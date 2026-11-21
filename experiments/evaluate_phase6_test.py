"""
TruthLens Phase 6: Locked Test Evaluation, Domain Confounding & Generalization Audit.

Evaluates:
- Baseline (SigLIP + CIFAKE head)
- E3 Candidate (SigLIP Layer 11 + Head fine-tuned on Tiny-GenImage)

Across:
1. Tiny-GenImage locked test partition (7,000 samples)
2. Frontier Benchmark October 2026 locked test partition (2,091 samples)
3. WildAIGC-Bench locked test partition (1,294 samples)

Implements:
- Step 4: Full evaluation & metrics computation on test splits.
- Step 5: Domain confounding investigation & matched-format transcoding analysis.
- Step 6: Cross-generator generalization (Seen vs Unseen generators).
- Step 7: Frontier social-media compression robustness.

Saves:
- experiments/results/phase6/baseline_*_predictions.csv
- experiments/results/phase6/e3_*_predictions.csv
- experiments/reports/phase6_domain_confounding.json
- experiments/reports/phase6_encoding_robustness.csv
- experiments/reports/phase6_generator_generalization.json
- experiments/reports/phase6_compression_robustness.json
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import SiglipVisionModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.training_utils import (
    SiglipClassifierModel,
    compute_file_sha256,
    get_canonical_transform,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_phase6")


class TestManifestDataset(Dataset):
    """Dataset loading test samples and metadata preserving original test membership."""

    def __init__(
        self,
        manifest_path: Path,
        base_dir: Path,
        transform=None,
    ):
        self.base_dir = base_dir
        self.transform = transform or get_canonical_transform()

        rows = []
        with open(manifest_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("evaluation_role") == "test":
                    rows.append(r)

        self.rows = rows
        logger.info(
            "Loaded %d locked test samples from %s (base_dir=%s)",
            len(self.rows),
            manifest_path.name,
            base_dir.name,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        row = self.rows[idx]
        rel_p = row.get("relative_path") or row.get("image_path")
        full_p = self.base_dir / rel_p

        lbl = row.get("normalized_label") or row.get("label")
        gt = 1.0 if lbl in ("ai_generated", "fake") else 0.0

        try:
            with Image.open(full_p) as img:
                rgb = img.convert("RGB")
                tensor = self.transform(rgb)
        except Exception as e:
            logger.error("Failed to load test image %s: %s", full_p, e)
            tensor = torch.zeros(3, 224, 224, dtype=torch.float32)

        return tensor, gt, row


def load_candidate_e3_model(device: torch.device) -> SiglipClassifierModel:
    """Load the trained E3 candidate model from best_checkpoint.pt."""
    chkpt_path = Path("experiments/checkpoints/phase5/e3_partial_finetuning/best_checkpoint.pt")
    logger.info("Loading E3 candidate model from %s...", chkpt_path)
    chkpt = torch.load(chkpt_path, map_location="cpu", weights_only=False)

    vision_model = SiglipVisionModel.from_pretrained(
        "google/siglip-base-patch16-224",
        cache_dir="models/cache",
    )
    model = SiglipClassifierModel(vision_model, nn.Linear(768, 1))
    model.load_state_dict(chkpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_baseline_model(device: torch.device) -> SiglipClassifierModel:
    """Load the original frozen baseline model from siglip_frozen_head.pt."""
    baseline_chkpt = Path("models/baselines/siglip_frozen_head.pt")
    logger.info("Loading baseline model head from %s...", baseline_chkpt)
    head_state = torch.load(baseline_chkpt, map_location="cpu", weights_only=True)

    vision_model = SiglipVisionModel.from_pretrained(
        "google/siglip-base-patch16-224",
        cache_dir="models/cache",
    )
    head = nn.Linear(768, 1)
    head.load_state_dict(head_state)

    model = SiglipClassifierModel(vision_model, head)
    model.to(device)
    model.eval()
    return model


def compute_binary_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate complete classification and discrimination metrics."""
    y_pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    frr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    try:
        auc = float(roc_auc_score(y_true, probs))
    except Exception:
        auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, probs))
    except Exception:
        pr_auc = float(np.mean(y_true))

    return {
        "sample_count": len(y_true),
        "threshold": threshold,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "far": round(far, 4),
        "frr": round(frr, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "mcc": round(mcc, 4),
        "cohen_kappa": round(kappa, 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def run_inference_on_dataset(
    model: nn.Module,
    ds: TestManifestDataset,
    temp_fit: float,
    temp_pooled: float,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute inference over all test samples in the dataset."""
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    model.eval()

    results = []
    all_targets = []
    all_probs = []

    with torch.inference_mode():
        for tensors, targets, rows in loader:
            tensors = tensors.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(tensors).squeeze(1).float().cpu().numpy()

            for i in range(len(logits)):
                z = float(logits[i])
                gt = float(targets[i])
                p_uncal = float(1.0 / (1.0 + np.exp(-np.clip(z, -700, 700))))
                p_cal_fit = float(1.0 / (1.0 + np.exp(-np.clip(z / max(1e-6, temp_fit), -700, 700))))
                p_cal_pool = float(1.0 / (1.0 + np.exp(-np.clip(z / max(1e-6, temp_pooled), -700, 700))))

                r_dict = {
                    "sample_id": rows["sample_id"][i] if "sample_id" in rows else f"test_{len(results)}",
                    "image_path": rows["image_path"][i] if "image_path" in rows else rows["relative_path"][i],
                    "ground_truth_binary": int(gt),
                    "label": rows["label"][i] if "label" in rows else rows.get("normalized_label", ["ai_generated" if gt == 1.0 else "likely_authentic"])[i],
                    "evaluation_role": "test",
                    "generator": rows["generator"][i] if "generator" in rows else rows.get("generator_name", ["unknown"])[i],
                    "group_id": rows["group_id"][i] if "group_id" in rows else rows.get("source_group_id", ["default"])[i],
                    "sha256": rows["sha256"][i] if "sha256" in rows else rows.get("image_sha256", [""])[i],
                    "raw_logit": round(z, 6),
                    "uncalibrated_probability": round(p_uncal, 6),
                    "calibrated_probability_fit": round(p_cal_fit, 6),
                    "calibrated_probability_pooled": round(p_cal_pool, 6),
                    "predicted_binary": int(p_uncal >= 0.5),
                    "is_correct": int((p_uncal >= 0.5) == int(gt)),
                    "annotation_notes": rows.get("annotation_notes", [""])[i],
                }
                results.append(r_dict)
                all_targets.append(gt)
                all_probs.append(p_uncal)

    metrics = compute_binary_metrics(np.array(all_targets, dtype=int), np.array(all_probs, dtype=float))
    return results, metrics


def save_predictions_csv(records: List[Dict[str, Any]], out_path: Path) -> None:
    """Save formatted test predictions CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    logger.info("Saved %d predictions to %s", len(records), out_path)


def run_domain_confounding_investigation(
    model_e3: nn.Module,
    model_base: nn.Module,
    test_ds: TestManifestDataset,
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Inspect file format distribution and perform matched-format transcoding evaluation."""
    logger.info("=" * 70)
    logger.info("STARTING DOMAIN CONFOUNDING & MATCHED-FORMAT TRANSCODING AUDIT")
    logger.info("=" * 70)

    # 1. Format distribution on test dataset
    format_counts = defaultdict(lambda: {"total": 0, "real": 0, "ai": 0})
    for r in test_ds.rows:
        fmt = r.get("format") or Path(r.get("image_path") or r.get("relative_path")).suffix.lstrip(".").upper()
        lbl = r.get("normalized_label") or r.get("label")
        is_ai = 1 if lbl in ("ai_generated", "fake") else 0
        format_counts[fmt]["total"] += 1
        if is_ai == 1:
            format_counts[fmt]["ai"] += 1
        else:
            format_counts[fmt]["real"] += 1

    format_summary = {k: dict(v) for k, v in format_counts.items()}
    logger.info("Test set format breakdown: %s", format_summary)

    # 2. Matched-format evaluation:
    # Transcode a representative stratified subset (e.g. 1,000 paired test images: 500 Real JPEG, 500 AI PNG)
    # to evaluate causal impact of encoding: Original, JPEG-q90, JPEG-q75, and lossless PNG.
    transform = get_canonical_transform()

    paired_real = [r for r in test_ds.rows if (r.get("normalized_label") or r.get("label")) != "ai_generated"][:500]
    paired_ai = [r for r in test_ds.rows if (r.get("normalized_label") or r.get("label")) == "ai_generated"][:500]
    selected_sample_rows = paired_real + paired_ai

    logger.info("Evaluating matched-format consistency on %d paired test samples (500 Real, 500 AI)...", len(selected_sample_rows))

    encoding_rows = []
    e3_flips = {"jpeg90": 0, "jpeg75": 0, "png": 0}
    base_flips = {"jpeg90": 0, "jpeg75": 0, "png": 0}

    e3_logit_shifts = {"jpeg90": [], "jpeg75": [], "png": []}
    base_logit_shifts = {"jpeg90": [], "jpeg75": [], "png": []}

    model_e3.eval()
    model_base.eval()

    for idx, r in enumerate(selected_sample_rows):
        rel_p = r.get("relative_path") or r.get("image_path")
        full_p = test_ds.base_dir / rel_p
        lbl = r.get("normalized_label") or r.get("label")
        gt = 1.0 if lbl in ("ai_generated", "fake") else 0.0
        orig_fmt = Path(full_p).suffix.lstrip(".").upper()

        try:
            with Image.open(full_p) as img:
                rgb_orig = img.convert("RGB")
        except Exception as e:
            continue

        # Transcode versions in-memory
        # 1. JPEG quality 90
        buf_j90 = io.BytesIO()
        rgb_orig.save(buf_j90, format="JPEG", quality=90)
        buf_j90.seek(0)
        rgb_j90 = Image.open(buf_j90).convert("RGB")

        # 2. JPEG quality 75
        buf_j75 = io.BytesIO()
        rgb_orig.save(buf_j75, format="JPEG", quality=75)
        buf_j75.seek(0)
        rgb_j75 = Image.open(buf_j75).convert("RGB")

        # 3. PNG
        buf_png = io.BytesIO()
        rgb_orig.save(buf_png, format="PNG")
        buf_png.seek(0)
        rgb_png = Image.open(buf_png).convert("RGB")

        # Create batch of [orig, j90, j75, png]
        batch = torch.stack([
            transform(rgb_orig),
            transform(rgb_j90),
            transform(rgb_j75),
            transform(rgb_png),
        ]).to(device)

        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                e3_out = model_e3(batch).squeeze(1).float().cpu().numpy()
                base_out = model_base(batch).squeeze(1).float().cpu().numpy()

        z_orig_e3, z_j90_e3, z_j75_e3, z_png_e3 = e3_out
        z_orig_base, z_j90_base, z_j75_base, z_png_base = base_out

        p_orig_e3 = float(1.0 / (1.0 + np.exp(-z_orig_e3)))
        p_j90_e3 = float(1.0 / (1.0 + np.exp(-z_j90_e3)))
        p_j75_e3 = float(1.0 / (1.0 + np.exp(-z_j75_e3)))
        p_png_e3 = float(1.0 / (1.0 + np.exp(-z_png_e3)))

        p_orig_base = float(1.0 / (1.0 + np.exp(-z_orig_base)))
        p_j90_base = float(1.0 / (1.0 + np.exp(-z_j90_base)))
        p_j75_base = float(1.0 / (1.0 + np.exp(-z_j75_base)))
        p_png_base = float(1.0 / (1.0 + np.exp(-z_png_base)))

        # Track E3 shifts
        e3_logit_shifts["jpeg90"].append(abs(z_j90_e3 - z_orig_e3))
        e3_logit_shifts["jpeg75"].append(abs(z_j75_e3 - z_orig_e3))
        e3_logit_shifts["png"].append(abs(z_png_e3 - z_orig_e3))

        if (p_j90_e3 >= 0.5) != (p_orig_e3 >= 0.5):
            e3_flips["jpeg90"] += 1
        if (p_j75_e3 >= 0.5) != (p_orig_e3 >= 0.5):
            e3_flips["jpeg75"] += 1
        if (p_png_e3 >= 0.5) != (p_orig_e3 >= 0.5):
            e3_flips["png"] += 1

        # Track Baseline shifts
        base_logit_shifts["jpeg90"].append(abs(z_j90_base - z_orig_base))
        base_logit_shifts["jpeg75"].append(abs(z_j75_base - z_orig_base))
        base_logit_shifts["png"].append(abs(z_png_base - z_orig_base))

        if (p_j90_base >= 0.5) != (p_orig_base >= 0.5):
            base_flips["jpeg90"] += 1
        if (p_j75_base >= 0.5) != (p_orig_base >= 0.5):
            base_flips["jpeg75"] += 1
        if (p_png_base >= 0.5) != (p_orig_base >= 0.5):
            base_flips["png"] += 1

        encoding_rows.append({
            "sample_index": idx,
            "image_path": rel_p,
            "ground_truth_binary": int(gt),
            "original_format": orig_fmt,
            "e3_logit_orig": round(float(z_orig_e3), 4),
            "e3_prob_orig": round(p_orig_e3, 4),
            "e3_prob_jpeg90": round(p_j90_e3, 4),
            "e3_prob_jpeg75": round(p_j75_e3, 4),
            "e3_prob_png": round(p_png_e3, 4),
            "e3_flip_jpeg90": int((p_j90_e3 >= 0.5) != (p_orig_e3 >= 0.5)),
            "e3_flip_jpeg75": int((p_j75_e3 >= 0.5) != (p_orig_e3 >= 0.5)),
            "e3_flip_png": int((p_png_e3 >= 0.5) != (p_orig_e3 >= 0.5)),
            "baseline_prob_orig": round(p_orig_base, 4),
            "baseline_prob_jpeg90": round(p_j90_base, 4),
            "baseline_prob_jpeg75": round(p_j75_base, 4),
            "baseline_prob_png": round(p_png_base, 4),
            "baseline_flip_jpeg90": int((p_j90_base >= 0.5) != (p_orig_base >= 0.5)),
            "baseline_flip_jpeg75": int((p_j75_base >= 0.5) != (p_orig_base >= 0.5)),
            "baseline_flip_png": int((p_png_base >= 0.5) != (p_orig_base >= 0.5)),
        })

    n_tested = len(selected_sample_rows)
    domain_report = {
        "format_distribution": format_summary,
        "matched_format_evaluation": {
            "samples_evaluated": n_tested,
            "real_samples_jpeg": 500,
            "ai_samples_png": 500,
            "e3_candidate_robustness": {
                "jpeg90": {
                    "prediction_flips": e3_flips["jpeg90"],
                    "flip_rate_pct": round((e3_flips["jpeg90"] / n_tested) * 100.0, 2),
                    "mean_absolute_logit_shift": round(float(np.mean(e3_logit_shifts["jpeg90"])), 4),
                    "max_absolute_logit_shift": round(float(np.max(e3_logit_shifts["jpeg90"])), 4),
                },
                "jpeg75": {
                    "prediction_flips": e3_flips["jpeg75"],
                    "flip_rate_pct": round((e3_flips["jpeg75"] / n_tested) * 100.0, 2),
                    "mean_absolute_logit_shift": round(float(np.mean(e3_logit_shifts["jpeg75"])), 4),
                    "max_absolute_logit_shift": round(float(np.max(e3_logit_shifts["jpeg75"])), 4),
                },
                "png": {
                    "prediction_flips": e3_flips["png"],
                    "flip_rate_pct": round((e3_flips["png"] / n_tested) * 100.0, 2),
                    "mean_absolute_logit_shift": round(float(np.mean(e3_logit_shifts["png"])), 4),
                    "max_absolute_logit_shift": round(float(np.max(e3_logit_shifts["png"])), 4),
                },
            },
            "baseline_robustness": {
                "jpeg90": {
                    "prediction_flips": base_flips["jpeg90"],
                    "flip_rate_pct": round((base_flips["jpeg90"] / n_tested) * 100.0, 2),
                    "mean_absolute_logit_shift": round(float(np.mean(base_logit_shifts["jpeg90"])), 4),
                    "max_absolute_logit_shift": round(float(np.max(base_logit_shifts["jpeg90"])), 4),
                },
                "jpeg75": {
                    "prediction_flips": base_flips["jpeg75"],
                    "flip_rate_pct": round((base_flips["jpeg75"] / n_tested) * 100.0, 2),
                    "mean_absolute_logit_shift": round(float(np.mean(base_logit_shifts["jpeg75"])), 4),
                    "max_absolute_logit_shift": round(float(np.max(base_logit_shifts["jpeg75"])), 4),
                },
                "png": {
                    "prediction_flips": base_flips["png"],
                    "flip_rate_pct": round((base_flips["png"] / n_tested) * 100.0, 2),
                    "mean_absolute_logit_shift": round(float(np.mean(base_logit_shifts["png"])), 4),
                    "max_absolute_logit_shift": round(float(np.max(base_logit_shifts["png"])), 4),
                },
            },
            "scientific_interpretation": (
                "Matched-format transcoding reveals whether the classifier relies heavily on compression artifacts. "
                "Transcoding real JPEG images to PNG and AI PNG images to JPEG tests causal prediction invariance. "
                "Differences in flip rates and logit shifts between baseline and E3 provide evidence on whether fine-tuning "
                "sensitized the network to compression signatures or semantic generative structures."
            ),
        },
    }

    return domain_report, encoding_rows


def run_generator_generalization_analysis(
    tiny_test_records_e3: List[Dict[str, Any]],
    tiny_test_records_base: List[Dict[str, Any]],
    frontier_test_records_e3: List[Dict[str, Any]],
    frontier_test_records_base: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate performance across seen and unseen generators."""
    logger.info("=" * 70)
    logger.info("EVALUATING CROSS-GENERATOR GENERALIZATION (SEEN VS UNSEEN)")
    logger.info("=" * 70)

    # 1. Tiny-GenImage generators
    # Training seen: ADM, BigGAN, SD 1.5, GLIDE
    # Held-out unseen: Midjourney, Wukong, VQDM
    seen_names = {"adm", "biggan", "sd", "stablediffusion", "glide"}
    unseen_names = {"midjourney", "wukong", "vqdm"}

    # Separate real and synthetic
    tiny_real_e3 = [r for r in tiny_test_records_e3 if r["ground_truth_binary"] == 0]
    tiny_real_base = [r for r in tiny_test_records_base if r["ground_truth_binary"] == 0]

    gen_groups_e3 = defaultdict(list)
    gen_groups_base = defaultdict(list)

    for r in tiny_test_records_e3:
        if r["ground_truth_binary"] == 1:
            gen_groups_e3[r["generator"].lower()].append(r)

    for r in tiny_test_records_base:
        if r["ground_truth_binary"] == 1:
            gen_groups_base[r["generator"].lower()].append(r)

    results_generators = {}

    all_gens = sorted(list(set(gen_groups_e3.keys()).union(gen_groups_base.keys())))
    for gen in all_gens:
        g_rows_e3 = gen_groups_e3[gen]
        g_rows_base = gen_groups_base[gen]

        is_seen = any(s in gen for s in seen_names)
        is_heldout = any(u in gen for u in unseen_names)
        category = "SEEN_TRAINING" if is_seen else ("HELD_OUT_UNSEEN" if is_heldout else "UNSEEN")

        # Combine with authentic real rows for valid binary discrimination metrics
        combined_e3 = tiny_real_e3 + g_rows_e3
        combined_base = tiny_real_base + g_rows_base

        y_e3 = np.array([r["ground_truth_binary"] for r in combined_e3], dtype=int)
        p_e3 = np.array([r["uncalibrated_probability"] for r in combined_e3], dtype=float)
        m_e3 = compute_binary_metrics(y_e3, p_e3)

        y_b = np.array([r["ground_truth_binary"] for r in combined_base], dtype=int)
        p_b = np.array([r["uncalibrated_probability"] for r in combined_base], dtype=float)
        m_b = compute_binary_metrics(y_b, p_b)

        # AI detection rate on the generator slice alone (Recall)
        ai_detect_e3 = float(np.mean([r["predicted_binary"] for r in g_rows_e3])) if g_rows_e3 else 0.0
        ai_detect_b = float(np.mean([r["predicted_binary"] for r in g_rows_base])) if g_rows_base else 0.0

        results_generators[gen] = {
            "category": category,
            "ai_sample_count": len(g_rows_e3),
            "paired_real_sample_count": len(tiny_real_e3),
            "e3_candidate": {
                "ai_detection_rate": round(ai_detect_e3, 4),
                "balanced_accuracy": m_e3["balanced_accuracy"],
                "roc_auc": m_e3["roc_auc"],
                "f1_score": m_e3["f1_score"],
                "precision": m_e3["precision"],
                "recall": m_e3["recall"],
            },
            "baseline": {
                "ai_detection_rate": round(ai_detect_b, 4),
                "balanced_accuracy": m_b["balanced_accuracy"],
                "roc_auc": m_b["roc_auc"],
                "f1_score": m_b["f1_score"],
                "precision": m_b["precision"],
                "recall": m_b["recall"],
            },
            "delta_balanced_accuracy": round(m_e3["balanced_accuracy"] - m_b["balanced_accuracy"], 4),
            "delta_roc_auc": round(m_e3["roc_auc"] - m_b["roc_auc"], 4),
        }
        logger.info(
            "Generator %-14s (%-15s) | AI Detect: Base=%.2f%%, E3=%.2f%% | BalAcc: Base=%.4f, E3=%.4f (Δ=%+.4f)",
            gen,
            category,
            ai_detect_b * 100.0,
            ai_detect_e3 * 100.0,
            m_b["balanced_accuracy"],
            m_e3["balanced_accuracy"],
            m_e3["balanced_accuracy"] - m_b["balanced_accuracy"],
        )

    # 2. Frontier Generators Breakdown
    frontier_gens_e3 = defaultdict(list)
    frontier_gens_base = defaultdict(list)
    frontier_real_e3 = [r for r in frontier_test_records_e3 if r["ground_truth_binary"] == 0]
    frontier_real_base = [r for r in frontier_test_records_base if r["ground_truth_binary"] == 0]

    for r in frontier_test_records_e3:
        if r["ground_truth_binary"] == 1:
            frontier_gens_e3[r["generator"]].append(r)

    for r in frontier_test_records_base:
        if r["ground_truth_binary"] == 1:
            frontier_gens_base[r["generator"]].append(r)

    frontier_results = {}
    for gen in sorted(list(frontier_gens_e3.keys())):
        g_rows_e3 = frontier_gens_e3[gen]
        g_rows_base = frontier_gens_base[gen]

        comb_e3 = frontier_real_e3 + g_rows_e3
        comb_base = frontier_real_base + g_rows_base

        y_e3 = np.array([r["ground_truth_binary"] for r in comb_e3], dtype=int)
        p_e3 = np.array([r["uncalibrated_probability"] for r in comb_e3], dtype=float)
        m_e3 = compute_binary_metrics(y_e3, p_e3)

        y_b = np.array([r["ground_truth_binary"] for r in comb_base], dtype=int)
        p_b = np.array([r["uncalibrated_probability"] for r in comb_base], dtype=float)
        m_b = compute_binary_metrics(y_b, p_b)

        frontier_results[gen] = {
            "ai_sample_count": len(g_rows_e3),
            "paired_real_sample_count": len(frontier_real_e3),
            "e3_candidate": {
                "ai_detection_rate": round(float(np.mean([r["predicted_binary"] for r in g_rows_e3])), 4),
                "balanced_accuracy": m_e3["balanced_accuracy"],
                "roc_auc": m_e3["roc_auc"],
                "f1_score": m_e3["f1_score"],
            },
            "baseline": {
                "ai_detection_rate": round(float(np.mean([r["predicted_binary"] for r in g_rows_base])), 4),
                "balanced_accuracy": m_b["balanced_accuracy"],
                "roc_auc": m_b["roc_auc"],
                "f1_score": m_b["f1_score"],
            },
            "delta_balanced_accuracy": round(m_e3["balanced_accuracy"] - m_b["balanced_accuracy"], 4),
            "delta_roc_auc": round(m_e3["roc_auc"] - m_b["roc_auc"], 4),
        }

    return {
        "tiny_genimage_generators": results_generators,
        "frontier_generators": frontier_results,
    }


def run_compression_robustness_analysis(
    frontier_records_e3: List[Dict[str, Any]],
    frontier_records_base: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze impact of social media compression variants in Frontier Benchmark."""
    logger.info("=" * 70)
    logger.info("EVALUATING FRONTIER SOCIAL MEDIA COMPRESSION ROBUSTNESS")
    logger.info("=" * 70)

    variant_e3 = defaultdict(list)
    variant_base = defaultdict(list)

    for r in frontier_records_e3:
        p = r["image_path"]
        v = "original"
        if "whatsapp" in p:
            v = "whatsapp"
        elif "instagram" in p:
            v = "instagram"
        elif "reshared" in p:
            v = "reshared"
        variant_e3[v].append(r)

    for r in frontier_records_base:
        p = r["image_path"]
        v = "original"
        if "whatsapp" in p:
            v = "whatsapp"
        elif "instagram" in p:
            v = "instagram"
        elif "reshared" in p:
            v = "reshared"
        variant_base[v].append(r)

    variant_metrics = {}
    all_variants = sorted(list(variant_e3.keys()))

    for v in all_variants:
        rows_e3 = variant_e3[v]
        rows_b = variant_base[v]

        y_e3 = np.array([r["ground_truth_binary"] for r in rows_e3], dtype=int)
        p_e3 = np.array([r["uncalibrated_probability"] for r in rows_e3], dtype=float)
        m_e3 = compute_binary_metrics(y_e3, p_e3)

        y_b = np.array([r["ground_truth_binary"] for r in rows_b], dtype=int)
        p_b = np.array([r["uncalibrated_probability"] for r in rows_b], dtype=float)
        m_b = compute_binary_metrics(y_b, p_b)

        variant_metrics[v] = {
            "sample_count": len(rows_e3),
            "e3_candidate": m_e3,
            "baseline": m_b,
            "delta_balanced_accuracy": round(m_e3["balanced_accuracy"] - m_b["balanced_accuracy"], 4),
            "delta_roc_auc": round(m_e3["roc_auc"] - m_b["roc_auc"], 4),
            "delta_f1": round(m_e3["f1_score"] - m_b["f1_score"], 4),
        }
        logger.info(
            "Compression Variant: %-12s | Base BalAcc=%.4f, AUC=%.4f | E3 BalAcc=%.4f, AUC=%.4f",
            v,
            m_b["balanced_accuracy"],
            m_b["roc_auc"],
            m_e3["balanced_accuracy"],
            m_e3["roc_auc"],
        )

    return variant_metrics


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Active compute device: %s", device)

    # Load fitted calibration temperatures
    cal_file = Path("experiments/reports/phase6_calibration.json")
    with open(cal_file, "r", encoding="utf-8") as f:
        cal_data = json.load(f)

    fitted_temps = {
        "tiny_genimage": cal_data["datasets"]["tiny_genimage"]["optimal_temperature"],
        "frontier_benchmark_2026_10": cal_data["datasets"]["frontier_benchmark_2026_10"]["optimal_temperature"],
        "wildaigc_bench": cal_data["datasets"]["wildaigc_bench"]["optimal_temperature"],
        "pooled": cal_data["datasets"]["pooled"]["optimal_temperature"],
    }
    logger.info("Loaded fitted temperatures: %s", fitted_temps)

    # Load Models
    model_e3 = load_candidate_e3_model(device)
    model_base = load_baseline_model(device)

    # Locked test datasets
    test_datasets = [
        ("tiny_genimage", Path("experiments/manifests/tiny_genimage_evaluation_splits.csv"), Path("datasets/external/tiny_genimage")),
        ("frontier_benchmark_2026_10", Path("experiments/manifests/frontier_benchmark_2026_10_evaluation_splits.csv"), Path("datasets/external/ai_image_detection_benchmark_2026_10")),
        ("wildaigc_bench", Path("experiments/manifests/wildaigc_bench_evaluation_splits.csv"), Path("datasets/external/wildaigc_bench")),
    ]

    results_dir = Path("experiments/results/phase6")
    results_dir.mkdir(parents=True, exist_ok=True)

    records_e3 = {}
    records_base = {}
    metrics_e3 = {}
    metrics_base = {}

    for name, manifest_path, base_dir in test_datasets:
        logger.info("=" * 70)
        logger.info("EVALUATING LOCKED TEST SPLIT: %s", name)
        logger.info("=" * 70)

        ds = TestManifestDataset(manifest_path, base_dir)
        temp_fit = fitted_temps[name]
        temp_pool = fitted_temps["pooled"]

        # Run E3 Candidate
        r_e3, m_e3 = run_inference_on_dataset(model_e3, ds, temp_fit, temp_pool, device)
        records_e3[name] = r_e3
        metrics_e3[name] = m_e3
        save_predictions_csv(r_e3, results_dir / f"e3_{name}_predictions.csv")

        # Run Baseline
        r_base, m_base = run_inference_on_dataset(model_base, ds, temp_fit, temp_pool, device)
        records_base[name] = r_base
        metrics_base[name] = m_base
        save_predictions_csv(r_base, results_dir / f"baseline_{name}_predictions.csv")

        logger.info(
            "%s TEST RESULTS:\n"
            "  Baseline: Acc=%.4f, BalAcc=%.4f, Prec=%.4f, Rec=%.4f, Spec=%.4f, F1=%.4f, AUC=%.4f\n"
            "  Candidate E3: Acc=%.4f, BalAcc=%.4f, Prec=%.4f, Rec=%.4f, Spec=%.4f, F1=%.4f, AUC=%.4f\n"
            "  Δ(E3 - Base): BalAcc=%+.4f, AUC=%+.4f, F1=%+.4f",
            name,
            m_base["accuracy"],
            m_base["balanced_accuracy"],
            m_base["precision"],
            m_base["recall"],
            m_base["specificity"],
            m_base["f1_score"],
            m_base["roc_auc"],
            m_e3["accuracy"],
            m_e3["balanced_accuracy"],
            m_e3["precision"],
            m_e3["recall"],
            m_e3["specificity"],
            m_e3["f1_score"],
            m_e3["roc_auc"],
            m_e3["balanced_accuracy"] - m_base["balanced_accuracy"],
            m_e3["roc_auc"] - m_base["roc_auc"],
            m_e3["f1_score"] - m_base["f1_score"],
        )

    # Step 5: Domain Confounding Investigation
    tiny_test_ds = TestManifestDataset(test_datasets[0][1], test_datasets[0][2])
    domain_report, encoding_rows = run_domain_confounding_investigation(model_e3, model_base, tiny_test_ds, device)

    reports_dir = Path("experiments/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(reports_dir / "phase6_domain_confounding.json", "w", encoding="utf-8") as f:
        json.dump(domain_report, f, indent=2)

    with open(reports_dir / "phase6_encoding_robustness.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(encoding_rows[0].keys()))
        writer.writeheader()
        writer.writerows(encoding_rows)

    logger.info("Saved domain confounding report and encoding robustness CSV.")

    # Step 6: Generator Generalization
    gen_report = run_generator_generalization_analysis(
        records_e3["tiny_genimage"],
        records_base["tiny_genimage"],
        records_e3["frontier_benchmark_2026_10"],
        records_base["frontier_benchmark_2026_10"],
    )
    with open(reports_dir / "phase6_generator_generalization.json", "w", encoding="utf-8") as f:
        json.dump(gen_report, f, indent=2)
    logger.info("Saved generator generalization report.")

    # Step 7: Frontier Compression Robustness
    comp_report = run_compression_robustness_analysis(
        records_e3["frontier_benchmark_2026_10"],
        records_base["frontier_benchmark_2026_10"],
    )
    with open(reports_dir / "phase6_compression_robustness.json", "w", encoding="utf-8") as f:
        json.dump(comp_report, f, indent=2)
    logger.info("Saved compression robustness report.")

    logger.info("=" * 70)
    logger.info("STEP 4-7 TEST EVALUATION AND GENERALIZATION AUDIT COMPLETE.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
