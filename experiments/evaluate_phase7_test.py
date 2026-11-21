"""
TruthLens Phase 7: Locked Test Evaluation, Generalization & Candidate Decision Audit.

Evaluates:
- S2: truthlens/siglip-base-v2-s2 (Selected Primary Candidate)
- S1: truthlens/siglip-base-v2-s1 (Audited Secondary Candidate)
- Baseline: truthlens/siglip-cifake-frozen-head (Original Frozen Production Baseline)

Partitions:
1. Tiny-GenImage locked test (7,000 samples)
   - Canonical raw evaluation
   - Matched in-memory augmentation evaluation
   - Seen vs Unseen generator breakdown
2. Frontier Benchmark October 2026 locked test (2,091 samples)
   - Canonical raw evaluation
   - Social media compression variants (Original, Instagram, Reshared, WhatsApp)
   - Real-image false alarm rate (FPR <= 5%)
3. WildAIGC-Bench locked test (1,294 samples)
   - Contextual label caveat noted

Performs:
- Group-aware paired bootstrap (1,000 resamples) for 95% CIs.
- Pre-registered decision criteria check (Conditions 1-4).
- Saves predictions to experiments/results/phase7/
- Saves reports to experiments/reports/
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
from datetime import datetime, timezone
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

from experiments.train_phase7 import apply_matched_augmentation
from experiments.training_utils import (
    SiglipClassifierModel,
    compute_file_sha256,
    get_canonical_transform,
    set_seed,
)

Image.MAX_IMAGE_PIXELS = 250_000_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_phase7")


class TestDataset(Dataset):
    """Dataset for locked test evaluation with optional matched augmentation."""

    def __init__(
        self,
        manifest_path: Path,
        base_dir: Path,
        apply_aug: bool = False,
    ):
        self.base_dir = base_dir
        self.apply_aug = apply_aug
        self.canonical_transform = get_canonical_transform()

        from torchvision import transforms
        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        rows = []
        with open(manifest_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("evaluation_role") == "test":
                    rows.append(r)

        self.rows = rows
        logger.info(
            "TestDataset loaded %d locked test samples from %s (apply_aug=%s)",
            len(self.rows),
            manifest_path.name,
            apply_aug,
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
                if self.apply_aug:
                    aug_img = apply_matched_augmentation(rgb)
                    tensor = self.to_tensor_norm(aug_img)
                else:
                    tensor = self.canonical_transform(rgb)
        except Exception as e:
            logger.error("Failed to load image %s: %s", full_p, e)
            tensor = torch.zeros(3, 224, 224, dtype=torch.float32)

        return tensor, gt, row


def compute_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
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


def find_operating_threshold_fpr5(
    y_true: np.ndarray,
    probs: np.ndarray,
    target_fpr: float = 0.05,
) -> Tuple[float, Dict[str, Any]]:
    """Find operating threshold where FPR <= 5% on authentic images."""
    real_probs = probs[y_true == 0]
    if len(real_probs) == 0:
        return 0.5, compute_metrics(y_true, probs, 0.5)

    # Threshold is the (1 - target_fpr) percentile of real probabilities
    t_fpr5 = float(np.percentile(real_probs, (1.0 - target_fpr) * 100.0))
    metrics_at_t = compute_metrics(y_true, probs, threshold=t_fpr5)
    return round(t_fpr5, 4), metrics_at_t


def run_inference(
    model: nn.Module,
    ds: TestDataset,
    temp_fit: float,
    temp_pooled: float,
    device: torch.device,
    dataset_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    model.eval()

    records = []
    all_logits = []
    all_targets = []

    with torch.inference_mode():
        for step, (tensors, targets, rows) in enumerate(loader):
            tensors = tensors.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(tensors)

            logits_np = logits.float().cpu().numpy().flatten()
            targets_np = targets.numpy().flatten()

            all_logits.extend(logits_np)
            all_targets.extend(targets_np)

            for i in range(len(logits_np)):
                z = float(logits_np[i])
                gt = int(targets_np[i])
                p_uncal = 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, z))))
                p_cal_fit = 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, z / max(1e-4, temp_fit)))))
                p_cal_pool = 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, z / max(1e-4, temp_pooled)))))

                r = {
                    "sample_id": rows.get("sample_id", [f"{dataset_name}_{len(records)}"])[i] if isinstance(rows.get("sample_id"), list) else f"{dataset_name}_{len(records)}",
                    "dataset_name": dataset_name,
                    "image_path": rows["image_path"][i] if isinstance(rows["image_path"], list) else str(rows["image_path"]),
                    "generator": rows.get("generator", ["unknown"])[i] if isinstance(rows.get("generator"), list) else rows.get("generator", "unknown"),
                    "group_id": rows.get("group_id", ["default"])[i] if isinstance(rows.get("group_id"), list) else rows.get("group_id", "default"),
                    "ground_truth_binary": gt,
                    "raw_logit": round(z, 5),
                    "uncalibrated_probability": round(p_uncal, 5),
                    "calibrated_probability_fitted": round(p_cal_fit, 5),
                    "calibrated_probability_pooled": round(p_cal_pool, 5),
                    "predicted_binary": int(p_uncal >= 0.5),
                }
                records.append(r)

    y_arr = np.array(all_targets, dtype=int)
    z_arr = np.array(all_logits, dtype=float)
    p_arr = 1.0 / (1.0 + np.exp(-np.clip(z_arr, -700, 700)))

    metrics = compute_metrics(y_arr, p_arr, threshold=0.5)
    return records, metrics


def paired_group_bootstrap(
    records_candidate: List[Dict[str, Any]],
    records_baseline: List[Dict[str, Any]],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(records_candidate)
    assert n == len(records_baseline)

    y_all = np.array([r["ground_truth_binary"] for r in records_candidate], dtype=int)
    p_c = np.array([r["uncalibrated_probability"] for r in records_candidate], dtype=float)
    p_b = np.array([r["uncalibrated_probability"] for r in records_baseline], dtype=float)

    real_idx = np.where(y_all == 0)[0]
    ai_idx = np.where(y_all == 1)[0]

    # Grouping by group_id
    group_map = defaultdict(list)
    for i in ai_idx:
        group_map[records_candidate[i].get("group_id", "default")].append(i)
    unique_groups = list(group_map.keys())

    delta_bal_acc = []
    delta_roc_auc = []
    delta_f1 = []

    for _ in range(n_resamples):
        b_real = rng.choice(real_idx, size=len(real_idx), replace=True)
        b_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        b_ai = []
        for g in b_groups:
            b_ai.extend(group_map[g])
        idx = np.concatenate([b_real, b_ai])

        y_boot = y_all[idx]
        pc_boot = p_c[idx]
        pb_boot = p_b[idx]

        mc = compute_metrics(y_boot, pc_boot)
        mb = compute_metrics(y_boot, pb_boot)

        delta_bal_acc.append(mc["balanced_accuracy"] - mb["balanced_accuracy"])
        delta_roc_auc.append(mc["roc_auc"] - mb["roc_auc"])
        delta_f1.append(mc["f1_score"] - mb["f1_score"])

    def ci(vals: List[float]) -> Dict[str, float]:
        return {
            "mean": round(float(np.mean(vals)), 4),
            "ci_lower": round(float(np.percentile(vals, 2.5)), 4),
            "ci_upper": round(float(np.percentile(vals, 97.5)), 4),
        }

    return {
        "n_resamples": n_resamples,
        "delta_balanced_accuracy": ci(delta_bal_acc),
        "delta_roc_auc": ci(delta_roc_auc),
        "delta_f1": ci(delta_f1),
    }


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting Phase 7 Locked Test Evaluation on device: %s", device)

    # 1. Load Calibration Temperatures
    with open("experiments/reports/phase7_calibration.json", "r", encoding="utf-8") as f:
        cal_meta = json.load(f)["datasets"]

    temp_s2 = {
        "tiny_genimage": cal_meta["tiny_genimage"]["fitted_temperature"],
        "frontier_benchmark_2026_10": cal_meta["frontier_benchmark_2026_10"]["fitted_temperature"],
        "wildaigc_bench": cal_meta["wildaigc_bench"]["fitted_temperature"],
        "pooled": cal_meta["pooled"]["fitted_temperature"],
    }
    logger.info("Loaded S2 calibration temperatures: %s", temp_s2)

    # 2. Load Models
    # A) S2: Primary Candidate
    s2_chkpt_p = Path("experiments/checkpoints/phase7/s2_siglip_partial_ft/best_checkpoint.pt")
    chkpt_s2 = torch.load(s2_chkpt_p, map_location="cpu", weights_only=False)
    v_model_s2 = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224", cache_dir="models/cache")
    model_s2 = SiglipClassifierModel(v_model_s2, nn.Linear(768, 1))
    model_s2.load_state_dict(chkpt_s2["model_state_dict"])
    model_s2.to(device)
    model_s2.eval()
    logger.info("Loaded S2 model (SHA-256: %s)", compute_file_sha256(s2_chkpt_p))

    # B) S1: Secondary Candidate
    s1_chkpt_p = Path("experiments/checkpoints/phase7/s1_siglip_linear_probe/best_checkpoint.pt")
    chkpt_s1 = torch.load(s1_chkpt_p, map_location="cpu", weights_only=False)
    v_model_s1 = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224", cache_dir="models/cache")
    model_s1 = SiglipClassifierModel(v_model_s1, nn.Linear(768, 1))
    model_s1.load_state_dict(chkpt_s1["model_state_dict"])
    model_s1.to(device)
    model_s1.eval()
    logger.info("Loaded S1 model (SHA-256: %s)", compute_file_sha256(s1_chkpt_p))

    # C) Frozen Production Baseline
    base_head_p = Path("models/baselines/siglip_frozen_head.pt")
    head_state_base = torch.load(base_head_p, map_location="cpu", weights_only=True)
    v_model_base = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224", cache_dir="models/cache")
    head_base = nn.Linear(768, 1)
    head_base.load_state_dict(head_state_base)
    model_base = SiglipClassifierModel(v_model_base, head_base)
    model_base.to(device)
    model_base.eval()
    logger.info("Loaded Baseline model (Head SHA-256: %s)", compute_file_sha256(base_head_p))

    # 3. Test Manifests
    manifests = {
        "tiny_genimage": (
            Path("experiments/manifests/tiny_genimage_evaluation_splits.csv"),
            Path("datasets/external/tiny_genimage"),
        ),
        "frontier_benchmark_2026_10": (
            Path("experiments/manifests/frontier_benchmark_2026_10_evaluation_splits.csv"),
            Path("datasets/external/ai_image_detection_benchmark_2026_10"),
        ),
        "wildaigc_bench": (
            Path("experiments/manifests/wildaigc_bench_evaluation_splits.csv"),
            Path("datasets/external/wildaigc_bench"),
        ),
    }

    results_dir = Path("experiments/results/phase7")
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("experiments/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_eval_results = {}
    all_predictions = {}

    for d_name, (m_path, b_dir) in manifests.items():
        logger.info("=" * 70)
        logger.info("EVALUATING ON LOCKED TEST PARTITION: %s", d_name)
        logger.info("=" * 70)

        t_fit = temp_s2[d_name]
        t_pool = temp_s2["pooled"]

        # Raw canonical evaluation
        ds_raw = TestDataset(m_path, b_dir, apply_aug=False)

        # Baseline
        recs_base, met_base = run_inference(model_base, ds_raw, 1.0, 1.0, device, d_name)
        # S2
        recs_s2, met_s2 = run_inference(model_s2, ds_raw, t_fit, t_pool, device, d_name)
        # S1
        recs_s1, met_s1 = run_inference(model_s1, ds_raw, 1.0, 1.0, device, d_name)

        all_predictions[f"{d_name}_base"] = recs_base
        all_predictions[f"{d_name}_s2"] = recs_s2
        all_predictions[f"{d_name}_s1"] = recs_s1

        # Write prediction CSVs
        for prefix, recs in [("baseline", recs_base), ("s2", recs_s2), ("s1", recs_s1)]:
            csv_file = results_dir / f"{prefix}_{d_name}_predictions.csv"
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
                writer.writeheader()
                writer.writerows(recs)
            logger.info("Saved %s predictions to %s", prefix, csv_file.name)

        # Bootstrap comparison (S2 vs Base, S1 vs Base)
        boot_s2_vs_base = paired_group_bootstrap(recs_s2, recs_base, n_resamples=1000)
        boot_s1_vs_base = paired_group_bootstrap(recs_s1, recs_base, n_resamples=1000)

        all_eval_results[d_name] = {
            "sample_count": len(recs_base),
            "baseline": met_base,
            "s2_candidate": met_s2,
            "s1_candidate": met_s1,
            "delta_s2_vs_base": {
                "delta_balanced_accuracy": round(met_s2["balanced_accuracy"] - met_base["balanced_accuracy"], 4),
                "delta_roc_auc": round(met_s2["roc_auc"] - met_base["roc_auc"], 4),
                "delta_f1": round(met_s2["f1_score"] - met_base["f1_score"], 4),
                "bootstrap_ci": boot_s2_vs_base,
            },
            "delta_s1_vs_base": {
                "delta_balanced_accuracy": round(met_s1["balanced_accuracy"] - met_base["balanced_accuracy"], 4),
                "delta_roc_auc": round(met_s1["roc_auc"] - met_base["roc_auc"], 4),
                "delta_f1": round(met_s1["f1_score"] - met_base["f1_score"], 4),
                "bootstrap_ci": boot_s1_vs_base,
            },
        }

        logger.info(
            "%s: Base BalAcc=%.4f (AUC=%.4f) | S2 BalAcc=%.4f (AUC=%.4f) | S1 BalAcc=%.4f (AUC=%.4f)",
            d_name,
            met_base["balanced_accuracy"],
            met_base["roc_auc"],
            met_s2["balanced_accuracy"],
            met_s2["roc_auc"],
            met_s1["balanced_accuracy"],
            met_s1["roc_auc"],
        )

    # 4. Tiny-GenImage Matched-Augmentation In-Memory Test Evaluation
    logger.info("=" * 70)
    logger.info("EVALUATING TINY-GENIMAGE UNDER MATCHED IN-MEMORY AUGMENTATION")
    logger.info("=" * 70)
    tiny_m_path, tiny_b_dir = manifests["tiny_genimage"]
    ds_tiny_aug = TestDataset(tiny_m_path, tiny_b_dir, apply_aug=True)

    recs_tiny_aug_base, met_tiny_aug_base = run_inference(model_base, ds_tiny_aug, 1.0, 1.0, device, "tiny_genimage_aug")
    recs_tiny_aug_s2, met_tiny_aug_s2 = run_inference(model_s2, ds_tiny_aug, temp_s2["tiny_genimage"], temp_s2["pooled"], device, "tiny_genimage_aug")
    recs_tiny_aug_s1, met_tiny_aug_s1 = run_inference(model_s1, ds_tiny_aug, 1.0, 1.0, device, "tiny_genimage_aug")

    all_eval_results["tiny_genimage_matched_augmented"] = {
        "sample_count": len(recs_tiny_aug_base),
        "baseline": met_tiny_aug_base,
        "s2_candidate": met_tiny_aug_s2,
        "s1_candidate": met_tiny_aug_s1,
    }
    logger.info(
        "Tiny-GenImage Matched-Augmented: Base BalAcc=%.4f (AUC=%.4f) | S2 BalAcc=%.4f (AUC=%.4f) | S1 BalAcc=%.4f (AUC=%.4f)",
        met_tiny_aug_base["balanced_accuracy"], met_tiny_aug_base["roc_auc"],
        met_tiny_aug_s2["balanced_accuracy"], met_tiny_aug_s2["roc_auc"],
        met_tiny_aug_s1["balanced_accuracy"], met_tiny_aug_s1["roc_auc"],
    )

    # 5. Generator Generalization Breakdown (Tiny-GenImage Seen vs Unseen)
    logger.info("=" * 70)
    logger.info("EVALUATING GENERATOR GENERALIZATION (TINY-GENIMAGE)")
    logger.info("=" * 70)
    gen_breakdown = {}
    tiny_recs_s2 = all_predictions["tiny_genimage_s2"]
    tiny_recs_base = all_predictions["tiny_genimage_base"]

    gen_groups_s2 = defaultdict(list)
    gen_groups_base = defaultdict(list)
    for r in tiny_recs_s2:
        gen_groups_s2[r["generator"]].append(r)
    for r in tiny_recs_base:
        gen_groups_base[r["generator"]].append(r)

    seen_gens = ["ADM", "BigGAN", "Stable Diffusion v1.5", "GLIDE"]
    unseen_gens = ["Midjourney", "Wukong", "VQDM"]

    unseen_recalls = {}
    for g in seen_gens + unseen_gens + ["ImageNet-Camera"]:
        if g not in gen_groups_s2:
            continue
        g_s2 = gen_groups_s2[g]
        g_base = gen_groups_base[g]
        y_g = np.array([r["ground_truth_binary"] for r in g_s2], dtype=int)
        p_s2 = np.array([r["uncalibrated_probability"] for r in g_s2], dtype=float)
        p_base = np.array([r["uncalibrated_probability"] for r in g_base], dtype=float)

        rec_s2 = float(np.mean(p_s2 >= 0.5)) if y_g[0] == 1 else float(np.mean(p_s2 < 0.5))
        rec_base = float(np.mean(p_base >= 0.5)) if y_g[0] == 1 else float(np.mean(p_base < 0.5))

        category = "real" if g == "ImageNet-Camera" else ("seen_generator" if g in seen_gens else "unseen_generator")
        gen_breakdown[g] = {
            "category": category,
            "sample_count": len(g_s2),
            "s2_accuracy": round(rec_s2, 4),
            "baseline_accuracy": round(rec_base, 4),
            "delta": round(rec_s2 - rec_base, 4),
        }
        if category == "unseen_generator":
            unseen_recalls[g] = round(rec_s2, 4)

        logger.info("Generator %-24s (%-16s) | Base Acc: %.4f | S2 Acc: %.4f", g, category, rec_base, rec_s2)

    with open(reports_dir / "phase7_generator_generalization.json", "w", encoding="utf-8") as f:
        json.dump(gen_breakdown, f, indent=2)

    # 6. Frontier Social Media Compression Robustness
    logger.info("=" * 70)
    logger.info("EVALUATING FRONTIER SOCIAL MEDIA COMPRESSION VARIANTS")
    logger.info("=" * 70)
    comp_breakdown = {}
    front_recs_s2 = all_predictions["frontier_benchmark_2026_10_s2"]
    front_recs_base = all_predictions["frontier_benchmark_2026_10_base"]

    v_map_s2 = defaultdict(list)
    v_map_base = defaultdict(list)
    for r in front_recs_s2:
        p = r["image_path"].lower()
        v = "whatsapp" if "whatsapp" in p else ("instagram" if "instagram" in p else ("reshared" if "reshared" in p else "original"))
        v_map_s2[v].append(r)
    for r in front_recs_base:
        p = r["image_path"].lower()
        v = "whatsapp" if "whatsapp" in p else ("instagram" if "instagram" in p else ("reshared" if "reshared" in p else "original"))
        v_map_base[v].append(r)

    for v in sorted(list(v_map_s2.keys())):
        rows_s2 = v_map_s2[v]
        rows_b = v_map_base[v]
        y_v = np.array([r["ground_truth_binary"] for r in rows_s2], dtype=int)
        p_s2 = np.array([r["uncalibrated_probability"] for r in rows_s2], dtype=float)
        p_b = np.array([r["uncalibrated_probability"] for r in rows_b], dtype=float)

        m_s2 = compute_metrics(y_v, p_s2)
        m_b = compute_metrics(y_v, p_b)

        comp_breakdown[v] = {
            "sample_count": len(rows_s2),
            "s2_metrics": m_s2,
            "baseline_metrics": m_b,
            "delta_balanced_accuracy": round(m_s2["balanced_accuracy"] - m_b["balanced_accuracy"], 4),
            "delta_roc_auc": round(m_s2["roc_auc"] - m_b["roc_auc"], 4),
        }
        logger.info(
            "Compression Variant: %-12s | Base BalAcc=%.4f (AUC=%.4f) | S2 BalAcc=%.4f (AUC=%.4f)",
            v, m_b["balanced_accuracy"], m_b["roc_auc"], m_s2["balanced_accuracy"], m_s2["roc_auc"],
        )

    with open(reports_dir / "phase7_compression_robustness.json", "w", encoding="utf-8") as f:
        json.dump(comp_breakdown, f, indent=2)

    # 7. Operating Threshold for 5% Real-Image False Positive Rate (Frontier Test)
    y_front = np.array([r["ground_truth_binary"] for r in front_recs_s2], dtype=int)
    p_front_s2 = np.array([r["uncalibrated_probability"] for r in front_recs_s2], dtype=float)
    t_fpr5_s2, m_fpr5_s2 = find_operating_threshold_fpr5(y_front, p_front_s2, target_fpr=0.05)
    far_standard_s2 = all_eval_results["frontier_benchmark_2026_10"]["s2_candidate"]["far"]
    logger.info("Frontier Standard FAR at threshold 0.5: %.4f | Operating threshold for FPR <= 5%%: %.4f", far_standard_s2, t_fpr5_s2)

    # 8. Check Pre-Registered Promotion Criteria
    # Condition 1: Frontier test generalization CI strictly positive
    front_boot = all_eval_results["frontier_benchmark_2026_10"]["delta_s2_vs_base"]["bootstrap_ci"]
    ci_bal_acc_s2 = front_boot["delta_balanced_accuracy"]
    ci_roc_auc_s2 = front_boot["delta_roc_auc"]
    cond1_passed = bool(ci_bal_acc_s2["ci_lower"] > 0 and ci_roc_auc_s2["ci_lower"] > 0)

    # Condition 2: Real-image FPR <= 5%
    cond2_passed = bool(far_standard_s2 <= 0.05)

    # Condition 3: JPEG-90 flip rate < 25%
    with open("experiments/reports/phase7_validation_robustness.json", "r", encoding="utf-8") as f:
        val_rob = json.load(f)
    jpeg90_flip = val_rob["s2_siglip_partial_ft"]["jpeg90"]["flip_rate_pct"]
    cond3_passed = bool(jpeg90_flip < 25.0)

    # Condition 4: Unseen generator recall >= 50%
    cond4_passed = bool(all(rec >= 0.50 for rec in unseen_recalls.values()))

    all_conditions_met = cond1_passed and cond2_passed and cond3_passed and cond4_passed
    final_decision = "PROMOTE_TO_PHASE_8_REVIEW" if all_conditions_met else "REJECT_CANDIDATE_FOR_PRODUCTION"

    candidate_decision = {
        "candidate": {
            "model_id": "truthlens/siglip-base-v2-s2",
            "checkpoint_path": str(s2_chkpt_p),
            "checkpoint_sha256": compute_file_sha256(s2_chkpt_p),
            "experiment_name": "s2_siglip_partial_ft",
        },
        "baseline": {
            "model_id": "truthlens/siglip-cifake-frozen-head",
            "head_path": str(base_head_p),
            "head_sha256": compute_file_sha256(base_head_p),
        },
        "decision": final_decision,
        "recommendation": "RETAIN_FROZEN_BASELINE_IN_PRODUCTION" if final_decision == "REJECT_CANDIDATE_FOR_PRODUCTION" else "PROCEED_TO_STAGING_DEPLOYMENT",
        "pre_registered_criteria_evaluation": {
            "condition_1_frontier_generalization": {
                "description": "Frontier test Balanced Accuracy and ROC-AUC 95% bootstrap CI strictly positive (Delta > 0)",
                "delta_balanced_accuracy_point": all_eval_results["frontier_benchmark_2026_10"]["delta_s2_vs_base"]["delta_balanced_accuracy"],
                "delta_balanced_accuracy_95ci": [ci_bal_acc_s2["ci_lower"], ci_bal_acc_s2["ci_upper"]],
                "delta_roc_auc_point": all_eval_results["frontier_benchmark_2026_10"]["delta_s2_vs_base"]["delta_roc_auc"],
                "delta_roc_auc_95ci": [ci_roc_auc_s2["ci_lower"], ci_roc_auc_s2["ci_upper"]],
                "passed": cond1_passed,
            },
            "condition_2_real_image_fpr": {
                "description": "Real-image false positive rate on Frontier test partition <= 5.0%",
                "frontier_test_far_standard_threshold": far_standard_s2,
                "operating_threshold_for_5pct_fpr": t_fpr5_s2,
                "passed": cond2_passed,
            },
            "condition_3_transcoding_robustness": {
                "description": "Under JPEG-90 transcoding, prediction flip rate < 25.0%",
                "jpeg90_flip_rate_pct": jpeg90_flip,
                "passed": cond3_passed,
            },
            "condition_4_unseen_generator_recall": {
                "description": "Detection recall on each held-out generator (Midjourney, Wukong, VQDM) >= 50.0%",
                "unseen_generator_recalls": unseen_recalls,
                "passed": cond4_passed,
            },
        },
        "all_conditions_satisfied": all_conditions_met,
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Save final reports
    with open(reports_dir / "phase7_test_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(all_eval_results, f, indent=2)

    with open(reports_dir / "phase7_candidate_decision.json", "w", encoding="utf-8") as f:
        json.dump(candidate_decision, f, indent=2)

    with open(reports_dir / "phase7_statistical_comparison.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_evaluations": all_eval_results,
            "generator_breakdown": gen_breakdown,
            "compression_robustness": comp_breakdown,
        }, f, indent=2)

    logger.info("=" * 70)
    logger.info("PHASE 7 LOCKED TEST EVALUATION COMPLETE")
    logger.info("FINAL DECISION: %s", final_decision)
    logger.info("RECOMMENDATION: %s", candidate_decision["recommendation"])
    logger.info("Condition 1 (Frontier Generalization CI > 0): %s", cond1_passed)
    logger.info("Condition 2 (Frontier Real FPR <= 5%%):        %s (FAR=%.4f)", cond2_passed, far_standard_s2)
    logger.info("Condition 3 (JPEG-90 Robustness Flip < 25%%):  %s (Flip=%.1f%%)", cond3_passed, jpeg90_flip)
    logger.info("Condition 4 (Unseen Recall >= 50%%):           %s (%s)", cond4_passed, unseen_recalls)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
