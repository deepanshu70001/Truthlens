"""
TruthLens Phase 3: Comprehensive Evaluation Analysis, Generalization & Robustness.

Performs:
1. Locked held-out test evaluation using frozen validation-selected thresholds.
2. Group-aware block bootstrapping (1,000 resamples) for 95% confidence intervals.
3. Social media compression robustness analysis (Original vs WhatsApp vs Instagram vs Reshared).
4. Cross-generator generalization analysis (Seen vs Unseen generator architectures).
5. Comprehensive failure analysis (False Positives, False Negatives, Borderline cases).
6. Generates publication-ready figures in experiments/reports/figures/.
7. Persists phase3_results.json, phase3_results.md, phase3_robustness.json,
   phase3_generalization.json, and phase3_failure_analysis.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("analyze_results")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


def load_dataset_predictions(csv_path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "image_path": r["image_path"],
                "label": r["label"],
                "ground_truth_binary": int(r["ground_truth_binary"]),
                "evaluation_role": r["evaluation_role"],
                "group_id": r["group_id"],
                "generator": r["generator"],
                "source_url": r.get("source_url", ""),
                "license_ref": r.get("license_ref", ""),
                "sha256": r["sha256"],
                "raw_logit_fake": float(r["raw_logit_fake"]),
                "raw_logit_real": float(r["raw_logit_real"]),
                "raw_margin": float(r["raw_margin"]),
                "authentic_score": float(r["authentic_score"]),
                "ai_generated_score": float(r["ai_generated_score"]),
                "model_prediction": r["model_prediction"],
                "is_correct": int(r["is_correct"]),
                "annotation_notes": r.get("annotation_notes", ""),
            })
    return rows


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    far = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    frr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = float(np.mean(y_true))

    return {
        "threshold": round(threshold, 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "false_acceptance_rate": round(far, 4),
        "false_rejection_rate": round(frr, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
    }


def compute_abstention_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auth_max: float,
    ai_min: float,
) -> Dict[str, Any]:
    n = len(y_true)
    if n == 0:
        return {}

    decided_real = y_prob <= auth_max
    decided_ai = y_prob >= ai_min
    decided_mask = decided_real | decided_ai
    inconclusive_mask = ~decided_mask

    coverage = float(decided_mask.mean())
    abstention_rate = float(inconclusive_mask.mean())

    if decided_mask.sum() > 0:
        y_true_decided = y_true[decided_mask]
        y_pred_decided = np.where(decided_ai[decided_mask], 1, 0)
        decided_acc = float(accuracy_score(y_true_decided, y_pred_decided))
        decided_bal_acc = float(balanced_accuracy_score(y_true_decided, y_pred_decided))
        decided_f1 = float(f1_score(y_true_decided, y_pred_decided, zero_division=0))
        decided_error = 1.0 - decided_acc
    else:
        decided_acc = 0.0
        decided_bal_acc = 0.0
        decided_f1 = 0.0
        decided_error = 0.0

    return {
        "authentic_max_threshold": round(auth_max, 4),
        "ai_generated_min_threshold": round(ai_min, 4),
        "coverage": round(coverage, 4),
        "abstention_rate": round(abstention_rate, 4),
        "total_samples": n,
        "decided_samples": int(decided_mask.sum()),
        "inconclusive_samples": int(inconclusive_mask.sum()),
        "accuracy_on_decided": round(decided_acc, 4),
        "balanced_accuracy_on_decided": round(decided_bal_acc, 4),
        "f1_score_on_decided": round(decided_f1, 4),
        "error_rate_on_decided": round(decided_error, 4),
    }


def group_aware_bootstrap_ci(
    rows: List[Dict[str, Any]],
    temperature: float,
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Perform 1,000 group-level bootstrap resamples.
    Groups are sampled with replacement to preserve intra-group correlation.
    """
    rng = np.random.RandomState(seed)

    # Group rows by group_id
    groups_dict = defaultdict(list)
    for r in rows:
        groups_dict[r["group_id"]].append(r)

    unique_groups = list(groups_dict.keys())
    n_groups = len(unique_groups)

    if n_groups == 0:
        return {}

    boot_accs = []
    boot_bal_accs = []
    boot_f1s = []
    boot_aucs = []

    for _ in range(n_bootstraps):
        sampled_group_indices = rng.choice(n_groups, size=n_groups, replace=True)
        sampled_rows = []
        for idx in sampled_group_indices:
            grp = unique_groups[idx]
            sampled_rows.extend(groups_dict[grp])

        y_true = np.array([r["ground_truth_binary"] for r in sampled_rows], dtype=int)
        margins = np.array([r["raw_margin"] for r in sampled_rows], dtype=float)
        y_prob = sigmoid(margins / temperature)
        y_pred = (y_prob >= threshold).astype(int)

        boot_accs.append(accuracy_score(y_true, y_pred))
        boot_bal_accs.append(balanced_accuracy_score(y_true, y_pred))
        boot_f1s.append(f1_score(y_true, y_pred, zero_division=0))

        if len(np.unique(y_true)) > 1:
            try:
                boot_aucs.append(roc_auc_score(y_true, y_prob))
            except Exception:
                pass

    def ci_95(data: List[float]) -> Dict[str, float]:
        if not data:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
        arr = np.array(data)
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "ci_lower": round(float(np.percentile(arr, 2.5)), 4),
            "ci_upper": round(float(np.percentile(arr, 97.5)), 4),
        }

    return {
        "bootstrap_iterations": n_bootstraps,
        "independent_clusters_sampled": n_groups,
        "metrics_95_ci": {
            "accuracy": ci_95(boot_accs),
            "balanced_accuracy": ci_95(boot_bal_accs),
            "f1_score": ci_95(boot_f1s),
            "roc_auc": ci_95(boot_aucs),
        },
    }


def analyze_frontier_compression_robustness(
    test_rows: List[Dict[str, Any]],
    temperature: float,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Analyze impact of social media compression variants in Frontier Benchmark."""
    # Organize by variant: original vs whatsapp vs instagram vs reshared
    variant_groups = defaultdict(list)
    for r in test_rows:
        notes = r.get("annotation_notes", "")
        variant = "original"
        if "variant: whatsapp" in notes or "images_social/whatsapp" in r["image_path"]:
            variant = "whatsapp"
        elif "variant: instagram" in notes or "images_social/instagram" in r["image_path"]:
            variant = "instagram"
        elif "variant: reshared" in notes or "images_social/reshared" in r["image_path"]:
            variant = "reshared"
        elif "variant: original" in notes or "images/" in r["image_path"]:
            variant = "original"
        variant_groups[variant].append(r)

    variant_metrics = {}
    for var_name, v_rows in variant_groups.items():
        y_true = np.array([r["ground_truth_binary"] for r in v_rows], dtype=int)
        margins = np.array([r["raw_margin"] for r in v_rows], dtype=float)
        y_prob = sigmoid(margins / temperature)
        metrics = compute_metrics_at_threshold(y_true, y_prob, threshold)
        brier = float(np.mean((y_prob - y_true) ** 2))
        metrics["brier_score"] = round(brier, 4)
        metrics["sample_count"] = len(v_rows)
        variant_metrics[var_name] = metrics

    # Variant consistency / flip analysis
    # Group by group_id
    by_family = defaultdict(dict)
    for r in test_rows:
        notes = r.get("annotation_notes", "")
        variant = "original"
        if "variant: whatsapp" in notes or "images_social/whatsapp" in r["image_path"]:
            variant = "whatsapp"
        elif "variant: instagram" in notes or "images_social/instagram" in r["image_path"]:
            variant = "instagram"
        elif "variant: reshared" in notes or "images_social/reshared" in r["image_path"]:
            variant = "reshared"
        prob = float(sigmoid(np.array([r["raw_margin"] / temperature]))[0])
        pred = 1 if prob >= threshold else 0
        by_family[r["group_id"]][variant] = {"pred": pred, "prob": prob, "gt": r["ground_truth_binary"]}

    flip_stats = {"whatsapp": 0, "instagram": 0, "reshared": 0}
    total_pairs = {"whatsapp": 0, "instagram": 0, "reshared": 0}

    for grp_id, var_dict in by_family.items():
        if "original" in var_dict:
            orig_pred = var_dict["original"]["pred"]
            for v in ["whatsapp", "instagram", "reshared"]:
                if v in var_dict:
                    total_pairs[v] += 1
                    if var_dict[v]["pred"] != orig_pred:
                        flip_stats[v] += 1

    flip_rates = {
        v: {
            "total_pairs": total_pairs[v],
            "prediction_flips": flip_stats[v],
            "flip_rate": round(flip_stats[v] / total_pairs[v], 4) if total_pairs[v] > 0 else 0.0,
        }
        for v in ["whatsapp", "instagram", "reshared"]
    }

    orig_acc = variant_metrics.get("original", {}).get("accuracy", 0.0)
    orig_auc = variant_metrics.get("original", {}).get("roc_auc", 0.0)

    delta_metrics = {}
    for v in ["whatsapp", "instagram", "reshared"]:
        if v in variant_metrics:
            delta_metrics[v] = {
                "delta_accuracy": round(variant_metrics[v]["accuracy"] - orig_acc, 4),
                "delta_roc_auc": round(variant_metrics[v]["roc_auc"] - orig_auc, 4),
            }

    return {
        "variant_evaluations": variant_metrics,
        "pairwise_consistency_against_original": flip_rates,
        "compression_degradation_deltas": delta_metrics,
    }


def analyze_cross_generator_generalization(
    tiny_test_rows: List[Dict[str, Any]],
    tiny_temp: float,
    frontier_test_rows: List[Dict[str, Any]],
    frontier_temp: float,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate cross-generator performance and seen vs unseen generalization."""
    # 1. Tiny-GenImage: Seen vs Unseen
    # Calibration generators (seen): Stable Diffusion v1.5, BigGAN, ADM, GLIDE
    # Held-out test generators (unseen): Midjourney, Wukong, VQDM
    seen_generators = {"stablediffusion", "sd", "biggan", "adm", "glide"}
    unseen_generators = {"midjourney", "wukong", "vqdm"}

    tiny_real_rows = [r for r in tiny_test_rows if r["ground_truth_binary"] == 0]
    tiny_gen_rows = defaultdict(list)
    for r in tiny_test_rows:
        if r["ground_truth_binary"] == 1:
            tiny_gen_rows[r["generator"]].append(r)

    def evaluate_slice(s_rows, temp):
        if not s_rows:
            return {}
        y_true = np.array([r["ground_truth_binary"] for r in s_rows], dtype=int)
        margins = np.array([r["raw_margin"] for r in s_rows], dtype=float)
        y_prob = sigmoid(margins / temp)
        res = compute_metrics_at_threshold(y_true, y_prob, threshold)
        res["sample_count"] = len(s_rows)
        res["ai_count"] = int((y_true == 1).sum())
        res["real_count"] = int((y_true == 0).sum())
        return res

    tiny_per_gen = {}
    seen_ai_rows = []
    unseen_ai_rows = []

    for gen, ai_rows in tiny_gen_rows.items():
        combined = ai_rows + tiny_real_rows
        tiny_per_gen[gen] = evaluate_slice(combined, tiny_temp)

        gen_clean = gen.lower().replace(" ", "").replace("_", "")
        if any(sg in gen_clean for sg in seen_generators):
            seen_ai_rows.extend(ai_rows)
        elif any(ug in gen_clean for ug in unseen_generators):
            unseen_ai_rows.extend(ai_rows)

    seen_metrics = evaluate_slice(seen_ai_rows + tiny_real_rows, tiny_temp)
    unseen_metrics = evaluate_slice(unseen_ai_rows + tiny_real_rows, tiny_temp)

    # 2. Frontier Benchmark per-generator metrics
    frontier_real_rows = [r for r in frontier_test_rows if r["ground_truth_binary"] == 0]
    frontier_gen_rows = defaultdict(list)
    for r in frontier_test_rows:
        if r["ground_truth_binary"] == 1:
            frontier_gen_rows[r["generator"]].append(r)

    frontier_per_gen = {}
    frontier_held_out_gens = {
        "openai/gpt-image-2", "blackforestlabs/flux-3", "grok-imagine-2",
        "google/nano-banana-2.1", "meta/muse-image", "xai/grok-imagine-image/v2.0"
    }

    frontier_held_out_ai_rows = []
    frontier_seen_ai_rows = []

    for gen, ai_rows in frontier_gen_rows.items():
        combined = ai_rows + frontier_real_rows
        frontier_per_gen[gen] = evaluate_slice(combined, frontier_temp)

        if gen in frontier_held_out_gens or any(h in gen for h in ["gpt-image", "flux-3", "grok", "nano-banana-2", "muse"]):
            frontier_held_out_ai_rows.extend(ai_rows)
        else:
            frontier_seen_ai_rows.extend(ai_rows)

    frontier_held_out_metrics = evaluate_slice(frontier_held_out_ai_rows + frontier_real_rows, frontier_temp)
    frontier_seen_metrics = evaluate_slice(frontier_seen_ai_rows + frontier_real_rows, frontier_temp)

    return {
        "tiny_genimage": {
            "per_generator": tiny_per_gen,
            "seen_training_generators_slice": seen_metrics,
            "unseen_held_out_generators_slice": unseen_metrics,
            "generalization_gap": {
                "accuracy_gap": round(seen_metrics.get("accuracy", 0.0) - unseen_metrics.get("accuracy", 0.0), 4),
                "balanced_accuracy_gap": round(seen_metrics.get("balanced_accuracy", 0.0) - unseen_metrics.get("balanced_accuracy", 0.0), 4),
                "roc_auc_gap": round(seen_metrics.get("roc_auc", 0.0) - unseen_metrics.get("roc_auc", 0.0), 4),
            },
        },
        "frontier_benchmark": {
            "per_generator": frontier_per_gen,
            "calibration_generator_families_slice": frontier_seen_metrics,
            "held_out_frontier_generators_slice": frontier_held_out_metrics,
            "generalization_gap": {
                "accuracy_gap": round(frontier_seen_metrics.get("accuracy", 0.0) - frontier_held_out_metrics.get("accuracy", 0.0), 4),
                "balanced_accuracy_gap": round(frontier_seen_metrics.get("balanced_accuracy", 0.0) - frontier_held_out_metrics.get("balanced_accuracy", 0.0), 4),
                "roc_auc_gap": round(frontier_seen_metrics.get("roc_auc", 0.0) - frontier_held_out_metrics.get("roc_auc", 0.0), 4),
            },
        },
    }


def generate_failure_analysis(
    all_test_rows: Dict[str, List[Dict[str, Any]]],
    temperatures: Dict[str, float],
    output_markdown_path: Path,
) -> None:
    """Generate extensive failure analysis documentation."""
    lines = [
        "# TruthLens Phase 3: Comprehensive Failure Analysis Report",
        "",
        "## 1. Executive Summary of Failure Modes",
        "",
        "This report investigates error patterns of the frozen SigLIP authenticity detector across the held-out test splits.",
        "We identify high-confidence false positives (authentic images misclassified as AI), high-confidence false negatives",
        "(AI-generated images misclassified as authentic), and borderline ambiguous cases.",
        "",
    ]

    for ds_id, rows in all_test_rows.items():
        temp = temperatures.get(ds_id, 1.0)
        fps = []
        fns = []
        borderlines = []

        for r in rows:
            prob = float(sigmoid(np.array([r["raw_margin"] / temp]))[0])
            gt = r["ground_truth_binary"]
            pred = 1 if prob >= 0.5 else 0

            item = {**r, "calibrated_ai_prob": prob}

            if gt == 0 and pred == 1:
                fps.append(item)
            elif gt == 1 and pred == 0:
                fns.append(item)

            borderlines.append((abs(prob - 0.5), item))

        fps.sort(key=lambda x: x["calibrated_ai_prob"], reverse=True)
        fns.sort(key=lambda x: x["calibrated_ai_prob"])
        borderlines.sort(key=lambda x: x[0])

        lines.extend([
            f"---",
            f"## 2. Dataset: {ds_id.upper()}",
            f"- **Held-out Test Samples**: {len(rows)}",
            f"- **False Positives**: {len(fps)} ({round(len(fps)/len(rows)*100, 2)}%)",
            f"- **False Negatives**: {len(fns)} ({round(len(fns)/len(rows)*100, 2)}%)",
            "",
            f"### Top High-Confidence False Positives (Authentic Misclassified as AI)",
            "| Image Path | Ground Truth | Calibrated AI Prob | Margin | Generator/Source | Group ID |",
            "|---|---|---|---|---|---|",
        ])

        for item in fps[:8]:
            lines.append(
                f"| `{item['image_path']}` | `{item['label']}` | `{item['calibrated_ai_prob']:.4f}` | `{item['raw_margin']:.4f}` | `{item['generator']}` | `{item['group_id']}` |"
            )

        lines.extend([
            "",
            f"### Top High-Confidence False Negatives (AI Misclassified as Authentic)",
            "| Image Path | Ground Truth | Calibrated AI Prob | Margin | Generator | Group ID |",
            "|---|---|---|---|---|---|",
        ])

        for item in fns[:8]:
            lines.append(
                f"| `{item['image_path']}` | `{item['label']}` | `{item['calibrated_ai_prob']:.4f}` | `{item['raw_margin']:.4f}` | `{item['generator']}` | `{item['group_id']}` |"
            )

        lines.extend([
            "",
            f"### Borderline / Most Ambiguous Samples (|p - 0.5| nearest 0)",
            "| Image Path | Ground Truth | Calibrated AI Prob | Margin | Generator | Group ID |",
            "|---|---|---|---|---|---|",
        ])

        for _, item in borderlines[:6]:
            lines.append(
                f"| `{item['image_path']}` | `{item['label']}` | `{item['calibrated_ai_prob']:.4f}` | `{item['raw_margin']:.4f}` | `{item['generator']}` | `{item['group_id']}` |"
            )
        lines.append("")

    lines.extend([
        "## 3. Systematic Architectural Vulnerabilities",
        "",
        "1. **Frequency Domain Blindspots**: As a patch-based Vision Transformer with 16x16 pixel patches and standard bicubic resizing to 224x224, the SigLIP backbone loses high-frequency Fourier artifacts and periodic resampling grids common in newer diffusion/autoregressive models.",
        "2. **Social Media Compression Degradation**: JPEG recompression (e.g. WhatsApp, Instagram) smooths subtle generation artifacts, pushing AI-generated images toward authentic scores.",
        "3. **Domain Shift & Stylistic Overlap**: Authentic images with heavy studio lighting, high contrast, or digital HDR post-processing frequently trigger false positive AI classifications.",
        "4. **Cross-Architecture Fragility**: Models trained on older diffusion (e.g. SD 1.5, BigGAN) experience sharp accuracy degradation when encountering modern rectified flow architectures (e.g. Flux-3) or large multimodal diffusion (e.g. GPT-Image-2, Grok Imagine 2).",
    ])

    output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_markdown_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info("Saved failure analysis to %s", output_markdown_path)


def generate_evaluation_figures(
    test_data: Dict[str, Dict[str, Any]],
    robustness_data: Dict[str, Any],
    generalization_data: Dict[str, Any],
    calibration_data: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate high-resolution evaluation figures using Matplotlib."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. ROC Curves Across Datasets
    plt.figure(figsize=(8, 6), dpi=300)
    for ds_id, data in test_data.items():
        y_true = data["y_true"]
        y_prob = data["y_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{ds_id.replace('_', ' ').title()} (AUC = {auc:.3f})", lw=2)

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title("TruthLens Detector: ROC Curves Across Held-Out Test Splits", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png")
    plt.close()

    # 2. Precision-Recall Curves
    plt.figure(figsize=(8, 6), dpi=300)
    for ds_id, data in test_data.items():
        y_true = data["y_true"]
        y_prob = data["y_prob"]
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(rec, prec, label=f"{ds_id.replace('_', ' ').title()} (AP = {ap:.3f})", lw=2)

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("TruthLens Detector: Precision-Recall Curves", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curves.png")
    plt.close()

    # 3. Compression Robustness (Frontier Benchmark)
    var_evals = robustness_data.get("variant_evaluations", {})
    if var_evals:
        variants = ["original", "whatsapp", "instagram", "reshared"]
        accs = [var_evals.get(v, {}).get("accuracy", 0.0) * 100 for v in variants]
        aucs = [var_evals.get(v, {}).get("roc_auc", 0.0) * 100 for v in variants]

        x = np.arange(len(variants))
        width = 0.35

        plt.figure(figsize=(8, 5), dpi=300)
        plt.bar(x - width/2, accs, width, label="Accuracy (%)", color="#1f77b4")
        plt.bar(x + width/2, aucs, width, label="ROC-AUC (%)", color="#ff7f0e")

        plt.xlabel("Compression Variant", fontsize=12)
        plt.ylabel("Score (%)", fontsize=12)
        plt.title("Frontier Benchmark: Robustness to Social Media Compression", fontsize=14, fontweight="bold")
        plt.xticks(x, [v.capitalize() for v in variants], fontsize=11)
        plt.ylim(0, 105)
        plt.legend(loc="lower right", fontsize=11)
        plt.tight_layout()
        plt.savefig(output_dir / "compression_robustness.png")
        plt.close()

    # 4. Cross-Generator Generalization Bar Chart
    gen_data = generalization_data.get("tiny_genimage", {})
    if "seen_training_generators_slice" in gen_data and "unseen_held_out_generators_slice" in gen_data:
        seen = gen_data["seen_training_generators_slice"]
        unseen = gen_data["unseen_held_out_generators_slice"]

        categories = ["Accuracy", "Balanced Acc", "F1-Score", "ROC-AUC"]
        seen_vals = [
            seen.get("accuracy", 0.0) * 100,
            seen.get("balanced_accuracy", 0.0) * 100,
            seen.get("f1_score", 0.0) * 100,
            seen.get("roc_auc", 0.0) * 100,
        ]
        unseen_vals = [
            unseen.get("accuracy", 0.0) * 100,
            unseen.get("balanced_accuracy", 0.0) * 100,
            unseen.get("f1_score", 0.0) * 100,
            unseen.get("roc_auc", 0.0) * 100,
        ]

        x = np.arange(len(categories))
        width = 0.35

        plt.figure(figsize=(9, 5), dpi=300)
        plt.bar(x - width/2, seen_vals, width, label="Seen Generators (SD1.5, BigGAN, ADM, GLIDE)", color="#2ca02c")
        plt.bar(x + width/2, unseen_vals, width, label="Unseen Generators (Midjourney, Wukong, VQDM)", color="#d62728")

        plt.ylabel("Percentage (%)", fontsize=12)
        plt.title("Tiny-GenImage: Seen vs Unseen Generator Architecture Generalization", fontsize=13, fontweight="bold")
        plt.xticks(x, categories, fontsize=11)
        plt.ylim(0, 105)
        plt.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        plt.savefig(output_dir / "generator_generalization.png")
        plt.close()

    # 5. Reliability Diagrams (Calibration Curve)
    calib_ds = calibration_data.get("datasets", {}).get("tiny_genimage", {})
    test_calib = calib_ds.get("calibration_diagnostics", {}).get("test_split", {})
    if test_calib:
        uncal_bins = test_calib.get("uncalibrated_T1", {}).get("reliability_bins", [])
        cal_bins = test_calib.get("calibrated_T_fit", {}).get("reliability_bins", [])

        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

        if uncal_bins:
            x_u = [b["mean_confidence"] for b in uncal_bins if b["mean_confidence"] is not None]
            y_u = [b["empirical_accuracy"] for b in uncal_bins if b["mean_confidence"] is not None]
            plt.plot(x_u, y_u, "s-", color="#d62728", lw=2, label=f"Uncalibrated (ECE = {test_calib.get('uncalibrated_T1', {}).get('ece_equal_width', 0):.3f})")

        if cal_bins:
            x_c = [b["mean_confidence"] for b in cal_bins if b["mean_confidence"] is not None]
            y_c = [b["empirical_accuracy"] for b in cal_bins if b["mean_confidence"] is not None]
            plt.plot(x_c, y_c, "o-", color="#1f77b4", lw=2, label=f"Temperature Scaled (ECE = {test_calib.get('calibrated_T_fit', {}).get('ece_equal_width', 0):.3f})")

        plt.xlabel("Mean Predicted Confidence", fontsize=12)
        plt.ylabel("Empirical Accuracy", fontsize=12)
        plt.title("Reliability Diagram: Tiny-GenImage Held-Out Test Split", fontsize=14, fontweight="bold")
        plt.legend(loc="upper left", fontsize=11)
        plt.tight_layout()
        plt.savefig(output_dir / "reliability_diagrams.png")
        plt.close()

    # 6. Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    for ax, (ds_id, data) in zip(axes, test_data.items()):
        cm = data["metrics_default"]["confusion_matrix"]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        im = ax.imshow(mat, cmap="Blues", interpolation="nearest")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center", color="white" if mat[i, j] > mat.max()/2 else "black", fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Real", "AI"], fontsize=10)
        ax.set_yticklabels(["Real", "AI"], fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title(f"{ds_id.replace('_', ' ').title()}", fontsize=12, fontweight="bold")

    plt.suptitle("TruthLens Detector: Confusion Matrices on Held-Out Test Splits", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png")
    plt.close()

    log.info("Saved all evaluation figures to %s", output_dir)


def main():
    parser = argparse.ArgumentParser(description="TruthLens Phase 3 Results Analysis")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "phase3",
    )
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "reports" / "phase3_calibration.json",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "reports",
    )
    args = parser.parse_args()

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Calibration Report
    if not args.calibration_report.is_file():
        raise FileNotFoundError(f"Calibration report not found: {args.calibration_report}. Run calibrate_evaluation.py first.")

    with open(args.calibration_report, "r", encoding="utf-8") as f:
        calib_artifact = json.load(f)

    fitted_temps = calib_artifact["fitted_temperatures"]

    # 2. Load Predictions CSVs
    dataset_files = {
        "tiny_genimage": args.results_dir / "tiny_genimage_predictions.csv",
        "frontier_benchmark_2026_10": args.results_dir / "frontier_predictions.csv",
        "wildaigc_bench": args.results_dir / "wildaigc_predictions.csv",
    }

    all_data = {}
    for ds_id, path in dataset_files.items():
        all_data[ds_id] = load_dataset_predictions(path)

    # Filter to held-out test splits
    test_rows_by_dataset = {}
    for ds_id, rows in all_data.items():
        test_rows_by_dataset[ds_id] = [r for r in rows if r["evaluation_role"] == "test"]
        log.info("[%s] Test samples: %d", ds_id, len(test_rows_by_dataset[ds_id]))

    # 3. Evaluate Locked Held-Out Test Splits
    test_eval_results = {}
    figure_test_data = {}

    for ds_id, t_rows in test_rows_by_dataset.items():
        temp = fitted_temps.get(ds_id, 1.0)
        y_true = np.array([r["ground_truth_binary"] for r in t_rows], dtype=int)
        margins = np.array([r["raw_margin"] for r in t_rows], dtype=float)
        y_prob_calib = sigmoid(margins / temp)

        # Retrieve validation-selected operating thresholds
        ds_calib_info = calib_artifact.get("datasets", {}).get(ds_id, {})
        val_thresh = ds_calib_info.get("validation_operating_thresholds", {})
        tau_bal = val_thresh.get("optimal_balanced_accuracy", {}).get("threshold", 0.5)
        tau_f1 = val_thresh.get("optimal_f1", {}).get("threshold", 0.5)
        tau_spec95 = val_thresh.get("high_specificity_95", {}).get("threshold", 0.5)
        auth_max = val_thresh.get("selective_abstention", {}).get("authentic_max_threshold", 0.35)
        ai_min = val_thresh.get("selective_abstention", {}).get("ai_generated_min_threshold", 0.65)

        # Metrics at candidate operating points
        met_default = compute_metrics_at_threshold(y_true, y_prob_calib, 0.5)
        met_bal = compute_metrics_at_threshold(y_true, y_prob_calib, tau_bal)
        met_f1 = compute_metrics_at_threshold(y_true, y_prob_calib, tau_f1)
        met_spec95 = compute_metrics_at_threshold(y_true, y_prob_calib, tau_spec95)

        # Selective abstention evaluation
        met_abstain = compute_abstention_metrics(y_true, y_prob_calib, auth_max, ai_min)

        # Group-aware bootstrap CI (1,000 resamples)
        boot_ci = group_aware_bootstrap_ci(t_rows, temp, threshold=0.5, n_bootstraps=1000)

        test_eval_results[ds_id] = {
            "dataset_id": ds_id,
            "test_sample_count": len(t_rows),
            "applied_temperature": temp,
            "validation_selected_thresholds": {
                "optimal_balanced_acc_threshold": tau_bal,
                "optimal_f1_threshold": tau_f1,
                "high_specificity_95_threshold": tau_spec95,
                "selective_abstention_bounds": [auth_max, ai_min],
            },
            "operating_point_evaluations": {
                "default_threshold_0_5": met_default,
                "optimal_balanced_acc_operating_point": met_bal,
                "optimal_f1_operating_point": met_f1,
                "high_specificity_operating_point": met_spec95,
            },
            "selective_abstention_evaluation": met_abstain,
            "group_aware_bootstrap_95_ci": boot_ci,
        }

        figure_test_data[ds_id] = {
            "y_true": y_true,
            "y_prob": y_prob_calib,
            "metrics_default": met_default,
        }

    # 4. Frontier Compression Robustness Analysis
    log.info("Analyzing Frontier social media compression robustness...")
    robustness_report = analyze_frontier_compression_robustness(
        test_rows_by_dataset["frontier_benchmark_2026_10"],
        fitted_temps.get("frontier_benchmark_2026_10", 1.0),
        threshold=0.5,
    )
    with open(args.reports_dir / "phase3_robustness.json", "w", encoding="utf-8") as f:
        json.dump(robustness_report, f, indent=2)

    # 5. Cross-Generator Generalization Analysis
    log.info("Analyzing cross-generator generalization...")
    generalization_report = analyze_cross_generator_generalization(
        test_rows_by_dataset["tiny_genimage"],
        fitted_temps.get("tiny_genimage", 1.0),
        test_rows_by_dataset["frontier_benchmark_2026_10"],
        fitted_temps.get("frontier_benchmark_2026_10", 1.0),
        threshold=0.5,
    )
    with open(args.reports_dir / "phase3_generalization.json", "w", encoding="utf-8") as f:
        json.dump(generalization_report, f, indent=2)

    # 6. Detailed Failure Analysis
    log.info("Generating failure analysis markdown...")
    generate_failure_analysis(
        test_rows_by_dataset,
        fitted_temps,
        args.reports_dir / "phase3_failure_analysis.md",
    )

    # 7. Generate Evaluation Figures
    log.info("Generating publication figures...")
    generate_evaluation_figures(
        figure_test_data,
        robustness_report,
        generalization_report,
        calib_artifact,
        figures_dir,
    )

    # 8. Persist Master Results JSON
    master_results = {
        "execution_date": datetime.now(timezone.utc).isoformat(),
        "phase": "Phase 3: Controlled Benchmark Evaluation & Calibration",
        "model_under_test": {
            "model_id": "truthlens/siglip-cifake-frozen-head",
            "backbone_model": "google/siglip-base-patch16-224",
            "backbone_revision": "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
            "head_sha256": "e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538",
            "preprocessing_version": "siglip-frozen-v1",
        },
        "locked_test_evaluations": test_eval_results,
        "robustness_summary": robustness_report["compression_degradation_deltas"],
        "generalization_summary": {
            "tiny_genimage_gap": generalization_report["tiny_genimage"]["generalization_gap"],
            "frontier_gap": generalization_report["frontier_benchmark"]["generalization_gap"],
        },
    }

    with open(args.reports_dir / "phase3_results.json", "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)

    # 9. Generate Master Results Markdown
    md_lines = [
        "# TruthLens Phase 3: Controlled Benchmark Evaluation Report",
        "",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        "**Target Detector:** `truthlens/siglip-cifake-frozen-head`  ",
        "**Backbone:** `google/siglip-base-patch16-224` (`7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`)  ",
        "**Classification Head:** `models/baselines/siglip_frozen_head.pt` (`e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`)  ",
        "**Evaluation Protocol:** Group-Aware Leakage-Safe Benchmarks (Tiny-GenImage, Frontier Benchmark 2026-10, WildAIGC-Bench)  ",
        "",
        "---",
        "",
        "## 1. Locked Held-Out Test Evaluation Results",
        "",
        "All test evaluations were conducted on strictly held-out partitions with frozen thresholds selected on validation data.",
        "",
        "| Dataset | Test Samples | Accuracy (95% CI) | Balanced Acc (95% CI) | F1-Score (95% CI) | ROC-AUC (95% CI) | PR-AUC | FAR | FRR |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for ds_id, res in test_eval_results.items():
        met = res["operating_point_evaluations"]["default_threshold_0_5"]
        ci = res["group_aware_bootstrap_95_ci"]["metrics_95_ci"]
        acc_str = f"{met['accuracy']:.4f} [{ci['accuracy']['ci_lower']:.4f}, {ci['accuracy']['ci_upper']:.4f}]"
        bal_str = f"{met['balanced_accuracy']:.4f} [{ci['balanced_accuracy']['ci_lower']:.4f}, {ci['balanced_accuracy']['ci_upper']:.4f}]"
        f1_str = f"{met['f1_score']:.4f} [{ci['f1_score']['ci_lower']:.4f}, {ci['f1_score']['ci_upper']:.4f}]"
        auc_str = f"{met['roc_auc']:.4f} [{ci['roc_auc']['ci_lower']:.4f}, {ci['roc_auc']['ci_upper']:.4f}]"

        md_lines.append(
            f"| **{ds_id}** | {res['test_sample_count']} | {acc_str} | {bal_str} | {f1_str} | {auc_str} | {met['pr_auc']:.4f} | {met['false_acceptance_rate']:.4f} | {met['false_rejection_rate']:.4f} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Confidence Calibration & Temperature Scaling",
        "",
        "Temperature parameters were fitted strictly on the calibration partitions to minimize NLL.",
        "",
        "| Dataset | Fitted T | Uncalibrated ECE (15-bin) | Calibrated ECE (15-bin) | Uncalibrated Brier | Calibrated Brier | Uncalibrated NLL | Calibrated NLL |",
        "|---|---|---|---|---|---|---|---|",
    ])

    for ds_id in test_rows_by_dataset:
        ds_cal = calib_artifact["datasets"].get(ds_id, {})
        t_val = ds_cal.get("fitted_temperature", 1.0)
        t_diag = ds_cal.get("calibration_diagnostics", {}).get("test_split", {})
        uncal = t_diag.get("uncalibrated_T1", {})
        cal = t_diag.get("calibrated_T_fit", {})

        md_lines.append(
            f"| **{ds_id}** | {t_val:.4f} | {uncal.get('ece_equal_width', 0):.4f} | {cal.get('ece_equal_width', 0):.4f} | {uncal.get('brier_score', 0):.4f} | {cal.get('brier_score', 0):.4f} | {uncal.get('nll', 0):.4f} | {cal.get('nll', 0):.4f} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Social Media Compression Robustness (Frontier Benchmark)",
        "",
        "Impact of standard social platform compression pipelines on detection performance:",
        "",
        "| Variant | Samples | Accuracy | Balanced Acc | F1-Score | ROC-AUC | Delta Acc vs Orig | Flip Rate vs Orig |",
        "|---|---|---|---|---|---|---|---|",
    ])

    v_evals = robustness_report["variant_evaluations"]
    v_flips = robustness_report["pairwise_consistency_against_original"]
    v_deltas = robustness_report["compression_degradation_deltas"]

    for v in ["original", "whatsapp", "instagram", "reshared"]:
        if v in v_evals:
            e = v_evals[v]
            d_acc = f"{v_deltas[v]['delta_accuracy']:+.4f}" if v in v_deltas else "Ref (0.000)"
            flip = f"{v_flips[v]['flip_rate']:.4f}" if v in v_flips else "Ref (0.000)"
            md_lines.append(
                f"| **{v.capitalize()}** | {e['sample_count']} | {e['accuracy']:.4f} | {e['balanced_accuracy']:.4f} | {e['f1_score']:.4f} | {e['roc_auc']:.4f} | {d_acc} | {flip} |"
            )

    md_lines.extend([
        "",
        "---",
        "",
        "## 4. Cross-Generator Generalization",
        "",
        "### Tiny-GenImage: Seen vs Unseen Architectures",
        f"- **Seen Training Generators (SD1.5, BigGAN, ADM, GLIDE):** Accuracy = {generalization_report['tiny_genimage']['seen_training_generators_slice'].get('accuracy', 0):.4f}, Balanced Acc = {generalization_report['tiny_genimage']['seen_training_generators_slice'].get('balanced_accuracy', 0):.4f}, AUC = {generalization_report['tiny_genimage']['seen_training_generators_slice'].get('roc_auc', 0):.4f}",
        f"- **Unseen Held-Out Generators (Midjourney, Wukong, VQDM):** Accuracy = {generalization_report['tiny_genimage']['unseen_held_out_generators_slice'].get('accuracy', 0):.4f}, Balanced Acc = {generalization_report['tiny_genimage']['unseen_held_out_generators_slice'].get('balanced_accuracy', 0):.4f}, AUC = {generalization_report['tiny_genimage']['unseen_held_out_generators_slice'].get('roc_auc', 0):.4f}",
        f"- **Generalization Gap:** Accuracy drop of {generalization_report['tiny_genimage']['generalization_gap']['accuracy_gap']:.4f}, AUC drop of {generalization_report['tiny_genimage']['generalization_gap']['roc_auc_gap']:.4f}",
        "",
        "### Frontier Benchmark: Seen vs Held-Out Frontier Generators",
        f"- **Calibration Families Slice:** Accuracy = {generalization_report['frontier_benchmark']['calibration_generator_families_slice'].get('accuracy', 0):.4f}, AUC = {generalization_report['frontier_benchmark']['calibration_generator_families_slice'].get('roc_auc', 0):.4f}",
        f"- **Held-Out Frontier Slice (GPT-Image-2, Flux-3, Grok-Imagine-2, etc.):** Accuracy = {generalization_report['frontier_benchmark']['held_out_frontier_generators_slice'].get('accuracy', 0):.4f}, AUC = {generalization_report['frontier_benchmark']['held_out_frontier_generators_slice'].get('roc_auc', 0):.4f}",
        f"- **Generalization Gap:** Accuracy drop of {generalization_report['frontier_benchmark']['generalization_gap']['accuracy_gap']:.4f}, AUC drop of {generalization_report['frontier_benchmark']['generalization_gap']['roc_auc_gap']:.4f}",
        "",
        "---",
        "",
        "## 5. Artifact Directory & Visualizations",
        "",
        "- Master Results: `experiments/reports/phase3_results.json`",
        "- Calibration Diagnostics: `experiments/reports/phase3_calibration.json`",
        "- Robustness Report: `experiments/reports/phase3_robustness.json`",
        "- Generalization Report: `experiments/reports/phase3_generalization.json`",
        "- Detailed Failure Analysis: `experiments/reports/phase3_failure_analysis.md`",
        "- Figures:",
        "  - `experiments/reports/figures/roc_curves.png`",
        "  - `experiments/reports/figures/pr_curves.png`",
        "  - `experiments/reports/figures/confusion_matrices.png`",
        "  - `experiments/reports/figures/reliability_diagrams.png`",
        "  - `experiments/reports/figures/compression_robustness.png`",
        "  - `experiments/reports/figures/generator_generalization.png`",
    ])

    with open(args.reports_dir / "phase3_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    log.info("Saved master results to %s and %s", args.reports_dir / "phase3_results.json", args.reports_dir / "phase3_results.md")


if __name__ == "__main__":
    main()
