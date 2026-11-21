"""
TruthLens Phase 6: Statistical Comparison, Group Bootstrap & Publication Figures.

Computes:
1. Paired metric differences (E3 - Baseline) on identical test samples.
2. Group-aware stratified bootstrap confidence intervals (1,000 resamples).
3. Step 9 Candidate Decision Report.
4. Generates 8 publication-quality evaluation figures.

Saves:
- experiments/reports/phase6_statistical_comparison.json
- experiments/reports/phase6_candidate_decision.json
- experiments/reports/figures/phase6_roc_curves.png
- experiments/reports/figures/phase6_pr_curves.png
- experiments/reports/figures/phase6_confusion_matrices.png
- experiments/reports/figures/phase6_reliability_diagrams.png
- experiments/reports/figures/phase6_generator_performance.png
- experiments/reports/figures/phase6_encoding_robustness.png
- experiments/reports/figures/phase6_compression_robustness.png
- experiments/reports/figures/phase6_paired_metric_differences.png
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("statistical_comparison_phase6")


def load_predictions(csv_path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["ground_truth_binary"] = int(r["ground_truth_binary"])
            r["raw_logit"] = float(r["raw_logit"])
            r["uncalibrated_probability"] = float(r["uncalibrated_probability"])
            r["predicted_binary"] = int(r["predicted_binary"])
            rows.append(r)
    return rows


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, probs))
    except Exception:
        auc = 0.5
    try:
        pr_auc = float(average_precision_score(y_true, probs))
    except Exception:
        pr_auc = float(np.mean(y_true))

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1_score": f1,
        "roc_auc": auc,
        "pr_auc": pr_auc,
    }


def stratified_group_bootstrap(
    records_base: List[Dict[str, Any]],
    records_e3: List[Dict[str, Any]],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Perform group-aware stratified bootstrap to compute paired delta 95% CIs."""
    rng = np.random.default_rng(seed)

    # Align rows by sample_id or index
    n_samples = len(records_base)
    assert n_samples == len(records_e3)

    # Grouping: split into Real vs AI groups
    real_indices = [i for i, r in enumerate(records_base) if r["ground_truth_binary"] == 0]
    ai_indices = [i for i, r in enumerate(records_base) if r["ground_truth_binary"] == 1]

    # Organize AI samples by group_id
    group_map = defaultdict(list)
    for i in ai_indices:
        group_map[records_base[i].get("group_id", "default")].append(i)
    unique_groups = list(group_map.keys())

    delta_metrics = defaultdict(list)
    base_metrics_list = defaultdict(list)
    e3_metrics_list = defaultdict(list)

    y_all = np.array([r["ground_truth_binary"] for r in records_base], dtype=int)
    probs_b = np.array([r["uncalibrated_probability"] for r in records_base], dtype=float)
    probs_e3 = np.array([r["uncalibrated_probability"] for r in records_e3], dtype=float)

    for b in range(n_resamples):
        # Resample real indices with replacement
        resampled_real = rng.choice(real_indices, size=len(real_indices), replace=True)

        # Resample AI groups with replacement
        resampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        resampled_ai = []
        for g in resampled_groups:
            resampled_ai.extend(group_map[g])

        boot_idx = np.concatenate([resampled_real, resampled_ai])

        y_boot = y_all[boot_idx]
        pb_boot = probs_b[boot_idx]
        pe3_boot = probs_e3[boot_idx]

        mb = compute_metrics(y_boot, pb_boot)
        me3 = compute_metrics(y_boot, pe3_boot)

        for k in mb:
            base_metrics_list[k].append(mb[k])
            e3_metrics_list[k].append(me3[k])
            delta_metrics[k].append(me3[k] - mb[k])

    def calc_ci(vals: List[float]) -> Dict[str, float]:
        return {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "ci_95_low": round(float(np.percentile(vals, 2.5)), 4),
            "ci_95_high": round(float(np.percentile(vals, 97.5)), 4),
        }

    summary = {}
    for k in delta_metrics:
        summary[k] = {
            "baseline": calc_ci(base_metrics_list[k]),
            "e3_candidate": calc_ci(e3_metrics_list[k]),
            "delta": calc_ci(delta_metrics[k]),
        }
    return summary


def generate_publication_figures(
    datasets_data: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
    domain_data: Dict[str, Any],
    gen_data: Dict[str, Any],
    comp_data: Dict[str, Any],
    stat_data: Dict[str, Any],
    figures_dir: Path,
) -> None:
    """Generate all 8 required publication-quality figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {"baseline": "#2b5c8f", "e3": "#d95f02", "real": "#2ca02c", "ai": "#d62728"}

    # 1. ROC Curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ds_titles = {
        "tiny_genimage": "Tiny-GenImage (N=7,000)",
        "frontier_benchmark_2026_10": "Frontier Benchmark (N=2,091)",
        "wildaigc_bench": "WildAIGC-Bench (N=1,294)",
    }
    for idx, (name, (rb, re3)) in enumerate(datasets_data.items()):
        ax = axes[idx]
        y_true = np.array([r["ground_truth_binary"] for r in rb])
        pb = np.array([r["uncalibrated_probability"] for r in rb])
        pe3 = np.array([r["uncalibrated_probability"] for r in re3])

        fpr_b, tpr_b, _ = roc_curve(y_true, pb)
        fpr_e3, tpr_e3, _ = roc_curve(y_true, pe3)
        auc_b = roc_auc_score(y_true, pb)
        auc_e3 = roc_auc_score(y_true, pe3)

        ax.plot(fpr_b, tpr_b, label=f"Baseline (AUC={auc_b:.3f})", color=colors["baseline"], linewidth=2)
        ax.plot(fpr_e3, tpr_e3, label=f"E3 Candidate (AUC={auc_e3:.3f})", color=colors["e3"], linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_title(ds_titles[name], fontsize=11, fontweight="bold")
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(figures_dir / "phase6_roc_curves.png", dpi=300)
    plt.close()

    # 2. Precision-Recall Curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, (name, (rb, re3)) in enumerate(datasets_data.items()):
        ax = axes[idx]
        y_true = np.array([r["ground_truth_binary"] for r in rb])
        pb = np.array([r["uncalibrated_probability"] for r in rb])
        pe3 = np.array([r["uncalibrated_probability"] for r in re3])

        pb_prec, pb_rec, _ = precision_recall_curve(y_true, pb)
        pe3_prec, pe3_rec, _ = precision_recall_curve(y_true, pe3)
        ap_b = average_precision_score(y_true, pb)
        ap_e3 = average_precision_score(y_true, pe3)

        ax.plot(pb_rec, pb_prec, label=f"Baseline (AP={ap_b:.3f})", color=colors["baseline"], linewidth=2)
        ax.plot(pe3_rec, pe3_prec, label=f"E3 Candidate (AP={ap_e3:.3f})", color=colors["e3"], linewidth=2)
        ax.set_title(ds_titles[name], fontsize=11, fontweight="bold")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(figures_dir / "phase6_pr_curves.png", dpi=300)
    plt.close()

    # 3. Confusion Matrices
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for col_idx, (name, (rb, re3)) in enumerate(datasets_data.items()):
        y_true = np.array([r["ground_truth_binary"] for r in rb])
        pb = (np.array([r["uncalibrated_probability"] for r in rb]) >= 0.5).astype(int)
        pe3 = (np.array([r["uncalibrated_probability"] for r in re3]) >= 0.5).astype(int)

        cm_b = confusion_matrix(y_true, pb, labels=[0, 1])
        cm_e3 = confusion_matrix(y_true, pe3, labels=[0, 1])

        # Plot Baseline
        ax_top = axes[0, col_idx]
        im0 = ax_top.imshow(cm_b, cmap="Blues", interpolation="nearest")
        ax_top.set_title(f"Baseline - {name}\n(BalAcc: {balanced_accuracy_score(y_true, pb):.3f})", fontsize=10, fontweight="bold")
        ax_top.set_xticks([0, 1])
        ax_top.set_yticks([0, 1])
        ax_top.set_xticklabels(["Real", "AI"])
        ax_top.set_yticklabels(["Real", "AI"])
        for i in range(2):
            for j in range(2):
                ax_top.text(j, i, f"{cm_b[i, j]}", ha="center", va="center", color="black" if cm_b[i, j] < cm_b.max()/2 else "white")

        # Plot E3
        ax_bot = axes[1, col_idx]
        im1 = ax_bot.imshow(cm_e3, cmap="Oranges", interpolation="nearest")
        ax_bot.set_title(f"E3 Candidate - {name}\n(BalAcc: {balanced_accuracy_score(y_true, pe3):.3f})", fontsize=10, fontweight="bold")
        ax_bot.set_xticks([0, 1])
        ax_bot.set_yticks([0, 1])
        ax_bot.set_xticklabels(["Real", "AI"])
        ax_bot.set_yticklabels(["Real", "AI"])
        for i in range(2):
            for j in range(2):
                ax_bot.text(j, i, f"{cm_e3[i, j]}", ha="center", va="center", color="black" if cm_e3[i, j] < cm_e3.max()/2 else "white")

    plt.tight_layout()
    plt.savefig(figures_dir / "phase6_confusion_matrices.png", dpi=300)
    plt.close()

    # 4. Reliability Diagrams
    rel_path = Path("experiments/reports/phase6_reliability_data.json")
    if rel_path.exists():
        with open(rel_path, "r", encoding="utf-8") as f:
            rel_json = json.load(f)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for idx, (name, title) in enumerate([("tiny_genimage", "Tiny-GenImage"), ("frontier_benchmark_2026_10", "Frontier"), ("pooled", "Pooled Calibration")]):
            ax = axes[idx]
            d = rel_json.get(name, {})
            u_bins = [b for b in d.get("uncalibrated_bins", []) if b["count"] > 0]
            c_bins = [b for b in d.get("calibrated_bins", []) if b["count"] > 0]

            if u_bins:
                u_conf = [b["mean_confidence"] for b in u_bins]
                u_acc = [b["empirical_accuracy"] for b in u_bins]
                ax.plot(u_conf, u_acc, "s--", label="Uncalibrated (T=1.0)", color="#7f7f7f", alpha=0.8)

            if c_bins:
                c_conf = [b["mean_confidence"] for b in c_bins]
                c_acc = [b["empirical_accuracy"] for b in c_bins]
                ax.plot(c_conf, c_acc, "o-", label=f"Calibrated (T={d.get('temperature', 1.0):.2f})", color=colors["e3"], linewidth=2)

            ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
            ax.set_title(f"Reliability Diagram: {title}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Mean Confidence")
            ax.set_ylabel("Empirical Accuracy")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(figures_dir / "phase6_reliability_diagrams.png", dpi=300)
        plt.close()

    # 5. Generator-Level Performance
    tiny_gens = gen_data.get("tiny_genimage_generators", {})
    gen_names = list(tiny_gens.keys())
    if gen_names:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(gen_names))
        width = 0.35

        base_rec = [tiny_gens[g]["baseline"]["recall"] * 100 for g in gen_names]
        e3_rec = [tiny_gens[g]["e3_candidate"]["recall"] * 100 for g in gen_names]

        rects1 = ax.bar(x - width/2, base_rec, width, label="Baseline Recall (%)", color=colors["baseline"])
        rects2 = ax.bar(x + width/2, e3_rec, width, label="E3 Candidate Recall (%)", color=colors["e3"])

        # Label categories
        labels_cat = [f"{g}\n({tiny_gens[g]['category']})" for g in gen_names]
        ax.set_xticks(x)
        ax.set_xticklabels(labels_cat, fontsize=9)
        ax.set_ylabel("AI Recall Rate (%)", fontsize=11)
        ax.set_title("Cross-Generator Generalization: Seen vs Held-Out Unseen Generators", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 115)
        ax.legend(loc="upper right", fontsize=10)

        # Annotate bars
        for r in rects1:
            ax.annotate(f"{r.get_height():.1f}%", xy=(r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for r in rects2:
            ax.annotate(f"{r.get_height():.1f}%", xy=(r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        plt.savefig(figures_dir / "phase6_generator_performance.png", dpi=300)
        plt.close()

    # 6. Encoding Robustness (Transcoding Flips)
    mf = domain_data.get("matched_format_evaluation", {})
    if mf:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        enc_variants = ["JPEG (Q=90)", "JPEG (Q=75)", "PNG (Lossless)"]
        e3_flips = [
            mf["e3_candidate_robustness"]["jpeg90"]["flip_rate_pct"],
            mf["e3_candidate_robustness"]["jpeg75"]["flip_rate_pct"],
            mf["e3_candidate_robustness"]["png"]["flip_rate_pct"],
        ]
        base_flips = [
            mf["baseline_robustness"]["jpeg90"]["flip_rate_pct"],
            mf["baseline_robustness"]["jpeg75"]["flip_rate_pct"],
            mf["baseline_robustness"]["png"]["flip_rate_pct"],
        ]

        x = np.arange(len(enc_variants))
        width = 0.35
        r1 = ax1.bar(x - width/2, base_flips, width, label="Baseline", color=colors["baseline"])
        r2 = ax1.bar(x + width/2, e3_flips, width, label="E3 Candidate", color=colors["e3"])
        ax1.set_ylabel("Prediction Flip Rate (%)", fontsize=11)
        ax1.set_title("Causal Impact of Transcoding: Prediction Flip Rate", fontsize=11, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(enc_variants, fontsize=10)
        ax1.set_ylim(0, 60)
        ax1.legend(loc="upper right")
        for r in r1:
            ax1.annotate(f"{r.get_height():.1f}%", xy=(r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for r in r2:
            ax1.annotate(f"{r.get_height():.1f}%", xy=(r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        # Logit Shifts
        e3_shifts = [
            mf["e3_candidate_robustness"]["jpeg90"]["mean_absolute_logit_shift"],
            mf["e3_candidate_robustness"]["jpeg75"]["mean_absolute_logit_shift"],
            mf["e3_candidate_robustness"]["png"]["mean_absolute_logit_shift"],
        ]
        base_shifts = [
            mf["baseline_robustness"]["jpeg90"]["mean_absolute_logit_shift"],
            mf["baseline_robustness"]["jpeg75"]["mean_absolute_logit_shift"],
            mf["baseline_robustness"]["png"]["mean_absolute_logit_shift"],
        ]
        r3 = ax2.bar(x - width/2, base_shifts, width, label="Baseline", color=colors["baseline"])
        r4 = ax2.bar(x + width/2, e3_shifts, width, label="E3 Candidate", color=colors["e3"])
        ax2.set_ylabel("Mean |Δ Logit| Shift", fontsize=11)
        ax2.set_title("Magnitude of Logit Perturbation from Encoding", fontsize=11, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(enc_variants, fontsize=10)
        ax2.legend(loc="upper right")
        for r in r3:
            ax2.annotate(f"{r.get_height():.2f}", xy=(r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for r in r4:
            ax2.annotate(f"{r.get_height():.2f}", xy=(r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        plt.savefig(figures_dir / "phase6_encoding_robustness.png", dpi=300)
        plt.close()

    # 7. Compression Robustness (Frontier)
    if comp_data:
        variants = sorted(list(comp_data.keys()))
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(variants))
        width = 0.35

        b_auc = [comp_data[v]["baseline"]["roc_auc"] for v in variants]
        e3_auc = [comp_data[v]["e3_candidate"]["roc_auc"] for v in variants]

        rects1 = ax.bar(x - width/2, b_auc, width, label="Baseline ROC-AUC", color=colors["baseline"])
        rects2 = ax.bar(x + width/2, e3_auc, width, label="E3 Candidate ROC-AUC", color=colors["e3"])

        ax.set_ylabel("ROC-AUC", fontsize=11)
        ax.set_title("Frontier Benchmark: Social Media Compression Robustness", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([v.capitalize() for v in variants], fontsize=11)
        ax.set_ylim(0, 0.85)
        ax.legend(loc="upper right", fontsize=10)

        for r in rects1:
            ax.annotate(f"{r.get_height():.3f}", xy=(r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for r in rects2:
            ax.annotate(f"{r.get_height():.3f}", xy=(r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        plt.savefig(figures_dir / "phase6_compression_robustness.png", dpi=300)
        plt.close()

    # 8. Paired Metric Differences (Forest Plot)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_to_plot = ["balanced_accuracy", "roc_auc", "f1_score", "recall", "specificity"]
    labels_clean = ["Balanced Acc", "ROC-AUC", "F1 Score", "Recall", "Specificity"]

    for idx, (name, title) in enumerate([("tiny_genimage", "Tiny-GenImage"), ("frontier_benchmark_2026_10", "Frontier"), ("wildaigc_bench", "WildAIGC")]):
        ax = axes[idx]
        d_stat = stat_data.get(name, {})
        means = []
        err_low = []
        err_high = []

        for m in metrics_to_plot:
            ci = d_stat.get(m, {}).get("delta", {})
            mean = ci.get("mean", 0.0)
            low = ci.get("ci_95_low", 0.0)
            high = ci.get("ci_95_high", 0.0)
            means.append(mean)
            err_low.append(mean - low)
            err_high.append(high - mean)

        y_pos = np.arange(len(metrics_to_plot))
        ax.errorbar(means, y_pos, xerr=[err_low, err_high], fmt="o", color=colors["e3"], ecolor=colors["e3"], elinewidth=2, capsize=4)
        ax.axvline(0, color="k", linestyle="--", alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_clean, fontsize=10)
        ax.set_xlabel("Paired Difference Δ (E3 - Baseline) [95% CI]")
        ax.set_title(f"{title} Paired Differences", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "phase6_paired_metric_differences.png", dpi=300)
    plt.close()

    logger.info("Saved all 8 publication figures in %s", figures_dir)


def main() -> None:
    results_dir = Path("experiments/results/phase6")
    reports_dir = Path("experiments/reports")
    figures_dir = reports_dir / "figures"

    # 1. Load predictions
    datasets_data = {
        "tiny_genimage": (
            load_predictions(results_dir / "baseline_tiny_genimage_predictions.csv"),
            load_predictions(results_dir / "e3_tiny_genimage_predictions.csv"),
        ),
        "frontier_benchmark_2026_10": (
            load_predictions(results_dir / "baseline_frontier_benchmark_2026_10_predictions.csv"),
            load_predictions(results_dir / "e3_frontier_benchmark_2026_10_predictions.csv"),
        ),
        "wildaigc_bench": (
            load_predictions(results_dir / "baseline_wildaigc_bench_predictions.csv"),
            load_predictions(results_dir / "e3_wildaigc_bench_predictions.csv"),
        ),
    }

    # 2. Run Group-Aware Stratified Bootstrap
    logger.info("=" * 70)
    logger.info("RUNNING GROUP-AWARE STRATIFIED BOOTSTRAP (1,000 RESAMPLES PER DATASET)")
    logger.info("=" * 70)

    stat_results = {}
    for name, (rb, re3) in datasets_data.items():
        logger.info("Running bootstrap for %s...", name)
        res = stratified_group_bootstrap(rb, re3, n_resamples=1000, seed=42)
        stat_results[name] = res
        logger.info(
            "%s Bootstrap Results (95%% CI):\n"
            "  Δ BalAcc: %+.4f [%+.4f, %+.4f]\n"
            "  Δ ROC-AUC: %+.4f [%+.4f, %+.4f]\n"
            "  Δ F1 Score: %+.4f [%+.4f, %+.4f]",
            name,
            res["balanced_accuracy"]["delta"]["mean"],
            res["balanced_accuracy"]["delta"]["ci_95_low"],
            res["balanced_accuracy"]["delta"]["ci_95_high"],
            res["roc_auc"]["delta"]["mean"],
            res["roc_auc"]["delta"]["ci_95_low"],
            res["roc_auc"]["delta"]["ci_95_high"],
            res["f1_score"]["delta"]["mean"],
            res["f1_score"]["delta"]["ci_95_low"],
            res["f1_score"]["delta"]["ci_95_high"],
        )

    with open(reports_dir / "phase6_statistical_comparison.json", "w", encoding="utf-8") as f:
        json.dump(stat_results, f, indent=2)

    # 3. Load other report files for figures
    with open(reports_dir / "phase6_domain_confounding.json", "r", encoding="utf-8") as f:
        domain_data = json.load(f)

    with open(reports_dir / "phase6_generator_generalization.json", "r", encoding="utf-8") as f:
        gen_data = json.load(f)

    with open(reports_dir / "phase6_compression_robustness.json", "r", encoding="utf-8") as f:
        comp_data = json.load(f)

    # 4. Generate Figures
    generate_publication_figures(datasets_data, domain_data, gen_data, comp_data, stat_results, figures_dir)

    # 5. Step 9 Candidate Decision
    # Synthesize findings:
    # 1. Tiny-GenImage in-distribution performance is high (+29.97% BalAcc, +33.30% AUC).
    # 2. However, cross-dataset generalization fails: on Frontier benchmark, E3 drops by -7.31% BalAcc and -9.55% AUC relative to baseline!
    # 3. Causal domain confounding analysis confirms extreme shortcut learning: JPEG transcoding causes 45.6% - 49.5% prediction flips with a massive 6.12 mean logit shift.
    # 4. Unseen generator recall collapses: Midjourney (51.6%), Wukong (54.0%), VQDM (20.6%).
    # Formal Decision: RETAIN_BASELINE (Candidate rejected for production deployment due to format shortcut learning and cross-dataset degradation).
    decision_report = {
        "candidate_model_id": "truthlens/siglip-base-tiny-genimage-e3_partial_finetuning",
        "baseline_model_id": "truthlens/siglip-cifake-frozen-head",
        "final_candidate_decision": "RETAIN_BASELINE",
        "candidate_disposition": "REJECT_CANDIDATE_FOR_PRODUCTION",
        "decision_summary": (
            "The E3 candidate achieved high in-distribution performance on Tiny-GenImage (BalAcc 0.8520, AUC 0.9453), "
            "but rigorous generalization audits revealed critical defects that prohibit production deployment: "
            "1. Severe cross-dataset degradation on the modern Frontier benchmark (BalAcc 0.5497 vs Baseline 0.6228; AUC 0.5546 vs Baseline 0.6501). "
            "2. Catastrophic shortcut learning on image format: matched transcoding showed that converting AI PNG images to JPEG caused 45.6% to 49.5% prediction flips, with a mean logit shift of 6.12 to 6.71 points. "
            "3. Sharp recall drop on held-out unseen generators: Midjourney (51.6%), Wukong (54.0%), and VQDM (20.6%). "
            "Consequently, the baseline model is retained in production, and E3 is rejected for deployment."
        ),
        "evaluation_criteria_audit": {
            "cross_dataset_performance": "FAIL (E3 underperformed Baseline on Frontier by -7.31% BalAcc and -9.55% AUC)",
            "unseen_generator_generalization": "PASS_WITH_LIMITATIONS (Improved BalAcc over baseline due to specificity, but recall collapsed to 20-54%)",
            "encoding_robustness": "FAIL (45.6% - 49.5% flip rate under JPEG transcoding; severe format confounding)",
            "compression_robustness": "FAIL (E3 dropped to ~0.54-0.56 AUC across social media variants, below baseline ~0.64-0.66)",
            "confidence_calibration": "PASS (Optimal T=2.0898 on Tiny-GenImage; pooled T=2.2729; significantly improved over baseline)",
        },
        "phase7_recommendations": [
            "1. Retain the frozen CIFAKE baseline (models/baselines/siglip_frozen_head.pt) in production.",
            "2. For future fine-tuning (Phase 7), apply aggressive data augmentation including random JPEG compression, format transcoding, and blur during training to break the 100% JPEG (real) vs 100% PNG (AI) artifact shortcut.",
            "3. Incorporate multi-generator, format-balanced training data across both real and synthetic partitions.",
        ],
    }

    with open(reports_dir / "phase6_candidate_decision.json", "w", encoding="utf-8") as f:
        json.dump(decision_report, f, indent=2)

    logger.info("Saved Phase 6 Candidate Decision Report to %s", reports_dir / "phase6_candidate_decision.json")


if __name__ == "__main__":
    main()
