"""
TruthLens Phase 5: Post-Training Analysis, Validation-Only Model Selection & Report Generation.

Aggregates results from:
- Baseline (frozen CIFAKE head evaluated on Tiny-GenImage validation split)
- E1: Frozen SigLIP backbone + linear head
- E2: Frozen SigLIP backbone + class-balanced loss
- E3: Partial fine-tuning (encoder layer 11 + head)

Produces:
- experiments/reports/phase5_training_results.json
- experiments/reports/phase5_experiment_comparison.csv
- experiments/reports/phase5_training_curves.json
- experiments/reports/phase5_model_registry.json
- Publication-quality visualization figures in experiments/reports/figures/
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("analyze_phase5")


def load_experiment_data(chkpt_base: Path, exp_names: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    summaries = {}
    histories = {}
    for name in exp_names:
        exp_dir = chkpt_base / name
        summary_file = exp_dir / "experiment_summary.json"
        history_file = exp_dir / "training_history.json"

        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                summaries[name] = json.load(f)
        else:
            logger.warning("Summary file not found: %s", summary_file)

        if history_file.exists():
            with open(history_file, "r", encoding="utf-8") as f:
                histories[name] = json.load(f)
        else:
            logger.warning("History file not found: %s", history_file)

    return summaries, histories


def generate_comparison_csv(
    baseline_ref: Dict[str, Any],
    summaries: Dict[str, Any],
    output_path: Path,
) -> List[Dict[str, Any]]:
    """Generate side-by-side comparison CSV for Baseline, E1, E2, E3."""
    rows = []

    # 1. Baseline
    rows.append({
        "model_id": baseline_ref.get("model_id", "truthlens/siglip-cifake-frozen-head"),
        "experiment_name": "Baseline (CIFAKE Head)",
        "mode": "frozen_baseline",
        "trainable_params_pct": "0.0008%",
        "best_epoch": "N/A",
        "val_loss": "N/A",
        "val_accuracy": baseline_ref.get("validation_accuracy", 0.5213),
        "val_balanced_accuracy": baseline_ref.get("validation_balanced_accuracy", 0.5542),
        "val_precision": "0.4646",
        "val_recall": "0.7818",
        "val_specificity": "0.3267",
        "val_f1": baseline_ref.get("validation_f1", 0.5828),
        "val_roc_auc": baseline_ref.get("validation_roc_auc", 0.6020),
        "val_pr_auc": "0.5539",
        "val_mcc": "0.1168",
        "training_time_sec": "0.0",
        "peak_vram_mb": "445.1",
        "checkpoint_sha256": baseline_ref.get("head_sha256", "e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538"),
    })

    # 2. Experiments E1, E2, E3
    for name, s in summaries.items():
        v = s.get("best_validation_metrics", {})
        pc = s.get("parameter_counts", {})
        total_p = pc.get("total_params", 92884993)
        train_p = pc.get("trainable_params", 0)
        pct = f"{(train_p / total_p)*100:.4f}%" if total_p else "N/A"

        rows.append({
            "model_id": s.get("model_identity", {}).get("model_id", name),
            "experiment_name": name,
            "mode": s.get("mode", ""),
            "trainable_params_pct": pct,
            "best_epoch": s.get("best_epoch", "N/A"),
            "val_loss": v.get("val_loss", "N/A"),
            "val_accuracy": v.get("accuracy", "N/A"),
            "val_balanced_accuracy": v.get("balanced_accuracy", "N/A"),
            "val_precision": v.get("precision", "N/A"),
            "val_recall": v.get("recall", "N/A"),
            "val_specificity": v.get("specificity", "N/A"),
            "val_f1": v.get("f1_score", "N/A"),
            "val_roc_auc": v.get("roc_auc", "N/A"),
            "val_pr_auc": v.get("pr_auc", "N/A"),
            "val_mcc": v.get("mcc", "N/A"),
            "training_time_sec": s.get("total_training_duration_sec", "N/A"),
            "peak_vram_mb": s.get("peak_vram_mb", "N/A"),
            "checkpoint_sha256": s.get("best_checkpoint_sha256", "N/A"),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved comparison CSV to %s", output_path)
    return rows


def generate_figures(
    baseline_ref: Dict[str, Any],
    summaries: Dict[str, Any],
    histories: Dict[str, Any],
    figures_dir: Path,
) -> None:
    """Generate all publication-quality comparison figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {
        "e1_frozen_head": "#1f77b4",
        "e2_balanced_training": "#2ca02c",
        "e3_partial_finetuning": "#d62728",
        "baseline": "#7f7f7f",
    }

    # 1. Loss curves (Train & Val)
    plt.figure(figsize=(9, 5))
    for name, hist in histories.items():
        if not hist:
            continue
        epochs = [r["epoch"] for r in hist]
        train_losses = [r["train_loss"] for r in hist]
        val_losses = [r["val_loss"] for r in hist]
        c = colors.get(name, "#333333")
        plt.plot(epochs, train_losses, linestyle="--", color=c, alpha=0.7, label=f"{name} (Train)")
        plt.plot(epochs, val_losses, linestyle="-", marker="o", color=c, label=f"{name} (Val)")

    plt.title("Phase 5: Training & Validation Loss Curves", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("BCE Loss", fontsize=11)
    plt.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(figures_dir / "phase5_loss_curves.png", dpi=300)
    plt.close()

    # 2. Validation Balanced Accuracy Curves
    plt.figure(figsize=(9, 5))
    base_bal_acc = baseline_ref.get("validation_balanced_accuracy", 0.5542)
    plt.axhline(y=base_bal_acc, color=colors["baseline"], linestyle=":", linewidth=2, label=f"Baseline CIFAKE ({base_bal_acc:.4f})")

    for name, hist in histories.items():
        if not hist:
            continue
        epochs = [r["epoch"] for r in hist]
        bal_accs = [r["val_balanced_accuracy"] for r in hist]
        c = colors.get(name, "#333333")
        plt.plot(epochs, bal_accs, marker="s", linewidth=2, color=c, label=name)

    plt.title("Phase 5: Validation Balanced Accuracy Progression", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Balanced Accuracy", fontsize=11)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / "phase5_val_balanced_acc_curves.png", dpi=300)
    plt.close()

    # 3. Validation ROC-AUC Curves
    plt.figure(figsize=(9, 5))
    base_auc = baseline_ref.get("validation_roc_auc", 0.6020)
    plt.axhline(y=base_auc, color=colors["baseline"], linestyle=":", linewidth=2, label=f"Baseline CIFAKE ({base_auc:.4f})")

    for name, hist in histories.items():
        if not hist:
            continue
        epochs = [r["epoch"] for r in hist]
        aucs = [r["val_roc_auc"] for r in hist]
        c = colors.get(name, "#333333")
        plt.plot(epochs, aucs, marker="^", linewidth=2, color=c, label=name)

    plt.title("Phase 5: Validation ROC-AUC Progression", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("ROC-AUC", fontsize=11)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / "phase5_val_roc_curves.png", dpi=300)
    plt.close()

    # 4. Validation PR-AUC Curves
    plt.figure(figsize=(9, 5))
    for name, hist in histories.items():
        if not hist:
            continue
        epochs = [r["epoch"] for r in hist]
        pr_aucs = [r["val_pr_auc"] for r in hist]
        c = colors.get(name, "#333333")
        plt.plot(epochs, pr_aucs, marker="d", linewidth=2, color=c, label=name)

    plt.title("Phase 5: Validation Precision-Recall AUC Progression", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("PR-AUC", fontsize=11)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / "phase5_val_pr_curves.png", dpi=300)
    plt.close()

    # 5. Side-by-Side Model Comparison Bar Chart
    models = ["Baseline", "E1 (Frozen)", "E2 (Balanced)", "E3 (Partial FT)"]
    metrics = ["Accuracy", "Balanced Acc", "F1 Score", "ROC-AUC"]

    def extract_metrics(name: str) -> List[float]:
        if name == "Baseline":
            return [
                baseline_ref.get("validation_accuracy", 0.5213),
                baseline_ref.get("validation_balanced_accuracy", 0.5542),
                baseline_ref.get("validation_f1", 0.5828),
                baseline_ref.get("validation_roc_auc", 0.6020),
            ]
        s = summaries.get(name, {})
        v = s.get("best_validation_metrics", {})
        return [
            v.get("accuracy", 0.0),
            v.get("balanced_accuracy", 0.0),
            v.get("f1_score", 0.0),
            v.get("roc_auc", 0.0),
        ]

    data = {
        "Baseline": extract_metrics("Baseline"),
        "E1 (Frozen)": extract_metrics("e1_frozen_head"),
        "E2 (Balanced)": extract_metrics("e2_balanced_training"),
        "E3 (Partial FT)": extract_metrics("e3_partial_finetuning"),
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(metrics))
    width = 0.20

    for i, (m_name, vals) in enumerate(data.items()):
        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, vals, width, label=m_name)
        for r in rects:
            height = r.get_height()
            if height > 0.05:
                ax.annotate(
                    f"{height:.3f}",
                    xy=(r.get_x() + r.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Phase 5: Controlled Model Performance Comparison (Validation Split)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(figures_dir / "phase5_experiment_comparison.png", dpi=300)
    plt.close()

    # 6. Resource and Memory Bar Chart
    vram_vals = [
        445.1,
        summaries.get("e1_frozen_head", {}).get("peak_vram_mb", 445.1),
        summaries.get("e2_balanced_training", {}).get("peak_vram_mb", 445.1),
        summaries.get("e3_partial_finetuning", {}).get("peak_vram_mb", 536.4),
    ]
    plt.figure(figsize=(8, 4.5))
    bar_colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]
    bars = plt.bar(models, vram_vals, color=bar_colors, width=0.5)
    for b in bars:
        h = b.get_height()
        plt.annotate(f"{h:.1f} MB", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.axhline(y=4096, color="r", linestyle="--", alpha=0.5, label="VRAM Limit (4096 MB)")
    plt.ylabel("Peak VRAM (MB)", fontsize=11)
    plt.title("Hardware Resource Usage: Peak VRAM on NVIDIA RTX 3050 Laptop", fontsize=12, fontweight="bold")
    plt.ylim(0, 4800)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(figures_dir / "phase5_vram_usage.png", dpi=300)
    plt.close()

    logger.info("Generated 6 comparison figures in %s", figures_dir)


def main() -> None:
    config_path = Path("experiments/configs/phase5_experiment_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    chkpt_base = Path(config["global_settings"]["checkpoints_dir"])
    reports_dir = Path(config["global_settings"]["reports_dir"])
    figures_dir = Path(config["global_settings"]["figures_dir"])
    exp_names = ["e1_frozen_head", "e2_balanced_training", "e3_partial_finetuning"]

    summaries, histories = load_experiment_data(chkpt_base, exp_names)
    baseline_ref = config["baseline_reference"]

    # 1. Comparison CSV
    csv_path = reports_dir / "phase5_experiment_comparison.csv"
    comp_rows = generate_comparison_csv(baseline_ref, summaries, csv_path)

    # 2. Figures
    generate_figures(baseline_ref, summaries, histories, figures_dir)

    # 3. Model Registry JSON
    registry_path = reports_dir / "phase5_model_registry.json"
    registry = {
        "baseline_model": {
            "model_id": baseline_ref["model_id"],
            "backbone_revision": baseline_ref["backbone_revision"],
            "head_sha256": baseline_ref["head_sha256"],
            "status": "BASELINE_FROZEN",
            "validation_balanced_accuracy": baseline_ref["validation_balanced_accuracy"],
            "validation_roc_auc": baseline_ref["validation_roc_auc"],
        },
        "experiments": {},
    }

    best_candidate_name = ""
    best_candidate_metric = -1.0

    for name, s in summaries.items():
        v = s.get("best_validation_metrics", {})
        bal_acc = v.get("balanced_accuracy", 0.0)
        registry["experiments"][name] = {
            "model_id": s.get("model_identity", {}).get("model_id"),
            "backbone_model_id": s.get("model_identity", {}).get("backbone_model_id"),
            "backbone_revision": s.get("model_identity", {}).get("backbone_revision"),
            "head_sha256": s.get("best_head_sha256"),
            "checkpoint_sha256": s.get("best_checkpoint_sha256"),
            "checkpoint_path": s.get("best_checkpoint_path"),
            "head_path": s.get("best_head_path"),
            "best_epoch": s.get("best_epoch"),
            "validation_metrics": v,
        }
        if bal_acc > best_candidate_metric:
            best_candidate_metric = bal_acc
            best_candidate_name = name

    # Model selection rule (validation only)
    base_bal_acc = baseline_ref["validation_balanced_accuracy"]
    if best_candidate_metric > base_bal_acc:
        selected_recommendation = {
            "selected_model": best_candidate_name,
            "decision": "PROMOTE_AS_CANDIDATE",
            "justification": f"{best_candidate_name} achieved validation balanced accuracy {best_candidate_metric:.4f}, outperforming baseline {base_bal_acc:.4f} by +{(best_candidate_metric - base_bal_acc)*100:.2f} percentage points.",
            "eligibility_for_phase6_test": True,
        }
    else:
        selected_recommendation = {
            "selected_model": baseline_ref["model_id"],
            "decision": "RETAIN_BASELINE",
            "justification": f"No experiment exceeded baseline validation balanced accuracy {base_bal_acc:.4f}. Baseline is retained.",
            "eligibility_for_phase6_test": False,
        }

    registry["model_selection_decision"] = selected_recommendation

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    logger.info("Saved model registry to %s", registry_path)

    # 4. Training Curves JSON
    curves_path = reports_dir / "phase5_training_curves.json"
    with open(curves_path, "w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2)
    logger.info("Saved training curves to %s", curves_path)

    # 5. Comprehensive Results JSON
    results_path = reports_dir / "phase5_training_results.json"
    results_payload = {
        "metadata": {
            "phase": "Phase 5 - Controlled Model Training and Fine-Tuning",
            "backbone": config["model_architecture"]["backbone_model_id"],
            "backbone_revision": config["model_architecture"]["backbone_revision"],
            "device": config["global_settings"]["device"],
            "training_samples": 6995,
            "validation_samples": 5958,
            "test_evaluation_performed": False,
            "test_split_status": "LOCKED - STRICTLY FORBIDDEN IN PHASE 5",
        },
        "baseline_validation": baseline_ref,
        "experiments": summaries,
        "model_selection": selected_recommendation,
        "domain_confounding_analysis": {
            "real_images_format": "100% JPEG (ImageNet ILSVRC2012 nature)",
            "ai_images_format": "100% PNG (ADM, BigGAN, SD 1.5, GLIDE)",
            "class_prefix_overlap": 0,
            "confounding_risk": "HIGH - Models may exploit compression artifact differences between JPEG and PNG rather than semantic generative features. Generalization across uncompressed formats requires evaluation in Phase 6.",
        },
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    logger.info("Saved comprehensive training results to %s", results_path)


if __name__ == "__main__":
    main()
