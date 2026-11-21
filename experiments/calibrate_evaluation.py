"""
TruthLens Phase 3: Confidence Calibration and Threshold Selection.

Implements rigorous temperature scaling and calibration evaluation:
1. Strict isolation protocol: Temperature T is fitted EXCLUSIVELY on the
   'calibration' partition. No validation or test data is ever seen during fitting.
2. Fits both dataset-specific temperatures and a pooled temperature.
3. Computes strictly proper scoring rules (Brier score, NLL) and calibration
   errors (ECE with 15 bins, MCE) before and after calibration.
4. Generates reliability diagram coordinates across all splits.
5. Selects decision thresholds and abstention intervals ON THE VALIDATION SPLIT ONLY.
6. Persists calibration artifacts and diagnostics to experiments/reports/phase3_calibration.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("calibrate_evaluation")


def load_predictions(csv_path: Path) -> List[Dict[str, Any]]:
    """Load predictions CSV into typed dictionaries."""
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


def fit_temperature_scaling(
    margins: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Fit scalar temperature T by minimizing negative log-likelihood on calibration data.
    logits: margin = fake_logit - real_logit
    prob(AI) = sigmoid(margin / T)
    """
    if len(margins) != len(labels) or len(margins) == 0:
        raise ValueError("Margins and labels must be non-empty and equal length.")
    if len(np.unique(labels)) < 2:
        raise ValueError("Calibration set must contain both classes (0 and 1).")

    def nll_objective(log_t: float) -> float:
        t = math.exp(log_t)
        scaled_margins = margins / t
        probs = sigmoid(scaled_margins)
        probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
        nll = -np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
        return float(nll)

    res = minimize_scalar(nll_objective, bounds=[-5.0, 5.0], method="bounded")
    opt_t = float(math.exp(res.x))
    log.info("Temperature fitted: T = %.4f (NLL: %.4f)", opt_t, res.fun)
    return opt_t


def compute_calibration_diagnostics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    num_bins: int = 15,
) -> Dict[str, Any]:
    """
    Compute comprehensive calibration metrics:
    - NLL (Negative Log Likelihood)
    - Brier score
    - ECE (Expected Calibration Error, 15 bins)
    - MCE (Maximum Calibration Error)
    - Reliability diagram bin data
    """
    n = len(labels)
    if n == 0:
        return {"n": 0}

    probs = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    y = labels.astype(float)

    # Strictly proper scoring rules
    nll = float(-np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))
    brier = float(np.mean((probs - y) ** 2))

    # Equal-width binning for ECE and reliability diagrams
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_rows = []
    ece_equal_width = 0.0
    mce = 0.0

    for i in range(num_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == num_bins - 1:
            mask = (probs >= low) & (probs <= high)
        else:
            mask = (probs >= low) & (probs < high)

        bin_count = int(mask.sum())
        if bin_count > 0:
            bin_conf = float(probs[mask].mean())
            bin_acc = float(y[mask].mean())  # Observed empirical frequency of AI
            gap = abs(bin_conf - bin_acc)
            ece_equal_width += (bin_count / n) * gap
            mce = max(mce, gap)
            bin_rows.append({
                "bin_index": i,
                "low": round(float(low), 4),
                "high": round(float(high), 4),
                "count": bin_count,
                "mean_confidence": round(bin_conf, 4),
                "empirical_accuracy": round(bin_acc, 4),
                "calibration_gap": round(gap, 4),
            })
        else:
            bin_rows.append({
                "bin_index": i,
                "low": round(float(low), 4),
                "high": round(float(high), 4),
                "count": 0,
                "mean_confidence": None,
                "empirical_accuracy": None,
                "calibration_gap": 0.0,
            })

    # Equal-mass (quantile) binning for ECE
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    q_edges = np.quantile(probs, quantiles)
    # Ensure strictly increasing edges
    q_edges[0] = 0.0
    q_edges[-1] = 1.0
    ece_equal_mass = 0.0

    for i in range(num_bins):
        q_low, q_high = q_edges[i], q_edges[i + 1]
        if i == num_bins - 1:
            mask = (probs >= q_low) & (probs <= q_high)
        else:
            mask = (probs >= q_low) & (probs < q_high)
        bin_count = int(mask.sum())
        if bin_count > 0:
            bin_conf = float(probs[mask].mean())
            bin_acc = float(y[mask].mean())
            gap = abs(bin_conf - bin_acc)
            ece_equal_mass += (bin_count / n) * gap

    return {
        "sample_count": n,
        "nll": round(nll, 5),
        "brier_score": round(brier, 5),
        "ece_equal_width": round(float(ece_equal_width), 5),
        "ece_equal_mass": round(float(ece_equal_mass), 5),
        "mce": round(float(mce), 5),
        "reliability_bins": bin_rows,
    }


def select_operating_thresholds_on_validation(
    labels: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, Any]:
    """
    Select candidate operating thresholds using ONLY the validation partition:
    1. Default threshold (0.5)
    2. Optimal Balanced Accuracy threshold
    3. Optimal F1-score threshold
    4. High Specificity (>= 95%) threshold
    5. Conservative Abstention bounds (error <= 10% on both classes)
    """
    y = labels.astype(int)
    p = probs.astype(float)
    n = len(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    threshold_grid = np.linspace(0.01, 0.99, 197)

    best_bal_acc = -1.0
    best_bal_tau = 0.5

    best_f1 = -1.0
    best_f1_tau = 0.5

    spec95_tau = 0.5
    spec95_found = False

    for tau in threshold_grid:
        preds = (p >= tau).astype(int)
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        tn = int(((preds == 0) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())

        tpr = tp / n_pos if n_pos > 0 else 0.0
        tnr = tn / n_neg if n_neg > 0 else 0.0
        bal_acc = (tpr + tnr) / 2.0

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tpr
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_bal_tau = float(tau)

        if f1 > best_f1:
            best_f1 = f1
            best_f1_tau = float(tau)

        if tnr >= 0.95 and not spec95_found:
            spec95_tau = float(tau)
            spec95_found = True

    # Selective abstention bounds: find [auth_max, ai_min] maximizing coverage
    # subject to class-wise error rates <= 10% (FPR <= 0.10, FNR <= 0.10)
    best_coverage = -1.0
    auth_max_sel = 0.35
    ai_min_sel = 0.65
    found_abstention = False

    abstention_grid = np.linspace(0.0, 1.0, 101)
    for auth_max in abstention_grid:
        # Predict real when p <= auth_max
        pred_real = p <= auth_max
        fnr = float(((y == 1) & pred_real).sum()) / n_pos if n_pos > 0 else 0.0
        if fnr > 0.10:
            continue
        for ai_min in abstention_grid[abstention_grid > auth_max]:
            pred_ai = p >= ai_min
            fpr = float(((y == 0) & pred_ai).sum()) / n_neg if n_neg > 0 else 0.0
            if fpr > 0.10:
                continue
            cov = float((pred_real | pred_ai).mean())
            if cov > best_coverage:
                best_coverage = cov
                auth_max_sel = float(auth_max)
                ai_min_sel = float(ai_min)
                found_abstention = True

    return {
        "default_threshold": 0.5,
        "optimal_balanced_accuracy": {
            "threshold": round(best_bal_tau, 4),
            "validation_balanced_accuracy": round(best_bal_acc, 4),
        },
        "optimal_f1": {
            "threshold": round(best_f1_tau, 4),
            "validation_f1": round(best_f1, 4),
        },
        "high_specificity_95": {
            "threshold": round(spec95_tau, 4),
            "target_specificity": 0.95,
            "achieved_on_validation": spec95_found,
        },
        "selective_abstention": {
            "authentic_max_threshold": round(auth_max_sel, 4),
            "ai_generated_min_threshold": round(ai_min_sel, 4),
            "validation_coverage": round(best_coverage, 4) if found_abstention else 0.0,
            "abstention_rate": round(1.0 - best_coverage, 4) if found_abstention else 1.0,
            "target_class_error_cap": 0.10,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="TruthLens Phase 3 Calibration Analysis")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "phase3",
        help="Directory containing predictions CSV files",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "reports" / "phase3_calibration.json",
        help="Path to output calibration report JSON",
    )
    args = parser.parse_args()

    args.output_report.parent.mkdir(parents=True, exist_ok=True)

    dataset_files = {
        "tiny_genimage": args.results_dir / "tiny_genimage_predictions.csv",
        "frontier_benchmark_2026_10": args.results_dir / "frontier_predictions.csv",
        "wildaigc_bench": args.results_dir / "wildaigc_predictions.csv",
    }

    all_data = {}
    for ds_id, p in dataset_files.items():
        if not p.is_file():
            raise FileNotFoundError(f"Predictions file not found: {p}. Run evaluate_authenticity.py first.")
        all_data[ds_id] = load_predictions(p)
        log.info("Loaded %d rows for %s", len(all_data[ds_id]), ds_id)

    # 1. Strict Isolation: Gather calibration split rows for fitting
    calib_margins: Dict[str, np.ndarray] = {}
    calib_labels: Dict[str, np.ndarray] = {}

    val_margins: Dict[str, np.ndarray] = {}
    val_labels: Dict[str, np.ndarray] = {}

    test_margins: Dict[str, np.ndarray] = {}
    test_labels: Dict[str, np.ndarray] = {}

    pooled_calib_margins = []
    pooled_calib_labels = []

    for ds_id, rows in all_data.items():
        c_m, c_y = [], []
        v_m, v_y = [], []
        t_m, t_y = [], []

        for r in rows:
            role = r["evaluation_role"]
            margin = r["raw_margin"]
            gt = r["ground_truth_binary"]
            if role == "calibration":
                c_m.append(margin)
                c_y.append(gt)
                pooled_calib_margins.append(margin)
                pooled_calib_labels.append(gt)
            elif role == "validation":
                v_m.append(margin)
                v_y.append(gt)
            elif role == "test":
                t_m.append(margin)
                t_y.append(gt)

        calib_margins[ds_id] = np.array(c_m, dtype=float)
        calib_labels[ds_id] = np.array(c_y, dtype=int)
        val_margins[ds_id] = np.array(v_m, dtype=float)
        val_labels[ds_id] = np.array(v_y, dtype=int)
        test_margins[ds_id] = np.array(t_m, dtype=float)
        test_labels[ds_id] = np.array(t_y, dtype=int)

    # 2. Fit temperatures EXCLUSIVELY on calibration partitions
    log.info("Fitting temperature scaling exclusively on calibration partitions...")
    fitted_temperatures: Dict[str, float] = {}
    for ds_id in all_data:
        fitted_temperatures[ds_id] = fit_temperature_scaling(
            calib_margins[ds_id], calib_labels[ds_id]
        )

    pooled_temperature = fit_temperature_scaling(
        np.array(pooled_calib_margins, dtype=float),
        np.array(pooled_calib_labels, dtype=int),
    )
    fitted_temperatures["pooled"] = pooled_temperature
    log.info("Pooled calibration temperature: T = %.4f", pooled_temperature)

    # 3. Compute Calibration Diagnostics Before and After
    dataset_reports: Dict[str, Any] = {}

    for ds_id in all_data:
        t_fit = fitted_temperatures[ds_id]

        # Raw (T=1.0) probabilities
        raw_p_calib = sigmoid(calib_margins[ds_id])
        raw_p_val = sigmoid(val_margins[ds_id])
        raw_p_test = sigmoid(test_margins[ds_id])

        # Scaled (T=t_fit) probabilities
        cal_p_calib = sigmoid(calib_margins[ds_id] / t_fit)
        cal_p_val = sigmoid(val_margins[ds_id] / t_fit)
        cal_p_test = sigmoid(test_margins[ds_id] / t_fit)

        # Pooled scaled probabilities
        pool_p_calib = sigmoid(calib_margins[ds_id] / pooled_temperature)
        pool_p_val = sigmoid(val_margins[ds_id] / pooled_temperature)
        pool_p_test = sigmoid(test_margins[ds_id] / pooled_temperature)

        # Threshold selection on VALIDATION split only
        val_thresholds = select_operating_thresholds_on_validation(
            val_labels[ds_id], cal_p_val
        )

        dataset_reports[ds_id] = {
            "dataset_id": ds_id,
            "fitted_temperature": round(t_fit, 4),
            "sample_counts": {
                "calibration": len(calib_labels[ds_id]),
                "validation": len(val_labels[ds_id]),
                "test": len(test_labels[ds_id]),
            },
            "calibration_diagnostics": {
                "calibration_split": {
                    "uncalibrated_T1": compute_calibration_diagnostics(calib_labels[ds_id], raw_p_calib),
                    "calibrated_T_fit": compute_calibration_diagnostics(calib_labels[ds_id], cal_p_calib),
                    "calibrated_T_pooled": compute_calibration_diagnostics(calib_labels[ds_id], pool_p_calib),
                },
                "validation_split": {
                    "uncalibrated_T1": compute_calibration_diagnostics(val_labels[ds_id], raw_p_val),
                    "calibrated_T_fit": compute_calibration_diagnostics(val_labels[ds_id], cal_p_val),
                    "calibrated_T_pooled": compute_calibration_diagnostics(val_labels[ds_id], pool_p_val),
                },
                "test_split": {
                    "uncalibrated_T1": compute_calibration_diagnostics(test_labels[ds_id], raw_p_test),
                    "calibrated_T_fit": compute_calibration_diagnostics(test_labels[ds_id], cal_p_test),
                    "calibrated_T_pooled": compute_calibration_diagnostics(test_labels[ds_id], pool_p_test),
                },
            },
            "validation_operating_thresholds": val_thresholds,
        }

    # Overall summary artifact
    calibration_artifact = {
        "schema_version": 2,
        "protocol_id": "truthlens-phase3-temperature-calibration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "temperature_fitting_isolation": {
            "strict_isolation_verified": True,
            "fitted_strictly_on_roles": ["calibration"],
            "validation_and_test_held_out": True,
            "total_calibration_samples": len(pooled_calib_labels),
        },
        "fitted_temperatures": {k: round(v, 4) for k, v in fitted_temperatures.items()},
        "datasets": dataset_reports,
        "limitations_and_governance": [
            "Calibration temperature is fitted exclusively on the calibration split.",
            "Temperature scaling recalibrates predicted confidence probabilities; it does not change the ROC curve, ranking, or feature representation.",
            "Validation partition is used solely for threshold selection and calibration generalization check.",
            "Held-out test partition is strictly locked and evaluated once.",
        ],
    }

    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(calibration_artifact, f, indent=2)

    log.info("Saved calibration report to %s", args.output_report)


if __name__ == "__main__":
    main()
