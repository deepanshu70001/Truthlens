"""Scientific Uncertainty and Calibration Research for TruthLens (Package C).

Evaluates the authorized training split (n=6,995) to quantify:
1. Uncalibrated SigLIP probability calibration (ECE, MCE, Brier score, reliability diagrams).
2. Predictive uncertainty distributions (Maximum Softmax Probability, Shannon Entropy).
3. Temperature scaling research exploration (optimal T*, calibration delta, NLL minimization).
4. Error analysis by confidence decile.
5. Selective prediction and risk-coverage curves under entropy-based abstention.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

def compute_entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p_clamped = np.clip(p, eps, 1.0 - eps)
    return -(p_clamped * np.log2(p_clamped) + (1.0 - p_clamped) * np.log2(1.0 - p_clamped))

def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Compute ECE, MCE, and reliability diagram bins for binary classification."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Predicted confidence for the chosen class
    y_pred = (y_prob >= 0.5).astype(int)
    confidences = np.where(y_pred == 1, y_prob, 1.0 - y_prob)
    accuracies = (y_pred == y_true).astype(float)
    
    bin_details = []
    ece = 0.0
    mce = 0.0
    total_samples = len(y_true)

    # Note: evaluate bins on confidences in [0.5, 1.0] for two-class prediction confidence
    conf_bins = np.linspace(0.5, 1.0, n_bins + 1)
    for i in range(n_bins):
        b_min, b_max = conf_bins[i], conf_bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= b_min) & (confidences <= b_max)
        else:
            mask = (confidences >= b_min) & (confidences < b_max)

        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            bin_err = abs(bin_acc - bin_conf)
            ece += (count / total_samples) * bin_err
            mce = max(mce, bin_err)
        else:
            bin_acc = None
            bin_conf = None
            bin_err = None

        bin_details.append({
            "bin_index": i + 1,
            "bin_range": [round(b_min, 4), round(b_max, 4)],
            "sample_count": count,
            "mean_confidence": round(bin_conf, 4) if bin_conf is not None else None,
            "empirical_accuracy": round(bin_acc, 4) if bin_acc is not None else None,
            "calibration_gap": round(bin_err, 4) if bin_err is not None else None
        })

    brier = float(np.mean((y_prob - y_true) ** 2))
    return {
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "brier_score": round(brier, 4),
        "n_bins": n_bins,
        "reliability_bins": bin_details
    }

def nll_objective(T: float, logits: np.ndarray, y_true: np.ndarray) -> float:
    scaled_probs = sigmoid(logits / T)
    eps = 1e-12
    p_c = np.clip(scaled_probs, eps, 1.0 - eps)
    nll = -np.mean(y_true * np.log(p_c) + (1.0 - y_true) * np.log(1.0 - p_c))
    return float(nll)

def run_research():
    print("=" * 70)
    print("TruthLens Scientific Uncertainty & Calibration Research (Package C)")
    print("=" * 70)

    # 1. Load authorized training split
    logits_path = Path("research/experiment2_forensic_screening/siglip_logits.csv")
    if not logits_path.exists():
        raise FileNotFoundError(f"Missing authorized logits: {logits_path}")

    df = pd.read_csv(logits_path)
    print(f"Loaded authorized dataset: {len(df)} samples from {logits_path}")
    
    y_true = df["label_int"].values.astype(int)
    logits = df["siglip_ai_logit"].values.astype(float)
    raw_probs = df["siglip_prob_ai"].values.astype(float)

    n_samples = len(y_true)
    n_auth = int(np.sum(y_true == 0))
    n_ai = int(np.sum(y_true == 1))
    print(f"Class distribution: {n_auth} authentic (0), {n_ai} AI-generated (1)")

    # 2. Uncalibrated Model Performance & Uncertainty Metrics
    y_pred_raw = (raw_probs >= 0.5).astype(int)
    raw_acc = float(np.mean(y_pred_raw == y_true))
    raw_msp = np.maximum(raw_probs, 1.0 - raw_probs)
    raw_entropy = compute_entropy(raw_probs)

    raw_calib_10 = compute_calibration_metrics(y_true, raw_probs, n_bins=10)
    raw_calib_15 = compute_calibration_metrics(y_true, raw_probs, n_bins=15)
    raw_nll = nll_objective(1.0, logits, y_true)

    print("\n--- Uncalibrated Baseline Metrics ---")
    print(f"Accuracy:           {raw_acc * 100:.2f}%")
    print(f"ECE (10 bins):      {raw_calib_10['ece']:.4f} ({raw_calib_10['ece']*100:.2f}%)")
    print(f"ECE (15 bins):      {raw_calib_15['ece']:.4f}")
    print(f"MCE (10 bins):      {raw_calib_10['mce']:.4f}")
    print(f"Brier Score:        {raw_calib_10['brier_score']:.4f}")
    print(f"Negative Log-Likelihood: {raw_nll:.4f}")
    print(f"Mean Predictive Entropy: {np.mean(raw_entropy):.4f} bits (Max: 1.0 bit)")
    print(f"Mean Confidence (MSP):   {np.mean(raw_msp):.4f}")

    # 3. Exploratory Temperature Scaling Optimization
    opt_res = minimize_scalar(nll_objective, bounds=(0.1, 50.0), args=(logits, y_true), method="bounded")
    optimal_T = float(opt_res.x)
    min_nll = float(opt_res.fun)

    calib_probs = sigmoid(logits / optimal_T)
    calib_metrics_10 = compute_calibration_metrics(y_true, calib_probs, n_bins=10)
    calib_metrics_15 = compute_calibration_metrics(y_true, calib_probs, n_bins=15)
    calib_entropy = compute_entropy(calib_probs)
    calib_msp = np.maximum(calib_probs, 1.0 - calib_probs)

    print("\n--- Temperature Scaling Exploration (Research Only) ---")
    print(f"Optimal Temperature (T*): {optimal_T:.4f}")
    print(f"NLL Reduction:            {raw_nll:.4f} -> {min_nll:.4f} (Delta: {raw_nll - min_nll:.4f})")
    print(f"ECE (10 bins) Delta:      {raw_calib_10['ece']:.4f} -> {calib_metrics_10['ece']:.4f} ({((raw_calib_10['ece'] - calib_metrics_10['ece'])/raw_calib_10['ece'])*100:.1f}% reduction)")
    print(f"MCE Delta:                {raw_calib_10['mce']:.4f} -> {calib_metrics_10['mce']:.4f}")
    print(f"Brier Score Delta:        {raw_calib_10['brier_score']:.4f} -> {calib_metrics_10['brier_score']:.4f}")

    # 4. Error Analysis by Confidence Decile
    decile_edges = np.linspace(0.5, 1.0, 11)
    error_analysis = []
    accuracies = (y_pred_raw == y_true).astype(int)

    for i in range(10):
        d_min, d_max = decile_edges[i], decile_edges[i + 1]
        if i == 9:
            mask = (raw_msp >= d_min) & (raw_msp <= d_max)
        else:
            mask = (raw_msp >= d_min) & (raw_msp < d_max)

        cnt = int(np.sum(mask))
        if cnt > 0:
            errs = int(cnt - np.sum(accuracies[mask]))
            err_rate = float(errs / cnt)
            mean_ent = float(np.mean(raw_entropy[mask]))
            mean_conf = float(np.mean(raw_msp[mask]))
        else:
            errs = 0
            err_rate = 0.0
            mean_ent = 0.0
            mean_conf = 0.0

        error_analysis.append({
            "confidence_decile": f"[{d_min:.2f}, {d_max:.2f}]",
            "sample_count": cnt,
            "error_count": errs,
            "error_rate": round(err_rate, 4),
            "accuracy": round(1.0 - err_rate, 4),
            "mean_confidence": round(mean_conf, 4),
            "mean_entropy": round(mean_ent, 4)
        })

    # 5. Selective Prediction & Risk-Coverage Curve
    # Sort samples by confidence descending (or entropy ascending)
    sorted_indices = np.argsort(-raw_msp)
    coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    risk_coverage_curve = []

    for cov in coverages:
        k = max(1, int(cov * n_samples))
        top_k_idx = sorted_indices[:k]
        cov_acc = float(np.mean(accuracies[top_k_idx]))
        cov_risk = 1.0 - cov_acc
        min_conf_threshold = float(raw_msp[top_k_idx[-1]])
        max_entropy_threshold = float(raw_entropy[top_k_idx[-1]])

        risk_coverage_curve.append({
            "target_coverage": cov,
            "decided_samples": k,
            "abstained_samples": n_samples - k,
            "effective_risk_error_rate": round(cov_risk, 4),
            "accuracy_on_decided": round(cov_acc, 4),
            "confidence_threshold": round(min_conf_threshold, 4),
            "entropy_threshold": round(max_entropy_threshold, 4)
        })

    # Effective error reduction at 50% coverage
    risk_full = 1.0 - raw_acc
    risk_50 = next(r["effective_risk_error_rate"] for r in risk_coverage_curve if r["target_coverage"] == 0.5)
    eer_50 = (risk_full - risk_50) / risk_full if risk_full > 0 else 0.0

    print("\n--- Selective Prediction (Risk-Coverage) ---")
    print(f"Error Rate at 100% Coverage: {risk_full * 100:.2f}%")
    print(f"Error Rate at 50% Coverage:  {risk_50 * 100:.2f}% (Error reduction: {eer_50 * 100:.1f}%)")

    # Assemble comprehensive research report
    research_report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "authorized_training_split (Tiny-GenImage training)",
            "source_path": str(logits_path),
            "total_samples": n_samples,
            "authentic_samples": n_auth,
            "ai_samples": n_ai,
            "policy_invariant_note": "Exploratory research only. Production thresholds, weights, and e11.conservative.v1 remain invariant."
        },
        "uncalibrated_baseline": {
            "accuracy": round(raw_acc, 4),
            "negative_log_likelihood": round(raw_nll, 4),
            "brier_score": raw_calib_10["brier_score"],
            "expected_calibration_error_10_bins": raw_calib_10["ece"],
            "expected_calibration_error_15_bins": raw_calib_15["ece"],
            "maximum_calibration_error": raw_calib_10["mce"],
            "mean_predictive_entropy": round(float(np.mean(raw_entropy)), 4),
            "median_predictive_entropy": round(float(np.median(raw_entropy)), 4),
            "mean_confidence_msp": round(float(np.mean(raw_msp)), 4),
            "reliability_diagram_10_bins": raw_calib_10["reliability_bins"]
        },
        "temperature_scaling_exploration": {
            "fitted_optimal_temperature": round(optimal_T, 4),
            "optimized_negative_log_likelihood": round(min_nll, 4),
            "calibrated_brier_score": calib_metrics_10["brier_score"],
            "calibrated_ece_10_bins": calib_metrics_10["ece"],
            "calibrated_ece_15_bins": calib_metrics_15["ece"],
            "calibrated_mce": calib_metrics_10["mce"],
            "ece_reduction_percentage": round(((raw_calib_10["ece"] - calib_metrics_10["ece"]) / raw_calib_10["ece"]) * 100, 2),
            "reliability_diagram_10_bins": calib_metrics_10["reliability_bins"]
        },
        "confidence_decile_error_analysis": error_analysis,
        "selective_prediction_risk_coverage": {
            "baseline_full_coverage_risk": round(risk_full, 4),
            "risk_coverage_curve": risk_coverage_curve,
            "error_reduction_at_50_coverage_percent": round(eer_50 * 100, 2)
        }
    }

    out_path = Path("experiments/reports/uncertainty_calibration_research.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(research_report, f, indent=2, ensure_ascii=False)

    print(f"\nResearch artifact saved to: {out_path}")

if __name__ == "__main__":
    run_research()
