"""
TruthLens Phase 3: Unit Tests for Evaluation Pipeline, Calibration & Generalization.

Tests cover:
1. Strict model identity enforcement and contract validation.
2. Calibration fitting isolation (fitting strictly on 'calibration' role, held-out validation/test).
3. Deterministic calculation of calibration diagnostics (ECE 15 bins, Brier score, NLL).
4. Threshold selection isolation on validation split.
5. Group-aware block bootstrap resampling integrity.
6. Robustness and consistency metrics for social media compression variants.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
from pathlib import Path

from experiments.calibrate_evaluation import (
    fit_temperature_scaling,
    compute_calibration_diagnostics,
    select_operating_thresholds_on_validation,
    sigmoid,
)
from experiments.analyze_results import (
    compute_metrics_at_threshold,
    compute_abstention_metrics,
    group_aware_bootstrap_ci,
    analyze_frontier_compression_robustness,
)


class TestModelIdentityVerification:
    """Tests for model identity verification in Phase 3 evaluation."""

    def test_verify_model_identity_success(self):
        from experiments.evaluate_authenticity import verify_model_identity
        from backend.app.services.authenticity.detector import AuthenticityDetector

        detector = AuthenticityDetector()
        detector.load("cpu")
        config_path = Path("experiments/configs/phase3_evaluation_config.yaml")

        identity = verify_model_identity(detector, config_path)
        assert identity["model_id"] == "truthlens/siglip-cifake-frozen-head"
        assert identity["head_sha256"] == "e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538"
        assert identity["backbone_model"] == "google/siglip-base-patch16-224"
        assert identity["backbone_revision"] == "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"

    def test_verify_model_identity_rejects_corrupted_head(self, tmp_path):
        from experiments.evaluate_authenticity import verify_model_identity
        from backend.app.services.authenticity.detector import AuthenticityDetector
        import yaml

        detector = AuthenticityDetector()
        detector.load("cpu")

        # Fake config with mismatched head hash
        bad_config = {
            "evaluation_target": {
                "model_id": "truthlens/siglip-cifake-frozen-head",
                "backbone_model": "google/siglip-base-patch16-224",
                "backbone_revision": "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
                "expected_head_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                "preprocessing_version": "siglip-frozen-v1",
            }
        }
        cfg_file = tmp_path / "bad_config.yaml"
        cfg_file.write_text(yaml.dump(bad_config), encoding="utf-8")

        with pytest.raises(ValueError, match="head_sha256 mismatch"):
            verify_model_identity(detector, cfg_file)


class TestCalibrationFittingAndDiagnostics:
    """Tests for calibration fitting and scoring metrics."""

    def test_fit_temperature_scaling_recovers_known_temperature(self):
        # Generate synthetic logits with known scale T=2.0
        np.random.seed(42)
        true_t = 2.0
        n_samples = 2000
        z = np.random.normal(0, 3.0, size=n_samples)
        p = 1.0 / (1.0 + np.exp(-z / true_t))
        y = (np.random.uniform(0, 1, size=n_samples) < p).astype(int)

        fitted_t = fit_temperature_scaling(z, y)
        assert abs(fitted_t - true_t) < 0.25

    def test_fit_temperature_requires_both_classes(self):
        margins = np.array([1.0, 2.0, 3.0])
        labels = np.array([1, 1, 1])
        with pytest.raises(ValueError, match="Calibration set must contain both classes"):
            fit_temperature_scaling(margins, labels)

    def test_compute_calibration_diagnostics_perfect_calibration(self):
        # When predicted probabilities match empirical outcomes exactly
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        # Two bins: bin 0 with prob 0.0, bin 1 with prob 1.0
        probs = np.array([0.01, 0.01, 0.01, 0.01, 0.99, 0.99, 0.99, 0.99])
        diag = compute_calibration_diagnostics(labels, probs, num_bins=15)

        assert diag["sample_count"] == 8
        assert diag["brier_score"] < 0.001
        assert diag["nll"] < 0.02
        assert diag["ece_equal_width"] < 0.02

    def test_select_operating_thresholds_on_validation(self):
        labels = np.array([0] * 50 + [1] * 50)
        # Separable with slight overlap
        probs = np.concatenate([
            np.random.uniform(0.05, 0.45, 50),
            np.random.uniform(0.55, 0.95, 50),
        ])
        thresholds = select_operating_thresholds_on_validation(labels, probs)
        assert "default_threshold" in thresholds
        assert "optimal_balanced_accuracy" in thresholds
        assert "optimal_f1" in thresholds
        assert "high_specificity_95" in thresholds
        assert "selective_abstention" in thresholds
        assert thresholds["optimal_balanced_accuracy"]["validation_balanced_accuracy"] >= 0.95


class TestResultsAnalysisAndBootstrap:
    """Tests for metrics computation, group bootstrap, and robustness."""

    def test_compute_metrics_at_threshold(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.2, 0.4, 0.6, 0.8])
        met = compute_metrics_at_threshold(y_true, y_prob, threshold=0.5)

        assert met["accuracy"] == 1.0
        assert met["balanced_accuracy"] == 1.0
        assert met["precision"] == 1.0
        assert met["recall"] == 1.0
        assert met["specificity"] == 1.0
        assert met["false_acceptance_rate"] == 0.0
        assert met["false_rejection_rate"] == 0.0
        assert met["f1_score"] == 1.0
        assert met["roc_auc"] == 1.0

    def test_group_aware_bootstrap_ci_preserves_clusters(self):
        # 10 groups, each group has 2 identical samples
        rows = []
        for g_idx in range(10):
            grp_id = f"group_{g_idx}"
            gt = g_idx % 2
            margin = 2.0 if gt == 1 else -2.0
            for i in range(2):
                rows.append({
                    "group_id": grp_id,
                    "ground_truth_binary": gt,
                    "raw_margin": margin,
                })

        boot_res = group_aware_bootstrap_ci(rows, temperature=1.0, threshold=0.5, n_bootstraps=50, seed=42)
        assert boot_res["bootstrap_iterations"] == 50
        assert boot_res["independent_clusters_sampled"] == 10
        ci = boot_res["metrics_95_ci"]["accuracy"]
        assert 0.0 <= ci["ci_lower"] <= ci["ci_upper"] <= 1.0

    def test_compute_abstention_metrics(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])
        # Abstention between 0.35 and 0.65
        res = compute_abstention_metrics(y_true, y_prob, auth_max=0.35, ai_min=0.65)
        assert res["total_samples"] == 4
        assert res["decided_samples"] == 2
        assert res["inconclusive_samples"] == 2
        assert res["coverage"] == 0.5
        assert res["abstention_rate"] == 0.5
        assert res["accuracy_on_decided"] == 1.0

    def test_frontier_compression_robustness(self):
        test_rows = [
            {"image_path": "images/1.png", "annotation_notes": "variant: original", "group_id": "g1", "ground_truth_binary": 1, "raw_margin": 2.0},
            {"image_path": "images_social/whatsapp/1.jpg", "annotation_notes": "variant: whatsapp", "group_id": "g1", "ground_truth_binary": 1, "raw_margin": 1.5},
            {"image_path": "images_social/instagram/1.jpg", "annotation_notes": "variant: instagram", "group_id": "g1", "ground_truth_binary": 1, "raw_margin": 1.8},
            {"image_path": "images_social/reshared/1.jpg", "annotation_notes": "variant: reshared", "group_id": "g1", "ground_truth_binary": 1, "raw_margin": 0.5},
        ]
        report = analyze_frontier_compression_robustness(test_rows, temperature=1.0, threshold=0.5)
        assert "original" in report["variant_evaluations"]
        assert "whatsapp" in report["variant_evaluations"]
        assert "pairwise_consistency_against_original" in report
