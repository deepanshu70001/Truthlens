"""
TruthLens Phase 7: Regression and Evaluation Integrity Tests.

Verifies:
1. Production safety: Baseline head is untouched and matches expected digest.
2. Pre-training shortcut audit report integrity.
3. Candidate identity freezing file integrity (S2 and S1 checkpoints and heads).
4. Calibration report convergence on calibration partitions (zero test data).
5. Locked test evaluation results across Tiny-GenImage, Frontier, and WildAIGC.
6. Candidate decision report: REJECT_CANDIDATE_FOR_PRODUCTION and RETAIN_FROZEN_BASELINE.
7. Prediction CSV artifacts exist and are non-empty.
"""

import csv
import hashlib
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class TestPhase7Integrity:
    """Automated verification suite for TruthLens Phase 7."""

    def test_production_baseline_untouched(self):
        """Verify original frozen baseline head is untouched and matches expected hash."""
        baseline_path = REPO_ROOT / "models" / "baselines" / "siglip_frozen_head.pt"
        assert baseline_path.exists(), "Baseline head file does not exist"
        sha = compute_sha256(baseline_path)
        expected_sha = "e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538"
        assert sha == expected_sha, f"Baseline head SHA-256 mismatch: {sha} vs {expected_sha}"

    def test_format_shortcut_audit_report(self):
        """Verify Phase 7 shortcut audit report exists and confirms container format shortcut."""
        report_path = REPO_ROOT / "experiments" / "reports" / "phase7_format_audit.json"
        assert report_path.exists(), "phase7_format_audit.json does not exist"
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "partition_summaries" in data
        train_audit = data["partition_summaries"]["training_manifest"]
        crosstab = train_audit["format_crosstab"]
        assert crosstab["likely_authentic"].get("JPEG") == 3515
        assert crosstab["ai_generated"].get("PNG") == 3480
        comb_cv = data["metadata_only_classifier"]["combined_cv_results_training_manifest"]
        assert comb_cv["random_forest_roc_auc"] == 1.0

    def test_candidate_identity_freezing(self):
        """Verify candidate identity freeze file matches checkpoint files on disk."""
        freeze_path = REPO_ROOT / "experiments" / "reports" / "phase7_selected_candidate_identity.json"
        assert freeze_path.exists(), "phase7_selected_candidate_identity.json does not exist"
        with open(freeze_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        s2_info = data["selected_primary_candidate"]
        assert s2_info["model_id"] == "truthlens/siglip-base-v2-s2"
        s2_chkpt = REPO_ROOT / s2_info["checkpoint_path"]
        assert s2_chkpt.exists(), f"S2 checkpoint missing: {s2_chkpt}"
        assert compute_sha256(s2_chkpt) == s2_info["checkpoint_sha256"]

        s1_info = data["selected_secondary_candidate"]
        assert s1_info["model_id"] == "truthlens/siglip-base-v2-s1"
        s1_chkpt = REPO_ROOT / s1_info["checkpoint_path"]
        assert s1_chkpt.exists(), f"S1 checkpoint missing: {s1_chkpt}"
        assert compute_sha256(s1_chkpt) == s1_info["checkpoint_sha256"]

    def test_calibration_report_convergence(self):
        """Verify temperature calibration converged on calibration partitions."""
        cal_path = REPO_ROOT / "experiments" / "reports" / "phase7_calibration.json"
        assert cal_path.exists(), "phase7_calibration.json does not exist"
        with open(cal_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dsets = data["datasets"]
        assert "tiny_genimage" in dsets
        assert "frontier_benchmark_2026_10" in dsets
        assert "wildaigc_bench" in dsets
        assert "pooled" in dsets

        for name, info in dsets.items():
            assert info["converged"] is True, f"Calibration did not converge for {name}"
            assert info["status"] == "CONVERGED", f"Status not converged for {name}"
            assert 0.1 <= info["fitted_temperature"] <= 45.0, f"Unbounded temperature for {name}"

    def test_locked_test_evaluation_results(self):
        """Verify locked test evaluation results across all partitions."""
        eval_path = REPO_ROOT / "experiments" / "reports" / "phase7_test_evaluation.json"
        assert eval_path.exists(), "phase7_test_evaluation.json does not exist"
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check sample counts
        assert data["tiny_genimage"]["sample_count"] == 7000
        assert data["frontier_benchmark_2026_10"]["sample_count"] == 2091
        assert data["wildaigc_bench"]["sample_count"] == 1294
        assert data["tiny_genimage_matched_augmented"]["sample_count"] == 7000

        # Check S2 metrics on Tiny-GenImage
        s2_tiny = data["tiny_genimage"]["s2_candidate"]
        assert s2_tiny["balanced_accuracy"] >= 0.80
        assert s2_tiny["roc_auc"] >= 0.90

        # Check S2 metrics under matched augmentation on Tiny-GenImage
        s2_aug = data["tiny_genimage_matched_augmented"]["s2_candidate"]
        assert s2_aug["balanced_accuracy"] >= 0.75
        assert s2_aug["roc_auc"] >= 0.85

    def test_candidate_decision_and_baseline_retention(self):
        """Verify pre-registered decision criteria and baseline retention."""
        dec_path = REPO_ROOT / "experiments" / "reports" / "phase7_candidate_decision.json"
        assert dec_path.exists(), "phase7_candidate_decision.json does not exist"
        with open(dec_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["decision"] == "REJECT_CANDIDATE_FOR_PRODUCTION"
        assert data["recommendation"] == "RETAIN_FROZEN_BASELINE_IN_PRODUCTION"
        assert data["all_conditions_satisfied"] is False

        crit = data["pre_registered_criteria_evaluation"]
        assert crit["condition_1_frontier_generalization"]["passed"] is False
        assert crit["condition_2_real_image_fpr"]["passed"] is False
        assert crit["condition_3_transcoding_robustness"]["passed"] is True

    def test_prediction_csv_artifacts(self):
        """Verify prediction CSVs exist, are non-empty, and contain required columns."""
        res_dir = REPO_ROOT / "experiments" / "results" / "phase7"
        expected_files = [
            ("s2_tiny_genimage_predictions.csv", 7000),
            ("s2_frontier_benchmark_2026_10_predictions.csv", 2091),
            ("s2_wildaigc_bench_predictions.csv", 1294),
            ("baseline_tiny_genimage_predictions.csv", 7000),
            ("baseline_frontier_benchmark_2026_10_predictions.csv", 2091),
            ("baseline_wildaigc_bench_predictions.csv", 1294),
        ]
        for fname, expected_count in expected_files:
            fpath = res_dir / fname
            assert fpath.exists(), f"Prediction file missing: {fname}"
            with open(fpath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == expected_count, f"Row count mismatch in {fname}: {len(rows)} vs {expected_count}"
                assert "raw_logit" in rows[0]
                assert "uncalibrated_probability" in rows[0]
                assert "predicted_binary" in rows[0]
                assert "ground_truth_binary" in rows[0]
