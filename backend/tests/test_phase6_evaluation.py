"""
TruthLens Phase 6: Calibration, Held-Out Evaluation & Generalization Regression Tests.

Verifies:
1. Model checkpoint identity and cryptographic reproducibility.
2. Calibration partition strict isolation.
3. Locked test split evaluation prediction schemas and sample counts.
4. Domain confounding and transcoding audit artifacts.
5. Generator generalization (seen vs unseen breakdown).
6. Compression robustness artifacts.
7. Group bootstrap statistics and formal candidate decision (RETAIN_BASELINE).
8. Publication figure generation.
"""

import csv
import json
from pathlib import Path

import pytest


def test_phase6_model_identity_audit():
    """Verify Phase 6 model identity audit report and cryptographic digests."""
    report_path = Path("experiments/reports/phase6_model_identity_audit.json")
    assert report_path.exists(), f"Missing {report_path}"

    with open(report_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert audit["status"] == "PASS"
    cand = audit["candidate_model"]
    assert cand["checkpoint_sha256"] == "541956a1f766d5093aaec4733f616afaf6deaf5aad1e0bbd6e59f95687a4b96d"
    assert cand["head_sha256"] == "38b12fb1931158a771cb6b64bd5745ef7f7796b27df6b9eb292c281566689b68"
    assert cand["checkpoint_sha256_match"] is True
    assert cand["head_sha256_match"] is True

    base = audit["baseline_model"]
    assert base["head_sha256"] == "e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538"
    assert base["head_sha256_match"] is True
    assert base["status"] == "PRESERVED_UNCHANGED"

    repro = audit["validation_reproducibility"]
    assert repro["metrics_reproduced_identically"] is True
    assert repro["reproduced_at_audit"]["balanced_accuracy"] == 0.8667
    assert repro["reproduced_at_audit"]["roc_auc"] == 0.9577


def test_phase6_calibration_partition_isolation():
    """Verify that calibration fitting was strictly isolated to calibration splits."""
    cal_path = Path("experiments/reports/phase6_calibration.json")
    rel_path = Path("experiments/reports/phase6_reliability_data.json")
    assert cal_path.exists()
    assert rel_path.exists()

    with open(cal_path, "r", encoding="utf-8") as f:
        cal = json.load(f)

    gov = cal["protocol_governance"]
    assert gov["fitted_exclusively_on_calibration"] is True
    assert gov["validation_data_used"] is False
    assert gov["test_data_used"] is False
    assert gov["status"] == "PASS"

    ds = cal["datasets"]
    assert "tiny_genimage" in ds
    assert "frontier_benchmark_2026_10" in ds
    assert "wildaigc_bench" in ds
    assert "pooled" in ds

    # Temperatures must be positive and finite
    for k, v in ds.items():
        assert v["optimal_temperature"] > 0.0
        assert v["optimizer_converged"] is True


def test_phase6_prediction_files_and_schemas():
    """Verify that all 6 prediction CSVs exist and have correct schemas and counts."""
    results_dir = Path("experiments/results/phase6")
    expected_files = {
        "e3_tiny_genimage_predictions.csv": 7000,
        "e3_frontier_predictions.csv": 2091,
        "e3_wildaigc_predictions.csv": 1294,
        "baseline_tiny_genimage_predictions.csv": 7000,
        "baseline_frontier_predictions.csv": 2091,
        "baseline_wildaigc_predictions.csv": 1294,
    }

    required_cols = {
        "sample_id",
        "image_path",
        "ground_truth_binary",
        "raw_logit",
        "uncalibrated_probability",
        "calibrated_probability_fit",
        "predicted_binary",
        "is_correct",
    }

    for fname, expected_count in expected_files.items():
        fpath = results_dir / fname
        assert fpath.exists(), f"Prediction file missing: {fpath}"

        with open(fpath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            assert required_cols.issubset(set(reader.fieldnames))
            rows = list(reader)
            assert len(rows) == expected_count, f"{fname} has {len(rows)} rows, expected {expected_count}"


def test_phase6_domain_confounding_artifacts():
    """Verify domain confounding and matched transcoding analysis results."""
    domain_path = Path("experiments/reports/phase6_domain_confounding.json")
    enc_path = Path("experiments/reports/phase6_encoding_robustness.csv")
    assert domain_path.exists()
    assert enc_path.exists()

    with open(domain_path, "r", encoding="utf-8") as f:
        dom = json.load(f)

    # Check that format distribution is 100% JPEG real vs 100% PNG AI in Tiny-GenImage
    fmt = dom["format_distribution"]
    assert fmt["JPEG"]["real"] == 3500 and fmt["JPEG"]["ai"] == 0
    assert fmt["PNG"]["ai"] == 3500 and fmt["PNG"]["real"] == 0

    # Check matched-format transcoding: E3 has high flip rate under JPEG transcoding (>40%)
    mf = dom["matched_format_evaluation"]
    assert mf["e3_candidate_robustness"]["jpeg90"]["flip_rate_pct"] > 40.0
    assert mf["e3_candidate_robustness"]["jpeg90"]["mean_absolute_logit_shift"] > 5.0


def test_phase6_generator_and_compression_reports():
    """Verify generator generalization and Frontier compression reports."""
    gen_path = Path("experiments/reports/phase6_generator_generalization.json")
    comp_path = Path("experiments/reports/phase6_compression_robustness.json")
    assert gen_path.exists()
    assert comp_path.exists()

    with open(gen_path, "r", encoding="utf-8") as f:
        gen = json.load(f)

    tiny_gens = gen["tiny_genimage_generators"]
    # Seen generators (adm, biggan, glide) have >90% recall on E3
    assert tiny_gens["adm"]["e3_candidate"]["ai_detection_rate"] > 0.90
    assert tiny_gens["biggan"]["e3_candidate"]["ai_detection_rate"] > 0.90
    assert tiny_gens["glide"]["e3_candidate"]["ai_detection_rate"] > 0.90

    # Held out generators (midjourney, wukong, vqdm) have lower recall on E3
    assert tiny_gens["vqdm"]["e3_candidate"]["ai_detection_rate"] < 0.30
    assert tiny_gens["midjourney"]["e3_candidate"]["ai_detection_rate"] < 0.60


def test_phase6_statistical_comparison_and_candidate_decision():
    """Verify group bootstrap statistical results and final candidate decision."""
    stat_path = Path("experiments/reports/phase6_statistical_comparison.json")
    dec_path = Path("experiments/reports/phase6_candidate_decision.json")
    assert stat_path.exists()
    assert dec_path.exists()

    with open(stat_path, "r", encoding="utf-8") as f:
        stat = json.load(f)

    # Frontier benchmark delta ROC-AUC 95% CI is strictly negative
    frontier_auc_delta = stat["frontier_benchmark_2026_10"]["roc_auc"]["delta"]
    assert frontier_auc_delta["mean"] < 0.0
    assert frontier_auc_delta["ci_95_high"] < 0.0

    with open(dec_path, "r", encoding="utf-8") as f:
        dec = json.load(f)

    assert dec["final_candidate_decision"] == "RETAIN_BASELINE"
    assert dec["candidate_disposition"] == "REJECT_CANDIDATE_FOR_PRODUCTION"


def test_phase6_figures_exist():
    """Verify that all 8 publication figures were generated and are non-empty."""
    figures_dir = Path("experiments/reports/figures")
    expected_figures = [
        "phase6_roc_curves.png",
        "phase6_pr_curves.png",
        "phase6_confusion_matrices.png",
        "phase6_reliability_diagrams.png",
        "phase6_generator_performance.png",
        "phase6_encoding_robustness.png",
        "phase6_compression_robustness.png",
        "phase6_paired_metric_differences.png",
    ]
    for fig in expected_figures:
        p = figures_dir / fig
        assert p.exists(), f"Missing figure: {p}"
        assert p.stat().st_size > 5000, f"Figure {p} is empty or too small"
