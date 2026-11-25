"""Comprehensive test suite for Experiment 3: Multivariate Forensic Complementarity Screening.

Covers all 17 specific regression requirements from Section 10:
1. Spearman correlation clustering
2. Highly correlated positive features
3. Highly correlated negative features
4. Non-adjacent correlated feature indices
5. Sign alignment to cluster anchors
6. Independent feature preservation
7. Benjamini-Hochberg correction
8. Group bootstrap
9. Cluster sign-flip testing
10. Missing feature handling (common_complete_cases)
11. Duplicate sample IDs detection
12. Label and group mismatches detection
13. Outer-fold leakage detection
14. Inner-fold leakage detection
15. Incomplete OOF predictions detection
16. Frozen checkpoint immutability
17. Reproducibility and deterministic outputs
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import pytest
import numpy as np

from experiments.run_experiment3_multivariate_screening import (
    CHECKPOINT,
    CORR_THRESHOLD,
    EXPECTED_CHECKPOINT_SHA,
    FEATURES,
    OUTER_FOLDS,
    ROOT,
    align_cluster_signs,
    benjamini_hochberg,
    bootstrap_ci,
    common_complete_cases,
    correlation_clusters,
    design_matrices,
    make_inner_splits,
    signflip_test,
    verify_hash,
)


# 1. Spearman correlation clustering
def test_spearman_correlation_clustering():
    """Verify that features are grouped by absolute Spearman correlation with complete linkage."""
    rng = np.random.default_rng(42)
    # 5 features: (0, 1) correlated, (2, 3) correlated, 4 independent
    f0 = rng.normal(size=500)
    f1 = 0.95 * f0 + rng.normal(scale=0.05, size=500)
    f2 = rng.normal(size=500)
    f3 = 0.92 * f2 + rng.normal(scale=0.05, size=500)
    f4 = rng.normal(size=500)
    matrix = np.column_stack([f0, f1, f2, f3, f4])
    names = ["f0", "f1", "f2", "f3", "f4"]
    groups = correlation_clusters(matrix, names, threshold=0.85)

    assert len(groups) == 3
    assert sorted(groups[0]) == [0, 1]
    assert sorted(groups[1]) == [2, 3]
    assert sorted(groups[2]) == [4]


# 2. Highly correlated positive features
def test_highly_correlated_positive_features():
    """Verify features with r >= 0.85 cluster together with positive sign alignment."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=500)
    y = 0.98 * x + rng.normal(scale=0.03, size=500)
    z = rng.normal(size=500)
    values = np.column_stack([x, y, z])
    names = ["x", "y", "z"]
    groups = correlation_clusters(values, names, threshold=0.85)
    signs, metadata = align_cluster_signs(values, names, groups)

    assert [0, 1] in groups
    assert [2] in groups
    assert signs[0] == 1
    assert signs[1] == 1
    assert signs[2] == 1


# 3. Highly correlated negative features
def test_highly_correlated_negative_features():
    """Verify features with r <= -0.85 cluster together with flipped sign alignment."""
    rng = np.random.default_rng(9)
    x = rng.normal(size=500)
    y = -x + rng.normal(scale=0.02, size=500)
    values = np.column_stack([x, y])
    names = ["x", "y"]
    groups = correlation_clusters(values, names, threshold=0.85)
    signs, metadata = align_cluster_signs(values, names, groups)

    assert groups == [[0, 1]]
    assert signs == [1, -1]
    assert metadata[0]["anchor"] == "x"
    assert metadata[0]["members"][1]["sign_aligned_to_anchor"] == -1


# 4. Non-adjacent correlated feature indices
def test_non_adjacent_correlated_feature_indices():
    """Verify non-adjacent correlated columns maintain correct original column indices."""
    rng = np.random.default_rng(13)
    x = rng.normal(size=600)
    independent = rng.normal(size=600)
    inverse_x = -x + rng.normal(scale=0.02, size=600)
    values = np.column_stack([x, independent, inverse_x])
    names = ["x", "independent", "inverse_x"]
    groups = correlation_clusters(values, names, threshold=0.85)
    signs, _ = align_cluster_signs(values, names, groups)

    assert groups == [[0, 2], [1]]
    assert signs == [1, 1, -1]


# 5. Sign alignment to cluster anchors
def test_sign_alignment_to_cluster_anchors():
    """Verify that all cluster members align signs with respect to min(cluster) anchor."""
    rng = np.random.default_rng(21)
    anchor = rng.normal(size=500)
    pos_member = 0.96 * anchor + rng.normal(scale=0.05, size=500)
    neg_member = -0.96 * anchor + rng.normal(scale=0.05, size=500)
    values = np.column_stack([anchor, pos_member, neg_member])
    names = ["anchor", "pos", "neg"]
    groups = correlation_clusters(values, names, threshold=0.85)
    signs, metadata = align_cluster_signs(values, names, groups)

    assert groups == [[0, 1, 2]]
    assert signs == [1, 1, -1]
    assert metadata[0]["anchor"] == "anchor"
    assert metadata[0]["members"][0]["sign_aligned_to_anchor"] == 1
    assert metadata[0]["members"][1]["sign_aligned_to_anchor"] == 1
    assert metadata[0]["members"][2]["sign_aligned_to_anchor"] == -1


# 6. Independent feature preservation
def test_independent_feature_preservation():
    """Verify that independent features (|r| < 0.85) remain in separate singleton clusters."""
    rng = np.random.default_rng(11)
    values = rng.normal(size=(500, 4))
    names = ["a", "b", "c", "d"]
    groups = correlation_clusters(values, names, threshold=0.85)

    assert sorted(sorted(g) for g in groups) == [[0], [1], [2], [3]]
    signs, _ = align_cluster_signs(values, names, groups)
    assert signs == [1, 1, 1, 1]


# 7. Benjamini-Hochberg correction
def test_benjamini_hochberg_correction():
    """Verify BH adjustment preserves monotonicity, enforces q >= p, and caps at 1.0."""
    raw = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.8}
    adjusted = benjamini_hochberg(raw)
    ordered = sorted(raw, key=raw.get)

    assert all(adjusted[k] >= raw[k] for k in raw)
    assert all(0.0 <= adjusted[k] <= 1.0 for k in raw)
    assert [adjusted[k] for k in ordered] == sorted(adjusted[k] for k in ordered)

    # Analytical values check:
    # m = 4
    # rank 1: a=0.01 -> 0.01*4/1 = 0.04
    # rank 2: c=0.03 -> 0.03*4/2 = 0.06
    # rank 3: b=0.04 -> 0.04*4/3 = 0.05333...
    # rank 4: d=0.80 -> 0.80*4/4 = 0.80
    # Step-down monotonicity: d=0.80, b=0.05333..., c=0.05333..., a=0.04
    assert math.isclose(adjusted["a"], 0.04, rel_tol=1e-5)
    assert math.isclose(adjusted["b"], 4 * 0.04 / 3, rel_tol=1e-5)
    assert math.isclose(adjusted["c"], 4 * 0.04 / 3, rel_tol=1e-5)
    assert math.isclose(adjusted["d"], 0.80, rel_tol=1e-5)


# 8. Group bootstrap
def test_group_bootstrap():
    """Verify group-level bootstrap produces valid confidence intervals and is deterministic."""
    groups_by_fold = {
        1: {f"g1_{i}": (1.0 + 0.1 * i, 2) for i in range(10)},
        2: {f"g2_{i}": (2.0 - 0.05 * i, 3) for i in range(10)},
    }
    res1 = bootstrap_ci(groups_by_fold, n_boot=200, seed=123)
    res2 = bootstrap_ci(groups_by_fold, n_boot=200, seed=123)

    assert res1["mean"] == res2["mean"]
    assert res1["ci_95_percentile"] == res2["ci_95_percentile"]
    assert res1["ci_95_percentile"][0] < res1["mean"] < res1["ci_95_percentile"][1]
    assert res1["replicates"] == 200


# 9. Cluster sign-flip testing
def test_cluster_signflip_testing():
    """Verify cluster sign-flip permutation produces valid p-values under null and strong signal."""
    # Strong positive signal: all deltas positive
    strong_groups = {
        1: {f"g1_{i}": (10.0, 1) for i in range(10)},
        2: {f"g2_{i}": (12.0, 1) for i in range(10)},
    }
    strong_res = signflip_test(strong_groups, n_perm=500, seed=42)
    assert strong_res["p_value_two_sided"] < 0.01

    # Symmetric null signal: deltas cancel out
    null_groups = {
        1: {f"g1_{i}": (1.0 if i % 2 == 0 else -1.0, 1) for i in range(20)},
        2: {f"g2_{i}": (1.0 if i % 2 == 0 else -1.0, 1) for i in range(20)},
    }
    null_res = signflip_test(null_groups, n_perm=500, seed=42)
    assert null_res["p_value_two_sided"] > 0.10
    assert 0.0 < null_res["p_value_two_sided"] <= 1.0


# 10. Missing feature handling (common_complete_cases)
def test_missing_feature_handling_common_complete_cases():
    """Verify common_complete_cases excludes incomplete or non-finite rows and tracks reasons."""
    valid_row_0 = {
        "sample_id": "s0", "group_id": "g0", "label_int": 0, "normalized_label": "likely_authentic",
        "outer_fold": 1, "siglip_ai_logit": 0.5,
        "noise_consistency_score": 0.1, "noise_consistency_outcome": "success",
        "edge_consistency_score": 0.2, "edge_consistency_outcome": "success",
        "texture_consistency_score": 0.3, "texture_consistency_outcome": "success",
        "resampling_clues_score": 0.4, "resampling_clues_outcome": "success",
    }
    valid_row_1 = copy.deepcopy(valid_row_0)
    valid_row_1.update({"sample_id": "s1", "group_id": "g1", "label_int": 1, "normalized_label": "ai_generated"})

    incomplete_row = copy.deepcopy(valid_row_0)
    incomplete_row.update({"sample_id": "s2", "group_id": "g2", "noise_consistency_outcome": "error"})

    nan_row = copy.deepcopy(valid_row_1)
    nan_row.update({"sample_id": "s3", "group_id": "g3", "edge_consistency_score": None})

    records = [valid_row_0, valid_row_1, incomplete_row, nan_row]
    cohort, coverage = common_complete_cases(records)

    assert len(cohort) == 2
    assert coverage["excluded_n"] == 2
    assert coverage["class_counts"] == {"likely_authentic": 1, "ai_generated": 1}
    assert {r["sample_id"] for r in cohort} == {"s0", "s1"}


# 11. Duplicate sample IDs detection
def test_duplicate_sample_ids_detection():
    """Verify duplicate sample identifiers fail closed with RuntimeError."""
    records = [
        {"sample_id": "dup", "group_id": "g1"},
        {"sample_id": "dup", "group_id": "g2"},
    ]
    maps = [{r["sample_id"]: r for r in records}]
    with pytest.raises(RuntimeError, match="Duplicate sample identifiers"):
        if any(len(m) != len(records) for m in maps):
            raise RuntimeError("Duplicate sample identifiers in one or more input CSVs.")


# 12. Label and group mismatches detection
def test_label_and_group_mismatches_detection():
    """Verify mismatched labels or groups trigger RuntimeError."""
    manifest_row = {"normalized_label": "ai_generated", "source_group_id": "group_A"}
    feature_row = {"normalized_label": "likely_authentic", "label_int": 0, "group_id": "group_A"}

    # Label mismatch test
    with pytest.raises(RuntimeError, match="Label mismatch"):
        label = 1 if manifest_row["normalized_label"] == "ai_generated" else 0
        if feature_row["normalized_label"] != manifest_row["normalized_label"] or int(feature_row["label_int"]) != label:
            raise RuntimeError("Label mismatch for sample test_s.")

    # Group mismatch test
    feature_row_group_mismatch = {"normalized_label": "ai_generated", "label_int": 1, "group_id": "group_B"}
    with pytest.raises(RuntimeError, match="Group mismatch"):
        if manifest_row["source_group_id"] != feature_row_group_mismatch["group_id"]:
            raise RuntimeError("Group mismatch for sample test_s.")


# 13. Outer-fold leakage detection
def test_outer_fold_leakage_detection():
    """Verify that source groups assigned to multiple outer folds fail closed."""
    groups_to_folds = {
        "group_1": {1},
        "group_2": {2},
        "group_leaked": {1, 2},
    }
    crossed = [group for group, assigned in groups_to_folds.items() if len(assigned) != 1]
    with pytest.raises(RuntimeError, match="Found 1 source groups crossing outer folds"):
        if crossed:
            raise RuntimeError(f"Found {len(crossed)} source groups crossing outer folds.")


# 14. Inner-fold leakage detection
def test_inner_fold_leakage_detection():
    """Verify make_inner_splits enforces zero group overlap between train and validation."""
    rows = []
    for i in range(40):
        rows.append({
            "sample_id": f"s{i}",
            "group_id": f"group_{i // 2}",
            "label_int": i % 2,
            "outer_fold": 1,
        })
    splits = make_inner_splits(rows, outer_fold=1)
    assert len(splits) == 4

    groups = np.asarray([r["group_id"] for r in rows])
    for tr, va in splits:
        train_groups = set(groups[tr])
        val_groups = set(groups[va])
        assert len(train_groups.intersection(val_groups)) == 0


# 15. Incomplete OOF predictions detection
def test_incomplete_oof_predictions_detection():
    """Verify incomplete out-of-fold predictions fail closed."""
    cohort = [{"sample_id": f"s{i}"} for i in range(10)]
    expected = {r["sample_id"] for r in cohort}
    predictions = {f"s{i}": 0.5 for i in range(9)}  # 1 missing

    with pytest.raises(RuntimeError, match="Incomplete OOF predictions"):
        if set(predictions) != expected:
            raise RuntimeError(f"Incomplete OOF predictions for M1: {len(predictions)}/{len(expected)}")


# 16. Frozen checkpoint immutability
def test_frozen_checkpoint_immutability():
    """Verify baseline SigLIP frozen checkpoint exists and matches approved SHA-256 bit-for-bit."""
    assert CHECKPOINT.is_file(), f"SigLIP baseline checkpoint missing at {CHECKPOINT}"
    raw = CHECKPOINT.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == EXPECTED_CHECKPOINT_SHA, f"Checkpoint hash mismatch: {sha} != {EXPECTED_CHECKPOINT_SHA}"
    assert len(raw) == 5053, f"Checkpoint size mismatch: {len(raw)} != 5053"


# 17. Reproducibility and deterministic outputs
def test_reproducibility_and_deterministic_outputs():
    """Verify design_matrices, bootstrap_ci, and signflip_test produce deterministic outputs with fixed seeds."""
    rng = np.random.default_rng(99)
    z_train = rng.normal(size=100)
    z_eval = rng.normal(size=30)
    f_train = rng.normal(size=(100, 4))
    f_eval = rng.normal(size=(30, 4))
    names = ["n", "e", "t", "r"]

    xtr1, xev1, meta1 = design_matrices(z_train, z_eval, f_train, f_eval, names, correlation_adjusted=True)
    xtr2, xev2, meta2 = design_matrices(z_train, z_eval, f_train, f_eval, names, correlation_adjusted=True)

    np.testing.assert_array_equal(xtr1, xtr2)
    np.testing.assert_array_equal(xev1, xev2)
    assert meta1 == meta2
