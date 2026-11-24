import numpy as np
from experiments.run_experiment3_multivariate_screening import (
    align_cluster_signs, benjamini_hochberg, correlation_clusters, signflip_test,
)


def test_correlation_clusters_detect_redundant_signals():
    rng = np.random.default_rng(7)
    x = rng.normal(size=500)
    y = 0.98 * x + rng.normal(scale=0.03, size=500)
    z = rng.normal(size=500)
    groups = correlation_clusters(np.column_stack([x, y, z]), ["x", "y", "z"])
    assert [0, 1] in groups
    assert [2] in groups


def test_negative_correlated_feature_is_sign_aligned():
    rng = np.random.default_rng(9)
    x = rng.normal(size=500)
    y = -x + rng.normal(scale=0.02, size=500)
    values = np.column_stack([x, y])
    groups = correlation_clusters(values, ["x", "y"])
    signs, _ = align_cluster_signs(values, ["x", "y"], groups)
    assert groups == [[0, 1]]
    assert signs == [1, -1]


def test_independent_features_remain_separate():
    rng = np.random.default_rng(11)
    groups = correlation_clusters(rng.normal(size=(500, 4)), ["a", "b", "c", "d"])
    assert sorted(sorted(g) for g in groups) == [[0], [1], [2], [3]]


def test_bh_adjustment_is_monotonic():
    raw = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.8}
    adjusted = benjamini_hochberg(raw)
    ordered = sorted(raw, key=raw.get)
    assert all(adjusted[k] >= raw[k] for k in raw)
    assert [adjusted[k] for k in ordered] == sorted(adjusted[k] for k in ordered)


def test_signflip_p_value_is_valid():
    groups = {
        1: {"g1": (2.0, 2), "g2": (1.0, 1)},
        2: {"g3": (1.5, 2), "g4": (0.5, 1)},
    }
    result = signflip_test(groups, n_perm=500, seed=9)
    assert 0.0 < result["p_value_two_sided"] <= 1.0
    assert result["replicates"] == 500

def test_non_adjacent_correlated_features_keep_original_column_sign_indices():
    rng = np.random.default_rng(13)
    x = rng.normal(size=600)
    independent = rng.normal(size=600)
    inverse_x = -x + rng.normal(scale=0.02, size=600)
    values = np.column_stack([x, independent, inverse_x])
    groups = correlation_clusters(values, ["x", "independent", "inverse_x"])
    signs, _ = align_cluster_signs(values, ["x", "independent", "inverse_x"], groups)
    assert groups == [[0, 2], [1]]
    assert signs == [1, 1, -1]
