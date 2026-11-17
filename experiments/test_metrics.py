"""Metric utility tests on toy vectors; no model or research sample is involved."""
from PIL import Image

from experiments.evaluate_authenticity import binary_metrics, calibration_diagnostics, transforms
from experiments.dataset_audit import near_candidate_pairs


def test_binary_metrics_known_toy_vector():
    result = binary_metrics([0, 0, 1, 1], [0, 1, 1, 1], [.1, .7, .8, .4])
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
    assert result["accuracy"] == .75
    assert result["roc_auc"] == .75
    assert abs(result["average_precision_pr_auc"] - 5/6) < 1e-9


def test_calibration_diagnostics_known_toy_vector():
    result = calibration_diagnostics([0, 1], [.1, .9], bins=10)
    assert abs(result["brier_score"] - .01) < 1e-9
    assert abs(result["ece"] - .1) < 1e-9


def test_fixed_robustness_transforms_keep_image_size():
    image = Image.new("RGB", (64, 48), "#789abc")
    results = transforms(image)
    assert len(results) == 7
    assert all(candidate.size == image.size for candidate in results.values())


def test_bk_tree_candidate_pair_count_known_toy_hashes():
    # 0↔1 and 1↔7 are within two bits; 0↔7 is not.
    assert near_candidate_pairs([0, 1, 7], radius=2) == 2
