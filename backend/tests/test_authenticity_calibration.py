import json

import pytest

from backend.app.services.authenticity.calibration import (
    CalibrationArtifactError,
    calibrated_ai_probability,
    load_calibration_artifact,
)
from experiments.calibration_utils import (
    binary_diagnostics,
    fit_temperature,
    probabilities_from_margins,
    select_abstention_thresholds,
)


def test_temperature_scaling_improves_validation_nll_on_overconfident_logits():
    margins = [-5.0, -4.0, 4.0, 5.0]
    labels = [0, 1, 0, 1]
    temperature = fit_temperature(margins, labels)
    before = binary_diagnostics(labels, probabilities_from_margins(margins, 1.0))
    after = binary_diagnostics(labels, probabilities_from_margins(margins, temperature))
    assert temperature > 1.0
    assert after["nll"] < before["nll"]
    assert set(("nll", "brier", "ece")) <= set(after)


def test_temperature_scaling_preserves_rank_and_conflicting_case_abstains():
    assert calibrated_ai_probability(2.0, 0.0, 2.0) > 0.5
    thresholds = select_abstention_thresholds([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    assert thresholds["coverage"] == 0.0
    assert thresholds["selection_status"] == "validation_selected_with_class_error_caps"
    assert thresholds["authentic_max"] < thresholds["ai_generated_min"]


def test_calibration_artifact_is_bound_to_model_and_checkpoint(tmp_path):
    artifact = {
        "schema_version": 1,
        "method": "temperature_scaling",
        "model_id": "org/model",
        "model_revision": "abc123",
        "validation_manifest_sha256": "a" * 64,
        "validation_split": "validation",
        "sample_count": 60,
        "class_counts": {"real": 30, "ai_generated": 30},
        "ai_generation_families": ["diffusion-family-a", "diffusion-family-b"],
        "temperature": 2.0,
        "thresholds": {"authentic_max": 0.3, "ai_generated_min": 0.7},
        "metrics_before": {"nll": 1.2, "brier": 0.4, "ece": 0.2},
        "metrics_after": {"nll": 0.8, "brier": 0.3, "ece": 0.1},
        "validated": True,
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert load_calibration_artifact(path, "org/model", "abc123")["temperature"] == 2.0
    with pytest.raises(CalibrationArtifactError, match="different model"):
        load_calibration_artifact(path, "other/model", "abc123")
    with pytest.raises(CalibrationArtifactError, match="exact model checkpoint"):
        load_calibration_artifact(path, "org/model", "def456")
