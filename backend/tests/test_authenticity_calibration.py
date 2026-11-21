import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest
import torch
from PIL import Image

from backend.app.services.authenticity.calibration import (
    CalibrationArtifactError,
    calibrated_ai_probability,
    load_calibration_artifact,
)
from backend.app.services.authenticity.detector import AuthenticityDetector
from experiments.calibration_utils import (
    binary_diagnostics,
    fit_temperature,
    probabilities_from_margins,
    select_abstention_thresholds,
)


def _make_sample_v2_artifact(overrides=None):
    base = {
        "schema_version": 2,
        "method": "temperature_scaling",
        "model_id": "truthlens/siglip-cifake-frozen-head",
        "model_revision": "backbone:google/siglip-base-patch16-224@abc123;head_sha256:123456",
        "backbone_model": "google/siglip-base-patch16-224",
        "backbone_revision": "abc123",
        "head_sha256": "123456",
        "preprocessing_version": "siglip-frozen-v1",
        "label_mapping": {"0": "real", "1": "ai_generated"},
        "inference_config": {
            "backend": "finetuned_siglip",
            "input_channels": 3,
            "input_size": 224,
        },
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
    if overrides:
        base.update(overrides)
    return base


def _make_mock_detector(fake_score=0.9, calibration_path=""):
    class FakeProcessor:
        def __call__(self, *, images, return_tensors):
            return {"pixel_values": torch.zeros((1, 3, 224, 224))}

    class FakeModel:
        config = SimpleNamespace(id2label={0: "real", 1: "ai_generated"}, _name_or_path="test-model")

        def __call__(self, **inputs):
            logits = torch.log(torch.tensor([[1.0 - fake_score, fake_score]]))
            return SimpleNamespace(logits=logits)

    detector = AuthenticityDetector()
    detector._lock = threading.RLock()
    detector._loaded = True
    detector._processor = FakeProcessor()
    detector._model = FakeModel()
    detector._device = torch.device("cpu")
    detector._model_id = "truthlens/siglip-cifake-frozen-head"
    detector._model_revision = "backbone:google/siglip-base-patch16-224@abc123;head_sha256:123456"
    detector._backbone_id = "google/siglip-base-patch16-224"
    detector._backbone_revision = "abc123"
    detector._checkpoint_digest = "123456"
    detector._preprocessing_version = "siglip-frozen-v1"
    detector._label_mapping = {0: "real", 1: "ai_generated"}
    return detector


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


def test_v1_legacy_artifact_is_safely_rejected(tmp_path):
    artifact = {
        "schema_version": 1,
        "method": "temperature_scaling",
        "model_id": "truthlens/siglip-cifake-frozen-head",
        "model_revision": "backbone:google/siglip-base-patch16-224@abc123;head_sha256:123456",
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
    path = tmp_path / "calibration_v1.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(CalibrationArtifactError, match="Legacy schema version 1"):
        load_calibration_artifact(path, "truthlens/siglip-cifake-frozen-head", "backbone:google/siglip-base-patch16-224@abc123;head_sha256:123456")


def test_calibration_artifact_is_bound_to_model_and_checkpoint(tmp_path):
    artifact = _make_sample_v2_artifact({"model_id": "org/model", "model_revision": "abc123"})
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert load_calibration_artifact(path, "org/model", "abc123")["temperature"] == 2.0
    with pytest.raises(CalibrationArtifactError, match="different model"):
        load_calibration_artifact(path, "other/model", "abc123")
    with pytest.raises(CalibrationArtifactError, match="exact model checkpoint"):
        load_calibration_artifact(path, "org/model", "def456")


def test_correct_matching_model_identity(tmp_path):
    artifact = _make_sample_v2_artifact()
    path = tmp_path / "calibration_v2.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    detector = _make_mock_detector()
    loaded = load_calibration_artifact(
        path,
        detector._model_id,
        detector._model_revision,
        detector.model_identity(),
    )
    assert loaded["schema_version"] == 2
    assert loaded["temperature"] == 2.0
    assert loaded["head_sha256"] == "123456"


def test_wrong_model_id(tmp_path):
    artifact = _make_sample_v2_artifact({"model_id": "wrong/model-id"})
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    detector = _make_mock_detector()
    with pytest.raises(CalibrationArtifactError, match="different model"):
        load_calibration_artifact(
            path,
            detector._model_id,
            detector._model_revision,
            detector.model_identity(),
        )


def test_wrong_backbone_revision(tmp_path):
    artifact = _make_sample_v2_artifact({"backbone_revision": "different_rev_999"})
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    detector = _make_mock_detector()
    with pytest.raises(CalibrationArtifactError, match="different backbone revision"):
        load_calibration_artifact(
            path,
            detector._model_id,
            detector._model_revision,
            detector.model_identity(),
        )


def test_wrong_classification_head_hash(tmp_path):
    artifact = _make_sample_v2_artifact({"head_sha256": "mismatched_sha_000"})
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    detector = _make_mock_detector()
    with pytest.raises(CalibrationArtifactError, match="different classification head"):
        load_calibration_artifact(
            path,
            detector._model_id,
            detector._model_revision,
            detector.model_identity(),
        )


def test_wrong_preprocessing_version(tmp_path):
    artifact = _make_sample_v2_artifact({"preprocessing_version": "siglip-legacy-v0"})
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    detector = _make_mock_detector()
    with pytest.raises(CalibrationArtifactError, match="different preprocessing version"):
        load_calibration_artifact(
            path,
            detector._model_id,
            detector._model_revision,
            detector.model_identity(),
        )


def test_missing_required_artifact_fields(tmp_path):
    artifact = _make_sample_v2_artifact()
    del artifact["head_sha256"]
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    detector = _make_mock_detector()
    with pytest.raises(CalibrationArtifactError, match="missing required fields"):
        load_calibration_artifact(
            path,
            detector._model_id,
            detector._model_revision,
            detector.model_identity(),
        )


@pytest.mark.parametrize(
    "bad_temperature", [0.0, -1.5, float("nan"), float("inf")]
)
def test_invalid_temperature_values(tmp_path, bad_temperature):
    artifact = _make_sample_v2_artifact({"temperature": bad_temperature})
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    detector = _make_mock_detector()
    with pytest.raises(CalibrationArtifactError, match="Temperature must be finite"):
        load_calibration_artifact(
            path,
            detector._model_id,
            detector._model_revision,
            detector.model_identity(),
        )


@pytest.mark.parametrize(
    "bad_thresholds", [
        {"authentic_max": 0.8, "ai_generated_min": 0.2},  # authentic >= ai
        {"authentic_max": -0.1, "ai_generated_min": 0.8}, # negative
        {"authentic_max": 0.5, "ai_generated_min": 0.5},  # no uncertainty band
    ]
)
def test_invalid_threshold_values(tmp_path, bad_thresholds):
    artifact = _make_sample_v2_artifact({"thresholds": bad_thresholds})
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    detector = _make_mock_detector()
    with pytest.raises(CalibrationArtifactError, match="Decision thresholds must define a non-empty uncertainty band"):
        load_calibration_artifact(
            path,
            detector._model_id,
            detector._model_revision,
            detector.model_identity(),
        )


def test_malformed_json(tmp_path):
    path = tmp_path / "malformed.json"
    path.write_text("{\"schema_version\": 2, not_valid_json", encoding="utf-8")

    detector = _make_mock_detector()
    with pytest.raises(CalibrationArtifactError, match="not valid JSON"):
        load_calibration_artifact(
            path,
            detector._model_id,
            detector._model_revision,
            detector.model_identity(),
        )


def test_calibration_artifact_not_found(tmp_path):
    path = tmp_path / "nonexistent_file.json"

    detector = _make_mock_detector()
    with pytest.raises(CalibrationArtifactError, match="could not be read"):
        load_calibration_artifact(
            path,
            detector._model_id,
            detector._model_revision,
            detector.model_identity(),
        )


def test_uncalibrated_detector_abstention(monkeypatch):
    from backend.app.config import settings

    monkeypatch.setattr(settings, "AUTHENTICITY_CALIBRATION_PATH", "")
    detector = _make_mock_detector(fake_score=0.9)
    result = detector.detect(Image.new("RGB", (8, 8)))

    assert result.label == "inconclusive"
    assert result.model_prediction == "likely_ai_generated"
    assert result.calibration_status == "not_calibrated"
    assert result.confidence is None
    assert result.calibrated_candidate_label == "inconclusive"
    assert result.domain_validation_status == "not_evaluated"


def test_temperature_calibration_without_domain_validation(monkeypatch, tmp_path):
    from backend.app.config import settings

    artifact = _make_sample_v2_artifact()
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    monkeypatch.setattr(settings, "AUTHENTICITY_CALIBRATION_PATH", str(path))
    detector = _make_mock_detector(fake_score=0.9)
    result = detector.detect(Image.new("RGB", (8, 8)))

    assert result.calibration_status == "validated_temperature_scaling"
    assert result.calibrated_candidate_label == "likely_ai_generated"
    assert result.label == "inconclusive"
    assert result.domain_validation_status == "not_evaluated"
    assert result.confidence is None


def test_validated_artifact_handling_promotes_only_with_domain_validation(monkeypatch, tmp_path):
    from backend.app.config import settings

    artifact = _make_sample_v2_artifact()
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    monkeypatch.setattr(settings, "AUTHENTICITY_CALIBRATION_PATH", str(path))
    detector = _make_mock_detector(fake_score=0.9)

    # Without domain validation: stays inconclusive
    result_unvalidated = detector.detect(Image.new("RGB", (8, 8)))
    assert result_unvalidated.label == "inconclusive"
    assert result_unvalidated.confidence is None
    assert result_unvalidated.calibrated_candidate_label == "likely_ai_generated"

    # With domain validation established: candidate label promoted to application label
    monkeypatch.setattr(detector, "_evaluate_domain_validation", lambda img: "validated_for_input_domain")
    result_validated = detector.detect(Image.new("RGB", (8, 8)))
    assert result_validated.domain_validation_status == "validated_for_input_domain"
    assert result_validated.label == "likely_ai_generated"
    assert result_validated.confidence is not None
    assert result_validated.confidence > 0.5


def test_status_flags_cannot_promote_without_domain_validation(monkeypatch, tmp_path):
    """Artifact validation status 'eligible_for_controlled_use' and validated=True describe
    curation of the calibration split; they cannot promote application label without domain validation.
    """
    from backend.app.config import settings

    artifact = _make_sample_v2_artifact({
        "validated": True,
        "status": "eligible_for_controlled_use",
    })
    path = tmp_path / "controlled_use_artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    monkeypatch.setattr(settings, "AUTHENTICITY_CALIBRATION_PATH", str(path))
    detector = _make_mock_detector(fake_score=0.99)

    result = detector.detect(Image.new("RGB", (8, 8)))
    assert result.calibration_status == "validated_temperature_scaling"
    assert result.calibrated_candidate_label == "likely_ai_generated"
    assert result.label == "inconclusive"
    assert result.confidence is None
    assert result.domain_validation_status == "not_evaluated"


def test_hosted_model_identity_consistency():
    from backend.app.config import settings
    detector = AuthenticityDetector()
    detector._model_id = "custom/hosted-model"
    detector._model_revision = "a" * 40
    detector._backbone_id = None
    detector._backbone_revision = None
    detector._checkpoint_digest = None
    detector._preprocessing_version = "huggingface-auto"
    detector._label_mapping = {0: "real", 1: "fake"}

    identity = detector.model_identity()
    assert identity["model_id"] == "custom/hosted-model"
    assert identity["model_revision"] == "a" * 40
    assert identity["backbone_model"] is None
    assert identity["backbone_revision"] is None
    assert identity["head_sha256"] is None
    assert identity["preprocessing_version"] == "huggingface-auto"
    assert identity["label_mapping"] == {0: "real", 1: "fake"}
    assert identity["inference_config"]["backend"] == settings.AUTHENTICITY_BACKEND


def test_finetuned_model_identity_consistency():
    detector = AuthenticityDetector()
    detector._model_id = "truthlens/siglip-cifake-frozen-head"
    detector._backbone_id = "google/siglip-base-patch16-224"
    detector._backbone_revision = "b" * 40
    detector._checkpoint_digest = "c" * 64
    detector._model_revision = f"backbone:google/siglip-base-patch16-224@{detector._backbone_revision};head_sha256:{detector._checkpoint_digest}"
    detector._preprocessing_version = "siglip-frozen-v1"
    detector._label_mapping = {0: "real", 1: "ai_generated"}

    identity = detector.model_identity()
    assert identity["model_id"] == "truthlens/siglip-cifake-frozen-head"
    assert identity["backbone_model"] == "google/siglip-base-patch16-224"
    assert identity["backbone_revision"] == "b" * 40
    assert identity["head_sha256"] == "c" * 64
    assert identity["preprocessing_version"] == "siglip-frozen-v1"
    assert identity["label_mapping"] == {0: "real", 1: "ai_generated"}
