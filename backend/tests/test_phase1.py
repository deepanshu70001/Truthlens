"""Deterministic regression coverage for the Phase 1 image-to-report contract."""

import json
import threading
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from backend.app.services.authenticity.detector import AuthenticityDetector
from backend.app.services.evidence.schema import EvidenceCollection, EvidenceItem
from backend.app.services.fusion.engine import fuse_evidence
from backend.app.services.reporting.builder import build_report


class FakeProcessor:
    def __call__(self, *, images, return_tensors):
        assert images.mode == "RGB"
        return {"pixel_values": torch.zeros((1, 3, 2, 2))}


class FakeModel:
    config = SimpleNamespace(id2label={0: "Real", 1: "Fake"}, _name_or_path="mock-checkpoint")

    def __init__(self, fake_score):
        self.fake_score = fake_score

    def __call__(self, **inputs):
        if self.fake_score == "error":
            raise RuntimeError("mock inference failure")
        if self.fake_score == "nan":
            logits = torch.tensor([[float("nan"), 0.0]])
        else:
            logits = torch.log(torch.tensor([[1.0 - self.fake_score, self.fake_score]]))
        return SimpleNamespace(logits=logits)


def make_detector(fake_score):
    detector = AuthenticityDetector()
    detector._lock = threading.RLock()
    detector._loaded = True
    detector._processor = FakeProcessor()
    detector._model = FakeModel(fake_score)
    detector._device = torch.device("cpu")
    return detector


@pytest.mark.parametrize(
    "fake_score", [0.80, 0.65, 0.35, 0.10, 0.50],
)
def test_detector_exposes_raw_argmax_but_abstains_without_calibration(fake_score):
    result = make_detector(fake_score).detect(Image.new("RGB", (8, 8)))
    assert result.label == "inconclusive"
    expected_prediction = "likely_ai_generated" if fake_score > 0.5 else "likely_authentic" if fake_score < 0.5 else "inconclusive"
    assert result.model_prediction == expected_prediction
    assert result.calibration_status == "not_calibrated"
    assert result.confidence is None
    assert result.raw_scores == {"real": pytest.approx(1 - fake_score), "fake": pytest.approx(fake_score)}


@pytest.mark.parametrize("fake_score", ["nan", "error"])
def test_detector_rejects_invalid_or_failed_inference(fake_score):
    with pytest.raises((ValueError, RuntimeError)):
        make_detector(fake_score).detect(Image.new("RGB", (8, 8)))


def test_detector_without_loaded_model_does_not_predict():
    detector = AuthenticityDetector()
    with pytest.raises(RuntimeError, match="Model not loaded"):
        detector.detect(Image.new("RGB", (8, 8)))


def test_empty_fusion_has_no_invented_numeric_score():
    result = fuse_evidence(EvidenceCollection())
    assert result.label == "inconclusive"
    assert result.overall_score is None
    assert result.confidence is None
    assert json.loads(json.dumps(result.to_dict()))["overall_score"] is None


def test_fusion_ignores_non_authenticity_and_invalid_duplicate_evidence():
    evidence = EvidenceCollection()
    item = EvidenceItem(
        type="factcheck_evidence", source="publisher", strength="high",
        direction="authentic", description="Claim rated accurate.",
    )
    evidence.add(item)
    evidence.add(item)
    evidence.add(EvidenceItem(
        type="model_evidence", source="bad", strength="unknown",
        direction="authentic", description="Invalid strength.",
    ))
    result = fuse_evidence(evidence)
    assert evidence.count == 2
    assert result.label == "inconclusive"
    assert result.overall_score is None


def test_fusion_rejects_non_finite_confidence():
    evidence = EvidenceCollection(items=[EvidenceItem(
        type="model_evidence", source="bad", strength="high", direction="authentic",
        description="Invalid confidence.", confidence=float("nan"),
    )])
    result = fuse_evidence(evidence)
    assert result.label == "inconclusive"
    assert result.overall_score is None


def test_report_is_partial_and_redacts_sensitive_values():
    report = build_report(
        analysis_id="analysis-1",
        status="completed",
        image_info={"original_filename": "sample.png", "sha256_hash": "abc"},
        stages={
            "authenticity": {"status": "unavailable", "error": "ImportError: stage execution failed"},
            "xai": {"status": "skipped", "result": {"heatmap_path": "C:/private/heat.png"}},
        },
        metadata={"has_exif": True, "_gps_data": {"GPSLatitude": [1, 2]}, "all_tags": {"GPS": "private"}},
    )
    encoded = json.dumps(report)
    assert report["status"] == "partial"
    assert report["authenticity"]["label"] is None
    assert report["authenticity"]["status"] == "unavailable"
    assert report["fusion"]["overall_score"] is None
    assert "C:/private" not in encoded
    assert "GPSLatitude" not in encoded
    assert "heatmap_path" not in encoded


@pytest.mark.asyncio
async def test_missing_optional_dependency_is_unavailable_stage():
    from backend.app.services.orchestrator import _run_stage

    class FakeDB:
        def add(self, _event):
            pass

        async def commit(self):
            pass

    analysis = SimpleNamespace(id="analysis-2", current_step=None, completed_steps="[]")
    results = {}

    def missing_dependency():
        raise ImportError("optional package missing")

    result = await _run_stage("optional", missing_dependency, analysis, FakeDB(), results)
    assert result is None
    assert results["optional"].status == "unavailable"
    assert "optional package missing" not in results["optional"].error
