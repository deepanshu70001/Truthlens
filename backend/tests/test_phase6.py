"""Controlled evidence-fusion behavior checks; not model-accuracy evaluation."""
from backend.app.services.evidence.schema import EvidenceCollection, EvidenceItem
from backend.app.services.fusion.engine import fuse_evidence


def _item(kind="model_evidence", source="detector", direction="synthetic", status="completed", description="signal", calibrated=False):
    return EvidenceItem(type=kind, source=source, strength="high", direction=direction,
                        description=description, status=status, confidence=0.9, calibrated=calibrated)


def test_unavailable_directional_evidence_does_not_vote():
    collection = EvidenceCollection(items=[_item(status="unavailable")])
    assessment = fuse_evidence(collection)
    assert assessment.label == "inconclusive"
    assert assessment.overall_score is None
    assert "Ignored unavailable evidence" in assessment.reasoning[0]


def test_failed_and_skipped_evidence_do_not_change_completed_result():
    base = fuse_evidence(EvidenceCollection(items=[_item()]))
    with_unavailable = fuse_evidence(EvidenceCollection(items=[
        _item(), _item(source="failed_detector", status="failed", description="failure"),
        _item(source="not_run", status="skipped", description="skip"),
    ]))
    assert with_unavailable.label == base.label
    assert with_unavailable.overall_score == base.overall_score


def test_completed_forensics_are_informational_not_authenticity_votes():
    model_only = fuse_evidence(EvidenceCollection(items=[_item(calibrated=True)]))
    with_forensics = fuse_evidence(EvidenceCollection(items=[
        _item(calibrated=True), _item(kind="forensic_evidence", source="ela", direction="authentic", description="forensic cue"),
    ]))
    assert with_forensics.label == model_only.label
    assert with_forensics.overall_score == model_only.overall_score


def test_duplicate_completed_evidence_is_counted_once():
    one = _item(calibrated=True)
    collection = EvidenceCollection(items=[one, _item(calibrated=True)])
    assert fuse_evidence(collection).overall_score == fuse_evidence(EvidenceCollection(items=[one])).overall_score


def test_uncalibrated_high_softmax_vote_cannot_determine_authenticity():
    assessment = fuse_evidence(EvidenceCollection(items=[_item()]))
    assert assessment.label == "inconclusive"
    assert assessment.confidence is None
    assert any("uncalibrated" in reason for reason in assessment.reasoning)


def test_calibration_without_input_domain_validation_does_not_cast_directional_vote():
    from types import SimpleNamespace
    from backend.app.services.evidence.schema import build_evidence_from_authenticity

    output = SimpleNamespace(
        label="inconclusive", model_prediction="likely_authentic", confidence=0.99,
        ai_generated_score=0.01, authentic_score=0.99, model_name="detector",
        calibrated_ai_generated_probability=0.01,
        calibration_status="validated_temperature_scaling",
        calibrated_candidate_label="likely_authentic",
        domain_validation_status="not_evaluated",
    )
    evidence = build_evidence_from_authenticity(output)
    assert evidence[0].direction == "neutral"
    assert evidence[0].calibrated is False
    assert fuse_evidence(EvidenceCollection(items=evidence)).label == "inconclusive"


def test_calibrated_candidate_is_not_application_label_without_domain_validation(monkeypatch):
    import torch
    from types import SimpleNamespace
    from PIL import Image
    from backend.app.config import settings
    import backend.app.services.authenticity.detector as detector_module
    from backend.app.services.authenticity.detector import AuthenticityDetector

    artifact = {"temperature": 1.0, "thresholds": {"authentic_max": 0.3, "ai_generated_min": 0.7}}
    monkeypatch.setattr(settings, "AUTHENTICITY_CALIBRATION_PATH", "reviewed-calibration.json")
    monkeypatch.setattr(detector_module, "load_calibration_artifact", lambda *args: artifact)

    class Processor:
        def __call__(self, **kwargs):
            return {"pixel_values": torch.zeros((1, 3, 4, 4))}

    class Model:
        config = SimpleNamespace(id2label={0: "Real", 1: "Fake"}, _name_or_path="checkpoint")

        def __call__(self, **kwargs):
            return SimpleNamespace(logits=torch.tensor([[8.0, -8.0]]))

    detector = AuthenticityDetector()
    detector._loaded = True
    detector._processor = Processor()
    detector._model = Model()
    detector._device = torch.device("cpu")
    result = detector.detect(Image.new("RGB", (8, 8)))
    assert result.calibration_status == "validated_temperature_scaling"
    assert result.calibrated_candidate_label == "likely_authentic"
    assert result.label == "inconclusive"
    assert result.domain_validation_status == "not_evaluated"
    assert result.confidence is None


def test_conflicting_calibrated_models_are_inconclusive():
    assessment = fuse_evidence(EvidenceCollection(items=[
        _item(source="model_a", direction="authentic", calibrated=True),
        _item(source="model_b", direction="synthetic", calibrated=True),
    ]))
    assert assessment.label == "inconclusive"
    assert assessment.conflicting is True


def test_report_never_promotes_raw_999_softmax_to_authentic_decision_band():
    from backend.app.services.reporting.builder import build_report

    raw_output = {
        "status": "completed",
        "label": "inconclusive",
        "model_prediction": "likely_authentic",
        "ai_generated_score": 0.001,
        "authentic_score": 0.999,
        "confidence": None,
        "raw_max_softmax_score": 0.999,
        "calibrated_ai_generated_probability": None,
        "calibration_status": "not_calibrated",
        "domain_validation_status": "not_evaluated",
        "score_semantics": "Raw scores only.",
    }
    report = build_report(
        analysis_id="raw-score-regression",
        status="completed",
        authenticity=raw_output,
        assessment=fuse_evidence(EvidenceCollection(items=[_item()] )).to_dict(),
        stages={"authenticity": {"status": "completed"}, "fusion": {"status": "completed"}},
    )
    assert report["authenticity"]["decision_band"] is None
    assert report["authenticity"]["label"] == "inconclusive"
    assert report["authenticity"]["model_prediction"] == "likely_authentic"


def test_detector_preserves_checkpoint_label_mapping_and_explicit_rgb_preprocessing():
    import torch
    from types import SimpleNamespace
    from PIL import Image
    from backend.app.services.authenticity.detector import AuthenticityDetector

    seen = {}

    class Processor:
        def __call__(self, *, images, return_tensors):
            seen["mode"] = images.mode
            seen["size"] = images.size
            return {"pixel_values": torch.zeros((1, 3, 224, 224))}

    class Model:
        config = SimpleNamespace(id2label={0: "Real", 1: "Fake"}, _name_or_path="checkpoint")

        def __call__(self, **inputs):
            return SimpleNamespace(logits=torch.tensor([[8.0, -8.0]]))

    detector = AuthenticityDetector()
    detector._loaded = True
    detector._processor = Processor()
    detector._model = Model()
    detector._device = torch.device("cpu")
    result = detector.detect(Image.new("L", (80, 50), color=127))

    assert seen == {"mode": "RGB", "size": (80, 50)}
    assert result.model_prediction == "likely_authentic"
    assert result.label == "inconclusive"
    assert result.ai_generated_score < result.authentic_score
    assert result.calibration_status == "not_calibrated"
    assert result.calibrated_ai_generated_probability is None
    assert result.confidence is None
    assert result.to_dict()["score_semantics"].startswith("ai_generated_score")
