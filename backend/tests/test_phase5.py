"""Phase 5 integration-contract and upload-boundary regression tests."""
import io

import pytest
from PIL import Image


def test_upload_rejects_over_dimension_limit(monkeypatch):
    from backend.app.core import security

    image = Image.new("RGB", (11, 10), color="blue")
    stream = io.BytesIO()
    image.save(stream, format="JPEG")
    monkeypatch.setattr(security.settings, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(ValueError, match="dimensions"):
        security.validate_image_file(stream.getvalue(), "sample.jpg", "image/jpeg")


def test_report_distinguishes_assessment_domains_and_preserves_partial_results():
    from backend.app.services.reporting.builder import build_report

    report = build_report(
        analysis_id="analysis-1",
        status="completed",
        image_info={"original_filename": "photo.jpg", "width": 120, "height": 80, "storage_path": "private/path"},
        authenticity={"label": "inconclusive", "ai_generated_score": 0.5},
        preprocessing={"status": "completed", "orientation_corrected": False},
        fact_checks={"status": "no_results", "claim_verifications": [{"verdict": "UNVERIFIABLE"}]},
        provenance={"status": "unavailable"},
        virality={"assessment_mode": "CONTENT_BASED_HEURISTIC", "virality_score": None},
        regional_spread={"data_status": "DATA_UNAVAILABLE", "regions": []},
        social_impact={"data_status": "DATA_UNAVAILABLE", "categories": []},
        stages={"ocr": {"status": "failed", "error": "OCR unavailable"}, "authenticity": {"status": "completed"}},
    )
    assert report["status"] == "partial"
    assert report["authenticity"]["label"] == "inconclusive"
    assert report["fact_checks"]["status"] == "no_results"
    assert report["preprocessing"]["status"] == "completed"
    assert report["virality"]["virality_score"] is None
    assert "storage_path" not in report["image"]
    assert report["summary"]["factual_claims"]["verdicts"] == ["UNVERIFIABLE"]
    assert report["summary"]["spread"]["regional_data_status"] == "DATA_UNAVAILABLE"


def test_report_keeps_stage_output_reference_without_duplicate_stage_payload():
    from backend.app.services.orchestrator import StageResult

    stage = StageResult("vision")
    stage.stage_id = "stage-1"
    stage.correlation_id = "analysis-1"
    stage.output_reference = "report.objects"
    stage.result = {"available": True, "type": "VisionResult"}
    value = stage.to_dict()
    assert value["stage_id"] == "stage-1"
    assert value["correlation_id"] == "analysis-1"
    assert value["output_reference"] == "report.objects"


def test_authenticity_detector_does_not_invent_manipulation_probability():
    from backend.app.services.authenticity.detector import AuthenticityResult

    data = AuthenticityResult(label="inconclusive", ai_generated_score=0.5, authentic_score=0.5).to_dict()
    assert data["manipulation_score"] is None
    assert "Uncalibrated" in data["score_semantics"]
