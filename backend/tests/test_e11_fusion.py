"""Controlled E11 contract tests; no model or protected evaluation data used."""

import json
from types import SimpleNamespace

from backend.app.services.evidence.schema import EvidenceCollection, EvidenceItem
from backend.app.services.fusion.engine import fuse_evidence


LABELS = {
    "authentic": "likely_authentic",
    "synthetic": "likely_ai_generated",
    "manipulated": "likely_manipulated",
}


def _vote(direction="authentic", *, status="completed", outcome="success", analysis_id="analysis-1", **overrides):
    probability = {"authentic": 0.1, "synthetic": 0.9, "manipulated": 0.7}[direction]
    data = {
        "label": LABELS[direction],
        "calibrated_candidate_label": LABELS[direction],
        "model": "truthlens/siglip-cifake-frozen-head",
        "calibration_status": "validated_temperature_scaling",
        "domain_validation_status": "validated_for_input_domain",
        "calibrated_ai_generated_probability": probability,
    }
    data.update(overrides.pop("data", {}))
    return EvidenceItem(
        type="model_evidence", source="authenticity_detector", strength="medium",
        direction=direction, description=f"validated {direction} detector finding",
        data=data, confidence=overrides.pop("confidence", probability), calibrated=True,
        analysis_id=analysis_id, stage_name="authenticity", status=status,
        outcome=outcome, **overrides,
    )


def _stage(status="completed", outcome="success", *, model_name="truthlens/siglip-cifake-frozen-head"):
    return SimpleNamespace(
        status=status, outcome=outcome, stage_id="stage-auth-1", runtime_ms=4,
        correlation_id="analysis-1",
        error_category=None, module_identity={"model_name": model_name, "model_version": "test-revision"},
    )


def test_failed_and_unavailable_stage_outputs_never_vote():
    for stage_status in ("failed", "unavailable", "skipped"):
        result = fuse_evidence(
            EvidenceCollection(items=[_vote()]),
            stage_results={"authenticity": _stage(stage_status)}, analysis_id="analysis-1",
        )
        assert result.label == "inconclusive"
        assert result.overall_score is None
        assert result.evidence_excluded[0]["reason"] == f"stage_{stage_status}"


def test_successful_empty_factcheck_is_informational_not_negative_evidence():
    from backend.app.services.evidence.schema import build_evidence_from_factcheck
    from backend.app.services.factcheck.service import FactCheckResult

    empty = build_evidence_from_factcheck(FactCheckResult(status="no_results", queries_performed=2))[0]
    result = fuse_evidence(EvidenceCollection(items=[empty]), stage_results={
        "fact_check": SimpleNamespace(status="completed", outcome="no_results"),
    })
    assert result.label == "inconclusive"
    assert not result.evidence_excluded
    assert result.informational_findings[0]["outcome"] == "no_results"
    assert any(item["stage_name"] == "fact_check" for item in result.missing_evidence)


def test_no_text_ocr_outcome_is_successful_context_not_a_vote():
    from backend.app.services.evidence.schema import build_evidence_from_ocr

    ocr = SimpleNamespace(status="completed", has_text=False, engine="tesseract", warnings=[])
    item = build_evidence_from_ocr(ocr)[0]
    result = fuse_evidence(EvidenceCollection(items=[item]))
    assert item.status == "completed" and item.outcome == "no_text"
    assert result.label == "inconclusive"
    assert result.informational_findings[0]["outcome"] == "no_text"


def test_not_applicable_evidence_is_excluded_without_direction():
    item = _vote(outcome="not_applicable")
    result = fuse_evidence(EvidenceCollection(items=[item]))
    assert result.label == "inconclusive"
    assert result.evidence_excluded[0]["reason"] == "not_applicable"


def test_completed_suboperation_can_be_retained_from_partial_stage():
    result = fuse_evidence(
        EvidenceCollection(items=[_vote()]),
        stage_results={"authenticity": _stage("partial", "partial_results")},
        analysis_id="analysis-1",
    )
    assert result.label == "likely_authentic"
    assert result.stage_execution_summary["authenticity"]["status"] == "partial"
    assert result.missing_evidence[0]["reason"].endswith("not negative evidence.")


def test_malformed_evidence_is_excluded_without_crashing():
    item = EvidenceItem(
        type="forensic_evidence", source="forensics.ela", strength="low",
        direction="informational", description="Malformed data fixture.", data=None,
    )
    result = fuse_evidence(EvidenceCollection(items=[item]))
    assert result.label == "inconclusive"
    assert result.evidence_excluded[0]["reason"] == "invalid_data_schema"


def test_duplicate_evidence_identity_is_counted_once():
    first = _vote()
    duplicate = _vote()
    duplicate.evidence_id = first.evidence_id
    result = fuse_evidence(EvidenceCollection(items=[first, duplicate]))
    single = fuse_evidence(EvidenceCollection(items=[first]))
    assert result.overall_score == single.overall_score
    assert result.confidence == single.confidence
    assert result.evidence_excluded[0]["reason"] == "duplicate_evidence"


def test_contradictory_valid_findings_are_explicit_and_inconclusive():
    authentic = _vote("authentic")
    synthetic = _vote("synthetic")
    synthetic.description = "conflicting validated output"
    result = fuse_evidence(EvidenceCollection(items=[authentic, synthetic]))
    assert result.label == "inconclusive"
    assert result.conflicting
    assert len(result.conflicting_findings) == 2


def test_directional_factcheck_rating_remains_attributed_information():
    from backend.app.services.evidence.schema import build_evidence_from_factcheck
    from backend.app.services.factcheck.service import FactCheckClaim, FactCheckResult

    claim = FactCheckClaim(claimed_text="A reviewed claim", publisher_name="Example Publisher", textual_rating="False")
    item = build_evidence_from_factcheck(FactCheckResult(status="success", claims=[claim]))[0]
    result = fuse_evidence(EvidenceCollection(items=[item]))
    assert result.label == "inconclusive"
    assert result.informational_findings[0]["source"] == "factcheck.Example Publisher"
    assert result.informational_findings[0]["direction"] == "informational"


def test_uncalibrated_classifier_output_is_visible_but_cannot_vote():
    item = _vote()
    item.calibrated = False
    item.data["calibration_status"] = "not_calibrated"
    result = fuse_evidence(EvidenceCollection(items=[item]))
    assert result.label == "inconclusive"
    assert result.informational_findings[0]["reason"] == "uncalibrated_model_output"
    assert result.to_dict()["assessment_confidence"] is None


def test_c2pa_signature_findings_and_signer_trust_remain_informational():
    from backend.app.services.evidence.schema import build_evidence_from_provenance
    from backend.app.services.provenance.service import ProvenanceResult

    items = build_evidence_from_provenance(ProvenanceResult(
        status="success",
        content_credentials={"status": "completed", "verification_outcome": "verified",
                             "signature_status": "valid", "content_binding_status": "valid",
                             "signer_trust": "unknown"},
        web_detection_status="no_results", web_detection_outcome="no_results",
    ))
    item = next(entry for entry in items if entry.source == "c2pa_content_credentials")
    result = fuse_evidence(EvidenceCollection(items=items))
    assert result.label == "inconclusive"
    assert result.informational_findings[0]["direction"] == "informational"
    assert item.data["signer_trust"] == "unknown"
    assert "signer trust" in item.description.lower()


def test_insufficient_evidence_contract_is_versioned_and_serializable():
    result = fuse_evidence(EvidenceCollection(), analysis_id="analysis-1", correlation_id="job-1")
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["schema_version"] == "truthlens.fusion.v1"
    assert payload["policy_version"] == "e11.conservative.v1"
    assert payload["category"] == "inconclusive"
    assert payload["assessment_confidence"] is None
    assert payload["correlation_id"] == "job-1"
    assert "probability" in payload["score_semantics"]


def test_stage_model_identity_and_analysis_id_are_verified():
    wrong_model_stage = _stage(model_name="different/model")
    result = fuse_evidence(
        EvidenceCollection(items=[_vote()]),
        stage_results={"authenticity": wrong_model_stage}, analysis_id="analysis-1",
    )
    assert result.label == "inconclusive"
    assert result.informational_findings[0]["reason"] == "stage_model_identity_mismatch"

    mismatched = _vote(analysis_id="other-analysis")
    result = fuse_evidence(EvidenceCollection(items=[mismatched]), analysis_id="analysis-1")
    assert result.evidence_excluded[0]["reason"] == "analysis_id_mismatch"


def test_older_report_and_new_fusion_fields_coexist():
    from backend.app.services.reporting.builder import build_report

    legacy = {"label": "inconclusive", "overall_score": None, "confidence": None}
    report = build_report(
        analysis_id="legacy-report", status="partial", assessment=legacy,
        stages={"fusion": {"status": "completed"}},
    )
    assert report["assessment"] == legacy
    assert report["fusion"]["label"] == "inconclusive"
    assert report["fusion"]["status"] == "completed"

    current = fuse_evidence(EvidenceCollection(items=[_vote()])).to_dict()
    current_report = build_report(
        analysis_id="current-report", status="completed", assessment=current,
        stages={"fusion": {"status": "completed"}},
    )
    assert current_report["fusion"]["policy_version"] == "e11.conservative.v1"


def test_versioned_report_survives_report_json_persistence_roundtrip():
    from backend.app.services.reporting.builder import build_report

    report = build_report(
        analysis_id="persist-e11", status="partial",
        assessment=fuse_evidence(EvidenceCollection(items=[_vote()])).to_dict(),
        stages={"fusion": {"status": "completed"}, "provenance": {"status": "unavailable"}},
    )
    persisted_json = json.dumps(report)
    restored = json.loads(persisted_json)
    assert restored["status"] == "partial"
    assert restored["fusion"]["schema_version"] == "truthlens.fusion.v1"
    assert restored["fusion"]["status"] == "completed"
    assert restored["fusion"]["assessment_confidence"] is None


def test_failed_fusion_report_has_safe_versioned_terminal_contract():
    from backend.app.services.reporting.builder import build_report

    report = build_report(
        analysis_id="fusion-failed", status="partial", assessment=None,
        stages={"fusion": {"status": "failed", "outcome": "stage_execution_error"},
                "fact_check": {"status": "unavailable", "outcome": "credentials_missing"}},
    )
    payload = report["fusion"]
    assert payload["schema_version"] == "truthlens.fusion.v1"
    assert payload["category"] is None
    assert payload["status"] == "failed"
    assert payload["correlation_id"] == "fusion-failed"
    assert payload["missing_evidence"][0]["stage_name"] == "fusion"
    assert "stack" not in json.dumps(payload).lower()


def test_controlled_authenticity_to_fusion_report_integration():
    from backend.app.services.evidence.schema import build_evidence_from_authenticity
    from backend.app.services.reporting.builder import build_report

    detector_output = SimpleNamespace(
        label="likely_authentic", model_prediction="likely_authentic",
        ai_generated_score=0.2, authentic_score=0.8, raw_logits={},
        calibrated_ai_generated_probability=0.1,
        calibrated_candidate_label="likely_authentic",
        calibration_status="validated_temperature_scaling",
        domain_validation_status="validated_for_input_domain",
        empirical_accuracy=None, model_name="truthlens/siglip-cifake-frozen-head",
        model_version="test-revision",
    )
    collection = EvidenceCollection(analysis_id="analysis-1")
    for item in build_evidence_from_authenticity(detector_output):
        collection.add(item)
    stages = {"authenticity": _stage()}
    assessment = fuse_evidence(
        collection, stage_results=stages, analysis_id="analysis-1", correlation_id="job-1",
    )
    report = build_report(
        analysis_id="analysis-1", status="completed", authenticity=detector_output.__dict__,
        evidence=collection.to_dict(), assessment=assessment.to_dict(),
        stages={"authenticity": {"status": "completed"}, "fusion": {"status": "completed"}},
    )
    persisted = json.loads(json.dumps(report))
    assert persisted["fusion"]["category"] == "likely_authentic"
    assert persisted["fusion"]["correlation_id"] == "job-1"
    assert persisted["fusion"]["evidence_considered"][0]["disposition"] == "directional_vote"


def test_explanation_omits_failed_module_results_from_findings():
    from backend.app.services.explanation.generator import generate_explanation

    collection = EvidenceCollection(items=[EvidenceItem(
        type="forensic_evidence", source="forensics.ela", strength="high",
        direction="synthetic", description="This failed finding must not surface.",
        status="failed", stage_name="forensics",
    )])
    result = generate_explanation(collection, "inconclusive")
    assert "This failed finding must not surface." not in result.key_findings
    assert all("failed finding" not in section["content"] for section in result.sections)
