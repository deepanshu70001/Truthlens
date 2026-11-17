"""Safety contract tests for Phase 4 outputs."""
from backend.app.services.regional.engine import estimate_regional_spread
from backend.app.services.social_impact.engine import assess_social_impact
from backend.app.services.virality.engine import estimate_virality


def test_missing_virality_modalities_are_unknown_not_negative():
    result = estimate_virality()
    assert result.virality_score is None
    assert result.risk_level == "unknown"
    assert all(feature.value is None for feature in result.features)
    assert result.to_dict()["historical_prediction_status"] == "DATA_UNAVAILABLE"


def test_virality_score_uses_only_observed_modalities():
    result = estimate_virality(vision_result={"object_count": 4})
    assert result.virality_score is not None
    assert all(feature.available for feature in result.features if feature.name in {"has_faces", "multiple_objects"})


def test_regional_contract_has_all_36_units_without_scores():
    result = estimate_regional_spread(content_language="ta")
    payload = result.to_dict()
    assert len(payload["regions"]) == 36
    assert len({region["region"] for region in payload["regions"]}) == 36
    assert payload["top_risk_regions"] == []
    assert all(region["risk_score"] is None and region["confidence"] is None for region in payload["regions"])


def test_social_category_does_not_claim_harm_or_falsity():
    result = assess_social_impact(ocr_text="Vaccine cure scam alert")
    payload = result.to_dict()
    assert payload["overall_risk"] is None
    assert payload["observed_public_impact"] == "UNAVAILABLE"
    assert payload["misinformation_severity"] == "unassessed"
    health = next(category for category in payload["categories"] if category["category"] == "public_health")
    assert health["indicators_found"] > 0
    assert health["risk_score"] is None


def test_image_authenticity_does_not_set_misinformation_severity():
    result = assess_social_impact(authenticity_label="likely_ai_generated", factcheck_rating="False")
    assert result.misinformation_severity == "unassessed"


def test_phase3_verification_is_linked_to_matching_claim():
    result = assess_social_impact(
        ocr_text="The vaccine causes severe disease",
        claims=[{"claim_id": "c-1", "text": "The vaccine causes disease"}],
        claim_verifications={"c-1": {"status": "UNVERIFIABLE", "evidence": [{"evidence_id": "ev-1"}]}},
    )
    health = next(c for c in result.to_dict()["categories"] if c["category"] == "public_health")
    assert health["supporting_claim_ids"] == ["c-1"]
    assert health["verification_statuses"] == ["UNVERIFIABLE"]
    assert health["evidence_ids"] == ["ev-1"]


def test_phase3_factcheck_list_shape_is_supported():
    result = assess_social_impact(
        ocr_text="A vaccine causes disease",
        claims=[{"claim_id": "c-2", "text": "A vaccine causes disease"}],
        claim_verifications=[{"claim_id": "c-2", "verification_status": "MIXED", "conflicting_evidence_ids": ["fc-x"]}],
    )
    health = next(c for c in result.to_dict()["categories"] if c["category"] == "public_health")
    assert health["verification_statuses"] == ["MIXED"]
    assert health["evidence_ids"] == ["fc-x"]


def test_social_keyword_matching_respects_token_boundaries():
    result = assess_social_impact(ocr_text="The banker discussed a scamming tactic")
    finance = next(c for c in result.categories if c.category == "financial_safety")
    assert finance.indicators_found == 0
