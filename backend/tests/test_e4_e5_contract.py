"""Focused E4/E5 claim extraction and provider contract tests."""
from unittest.mock import patch

import httpx
import pytest


def test_normalization_preserves_abbreviation_safe_sentence_offsets():
    from backend.app.services.text.normalizer import normalize_text

    result = normalize_text("Dr. Rao confirmed 25 clinics opened in Delhi. A second report followed.")
    assert len(result.sentences) == 2
    first = result.sentence_spans[0]
    assert result.normalized_text[first["start"]:first["end"]] == first["text"]


def test_claims_retain_offsets_and_explicit_language_confidence():
    from backend.app.services.claims.extractor import extract_claims
    from backend.app.services.text.normalizer import normalize_text

    text = normalize_text("Delhi is a city. Clinics opened in 2025.")
    claims = extract_claims(
        text.normalized_text, sentence_spans=text.sentence_spans,
        language="unknown", language_detection_status="undetermined",
        source_regions=[{"region_id": "r1", "text_start": 0, "text_end": 16, "bbox": [[1, 2], [3, 4]]}],
    )
    assert claims.claims
    item = claims.claims[0].to_dict()
    assert item["source_start"] == 0
    assert item["source_end"] == len("Delhi is a city.")
    assert item["source_regions"][0]["bbox"] == [[1, 2], [3, 4]]
    assert item["language_detection_status"] == "undetermined"
    assert item["extraction_method"] == "conservative_rules_v2"
    assert item["confidence_semantics"].endswith("not truth probability")


def test_factcheck_omits_language_without_confident_language_and_deduplicates():
    from backend.app.services.factcheck.service import search_fact_checks

    with patch("backend.app.services.factcheck.service.settings") as cfg:
        cfg.GOOGLE_API_KEY, cfg.FACTCHECK_API_KEY = "configured", ""
        with patch("backend.app.services.factcheck.service._search_single_claim", return_value=[]) as search:
            result = search_fact_checks(["Delhi opened 25 clinics in 2025", "  delhi OPENED 25 clinics in 2025  "])
    assert result.status == "no_results"
    assert search.call_count == 1
    assert search.call_args.kwargs["language_code"] is None


def test_factcheck_retains_success_when_later_provider_query_fails():
    from backend.app.services.factcheck.service import FactCheckClaim, search_fact_checks

    good = FactCheckClaim(query="Delhi opened 25 clinics in 2025", claimed_text="Delhi opened 25 clinics in 2025", url="https://example.org/review")
    def provider(query, **kwargs):
        if query.startswith("Delhi"):
            return [good]
        raise httpx.TimeoutException("private")
    with patch("backend.app.services.factcheck.service.settings") as cfg:
        cfg.GOOGLE_API_KEY, cfg.FACTCHECK_API_KEY = "configured", ""
        cfg.FACTCHECK_MAX_CLAIMS = 5
        with patch("backend.app.services.factcheck.service._search_single_claim", side_effect=provider):
            result = search_fact_checks(["Delhi opened 25 clinics in 2025", "Mumbai opened 12 clinics in 2024"], claim_ids=["c1", "c2"])
    assert result.status == "partial"
    assert len(result.claims) == 1
    assert result.claim_verifications[1]["retrieval_error_category"] == "PROVIDER_TIMEOUT"


def test_factcheck_stage_endpoint_accepts_bounded_requests():
    from backend.app.api.v1.endpoints.stubs import FactCheckRequest

    request = FactCheckRequest(claims=["A factual statement"], language_code=None)
    assert request.claims == ["A factual statement"]


def test_failed_factcheck_evidence_is_informational_and_ignored_by_fusion():
    from backend.app.services.evidence.schema import EvidenceCollection, build_evidence_from_factcheck
    from backend.app.services.factcheck.service import FactCheckResult
    from backend.app.services.fusion.engine import fuse_evidence

    collection = EvidenceCollection(analysis_id="analysis-failed-provider")
    for item in build_evidence_from_factcheck(FactCheckResult(status="failed", error="safe failure")):
        collection.add(item)
    assert collection.items[0].direction == "informational"
    assessment = fuse_evidence(collection)
    assert assessment.label == "inconclusive"


@pytest.mark.asyncio
async def test_factcheck_api_route_returns_normalized_service_payload(client):
    from backend.app.services.factcheck.service import FactCheckResult

    with patch("backend.app.services.factcheck.service.search_fact_checks", return_value=FactCheckResult(status="no_results")):
        response = await client.post("/api/v1/fact-check", json={"claims": ["A verifiable public statement"]})
    assert response.status_code == 200
    assert response.json()["status"] == "no_results"
    assert response.json()["provider"] == "google_fact_check_tools"
