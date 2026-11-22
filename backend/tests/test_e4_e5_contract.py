"""Focused E4/E5 claim extraction and provider contract tests."""
from unittest.mock import patch

import httpx
import pytest


def _mock_stream_client(payload: bytes, *, status_error=None, headers=None, chunk_size=16384):
    class Response:
        def __init__(self):
            self.headers = {"content-length": str(len(payload)), **(headers or {})}
        def raise_for_status(self):
            if status_error:
                raise status_error
        def iter_bytes(self):
            for index in range(0, len(payload), chunk_size):
                yield payload[index:index + chunk_size]
        def __enter__(self): return self
        def __exit__(self, *args): return False
    response = Response()
    class Client:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def stream(self, *args, **kwargs): return response
    return Client


def test_normalization_preserves_abbreviation_safe_sentence_offsets():
    from backend.app.services.text.normalizer import normalize_text

    result = normalize_text("Dr. Rao confirmed 25 clinics opened in Delhi. A second report followed.")
    assert len(result.sentences) == 2
    first = result.sentence_spans[0]
    assert result.normalized_text[first["start"]:first["end"]] == first["text"]


def test_sentence_boundaries_cover_punctuation_initials_abbreviation_end_and_numbers():
    from backend.app.services.text.normalizer import normalize_text

    text = normalize_text("Dr. A. Rao reported 3.5 cases on 04/05/2025. Clinics reopened, etc. The review ended!")
    assert text.sentences == [
        "Dr. A. Rao reported 3.5 cases on 04/05/2025.",
        "Clinics reopened, etc.",
        "The review ended!",
    ]
    assert all(text.normalized_text[s["start"]:s["end"]] == s["text"] for s in text.sentence_spans)


def test_claim_deduplication_preserves_punctuation_numbers_and_wording():
    from backend.app.services.claims.extractor import extract_claims

    claims = extract_claims([
        "The city opened 25 clinics in 2025.",
        "  THE city opened 25 clinics in 2025.  ",
        "The city opened 25 clinics in 2025!",
        "The city opened 26 clinics in 2025.",
        "The city officially opened 25 clinics in 2025.",
    ])
    assert [claim.original_text for claim in claims.claims] == [
        "The city opened 25 clinics in 2025.",
        "The city opened 25 clinics in 2025!",
        "The city opened 26 clinics in 2025.",
        "The city officially opened 25 clinics in 2025.",
    ]


def test_malformed_sentence_spans_and_ocr_regions_are_ignored_safely():
    from backend.app.services.claims.extractor import extract_claims

    source = "The city opened 25 clinics in 2025. A second claim was reported."
    spans = [
        {"text": "The city opened 25 clinics in 2025.", "start": 0, "end": 35},
        {"text": "A second claim was reported.", "start": "bad", "end": 62},
        None,
    ]
    result = extract_claims(
        source,
        sentence_spans=spans,
        source_regions=[
            {"region_id": "valid", "text_start": 0, "text_end": 10},
            {"region_id": "bad-range", "text_start": -1, "text_end": 5},
            {"region_id": "", "text_start": 0, "text_end": 5},
        ],
    )
    assert len(result.claims) == 1
    assert result.claims[0].source_start == 0
    assert result.claims[0].source_regions[0]["region_id"] == "valid"
    assert result.warnings
    malformed_containers = extract_claims(source, sentence_spans=42, source_regions=42)
    assert malformed_containers.claims == []
    assert malformed_containers.warnings


def test_claim_language_provenance_is_copied_only_from_successful_supported_e3():
    from types import SimpleNamespace
    from backend.app.services.claims.extractor import extract_claims

    text = "The city opened 25 clinics in 2025."
    success = SimpleNamespace(
        status="completed", outcome="success", detected_language="en", language_name="English",
        model_name="papluca/xlm-roberta-base-language-detection",
        model_revision="immutable-revision", confidence=0.91,
    )
    successful = extract_claims(text, language_result=success).claims[0].to_dict()
    assert successful["language_code"] == "en"
    assert successful["language_name"] == "English"
    assert successful["language_source"] == "e3_language_identification"
    assert successful["language_model_id"] == success.model_name
    assert successful["language_model_revision"] == success.model_revision
    assert successful["language_confidence"] == 0.91
    assert successful["language_detection_outcome"] == "success"

    for outcome in ("unavailable", "failed", "insufficient_input", "undetermined"):
        failed = SimpleNamespace(
            status="unavailable" if outcome == "unavailable" else "completed",
            outcome=outcome, detected_language="unknown", model_revision=None, confidence=None,
        )
        data = extract_claims(text, language_result=failed).claims[0].to_dict()
        assert data["language"] == "unknown"
        assert data["language_code"] is None
        assert data["language_model_revision"] is None
        assert data["language_confidence"] is None
        assert data["language_detection_outcome"] == outcome


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

    request = FactCheckRequest(claims=["The city opened 25 clinics in 2025."], language_code=None)
    assert request.claims == ["The city opened 25 clinics in 2025."]


@pytest.mark.asyncio
async def test_standalone_endpoint_skips_ineligible_and_submits_only_eligible_claims(client):
    from backend.app.services.factcheck.service import FactCheckResult

    eligible = "The city opened 25 clinics in 2025."
    ineligible = "I think this policy is beautiful."
    with patch("backend.app.services.factcheck.service.search_fact_checks", return_value=FactCheckResult(status="no_results")) as search:
        response = await client.post("/api/v1/fact-check", json={"claims": [ineligible, eligible]})
    assert response.status_code == 200
    assert search.call_count == 1
    assert search.call_args.args[0] == [eligible]
    entries = response.json()["claim_verifications"]
    assert entries[0]["verification_status"] == "SKIPPED"
    assert entries[0]["eligibility_status"] == "INELIGIBLE"
    assert entries[0]["eligibility_reason"]
    assert entries[1]["eligibility_status"] == "ELIGIBLE"


@pytest.mark.asyncio
async def test_standalone_endpoint_skips_provider_when_all_claims_are_ineligible(client):
    with patch("backend.app.services.factcheck.service.search_fact_checks") as search:
        response = await client.post("/api/v1/fact-check", json={"claims": ["I think this is beautiful."]})
    assert response.status_code == 200
    assert response.json()["status"] == "skipped"
    assert response.json()["claim_verifications"][0]["verification_status"] == "SKIPPED"
    search.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {"claims": []},
    {"claims": ["tiny"]},
    {"claims": ["The city opened 25 clinics in 2025."] * 21},
    {"claims": ["The city opened 25 clinics in 2025."], "language_code": "xx"},
    {"claims": ["The city opened 25 clinics in 2025."], "language_code": "EN"},
])
async def test_standalone_endpoint_rejects_invalid_bounds_and_language_codes(client, payload):
    response = await client.post("/api/v1/fact-check", json=payload)
    assert response.status_code == 422


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
        response = await client.post("/api/v1/fact-check", json={"claims": ["The city opened 25 clinics in 2025."]})
    assert response.status_code == 200
    assert response.json()["status"] == "no_results"
    assert response.json()["provider"] == "google_fact_check_tools"


def test_review_url_sanitization_removes_secrets_and_tracking_but_keeps_navigation():
    from backend.app.services.factcheck.service import _sanitize_review_url

    value = "https://publisher.example/review?id=42&key=secret&access_token=x&utm_source=mail&gclid=abc&tab=ratings#frag"
    assert _sanitize_review_url(value) == "https://publisher.example/review?id=42&tab=ratings"
    assert _sanitize_review_url("http://publisher.example/review") is None
    assert _sanitize_review_url("https://user:pass@publisher.example/review") is None
    assert _sanitize_review_url("https://publisher.example:bad/review") is None
    assert _sanitize_review_url("https://publisher.example/review?" + "&".join(f"x{i}=1" for i in range(101))) is None


def test_provider_stream_rejects_oversized_body_before_json_parsing():
    import json
    from backend.app.services.factcheck.service import search_fact_checks

    payload = json.dumps({"claims": []}).encode() + b" " * 1100
    with patch("backend.app.services.factcheck.service.settings") as cfg:
        cfg.GOOGLE_API_KEY, cfg.FACTCHECK_API_KEY = "configured", ""
        cfg.FACTCHECK_MAX_RESPONSE_BYTES = 1024
        cfg.FACTCHECK_RETRY_ATTEMPTS = 0
        cfg.FACTCHECK_MAX_CLAIMS = 5
        cfg.FACTCHECK_MAX_TOTAL_SECONDS = 25
        with patch("backend.app.services.factcheck.service.httpx.Client", _mock_stream_client(payload)):
            result = search_fact_checks(["The city opened 25 clinics in 2025."])
    assert result.status == "failed"
    assert result.claim_verifications[0]["retrieval_error_category"] == "PROVIDER_RESPONSE_TOO_LARGE"
    assert "secret" not in str(result.to_dict())


def test_provider_invalid_json_is_failed_and_redacted():
    from backend.app.services.factcheck.service import search_fact_checks

    payload = b'{"claims": [broken'
    with patch("backend.app.services.factcheck.service.settings") as cfg:
        cfg.GOOGLE_API_KEY, cfg.FACTCHECK_API_KEY = "configured", ""
        cfg.FACTCHECK_RETRY_ATTEMPTS = 0
        cfg.FACTCHECK_MAX_CLAIMS = 5
        cfg.FACTCHECK_MAX_TOTAL_SECONDS = 25
        with patch("backend.app.services.factcheck.service.httpx.Client", _mock_stream_client(payload)):
            result = search_fact_checks(["The city opened 25 clinics in 2025."])
    assert result.status == "failed"
    assert result.claim_verifications[0]["retrieval_error_category"] == "PROVIDER_INVALID_RESPONSE"
    assert "Expecting" not in str(result.to_dict())


def test_malformed_nested_record_is_skipped_while_valid_review_is_retained():
    import json
    from backend.app.services.factcheck.service import _search_single_claim

    query = "The city opened 25 clinics in 2025."
    payload = {"claims": [
        {"text": query, "claimReview": [
            {"publisher": {"name": "Publisher", "site": "publisher.example"},
             "url": "https://publisher.example/review?tab=ratings&token=secret",
             "title": "Review", "textualRating": "False"},
            {"publisher": "malformed", "url": "https://bad.example/review"},
            "not-a-record",
        ]},
        None,
    ]}
    encoded = json.dumps(payload).encode()
    with patch("backend.app.services.factcheck.service.settings") as cfg:
        cfg.FACTCHECK_API_TIMEOUT_SECONDS = 3
        with patch("backend.app.services.factcheck.service.httpx.Client", _mock_stream_client(encoded)):
            result = _search_single_claim(query, "mock-key", claim_id="c1")
    assert len(result) == 1
    assert result.malformed_items == 3
    assert result[0].url == "https://publisher.example/review?tab=ratings"
    assert "secret" not in result[0].url


def test_mixed_valid_and_malformed_provider_records_return_partial():
    from backend.app.services.factcheck.service import FactCheckClaim, ParsedReviews, search_fact_checks

    claim = FactCheckClaim(query="The city opened 25 clinics in 2025.", claimed_text="The city opened 25 clinics in 2025.", url="https://publisher.example/review")
    parsed = ParsedReviews([claim], malformed_items=1)
    with patch("backend.app.services.factcheck.service.settings") as cfg:
        cfg.GOOGLE_API_KEY, cfg.FACTCHECK_API_KEY = "configured", ""
        cfg.FACTCHECK_RETRY_ATTEMPTS = 0
        cfg.FACTCHECK_MAX_CLAIMS = 5
        cfg.FACTCHECK_MAX_TOTAL_SECONDS = 25
        with patch("backend.app.services.factcheck.service._search_single_claim", return_value=parsed):
            result = search_fact_checks([claim.query])
    assert result.status == "partial"
    assert result.claims == [claim]
    assert any("malformed" in warning for warning in result.warnings)


def test_provider_retries_are_bounded_and_backoff_is_capped():
    from backend.app.services.factcheck.service import FactCheckClaim, search_fact_checks

    good = FactCheckClaim(query="The city opened 25 clinics in 2025.", claimed_text="The city opened 25 clinics in 2025.", url="https://publisher.example/review")
    with patch("backend.app.services.factcheck.service.settings") as cfg:
        cfg.GOOGLE_API_KEY, cfg.FACTCHECK_API_KEY = "configured", ""
        cfg.FACTCHECK_RETRY_ATTEMPTS = 2
        cfg.FACTCHECK_MAX_CLAIMS = 5
        cfg.FACTCHECK_MAX_TOTAL_SECONDS = 25
        cfg.FACTCHECK_API_TIMEOUT_SECONDS = 5
        with patch("backend.app.services.factcheck.service._search_single_claim", side_effect=[httpx.TimeoutException("hidden"), [good]]) as call, \
             patch("backend.app.services.factcheck.service.time.sleep") as sleep:
            result = search_fact_checks([good.query])
    assert result.status == "success"
    assert call.call_count == 2
    assert sleep.call_count == 1
    assert sleep.call_args.args[0] <= 1.0


def test_retry_exhaustion_is_failed_with_safe_error_category():
    from backend.app.services.factcheck.service import search_fact_checks

    with patch("backend.app.services.factcheck.service.settings") as cfg:
        cfg.GOOGLE_API_KEY, cfg.FACTCHECK_API_KEY = "configured", ""
        cfg.FACTCHECK_RETRY_ATTEMPTS = 2
        cfg.FACTCHECK_MAX_CLAIMS = 5
        cfg.FACTCHECK_MAX_TOTAL_SECONDS = 25
        cfg.FACTCHECK_API_TIMEOUT_SECONDS = 5
        with patch("backend.app.services.factcheck.service._search_single_claim", side_effect=httpx.TimeoutException("credential=secret")) as call, \
             patch("backend.app.services.factcheck.service.time.sleep"):
            result = search_fact_checks(["The city opened 25 clinics in 2025."])
    assert result.status == "failed"
    assert call.call_count == 3
    assert result.claim_verifications[0]["retrieval_error_category"] == "PROVIDER_TIMEOUT"
    assert "secret" not in str(result.to_dict())
