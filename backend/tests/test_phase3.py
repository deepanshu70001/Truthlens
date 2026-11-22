"""Offline coverage for Phase 3 claims, retrieval, verdicts and provenance."""
import io
import json
from unittest.mock import patch

from PIL import Image


def test_claim_extraction_classifies_and_preserves_text():
    from backend.app.services.claims.extractor import extract_claims

    result = extract_claims([
        "Government announced 120 new clinics in Delhi in 2025.",
        "I think this is the best policy.",
        "Share this now!",
        "Did the minister resign?",
        "Government announced 120 new clinics in Delhi in 2025.",
        "खबर सरकार ने 2025 में नई नीति घोषित की।",
    ], language="hi", ocr_confidence=0.42)
    assert result.status == "completed"
    assert len(result.claims) == 5
    factual = result.claims[0]
    assert factual.classification == "FACTUAL"
    assert factual.verification_eligibility == "ELIGIBLE"
    assert factual.original_text.startswith("Government announced")
    assert factual.numbers == ["120", "2025"]
    assert factual.language == "hi" and factual.translated_text is None
    assert factual.ocr_confidence == 0.42
    assert all(c.claim_id for c in result.claims)
    assert [c.classification for c in result.claims[1:4]] == ["OPINION", "SLOGAN", "QUESTION"]
    assert all(c.verification_eligibility == "INELIGIBLE" for c in result.claims[1:4])


def test_empty_and_ambiguous_ocr_are_successful_and_not_searchable():
    from backend.app.services.claims.extractor import extract_claims

    empty = extract_claims([])
    ambiguous = extract_claims(["Breaking news", "Something happened."])
    assert empty.status == "completed" and empty.claims == []
    assert ambiguous.claims[0].verification_eligibility == "INELIGIBLE"
    assert ambiguous.claims[0].classification in {"SLOGAN", "AMBIGUOUS"}


def test_attributed_quote_can_be_checked_but_unattributed_quote_cannot():
    from backend.app.services.claims.extractor import extract_claims

    result = extract_claims(['The minister said, "The bridge opened in 2025."', '"The bridge opened in 2025."'])
    assert result.claims[0].classification == "QUOTE"
    assert result.claims[0].verification_eligibility == "ELIGIBLE"
    assert result.claims[1].classification == "QUOTE"
    assert result.claims[1].verification_eligibility == "INELIGIBLE"


def test_factcheck_verdicts_link_to_real_review_ids_and_reject_irrelevant_hits():
    from backend.app.services.factcheck.service import FactCheckClaim, search_fact_checks, _assess_ratings

    query = "Government announced 120 new clinics in Delhi in 2025"
    review = FactCheckClaim(query=query, claimed_text=query, publisher_name="Example Review",
        url="https://example.org/review", title="Review", textual_rating="False")
    with patch("backend.app.services.factcheck.service.settings") as cfg:
        cfg.GOOGLE_API_KEY, cfg.FACTCHECK_API_KEY = "configured", ""
        with patch("backend.app.services.factcheck.service._search_single_claim", return_value=[review]):
            result = search_fact_checks([query], claim_ids=["claim-test"])
    data = result.to_dict()
    verdict = data["claim_verifications"][0]
    assert result.status == "success"
    assert verdict["verdict"] == "CONTRADICTED"
    assert verdict["contradicting_evidence_ids"] == [review.evidence_id]
    assert data["claims"][0]["claim_id"] == "claim-test"
    assert data["claims"][0]["claim_match"] == "strong_match"

    unrelated = FactCheckClaim(query=query, claimed_text="Different event in another country", url="https://example.org/other", textual_rating="False")
    assert unrelated.evidence_id
    verdict, *_ = _assess_ratings(query, [unrelated])
    assert verdict == "INSUFFICIENT_EVIDENCE"


def test_factcheck_unavailable_and_successful_empty_are_different():
    from backend.app.services.factcheck.service import search_fact_checks

    with patch("backend.app.services.factcheck.service.settings") as cfg:
        cfg.GOOGLE_API_KEY, cfg.FACTCHECK_API_KEY = "", ""
        unavailable = search_fact_checks(["A factual claim"], claim_ids=["c1"])
    assert unavailable.status == "unavailable"
    assert unavailable.claim_verifications[0]["verdict"] == "RETRIEVAL_UNAVAILABLE"

    with patch("backend.app.services.factcheck.service.settings") as cfg:
        cfg.GOOGLE_API_KEY, cfg.FACTCHECK_API_KEY = "configured", ""
        with patch("backend.app.services.factcheck.service._search_single_claim", return_value=[]):
            empty = search_fact_checks(["A factual claim"], claim_ids=["c1"])
    assert empty.status == "no_results"
    assert empty.claim_verifications[0]["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_factcheck_provider_normalizes_only_safe_real_review_links():
    from backend.app.services.factcheck.service import _search_single_claim

    query = "A public factual statement with 27 clinics in Delhi"
    payload = {"claims": [{"text": query, "claimReview": [
        {"publisher": {"name": "Publisher", "site": "publisher.example"},
         "url": "https://publisher.example/review", "title": "Review title",
         "reviewDate": "2026-05-01", "textualRating": "False", "languageCode": "en"},
        {"publisher": {"name": "Unsafe"}, "url": "http://localhost/private", "textualRating": "True"},
    ]}]}
    mock_response = type("Response", (), {"raise_for_status": lambda self: None, "json": lambda self: payload})()
    mock_client = type("Client", (), {
        "__enter__": lambda self: self, "__exit__": lambda *args: False,
        "get": lambda *args, **kwargs: mock_response,
    })()
    with patch("backend.app.services.factcheck.service.settings") as cfg:
        cfg.FACTCHECK_API_TIMEOUT_SECONDS = 3
        with patch("backend.app.services.factcheck.service.httpx.Client", return_value=mock_client):
            reviews = _search_single_claim(query, "mock-secret", claim_id="claim-safe")
    assert len(reviews) == 1
    assert reviews[0].url == "https://publisher.example/review"
    assert reviews[0].claim_id == "claim-safe"
    assert reviews[0].evidence_id and reviews[0].retrieved_at
    assert reviews[0].claim_match == "strong_match"


def test_factcheck_request_timeout_is_unavailable_not_a_false_verdict():
    import httpx
    from backend.app.services.factcheck.service import search_fact_checks

    with patch("backend.app.services.factcheck.service.settings") as cfg:
        cfg.GOOGLE_API_KEY, cfg.FACTCHECK_API_KEY = "configured", ""
        with patch("backend.app.services.factcheck.service._search_single_claim", side_effect=httpx.TimeoutException("private detail")):
            result = search_fact_checks(["A factual claim about Delhi"], claim_ids=["claim-timeout"])
    assert result.status == "failed"
    assert result.claim_verifications[0]["verdict"] == "RETRIEVAL_UNAVAILABLE"
    assert "private detail" not in json.dumps(result.to_dict())


def test_factcheck_claim_cap_and_short_text_are_reported_without_false_results():
    from backend.app.services.factcheck.service import search_fact_checks

    with patch("backend.app.services.factcheck.service.settings") as cfg:
        cfg.GOOGLE_API_KEY, cfg.FACTCHECK_API_KEY = "configured", ""
        cfg.FACTCHECK_MAX_CLAIMS = 1
        with patch("backend.app.services.factcheck.service._search_single_claim", return_value=[]) as search:
            result = search_fact_checks(["A long factual assertion", "Another factual assertion"], claim_ids=["a", "b"])
    assert result.status == "partial"
    assert search.call_count == 1
    assert result.claim_verifications[1]["verdict"] == "INSUFFICIENT_EVIDENCE"

    with patch("backend.app.services.factcheck.service.settings") as cfg:
        cfg.GOOGLE_API_KEY, cfg.FACTCHECK_API_KEY = "configured", ""
        skipped = search_fact_checks(["No"], claim_ids=["short"])
    assert skipped.claim_verifications[0]["verdict"] == "UNVERIFIABLE"


def test_verdict_abstains_on_conflicting_review_ratings():
    from backend.app.services.factcheck.service import FactCheckClaim, _assess_ratings

    claim = "The city opened 25 schools in Delhi in 2025"
    reviews = [
        FactCheckClaim(claimed_text=claim, textual_rating="True", url="https://a.example/review"),
        FactCheckClaim(claimed_text=claim, textual_rating="False", url="https://b.example/review"),
    ]
    verdict, supporting, contradicting, conflicting, _ = _assess_ratings(claim, reviews)
    assert verdict == "MIXED"
    assert supporting and contradicting and set(conflicting) == {r.evidence_id for r in reviews}


def test_evidence_ids_are_preserved_and_factcheck_stays_informational():
    from backend.app.services.evidence.schema import build_evidence_from_factcheck
    from backend.app.services.factcheck.service import FactCheckClaim, FactCheckResult

    review = FactCheckClaim(claimed_text="A claim", publisher_name="Publisher", url="https://example.org/r", textual_rating="False")
    items = build_evidence_from_factcheck(FactCheckResult(status="success", claims=[review]))
    assert len(items) == 1
    assert items[0].evidence_id == review.evidence_id
    assert items[0].direction == "informational"


def test_provenance_response_marks_parser_unavailable_and_filters_unsafe_urls():
    from backend.app.services.provenance.service import search_provenance

    response_payload = {"responses": [{"webDetection": {
        "pagesWithMatchingImages": [
            {"url": "https://publisher.example/story", "pageTitle": "A story"},
            {"url": "http://localhost/private", "pageTitle": "Unsafe"},
        ],
        "visuallySimilarImages": [{"url": "javascript:alert(1)"}],
    }}]}
    mock_response = type("Response", (), {"raise_for_status": lambda self: None, "json": lambda self: response_payload})()
    mock_client = type("Client", (), {
        "__enter__": lambda self: self, "__exit__": lambda *args: False,
        "post": lambda *args, **kwargs: mock_response,
    })()
    with patch("backend.app.services.provenance.service.settings") as cfg, patch(
        "backend.app.services.provenance.service.verify_content_credentials"
    ) as mock_c2pa:
        cfg.GOOGLE_API_KEY = "configured"
        cfg.PROVENANCE_API_TIMEOUT_SECONDS = 1
        cfg.max_upload_bytes = 5 * 1024 * 1024
        from backend.app.services.provenance.c2pa_verifier import C2PAResult
        mock_c2pa.return_value = C2PAResult(status="unavailable", outcome="unavailable")
        with patch("backend.app.services.provenance.service.httpx.Client", return_value=mock_client):
            result = search_provenance(b"image bytes")
    assert result.status == "partial"
    assert result.content_credentials["status"] == "unavailable"
    assert result.web_detection_status == "success"
    data = result.to_dict()
    assert len(data["matching_pages"]) == 1
    assert data["matching_pages"][0]["publisher"] == "publisher.example"
    assert data["retrieved_at"]
    assert data["content_credentials"]["status"] == "unavailable"


def test_end_to_end_claim_report_keeps_caption_verdict_separate_from_image_score():
    """Exercise normalization → extraction → mock provider → evidence → report."""
    from backend.app.services.text.normalizer import normalize_text
    from backend.app.services.claims.extractor import extract_claims
    from backend.app.services.factcheck.service import FactCheckClaim, search_fact_checks
    from backend.app.services.evidence.schema import EvidenceCollection, build_evidence_from_factcheck
    from backend.app.services.reporting.builder import build_report

    text = normalize_text("Government announced 120 new clinics in Delhi in 2025.")
    claims = extract_claims(text.sentences, language="en", ocr_confidence=0.9)
    claim = claims.claims[0]
    review = FactCheckClaim(query=claim.normalized_text, claimed_text=claim.normalized_text,
        publisher_name="Mocked Fact Check", url="https://example.org/check", title="Check", textual_rating="False")
    with patch("backend.app.services.factcheck.service.settings") as cfg:
        cfg.GOOGLE_API_KEY, cfg.FACTCHECK_API_KEY = "configured", ""
        with patch("backend.app.services.factcheck.service._search_single_claim", return_value=[review]):
            checks = search_fact_checks([claim.normalized_text], claim_ids=[claim.claim_id])
    evidence = EvidenceCollection(analysis_id="analysis-test")
    for item in build_evidence_from_factcheck(checks):
        evidence.add(item)
    report = build_report("analysis-test", "completed", claims=claims.to_dict(),
        fact_checks=checks.to_dict(), evidence=evidence.to_dict(),
        authenticity={"status": "completed", "ai_generated_score": 0.1},
        assessment={"label": "likely_authentic", "overall_score": 0.9})
    serialized = json.loads(json.dumps(report))
    assert serialized["claims"]["claims"][0]["claim_id"] == claim.claim_id
    assert serialized["fact_checks"]["claim_verifications"][0]["verdict"] == "CONTRADICTED"
    assert serialized["assessment"]["overall_score"] == 0.9
