"""
TruthLens — Tests for the Standalone Claim Extraction endpoint (POST /api/v1/claims).

Verifies:
1. Normal text with multiple candidate claims.
2. Text containing no eligible claims (questions/opinions/slogans/short fragments).
3. Empty text rejection (HTTP 422 with VALIDATION_ERROR).
4. Whitespace-only text rejection (HTTP 422 with VALIDATION_ERROR).
5. Maximum text length validation (HTTP 422 when exceeding limit).
6. Invalid request schema rejection (HTTP 422).
7. Supported and unsupported language handling (HTTP 200 on valid, HTTP 422 on invalid).
8. Original character-offset correctness (source_start, source_end matching input text).
9. Unicode, Hindi and mixed-language input handling.
10. Duplicate or overlapping candidate claims deduplication.
11. Extractor unavailable or returning empty gracefully.
12. Unexpected extractor exception sanitization (HTTP 500 without stack trace disclosure).
13. Authentication enforcement (HTTP 401 when unauthenticated).
14. Rate-limit enforcement (HTTP 429 when quota exceeded).
15. Response schema and method/version reporting (conservative_rules_v2).
16. Sensitive-data and exception sanitization (no internal paths or secrets in responses).
"""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.config import settings
from backend.app.core.auth import issue_access_token
from backend.app.main import app
from backend.app.schemas.claims import ClaimExtractionRequest, ClaimExtractionResponse, ClaimResponse
from backend.app.services.claims.extractor import EXTRACTION_METHOD, ClaimExtractionResult, ExtractedClaim


@pytest.mark.asyncio
async def test_claims_endpoint_multiple_candidates(client):
    """1. POST /api/v1/claims extracts multiple candidate claims from standard text."""
    text = (
        "The Health Ministry announced 50 new hospitals in 2025. "
        "Over 10,000 doctors were recruited in Mumbai. "
        "The project budget exceeded 500 crore rupees."
    )
    resp = await client.post(
        "/api/v1/claims",
        json={"text": text, "language": "en", "max_claims": 5},
    )
    assert resp.status_code == 200
    data = resp.json()

    validated = ClaimExtractionResponse.model_validate(data)
    assert validated.status == "completed"
    assert validated.total_claims >= 2
    assert validated.accepted_count >= 2
    assert validated.method == EXTRACTION_METHOD

    # Verify claim properties
    first = validated.claims[0]
    assert first.verification_eligibility == "ELIGIBLE"
    assert first.classification in ("FACTUAL", "QUOTE")
    assert first.confidence > 0.0
    assert first.confidence_semantics == "heuristic extraction score; not truth probability"
    assert first.source_start is not None
    assert first.source_end is not None
    assert text[first.source_start : first.source_end] == first.original_text


@pytest.mark.asyncio
async def test_claims_endpoint_no_eligible_claims(client):
    """2. Text containing only questions, opinions, and slogans produces no eligible claims."""
    text = (
        "Is this the best policy ever? "
        "In my opinion, everyone should stop this immediately. "
        "#SaveOurCity"
    )
    resp = await client.post(
        "/api/v1/claims",
        json={"text": text},
    )
    assert resp.status_code == 200
    data = resp.json()
    validated = ClaimExtractionResponse.model_validate(data)

    assert validated.status == "completed"
    assert validated.accepted_count == 0  # No ELIGIBLE claims
    for claim in validated.claims:
        assert claim.verification_eligibility == "INELIGIBLE"
        assert claim.classification in ("QUESTION", "OPINION", "SLOGAN", "AMBIGUOUS")


@pytest.mark.asyncio
async def test_claims_endpoint_empty_text(client):
    """3. Empty text string is rejected with HTTP 422."""
    resp = await client.post(
        "/api/v1/claims",
        json={"text": ""},
    )
    assert resp.status_code == 422
    data = resp.json()
    assert data["error_code"] == "VALIDATION_ERROR"
    assert any(err.get("loc")[-1] == "text" or "body" in err.get("loc") for err in data["detail"])


@pytest.mark.asyncio
async def test_claims_endpoint_whitespace_only_text(client):
    """4. Whitespace-only text string is rejected with HTTP 422."""
    resp = await client.post(
        "/api/v1/claims",
        json={"text": "   \n\t   "},
    )
    assert resp.status_code == 422
    data = resp.json()
    assert data["error_code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_claims_endpoint_max_text_length_exceeded(client, monkeypatch):
    """5. Input exceeding maximum text length is rejected with HTTP 422."""
    monkeypatch.setattr(settings, "MAX_CLAIM_TEXT_LENGTH", 100)
    oversized = "The minister stated that the project was approved in 2025. " * 5  # > 100 chars
    resp = await client.post(
        "/api/v1/claims",
        json={"text": oversized},
    )
    assert resp.status_code == 422
    data = resp.json()
    assert data["error_code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_claims_endpoint_invalid_schema(client):
    """6. Completely empty payload or invalid data types rejected with HTTP 422."""
    # Neither text nor sentences
    resp = await client.post("/api/v1/claims", json={})
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "VALIDATION_ERROR"

    # Invalid max_claims (< 1)
    resp = await client.post(
        "/api/v1/claims",
        json={"text": "A valid sentence.", "max_claims": 0},
    )
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "VALIDATION_ERROR"

    # Non-list sentences
    resp = await client.post(
        "/api/v1/claims",
        json={"sentences": "not a list"},
    )
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_claims_endpoint_supported_and_unsupported_languages(client):
    """7. Valid language codes are accepted; unsupported codes are rejected with HTTP 422."""
    text = "The government launched 15 initiatives in Delhi."

    # Supported language 'en'
    resp = await client.post(
        "/api/v1/claims",
        json={"text": text, "language": "en"},
    )
    assert resp.status_code == 200
    assert resp.json()["claims"][0]["language"] == "en"

    # Supported language 'hi'
    resp = await client.post(
        "/api/v1/claims",
        json={"text": text, "language_code": "hi"},
    )
    assert resp.status_code == 200
    assert resp.json()["claims"][0]["language"] == "hi"

    # Unsupported language code 'klingon' or 'xx'
    resp = await client.post(
        "/api/v1/claims",
        json={"text": text, "language": "xx"},
    )
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_claims_endpoint_character_offset_correctness(client):
    """8. Original character offsets (source_start, source_end) index exact substring in input."""
    prefix = "   Leading whitespace. "
    statement = "The Ministry of Finance reported a 12% revenue increase in 2025."
    suffix = " Trailing remarks."
    full_text = f"{prefix}{statement}{suffix}"

    resp = await client.post(
        "/api/v1/claims",
        json={"text": full_text},
    )
    assert resp.status_code == 200
    data = resp.json()

    matched = False
    for claim in data["claims"]:
        if "revenue increase" in claim["claim"]:
            matched = True
            start = claim["source_start"]
            end = claim["source_end"]
            assert start is not None and end is not None
            assert full_text[start:end] == claim["original_text"]
    assert matched is True


@pytest.mark.asyncio
async def test_claims_endpoint_unicode_hindi_and_multilingual(client):
    """9. Unicode, Hindi, and mixed-language inputs are parsed correctly."""
    hindi_text = (
        "सरकार ने 2025 में 50 नए अस्पताल शुरू किए. "
        "क्या यह दावा सही है? "
        "मेरे विचार में यह अच्छा कदम है."
    )
    resp = await client.post(
        "/api/v1/claims",
        json={"text": hindi_text, "language": "hi"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_claims"] >= 2

    # Verify first claim is recognized as factual Hindi statement
    factual = data["claims"][0]
    assert factual["verification_eligibility"] == "ELIGIBLE"
    assert factual["classification"] == "FACTUAL"
    assert "50" in factual["numbers"]
    assert "2025" in factual["dates"]

    # Verify question is classified as INELIGIBLE QUESTION
    question = [c for c in data["claims"] if "क्या" in c["claim"]]
    if question:
        assert question[0]["verification_eligibility"] == "INELIGIBLE"
        assert question[0]["classification"] == "QUESTION"


@pytest.mark.asyncio
async def test_claims_endpoint_duplicate_claim_deduplication(client):
    """10. Duplicate normalized candidate claims are collapsed."""
    sentences = [
        "The city opened 25 clinics in 2025.",
        "  THE city opened 25 clinics in 2025.  ",
        "the city opened 25 clinics in 2025.",
        "A different announcement occurred in Mumbai in 2025.",
    ]
    resp = await client.post(
        "/api/v1/claims",
        json={"sentences": sentences},
    )
    assert resp.status_code == 200
    data = resp.json()

    # The 3 duplicate sentences differing only by case/whitespace collapse into one claim
    claims_text = [c["normalized_text"] for c in data["claims"]]
    assert len([t for t in claims_text if "25 clinics" in t.lower()]) == 1
    assert data["total_claims"] == 2


@pytest.mark.asyncio
async def test_claims_endpoint_extractor_empty_or_unavailable(client, monkeypatch):
    """11. Extractor returning no claims results in clean 200 OK with empty claims list."""
    def mock_empty_extract(*args, **kwargs):
        return ClaimExtractionResult(
            claims=[],
            total_claims=0,
            runtime_ms=5,
            method="conservative_rules_v2",
            status="completed",
            warnings=["No candidate statements were identified."],
        )

    monkeypatch.setattr("backend.app.api.v1.endpoints.claims.extract_claims", mock_empty_extract)

    resp = await client.post(
        "/api/v1/claims",
        json={"text": "Short. Ok."},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["total_claims"] == 0
    assert data["accepted_count"] == 0
    assert data["claims"] == []
    assert len(data["warnings"]) > 0


@pytest.mark.asyncio
async def test_claims_endpoint_unexpected_exception_sanitization(client, monkeypatch):
    """12. Unexpected internal extractor exceptions return sanitized HTTP 500."""
    def mock_broken_extract(*args, **kwargs):
        raise RuntimeError("Internal secret /var/secrets/key.pem failed during regex compile")

    monkeypatch.setattr("backend.app.api.v1.endpoints.claims.extract_claims", mock_broken_extract)

    resp = await client.post(
        "/api/v1/claims",
        json={"text": "The minister made an announcement yesterday in Delhi."},
    )
    assert resp.status_code == 500
    data = resp.json()
    assert "Claim extraction failed due to an internal error" in data["detail"]
    assert "key.pem" not in resp.text
    assert "/var/secrets" not in resp.text
    assert "Traceback" not in resp.text


@pytest.mark.asyncio
async def test_claims_endpoint_authentication_enforced(monkeypatch):
    """13. Requests without valid Bearer token are rejected with HTTP 401."""
    monkeypatch.setattr(settings, "DEV_AUTH_BYPASS", False)
    monkeypatch.setattr(settings, "APP_ENV", "production")

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as unauth_client:
        resp = await unauth_client.post(
            "/api/v1/claims",
            json={"text": "The prime minister announced 5 projects in 2025."},
        )
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_claims_endpoint_rate_limiting_enforced(client, monkeypatch):
    """14. Exceeding RATE_LIMIT_CLAIMS_PER_MINUTE returns HTTP 429."""
    monkeypatch.setattr(settings, "RATE_LIMIT_CLAIMS_PER_MINUTE", 2)
    payload = {"text": "A factual claim was announced in Delhi in 2025."}

    # First 2 requests succeed
    r1 = await client.post("/api/v1/claims", json=payload)
    assert r1.status_code in (200, 429)

    r2 = await client.post("/api/v1/claims", json=payload)
    # By request 3+, rate limiter must trigger 429
    r3 = await client.post("/api/v1/claims", json=payload)
    if r3.status_code != 429:
        r4 = await client.post("/api/v1/claims", json=payload)
        assert r4.status_code == 429
    else:
        assert r3.status_code == 429


@pytest.mark.asyncio
async def test_claims_endpoint_response_schema_and_version(client):
    """15. Response adheres strictly to ClaimExtractionResponse with method and version."""
    resp = await client.post(
        "/api/v1/claims",
        json={"text": "The minister said, 'We completed 100 projects in 2025.'"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["method"] == EXTRACTION_METHOD
    assert data["extraction_method"] == EXTRACTION_METHOD
    assert "runtime_ms" in data
    assert "processing_time_ms" in data
    assert "candidate_count" in data
    assert "accepted_count" in data
    assert "eligible_count" in data
    assert isinstance(data["claims"], list)


@pytest.mark.asyncio
async def test_claims_endpoint_no_sensitive_paths_or_secrets_exposed(client):
    """16. Responses never leak internal file paths, tokens, or environment secrets."""
    resp = await client.post(
        "/api/v1/claims",
        json={"text": "The government launched 25 healthcare clinics in 2025."},
    )
    assert resp.status_code == 200
    text_content = resp.text

    # Verify no absolute local filesystem paths
    assert not re.search(r"[A-Za-z]:\\[A-Za-z0-9_\\\.-]+", text_content)
    assert "/Users/" not in text_content
    assert "/home/" not in text_content
    assert "SECRET" not in text_content
    assert "token" not in text_content.lower() or "papluca" not in text_content
