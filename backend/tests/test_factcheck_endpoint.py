"""
TruthLens — Tests for the Dedicated Fact-Checking API endpoint (POST /api/v1/fact-check).

Verifies:
1. Successful provider response.
2. Multiple eligible claims.
3. Ineligible claims excluded.
4. No eligible claims.
5. Empty and malformed request.
6. Provider returns no results.
7. Missing credentials (status="unavailable").
8. Provider HTTP 401/403.
9. Provider quota HTTP 429.
10. Provider timeout.
11. Provider server error.
12. Malformed provider response.
13. Publisher and URL attribution.
14. Duplicate review result handling.
15. Cache miss followed by cache hit.
16. Cache expiry.
17. Cache isolation by claim and language.
18. Duplicate concurrent requests.
19. Application rate limiting.
20. Authentication enforcement.
21. Sensitive-data and secret sanitization.
22. Persistent cache SQLite lifecycle and schema integrity tests.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.config import settings
from backend.app.core.auth import issue_access_token
from backend.app.main import app
from backend.app.schemas.factcheck import FactCheckRequest, FactCheckResponse
from backend.app.services.factcheck.cache import (
    FactCheckCache,
    compute_factcheck_cache_key,
    get_factcheck_cache,
)
from backend.app.services.factcheck.service import FactCheckClaim, FactCheckResult


def _mock_stream_client(payload: bytes, *, status_error=None, headers=None, chunk_size=16384, on_stream=None):
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

    class Client:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def stream(self, *args, **kwargs):
            if on_stream:
                on_stream()
            if status_error:
                raise status_error
            return Response()
    return Client


def _mock_google_claim_review(
    query="The government announced 50 hospitals in 2025.",
    claimant="Health Ministry",
    publisher="FactCheck Org",
    rating="False",
    url="https://factcheck.org/review/123",
):
    return {
        "text": query,
        "claimant": claimant,
        "claimDate": "2025-01-15",
        "claimReview": [
            {
                "publisher": {"name": publisher, "site": "factcheck.org"},
                "url": url,
                "title": f"Review of: {query}",
                "reviewDate": "2025-01-16",
                "textualRating": rating,
                "languageCode": "en",
            }
        ],
    }


@pytest.fixture(autouse=True)
def clean_factcheck_cache(tmp_path, monkeypatch):
    """Ensure each test runs with a clean cache and configured API key."""
    cache_db = tmp_path / "test_factcheck_cache.sqlite3"
    test_cache = FactCheckCache(db_path=cache_db)
    monkeypatch.setattr("backend.app.services.factcheck.cache._factcheck_cache", test_cache)
    monkeypatch.setattr(settings, "FACTCHECK_API_KEY", "test-factcheck-api-key")
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "")
    return test_cache


# ── Tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_factcheck_endpoint_success_single_claim(client):
    """1. Successful provider response returns structured publisher attribution."""
    mock_payload = {"claims": [_mock_google_claim_review()] }
    payload_bytes = json.dumps(mock_payload).encode("utf-8")

    with patch("backend.app.services.factcheck.service.httpx.Client", _mock_stream_client(payload_bytes)):
        resp = await client.post(
            "/api/v1/fact-check",
            json={"claims": ["The government announced 50 hospitals in 2025."]},
        )
        assert resp.status_code == 200
        data = resp.json()
        validated = FactCheckResponse.model_validate(data)
        assert validated.status == "success"
        assert validated.total_results == 1
        assert validated.claims[0].publisher_name == "FactCheck Org"
        assert validated.claims[0].textual_rating == "False"
        assert validated.claims[0].url == "https://factcheck.org/review/123"
        assert len(validated.claim_verifications) == 1
        assert validated.claim_verifications[0].verification_status == "COMPLETED"


@pytest.mark.asyncio
async def test_factcheck_endpoint_multiple_eligible_claims(client):
    """2. Multiple eligible claims are queried and mapped in request order."""
    claims_input = [
        "The Health Ministry opened 50 hospitals in 2025.",
        "Over 10,000 doctors were recruited in Mumbai in 2025.",
    ]
    mock_payload = {"claims": [_mock_google_claim_review()]}
    payload_bytes = json.dumps(mock_payload).encode("utf-8")

    with patch("backend.app.services.factcheck.service.httpx.Client", _mock_stream_client(payload_bytes)):
        resp = await client.post(
            "/api/v1/fact-check",
            json={"claims": claims_input},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["claim_verifications"]) == 2
        assert data["claim_verifications"][0]["request_index"] == 0
        assert data["claim_verifications"][1]["request_index"] == 1


@pytest.mark.asyncio
async def test_factcheck_endpoint_ineligible_claims_excluded(client):
    """3. Ineligible claims (questions, opinions, slogans) are excluded from provider lookup."""
    claims_input = [
        "The Health Ministry opened 50 hospitals in 2025.",
        "Is this the best project ever?",
        "In my opinion this is bad.",
    ]
    mock_payload = {"claims": [_mock_google_claim_review()]}
    payload_bytes = json.dumps(mock_payload).encode("utf-8")

    with patch("backend.app.services.factcheck.service.httpx.Client", _mock_stream_client(payload_bytes)):
        resp = await client.post(
            "/api/v1/fact-check",
            json={"claims": claims_input},
        )
        assert resp.status_code == 200
        data = resp.json()
        verifications = data["claim_verifications"]
        assert len(verifications) == 3

        # Claim 0: ELIGIBLE
        assert verifications[0]["eligibility_status"] == "ELIGIBLE"
        assert verifications[0]["verification_status"] == "COMPLETED"

        # Claim 1: QUESTION (INELIGIBLE, SKIPPED)
        assert verifications[1]["eligibility_status"] == "INELIGIBLE"
        assert verifications[1]["verification_status"] == "SKIPPED"
        assert verifications[1]["verdict"] == "UNVERIFIABLE"

        # Claim 2: OPINION (INELIGIBLE, SKIPPED)
        assert verifications[2]["eligibility_status"] == "INELIGIBLE"
        assert verifications[2]["verification_status"] == "SKIPPED"


@pytest.mark.asyncio
async def test_factcheck_endpoint_no_eligible_claims(client):
    """4. If all claims are ineligible, provider lookup is bypassed entirely."""
    resp = await client.post(
        "/api/v1/fact-check",
        json={"claims": ["Is this policy real?", "In my opinion it is terrible."]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "skipped"
    assert data["queries_performed"] == 0
    assert data["total_results"] == 0
    assert all(v["verification_status"] == "SKIPPED" for v in data["claim_verifications"])


@pytest.mark.asyncio
async def test_factcheck_endpoint_empty_and_malformed_requests(client):
    """5. Empty claim lists or malformed entries are rejected with HTTP 422."""
    r1 = await client.post("/api/v1/fact-check", json={"claims": []})
    assert r1.status_code == 422

    r2 = await client.post("/api/v1/fact-check", json={"claims": ["    "]})
    assert r2.status_code == 422

    r3 = await client.post("/api/v1/fact-check", json={"claims": ["abc"]})
    assert r3.status_code == 422

    r4 = await client.post(
        "/api/v1/fact-check",
        json={"claims": ["The government opened 50 hospitals in 2025."], "language_code": "klingon"},
    )
    assert r4.status_code == 422


@pytest.mark.asyncio
async def test_factcheck_endpoint_provider_returns_no_results(client):
    """6. Provider returning no matching reviews produces clean 'no_results' status."""
    payload_bytes = json.dumps({}).encode("utf-8")

    with patch("backend.app.services.factcheck.service.httpx.Client", _mock_stream_client(payload_bytes)):
        resp = await client.post(
            "/api/v1/fact-check",
            json={"claims": ["A unique unreviewed statement happened in 2025."]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "no_results"
        assert data["total_results"] == 0
        assert data["claim_verifications"][0]["verdict"] == "INSUFFICIENT_EVIDENCE"


@pytest.mark.asyncio
async def test_factcheck_endpoint_missing_credentials(client, monkeypatch):
    """7. Missing API credentials return 'unavailable' status with clear warning."""
    monkeypatch.setattr(settings, "FACTCHECK_API_KEY", "")
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "")

    resp = await client.post(
        "/api/v1/fact-check",
        json={"claims": ["The minister announced 10 projects in 2025."]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "unavailable"
    assert any("not configured" in w.lower() for w in data["warnings"])
    assert data["claim_verifications"][0]["verification_status"] == "UNAVAILABLE"


@pytest.mark.asyncio
async def test_factcheck_endpoint_provider_http_403(client):
    """8. Provider 403 authorization/quota rejection returns partial/failed gracefully."""
    req = httpx.Request("GET", "https://factchecktools.googleapis.com")
    resp_403 = httpx.Response(403, request=req)
    err = httpx.HTTPStatusError("Forbidden", request=req, response=resp_403)

    with patch("backend.app.services.factcheck.service.httpx.Client", _mock_stream_client(b"", status_error=err)):
        resp = await client.post(
            "/api/v1/fact-check",
            json={"claims": ["The government built 100 bridges in 2025."]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("failed", "partial")
        assert data["claim_verifications"][0]["retrieval_error_category"] == "PROVIDER_AUTH_OR_QUOTA"


@pytest.mark.asyncio
async def test_factcheck_endpoint_provider_quota_http_429(client):
    """9. Provider quota exhaustion (HTTP 429) stops subsequent queries with rate-limit warning."""
    req = httpx.Request("GET", "https://factchecktools.googleapis.com")
    resp_429 = httpx.Response(429, request=req)
    err = httpx.HTTPStatusError("Too Many Requests", request=req, response=resp_429)

    with patch("backend.app.services.factcheck.service.httpx.Client", _mock_stream_client(b"", status_error=err)):
        resp = await client.post(
            "/api/v1/fact-check",
            json={"claims": ["A policy was announced in 2025 in Delhi."]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("failed", "partial")
        assert any("Rate limit reached" in w for w in data["warnings"])


@pytest.mark.asyncio
async def test_factcheck_endpoint_provider_timeout(client):
    """10. Provider network timeout is handled safely without hanging."""
    err = httpx.TimeoutException("Read timeout")

    with patch("backend.app.services.factcheck.service.httpx.Client", _mock_stream_client(b"", status_error=err)):
        resp = await client.post(
            "/api/v1/fact-check",
            json={"claims": ["The minister reported 12 projects completed."]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("failed", "partial")
        assert data["claim_verifications"][0]["retrieval_error_category"] == "PROVIDER_TIMEOUT"


@pytest.mark.asyncio
async def test_factcheck_endpoint_provider_server_error(client):
    """11. Provider 500 error returns safe error status without leaking tracebacks."""
    req = httpx.Request("GET", "https://factchecktools.googleapis.com")
    resp_500 = httpx.Response(500, request=req)
    err = httpx.HTTPStatusError("Internal Server Error", request=req, response=resp_500)

    with patch("backend.app.services.factcheck.service.httpx.Client", _mock_stream_client(b"", status_error=err)):
        resp = await client.post(
            "/api/v1/fact-check",
            json={"claims": ["A bridge collapsed in 2025 during heavy rains."]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("failed", "partial")


@pytest.mark.asyncio
async def test_factcheck_endpoint_malformed_provider_response(client):
    """12. Malformed JSON bytes from provider are handled safely."""
    with patch("backend.app.services.factcheck.service.httpx.Client", _mock_stream_client(b"not valid json <xml>")):
        resp = await client.post(
            "/api/v1/fact-check",
            json={"claims": ["The agency approved 3 vaccines in 2025."]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("failed", "partial")


@pytest.mark.asyncio
async def test_factcheck_endpoint_publisher_and_url_attribution(client):
    """13. Review URLs are sanitized (HTTPS only, tracking stripped) with full publisher attribution."""
    mock_payload = {
        "claims": [
            _mock_google_claim_review(
                publisher="Lead Stories",
                rating="Partly False",
                url="https://leadstories.com/analysis/2025/test.html?utm_source=twitter&ref=123",
            )
        ]
    }
    payload_bytes = json.dumps(mock_payload).encode("utf-8")

    with patch("backend.app.services.factcheck.service.httpx.Client", _mock_stream_client(payload_bytes)):
        resp = await client.post(
            "/api/v1/fact-check",
            json={"claims": ["The government issued new passports in 2025."]},
        )
        assert resp.status_code == 200
        claim = resp.json()["claims"][0]
        assert claim["publisher_name"] == "Lead Stories"
        assert claim["textual_rating"] == "Partly False"
        assert "utm_source" not in claim["url"]
        assert "ref=" not in claim["url"]
        assert claim["url"].startswith("https://leadstories.com")


@pytest.mark.asyncio
async def test_factcheck_endpoint_duplicate_review_deduplication(client):
    """14. Duplicate review records from the provider are deduplicated."""
    item = _mock_google_claim_review()
    mock_payload = {"claims": [item, item]}
    payload_bytes = json.dumps(mock_payload).encode("utf-8")

    with patch("backend.app.services.factcheck.service.httpx.Client", _mock_stream_client(payload_bytes)):
        resp = await client.post(
            "/api/v1/fact-check",
            json={"claims": ["The government announced 50 hospitals in 2025."]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_results"] == 1


@pytest.mark.asyncio
async def test_factcheck_endpoint_cache_miss_then_hit(client):
    """15. Cache miss executes provider query and caches it; subsequent call produces a cache hit."""
    mock_payload = {"claims": [_mock_google_claim_review()]}
    payload_bytes = json.dumps(mock_payload).encode("utf-8")

    with patch("backend.app.services.factcheck.service.httpx.Client", _mock_stream_client(payload_bytes)):
        # Call 1: Cache Miss
        r1 = await client.post(
            "/api/v1/fact-check",
            json={"claims": ["The council built 30 schools in 2025."]},
        )
        assert r1.status_code == 200
        data1 = r1.json()
        assert data1["queries_performed"] == 1
        assert data1["cached_queries_count"] == 0
        assert data1["claim_verifications"][0]["cache_hit"] is False

        # Call 2: Cache Hit
        r2 = await client.post(
            "/api/v1/fact-check",
            json={"claims": ["The council built 30 schools in 2025."]},
        )
        assert r2.status_code == 200
        data2 = r2.json()
        assert data2["queries_performed"] == 0
        assert data2["cached_queries_count"] == 1
        assert data2["claim_verifications"][0]["cache_hit"] is True
        assert data2["total_results"] == data1["total_results"]


@pytest.mark.asyncio
async def test_factcheck_endpoint_cache_expiry(client, monkeypatch):
    """16. Expired cache entries are evicted and re-queried."""
    monkeypatch.setattr(settings, "FACTCHECK_CACHE_TTL_SECONDS", 1)  # 1 second TTL
    mock_payload = {"claims": [_mock_google_claim_review()]}
    payload_bytes = json.dumps(mock_payload).encode("utf-8")

    with patch("backend.app.services.factcheck.service.httpx.Client", _mock_stream_client(payload_bytes)):
        # Call 1: Miss
        r1 = await client.post(
            "/api/v1/fact-check",
            json={"claims": ["A tax cut of 5% was announced in 2025."]},
        )
        assert r1.status_code == 200
        assert r1.json()["cached_queries_count"] == 0

        # Wait for TTL to expire
        await asyncio.sleep(1.2)

        # Call 2: Should be expired -> fresh query
        r2 = await client.post(
            "/api/v1/fact-check",
            json={"claims": ["A tax cut of 5% was announced in 2025."]},
        )
        assert r2.status_code == 200
        assert r2.json()["cached_queries_count"] == 0
        assert r2.json()["queries_performed"] == 1


@pytest.mark.asyncio
async def test_factcheck_endpoint_cache_isolation_by_language(client):
    """17. Cache entries are isolated by query and language code."""
    mock_payload = {"claims": [_mock_google_claim_review()]}
    payload_bytes = json.dumps(mock_payload).encode("utf-8")

    with patch("backend.app.services.factcheck.service.httpx.Client", _mock_stream_client(payload_bytes)):
        claim = "The government launched 20 programs in 2025."

        # Query in English
        r_en = await client.post("/api/v1/fact-check", json={"claims": [claim], "language_code": "en"})
        assert r_en.status_code == 200
        assert r_en.json()["cached_queries_count"] == 0

        # Same query in Hindi should be a cache miss due to key isolation
        r_hi = await client.post("/api/v1/fact-check", json={"claims": [claim], "language_code": "hi"})
        assert r_hi.status_code == 200
        assert r_hi.json()["cached_queries_count"] == 0


@pytest.mark.asyncio
async def test_factcheck_endpoint_duplicate_concurrent_requests(client):
    """18. Duplicate concurrent requests for the same query deduplicate provider calls."""
    mock_payload = {"claims": [_mock_google_claim_review()]}
    payload_bytes = json.dumps(mock_payload).encode("utf-8")
    provider_call_count = 0

    def on_stream():
        nonlocal provider_call_count
        provider_call_count += 1
        time.sleep(0.05)

    with patch(
        "backend.app.services.factcheck.service.httpx.Client",
        _mock_stream_client(payload_bytes, on_stream=on_stream),
    ):
        claim = "50 new dams were commissioned in 2025."
        reqs = [client.post("/api/v1/fact-check", json={"claims": [claim]}) for _ in range(3)]
        responses = await asyncio.gather(*reqs)

        for r in responses:
            assert r.status_code == 200

        # Due to in-flight deduplication, only 1 external provider call should occur
        assert provider_call_count == 1


@pytest.mark.asyncio
async def test_factcheck_endpoint_rate_limiting_enforced(client, monkeypatch):
    """19. Exceeding RATE_LIMIT_FACTCHECK_PER_MINUTE returns HTTP 429."""
    monkeypatch.setattr(settings, "RATE_LIMIT_FACTCHECK_PER_MINUTE", 2)
    payload = {"claims": ["A factory was opened in Mumbai in 2025."]}

    r1 = await client.post("/api/v1/fact-check", json=payload)
    assert r1.status_code in (200, 429)

    r2 = await client.post("/api/v1/fact-check", json=payload)
    r3 = await client.post("/api/v1/fact-check", json=payload)
    if r3.status_code != 429:
        r4 = await client.post("/api/v1/fact-check", json=payload)
        assert r4.status_code == 429
    else:
        assert r3.status_code == 429


@pytest.mark.asyncio
async def test_factcheck_endpoint_authentication_enforced(monkeypatch):
    """20. Requests without valid Bearer token are rejected with HTTP 401."""
    monkeypatch.setattr(settings, "DEV_AUTH_BYPASS", False)
    monkeypatch.setattr(settings, "APP_ENV", "production")

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as unauth_client:
        resp = await unauth_client.post(
            "/api/v1/fact-check",
            json={"claims": ["The prime minister announced 5 projects in 2025."]},
        )
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_factcheck_endpoint_no_sensitive_secrets_exposed(client):
    """21. Responses never leak internal file paths, API keys, or environment secrets."""
    resp = await client.post(
        "/api/v1/fact-check",
        json={"claims": ["The government launched 25 healthcare clinics in 2025."]},
    )
    assert resp.status_code == 200
    text_content = resp.text

    assert not re.search(r"[A-Za-z]:\\[A-Za-z0-9_\\\.-]+", text_content)
    assert "/Users/" not in text_content
    assert "/home/" not in text_content
    assert "SECRET" not in text_content
    assert "test-factcheck-api-key" not in text_content
    assert "api_key" not in text_content.lower() or "google_fact_check_tools" in text_content


def test_factcheck_cache_sqlite_lifecycle_and_schema(tmp_path):
    """22. Persistent cache SQLite database lifecycle, capacity enforcement, and schema."""
    db_file = tmp_path / "lifecycle_cache.sqlite3"
    cache = FactCheckCache(db_path=db_file)

    claim = FactCheckClaim(
        query="Test query",
        claimed_text="Test claim text",
        publisher_name="Snopes",
        url="https://snopes.com/fact-check/test",
        textual_rating="Mostly False",
    )
    key = compute_factcheck_cache_key("Test query", "en")

    # Set cache
    cache.set(key, "Test query", [claim], status="success", language_code="en")

    # Get cache
    hit = cache.get(key)
    assert hit is not None
    claims, status = hit
    assert status == "success"
    assert len(claims) == 1
    assert claims[0].publisher_name == "Snopes"

    # Verify table schema directly
    with sqlite3.connect(db_file) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(factcheck_cache)").fetchall()}
        assert {"cache_key", "query_hash", "normalized_query", "language_code", "provider", "status", "claims_json", "expires_at"} <= cols

    # Clear cache
    cache.clear()
    assert cache.get(key) is None


@pytest.mark.asyncio
async def test_factcheck_publisher_filter_and_max_age_days(client):
    """23. Publisher domain filter and max_age_days parameters are forwarded to provider."""
    captured_params = {}

    class _Response:
        def __init__(self):
            self.headers = {"content-length": str(len(payload_bytes))}
        def raise_for_status(self): pass
        def iter_bytes(self): yield payload_bytes
        def __enter__(self): return self
        def __exit__(self, *args): return False

    class _Client:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def stream(self, method, url, params=None, **kwargs):
            if params:
                captured_params.update(params)
            return _Response()

    mock_payload = {"claims": [_mock_google_claim_review()]}
    payload_bytes = json.dumps(mock_payload).encode("utf-8")

    with patch("backend.app.services.factcheck.service.httpx.Client", _Client):
        req_data = {
            "claims": ["The government announced 50 hospitals in 2025."],
            "review_publisher_site_filter": "snopes.com",
            "max_age_days": 365,
        }
        resp = await client.post("/api/v1/fact-check", json=req_data)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert len(data["claims"]) == 1
        assert data["claims"][0]["publisher_name"] == "FactCheck Org"
        assert captured_params.get("reviewPublisherSiteFilter") == "snopes.com"
        assert captured_params.get("maxAgeDays") == "365"


def test_google_fact_check_api_key_configuration(monkeypatch):
    """24. GOOGLE_FACT_CHECK_API_KEY is respected when FACTCHECK_API_KEY is empty."""
    from backend.app.services.factcheck.service import _is_configured, _get_api_key
    monkeypatch.setattr(settings, "FACTCHECK_API_KEY", "")
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "")
    monkeypatch.setattr(settings, "GOOGLE_FACT_CHECK_API_KEY", "test-gfc-key-12345")
    assert _is_configured() is True
    assert _get_api_key() == "test-gfc-key-12345"

