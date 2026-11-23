"""
TruthLens — Security Hardening and Production Readiness Test Suite.

Verifies the 22 required security controls from Part 9 of the security audit:
1. Missing authentication
2. Invalid authentication (malformed, bad signature, unsupported algorithm)
3. Expired token
4. Cross-user analysis access
5. Cross-user report access
6. Cross-user artifact access (image, heatmap, ELA)
7. Path traversal prevention (filename & artifact path)
8. Malicious filename sanitization (null bytes, Windows reserved, pure dots)
9. MIME/signature mismatch rejection
10. Corrupt image upload rejection
11. Oversized upload rejection
12. Decompression-bomb protection
13. Rate-limit enforcement
14. Malformed request handling and secret non-echoing
15. Secret and path redaction (payloads and logs)
16. Stack-trace sanitization
17. CORS origin restriction and credential protection
18. Trusted-host enforcement
19. Production configuration validation
20. Provider URL credential and sensitive parameter stripping
21. Unauthorized job status access
22. Artifact path containment
"""

from __future__ import annotations

import base64
import io
import json
import logging
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from PIL import Image

from backend.app.config import Settings, settings
from backend.app.core.auth import decode_access_token, hash_password, issue_access_token
from backend.app.core.logging import SensitiveDataFilter
from backend.app.core.public_output import safe_public_text, sanitize_public_payload
from backend.app.core.rate_limit import check_rate_limit
from backend.app.core.security import sanitize_filename, validate_image_file
from backend.app.services.factcheck.service import _sanitize_review_url
from backend.app.services.provenance.service import _safe_external_url
from backend.tests.conftest import create_test_jpeg, create_test_png


# ── 1. Missing authentication ──────────────────────────────────────────────
@pytest.mark.asyncio
async def test_01_missing_authentication_rejected_on_all_protected_endpoints(
    client_without_worker, monkeypatch
):
    monkeypatch.setattr(settings, "DEV_AUTH_BYPASS", False)
    monkeypatch.setattr(settings, "SECRET_KEY", "test-secret-key-with-at-least-32-bytes")

    fake_id = str(uuid4())
    protected_endpoints = [
        ("GET", f"/api/v1/analysis/{fake_id}"),
        ("GET", f"/api/v1/analysis/{fake_id}/report"),
        ("GET", f"/api/v1/analysis/{fake_id}/image"),
        ("GET", f"/api/v1/analysis/{fake_id}/heatmap"),
        ("GET", f"/api/v1/analysis/{fake_id}/forensics/ela"),
        ("GET", "/api/v1/analysis"),
        ("DELETE", f"/api/v1/analysis/{fake_id}"),
        ("GET", "/api/v1/models"),
        ("GET", "/api/v1/health/models"),
        ("POST", "/api/v1/claims"),
        ("POST", "/api/v1/fact-check"),
        ("POST", "/api/v1/nlp/language"),
    ]

    for method, path in protected_endpoints:
        if method == "GET":
            resp = await client_without_worker.get(path)
        elif method == "DELETE":
            resp = await client_without_worker.delete(path)
        else:
            resp = await client_without_worker.post(path, json={"text": "test claim"})
        assert resp.status_code == 401, f"{method} {path} did not reject missing authentication"
        assert resp.headers.get("www-authenticate") == "Bearer"


# ── 2. Invalid authentication (malformed, bad signature, unsupported alg) ───
@pytest.mark.asyncio
async def test_02_invalid_authentication_rejected(client_without_worker, monkeypatch):
    monkeypatch.setattr(settings, "DEV_AUTH_BYPASS", False)
    monkeypatch.setattr(settings, "SECRET_KEY", "test-secret-key-with-at-least-32-bytes")

    fake_id = str(uuid4())

    # Malformed token (not 3 dot-separated parts)
    res1 = await client_without_worker.get(
        f"/api/v1/analysis/{fake_id}", headers={"Authorization": "Bearer not-three-parts"}
    )
    assert res1.status_code == 401

    # Non-bearer scheme
    res2 = await client_without_worker.get(
        f"/api/v1/analysis/{fake_id}", headers={"Authorization": "Basic dXNlcjpwYXNz"}
    )
    assert res2.status_code == 401

    # Valid token structure signed with a different key
    token_diff_key, _ = issue_access_token("alice")
    monkeypatch.setattr(settings, "SECRET_KEY", "a-completely-different-signing-key-32b")
    with pytest.raises(HTTPException) as exc_diff:
        decode_access_token(token_diff_key)
    assert exc_diff.value.status_code == 401

    # Token with unsupported algorithm (e.g. alg=none)
    hdr = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(b"=").decode()
    claims = base64.urlsafe_b64encode(
        json.dumps({
            "iss": "truthlens", "aud": "truthlens-api", "sub": "alice",
            "exp": 9999999999, "iat": 1000000000, "jti": "test-jti",
        }).encode()
    ).rstrip(b"=").decode()
    none_token = f"{hdr}.{claims}."
    with pytest.raises(HTTPException) as exc_none:
        decode_access_token(none_token)
    assert exc_none.value.status_code == 401


# ── 3. Expired token ────────────────────────────────────────────────────────
def test_03_expired_token_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "SECRET_KEY", "test-secret-key-with-at-least-32-bytes")
    token, _ = issue_access_token("alice", now=1000)
    with pytest.raises(HTTPException) as exc:
        # Simulate decode when current time is past expiry
        decode_access_token(token, now=1000 + settings.ACCESS_TOKEN_TTL_SECONDS + 1)
    assert exc.value.status_code == 401
    assert "Invalid or expired" in exc.value.detail


# ── 4. Cross-user analysis access ──────────────────────────────────────────
@pytest.mark.asyncio
async def test_04_cross_user_analysis_access_denied_with_safe_404(client_without_worker, monkeypatch):
    monkeypatch.setattr(settings, "DEV_AUTH_BYPASS", False)
    monkeypatch.setattr(settings, "SECRET_KEY", "test-secret-key-with-at-least-32-bytes")
    monkeypatch.setattr(
        settings,
        "AUTH_USERS_JSON",
        json.dumps({
            "alice": hash_password("alice-secure-password"),
            "bob": hash_password("bob-secure-password"),
        }),
    )

    alice_token, _ = issue_access_token("alice")
    bob_token, _ = issue_access_token("bob")
    alice_hdrs = {"Authorization": f"Bearer {alice_token}"}
    bob_hdrs = {"Authorization": f"Bearer {bob_token}"}

    # Alice uploads an analysis
    sub = await client_without_worker.post(
        "/api/v1/analyze",
        files={"image": ("alice_doc.jpg", create_test_jpeg(), "image/jpeg")},
        headers=alice_hdrs,
    )
    assert sub.status_code == 201
    analysis_id = sub.json()["analysis_id"]

    # Alice can view it
    assert (await client_without_worker.get(f"/api/v1/analysis/{analysis_id}", headers=alice_hdrs)).status_code == 200

    # Bob cannot view it; receives 404 (does not leak existence)
    bob_resp = await client_without_worker.get(f"/api/v1/analysis/{analysis_id}", headers=bob_hdrs)
    assert bob_resp.status_code == 404
    assert bob_resp.json()["detail"] == "Analysis not found."


# ── 5. Cross-user report access ────────────────────────────────────────────
@pytest.mark.asyncio
async def test_05_cross_user_report_access_denied_with_safe_404(client_without_worker, monkeypatch):
    monkeypatch.setattr(settings, "DEV_AUTH_BYPASS", False)
    monkeypatch.setattr(settings, "SECRET_KEY", "test-secret-key-with-at-least-32-bytes")

    alice_token, _ = issue_access_token("alice")
    bob_token, _ = issue_access_token("bob")
    alice_hdrs = {"Authorization": f"Bearer {alice_token}"}
    bob_hdrs = {"Authorization": f"Bearer {bob_token}"}

    sub = await client_without_worker.post(
        "/api/v1/analyze",
        files={"image": ("report_test.jpg", create_test_jpeg(), "image/jpeg")},
        headers=alice_hdrs,
    )
    analysis_id = sub.json()["analysis_id"]

    # Bob cannot retrieve Alice's report
    bob_report = await client_without_worker.get(f"/api/v1/analysis/{analysis_id}/report", headers=bob_hdrs)
    assert bob_report.status_code == 404


# ── 6. Cross-user artifact access (image, heatmap, ELA) ────────────────────
@pytest.mark.asyncio
async def test_06_cross_user_artifact_access_denied_with_safe_404(client_without_worker, monkeypatch):
    monkeypatch.setattr(settings, "DEV_AUTH_BYPASS", False)
    monkeypatch.setattr(settings, "SECRET_KEY", "test-secret-key-with-at-least-32-bytes")

    alice_token, _ = issue_access_token("alice")
    bob_token, _ = issue_access_token("bob")
    alice_hdrs = {"Authorization": f"Bearer {alice_token}"}
    bob_hdrs = {"Authorization": f"Bearer {bob_token}"}

    sub = await client_without_worker.post(
        "/api/v1/analyze",
        files={"image": ("artifacts_test.jpg", create_test_jpeg(), "image/jpeg")},
        headers=alice_hdrs,
    )
    analysis_id = sub.json()["analysis_id"]

    for artifact_path in (
        f"/api/v1/analysis/{analysis_id}/image",
        f"/api/v1/analysis/{analysis_id}/heatmap",
        f"/api/v1/analysis/{analysis_id}/forensics/ela",
    ):
        res = await client_without_worker.get(artifact_path, headers=bob_hdrs)
        assert res.status_code == 404, f"Cross-user access leaked artifact at {artifact_path}"


# ── 7. Path traversal attempt ──────────────────────────────────────────────
def test_07_path_traversal_in_filename_is_neutralized():
    assert ".." not in sanitize_filename("../../../../etc/passwd")
    assert ".." not in sanitize_filename("..\\..\\..\\windows\\system32\\config\\SAM")
    assert ".." not in sanitize_filename("..//..//data//secret.jpg")
    safe = sanitize_filename(r"C:\Windows\System32\drivers\etc\hosts.jpg")
    assert "Windows" not in safe
    assert "hosts.jpg" in safe


# ── 8. Malicious filename sanitization ─────────────────────────────────────
def test_08_malicious_filename_sanitization():
    # Null bytes and control characters
    assert "\x00" not in sanitize_filename("exploit\x00.jpg")
    # Pure dots and slashes
    assert sanitize_filename("...") == "unnamed"
    assert sanitize_filename("....") == "unnamed"
    assert sanitize_filename("/") == "unnamed"
    assert sanitize_filename("\\") == "unnamed"
    # Windows reserved device names
    assert sanitize_filename("CON") == "unnamed"
    assert sanitize_filename("PRN") == "unnamed"
    assert sanitize_filename("NUL") == "unnamed"
    assert sanitize_filename("AUX") == "unnamed"


# ── 9. MIME/signature mismatch rejection ───────────────────────────────────
def test_09_mime_signature_mismatch_rejected():
    jpeg_data = create_test_jpeg()
    png_data = create_test_png()

    # Declared PNG but content is JPEG
    with pytest.raises(ValueError, match="MIME type"):
        validate_image_file(jpeg_data, "photo.jpg", "image/png")

    # Declared JPEG but content is PNG
    with pytest.raises(ValueError, match="MIME type"):
        validate_image_file(png_data, "photo.png", "image/jpeg")

    # Extension mismatch: declared JPEG, detected JPEG, but filename has .png
    with pytest.raises(ValueError, match="Filename extension"):
        validate_image_file(jpeg_data, "photo.png", "image/jpeg")


# ── 10. Corrupt image upload rejection ─────────────────────────────────────
def test_10_corrupt_image_upload_rejected():
    corrupt_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 50  # Corrupted JPEG header
    with pytest.raises(ValueError, match="File could not be decoded"):
        validate_image_file(corrupt_bytes, "corrupt.jpg", "image/jpeg")

    random_garbage = b"this is obviously not an image at all but has 32 bytes"
    with pytest.raises(ValueError):
        validate_image_file(random_garbage, "garbage.jpg", "image/jpeg")


# ── 11. Oversized upload rejection ─────────────────────────────────────────
def test_11_oversized_upload_rejected(monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 1)  # 1 MB
    oversized = b"\x00" * (1 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="File too large"):
        validate_image_file(oversized, "huge.jpg", "image/jpeg")


# ── 12. Decompression-bomb protection ──────────────────────────────────────
def test_12_decompression_bomb_protection(monkeypatch):
    monkeypatch.setattr(settings, "MAX_IMAGE_PIXELS", 500)
    img_buf = io.BytesIO()
    # 50x50 = 2500 pixels > 500 limit
    Image.new("RGB", (50, 50), "black").save(img_buf, format="JPEG")
    with pytest.raises(ValueError, match="safety limit"):
        validate_image_file(img_buf.getvalue(), "bomb.jpg", "image/jpeg")


# ── 13. Rate-limit enforcement ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_13_rate_limit_enforcement():
    from fastapi import Request

    req = Request({
        "type": "http", "http_version": "1.1", "method": "POST",
        "scheme": "http", "path": "/api/v1/test-limit", "raw_path": b"/api/v1/test-limit",
        "query_string": b"", "headers": [], "client": ("198.51.100.42", 5000), "server": ("test", 80),
    })
    scope = f"test-audit-scope-{uuid4()}"
    # Limit to 2 requests
    await check_rate_limit(req, scope=scope, limit=2, window_seconds=60)
    await check_rate_limit(req, scope=scope, limit=2, window_seconds=60)
    with pytest.raises(HTTPException) as limited:
        await check_rate_limit(req, scope=scope, limit=2, window_seconds=60)
    assert limited.value.status_code == 429
    assert "Retry-After" in limited.value.headers


# ── 14. Malformed request handling and secret non-echoing ──────────────────
@pytest.mark.asyncio
async def test_14_malformed_request_does_not_echo_sensitive_data(client_without_worker):
    res = await client_without_worker.post(
        "/api/v1/auth/token",
        json={"username": "alice", "password": {"secret_token": "super-confidential-credential"}},
    )
    assert res.status_code == 422
    assert "super-confidential-credential" not in res.text
    assert "secret_token" not in res.text


# ── 15. Secret and path redaction (payloads and logs) ───────────────────────
def test_15_secret_and_path_redaction_in_payloads_and_logs():
    sensitive_dict = {
        "api_key": "secret-12345",
        "password": "my-secret-password",
        "access_token": "jwt-token-string",
        "nested": {"secretKey": "hidden", "storage_path": "D:\\Projects\\Truthlens\\data\\photo.jpg"},
        "clean_field": "public information",
    }
    sanitized = sanitize_public_payload(sensitive_dict)
    assert "api_key" not in sanitized
    assert "password" not in sanitized
    assert "access_token" not in sanitized
    assert "secretKey" not in sanitized["nested"]
    assert "storage_path" not in sanitized["nested"]
    assert sanitized["clean_field"] == "public information"

    # Log filter redaction
    log_filter = SensitiveDataFilter()
    record1 = logging.LogRecord("test", logging.INFO, "path", 1, "User logged in with bearer eyJhbGciOiJIUzI1NiJ9.test.sig", (), None)
    log_filter.filter(record1)
    assert "eyJhbGci" not in record1.msg
    assert "[redacted]" in record1.msg

    record2 = logging.LogRecord("test", logging.INFO, "path", 1, "Calling external API with api_key=AIzaSySecretKeyHere", (), None)
    log_filter.filter(record2)
    assert "AIzaSy" not in record2.msg
    assert "[redacted]" in record2.msg


# ── 16. Stack-trace sanitization ───────────────────────────────────────────
def test_16_stack_trace_sanitization():
    raw_error = "Traceback (most recent call last):\n  File 'D:\\app\\secret.py', line 42\nException: boom"
    safe = safe_public_text(raw_error)
    assert "Traceback" not in safe
    assert "secret.py" not in safe
    assert "Internal error details are hidden." == safe


# ── 17. CORS origin restriction and credential protection ──────────────────
@pytest.mark.asyncio
async def test_17_cors_origin_restriction_and_credentials_forbidden(client_without_worker):
    # Allowed origin gets CORS header
    res_allowed = await client_without_worker.get(
        "/api/v1/health", headers={"Origin": "http://localhost:5173"}
    )
    assert res_allowed.status_code == 200
    assert res_allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    # Credentials must be False to prevent credential exposure attacks
    assert res_allowed.headers.get("access-control-allow-credentials") is None

    # Untrusted origin does not get allow-origin header
    res_untrusted = await client_without_worker.get(
        "/api/v1/health", headers={"Origin": "https://evil-attacker.com"}
    )
    assert res_untrusted.status_code == 200
    assert "access-control-allow-origin" not in res_untrusted.headers


# ── 18. Trusted-host enforcement ───────────────────────────────────────────
@pytest.mark.asyncio
async def test_18_trusted_host_enforcement(client_without_worker):
    # Legitimate host succeeds
    ok_host = await client_without_worker.get("/api/v1/health", headers={"Host": "localhost"})
    assert ok_host.status_code == 200

    # Spoofed Host header rejected with 400
    bad_host = await client_without_worker.get("/api/v1/health", headers={"Host": "untrusted-host.net"})
    assert bad_host.status_code == 400


# ── 19. Production configuration validation ────────────────────────────────
def test_19_production_configuration_validation():
    # Production with debug mode enabled must fail
    with pytest.raises(ValueError, match="APP_DEBUG"):
        Settings(
            APP_ENV="production", APP_DEBUG=True, SECRET_KEY="s" * 40,
            DEV_AUTH_BYPASS=False, AUTH_USERS_JSON=json.dumps({"u": hash_password("pass12345678")}),
            TRUSTED_HOSTS="truthlens.org", CORS_ORIGINS="https://truthlens.org",
            DATABASE_URL="postgresql+asyncpg://user:pass@db:5432/truthlens",
        ).validate_runtime_security()

    # Production with dev bypass enabled must fail
    with pytest.raises(ValueError, match="Development authentication bypass"):
        Settings(
            APP_ENV="production", APP_DEBUG=False, SECRET_KEY="s" * 40,
            DEV_AUTH_BYPASS=True, AUTH_USERS_JSON=json.dumps({"u": hash_password("pass12345678")}),
            TRUSTED_HOSTS="truthlens.org", CORS_ORIGINS="https://truthlens.org",
            DATABASE_URL="postgresql+asyncpg://user:pass@db:5432/truthlens",
        ).validate_runtime_security()

    # Production with SQLite database URL must fail
    with pytest.raises(ValueError, match="PostgreSQL"):
        Settings(
            APP_ENV="production", APP_DEBUG=False, SECRET_KEY="s" * 40,
            DEV_AUTH_BYPASS=False, AUTH_USERS_JSON=json.dumps({"u": hash_password("pass12345678")}),
            TRUSTED_HOSTS="truthlens.org", CORS_ORIGINS="https://truthlens.org",
            DATABASE_URL="sqlite+aiosqlite:///./truthlens.db",
        ).validate_runtime_security()

    # Production with wildcard CORS must fail
    with pytest.raises(ValueError, match="CORS_ORIGINS"):
        Settings(
            APP_ENV="production", APP_DEBUG=False, SECRET_KEY="s" * 40,
            DEV_AUTH_BYPASS=False, AUTH_USERS_JSON=json.dumps({"u": hash_password("pass12345678")}),
            TRUSTED_HOSTS="truthlens.org", CORS_ORIGINS="*",
            DATABASE_URL="postgresql+asyncpg://user:pass@db:5432/truthlens",
        ).validate_runtime_security()


# ── 20. Provider URL credential and sensitive parameter stripping ──────────
def test_20_provider_url_credential_and_parameter_stripping():
    # Provenance external URLs strip credentials and local loopback
    assert _safe_external_url("http://127.0.0.1:8080/internal") == ""
    assert _safe_external_url("https://user:pass@public-site.com/image.jpg") == ""
    # Sensitive query parameters removed
    sanitized = _safe_external_url("https://public-site.com/review?token=secret123&api_key=456&page=1")
    assert "token" not in sanitized
    assert "api_key" not in sanitized
    assert "page=1" in sanitized

    # Fact-check review URLs strip tracking and sensitive params
    fc_clean = _sanitize_review_url("https://factcheck.org/review?utm_source=twitter&secret=abc&id=42")
    assert "utm_source" not in fc_clean
    assert "secret" not in fc_clean
    assert "id=42" in fc_clean


# ── 21. Unauthorized job status access ─────────────────────────────────────
@pytest.mark.asyncio
async def test_21_unauthorized_job_status_access_denied(client_without_worker, monkeypatch):
    monkeypatch.setattr(settings, "DEV_AUTH_BYPASS", False)
    monkeypatch.setattr(settings, "SECRET_KEY", "test-secret-key-with-at-least-32-bytes")

    alice_token, _ = issue_access_token("alice")
    bob_token, _ = issue_access_token("bob")

    # Alice creates a job
    sub = await client_without_worker.post(
        "/api/v1/analyze",
        files={"image": ("job_test.jpg", create_test_jpeg(), "image/jpeg")},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    analysis_id = sub.json()["analysis_id"]

    # Bob attempts to inspect status / job
    bob_check = await client_without_worker.get(
        f"/api/v1/analysis/{analysis_id}",
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert bob_check.status_code == 404
    assert "job" not in bob_check.text


# ── 22. Artifact path containment ──────────────────────────────────────────
@pytest.mark.asyncio
async def test_22_artifact_path_containment_prevents_escape(client_without_worker, monkeypatch):
    monkeypatch.setattr(settings, "DEV_AUTH_BYPASS", False)
    monkeypatch.setattr(settings, "SECRET_KEY", "test-secret-key-with-at-least-32-bytes")

    alice_token, _ = issue_access_token("alice")
    alice_hdrs = {"Authorization": f"Bearer {alice_token}"}

    # Attempt path traversal via analysis_id in artifact endpoints
    for traversal_id in ("..", "..%2F..", "some-uuid/../../etc/passwd"):
        resp = await client_without_worker.get(
            f"/api/v1/analysis/{traversal_id}/image", headers=alice_hdrs
        )
        assert resp.status_code in {404, 422}
