"""Security-boundary tests for auth, ownership, upload validation and limits."""

from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException, Request

from backend.app.config import Settings, settings
from backend.app.core.auth import decode_access_token, hash_password, issue_access_token
from backend.app.core.security import sanitize_filename, validate_image_file
from backend.app.services.provenance.service import _safe_external_url


def _request():
    return Request({
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/v1/test",
        "raw_path": b"/api/v1/test",
        "query_string": b"",
        "headers": [],
        "client": ("192.0.2.10", 1234),
        "server": ("test", 80),
    })


@pytest.mark.asyncio
async def test_authentication_rejects_missing_malformed_and_expired_tokens(client_without_worker, monkeypatch):
    monkeypatch.setattr(settings, "DEV_AUTH_BYPASS", False)
    monkeypatch.setattr(settings, "SECRET_KEY", "test-secret-key-with-at-least-32-bytes")

    missing = await client_without_worker.get(f"/api/v1/analysis/{uuid4()}")
    assert missing.status_code == 401

    malformed = await client_without_worker.get(
        f"/api/v1/analysis/{uuid4()}", headers={"Authorization": "Bearer not.a.jwt"}
    )
    assert malformed.status_code == 401

    token, _ = issue_access_token("alice")
    with pytest.raises(HTTPException) as expired:
        decode_access_token(token, now=10**12)
    assert expired.value.status_code == 401

    malformed_body = await client_without_worker.post(
        "/api/v1/auth/token",
        json={"username": "alice", "password": {"api_key": "do-not-echo-this"}},
    )
    assert malformed_body.status_code == 422
    assert "do-not-echo-this" not in malformed_body.text
    assert "api_key" not in malformed_body.text


@pytest.mark.asyncio
async def test_owner_is_enforced_for_status_report_image_and_delete(client_without_worker, monkeypatch):
    from backend.tests.conftest import create_test_jpeg

    monkeypatch.setattr(settings, "DEV_AUTH_BYPASS", False)
    monkeypatch.setattr(settings, "SECRET_KEY", "test-secret-key-with-at-least-32-bytes")
    alice_password = hash_password("alice-password-for-tests")
    bob_password = hash_password("bob-password-for-tests")
    monkeypatch.setattr(settings, "AUTH_USERS_JSON", json.dumps({"alice": alice_password, "bob": bob_password}))

    alice_login = await client_without_worker.post(
        "/api/v1/auth/token", json={"username": "alice", "password": "alice-password-for-tests"}
    )
    bob_login = await client_without_worker.post(
        "/api/v1/auth/token", json={"username": "bob", "password": "bob-password-for-tests"}
    )
    assert alice_login.status_code == bob_login.status_code == 200
    alice_headers = {"Authorization": f"Bearer {alice_login.json()['access_token']}"}
    bob_headers = {"Authorization": f"Bearer {bob_login.json()['access_token']}"}

    submitted = await client_without_worker.post(
        "/api/v1/analyze",
        files={"image": ("owned.jpg", create_test_jpeg(), "image/jpeg")},
        headers=alice_headers,
    )
    assert submitted.status_code == 201
    analysis_id = submitted.json()["analysis_id"]
    assert (await client_without_worker.get(f"/api/v1/analysis/{analysis_id}", headers=alice_headers)).status_code == 200
    for suffix in ("", "/report", "/image"):
        response = await client_without_worker.get(f"/api/v1/analysis/{analysis_id}{suffix}", headers=bob_headers)
        assert response.status_code == 404
    assert (await client_without_worker.delete(f"/api/v1/analysis/{analysis_id}", headers=bob_headers)).status_code == 404
    assert (await client_without_worker.delete(f"/api/v1/analysis/{analysis_id}", headers=alice_headers)).status_code == 204


@pytest.mark.asyncio
async def test_shared_rate_limit_enforces_threshold_and_fails_closed(monkeypatch):
    from backend.app.core import rate_limit

    scope = f"test-{uuid4()}"
    await rate_limit.check_rate_limit(_request(), scope=scope, limit=1)
    with pytest.raises(HTTPException) as limited:
        await rate_limit.check_rate_limit(_request(), scope=scope, limit=1)
    assert limited.value.status_code == 429
    assert int(limited.value.headers["Retry-After"]) > 0

    class BrokenFactory:
        kw = {"bind": SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))}

        def __call__(self):
            raise OSError("database unavailable")

    monkeypatch.setattr(rate_limit, "async_session_factory", BrokenFactory())
    with pytest.raises(HTTPException) as unavailable:
        await rate_limit.check_rate_limit(_request(), scope=f"fail-{uuid4()}", limit=1)
    assert unavailable.value.status_code == 503


def test_production_configuration_requires_provisioned_secret_and_auth():
    unsafe = Settings(
        APP_ENV="production",
        APP_DEBUG=True,
        SECRET_KEY="",
        DEV_AUTH_BYPASS=False,
        AUTH_USERS_JSON="{}",
        TRUSTED_HOSTS="api.example.test",
        CORS_ORIGINS="https://app.example.test",
        DATABASE_URL="postgresql+asyncpg://user:pass@db.example.test/truthlens",
    )
    with pytest.raises(ValueError, match="APP_DEBUG"):
        unsafe.validate_runtime_security()

    valid_base = {
        "APP_ENV": "production",
        "APP_DEBUG": False,
        "SECRET_KEY": "s" * 40,
        "DEV_AUTH_BYPASS": False,
        "AUTH_USERS_JSON": json.dumps({"operator": hash_password("long-enough-password")}),
        "TRUSTED_HOSTS": "api.example.test",
        "CORS_ORIGINS": "https://app.example.test",
        "DATABASE_URL": "postgresql+asyncpg://user:pass@db.example.test/truthlens",
    }
    Settings(**valid_base).validate_runtime_security()
    with pytest.raises(ValueError, match="SECRET_KEY"):
        Settings(**{**valid_base, "SECRET_KEY": "short"}).validate_runtime_security()


def test_upload_mime_magic_dimensions_and_filename_protection(monkeypatch):
    from backend.tests.conftest import create_test_jpeg

    payload = create_test_jpeg()
    safe_name, actual_type = validate_image_file(payload, r"..\..\private\photo.jpg", "image/jpeg")
    assert safe_name == "photo.jpg"
    assert actual_type == "image/jpeg"
    with pytest.raises(ValueError, match="MIME type"):
        validate_image_file(payload, "photo.jpg", "image/png")
    with pytest.raises(ValueError):
        validate_image_file(b"not-an-image-file", "photo.jpg", "image/jpeg")
    assert ".." not in sanitize_filename(r"..\..\secret.jpg")

    from PIL import Image
    import io
    oversized_dimensions = io.BytesIO()
    Image.new("RGB", (40, 40), "black").save(oversized_dimensions, format="JPEG")
    monkeypatch.setattr(settings, "MAX_IMAGE_PIXELS", 1_000)
    with pytest.raises(ValueError, match="dimensions"):
        validate_image_file(oversized_dimensions.getvalue(), "large.jpg", "image/jpeg")

    from backend.app.core.middleware import apply_security_headers
    from starlette.responses import Response
    response = apply_security_headers(Response())
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.asyncio
async def test_request_body_limit_cors_and_trusted_host(client_without_worker):
    from backend.app.core.middleware import RequestBodyLimitMiddleware

    sent = []
    async def inner_app(_scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive_body():
        body = b"0" * (8 + 128 * 1024 + 1)
        return {"type": "http.request", "body": body, "more_body": False}

    async def collect(message):
        sent.append(message)

    middleware = RequestBodyLimitMiddleware(inner_app, upload_limit=8)
    scope = {"type": "http", "path": "/api/v1/analyze", "headers": []}
    await middleware(scope, receive_body, collect)
    assert sent[0]["status"] == 413

    bad_host = await client_without_worker.get("/api/v1/health", headers={"Host": "attacker.invalid"})
    assert bad_host.status_code == 400
    blocked_origin = await client_without_worker.get(
        "/api/v1/health", headers={"Origin": "https://attacker.invalid"}
    )
    assert blocked_origin.status_code == 200
    assert "access-control-allow-origin" not in blocked_origin.headers
    assert blocked_origin.headers["X-Content-Type-Options"] == "nosniff"


def test_provider_urls_are_not_backend_fetch_targets_and_sensitive_query_is_removed():
    assert _safe_external_url("http://127.0.0.1/admin") == ""
    assert _safe_external_url("https://user:pass@example.test/claim") == ""
    assert _safe_external_url("https://example.test/review?token=secret&id=2").startswith(
        "https://example.test/review?id=2"
    )


def test_public_report_sanitization_removes_secret_fields_paths_and_tracebacks():
    from backend.app.core.public_output import sanitize_public_payload

    value = sanitize_public_payload({
        "apiKey": "must-not-escape",
        "nested": {"secret_key": "also-hidden", "storagePath": "C:\\private\\upload.jpg"},
        "warning": "Traceback (most recent call last): C:\\private\\module.py",
    })
    assert "apiKey" not in value
    assert "secret_key" not in value["nested"]
    assert "storagePath" not in value["nested"]
    assert "must-not-escape" not in json.dumps(value)
    assert "module.py" not in json.dumps(value)

