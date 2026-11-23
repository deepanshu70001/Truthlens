"""
TruthLens — Tests for the Standalone OCR endpoint (POST /api/v1/ocr).

Verifies:
1. Successful OCR request (text, language, engine, confidence, bounding boxes).
2. Valid image containing no text.
3. Invalid file type rejection (HTTP 422).
4. Corrupt image rejection (HTTP 422).
5. Empty upload rejection (HTTP 422).
6. Oversized upload rejection (HTTP 413).
7. Unsupported image dimensions rejection (HTTP 422).
8. OCR service unavailable (graceful status reporting).
9. EasyOCR failure and Tesseract fallback preservation.
10. Unexpected OCR service exception (safe HTTP 500 without stack trace disclosure).
11. Authentication enforcement (HTTP 401).
12. Rate-limit enforcement (HTTP 429).
13. Response schema validation (OCRResponse).
14. No sensitive paths, stack traces, or secrets leaked in responses.
"""

from __future__ import annotations

import io
import re
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from backend.app.config import settings
from backend.app.core.auth import issue_access_token
from backend.app.main import app
from backend.app.schemas.ocr import OCRResponse
from backend.app.services.ocr.service import OCRBlock, OCRResult
from backend.tests.conftest import create_test_jpeg, create_test_png


@pytest.mark.asyncio
async def test_ocr_endpoint_success(client, mock_optional_analysis_modules):
    """POST /api/v1/ocr returns 200 with extracted text and bounding polygon boxes."""
    test_blocks = [
        OCRBlock(text="TRUTHLENS", confidence=0.985, bbox=[[10, 10], [90, 10], [90, 30], [10, 30]]),
        OCRBlock(text="FORENSICS", confidence=0.942, bbox=[[10, 40], [90, 40], [90, 60], [10, 60]]),
    ]
    mock_optional_analysis_modules["ocr"].extract = lambda _img: OCRResult(
        status="completed",
        text="TRUTHLENS\nFORENSICS",
        normalized_text="TRUTHLENS FORENSICS",
        language="en",
        has_text=True,
        word_count=2,
        total_confidence=0.9635,
        engine="EasyOCR",
        blocks=test_blocks,
        language_configuration=["en", "hi"],
        warnings=[],
    )

    jpeg_bytes = create_test_jpeg(width=120, height=80)
    resp = await client.post(
        "/api/v1/ocr",
        files={"image": ("sample.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()

    # Schema validation
    validated = OCRResponse.model_validate(data)
    assert validated.status == "completed"
    assert validated.has_text is True
    assert validated.detected is True
    assert validated.text == "TRUTHLENS\nFORENSICS"
    assert validated.normalized_text == "TRUTHLENS FORENSICS"
    assert validated.language == "en"
    assert validated.engine == "EasyOCR"
    assert validated.word_count == 2
    assert validated.confidence == 0.9635
    assert len(validated.blocks) == 2
    assert len(validated.results) == 2
    assert validated.results[0].text == "TRUTHLENS"
    assert validated.results[0].bbox == [[10, 10], [90, 10], [90, 30], [10, 30]]


@pytest.mark.asyncio
async def test_ocr_endpoint_no_text_in_image(client, mock_optional_analysis_modules):
    """Valid image with no text returns 200 with has_text=False and informative reason."""
    reason = "EasyOCR ran for en, hi but found no readable text regions."
    mock_optional_analysis_modules["ocr"].extract = lambda _img: OCRResult(
        status="completed",
        text="",
        normalized_text="",
        language="unknown",
        has_text=False,
        no_text_reason=reason,
        word_count=0,
        total_confidence=None,
        engine="EasyOCR",
        blocks=[],
        language_configuration=["en", "hi"],
    )

    jpeg_bytes = create_test_jpeg(width=100, height=100)
    resp = await client.post(
        "/api/v1/ocr",
        files={"image": ("blank.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()

    validated = OCRResponse.model_validate(data)
    assert validated.status == "completed"
    assert validated.has_text is False
    assert validated.text == ""
    assert validated.no_text_reason == reason
    assert validated.confidence is None
    assert len(validated.blocks) == 0


@pytest.mark.asyncio
async def test_ocr_endpoint_invalid_file_type(client):
    """Uploading non-image file type (e.g. text/plain) is rejected with HTTP 422."""
    resp = await client.post(
        "/api/v1/ocr",
        files={"image": ("document.txt", io.BytesIO(b"Hello from a text file!"), "text/plain")},
    )
    assert resp.status_code == 422
    assert "detail" in resp.json()


@pytest.mark.asyncio
async def test_ocr_endpoint_corrupt_image(client):
    """Uploading corrupted image bytes is rejected with HTTP 422."""
    corrupt_data = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00corrupted_truncated_bytes"
    resp = await client.post(
        "/api/v1/ocr",
        files={"image": ("corrupted.jpg", io.BytesIO(corrupt_data), "image/jpeg")},
    )
    assert resp.status_code == 422
    assert "File could not be decoded" in resp.json()["detail"] or "header does not match" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_ocr_endpoint_empty_upload(client):
    """Uploading 0-byte file is rejected with HTTP 422."""
    resp = await client.post(
        "/api/v1/ocr",
        files={"image": ("empty.png", io.BytesIO(b""), "image/png")},
    )
    assert resp.status_code == 422
    assert "empty" in resp.json()["detail"].lower() or "too small" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_ocr_endpoint_oversized_upload(client, monkeypatch):
    """Uploading file exceeding max_upload_bytes is rejected with HTTP 413."""
    # Setting MAX_UPLOAD_SIZE_MB = 0 forces max_upload_bytes = 0
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 0)

    jpeg_bytes = create_test_jpeg(width=100, height=100)
    resp = await client.post(
        "/api/v1/ocr",
        files={"image": ("oversized.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    assert resp.status_code == 413
    assert "exceeds" in resp.json()["detail"].lower() or "too large" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_ocr_endpoint_unsupported_dimensions(client, monkeypatch):
    """Image with dimensions exceeding MAX_IMAGE_PIXELS is rejected with HTTP 422."""
    jpeg_bytes = create_test_jpeg(width=50, height=50)

    # Mock Image.open to return an image reporting oversized dimensions
    real_open = Image.open

    def mock_oversized_open(fp, *args, **kwargs):
        img = real_open(fp, *args, **kwargs)
        img._size = (50000, 50000)  # 2.5 billion pixels > MAX_IMAGE_PIXELS
        return img

    monkeypatch.setattr("PIL.Image.open", mock_oversized_open)

    resp = await client.post(
        "/api/v1/ocr",
        files={"image": ("huge.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    assert resp.status_code == 422
    assert "dimensions exceed" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_ocr_endpoint_service_unavailable(client, mock_optional_analysis_modules):
    """When OCR engine is unavailable, endpoint returns status='unavailable' gracefully."""
    mock_optional_analysis_modules["ocr"].extract = lambda _img: OCRResult(
        status="unavailable",
        text="",
        has_text=False,
        no_text_reason="OCR could not run because no configured OCR engine is available.",
        warnings=["No configured OCR engine is available."],
    )

    jpeg_bytes = create_test_jpeg(width=80, height=80)
    resp = await client.post(
        "/api/v1/ocr",
        files={"image": ("sample.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "unavailable"
    assert data["has_text"] is False
    assert "not run" in data["no_text_reason"]


@pytest.mark.asyncio
async def test_ocr_endpoint_tesseract_fallback(client, mock_optional_analysis_modules):
    """When EasyOCR produces no text and Tesseract succeeds, endpoint returns fallback output."""
    mock_optional_analysis_modules["ocr"].extract = lambda _img: OCRResult(
        status="completed",
        text="FALLBACK TEXT",
        normalized_text="FALLBACK TEXT",
        language="en",
        has_text=True,
        word_count=2,
        total_confidence=0.885,
        engine="PyTesseract",
        blocks=[OCRBlock(text="FALLBACK TEXT", confidence=0.885, bbox=[[5, 5], [95, 5], [95, 25], [5, 25]])],
        warnings=[
            "EasyOCR produced no usable text; attempting the Tesseract fallback.",
            "Readable OCR text was produced by the Tesseract fallback.",
        ],
    )

    jpeg_bytes = create_test_jpeg(width=100, height=50)
    resp = await client.post(
        "/api/v1/ocr",
        files={"image": ("sign.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["engine"] == "PyTesseract"
    assert data["text"] == "FALLBACK TEXT"
    assert any("fallback" in w.lower() for w in data["warnings"])


@pytest.mark.asyncio
async def test_ocr_endpoint_unexpected_exception(client, mock_optional_analysis_modules):
    """Unexpected exception inside OCR extraction returns safe HTTP 500 without stack trace disclosure."""
    def fail_extract(_img):
        raise RuntimeError("Private GPU driver failure: /internal/path/crash.cpp:42")

    mock_optional_analysis_modules["ocr"].extract = fail_extract

    jpeg_bytes = create_test_jpeg(width=100, height=100)
    resp = await client.post(
        "/api/v1/ocr",
        files={"image": ("test.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "unexpected error" in detail.lower()
    # Ensure internal exception details and private paths are NOT leaked
    assert "crash.cpp" not in detail
    assert "Private GPU driver" not in detail


@pytest.mark.asyncio
async def test_ocr_endpoint_authentication_enforced(monkeypatch):
    """POST /api/v1/ocr requires valid Bearer authentication in production mode."""
    monkeypatch.setattr(settings, "SECRET_KEY", "test-secret-key-that-is-at-least-32-bytes-long!")
    monkeypatch.setattr(settings, "DEV_AUTH_BYPASS", False)
    monkeypatch.setattr(settings, "APP_ENV", "production")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as unauthed_client:
        jpeg_bytes = create_test_jpeg(width=50, height=50)

        # Unauthenticated request
        resp_unauthed = await unauthed_client.post(
            "/api/v1/ocr",
            files={"image": ("test.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
        )
        assert resp_unauthed.status_code == 401
        assert "Authentication is required" in resp_unauthed.json()["detail"]

        # Authenticated request
        valid_token, _ = issue_access_token("test-operator-ocr")
        resp_authed = await unauthed_client.post(
            "/api/v1/ocr",
            files={"image": ("test.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
            headers={"Authorization": f"Bearer {valid_token}"},
        )
        assert resp_authed.status_code == 200


@pytest.mark.asyncio
async def test_ocr_endpoint_rate_limiting_enforced(monkeypatch):
    """POST /api/v1/ocr enforces configured rate limit."""
    monkeypatch.setattr(settings, "SECRET_KEY", "test-secret-key-that-is-at-least-32-bytes-long!")
    monkeypatch.setattr(settings, "RATE_LIMIT_OCR_PER_MINUTE", 2)

    user_token, _ = issue_access_token("ocr-rate-limit-isolated-user")
    headers = {"Authorization": f"Bearer {user_token}"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        jpeg_bytes = create_test_jpeg(width=50, height=50)

        # 1st request succeeds
        r1 = await test_client.post(
            "/api/v1/ocr",
            files={"image": ("test1.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
            headers=headers,
        )
        assert r1.status_code == 200

        # 2nd request succeeds
        r2 = await test_client.post(
            "/api/v1/ocr",
            files={"image": ("test2.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
            headers=headers,
        )
        assert r2.status_code == 200

        # 3rd request exceeds rate limit
        r3 = await test_client.post(
            "/api/v1/ocr",
            files={"image": ("test3.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
            headers=headers,
        )
        assert r3.status_code == 429
        assert "Rate limit exceeded" in r3.json()["detail"]


@pytest.mark.asyncio
async def test_ocr_endpoint_no_sensitive_paths_or_secrets_exposed(client, mock_optional_analysis_modules):
    """Response must not contain Windows/Unix absolute paths or configuration secrets."""
    mock_optional_analysis_modules["ocr"].extract = lambda _img: OCRResult(
        status="completed",
        text="Privacy Check",
        normalized_text="Privacy Check",
        language="en",
        has_text=True,
        word_count=2,
        total_confidence=0.99,
        engine="EasyOCR",
        blocks=[OCRBlock(text="Privacy Check", confidence=0.99, bbox=[[0, 0], [50, 0], [50, 20], [0, 20]])],
    )

    jpeg_bytes = create_test_jpeg(width=60, height=40)
    resp = await client.post(
        "/api/v1/ocr",
        files={"image": ("privacy.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    assert resp.status_code == 200
    raw_text = resp.text

    # No Windows drive paths
    assert not re.search(r"[A-Za-z]:\\[A-Za-z0-9_.\\]+", raw_text), "Windows filesystem path leaked"
    # No Unix home paths
    assert not re.search(r"/(?:home|Users)/[A-Za-z0-9_.-]+", raw_text), "Unix filesystem path leaked"

    # No secret key or internal credentials
    if len(settings.SECRET_KEY) >= 8:
        assert settings.SECRET_KEY not in raw_text
    assert "SECRET_KEY" not in raw_text
