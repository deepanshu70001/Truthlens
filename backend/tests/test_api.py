"""
TruthLens — Tests for the image upload and analysis endpoints.

Tests cover:
- Valid image upload (JPEG, PNG)
- Invalid file type rejection
- Oversized file rejection
- Hash generation
- Analysis status lifecycle
- Health endpoint
"""

import io
import os

import pytest
import pytest_asyncio

from backend.tests.conftest import create_test_jpeg, create_test_png


# ── Health endpoint ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_endpoint(client):
    """GET /api/v1/health returns healthy status."""
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "gpu" in data
    assert "database" in data
    assert data["database"] == "connected"


@pytest.mark.asyncio
async def test_health_models_endpoint(client):
    """GET /api/v1/health/models returns stub response."""
    resp = await client.get("/api/v1/health/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data


# ── Valid image upload ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_valid_jpeg(client):
    """POST /api/v1/analyze with valid JPEG returns 201 + analysis_id."""
    jpeg_bytes = create_test_jpeg()

    resp = await client.post(
        "/api/v1/analyze",
        files={"image": ("test_photo.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "analysis_id" in data
    assert data["status"] == "queued"
    assert len(data["analysis_id"]) == 36  # UUID format


@pytest.mark.asyncio
async def test_upload_valid_png(client):
    """POST /api/v1/analyze with valid PNG returns 201."""
    png_bytes = create_test_png()

    resp = await client.post(
        "/api/v1/analyze",
        files={"image": ("test_photo.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "analysis_id" in data
    assert data["status"] == "queued"


# ── Invalid file type ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_invalid_file_type(client):
    """POST /api/v1/analyze with a text file should be rejected."""
    text_data = b"This is not an image file at all."

    resp = await client.post(
        "/api/v1/analyze",
        files={"image": ("malicious.txt", io.BytesIO(text_data), "text/plain")},
    )
    assert resp.status_code == 422
    data = resp.json()
    assert "detail" in data


@pytest.mark.asyncio
async def test_upload_fake_mime_header(client):
    """
    POST /api/v1/analyze with JPEG content-type but non-image content
    should be rejected by magic-byte validation.
    """
    fake_data = b"Not a real image content - just random bytes here"

    resp = await client.post(
        "/api/v1/analyze",
        files={"image": ("fake.jpg", io.BytesIO(fake_data), "image/jpeg")},
    )
    assert resp.status_code == 422
    data = resp.json()
    assert "detail" in data


# ── Oversized file ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_oversized_file(client):
    """POST /api/v1/analyze with file larger than MAX_UPLOAD_SIZE_MB should be rejected."""
    # Create a file just over the limit (default 20 MB)
    # For testing, we set a small limit via env
    original_max = os.environ.get("MAX_UPLOAD_SIZE_MB")
    os.environ["MAX_UPLOAD_SIZE_MB"] = "0"  # 0 MB = reject everything

    # Need to reload settings
    from backend.app.config import Settings
    test_settings = Settings()

    # Create data larger than 0 MB
    from unittest.mock import patch
    with patch("backend.app.core.security.settings", test_settings):
        resp = await client.post(
            "/api/v1/analyze",
            files={"image": ("big.jpg", io.BytesIO(create_test_jpeg()), "image/jpeg")},
        )
    assert resp.status_code == 422

    # Restore
    if original_max is not None:
        os.environ["MAX_UPLOAD_SIZE_MB"] = original_max
    elif "MAX_UPLOAD_SIZE_MB" in os.environ:
        del os.environ["MAX_UPLOAD_SIZE_MB"]


# ── Analysis status lifecycle ────────────────────────────────────

@pytest.mark.asyncio
async def test_analysis_status_lifecycle(client):
    """
    Upload an image, then GET the analysis status.
    The stub pipeline should transition to completed.
    """
    jpeg_bytes = create_test_jpeg()

    # Upload
    resp = await client.post(
        "/api/v1/analyze",
        files={"image": ("lifecycle.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    assert resp.status_code == 201
    analysis_id = resp.json()["analysis_id"]

    # Wait briefly for the background pipeline to complete
    import asyncio
    await asyncio.sleep(1.0)

    # Check status
    resp = await client.get(f"/api/v1/analysis/{analysis_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["analysis_id"] == analysis_id
    assert data["status"] in ("queued", "processing", "completed")
    assert "progress" in data
    assert "image" in data
    assert data.get("caveat") is not None  # Always present


@pytest.mark.asyncio
async def test_analysis_not_found(client):
    """GET /api/v1/analysis/{bad_id} returns 404."""
    resp = await client.get("/api/v1/analysis/nonexistent-id-12345")
    assert resp.status_code == 404
    data = resp.json()
    assert "detail" in data


# ── Hash generation ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hash_generation(client):
    """Upload should generate SHA-256 and perceptual hash."""
    jpeg_bytes = create_test_jpeg()

    resp = await client.post(
        "/api/v1/analyze",
        files={"image": ("hash_test.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    assert resp.status_code == 201
    analysis_id = resp.json()["analysis_id"]

    import asyncio
    await asyncio.sleep(0.5)

    resp = await client.get(f"/api/v1/analysis/{analysis_id}")
    assert resp.status_code == 200
    data = resp.json()
    image_data = data.get("image")
    assert image_data is not None
    assert image_data["sha256_hash"] is not None
    assert len(image_data["sha256_hash"]) == 64  # SHA-256 hex length
    assert image_data["perceptual_hash"] is not None


# ── Metadata extraction ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_image_dimensions_extracted(client):
    """Upload should extract width and height."""
    jpeg_bytes = create_test_jpeg(width=320, height=240)

    resp = await client.post(
        "/api/v1/analyze",
        files={"image": ("dims_test.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    assert resp.status_code == 201
    analysis_id = resp.json()["analysis_id"]

    import asyncio
    await asyncio.sleep(0.5)

    resp = await client.get(f"/api/v1/analysis/{analysis_id}")
    data = resp.json()
    image_data = data.get("image")
    assert image_data is not None
    assert image_data["width"] == 320
    assert image_data["height"] == 240
    assert image_data["format"] == "JPEG"
    assert image_data["mime_type"] == "image/jpeg"


# ── Database creation ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_database_record_created(client):
    """Upload should create both Analysis and ImageAsset records."""
    jpeg_bytes = create_test_jpeg()

    resp = await client.post(
        "/api/v1/analyze",
        files={"image": ("db_test.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    assert resp.status_code == 201
    analysis_id = resp.json()["analysis_id"]

    import asyncio
    await asyncio.sleep(0.5)

    # Verify via API that both records exist
    resp = await client.get(f"/api/v1/analysis/{analysis_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["analysis_id"] == analysis_id
    assert data["image"] is not None
    assert data["image"]["file_size_bytes"] > 0


# ── Report and heatmap stubs ────────────────────────────────────

@pytest.mark.asyncio
async def test_report_stub(client):
    """GET /analysis/{id}/report returns stub message."""
    jpeg_bytes = create_test_jpeg()
    resp = await client.post(
        "/api/v1/analyze",
        files={"image": ("report_test.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    analysis_id = resp.json()["analysis_id"]

    resp = await client.get(f"/api/v1/analysis/{analysis_id}/report")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_heatmap_stub(client):
    """GET /analysis/{id}/heatmap returns stub message."""
    jpeg_bytes = create_test_jpeg()
    resp = await client.post(
        "/api/v1/analyze",
        files={"image": ("heatmap_test.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    analysis_id = resp.json()["analysis_id"]

    resp = await client.get(f"/api/v1/analysis/{analysis_id}/heatmap")
    assert resp.status_code == 200
