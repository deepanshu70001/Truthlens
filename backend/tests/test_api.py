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
import json
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


@pytest.mark.asyncio
async def test_upload_empty_file_is_rejected(client):
    resp = await client.post(
        "/api/v1/analyze",
        files={"image": ("empty.png", io.BytesIO(b""), "image/png")},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_upload_extension_must_match_image_content(client):
    resp = await client.post(
        "/api/v1/analyze",
        files={"image": ("image.txt", io.BytesIO(create_test_jpeg()), "image/jpeg")},
    )
    assert resp.status_code == 422


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
    The pipeline should expose a valid queued, processing, completed, or partial state.
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
    assert data["status"] in ("queued", "processing", "completed", "partial")
    assert "progress" in data
    assert "image" in data
    assert data.get("caveat") is not None  # Always present


@pytest.mark.asyncio
async def test_upload_to_partial_report_and_image_retrieval(client):
    """Exercise the integrated path with optional providers unavailable."""
    import asyncio

    image_bytes = create_test_jpeg(width=160, height=120)
    response = await client.post(
        "/api/v1/analyze",
        files={"image": ("traceability.jpg", io.BytesIO(image_bytes), "image/jpeg")},
    )
    assert response.status_code == 201
    analysis_id = response.json()["analysis_id"]

    status = None
    for _ in range(150):
        await asyncio.sleep(0.1)
        status_response = await client.get(f"/api/v1/analysis/{analysis_id}")
        assert status_response.status_code == 200
        status = status_response.json()
        if status["status"] in {"completed", "partial", "failed"}:
            break
    assert status and status["status"] in {"completed", "partial"}

    report_response = await client.get(f"/api/v1/analysis/{analysis_id}/report")
    assert report_response.status_code == 200
    report = report_response.json()
    assert report["analysis_id"] == analysis_id
    assert report["stages"]["preprocessing"]["status"] == "completed"
    assert report["stages"]["ocr"]["status"] == "completed"
    assert report["stages"]["ocr"]["outcome"] == "no_text"
    assert report["stages"]["language_identification"]["status"] == "completed"
    assert report["stages"]["language_identification"]["outcome"] == "no_text"
    assert report["preprocessing"]["original_width"] == 160
    assert report["preprocessing"]["original_height"] == 120
    assert len(report["stages"]) == 21
    assert report["stages"]["preprocessing"]["correlation_id"] == analysis_id
    assert report["summary"]["spread"]["regional_data_status"] == "DATA_UNAVAILABLE"
    assert "storage_path" not in json.dumps(report)

    image_response = await client.get(f"/api/v1/analysis/{analysis_id}/image")
    assert image_response.status_code == 200
    assert image_response.content == image_bytes


@pytest.mark.asyncio
async def test_ocr_text_is_passed_to_text_normalization(client, mock_optional_analysis_modules):
    import asyncio
    from backend.app.services.ocr.service import OCRResult

    raw_text = "The city opened 125 new schools in 2026."
    mock_optional_analysis_modules["ocr"].extract = lambda _image: OCRResult(
        text=raw_text, normalized_text=raw_text, has_text=True, word_count=8,
        engine="mock-ocr", status="completed", language_configuration=["en"],
    )
    response = await client.post(
        "/api/v1/analyze",
        files={"image": ("ocr-text.jpg", io.BytesIO(create_test_jpeg()), "image/jpeg")},
    )
    assert response.status_code == 201
    analysis_id = response.json()["analysis_id"]
    for _ in range(100):
        await asyncio.sleep(0.05)
        status = (await client.get(f"/api/v1/analysis/{analysis_id}")).json()
        if status["status"] in {"completed", "partial", "failed"}:
            break
    assert status["status"] in {"completed", "partial"}
    report = (await client.get(f"/api/v1/analysis/{analysis_id}/report")).json()
    assert report["ocr"]["text"] == raw_text
    assert mock_optional_analysis_modules["language"].received_text == raw_text
    assert report["language_identification"]["language_code"] == "en"
    assert report["language_identification"]["language_name"] == "English"
    assert report["stages"]["language_identification"]["status"] == "completed"
    assert report["stages"]["language_identification"]["outcome"] == "success"
    assert report["text_normalized"]["raw_text"] == raw_text
    assert report["claims"]["claims"]
    assert all(claim["language"] == "en" for claim in report["claims"]["claims"])
    assert report["stages"]["ocr"]["status"] == "completed"
    assert report["stages"]["ocr"]["outcome"] == "success"


@pytest.mark.asyncio
async def test_unavailable_language_model_does_not_block_image_or_text_stages(
    client, mock_optional_analysis_modules
):
    import asyncio
    from backend.app.services.nlp.language_detector import LanguageDetectionResult
    from backend.app.services.ocr.service import OCRResult

    raw_text = "The city opened 125 new schools in 2026."
    mock_optional_analysis_modules["ocr"].extract = lambda _image: OCRResult(
        text=raw_text, normalized_text=raw_text, has_text=True, word_count=8,
        engine="mock-ocr", status="completed", language_configuration=["en"],
    )

    def unavailable(_text, **_kwargs):
        return LanguageDetectionResult(
            status="unavailable", outcome="unavailable",
            warnings=["The local language model is unavailable."],
        )

    mock_optional_analysis_modules["language"].detect = unavailable
    response = await client.post(
        "/api/v1/analyze",
        files={"image": ("language-unavailable.jpg", io.BytesIO(create_test_jpeg()), "image/jpeg")},
    )
    assert response.status_code == 201
    analysis_id = response.json()["analysis_id"]
    for _ in range(100):
        await asyncio.sleep(0.05)
        status = (await client.get(f"/api/v1/analysis/{analysis_id}")).json()
        if status["status"] in {"completed", "partial", "failed"}:
            break
    report = (await client.get(f"/api/v1/analysis/{analysis_id}/report")).json()
    assert status["status"] in {"completed", "partial"}
    assert report["stages"]["language_identification"]["status"] == "unavailable"
    assert report["stages"]["language_identification"]["outcome"] == "unavailable"
    assert report["stages"]["vision"]["status"] == "completed"
    assert report["text_normalized"]["raw_text"] == raw_text


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


# ── Report and heatmap artifact routes ──────────────────────────

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
async def test_heatmap_and_ela_routes_serve_analysis_artifacts(client):
    """Artifact endpoints serve files associated with the requested analysis."""
    jpeg_bytes = create_test_jpeg()
    resp = await client.post(
        "/api/v1/analyze",
        files={"image": ("heatmap_test.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    analysis_id = resp.json()["analysis_id"]

    # The durable worker may generate its own ELA artifact. Wait for the
    # persisted terminal state before replacing it with a route fixture.
    from backend.tests.conftest import await_analysis_terminal
    await await_analysis_terminal(client, analysis_id)

    from backend.app.config import settings
    from PIL import Image

    heatmap_dir = settings.heatmaps_dir / analysis_id
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), "red").save(heatmap_dir / "overlay.png")
    Image.new("L", (12, 8), 128).save(heatmap_dir / "attribution.png")
    resp = await client.get(f"/api/v1/analysis/{analysis_id}/heatmap")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/png")
    assert Image.open(io.BytesIO(resp.content)).size == (12, 8)
    raw = await client.get(f"/api/v1/analysis/{analysis_id}/heatmap?variant=raw")
    assert raw.status_code == 200
    invalid_variant = await client.get(f"/api/v1/analysis/{analysis_id}/heatmap?variant=invalid")
    assert invalid_variant.status_code == 422

    ela_dir = settings.artifacts_dir / "forensics"
    ela_dir.mkdir(parents=True, exist_ok=True)
    Image.new("L", (12, 8), 64).save(ela_dir / f"{analysis_id}_ela.png")
    ela = await client.get(f"/api/v1/analysis/{analysis_id}/forensics/ela")
    assert ela.status_code == 200
    assert Image.open(io.BytesIO(ela.content)).size == (12, 8)
