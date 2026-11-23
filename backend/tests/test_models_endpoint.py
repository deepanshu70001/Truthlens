"""
TruthLens — Tests for the Model Registry endpoint (GET /api/v1/models).

Verifies:
1. Complete model inventory reporting (SigLIP, YOLOv8, EasyOCR, Tesseract, Language detector).
2. Schema validation and HTTP response structure.
3. Critical non-eager loading guarantee: Calling the endpoint never loads model weights.
4. Privacy and safety: Zero private absolute paths, credentials, or secrets in response.
5. Accurate lifecycle status reporting (loaded, cached_not_loaded, available_not_loaded, checkpoint_missing, dependency_unavailable, failed).
6. Execution device reporting when loaded vs null when not loaded.
7. Authentication and rate-limiting enforcement.
8. Partial or altered configuration handling.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from backend.app.config import settings
from backend.app.core.auth import issue_access_token
from backend.app.main import app
from backend.app.schemas.models import ModelRegistryResponse


@pytest.mark.asyncio
async def test_models_registry_inventory_and_schema(client):
    """GET /api/v1/models returns 200 with all 5 configured models adhering to schema."""
    resp = await client.get("/api/v1/models")
    assert resp.status_code == 200

    data = resp.json()
    # Validate against Pydantic schema
    validated = ModelRegistryResponse.model_validate(data)
    assert validated.status == "ok"
    assert validated.total_models == 5
    assert len(validated.models) == 5

    model_ids = {m.id for m in validated.models}
    expected_ids = {
        "authenticity",
        "object_detection",
        "ocr_easyocr",
        "ocr_tesseract",
        "language_detection",
    }
    assert model_ids == expected_ids

    # Verify high-level counters match
    assert validated.loaded_count == sum(1 for m in validated.models if m.loaded)
    assert validated.ready_count == sum(1 for m in validated.models if m.readiness in ("ready", "available"))


@pytest.mark.asyncio
async def test_models_no_eager_loading(client):
    """Calling GET /api/v1/models must NEVER trigger model loading."""
    from backend.app.services.authenticity.detector import AuthenticityDetector
    from backend.app.services.vision.detector import ObjectDetector
    from backend.app.services.ocr.service import OCRService
    from backend.app.services.nlp.language_detector import LanguageDetector

    # Wrap load methods on the actual service classes to assert they are NEVER called
    with patch.object(AuthenticityDetector, "load", autospec=True) as spy_auth_load, \
         patch.object(ObjectDetector, "load", autospec=True) as spy_vision_load, \
         patch.object(OCRService, "load", autospec=True) as spy_ocr_load, \
         patch.object(LanguageDetector, "load", autospec=True) as spy_lang_load:

        resp = await client.get("/api/v1/models")
        assert resp.status_code == 200

        # Assert no .load() methods were invoked
        spy_auth_load.assert_not_called()
        spy_vision_load.assert_not_called()
        spy_ocr_load.assert_not_called()
        spy_lang_load.assert_not_called()


@pytest.mark.asyncio
async def test_models_no_sensitive_paths_or_secrets_exposed(client):
    """Ensure response payload contains zero absolute paths, secrets, or API keys."""
    resp = await client.get("/api/v1/models")
    assert resp.status_code == 200
    raw_text = resp.text

    # No Windows drive paths (e.g. D:\... or C:\...)
    assert not re.search(r"[A-Za-z]:\\[A-Za-z0-9_.\\]+", raw_text), "Windows filesystem path leaked in response"
    # No Unix home paths (e.g. /home/... or /Users/...)
    assert not re.search(r"/(?:home|Users)/[A-Za-z0-9_.-]+", raw_text), "Unix filesystem path leaked in response"

    # No secret key or auth tokens leaked
    if len(settings.SECRET_KEY) >= 8:
        assert settings.SECRET_KEY not in raw_text
    assert "SECRET_KEY" not in raw_text


@pytest.mark.asyncio
async def test_models_missing_checkpoint_status(client, monkeypatch):
    """When a model's configured checkpoint file is missing, it reports checkpoint_missing."""
    monkeypatch.setattr(
        "backend.app.config.settings.AUTHENTICITY_CHECKPOINT_PATH",
        "models/baselines/non_existent_checkpoint_file.pt",
    )
    monkeypatch.setattr(
        "backend.app.config.settings.VISION_MODEL",
        "models/baselines/non_existent_yolo.pt",
    )

    resp = await client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()

    auth_entry = next(m for m in data["models"] if m["id"] == "authenticity")
    vision_entry = next(m for m in data["models"] if m["id"] == "object_detection")

    assert auth_entry["status"] == "checkpoint_missing"
    assert auth_entry["readiness"] == "unavailable"
    assert auth_entry["loaded"] is False
    assert "non_existent_checkpoint_file.pt" in auth_entry["reason"]
    # Checkpoint filename is public, not full path
    assert auth_entry["details"]["checkpoint_filename"] == "non_existent_checkpoint_file.pt"
    assert auth_entry["details"]["checkpoint_present"] is False

    assert vision_entry["status"] == "checkpoint_missing"
    assert vision_entry["readiness"] == "unavailable"
    assert vision_entry["loaded"] is False
    assert "non_existent_yolo.pt" in vision_entry["reason"]


@pytest.mark.asyncio
async def test_models_dependency_unavailable_status(client, monkeypatch):
    """When required Python packages are missing, status is dependency_unavailable."""
    from backend.app.services.ml import registry

    orig_is_pkg = registry._is_package_available

    def mock_is_pkg(pkg_name):
        if pkg_name in ("ultralytics", "easyocr", "pytesseract"):
            return False
        return orig_is_pkg(pkg_name)

    monkeypatch.setattr(registry, "_is_package_available", mock_is_pkg)

    resp = await client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()

    vision = next(m for m in data["models"] if m["id"] == "object_detection")
    easyocr = next(m for m in data["models"] if m["id"] == "ocr_easyocr")
    tesseract = next(m for m in data["models"] if m["id"] == "ocr_tesseract")

    assert vision["status"] == "dependency_unavailable"
    assert vision["readiness"] == "unavailable"
    assert "Ultralytics package is not installed" in vision["reason"]

    assert easyocr["status"] == "dependency_unavailable"
    assert easyocr["readiness"] == "unavailable"
    assert "EasyOCR package is not installed" in easyocr["reason"]

    assert tesseract["status"] == "dependency_unavailable"
    assert tesseract["readiness"] == "unavailable"
    assert "pytesseract package is not installed" in tesseract["reason"]


@pytest.mark.asyncio
async def test_models_failed_initialization_status(client, monkeypatch):
    """When a model has recorded a load failure, status is failed."""
    from backend.app.services.nlp.language_detector import get_language_detector

    lang = get_language_detector()
    monkeypatch.setattr(lang, "_load_failure", "OSError: corrupt index")

    resp = await client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()

    lang_entry = next(m for m in data["models"] if m["id"] == "language_detection")
    assert lang_entry["status"] == "failed"
    assert lang_entry["readiness"] == "failed"
    assert "Language detector initialization failed" in lang_entry["reason"]


@pytest.mark.asyncio
async def test_models_loaded_state_reporting(client, monkeypatch):
    """When models are resident in memory, registry reports loaded=True, readiness=ready, and device."""
    from backend.app.services.authenticity.detector import get_detector
    from backend.app.services.vision.detector import get_object_detector
    from backend.app.services.ocr.service import get_ocr_service
    from backend.app.services.nlp.language_detector import get_language_detector

    auth_det = get_detector()
    vision_det = get_object_detector()
    ocr_srv = get_ocr_service()
    lang_det = get_language_detector()

    # Simulate loaded singletons
    monkeypatch.setattr(auth_det, "is_loaded", True)
    monkeypatch.setattr(auth_det, "get_device", lambda: "cuda:0")

    monkeypatch.setattr(vision_det, "is_loaded", True)
    dummy_yolo_model = MagicMock()
    dummy_yolo_model.device = "cuda:0"
    monkeypatch.setattr(vision_det, "_model", dummy_yolo_model)

    monkeypatch.setattr(ocr_srv, "_reader", object())
    monkeypatch.setattr(ocr_srv, "_gpu", True)

    monkeypatch.setattr(lang_det, "is_loaded", True)
    monkeypatch.setattr(lang_det, "_device", "cuda:0")

    resp = await client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()

    auth = next(m for m in data["models"] if m["id"] == "authenticity")
    assert auth["loaded"] is True
    assert auth["status"] == "loaded"
    assert auth["readiness"] == "ready"
    assert auth["device"] == "cuda:0"

    vision = next(m for m in data["models"] if m["id"] == "object_detection")
    assert vision["loaded"] is True
    assert vision["status"] == "loaded"
    assert vision["readiness"] == "ready"
    assert vision["device"] == "cuda:0"

    ocr = next(m for m in data["models"] if m["id"] == "ocr_easyocr")
    assert ocr["loaded"] is True
    assert ocr["status"] == "loaded"
    assert ocr["readiness"] == "ready"
    assert ocr["device"] == "cuda"

    lang = next(m for m in data["models"] if m["id"] == "language_detection")
    assert lang["loaded"] is True
    assert lang["status"] == "loaded"
    assert lang["readiness"] == "ready"
    assert lang["device"] == "cuda:0"

    assert data["loaded_count"] >= 4


@pytest.mark.asyncio
async def test_models_authentication_enforced(monkeypatch):
    """GET /api/v1/models requires valid Bearer authentication in production mode."""
    # Temporarily disable development bypass to test auth enforcement
    monkeypatch.setattr(settings, "DEV_AUTH_BYPASS", False)
    monkeypatch.setattr(settings, "APP_ENV", "production")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as unauthed_client:
        # Request with no Authorization header
        resp = await unauthed_client.get("/api/v1/models")
        assert resp.status_code == 401
        assert "Authentication is required" in resp.json()["detail"]

        # Request with invalid token
        resp_bad = await unauthed_client.get(
            "/api/v1/models",
            headers={"Authorization": "Bearer invalid.jwt.token"},
        )
        assert resp_bad.status_code == 401

        # Request with valid token
        valid_token, _ = issue_access_token("test-operator")
        resp_auth = await unauthed_client.get(
            "/api/v1/models",
            headers={"Authorization": f"Bearer {valid_token}"},
        )
        assert resp_auth.status_code == 200
        assert resp_auth.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_models_rate_limiting_enforced(monkeypatch):
    """GET /api/v1/models enforces configured rate limit."""
    monkeypatch.setattr(settings, "RATE_LIMIT_MODELS_PER_MINUTE", 2)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        # 1st request succeeds
        r1 = await test_client.get("/api/v1/models")
        assert r1.status_code == 200

        # 2nd request succeeds
        r2 = await test_client.get("/api/v1/models")
        assert r2.status_code == 200

        # 3rd request exceeds limit
        r3 = await test_client.get("/api/v1/models")
        assert r3.status_code == 429
        assert "Rate limit exceeded" in r3.json()["detail"]


@pytest.mark.asyncio
async def test_models_alternate_authenticity_backend(client, monkeypatch):
    """When AUTHENTICITY_BACKEND is set to huggingface classifier, registry adapts cleanly."""
    monkeypatch.setattr(settings, "AUTHENTICITY_BACKEND", "huggingface")
    monkeypatch.setattr(settings, "AUTHENTICITY_MODEL", "custom-org/deepfake-detector-v2")
    monkeypatch.setattr(settings, "AUTHENTICITY_MODEL_REVISION", "v2.1")

    resp = await client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()

    auth = next(m for m in data["models"] if m["id"] == "authenticity")
    assert auth["model_id"] == "custom-org/deepfake-detector-v2"
    assert auth["version"] == "v2.1"
    assert auth["architecture"] == "SigLIP (Vision Transformer Classifier)"
    assert auth["details"]["backbone_model_id"] is None
