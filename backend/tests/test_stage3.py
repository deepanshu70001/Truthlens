"""
TruthLens — Stage 3 tests: ML pipeline integration tests.

Tests cover:
- Authenticity model loads and runs inference
- Forensic analysis produces signals
- OCR extracts text
- XAI generates a heatmap
- Vision (object detection) works
- Full pipeline integration
"""

import io
import os
import os
import sys
import time
from pathlib import Path

import pytest
import numpy as np
from PIL import Image, ImageDraw

# Fixture path
FIXTURE_DIR = Path(__file__).parent / "fixtures"
TEST_IMAGE_PATH = FIXTURE_DIR / "test_photo.jpg"


def _ensure_test_image():
    """Create a test image if it doesn't exist."""
    if TEST_IMAGE_PATH.exists():
        return
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    width, height = 640, 480
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height // 2):
        img_array[y, :] = [int(200 - y * 0.3), int(200 - y * 0.3 + 20), 255]
    for y in range(height // 2, height):
        img_array[y, :] = [50, int(100 + (y - height // 2) * 0.3), 30]
    noise = np.random.randint(-20, 20, img_array.shape, dtype=np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "Breaking News: Test Image", fill=(255, 255, 0))
    draw.text((50, 80), "This is a test for TruthLens", fill=(255, 255, 255))
    img.save(str(TEST_IMAGE_PATH), "JPEG", quality=90)


_ensure_test_image()


# ── Forensic Analysis Tests ──────────────────────────────────────

class TestForensics:
    def test_forensics_produces_signals(self):
        """Forensic analysis should return multiple signals."""
        from backend.app.services.forensics.analyzer import analyze_forensics

        image_data = TEST_IMAGE_PATH.read_bytes()
        result = analyze_forensics(image_data, "JPEG")

        assert len(result.signals) >= 5
        assert result.overall_anomaly_score is None
        assert result.runtime_ms >= 0
        assert result.caveat  # Always has a caveat

    def test_forensic_signal_structure(self):
        """Each signal should have name, score, interpretation, limitations."""
        from backend.app.services.forensics.analyzer import analyze_forensics

        image_data = TEST_IMAGE_PATH.read_bytes()
        result = analyze_forensics(image_data, "JPEG")

        for signal in result.signals:
            assert signal.name, f"Signal missing name"
            assert 0.0 <= signal.score <= 1.0, f"{signal.name} score out of range: {signal.score}"
            assert signal.interpretation, f"{signal.name} missing interpretation"
            assert signal.limitations, f"{signal.name} missing limitations"

    def test_ela_signal_present(self):
        """Error Level Analysis should be one of the signals."""
        from backend.app.services.forensics.analyzer import analyze_forensics

        image_data = TEST_IMAGE_PATH.read_bytes()
        result = analyze_forensics(image_data, "JPEG")

        signal_names = [s.name for s in result.signals]
        assert "error_level_analysis" in signal_names

    def test_jpeg_artifact_analysis_for_jpeg(self):
        """JPEG artifact analysis should run for JPEG images."""
        from backend.app.services.forensics.analyzer import analyze_forensics

        image_data = TEST_IMAGE_PATH.read_bytes()
        result = analyze_forensics(image_data, "JPEG")

        jpeg_sig = [s for s in result.signals if s.name == "jpeg_artifacts"]
        assert len(jpeg_sig) == 1
        assert jpeg_sig[0].score >= 0.0

    def test_forensics_to_dict(self):
        """to_dict() should produce a serializable dictionary."""
        from backend.app.services.forensics.analyzer import analyze_forensics
        import json

        image_data = TEST_IMAGE_PATH.read_bytes()
        result = analyze_forensics(image_data, "JPEG")

        d = result.to_dict()
        json_str = json.dumps(d)  # Should not raise
        assert "signals" in d
        assert "overall_anomaly_score" in d
        assert "caveat" in d

    def test_forensics_png_skips_jpeg_analysis(self):
        """PNG image should skip JPEG artifact analysis."""
        from backend.app.services.forensics.analyzer import analyze_forensics

        buf = io.BytesIO()
        Image.new("RGB", (200, 200)).save(buf, format="PNG")
        result = analyze_forensics(buf.getvalue(), "PNG")

        jpeg_sig = [s for s in result.signals if s.name == "jpeg_artifacts"]
        assert len(jpeg_sig) == 1
        assert "not applicable" in jpeg_sig[0].interpretation.lower()


# ── Authenticity Detector Tests ──────────────────────────────────

@pytest.mark.skipif(os.environ.get("TRUTHLENS_REAL_MODEL_TESTS") != "1", reason="Set TRUTHLENS_REAL_MODEL_TESTS=1 to run cache-backed model smoke tests")
class TestAuthenticity:
    """
    Tests that require the authenticity model to be downloaded.
    Marked with 'slow' for CI environments without model cache.
    """

    @pytest.fixture(autouse=True)
    def _check_model_available(self):
        """Skip if model not cached (first-time download might timeout)."""
        from backend.app.config import settings
        cache_dir = Path(settings.MODEL_CACHE_DIR)
        # Check if any model files exist in cache
        if not cache_dir.exists() or not any(cache_dir.rglob("*.safetensors")):
            pytest.skip("Authenticity model not cached — run download_models.py first")

    def test_model_loads(self):
        """Authenticity model should load successfully."""
        from backend.app.services.authenticity.detector import AuthenticityDetector

        detector = AuthenticityDetector()
        detector.load(device="cpu")  # Force CPU for tests
        assert detector.is_loaded
        assert detector.get_internal_model() is not None
        assert detector.get_processor() is not None
        detector.unload()
        assert not detector.is_loaded

    def test_inference_returns_result(self):
        """Model inference should return an AuthenticityResult."""
        from backend.app.services.authenticity.detector import AuthenticityDetector

        detector = AuthenticityDetector()
        detector.load(device="cpu")

        img = Image.open(str(TEST_IMAGE_PATH)).convert("RGB")
        result = detector.detect(img)

        assert result.label in ("likely_ai_generated", "likely_manipulated", "likely_authentic", "inconclusive")
        assert 0.0 <= result.ai_generated_score <= 1.0
        assert 0.0 <= result.authentic_score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.model_name
        assert result.runtime_ms >= 0

        detector.unload()

    def test_result_to_dict_has_caveat(self):
        """Result dict must always include a caveat about AI-assisted assessment."""
        from backend.app.services.authenticity.detector import AuthenticityDetector

        detector = AuthenticityDetector()
        detector.load(device="cpu")

        img = Image.open(str(TEST_IMAGE_PATH)).convert("RGB")
        result = detector.detect(img)

        d = result.to_dict()
        assert "caveat" in d
        assert "proof" in d["caveat"].lower() or "probabilistic" in d["caveat"].lower()

        detector.unload()

    def test_xai_heatmap_generation(self):
        """Grad-CAM should generate a heatmap from the model."""
        from backend.app.services.authenticity.detector import AuthenticityDetector
        from backend.app.services.xai.gradcam import generate_heatmap
        import torch

        detector = AuthenticityDetector()
        detector.load(device="cpu")

        img = Image.open(str(TEST_IMAGE_PATH)).convert("RGB")
        xai_result = generate_heatmap(
            model=detector.get_internal_model(),
            processor=detector.get_processor(),
            image=img,
            analysis_id="test-xai-123",
            device=detector.get_device(),
        )

        assert xai_result is not None
        assert xai_result.method == "grad_cam"

        # Check if heatmap file was created
        if xai_result.heatmap_path:
            assert Path(xai_result.heatmap_path).exists()
        if xai_result.overlay_path:
            assert Path(xai_result.overlay_path).exists()

        detector.unload()


# ── OCR Tests ────────────────────────────────────────────────────

@pytest.mark.skipif(os.environ.get("TRUTHLENS_REAL_MODEL_TESTS") != "1", reason="Set TRUTHLENS_REAL_MODEL_TESTS=1 to run cache-backed model smoke tests")
class TestOCR:
    @pytest.fixture(autouse=True)
    def _check_easyocr(self):
        try:
            import easyocr
        except ImportError:
            pytest.skip("EasyOCR not installed")

    def test_ocr_extracts_text(self):
        """OCR should extract text from the test image."""
        from backend.app.services.ocr.service import OCRService

        service = OCRService(languages=["en"], gpu=False)
        service.load()

        img = Image.open(str(TEST_IMAGE_PATH)).convert("RGB")
        result = service.extract(img)

        assert result.has_text
        assert len(result.blocks) > 0
        assert result.word_count > 0
        assert result.total_confidence > 0
        assert result.runtime_ms >= 0

        service.unload()

    def test_ocr_block_structure(self):
        """Each OCR block should have text, confidence, and bbox."""
        from backend.app.services.ocr.service import OCRService

        service = OCRService(languages=["en"], gpu=False)
        service.load()

        img = Image.open(str(TEST_IMAGE_PATH)).convert("RGB")
        result = service.extract(img)

        for block in result.blocks:
            assert block.text
            assert 0.0 <= block.confidence <= 1.0
            assert len(block.bbox) == 4  # Four corners

        service.unload()

    def test_ocr_to_dict_serializable(self):
        """OCR result to_dict should be JSON-serializable."""
        from backend.app.services.ocr.service import OCRService
        import json

        service = OCRService(languages=["en"], gpu=False)
        service.load()

        img = Image.open(str(TEST_IMAGE_PATH)).convert("RGB")
        result = service.extract(img)

        d = result.to_dict()
        json_str = json.dumps(d)  # Should not raise
        assert "text" in d
        assert "blocks" in d
        assert "language" in d

        service.unload()

    def test_ocr_empty_image(self):
        """OCR on a blank image should return empty result gracefully."""
        from backend.app.services.ocr.service import OCRService

        service = OCRService(languages=["en"], gpu=False)
        service.load()

        blank = Image.new("RGB", (200, 200), color=(0, 0, 0))
        result = service.extract(blank)

        # Should not crash, may or may not find text
        assert isinstance(result.blocks, list)
        assert result.runtime_ms >= 0

        service.unload()


# ── Vision (Object Detection) Tests ──────────────────────────────

@pytest.mark.skipif(os.environ.get("TRUTHLENS_REAL_MODEL_TESTS") != "1", reason="Set TRUTHLENS_REAL_MODEL_TESTS=1 to run cache-backed model smoke tests")
class TestVision:
    @pytest.fixture(autouse=True)
    def _check_ultralytics(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            pytest.skip("ultralytics not installed")

    def test_object_detection_runs(self):
        """Object detection should run without errors."""
        from backend.app.services.vision.detector import ObjectDetector

        detector = ObjectDetector()
        detector.load()

        img = Image.open(str(TEST_IMAGE_PATH)).convert("RGB")
        result = detector.detect(img)

        assert result is not None
        assert isinstance(result.objects, list)
        assert result.runtime_ms >= 0
        assert result.model_name

        detector.unload()

    def test_vision_result_to_dict(self):
        """Vision result should be serializable."""
        from backend.app.services.vision.detector import ObjectDetector
        import json

        detector = ObjectDetector()
        detector.load()

        img = Image.open(str(TEST_IMAGE_PATH)).convert("RGB")
        result = detector.detect(img)

        d = result.to_dict()
        json_str = json.dumps(d)  # Should not raise
        assert "objects" in d
        assert "note" in d

        detector.unload()


# ── Device Manager Tests ─────────────────────────────────────────

class TestDeviceManager:
    def test_resolve_cpu(self):
        from backend.app.utils.device import resolve_device
        import torch
        device = resolve_device("cpu")
        assert device.type == "cpu"

    def test_resolve_auto(self):
        from backend.app.utils.device import resolve_device
        device = resolve_device("auto")
        # Should resolve to something valid
        assert device.type in ("cpu", "cuda", "mps")

    def test_gpu_memory_info(self):
        from backend.app.utils.device import get_gpu_memory_info
        info = get_gpu_memory_info()
        # May or may not have GPU
        assert isinstance(info, dict)


# ── Integration Test ─────────────────────────────────────────────

class TestIntegration:
    """Pipeline integration test using the API and mocked detector."""

    @pytest.mark.asyncio
    async def test_full_pipeline_via_api(self, monkeypatch):
        """Upload an image and verify the pipeline under a deterministic mock detector."""
        import asyncio
        from httpx import AsyncClient, ASGITransport
        from backend.app.services.authenticity.detector import AuthenticityResult

        class MockDetector:
            is_loaded = False

            def load(self, device="auto"):
                return None

            def detect(self, image):
                return AuthenticityResult(
                    label="inconclusive", ai_generated_score=0.5,
                    authentic_score=0.5, confidence=0.5,
                    model_name="mock-detector", model_version="test",
                )

            def get_device(self):
                return None

        monkeypatch.setattr(
            "backend.app.services.authenticity.detector.get_detector",
            lambda: MockDetector(),
        )

        os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_stage3.db"
        os.environ["APP_ENV"] = "test"

        # Re-import to get fresh settings
        from backend.app.main import app
        from backend.app.db.session import engine
        from backend.app.db.base import Base
        import backend.app.models

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Upload
            image_data = TEST_IMAGE_PATH.read_bytes()
            resp = await client.post(
                "/api/v1/analyze",
                files={"image": ("test_photo.jpg", io.BytesIO(image_data), "image/jpeg")},
            )
            assert resp.status_code == 201
            analysis_id = resp.json()["analysis_id"]

            # Wait for pipeline (may take time with model inference)
            for _ in range(60):  # Wait up to 60 seconds
                await asyncio.sleep(1)
                resp = await client.get(f"/api/v1/analysis/{analysis_id}")
                data = resp.json()
                if data["status"] in ("completed", "partial", "failed"):
                    break

            assert data["status"] in ("completed", "partial"), f"Pipeline ended with status: {data['status']}"
            assert data["progress"]["completed_steps"]
            if data["status"] == "partial":
                report_resp = await client.get(f"/api/v1/analysis/{analysis_id}/report")
                report = report_resp.json()
                assert report["analysis_id"] == analysis_id
                assert report["status"] == "partial"
                assert report["stages"]["ocr"]["status"] in ("unavailable", "failed")

        from backend.app.api.v1.endpoints.analyze import wait_for_analysis_tasks
        await wait_for_analysis_tasks()

        # Cleanup
        await engine.dispose()
        db_path = Path("./test_stage3.db")
        if db_path.exists():
            try:
                db_path.unlink()
            except OSError:
                pass
