"""
TruthLens — Test configuration and fixtures.
"""

import io
import os
import sys
import asyncio
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# Ensure we use SQLite for tests
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_truthlens.db"
os.environ["STORAGE_PATH"] = str(Path(tempfile.gettempdir()) / "truthlens-phase1-tests")
os.environ["APP_DEBUG"] = "true"
os.environ["APP_ENV"] = "test"

from backend.app.main import app  # noqa: E402
from backend.app.db.session import engine  # noqa: E402
from backend.app.db.base import Base  # noqa: E402
import backend.app.models  # noqa: E402, F401


@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_db():
    """Create all tables before tests, drop after."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Dispose engine connections first to avoid SQLite lock
    await engine.dispose()
    # Clean up test database file
    db_path = Path("./test_truthlens.db")
    if db_path.exists():
        try:
            db_path.unlink()
        except OSError:
            pass  # File may still be locked on Windows; acceptable


@pytest.fixture
def mock_auth_detector(monkeypatch):
    """Keep API tests deterministic and prevent implicit checkpoint downloads."""
    from backend.app.services.authenticity.detector import AuthenticityResult

    class MockDetector:
        is_loaded = False

        def load(self, device="auto"):
            return None

        def detect(self, image):
            return AuthenticityResult(
                label="inconclusive",
                ai_generated_score=0.5,
                authentic_score=0.5,
                confidence=0.5,
                model_name="mock-detector",
                model_version="test",
            )

        def get_device(self):
            return None

    mock = MockDetector()
    monkeypatch.setattr(
        "backend.app.services.authenticity.detector.get_detector", lambda: mock
    )
    return mock


@pytest_asyncio.fixture
async def client(mock_auth_detector, mock_optional_analysis_modules):
    """Async HTTP test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    from backend.app.api.v1.endpoints.analyze import wait_for_analysis_tasks
    await wait_for_analysis_tasks()


@pytest.fixture
def mock_optional_analysis_modules(monkeypatch):
    """Keep API/background-pipeline tests offline and independent of model caches."""
    from backend.app.services.ocr.service import OCRResult
    from backend.app.services.vision.detector import VisionResult

    class MockOCR:
        is_loaded = False
        def load(self): self.is_loaded = True
        def extract(self, image):
            return OCRResult(status="completed", engine="mock-ocr", language_configuration=["en", "hi"])

    class MockVision:
        is_loaded = False
        def load(self, device="auto"): self.is_loaded = True
        def detect(self, image):
            return VisionResult(status="completed", model_name="mock-yolov8n", image_width=image.width, image_height=image.height)

    ocr = MockOCR()
    vision = MockVision()
    monkeypatch.setattr("backend.app.services.ocr.service.get_ocr_service", lambda **kwargs: ocr)
    monkeypatch.setattr("backend.app.services.vision.detector.get_object_detector", lambda: vision)
    return {"ocr": ocr, "vision": vision}


def create_test_jpeg(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal valid JPEG image in memory."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def create_test_png(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal valid PNG image in memory."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(64, 128, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
