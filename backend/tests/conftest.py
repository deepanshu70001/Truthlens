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
os.environ["DEV_AUTH_BYPASS"] = "true"
os.environ["TRUSTED_HOSTS"] = "localhost,127.0.0.1,testserver,test"
os.environ["RATE_LIMIT_ANALYSIS_PER_MINUTE"] = "500"
os.environ["RATE_LIMIT_FACTCHECK_PER_MINUTE"] = "500"
os.environ["RATE_LIMIT_PROVENANCE_PER_MINUTE"] = "500"
os.environ["RATE_LIMIT_AUTH_PER_MINUTE"] = "500"
os.environ["RATE_LIMIT_RETRIEVAL_PER_MINUTE"] = "1000"
os.environ["RATE_LIMIT_MODELS_PER_MINUTE"] = "500"
os.environ["RATE_LIMIT_OCR_PER_MINUTE"] = "500"
os.environ["RATE_LIMIT_CLAIMS_PER_MINUTE"] = "500"
os.environ["RATE_LIMIT_LANGUAGE_PER_MINUTE"] = "500"

from backend.app.main import app  # noqa: E402
from backend.app.db.session import engine  # noqa: E402
from backend.app.db.session import async_session_factory  # noqa: E402
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
    from sqlalchemy import select
    from backend.app.models import QueueCapacity, WorkerLease
    from backend.app.db.session import async_session_factory
    async with async_session_factory() as db:
        if await db.get(QueueCapacity, 1) is None:
            db.add(QueueCapacity(id=1, outstanding_count=0))
        if await db.get(WorkerLease, "analysis-pipeline") is None:
            db.add(WorkerLease(name="analysis-pipeline"))
        await db.commit()
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


@pytest.fixture(autouse=True)
def reset_factcheck_cache():
    """Ensure factcheck query cache isolation between test runs."""
    try:
        from backend.app.services.factcheck.cache import get_factcheck_cache
        cache = get_factcheck_cache()
        cache.clear()
    except Exception:
        pass
    yield
    try:
        from backend.app.services.factcheck.cache import get_factcheck_cache
        cache = get_factcheck_cache()
        cache.clear()
    except Exception:
        pass


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
    """API client plus an explicitly separate deterministic test worker."""
    transport = ASGITransport(app=app)
    from backend.app.services.job_queue import run_worker
    stop_event = asyncio.Event()
    worker_task = asyncio.create_task(run_worker(async_session_factory, stop_event=stop_event))
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    stop_event.set()
    await worker_task


@pytest_asyncio.fixture
async def client_without_worker(mock_auth_detector, mock_optional_analysis_modules):
    """Client for tests that assert durable submission before any worker claims it."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def await_analysis_terminal(client, analysis_id: str, *, timeout: float = 30.0):
    """Poll the durable status API until a terminal analysis state is persisted."""
    deadline = asyncio.get_running_loop().time() + timeout
    last_response = None
    while asyncio.get_running_loop().time() < deadline:
        last_response = await client.get(f"/api/v1/analysis/{analysis_id}")
        if last_response.status_code == 200 and last_response.json().get("status") in {
            "completed", "partial", "failed"
        }:
            return last_response
        await asyncio.sleep(0.05)
    raise AssertionError(
        f"Analysis {analysis_id} did not reach a terminal status; last response: "
        f"{last_response.status_code if last_response else 'none'}"
    )


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

    class MockLanguageDetector:
        received_text = None
        def detect(self, text, **_kwargs):
            from backend.app.services.nlp.language_detector import LanguageDetectionResult
            self.received_text = text
            if not text:
                return LanguageDetectionResult(status="completed", outcome="no_text")
            return LanguageDetectionResult(
                status="completed", outcome="success", detected_language="en",
                language_name="English", confidence=0.96,
            )

    ocr = MockOCR()
    vision = MockVision()
    language = MockLanguageDetector()
    monkeypatch.setattr("backend.app.services.ocr.service.get_ocr_service", lambda **kwargs: ocr)
    monkeypatch.setattr("backend.app.services.vision.detector.get_object_detector", lambda: vision)
    monkeypatch.setattr("backend.app.services.nlp.language_detector.get_language_detector", lambda: language)
    return {"ocr": ocr, "vision": vision, "language": language}


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
