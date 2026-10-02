"""
TruthLens — Test configuration and fixtures.
"""

import io
import os
import sys
import asyncio
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# Ensure we use SQLite for tests
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_truthlens.db"
os.environ["STORAGE_PATH"] = str(Path(__file__).parent.parent.parent / "test_data")
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


@pytest_asyncio.fixture
async def client():
    """Async HTTP test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


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
