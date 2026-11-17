"""
TruthLens — FastAPI application factory.

Entry point for the backend server.
Run with: uvicorn backend.app.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.core.exceptions import register_exception_handlers
from backend.app.core.logging import get_logger, setup_logging

log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # ── Startup ──────────────────────────────────────────────────
    setup_logging()
    log.info("Starting TruthLens v%s [%s]", settings.APP_VERSION, settings.APP_ENV)

    # Ensure storage directories exist
    settings.ensure_storage_dirs()
    log.info("Storage directory: %s", settings.STORAGE_PATH)

    # Create database tables (dev convenience — use Alembic in production)
    from backend.app.db.session import engine
    from backend.app.db.base import Base
    # Import all models so Base.metadata knows about them
    import backend.app.models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Database tables ensured")

    # Log GPU status
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**2)
            log.info("GPU: %s (%d MB VRAM)", gpu_name, gpu_mem)
        else:
            log.info("GPU: Not available — using CPU")
    except ImportError:
        log.info("PyTorch not installed — GPU detection skipped")

    yield

    # ── Shutdown ─────────────────────────────────────────────────
    log.info("Shutting down TruthLens")
    from backend.app.api.v1.endpoints.analyze import wait_for_analysis_tasks
    await wait_for_analysis_tasks()
    from backend.app.db.session import engine
    await engine.dispose()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        description=(
            "AI-Assisted Image Verification, Analysis and Virality Estimation. "
            "TruthLens is a research tool — it provides probabilistic assessments, "
            "not absolute truth."
        ),
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS ─────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    # ── Exception handlers ───────────────────────────────────────
    register_exception_handlers(app)

    # ── Routers ──────────────────────────────────────────────────
    from backend.app.api.v1.router import api_router
    app.include_router(api_router)

    return app


# The app instance Uvicorn will serve
app = create_app()
