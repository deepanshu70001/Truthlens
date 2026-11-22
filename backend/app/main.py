"""
TruthLens — FastAPI application factory.

Entry point for the backend server.
Run with: uvicorn backend.app.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

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
    from backend.app.services.storage_lifecycle import cleanup_abandoned_temp_uploads
    removed_temp_uploads = cleanup_abandoned_temp_uploads()
    if removed_temp_uploads:
        log.info("Removed %d expired upload staging files", removed_temp_uploads)

    settings.validate_runtime_security()
    if settings.APP_ENV.lower() == "production":
        from sqlalchemy import text
        from alembic.config import Config
        from alembic.script import ScriptDirectory
        from backend.app.db.session import async_session_factory
        migration_config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
        expected_heads = set(ScriptDirectory.from_config(migration_config).get_heads())
        try:
            async with async_session_factory() as db:
                actual_revisions = set((await db.scalars(text("SELECT version_num FROM alembic_version"))).all())
                if actual_revisions != expected_heads:
                    raise RuntimeError("Database schema is not at the application migration head.")
        except Exception:
            raise RuntimeError("Database migrations are missing, outdated, or unavailable; apply migrations before starting the API.") from None

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
    from backend.app.db.session import engine
    await engine.dispose()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings.validate_runtime_security()
    app = FastAPI(
        title=settings.APP_NAME,
        description=(
            "AI-Assisted Image Verification, Analysis and Virality Estimation. "
            "TruthLens is a research tool — it provides probabilistic assessments, "
            "not absolute truth."
        ),
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url=None if settings.APP_ENV.lower() == "production" else "/docs",
        redoc_url=None if settings.APP_ENV.lower() == "production" else "/redoc",
        openapi_url=None if settings.APP_ENV.lower() == "production" else "/openapi.json",
    )

    from backend.app.core.middleware import RequestBodyLimitMiddleware, apply_security_headers
    app.add_middleware(
        RequestBodyLimitMiddleware,
        upload_limit=settings.max_upload_bytes,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_host_list)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        apply_security_headers(response)
        if request.url.scheme == "https":
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response

    # ── CORS ─────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
        expose_headers=["Retry-After"],
    )

    # ── Exception handlers ───────────────────────────────────────
    register_exception_handlers(app)

    # ── Routers ──────────────────────────────────────────────────
    from backend.app.api.v1.router import api_router
    app.include_router(api_router)

    return app


# The app instance Uvicorn will serve
app = create_app()
