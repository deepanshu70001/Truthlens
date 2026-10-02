"""
TruthLens — API endpoint: Health checks.

GET /api/v1/health
GET /api/v1/health/models
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.app.db.session import get_db
from backend.app.schemas import GPUInfo, HealthResponse

router = APIRouter()

_start_time = time.time()


def _get_gpu_info() -> GPUInfo:
    """Detect GPU availability and specs."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            used = torch.cuda.memory_allocated(0) // (1024 * 1024)
            return GPUInfo(
                available=True, name=name, memory_total_mb=total, memory_used_mb=used
            )
    except ImportError:
        pass
    return GPUInfo(available=False)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    description="Returns system health status including GPU, database, and uptime.",
)
async def health_check(db: AsyncSession = Depends(get_db)):
    gpu = _get_gpu_info()

    # Test DB connection
    db_status = "unknown"
    try:
        await db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        gpu=gpu,
        database=db_status,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@router.get(
    "/health/models",
    summary="Model health status",
    description="Returns loaded/available model status. (Stub — models loaded in Stage 3.)",
)
async def model_health():
    return {
        "status": "no_models_loaded",
        "message": "ML models will be loaded in Stage 3.",
        "models": [],
    }
