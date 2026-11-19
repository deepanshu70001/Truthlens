"""
TruthLens — API endpoint: Health checks.

GET /api/v1/health
GET /api/v1/health/models
"""

from __future__ import annotations

import time
import importlib.util
from pathlib import Path

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
        status="healthy" if db_status == "connected" else "degraded",
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
    model_root = Path(settings.MODEL_CACHE_DIR)
    finetuned_backend = settings.AUTHENTICITY_BACKEND.strip().lower() == "finetuned_siglip"
    if finetuned_backend:
        authenticity_model_id = "truthlens/siglip-cifake-frozen-head"
        authenticity_backbone_id = settings.AUTHENTICITY_BACKBONE_MODEL
        authenticity_checkpoint = Path(settings.AUTHENTICITY_CHECKPOINT_PATH)
        authenticity_checkpoint_present = authenticity_checkpoint.is_file()
    else:
        authenticity_model_id = settings.AUTHENTICITY_MODEL
        authenticity_backbone_id = None
        authenticity_checkpoint = None
        authenticity_checkpoint_present = None
    authenticity_cached = (
        model_root / f"models--{authenticity_backbone_id.replace('/', '--')}"
        if authenticity_backbone_id
        else model_root / f"models--{settings.AUTHENTICITY_MODEL.replace('/', '--')}"
    ).exists()
    yolo_path = Path(settings.VISION_MODEL)
    if not yolo_path.is_absolute():
        yolo_path = Path.cwd() / yolo_path
    easyocr_installed = importlib.util.find_spec("easyocr") is not None
    tesseract_installed = importlib.util.find_spec("pytesseract") is not None
    ultralytics_installed = importlib.util.find_spec("ultralytics") is not None
    from backend.app.services.authenticity.detector import get_detector
    from backend.app.services.vision.detector import get_object_detector
    from backend.app.services.ocr.service import get_ocr_service
    auth_loaded = get_detector().is_loaded
    vision_loaded = get_object_detector().is_loaded
    ocr_languages = [code.strip() for code in settings.OCR_LANGUAGES.split(",") if code.strip()]
    ocr_loaded = get_ocr_service(gpu=settings.DEVICE != "cpu", languages=ocr_languages).is_loaded
    authenticity_status = (
        "loaded" if auth_loaded
        else "checkpoint_missing" if finetuned_backend and not authenticity_checkpoint_present
        else "cached_not_loaded" if authenticity_cached
        else "download_required"
    )
    return {
        "status": "model_readiness_report",
        "models": [
            {"name": "authenticity", "model_id": authenticity_model_id,
             "backbone_model_id": authenticity_backbone_id,
             "checkpoint_path": str(authenticity_checkpoint) if authenticity_checkpoint else None,
             "checkpoint_present": authenticity_checkpoint_present,
             "status": authenticity_status,
             "dependency_available": importlib.util.find_spec("transformers") is not None,
             "loaded": auth_loaded},
            {"name": "object_detection", "model_id": settings.VISION_MODEL,
             "status": "loaded" if vision_loaded else "available_not_loaded" if ultralytics_installed and yolo_path.exists() else "unavailable_or_download_required",
             "dependency_available": ultralytics_installed, "checkpoint_present": yolo_path.exists(), "loaded": vision_loaded},
            {"name": "ocr", "model_id": None,
             "status": "loaded" if ocr_loaded else "available_not_loaded" if easyocr_installed or tesseract_installed else "dependency_unavailable",
             "easyocr_available": easyocr_installed, "tesseract_wrapper_available": tesseract_installed,
             "loaded": ocr_loaded},
        ],
    }
