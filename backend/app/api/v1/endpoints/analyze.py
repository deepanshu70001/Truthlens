"""
TruthLens — API endpoint: Image analysis (upload + start).

POST /api/v1/analyze
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

import imagehash
from fastapi import APIRouter, Depends, File, UploadFile
from PIL import Image as PILImage
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.app.core.exceptions import ImageValidationError, StorageError
from backend.app.core.logging import get_logger
from backend.app.core.security import (
    compute_sha256,
    generate_storage_filename,
    validate_image_file,
)
from backend.app.db.session import get_db
from backend.app.models import Analysis, AnalysisStatus, ImageAsset, SystemEvent
from backend.app.schemas import AnalysisCreate
from backend.app.services.preprocessing.service import preprocess_image

log = get_logger("api.analyze")
router = APIRouter()
_analysis_tasks: set[asyncio.Task] = set()


async def wait_for_analysis_tasks() -> None:
    """Wait until all in-process analysis tasks have finished."""
    while _analysis_tasks:
        await asyncio.gather(*tuple(_analysis_tasks), return_exceptions=True)


def _track_analysis_task(task: asyncio.Task) -> None:
    _analysis_tasks.add(task)

    def _finished(done_task: asyncio.Task) -> None:
        _analysis_tasks.discard(done_task)
        if not done_task.cancelled():
            error = done_task.exception()
            if error is not None:
                log.error("Analysis background task ended unexpectedly (%s)", type(error).__name__)

    task.add_done_callback(_finished)


@router.post(
    "/analyze",
    response_model=AnalysisCreate,
    status_code=201,
    summary="Submit image for analysis",
    description=(
        "Upload an image file for comprehensive AI-assisted verification. "
        "Returns an analysis ID to track progress."
    ),
)
async def create_analysis(
    image: UploadFile = File(
        ..., description="Image file (JPEG, PNG, WebP, GIF, BMP, TIFF)"
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Full image upload and analysis creation endpoint.

    Pipeline:
    1. Read file bytes
    2. Validate (MIME, magic bytes, size, Pillow decode)
    3. Sanitize filename → generate UUID storage name
    4. Compute SHA-256 and perceptual hash
    5. Save original to storage
    6. Run preprocessing (thumbnail, resized, quality stats)
    7. Create Analysis + ImageAsset DB records
    8. Kick off analysis pipeline (async background)
    9. Return analysis_id
    """
    # ── 1. Read file data ────────────────────────────────────────
    try:
        data = await image.read(settings.max_upload_bytes + 1)
    except Exception:
        log.exception("Failed to read uploaded file")
        raise ImageValidationError("Failed to read the uploaded file.")

    # Reject oversized input before buffering the remainder into memory.
    if len(data) > settings.max_upload_bytes:
        raise ImageValidationError(
            f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE_MB} MB."
        )

    # ── 2. Validate ──────────────────────────────────────────────
    try:
        safe_filename, detected_mime = await asyncio.to_thread(
            validate_image_file, data, image.filename or "unnamed", image.content_type
        )
    except ValueError as e:
        raise ImageValidationError(str(e))

    # ── 3. Generate storage filename ─────────────────────────────
    storage_filename = generate_storage_filename(safe_filename)
    analysis_id = str(uuid.uuid4())

    # ── 4. Compute hashes ────────────────────────────────────────
    sha256 = compute_sha256(data)

    def inspect_validated_image():
        import io
        with PILImage.open(io.BytesIO(data)) as pil_img:
            return pil_img.size, pil_img.format or "UNKNOWN", str(imagehash.phash(pil_img))

    try:
        (width, height), img_format, perceptual_hash_str = await asyncio.to_thread(inspect_validated_image)
    except Exception as e:
        log.warning("Image metadata/hash extraction failed (%s)", type(e).__name__)
        width, height, img_format, perceptual_hash_str = None, None, "UNKNOWN", None

    # ── 5. Save original ─────────────────────────────────────────
    settings.ensure_storage_dirs()
    original_path = settings.originals_dir / storage_filename

    try:
        await asyncio.to_thread(original_path.write_bytes, data)
    except Exception:
        log.exception("Failed to store uploaded image for analysis %s", analysis_id)
        raise StorageError("Failed to store the uploaded image.")

    # ── 7. Run preprocessing ─────────────────────────────────────
    preprocessing_started = time.perf_counter()
    preprocessing_info = {"status": "failed", "warnings": ["Preprocessing did not complete."]}
    try:
        prep_result = await asyncio.to_thread(
            preprocess_image, data, analysis_id, settings.thumbnails_dir, settings.processed_dir
        )
        thumbnail_path = prep_result.thumbnail_path
        preprocessing_info = {
            "status": "completed", "runtime_ms": int((time.perf_counter() - preprocessing_started) * 1000),
            "original_width": prep_result.original_width, "original_height": prep_result.original_height,
            "format": prep_result.format, "has_exif": prep_result.has_exif,
            "orientation_corrected": prep_result.orientation_corrected,
            "quality_stats": prep_result.quality_stats,
            "thumbnail_available": bool(prep_result.thumbnail_path),
            "resized_variants": sorted(prep_result.resized_paths.keys()), "warnings": [],
        }
    except Exception as e:
        preprocessing_info["runtime_ms"] = int((time.perf_counter() - preprocessing_started) * 1000)
        log.warning("Preprocessing failed (non-fatal, %s)", type(e).__name__)
        thumbnail_path = None

    # ── 8. Create DB records ─────────────────────────────────────
    analysis = Analysis(
        id=analysis_id,
        status=AnalysisStatus.QUEUED,
        completed_steps=json.dumps([]),
    )
    db.add(analysis)

    image_asset = ImageAsset(
        analysis_id=analysis_id,
        original_filename=safe_filename,
        storage_filename=storage_filename,
        storage_path=str(original_path),
        sha256_hash=sha256,
        perceptual_hash=perceptual_hash_str,
        mime_type=detected_mime,
        file_size_bytes=len(data),
        width=width,
        height=height,
        format=img_format,
        thumbnail_path=thumbnail_path,
    )
    db.add(image_asset)

    event = SystemEvent(
        analysis_id=analysis_id,
        event_type="analysis_created",
        severity="info",
        source="api.analyze",
        message=f"Analysis created ({detected_mime}, {len(data)} bytes).",
    )
    db.add(event)
    db.add(SystemEvent(
        analysis_id=analysis_id,
        event_type="preprocessing",
        severity="info" if preprocessing_info["status"] == "completed" else "warning",
        source="upload.preprocess",
        message="Image preprocessing completed." if preprocessing_info["status"] == "completed" else "Image preprocessing failed; remaining analysis will continue.",
        metadata_json=json.dumps(preprocessing_info),
    ))

    await db.commit()

    # ── 9. Kick off background pipeline (stub) ───────────────────
    from backend.app.services.orchestrator import run_analysis_pipeline
    from backend.app.db.session import async_session_factory

    async def _run_pipeline():
        async with async_session_factory() as pipeline_db:
            try:
                await run_analysis_pipeline(analysis_id, pipeline_db)
            except Exception as e:
                log.error("Background pipeline error: %s", e)

    _track_analysis_task(asyncio.create_task(_run_pipeline()))

    log.info(
        "Analysis created: id=%s mime=%s size=%d",
        analysis_id, detected_mime, len(data),
    )

    return AnalysisCreate(analysis_id=analysis_id, status="queued")
