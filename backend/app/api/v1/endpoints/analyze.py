"""
TruthLens — API endpoint: Image analysis (upload + start).

POST /api/v1/analyze
"""

from __future__ import annotations

import json
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
        data = await image.read()
    except Exception as e:
        raise ImageValidationError(f"Failed to read uploaded file: {e}")

    # ── 2. Validate ──────────────────────────────────────────────
    try:
        safe_filename, detected_mime = validate_image_file(
            data=data,
            filename=image.filename or "unnamed",
            content_type=image.content_type,
        )
    except ValueError as e:
        raise ImageValidationError(str(e))

    # ── 3. Generate storage filename ─────────────────────────────
    storage_filename = generate_storage_filename(safe_filename)
    analysis_id = str(uuid.uuid4())

    # ── 4. Compute hashes ────────────────────────────────────────
    sha256 = compute_sha256(data)

    perceptual_hash_str = None
    try:
        import io
        pil_img = PILImage.open(io.BytesIO(data))
        phash = imagehash.phash(pil_img)
        perceptual_hash_str = str(phash)
    except Exception as e:
        log.warning("Perceptual hash computation failed: %s", e)

    # ── 5. Save original ─────────────────────────────────────────
    settings.ensure_storage_dirs()
    original_path = settings.originals_dir / storage_filename

    try:
        original_path.write_bytes(data)
    except Exception as e:
        raise StorageError(f"Failed to save uploaded file: {e}")

    # ── 6. Get dimensions ────────────────────────────────────────
    import io
    try:
        pil_img = PILImage.open(io.BytesIO(data))
        width, height = pil_img.size
        img_format = pil_img.format or "UNKNOWN"
    except Exception:
        width, height, img_format = None, None, "UNKNOWN"

    # ── 7. Run preprocessing ─────────────────────────────────────
    try:
        prep_result = preprocess_image(
            image_data=data,
            analysis_id=analysis_id,
            thumbnails_dir=settings.thumbnails_dir,
            processed_dir=settings.processed_dir,
        )
        thumbnail_path = prep_result.thumbnail_path
    except Exception as e:
        log.warning("Preprocessing failed (non-fatal): %s", e)
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
        message=f"Analysis created: {safe_filename} ({detected_mime}, {len(data)} bytes)",
    )
    db.add(event)

    await db.commit()

    # ── 9. Kick off background pipeline (stub) ───────────────────
    import asyncio
    from backend.app.services.orchestrator import run_analysis_pipeline
    from backend.app.db.session import async_session_factory

    async def _run_pipeline():
        async with async_session_factory() as pipeline_db:
            try:
                await run_analysis_pipeline(analysis_id, pipeline_db)
            except Exception as e:
                log.error("Background pipeline error: %s", e)

    asyncio.create_task(_run_pipeline())

    log.info(
        "Analysis created: id=%s file=%s mime=%s size=%d sha256=%s",
        analysis_id, safe_filename, detected_mime, len(data), sha256[:16],
    )

    return AnalysisCreate(analysis_id=analysis_id, status="queued")
