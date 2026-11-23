"""
TruthLens — API endpoint: Image analysis (upload + start).

POST /api/v1/analyze
"""

from __future__ import annotations

import json
import asyncio
import os
import re
import time
import uuid

import imagehash
from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile
from PIL import Image as PILImage
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.app.core.exceptions import ImageValidationError, StorageError
from backend.app.core.logging import get_logger
from backend.app.core.security import (
    compute_sha256,
    generate_storage_filename,
    validate_image_file,
)
from backend.app.core.auth import Principal, get_current_principal
from backend.app.core.rate_limit import check_rate_limit
from backend.app.db.session import get_db
from backend.app.models import Analysis, AnalysisJob, AnalysisStatus, ImageAsset, SystemEvent
from backend.app.schemas import AnalysisCreate
from backend.app.services.job_queue import reserve_job_capacity
from backend.app.services.preprocessing.service import preprocess_image
from backend.app.services.storage_lifecycle import cleanup_uncommitted_upload

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
    request: Request,
    image: UploadFile = File(
        ..., description="Image file (JPEG, PNG, WebP, GIF, BMP, TIFF)"
    ),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", min_length=1, max_length=128),
    db: AsyncSession = Depends(get_db),
    principal: Principal = Depends(get_current_principal),
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
    await check_rate_limit(
        request, scope="analysis-submit", limit=settings.RATE_LIMIT_ANALYSIS_PER_MINUTE,
        principal=principal,
    )
    if idempotency_key is not None and not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", idempotency_key):
        raise HTTPException(status_code=422, detail="Idempotency-Key contains unsupported characters.")

    try:
        data = await image.read(settings.max_upload_bytes + 1)
    except Exception as exc:
        log.error("Failed to read uploaded file (%s)", type(exc).__name__)
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

    if idempotency_key:
        existing_job = await db.scalar(select(AnalysisJob).where(
            AnalysisJob.owner_id == principal.subject,
            AnalysisJob.idempotency_key == idempotency_key,
        ))
        if existing_job is not None:
            existing_analysis = await db.get(Analysis, existing_job.analysis_id)
            if existing_job.payload_sha256 != sha256:
                raise HTTPException(status_code=409, detail="Idempotency-Key was already used for a different upload.")
            if existing_analysis is None:
                raise HTTPException(status_code=503, detail="The prior submission record is unavailable.")
            status_value = existing_analysis.status.value if hasattr(existing_analysis.status, "value") else existing_analysis.status
            if status_value == "completed" and existing_analysis.report_json:
                try:
                    if json.loads(existing_analysis.report_json).get("status") == "partial":
                        status_value = "partial"
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass
            return AnalysisCreate(analysis_id=existing_job.analysis_id, job_id=existing_job.analysis_id, status=status_value)

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
    staged_path = settings.temp_uploads_dir / f"{uuid.uuid4().hex}.uploading"

    try:
        await asyncio.to_thread(staged_path.write_bytes, data)
        await asyncio.to_thread(os.replace, staged_path, original_path)
    except Exception as exc:
        cleanup_uncommitted_upload(analysis_id, staged_path)
        log.error("Failed to store uploaded image for analysis %s (%s)", analysis_id, type(exc).__name__)
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
        owner_id=principal.subject,
        status=AnalysisStatus.QUEUED,
        completed_steps=json.dumps([]),
    )
    db.add(analysis)

    if not await reserve_job_capacity(db):
        await db.rollback()
        cleanup_uncommitted_upload(analysis_id)
        raise HTTPException(
            status_code=429,
            detail="Analysis capacity is full. Retry after queued work completes.",
            headers={"Retry-After": "5"},
        )

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
    db.add(AnalysisJob(
        analysis_id=analysis_id,
        owner_id=principal.subject,
        idempotency_key=idempotency_key,
        payload_sha256=sha256,
        state="queued",
        attempt_count=0,
        max_attempts=settings.ANALYSIS_JOB_MAX_ATTEMPTS,
    ))

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

    try:
        await db.commit()
    except Exception as exc:
        await db.rollback()
        cleanup_uncommitted_upload(analysis_id)
        if isinstance(exc, IntegrityError) and idempotency_key:
            existing_job = await db.scalar(select(AnalysisJob).where(
                AnalysisJob.owner_id == principal.subject,
                AnalysisJob.idempotency_key == idempotency_key,
            ))
            if existing_job is not None and existing_job.payload_sha256 == sha256:
                existing_analysis = await db.get(Analysis, existing_job.analysis_id)
                if existing_analysis is not None:
                    existing_status = existing_analysis.status.value if hasattr(existing_analysis.status, "value") else existing_analysis.status
                    return AnalysisCreate(analysis_id=existing_job.analysis_id, job_id=existing_job.analysis_id, status=existing_status)
            if existing_job is not None:
                raise HTTPException(status_code=409, detail="Idempotency-Key was already used for a different upload.") from None
        log.error("Durable analysis submission failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Analysis submission could not be persisted. Please retry.") from None

    log.info(
        "Analysis created: id=%s mime=%s size=%d",
        analysis_id, detected_mime, len(data),
    )

    return AnalysisCreate(analysis_id=analysis_id, job_id=analysis_id, status="queued")
