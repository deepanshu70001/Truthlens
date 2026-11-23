"""
TruthLens — Standalone OCR API endpoint.

POST /api/v1/ocr
"""

from __future__ import annotations

import asyncio
import io
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from PIL import Image

from backend.app.config import settings
from backend.app.core.auth import Principal, get_current_principal
from backend.app.core.logging import get_logger
from backend.app.core.rate_limit import check_rate_limit
from backend.app.core.security import validate_image_file
from backend.app.schemas.ocr import OCRResponse
from backend.app.services.ocr.service import OCRResult

log = get_logger("api.ocr")
router = APIRouter()


@router.post(
    "/ocr",
    response_model=OCRResponse,
    summary="Extract text and spatial bounding boxes from an image",
    description=(
        "Performs optical character recognition (OCR) on an uploaded image using the existing "
        "TruthLens OCR service (EasyOCR primary with PyTesseract fallback). Returns extracted "
        "text, per-region bounding box polygon coordinates, recognition confidence scores, "
        "detected language script, and execution metadata."
    ),
)
async def ocr_endpoint(
    http_request: Request,
    image: UploadFile = File(
        ..., description="Image file (JPEG, PNG, WebP, GIF, BMP, TIFF)"
    ),
    languages: Optional[str] = Query(
        None, description="Comma-separated ISO 639-1 language codes, e.g. 'en,hi'"
    ),
    principal: Principal = Depends(get_current_principal),
) -> OCRResponse:
    # 1. Rate limiting
    await check_rate_limit(
        http_request,
        scope="ocr",
        limit=settings.RATE_LIMIT_OCR_PER_MINUTE,
        principal=principal,
    )

    # 2. Upload size bounding
    image_data = await image.read(settings.max_upload_bytes + 1)
    if len(image_data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the configured upload size limit of {settings.MAX_UPLOAD_SIZE_MB} MB.",
        )

    # 3. Empty file rejection
    if len(image_data) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    # 4. Strict format, MIME, header magic bytes, and file extension validation
    try:
        validate_image_file(image_data, image.filename or "image.jpg", image.content_type)
    except ValueError as exc:
        err_msg = str(exc)
        if "too large" in err_msg.lower():
            raise HTTPException(status_code=413, detail=err_msg) from None
        raise HTTPException(status_code=422, detail=err_msg) from None

    # 5. Image decode verification
    try:
        pil_image = Image.open(io.BytesIO(image_data))
        pil_image.load()
    except Exception:
        raise HTTPException(status_code=422, detail="File could not be decoded as a valid image.") from None

    # 6. Dimension safety limits
    width, height = pil_image.size
    if width <= 0 or height <= 0 or width * height > settings.MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=422, detail="Image dimensions exceed the configured safety limit.")

    # 7. Language configuration resolution
    if languages is not None and languages.strip():
        configured_languages = [c.strip() for c in languages.split(",") if c.strip()]
    else:
        configured_languages = [c.strip() for c in settings.OCR_LANGUAGES.split(",") if c.strip()]

    # 8. Non-blocking OCR execution via worker thread
    def _run_extract() -> OCRResult:
        from backend.app.services.ocr.service import get_ocr_service

        service = get_ocr_service(gpu=settings.DEVICE != "cpu", languages=configured_languages)
        if not service.is_loaded:
            service.load()
        return service.extract(pil_image)

    timeout = max(1.0, min(120.0, float(settings.OCR_TIMEOUT_SECONDS) + 5.0))
    try:
        result = await asyncio.wait_for(asyncio.to_thread(_run_extract), timeout=timeout)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="OCR processing exceeded the configured time limit.",
        ) from None
    except HTTPException:
        raise
    except Exception as exc:
        log.error("Unexpected OCR service failure: %s", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail="OCR service encountered an unexpected error.",
        ) from None

    # 9. Format structured response
    raw = result.to_dict()
    return OCRResponse(
        status=raw.get("status", "completed"),
        text=raw.get("text", "") or "",
        normalized_text=raw.get("normalized_text", "") or "",
        language=raw.get("language", "unknown") or "unknown",
        engine=raw.get("engine"),
        confidence=raw.get("total_confidence"),
        total_confidence=raw.get("total_confidence"),
        has_text=bool(raw.get("has_text", False)),
        detected=bool(raw.get("has_text", False)),
        no_text_reason=raw.get("no_text_reason"),
        word_count=int(raw.get("word_count", 0)),
        processing_time_ms=int(raw.get("runtime_ms", 0)),
        runtime_ms=int(raw.get("runtime_ms", 0)),
        blocks=raw.get("blocks", []),
        results=raw.get("blocks", []),
        language_configuration=raw.get("language_configuration", configured_languages),
        warnings=raw.get("warnings", []),
    )
