"""
TruthLens — Standalone Language Identification API endpoint.

POST /api/v1/nlp/language
"""

from __future__ import annotations

import asyncio
import time
from typing import List

from fastapi import APIRouter, Depends, Request

from backend.app.config import settings
from backend.app.core.auth import Principal, get_current_principal
from backend.app.core.logging import get_logger
from backend.app.core.rate_limit import check_rate_limit
from backend.app.schemas.language import (
    LanguageCandidate,
    LanguageDetectionRequest,
    LanguageDetectionResponse,
)
from backend.app.services.nlp.language_detector import (
    MODEL_ID,
    get_language_detector,
)

log = get_logger("api.language")
router = APIRouter()


@router.post(
    "/nlp/language",
    response_model=LanguageDetectionResponse,
    summary="Identify document-level language of text",
    description=(
        "Performs document-level language identification across 20 supported languages "
        f"using the local XLM-RoBERTa language detection model ({MODEL_ID}). "
        "Supports returning top-k language candidates with softmax confidence scores. "
        "Model loading is local and cache-only (no dynamic weight downloads). "
        "Softmax confidence scores reflect relative classification likelihood, not calibrated veracity."
    ),
)
async def detect_language_endpoint(
    request: LanguageDetectionRequest,
    http_request: Request,
    principal: Principal = Depends(get_current_principal),
) -> LanguageDetectionResponse:
    # 1. Rate limiting
    await check_rate_limit(
        http_request,
        scope="language",
        limit=settings.RATE_LIMIT_LANGUAGE_PER_MINUTE,
        principal=principal,
    )

    start_wall = time.perf_counter()
    detector = get_language_detector()

    # 2. Run detection non-blockingly on worker thread
    top_k = request.top_k or 3
    text = request.text

    try:
        def _run_detect():
            return detector.detect(text=text, top_k=top_k)

        result = await asyncio.to_thread(_run_detect)
    except Exception as exc:
        log.warning("Language detection unhandled exception: %s", type(exc).__name__)
        total_ms = max(0, int((time.perf_counter() - start_wall) * 1000))
        is_load_err = not getattr(detector, "is_loaded", False)
        status_val = "unavailable" if is_load_err else "failed"
        reason_val = "Model initialization failed or dependencies unavailable." if is_load_err else None
        return LanguageDetectionResponse(
            status=status_val,
            outcome=status_val,
            detected_language="unknown",
            language_code=None,
            language_name=None,
            confidence=None,
            top_languages=[],
            top_k_candidates=[],
            model_id=MODEL_ID,
            model_revision=getattr(detector, "model_revision", None) or "9865598389ca9d95637462f743f683b51d75b87b",
            device=str(detector._device) if getattr(detector, "_device", None) else None,
            runtime_ms=total_ms,
            processing_time_ms=total_ms,
            input_characters=len(text),
            input_truncated=False,
            warnings=["Language identification inference could not complete."],
            limitations=["No prediction was substituted."],
            unavailable_reason=reason_val,
        )

    total_ms = max(0, int((time.perf_counter() - start_wall) * 1000))

    # 3. Format candidates
    candidates: List[LanguageCandidate] = [
        LanguageCandidate(
            language_code=c["language"],
            language=c["language"],
            language_name=c["language_name"],
            confidence=c["confidence"],
        )
        for c in (result.top_languages or [])
    ]

    # Map status
    if result.status == "unavailable":
        resp_status = "unavailable"
        unavailable_reason = "Model weights or dependencies are unavailable in local cache."
    elif result.status == "failed":
        resp_status = "failed"
        unavailable_reason = None
    elif result.outcome in ("insufficient_input", "no_text"):
        resp_status = "insufficient_text"
        unavailable_reason = None
    else:
        resp_status = "completed"
        unavailable_reason = None

    device_str = str(detector._device) if getattr(detector, "_device", None) else None

    return LanguageDetectionResponse(
        status=resp_status,
        outcome=result.outcome,
        detected_language=result.detected_language,
        language_code=result.detected_language if result.detected_language != "unknown" else None,
        language_name=result.language_name,
        confidence=result.confidence,
        top_languages=candidates,
        top_k_candidates=candidates,
        model_id=result.model_name or MODEL_ID,
        model_revision=result.model_revision or getattr(detector, "model_revision", None) or "9865598389ca9d95637462f743f683b51d75b87b",
        device=device_str,
        runtime_ms=result.runtime_ms,
        processing_time_ms=total_ms,
        input_characters=result.input_characters or len(text),
        input_truncated=result.input_truncated,
        confidence_semantics=result.confidence_semantics,
        warnings=result.warnings,
        limitations=result.limitations,
        unavailable_reason=unavailable_reason,
    )
