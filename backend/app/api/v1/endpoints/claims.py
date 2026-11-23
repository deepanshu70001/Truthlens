"""
TruthLens — Standalone Claim Extraction API endpoint.

POST /api/v1/claims
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from backend.app.config import settings
from backend.app.core.auth import Principal, get_current_principal
from backend.app.core.logging import get_logger
from backend.app.core.rate_limit import check_rate_limit
from backend.app.schemas.claims import (
    ClaimExtractionRequest,
    ClaimExtractionResponse,
    ClaimResponse,
)
from backend.app.services.claims.extractor import EXTRACTION_METHOD, extract_claims

log = get_logger("api.claims")
router = APIRouter()


@router.post(
    "/claims",
    response_model=ClaimExtractionResponse,
    summary="Extract verifiable factual claims and candidate statements from text",
    description=(
        "Extracts candidate factual assertions, attributed quotations, and statistical propositions "
        "from submitted text or sentences using the deterministic TruthLens claim extraction engine "
        f"({EXTRACTION_METHOD}). Classifies candidate statements as FACTUAL, QUOTE, OPINION, QUESTION, "
        "or SLOGAN, assigns verification eligibility, and computes original character offsets. "
        "This service identifies candidate statements; it does NOT verify whether claims are true or false."
    ),
)
async def claims_endpoint(
    request: ClaimExtractionRequest,
    http_request: Request,
    principal: Principal = Depends(get_current_principal),
) -> ClaimExtractionResponse:
    # 1. Rate limiting
    await check_rate_limit(
        http_request,
        scope="claims",
        limit=settings.RATE_LIMIT_CLAIMS_PER_MINUTE,
        principal=principal,
    )

    # 2. Extract inputs
    source_input = request.text if request.text is not None else (request.sentences or [])
    lang = (request.language_code or request.language or "").strip().lower() or "unknown"
    lang_status = "provided_by_client" if lang != "unknown" else "unavailable"

    # 3. Execute extraction (deterministic CPU execution wrapped in thread for async safety)
    try:
        def _run_extract():
            return extract_claims(
                sentences=source_input,
                min_confidence=request.min_confidence,
                max_claims=request.max_claims,
                language=lang,
                language_detection_status=lang_status,
                ocr_confidence=request.ocr_confidence,
            )

        result = await asyncio.to_thread(_run_extract)
    except Exception as exc:
        log.error("Claim extraction failed unexpectedly: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Claim extraction failed due to an internal error.",
        ) from None

    # 4. Map to response model
    claims_list: List[ClaimResponse] = []
    eligible_count = 0

    for c in result.claims:
        claim_dict = c.to_dict()
        # If request provided custom source and claim defaults to ocr, harmonize
        if request.source and claim_dict.get("source") == "ocr" and request.text is not None:
            claim_dict["source"] = request.source

        claim_model = ClaimResponse(**claim_dict)
        claims_list.append(claim_model)
        if claim_model.verification_eligibility == "ELIGIBLE":
            eligible_count += 1

    method_name = result.method or EXTRACTION_METHOD

    return ClaimExtractionResponse(
        status=result.status,
        claims=claims_list,
        total_claims=len(claims_list),
        candidate_count=len(claims_list),
        accepted_count=eligible_count,
        eligible_count=eligible_count,
        method=method_name,
        extraction_method=method_name,
        runtime_ms=result.runtime_ms,
        processing_time_ms=result.runtime_ms,
        warnings=result.warnings,
    )
