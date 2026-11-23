"""
TruthLens — API endpoint stubs for future stages.

POST /api/v1/ocr
POST /api/v1/fact-check
POST /api/v1/provenance
"""

from __future__ import annotations

import asyncio
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, StringConstraints, field_validator

from backend.app.config import settings
from backend.app.core.auth import Principal, get_current_principal
from backend.app.core.rate_limit import check_rate_limit

router = APIRouter()


class FactCheckRequest(BaseModel):
    claims: List[Annotated[str, StringConstraints(strip_whitespace=True, min_length=5, max_length=2000)]] = Field(..., min_length=1, max_length=20)
    language_code: Optional[str] = Field(None, min_length=2, max_length=5, pattern=r"^[a-z]{2,3}(-[A-Z]{2})?$")

    @field_validator("language_code")
    @classmethod
    def language_code_must_be_supported(cls, value):
        if value is None:
            return value
        from backend.app.services.nlp.language_detector import LANGUAGE_NAMES

        if value not in LANGUAGE_NAMES:
            raise ValueError("Unsupported fact-check language code.")
        return value


class FactCheckResponse(BaseModel):
    status: str
    provider: Optional[str] = None
    claims: list
    total_results: int
    queries_performed: int
    runtime_ms: int
    error: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    claim_verifications: list = Field(default_factory=list)
    note: Optional[str] = None


class ProvenanceResponse(BaseModel):
    status: str
    outcome: Optional[str] = None
    provider: Optional[str] = None
    image_integrity: Optional[dict] = None
    metadata_status: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    content_credentials: dict = Field(default_factory=dict)
    web_detection: dict = Field(default_factory=dict)
    web_entities: list = Field(default_factory=list)
    matching_pages: list = Field(default_factory=list)
    matching_images: list = Field(default_factory=list)
    similar_images: list = Field(default_factory=list)
    best_guess_labels: list = Field(default_factory=list)
    total_matches: int = 0
    runtime_ms: int = 0
    warnings: List[str] = Field(default_factory=list)
    note: Optional[str] = None


@router.post(
    "/ocr",
    summary="Standalone OCR extraction",
    description="Extract text from an image. (Will be implemented in Stage 4.)",
    status_code=501,
)
async def ocr_endpoint(principal: Principal = Depends(get_current_principal)):
    return {"detail": "OCR endpoint will be implemented in Stage 4.", "error_code": "NOT_IMPLEMENTED"}


@router.post(
    "/fact-check",
    summary="Standalone fact-check query",
    description="Retrieve existing third-party fact-check reviews for supplied factual claims.",
    response_model=FactCheckResponse,
)
async def factcheck_endpoint(
    request: FactCheckRequest,
    http_request: Request,
    principal: Principal = Depends(get_current_principal),
):
    from backend.app.services.claims.extractor import extract_claims
    from backend.app.services.factcheck.service import FactCheckResult, search_fact_checks

    await check_rate_limit(
        http_request, scope="fact-check", limit=settings.RATE_LIMIT_FACTCHECK_PER_MINUTE,
        principal=principal,
    )
    prepared = []
    eligible = []
    for request_index, raw_claim in enumerate(request.claims):
        # Treat each API list entry as one claim, even if it contains periods;
        # this avoids silently checking only the first sentence of user input.
        extraction = extract_claims([raw_claim], max_claims=1)
        claim = extraction.claims[0] if extraction.claims else None
        if claim is None:
            prepared.append({
                "request_index": request_index,
                "original_claim": raw_claim,
                "normalized_claim": " ".join(raw_claim.split()),
                "verification_status": "SKIPPED",
                "verdict": "UNVERIFIABLE",
                "eligibility_status": "INELIGIBLE",
                "eligibility_reason": "No usable claim text was produced.",
                "confidence": None,
            })
            continue
        entry = {"request_index": request_index, "claim": claim}
        prepared.append(entry)
        if claim.verification_eligibility == "ELIGIBLE":
            eligible.append(entry)
        else:
            entry["ineligible_result"] = {
                "request_index": request_index,
                "claim_id": claim.claim_id,
                "original_claim": claim.original_text,
                "normalized_claim": claim.normalized_text,
                "verification_status": "SKIPPED",
                "verdict": "UNVERIFIABLE",
                "eligibility_status": "INELIGIBLE",
                "eligibility_reason": claim.eligibility_reason,
                "confidence": None,
                "limitations": ["No provider lookup was performed for this ineligible claim."],
            }

    if eligible:
        result = await asyncio.to_thread(
            search_fact_checks,
            [entry["claim"].normalized_text for entry in eligible],
            language_code=request.language_code,
            claim_ids=[entry["claim"].claim_id for entry in eligible],
        )
    else:
        result = FactCheckResult(
            status="skipped",
            warnings=["No eligible claims were submitted; the provider was not queried."],
        )

    by_claim_id = {
        item.get("claim_id"): item
        for item in result.claim_verifications
        if isinstance(item, dict) and item.get("claim_id")
    }
    ordered_verifications = []
    for entry in prepared:
        if "ineligible_result" in entry:
            ordered_verifications.append(entry["ineligible_result"])
            continue
        if "claim" not in entry:
            ordered_verifications.append(entry)
            continue
        claim = entry["claim"]
        verification = dict(by_claim_id.get(claim.claim_id, {
            "claim_id": claim.claim_id,
            "original_claim": claim.original_text,
            "verdict": "INSUFFICIENT_EVIDENCE" if result.status == "no_results" else "RETRIEVAL_UNAVAILABLE",
            "verification_status": "COMPLETED" if result.status == "no_results" else "UNAVAILABLE",
            "confidence": None,
        }))
        verification.update({
            "request_index": entry["request_index"],
            "eligibility_status": "ELIGIBLE",
            "eligibility_reason": claim.eligibility_reason,
        })
        ordered_verifications.append(verification)
    result.claim_verifications = ordered_verifications
    ineligible_count = sum("ineligible_result" in entry or "claim" not in entry for entry in prepared)
    if ineligible_count:
        result.warnings.append(f"{ineligible_count} ineligible claim(s) were skipped before provider submission.")
    return result.to_dict()


@router.post(
    "/provenance",
    summary="Image provenance and content credentials",
    description="Inspect original image bytes for C2PA credentials and search Google Vision Web Detection when configured.",
    response_model=ProvenanceResponse,
)
async def provenance_endpoint(
    http_request: Request,
    image: UploadFile = File(...),
    principal: Principal = Depends(get_current_principal),
):
    from backend.app.core.security import validate_image_file
    from backend.app.services.provenance.service import search_provenance

    await check_rate_limit(
        http_request, scope="provenance", limit=settings.RATE_LIMIT_PROVENANCE_PER_MINUTE,
        principal=principal,
    )
    image_data = await image.read(settings.max_upload_bytes + 1)
    if len(image_data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File exceeds the configured upload size limit.")
    try:
        validate_image_file(image_data, image.filename or "image", image.content_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    timeout = max(1.0, min(125.0, float(settings.PROVENANCE_API_TIMEOUT_SECONDS) + 5.0))
    try:
        result = await asyncio.wait_for(asyncio.to_thread(search_provenance, image_data), timeout=timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Image provenance checks exceeded the configured time limit.") from None
    return result.to_dict()


# Re-exported from dedicated models endpoint module for backward compatibility
from backend.app.api.v1.endpoints.models import list_models
