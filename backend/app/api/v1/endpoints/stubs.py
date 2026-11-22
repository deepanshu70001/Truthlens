"""
TruthLens — API endpoint stubs for future stages.

POST /api/v1/ocr
POST /api/v1/fact-check
POST /api/v1/provenance
GET  /api/v1/models
"""

from __future__ import annotations

import asyncio
from typing import Annotated, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, StringConstraints

router = APIRouter()


class FactCheckRequest(BaseModel):
    claims: List[Annotated[str, StringConstraints(strip_whitespace=True, min_length=5, max_length=2000)]] = Field(..., min_length=1, max_length=20)
    language_code: Optional[str] = Field(None, min_length=2, max_length=5, pattern=r"^[a-z]{2,3}(-[A-Z]{2})?$")


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
async def ocr_endpoint():
    return {"detail": "OCR endpoint will be implemented in Stage 4.", "error_code": "NOT_IMPLEMENTED"}


@router.post(
    "/fact-check",
    summary="Standalone fact-check query",
    description="Retrieve existing third-party fact-check reviews for supplied factual claims.",
    response_model=FactCheckResponse,
)
async def factcheck_endpoint(request: FactCheckRequest):
    from backend.app.services.factcheck.service import search_fact_checks

    result = await asyncio.to_thread(
        search_fact_checks, request.claims, language_code=request.language_code,
    )
    return result.to_dict()


@router.post(
    "/provenance",
    summary="Image provenance and content credentials",
    description="Inspect original image bytes for C2PA credentials and search Google Vision Web Detection when configured.",
    response_model=ProvenanceResponse,
)
async def provenance_endpoint(image: UploadFile = File(...)):
    from backend.app.config import settings
    from backend.app.core.security import validate_image_file
    from backend.app.services.provenance.service import search_provenance

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


@router.get(
    "/models",
    summary="List available models",
    description="Returns available ML models. (Will be implemented in Stage 3.)",
)
async def list_models():
    return {
        "models": [],
        "message": "ML models will be available in Stage 3.",
    }
