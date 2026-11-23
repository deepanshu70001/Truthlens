"""
TruthLens — API endpoint stubs for future stages.

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
from backend.app.api.v1.endpoints.fact_check import factcheck_endpoint
from backend.app.api.v1.endpoints.models import list_models
from backend.app.api.v1.endpoints.ocr import ocr_endpoint

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


__all__ = [
    "FactCheckRequest",
    "FactCheckResponse",
    "ProvenanceResponse",
    "ocr_endpoint",
    "factcheck_endpoint",
    "provenance_endpoint",
    "list_models",
]
