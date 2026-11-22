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

from fastapi import APIRouter
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
    summary="Reverse image search",
    description="Search for image provenance. (Will be implemented in Stage 4.)",
    status_code=501,
)
async def provenance_endpoint():
    return {"detail": "Provenance endpoint will be implemented in Stage 4.", "error_code": "NOT_IMPLEMENTED"}


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
