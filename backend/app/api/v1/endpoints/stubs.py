"""
TruthLens — API endpoint stubs for future stages.

POST /api/v1/ocr
POST /api/v1/fact-check
POST /api/v1/provenance
GET  /api/v1/models
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


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
    description="Check claims against fact-check databases. (Will be implemented in Stage 4.)",
    status_code=501,
)
async def factcheck_endpoint():
    return {"detail": "Fact-check endpoint will be implemented in Stage 4.", "error_code": "NOT_IMPLEMENTED"}


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
