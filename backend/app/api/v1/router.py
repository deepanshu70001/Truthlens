"""
TruthLens — API v1 router.

Aggregates all v1 endpoint routers under /api/v1.
"""

from fastapi import APIRouter

from backend.app.api.v1.endpoints import analyze, analysis, auth, claims, fact_check, health, models, ocr, stubs

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth.router, tags=["Authentication"])
api_router.include_router(analyze.router, tags=["Analysis"])
api_router.include_router(analysis.router, tags=["Analysis"])
api_router.include_router(claims.router, tags=["Claims"])
api_router.include_router(fact_check.router, tags=["Fact-Checking"])
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(models.router, tags=["Models"])
api_router.include_router(ocr.router, tags=["OCR"])
api_router.include_router(stubs.router, tags=["Stubs"])
