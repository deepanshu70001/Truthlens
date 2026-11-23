"""
TruthLens — API endpoint: Models registry.

GET /api/v1/models
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from backend.app.config import settings
from backend.app.core.auth import Principal, get_current_principal
from backend.app.core.rate_limit import check_rate_limit
from backend.app.schemas.models import ModelRegistryResponse
from backend.app.services.ml.registry import get_model_registry

router = APIRouter()


@router.get(
    "/models",
    response_model=ModelRegistryResponse,
    summary="List configured ML models and runtime readiness",
    description=(
        "Returns the complete inventory of configured machine learning models and OCR engines, "
        "including display metadata, provider architecture, configured parameters, and current "
        "in-memory readiness status. This endpoint inspects singleton states and filesystem presence "
        "without triggering eager model loading or executing inference passes."
    ),
)
async def list_models(
    http_request: Request,
    principal: Principal = Depends(get_current_principal),
) -> ModelRegistryResponse:
    await check_rate_limit(
        http_request,
        scope="models",
        limit=settings.RATE_LIMIT_MODELS_PER_MINUTE,
        principal=principal,
    )
    return get_model_registry()
