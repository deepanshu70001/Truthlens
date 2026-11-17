"""
TruthLens — Custom exception types and FastAPI exception handlers.

All exceptions return structured JSON. No stack traces leak in production.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class TruthLensError(Exception):
    """Base exception for TruthLens application errors."""

    def __init__(self, detail: str, status_code: int = 500, error_code: str | None = None):
        self.detail = detail
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(detail)


class ImageValidationError(TruthLensError):
    """Raised when image upload validation fails."""

    def __init__(self, detail: str):
        super().__init__(detail=detail, status_code=422, error_code="IMAGE_VALIDATION_ERROR")


class AnalysisNotFoundError(TruthLensError):
    """Raised when an analysis ID is not found."""

    def __init__(self, analysis_id: str):
        super().__init__(
            detail=f"Analysis '{analysis_id}' not found.",
            status_code=404,
            error_code="ANALYSIS_NOT_FOUND",
        )


class StorageError(TruthLensError):
    """Raised when file storage operations fail."""

    def __init__(self, detail: str):
        super().__init__(detail=detail, status_code=500, error_code="STORAGE_ERROR")


def register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers on the FastAPI application."""

    @app.exception_handler(TruthLensError)
    async def truthlens_error_handler(request: Request, exc: TruthLensError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "error_code": exc.error_code,
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=422,
            content={
                "detail": str(exc),
                "error_code": "VALIDATION_ERROR",
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        from backend.app.core.logging import get_logger

        get_logger("api.errors").exception("Unhandled API exception")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An internal error occurred.",
                "error_code": "INTERNAL_ERROR",
            },
        )
