"""
TruthLens — Pydantic schemas for API request/response validation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Analysis ─────────────────────────────────────────────────────

class AnalysisCreate(BaseModel):
    """Returned immediately after image upload."""
    analysis_id: str
    status: str = "queued"


class ImageAssetOut(BaseModel):
    """Public view of an uploaded image's metadata."""
    id: str
    original_filename: str
    sha256_hash: str
    perceptual_hash: Optional[str] = None
    mime_type: str
    file_size_bytes: int
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ModelRunOut(BaseModel):
    """Public view of a single model inference run."""
    id: str
    model_id: str
    model_version: Optional[str] = None
    task: str
    device: str
    confidence: Optional[float] = None
    runtime_ms: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class AnalysisProgress(BaseModel):
    """Tracks pipeline progress."""
    current_step: Optional[str] = None
    completed_steps: List[str] = Field(default_factory=list)
    total_steps: int = 0
    percentage: float = 0.0


class AnalysisOut(BaseModel):
    """Full analysis response."""
    analysis_id: str
    status: str
    progress: AnalysisProgress
    truth_score: Optional[float] = None
    truth_confidence: Optional[float] = None
    truth_label: Optional[str] = None
    image: Optional[ImageAssetOut] = None
    model_runs: List[ModelRunOut] = Field(default_factory=list)
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    caveat: str = (
        "This is an AI-assisted probabilistic assessment, not proof. "
        "Model predictions can be wrong."
    )

    model_config = {"from_attributes": True}


# ── Health ───────────────────────────────────────────────────────

class GPUInfo(BaseModel):
    available: bool = False
    name: Optional[str] = None
    memory_total_mb: Optional[int] = None
    memory_used_mb: Optional[int] = None


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    environment: str
    gpu: GPUInfo
    database: str = "unknown"
    uptime_seconds: float = 0.0


# ── Error ────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
