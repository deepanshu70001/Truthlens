"""
TruthLens — Pydantic schemas for the Model Registry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Public, sanitized information regarding a configured ML model or engine."""

    id: str = Field(..., description="Stable unique identifier for the model or service")
    name: str = Field(..., description="Human-readable display name")
    task: str = Field(..., description="Primary capability or task (e.g. authenticity_classification, object_detection)")
    provider: str = Field(..., description="Underlying provider or organization")
    architecture: str = Field(..., description="Model architecture or engine design")
    model_id: Optional[str] = Field(None, description="Public model identifier or repository reference")
    version: Optional[str] = Field(None, description="Model revision, checkpoint digest summary, or engine version")
    configured: bool = Field(True, description="Whether this service is configured in application settings")
    loaded: bool = Field(False, description="Strictly true iff model weights are resident in memory; false otherwise")
    readiness: Literal["ready", "available", "unavailable", "failed"] = Field(
        ...,
        description="High-level operational readiness category: 'ready' (loaded in memory), 'available' (present on disk, ready to load), 'unavailable' (missing dependency or weights), 'failed' (load failed)",
    )
    status: str = Field(
        ...,
        description="Detailed lifecycle status: 'loaded', 'available_not_loaded', 'cached_not_loaded', 'checkpoint_missing', 'dependency_unavailable', 'download_required', 'failed', 'disabled'",
    )
    device: Optional[str] = Field(None, description="Execution device if loaded (e.g. cuda:0, cpu); null if not loaded")
    capabilities: List[str] = Field(default_factory=list, description="Specific capabilities supported by this model")
    details: Dict[str, Any] = Field(default_factory=dict, description="Safe, non-sensitive operational details")
    reason: Optional[str] = Field(None, description="Safe explanatory status reason or explanation when not loaded/unavailable")


class ModelRegistryResponse(BaseModel):
    """Response payload for GET /api/v1/models."""

    status: str = Field("ok", description="Response status ('ok')")
    models: List[ModelInfo] = Field(default_factory=list, description="List of configured models and services")
    total_models: int = Field(0, description="Total number of models reported")
    loaded_count: int = Field(0, description="Number of models currently resident in memory")
    available_count: int = Field(0, description="Number of models verified on disk and available for on-demand loading")
    ready_count: int = Field(0, description="Number of models operational (loaded or ready for immediate on-demand loading)")
    environment: str = Field("development", description="Current application environment")
    message: Optional[str] = Field(None, description="Informational message about the model registry")
