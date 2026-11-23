"""
TruthLens — Pydantic schemas for Standalone Language Identification API.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class LanguageCandidate(BaseModel):
    """Candidate language prediction."""
    language_code: str = Field(..., description="Two-letter ISO 639-1 language code or label")
    language: Optional[str] = Field(None, description="Alias for language_code")
    language_name: str = Field(..., description="Full English name of candidate language")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax confidence score")

    def model_post_init(self, __context: Any) -> None:
        if self.language is None:
            self.language = self.language_code


class LanguageDetectionRequest(BaseModel):
    """Request payload for POST /api/v1/nlp/language."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to analyze for document-level language identification.",
    )
    top_k: Optional[int] = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of top candidate languages to return (1-20).",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Input text cannot be empty or whitespace-only.")
        return v


class LanguageDetectionResponse(BaseModel):
    """Response payload for POST /api/v1/nlp/language."""
    status: Literal["completed", "insufficient_text", "unavailable", "failed"] = Field(
        default="completed",
        description="Execution status: completed, insufficient_text, unavailable, or failed.",
    )
    outcome: str = Field(
        default="undetermined",
        description="Operational outcome: success, undetermined, insufficient_input, no_text, unavailable, failed.",
    )
    detected_language: str = Field(
        default="unknown",
        description="Identified primary language code, or 'unknown'.",
    )
    language_code: Optional[str] = Field(
        default=None,
        description="Primary language code if identified, else None.",
    )
    language_name: Optional[str] = Field(
        default=None,
        description="Full English name of identified primary language, else None.",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score of top prediction (0.0 to 1.0) if confident, else None.",
    )
    top_languages: List[LanguageCandidate] = Field(
        default_factory=list,
        description="List of top candidate languages ranked by softmax score.",
    )
    top_k_candidates: List[LanguageCandidate] = Field(
        default_factory=list,
        description="Alias for top_languages.",
    )
    model_id: str = Field(
        default="papluca/xlm-roberta-base-language-detection",
        description="Underlying model identifier.",
    )
    model_revision: Optional[str] = Field(
        default="9865598389ca9d95637462f743f683b51d75b87b",
        description="Pinned model commit hash or revision.",
    )
    device: Optional[str] = Field(
        default=None,
        description="Inference execution device (e.g., cpu, cuda).",
    )
    runtime_ms: int = Field(
        default=0,
        ge=0,
        description="Inference runtime in milliseconds.",
    )
    processing_time_ms: int = Field(
        default=0,
        ge=0,
        description="Total processing time in milliseconds.",
    )
    input_characters: int = Field(
        default=0,
        ge=0,
        description="Length of input text in characters.",
    )
    input_truncated: bool = Field(
        default=False,
        description="Whether input text exceeded max model token or character limit.",
    )
    confidence_semantics: str = Field(
        default="Maximum softmax probability from the model; not calibrated.",
        description="Interpretation warning for raw softmax scores.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Operational or input warnings.",
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Documented architectural or scope limitations.",
    )
    unavailable_reason: Optional[str] = Field(
        default=None,
        description="Reason why the model or service is unavailable, if status is unavailable.",
    )
