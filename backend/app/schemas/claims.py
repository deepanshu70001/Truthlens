"""
TruthLens — Pydantic schemas for the Standalone Claim Extraction endpoint.

POST /api/v1/claims
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class ClaimExtractionRequest(BaseModel):
    """Request payload for POST /api/v1/claims."""

    text: Optional[str] = Field(
        None,
        description="Full text string to extract claims from (enables sentence splitting and character offset tracking)",
    )
    sentences: Optional[List[str]] = Field(
        None,
        description="Pre-split sentence list to extract claims from",
    )
    language: Optional[str] = Field(
        None,
        description="Optional ISO 639-1 language code (e.g. 'en', 'hi')",
    )
    language_code: Optional[str] = Field(
        None,
        description="Alias for language parameter",
    )
    source: Optional[str] = Field(
        "text",
        description="Source modality of text ('text', 'ocr', 'user')",
    )
    ocr_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional upstream OCR confidence score between 0.0 and 1.0",
    )
    max_claims: int = Field(
        5,
        ge=1,
        le=50,
        description="Maximum number of candidate claims to extract (clamped between 1 and 50)",
    )
    min_confidence: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="Minimum heuristic extraction confidence threshold",
    )

    @model_validator(mode="after")
    def validate_inputs_and_language(self) -> ClaimExtractionRequest:
        from backend.app.config import settings
        from backend.app.services.nlp.language_detector import LANGUAGE_NAMES

        # 1. Ensure at least one input source is provided
        has_text = self.text is not None
        has_sentences = self.sentences is not None

        if not has_text and not has_sentences:
            raise ValueError("At least one of 'text' or 'sentences' must be provided.")

        # 2. Validate text if provided
        if has_text:
            cleaned = self.text.strip()
            if not cleaned:
                raise ValueError("Text input must not be empty or whitespace-only.")
            max_len = getattr(settings, "MAX_CLAIM_TEXT_LENGTH", 10000)
            if len(self.text) > max_len:
                raise ValueError(
                    f"Text length ({len(self.text)} characters) exceeds maximum allowed length of {max_len}."
                )

        # 3. Validate sentences if provided
        if has_sentences:
            if len(self.sentences) == 0:
                raise ValueError("Sentences list must not be empty.")
            if len(self.sentences) > 100:
                raise ValueError(
                    f"Sentences list length ({len(self.sentences)}) exceeds maximum allowed count of 100."
                )
            non_empty = [s for s in self.sentences if isinstance(s, str) and s.strip()]
            if not non_empty:
                raise ValueError("Sentences list must contain at least one non-empty string.")

        # 4. Validate language / language_code if provided
        lang = self.language_code or self.language
        if lang is not None:
            normalized_lang = lang.strip().lower()
            if normalized_lang not in LANGUAGE_NAMES:
                supported = ", ".join(sorted(LANGUAGE_NAMES.keys()))
                raise ValueError(
                    f"Unsupported language code '{lang}'. Supported language codes: {supported}"
                )

        return self


class ClaimResponse(BaseModel):
    """Individual extracted claim candidate with offsets, eligibility, and metadata."""

    claim_id: str = Field(
        ...,
        description="Deterministic unique claim ID derived from SHA-256 of normalized text",
    )
    claim: str = Field(..., description="Extracted claim text")
    original_text: str = Field(..., description="Raw text segment before normalization")
    normalized_text: str = Field(..., description="Unicode NFC and whitespace-normalized text")
    classification: str = Field(
        ...,
        description="Semantic classification: 'FACTUAL', 'QUOTE', 'OPINION', 'QUESTION', 'SLOGAN', 'AMBIGUOUS'",
    )
    verification_eligibility: str = Field(
        ...,
        description="Eligibility status for fact-check verification: 'ELIGIBLE' or 'INELIGIBLE'",
    )
    eligibility_reason: str = Field(
        "",
        description="Deterministic explanation for verification eligibility or rejection",
    )
    claim_type: str = Field(
        "unclassified",
        description="Legacy classification subtype: 'factual', 'statistical', 'attribution', 'opinion', 'unclassified'",
    )
    confidence: float = Field(
        ...,
        description="Heuristic extraction score (0.0 to 1.0); explicitly NOT a truth probability",
    )
    confidence_semantics: str = Field(
        "heuristic extraction score; not truth probability",
        description="Semantic definition of the confidence score",
    )
    entities: List[str] = Field(
        default_factory=list,
        description="Capitalized proper nouns identified as entity candidates",
    )
    dates: List[str] = Field(
        default_factory=list,
        description="Extracted date expressions matching recognized patterns",
    )
    locations: List[str] = Field(
        default_factory=list,
        description="Extracted geographical location names",
    )
    numbers: List[str] = Field(
        default_factory=list,
        description="Extracted numeric values and percentages",
    )
    source: str = Field("text", description="Source modality of claim ('text', 'ocr', etc.)")
    source_start: Optional[int] = Field(
        None,
        description="0-based start character offset in original input text",
    )
    source_end: Optional[int] = Field(
        None,
        description="0-based end character offset in original input text",
    )
    source_text_reference: Optional[str] = Field(
        None,
        description="Internal reference identifier for source sentence or segment",
    )
    source_region_refs: List[str] = Field(
        default_factory=list,
        description="Spatial region identifiers from OCR if provided",
    )
    source_regions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Spatial region objects including bounding boxes if provided",
    )
    extraction_method: str = Field(
        "conservative_rules_v2",
        description="Claim extraction algorithm identifier and version",
    )
    language: str = Field("unknown", description="Language code")
    language_code: Optional[str] = Field(None, description="ISO language code")
    language_name: Optional[str] = Field(None, description="Human-readable language name")
    language_source: Optional[str] = Field(None, description="Provenance source of language identification")
    language_model_id: Optional[str] = Field(None, description="Model ID used for language detection")
    language_model_revision: Optional[str] = Field(None, description="Model revision used for language detection")
    language_confidence: Optional[float] = Field(None, description="Language identification confidence score")
    language_detection_status: str = Field("unavailable", description="Language detection pipeline status")
    language_detection_outcome: str = Field("unavailable", description="Language detection outcome")
    ocr_confidence: Optional[float] = Field(None, description="OCR confidence score if derived from OCR")
    translated_text: Optional[str] = Field(None, description="Translated text if applicable")
    warnings: List[str] = Field(default_factory=list, description="Per-claim warnings or caveats")


class ClaimExtractionResponse(BaseModel):
    """Response payload for POST /api/v1/claims."""

    status: str = Field(
        ...,
        description="Execution status ('completed', 'unavailable', 'failed')",
    )
    claims: List[ClaimResponse] = Field(
        default_factory=list,
        description="List of extracted candidate claims",
    )
    total_claims: int = Field(
        0,
        description="Total candidate claims extracted",
    )
    candidate_count: int = Field(
        0,
        description="Alias for total_claims",
    )
    accepted_count: int = Field(
        0,
        description="Number of extracted claims deemed ELIGIBLE for fact-check verification",
    )
    eligible_count: int = Field(
        0,
        description="Alias for accepted_count",
    )
    method: str = Field(
        "conservative_rules_v2",
        description="Extraction method identifier and version",
    )
    extraction_method: str = Field(
        "conservative_rules_v2",
        description="Alias for method",
    )
    runtime_ms: int = Field(
        0,
        description="Processing runtime in milliseconds",
    )
    processing_time_ms: int = Field(
        0,
        description="Alias for runtime_ms",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal warnings or extraction caveats",
    )
