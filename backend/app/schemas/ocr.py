"""
TruthLens — Pydantic schemas for the Standalone OCR endpoint.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class OCRBlockResponse(BaseModel):
    """A detected text region with recognized content, confidence, and bounding polygon."""

    text: str = Field(..., description="Recognized text in this region")
    confidence: Optional[float] = Field(None, description="Recognition confidence between 0.0 and 1.0, if available")
    bbox: List[List[int]] = Field(
        ...,
        description="Bounding box polygon coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] in source image pixels",
    )


class OCRResponse(BaseModel):
    """Response payload for POST /api/v1/ocr."""

    status: str = Field(..., description="Execution status ('completed', 'partial', 'unavailable', 'failed')")
    text: str = Field(..., description="Full concatenated extracted text")
    normalized_text: Optional[str] = Field(None, description="Unicode-normalized text with canonical whitespace")
    language: str = Field("unknown", description="Primary script or language detected from OCR text")
    engine: Optional[str] = Field(None, description="OCR engine utilized ('EasyOCR', 'PyTesseract')")
    confidence: Optional[float] = Field(None, description="Average recognition confidence across valid blocks")
    total_confidence: Optional[float] = Field(None, description="Alias for confidence")
    has_text: bool = Field(False, description="Whether any readable text was detected in the image")
    detected: bool = Field(False, description="Alias for has_text")
    no_text_reason: Optional[str] = Field(None, description="Detailed explanation when no readable text was found")
    word_count: int = Field(0, description="Count of words in extracted text")
    processing_time_ms: int = Field(0, description="Processing duration in milliseconds")
    runtime_ms: int = Field(0, description="Alias for processing_time_ms")
    blocks: List[OCRBlockResponse] = Field(default_factory=list, description="Extracted text blocks with coordinates")
    results: List[OCRBlockResponse] = Field(default_factory=list, description="Alias for blocks")
    language_configuration: List[str] = Field(default_factory=list, description="Configured language codes for the OCR engine")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings or operational caveats")
