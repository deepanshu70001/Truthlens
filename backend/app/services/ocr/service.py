"""
TruthLens — OCR service using EasyOCR.

Supports English and Hindi/Devanagari. Language list is extensible.
Returns raw text, normalized text, language, bounding boxes, and confidence.
"""

from __future__ import annotations

import io
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from backend.app.core.logging import get_logger

log = get_logger("ocr")

# Languages supported — add more Indian languages by appending to this list
DEFAULT_LANGUAGES = ["en", "hi"]

# EasyOCR language codes for Indian languages (for future expansion)
INDIAN_LANGUAGES = {
    "hi": "Hindi (Devanagari)",
    "mr": "Marathi",
    "ne": "Nepali",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "gu": "Gujarati",
    "pa": "Punjabi (Gurmukhi)",
    "bn": "Bengali",
    "as": "Assamese",
}


@dataclass
class OCRBlock:
    """A single detected text block."""
    text: str
    confidence: float
    bbox: List[List[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]


@dataclass
class OCRResult:
    """Complete OCR extraction result."""
    text: str = ""
    normalized_text: str = ""
    language: str = "unknown"
    blocks: List[OCRBlock] = field(default_factory=list)
    total_confidence: float = 0.0
    word_count: int = 0
    runtime_ms: int = 0
    has_text: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "normalized_text": self.normalized_text,
            "language": self.language,
            "blocks": [
                {
                    "text": b.text,
                    "confidence": round(b.confidence, 4),
                    "bbox": b.bbox,
                }
                for b in self.blocks
            ],
            "total_confidence": round(self.total_confidence, 4),
            "word_count": self.word_count,
            "runtime_ms": self.runtime_ms,
            "has_text": self.has_text,
            "warnings": self.warnings,
        }


def _normalize_text(text: str) -> str:
    """
    Normalize extracted text:
    - Unicode NFC normalization
    - Collapse whitespace
    - Strip leading/trailing whitespace
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _detect_primary_script(text: str) -> str:
    """Detect primary script (Latin vs Devanagari etc.) from characters."""
    script_counts: Dict[str, int] = {}
    for char in text:
        if char.isspace() or char in ".,!?;:-'\"()[]{}":
            continue
        cat = unicodedata.category(char)
        name = unicodedata.name(char, "UNKNOWN")
        if "DEVANAGARI" in name:
            script_counts["hi"] = script_counts.get("hi", 0) + 1
        elif "LATIN" in name or cat.startswith("L") and ord(char) < 0x0250:
            script_counts["en"] = script_counts.get("en", 0) + 1
        elif "TAMIL" in name:
            script_counts["ta"] = script_counts.get("ta", 0) + 1
        elif "TELUGU" in name:
            script_counts["te"] = script_counts.get("te", 0) + 1
        elif "BENGALI" in name:
            script_counts["bn"] = script_counts.get("bn", 0) + 1
        elif "GURMUKHI" in name:
            script_counts["pa"] = script_counts.get("pa", 0) + 1
        elif "GUJARATI" in name:
            script_counts["gu"] = script_counts.get("gu", 0) + 1
        elif "KANNADA" in name:
            script_counts["kn"] = script_counts.get("kn", 0) + 1

    if not script_counts:
        return "unknown"

    return max(script_counts, key=script_counts.get)


class OCRService:
    """
    OCR service using EasyOCR.

    Lazy-loads the EasyOCR reader on first use.
    Supports English + Hindi by default. More languages can be added.
    """

    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = True):
        self._languages = languages or DEFAULT_LANGUAGES
        self._gpu = gpu
        self._reader = None

    @property
    def is_loaded(self) -> bool:
        return self._reader is not None

    def load(self) -> None:
        """Load EasyOCR reader (downloads models on first use)."""
        if self._reader is not None:
            return

        import easyocr

        log.info("Loading EasyOCR with languages: %s (GPU=%s)", self._languages, self._gpu)
        t0 = time.time()

        self._reader = easyocr.Reader(
            self._languages,
            gpu=self._gpu,
            verbose=False,
        )

        elapsed = int((time.time() - t0) * 1000)
        log.info("EasyOCR loaded in %d ms", elapsed)

    def unload(self) -> None:
        """Unload reader to free memory."""
        if self._reader is not None:
            del self._reader
            self._reader = None
            log.info("EasyOCR unloaded")

    def extract(self, image: Image.Image) -> OCRResult:
        """
        Extract text from a PIL Image using EasyOCR.

        Returns OCRResult with blocks, confidence, and detected language.
        """
        if self._reader is None:
            raise RuntimeError("OCR reader not loaded. Call load() first.")

        t0 = time.time()
        warnings: List[str] = []

        # Convert to numpy array
        img_array = np.array(image.convert("RGB"))

        # Run EasyOCR
        try:
            results = self._reader.readtext(img_array)
        except Exception as e:
            log.error("EasyOCR extraction failed: %s", e)
            return OCRResult(
                runtime_ms=int((time.time() - t0) * 1000),
                warnings=[f"OCR extraction failed: {e}"],
            )

        if not results:
            return OCRResult(
                runtime_ms=int((time.time() - t0) * 1000),
                has_text=False,
            )

        # Build blocks
        blocks: List[OCRBlock] = []
        all_text_parts: List[str] = []
        total_conf = 0.0

        for bbox, text, conf in results:
            # Convert bbox to list of [x, y] pairs
            bbox_int = [[int(p[0]), int(p[1])] for p in bbox]
            blocks.append(OCRBlock(
                text=text,
                confidence=float(conf),
                bbox=bbox_int,
            ))
            all_text_parts.append(text)
            total_conf += float(conf)

        raw_text = " ".join(all_text_parts)
        normalized = _normalize_text(raw_text)

        # Detect language from script
        detected_lang = _detect_primary_script(raw_text)

        avg_confidence = total_conf / len(blocks) if blocks else 0.0

        runtime_ms = int((time.time() - t0) * 1000)

        result = OCRResult(
            text=raw_text,
            normalized_text=normalized,
            language=detected_lang,
            blocks=blocks,
            total_confidence=avg_confidence,
            word_count=len(normalized.split()) if normalized else 0,
            runtime_ms=runtime_ms,
            has_text=bool(normalized),
            warnings=warnings,
        )

        log.info(
            "OCR: %d blocks, %d words, lang=%s, confidence=%.2f [%d ms]",
            len(blocks), result.word_count, detected_lang, avg_confidence, runtime_ms,
        )

        return result


# Module-level singleton
_ocr_service: Optional[OCRService] = None


def get_ocr_service(gpu: bool = True) -> OCRService:
    """Get or create the singleton OCR service."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService(gpu=gpu)
    return _ocr_service
