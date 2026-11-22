"""
TruthLens — Text normalization service.

Performs reasonable cleanup of OCR-extracted text while
preserving the raw text separately.

Operations:
- Unicode NFC normalization
- Whitespace collapse
- Common OCR error correction (optional)
- Case normalization for comparison (separate field)
- Script-specific cleanup

Raw text is NEVER modified. Normalized text is a parallel field.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.app.core.logging import get_logger

log = get_logger("text_normalize")

# Common OCR substitution errors (conservative list)
_OCR_CORRECTIONS = {
    "l": "I",   # Only in all-caps context
    "0": "O",   # Only in all-caps context
    "|": "I",
    "rn": "m",  # Very common OCR error
}


@dataclass
class NormalizedTextResult:
    """Result of text normalization."""
    raw_text: str = ""
    normalized_text: str = ""
    sentences: List[str] = field(default_factory=list)
    sentence_spans: List[Dict[str, Any]] = field(default_factory=list)
    word_count: int = 0
    char_count: int = 0
    has_urls: bool = False
    extracted_urls: List[str] = field(default_factory=list)
    has_hashtags: bool = False
    extracted_hashtags: List[str] = field(default_factory=list)
    has_mentions: bool = False
    extracted_mentions: List[str] = field(default_factory=list)
    detected_numbers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "sentences": self.sentences,
            "sentence_spans": self.sentence_spans,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "has_urls": self.has_urls,
            "extracted_urls": self.extracted_urls,
            "has_hashtags": self.has_hashtags,
            "extracted_hashtags": self.extracted_hashtags,
            "has_mentions": self.has_mentions,
            "extracted_mentions": self.extracted_mentions,
            "detected_numbers": self.detected_numbers,
            "warnings": self.warnings,
        }


def _unicode_normalize(text: str) -> str:
    """Apply Unicode NFC normalization."""
    return unicodedata.normalize("NFC", text)


def _collapse_whitespace(text: str) -> str:
    """Replace multiple whitespace with single space."""
    return re.sub(r"\s+", " ", text).strip()


def _extract_urls(text: str) -> List[str]:
    """Extract URLs from text."""
    url_pattern = re.compile(
        r"https?://[^\s<>\"')\]]+|www\.[^\s<>\"')\]]+",
        re.IGNORECASE,
    )
    return url_pattern.findall(text)


def _extract_hashtags(text: str) -> List[str]:
    """Extract hashtags from text."""
    return re.findall(r"#\w+", text)


def _extract_mentions(text: str) -> List[str]:
    """Extract @mentions from text."""
    return re.findall(r"@\w+", text)


def _extract_numbers(text: str) -> List[str]:
    """Extract numeric values (for potential claim verification)."""
    return re.findall(r"\b\d[\d,]*\.?\d*\b", text)


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    Simple rule-based approach that handles common cases.
    """
    if not text:
        return []

    # Split on sentence-ending punctuation followed by space/newline
    return [span["text"] for span in _sentence_spans(text)]


def _sentence_spans(text: str) -> List[Dict[str, Any]]:
    # Keep source offsets and avoid splitting common title/initial abbreviations.
    if not text:
        return []
    pattern = re.compile(r"(?<=[.!?])\s+(?=[^\W_\"'“‘])")
    spans: List[Dict[str, Any]] = []
    start = 0
    for match in pattern.finditer(text):
        prefix = text[max(start, match.start() - 6):match.start()]
        abbreviation = re.search(r"(?:\b(?P<word>Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e)\.)$|\b(?P<initial>[A-Z])\.$", prefix, re.I)
        if abbreviation:
            word = (abbreviation.group("word") or "").casefold()
            # Titles and initials before a name are not sentence boundaries.
            if word in {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st"} or abbreviation.group("initial"):
                continue
            # Some abbreviations also end sentences. A following capitalized
            # word is treated as a boundary for etc./vs./e.g./i.e.; this remains
            # a deliberately small heuristic, not a linguistic parser.
            following = text[match.end():]
            if not re.match(r"[A-Z]", following):
                continue
        end = match.start() + 1
        value = text[start:end].strip()
        left = start + len(text[start:end]) - len(text[start:end].lstrip())
        if len(value) > 2:
            spans.append({"text": value, "start": left, "end": left + len(value)})
        start = match.end()
    value = text[start:].strip()
    left = start + len(text[start:]) - len(text[start:].lstrip())
    if len(value) > 2:
        spans.append({"text": value, "start": left, "end": left + len(value)})
    return spans


def normalize_text(raw_text: str) -> NormalizedTextResult:
    """
    Normalize OCR-extracted text.

    The raw_text is preserved as-is.
    The normalized_text is a cleaned version for downstream processing.
    """
    if not raw_text or not raw_text.strip():
        return NormalizedTextResult(
            raw_text=raw_text or "",
            normalized_text="",
        )

    result = NormalizedTextResult(raw_text=raw_text)

    # Step 1: Unicode normalization
    normalized = _unicode_normalize(raw_text)

    # Step 2: Collapse whitespace
    normalized = _collapse_whitespace(normalized)

    # Step 3: Remove control characters (keep newlines for sentence splitting)
    normalized = "".join(
        c for c in normalized
        if not unicodedata.category(c).startswith("C") or c in "\n\r\t "
    )
    normalized = _collapse_whitespace(normalized)

    result.normalized_text = normalized
    result.word_count = len(normalized.split()) if normalized else 0
    result.char_count = len(normalized)

    # Extract structured elements
    result.extracted_urls = _extract_urls(raw_text)
    result.has_urls = bool(result.extracted_urls)

    result.extracted_hashtags = _extract_hashtags(raw_text)
    result.has_hashtags = bool(result.extracted_hashtags)

    result.extracted_mentions = _extract_mentions(raw_text)
    result.has_mentions = bool(result.extracted_mentions)

    result.detected_numbers = _extract_numbers(raw_text)

    result.sentence_spans = _sentence_spans(normalized)
    result.sentences = [item["text"] for item in result.sentence_spans]

    log.info(
        "Text normalized: %d chars, %d words, %d sentences, "
        "%d URLs, %d hashtags",
        result.char_count, result.word_count, len(result.sentences),
        len(result.extracted_urls), len(result.extracted_hashtags),
    )

    return result
