"""
TruthLens — Claim extraction service.

Extracts structured claims from OCR/normalized text.
Uses rule-based extraction (no ML required).

The claim extractor identifies factual assertions — it does NOT
determine whether claims are true or false.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.app.core.logging import get_logger

log = get_logger("claims")

# Patterns that indicate factual claims
_CLAIM_INDICATORS = [
    # Statistics / numbers
    r"\d+\s*%",
    r"\d[\d,]*\s+(people|cases|deaths|infected|killed|injured|crores?|lakhs?)",
    # Attribution
    r"(according to|says?|said|stated|reported|announced|confirmed|revealed)\b",
    # Factual assertion patterns
    r"\b(is|are|was|were|has|have|had)\s+(the|a)\s+",
    r"\b(proved?n?|confirmed|established|discovered|found)\b",
    # Breaking news patterns
    r"\b(breaking|exclusive|alert|urgent|official)\b",
    # Temporal claims
    r"\b(today|yesterday|tomorrow|this (week|month|year))\b",
    r"\b(just (in|now|happened|announced))\b",
]

# Entity extraction patterns (basic — not a full NER)
_DATE_PATTERNS = [
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}\b",
    r"\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}\b",
    r"\b\d{4}\b",  # Standalone years
]

_LOCATION_INDICATORS = [
    r"\b(in|at|from|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
    # Indian states and cities
    r"\b(Delhi|Mumbai|Bengaluru|Bangalore|Chennai|Kolkata|Hyderabad|Pune|Ahmedabad|Jaipur)\b",
    r"\b(Uttar Pradesh|Madhya Pradesh|Maharashtra|Karnataka|Tamil Nadu|Kerala|Gujarat|Rajasthan)\b",
    r"\b(Bihar|West Bengal|Assam|Odisha|Punjab|Haryana|Telangana|Andhra Pradesh)\b",
    r"\b(India|Pakistan|China|Bangladesh|Nepal|Sri Lanka)\b",
]

# Claim type classification keywords
_CLAIM_TYPE_MAP = {
    "statistical": [r"\d+\s*%", r"\d[\d,]*\s+(crore|lakh|million|billion|thousand)"],
    "attribution": [r"(said|says|stated|according|reported|announced)"],
    "factual": [r"(is|are|was|were|has been|will be)\s+"],
    "temporal": [r"(today|yesterday|tomorrow|just|recently|breaking)"],
    "causal": [r"(because|caused|due to|resulted|led to|effect)"],
}


@dataclass
class ExtractedClaim:
    """A single extracted claim."""
    claim: str
    entities: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    claim_type: str = "unclassified"
    confidence: float = 0.0
    source: str = "ocr"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "entities": self.entities,
            "dates": self.dates,
            "locations": self.locations,
            "claim_type": self.claim_type,
            "confidence": round(self.confidence, 4),
            "source": self.source,
        }


@dataclass
class ClaimExtractionResult:
    """Result of claim extraction."""
    claims: List[ExtractedClaim] = field(default_factory=list)
    total_claims: int = 0
    runtime_ms: int = 0
    method: str = "rule_based"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claims": [c.to_dict() for c in self.claims],
            "total_claims": self.total_claims,
            "runtime_ms": self.runtime_ms,
            "method": self.method,
            "warnings": self.warnings,
        }


def _extract_dates(text: str) -> List[str]:
    """Extract date mentions from text."""
    dates = []
    for pattern in _DATE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            date_str = m if isinstance(m, str) else m[0] if m else ""
            if date_str and date_str not in dates:
                dates.append(date_str)
    return dates


def _extract_locations(text: str) -> List[str]:
    """Extract location mentions from text."""
    locations = []
    for pattern in _LOCATION_INDICATORS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            loc = m if isinstance(m, str) else m[-1] if isinstance(m, tuple) else str(m)
            loc = loc.strip()
            if loc and len(loc) > 2 and loc not in locations:
                locations.append(loc)
    return locations


def _extract_entities(text: str) -> List[str]:
    """Extract potential named entities (capitalized words/phrases)."""
    # Simple approach: sequences of capitalized words
    entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    # Deduplicate, exclude very short
    seen = set()
    result = []
    for e in entities:
        if e not in seen and len(e) > 2:
            seen.add(e)
            result.append(e)
    return result[:20]  # Cap at 20 entities


def _classify_claim(text: str) -> str:
    """Classify a claim into a type based on content patterns."""
    for claim_type, patterns in _CLAIM_TYPE_MAP.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return claim_type
    return "unclassified"


def _score_claim_confidence(sentence: str) -> float:
    """
    Score how likely a sentence is to contain a verifiable claim.
    Higher = more likely to be a factual claim worth checking.
    """
    score = 0.0
    text_lower = sentence.lower()

    # Has numbers/statistics → more likely factual
    if re.search(r"\d+", sentence):
        score += 0.2

    # Has attribution → more likely a claim
    if re.search(r"(said|says|stated|according|reported)", text_lower):
        score += 0.2

    # Has factual language
    if re.search(r"(confirmed|proved|established|discovered|found|revealed)", text_lower):
        score += 0.15

    # Has temporal markers → newsworthy
    if re.search(r"(today|yesterday|breaking|just|recently)", text_lower):
        score += 0.15

    # Has location context
    if re.search(r"(in|at|from)\s+[A-Z]", sentence):
        score += 0.1

    # Is a question → less likely a claim
    if sentence.strip().endswith("?"):
        score -= 0.3

    # Very short → less useful
    if len(sentence.split()) < 5:
        score -= 0.2

    # Adequate length
    if len(sentence.split()) >= 8:
        score += 0.1

    return max(0.0, min(1.0, score))


def extract_claims(
    sentences: List[str],
    min_confidence: float = 0.1,
    max_claims: int = 20,
) -> ClaimExtractionResult:
    """
    Extract structured claims from a list of sentences.

    The extractor identifies potential factual claims. It does NOT
    determine whether any claim is true or false.

    Args:
        sentences: List of sentences (from text normalization)
        min_confidence: Minimum score to include a claim
        max_claims: Maximum number of claims to return
    """
    t0 = time.time()
    warnings: List[str] = []

    if not sentences:
        return ClaimExtractionResult(
            runtime_ms=int((time.time() - t0) * 1000),
            warnings=["No sentences provided for claim extraction."],
        )

    claims: List[ExtractedClaim] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            continue

        confidence = _score_claim_confidence(sentence)

        if confidence < min_confidence:
            continue

        claim = ExtractedClaim(
            claim=sentence,
            entities=_extract_entities(sentence),
            dates=_extract_dates(sentence),
            locations=_extract_locations(sentence),
            claim_type=_classify_claim(sentence),
            confidence=confidence,
            source="ocr",
        )
        claims.append(claim)

    # Sort by confidence descending, cap at max_claims
    claims.sort(key=lambda c: c.confidence, reverse=True)
    claims = claims[:max_claims]

    runtime_ms = int((time.time() - t0) * 1000)

    result = ClaimExtractionResult(
        claims=claims,
        total_claims=len(claims),
        runtime_ms=runtime_ms,
        warnings=warnings,
    )

    log.info(
        "Claims extracted: %d claims from %d sentences [%d ms]",
        len(claims), len(sentences), runtime_ms,
    )

    return result
