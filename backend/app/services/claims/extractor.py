"""Conservative, deterministic extraction and classification of OCR claims.

This service identifies candidate statements; it does not decide whether they
are true. It uses no language model and never expands OCR into new assertions.
"""
from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from backend.app.core.logging import get_logger

log = get_logger("claims")

_DATE_PATTERNS = (
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}\b",
    r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}\b",
    r"\b(?:19|20)\d{2}\b",
)
_LOCATION_PATTERNS = (
    r"\b(?:Delhi|Mumbai|Bengaluru|Bangalore|Chennai|Kolkata|Hyderabad|Pune|Ahmedabad|Jaipur|India|Pakistan|China|Bangladesh|Nepal|Sri Lanka)\b",
    r"\b(?:Uttar Pradesh|Madhya Pradesh|Maharashtra|Karnataka|Tamil Nadu|Kerala|Gujarat|Rajasthan|Bihar|West Bengal|Assam|Odisha|Punjab|Haryana|Telangana|Andhra Pradesh)\b",
)
_QUESTION = re.compile(r"(?:\?\s*$|^\s*(?:क्या|कौन|कब|कहाँ|क्यों|कैसे)\b)", re.I)
_OPINION = re.compile(r"\b(?:I think|I believe|in my opinion|should|must|best|worst|beautiful|terrible|good|bad|मुझे लगता है|मेरे विचार में|चाहिए|सबसे अच्छा|सबसे बुरा)\b", re.I)
_SLOGAN = re.compile(r"^(?:#\w+|(?:save|protect|support|stop|boycott|say no to|share|like|subscribe|follow|watch|read)\b.*)$", re.I)
_FACTUAL = re.compile(r"\b(?:is|are|was|were|has|have|had|will|did|does|reported|announced|confirmed|said|according to|killed|injured|occurred|happened|won|lost|launched|approved|banned|increased|decreased|ने|है|हैं|थे|थी|था|हुए|हुई|किया|किए|की|दिया|दर्ज|घोषित|मंजूर|बैन|जीता|हारा|घटना)\b", re.I)
_FRAGMENT = re.compile(r"^(?:breaking|exclusive|urgent|alert|viral|shocking|must watch|share now)[!:.\s]*$", re.I)
_NUMBER = re.compile(r"(?<!\w)\d[\d,]*(?:\.\d+)?%?(?!\w)")
_ENTITY = re.compile(r"\b[A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,3}\b")
EXTRACTION_METHOD = "conservative_rules_v2"


@dataclass
class ExtractedClaim:
    claim: str
    entities: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    claim_type: str = "unclassified"  # legacy subtype retained for clients
    classification: str = "AMBIGUOUS"
    confidence: float = 0.0            # extraction heuristic, not truth probability
    source: str = "ocr"
    claim_id: str = ""
    original_text: str = ""
    normalized_text: str = ""
    language: str = "unknown"
    translated_text: Optional[str] = None
    numbers: List[str] = field(default_factory=list)
    source_text_reference: Optional[str] = None
    verification_eligibility: str = "INELIGIBLE"
    eligibility_reason: str = ""
    ocr_confidence: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    source_start: Optional[int] = None
    source_end: Optional[int] = None
    source_region_refs: List[str] = field(default_factory=list)
    source_regions: List[Dict[str, Any]] = field(default_factory=list)
    extraction_method: str = EXTRACTION_METHOD
    language_detection_status: str = "unavailable"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id, "claim": self.claim,
            "original_text": self.original_text or self.claim,
            "normalized_text": self.normalized_text or self.claim,
            "language": self.language, "translated_text": self.translated_text,
            "entities": self.entities, "dates": self.dates,
            "locations": self.locations, "numbers": self.numbers,
            "claim_type": self.claim_type, "confidence": round(self.confidence, 4),
            "classification": self.classification,
            "language_detection_status": self.language_detection_status,
            "source_start": self.source_start, "source_end": self.source_end,
            "source_region_refs": self.source_region_refs,
            "source_regions": self.source_regions,
            "extraction_method": self.extraction_method,
            "confidence_semantics": "heuristic extraction score; not truth probability",
            "source": self.source, "source_text_reference": self.source_text_reference,
            "ocr_confidence": round(self.ocr_confidence, 4) if self.ocr_confidence is not None else None,
            "verification_eligibility": self.verification_eligibility,
            "eligibility_reason": self.eligibility_reason, "warnings": self.warnings,
        }


@dataclass
class ClaimExtractionResult:
    claims: List[ExtractedClaim] = field(default_factory=list)
    total_claims: int = 0
    runtime_ms: int = 0
    method: str = "conservative_rules"
    status: str = "completed"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "claims": [c.to_dict() for c in self.claims],
                "total_claims": self.total_claims, "runtime_ms": self.runtime_ms,
                "method": self.method, "warnings": self.warnings}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def _classify(text: str) -> tuple[str, str, str, str]:
    if re.search(r'["“”‘’]', text):
        if re.search(r"\b(?:said|says|according to|quoted|stated|announced)\b", text, re.I):
            return "attribution", "QUOTE", "ELIGIBLE", "Attributed quotation candidate; verify wording and attribution against retrieved reviews."
        return "unclassified", "QUOTE", "INELIGIBLE", "Quotation has no identifiable attribution for verification."
    if _QUESTION.search(text):
        return "unclassified", "QUESTION", "INELIGIBLE", "Questions are not treated as factual assertions."
    if _OPINION.search(text):
        return "opinion", "OPINION", "INELIGIBLE", "Subjective language is not eligible for factual verification."
    if _FRAGMENT.fullmatch(text):
        return "unclassified", "SLOGAN", "INELIGIBLE", "Promotional or slogan-like fragment has no checkable proposition."
    if _SLOGAN.fullmatch(text) and not re.search(r"\d|\b(?:is|was|were|has|will|did)\b", text, re.I):
        return "unclassified", "SLOGAN", "INELIGIBLE", "Slogan-like text does not state a verifiable proposition."
    if _FACTUAL.search(text) or _NUMBER.search(text):
        # Short claims without an event, entity, number or attribution can be
        # incomplete; mark ambiguous rather than dispatching noisy searches.
        words = text.split()
        if len(words) < 3:
            return "unclassified", "AMBIGUOUS", "INELIGIBLE", "Statement is too short to identify a verifiable proposition."
        subtype = "statistical" if _NUMBER.search(text) else "factual"
        return subtype, "FACTUAL", "ELIGIBLE", "Text contains a factual assertion candidate."
    return "unclassified", "AMBIGUOUS", "INELIGIBLE", "No clear factual assertion could be identified from the OCR text."


def _entities(text: str) -> List[str]:
    # Preserve deduplicated capitalized spans; this is a lightweight heuristic,
    # not named-entity recognition.
    stop = {"the", "government", "officials", "according", "breaking", "exclusive", "urgent", "today", "yesterday"}
    return list(dict.fromkeys(match for match in _ENTITY.findall(text)
                              if len(match) > 2 and match.casefold() not in stop))[:20]


def _extract_dates(text: str) -> List[str]:
    return list(dict.fromkeys(m.group(0) for pattern in _DATE_PATTERNS for m in re.finditer(pattern, text, re.I)))


def _extract_locations(text: str) -> List[str]:
    return list(dict.fromkeys(m.group(0) for pattern in _LOCATION_PATTERNS for m in re.finditer(pattern, text, re.I)))


def _score(text: str, ocr_confidence: Optional[float]) -> float:
    # Ranking hint only; explicitly not a truth probability.
    score = 0.35 + min(len(text.split()), 15) * 0.02
    if _FACTUAL.search(text):
        score += 0.2
    if _NUMBER.search(text):
        score += 0.1
    if text.endswith("?"):
        score -= 0.4
    if ocr_confidence is not None:
        score = score * 0.75 + max(0.0, min(1.0, ocr_confidence)) * 0.25
    return round(max(0.0, min(score, 1.0)), 4)


def extract_claims(
    sentences: Sequence[str] | str, min_confidence: float = 0.1, max_claims: int = 5,
    *, language: str = "unknown", language_detection_status: str = "unavailable",
    ocr_confidence: Optional[float] = None, sentence_spans: Optional[Sequence[Dict[str, Any]]] = None,
    source_regions: Optional[Sequence[Dict[str, Any]]] = None,
) -> ClaimExtractionResult:
    """Extract OCR-grounded candidates while preserving ineligible text.

    Duplicate normalized statements are collapsed. Candidate IDs derive from
    normalized text, making them stable across reruns of the same analysis.
    """
    started = time.perf_counter()
    warnings: List[str] = []
    found: Dict[str, ExtractedClaim] = {}
    items: List[Dict[str, Any]] = []
    if isinstance(sentences, str):
        from backend.app.services.text.normalizer import _sentence_spans
        items = _sentence_spans(sentences)
    elif sentence_spans:
        items = [dict(item) for item in sentence_spans]
    else:
        items = [{"text": raw} for raw in (sentences or [])]
    for index, item in enumerate(items):
        raw = item.get("text", "")
        original = str(raw or "").strip()
        normalized = _clean(original)
        if not normalized or len(normalized) < 5:
            continue
        key = " ".join(normalized.casefold().split())
        if key in found:
            continue
        legacy_type, classification, eligibility, reason = _classify(normalized)
        confidence = _score(normalized, ocr_confidence)
        # Keep slogans/opinions/questions/ambiguous text visible, but never
        # send them to ordinary fact-check retrieval.
        if confidence < min_confidence and eligibility == "ELIGIBLE":
            eligibility, reason = "INELIGIBLE", "OCR candidate confidence is too low for retrieval."
        claim_id = "claim_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        claim_warnings = []
        if ocr_confidence is not None and ocr_confidence < 0.5:
            claim_warnings.append("OCR confidence is low; verify the recognized wording.")
        if language == "unknown" or language_detection_status != "success":
            claim_warnings.append("Original language could not be identified confidently.")
        start, end = item.get("start"), item.get("end")
        region_refs = []
        matched_regions = []
        if start is not None and end is not None:
            for region in source_regions or []:
                if region.get("text_start") is not None and region.get("text_end") is not None and region["text_start"] < end and region["text_end"] > start:
                    ref = region.get("region_id") or region.get("id")
                    if ref:
                        region_refs.append(str(ref))
                        matched_regions.append({"region_id": str(ref), "bbox": region.get("bbox")})
        found[key] = ExtractedClaim(
            claim=original, original_text=original, normalized_text=normalized,
            claim_id=claim_id, claim_type=legacy_type, classification=classification, confidence=confidence,
            entities=_entities(normalized), dates=_extract_dates(normalized),
            locations=_extract_locations(normalized), numbers=list(dict.fromkeys(_NUMBER.findall(normalized))),
            language=language, source_text_reference=f"ocr_sentence_{index}",
            verification_eligibility=eligibility, eligibility_reason=reason,
            ocr_confidence=ocr_confidence, warnings=claim_warnings,
            source_start=start, source_end=end, source_region_refs=region_refs,
            source_regions=matched_regions,
            language_detection_status=language_detection_status,
        )
    claims = list(found.values())[:max(0, max_claims)]
    if not claims:
        warnings.append("No candidate statements were identified in the OCR text.")
    if len(found) > max_claims:
        warnings.append(f"Candidate list capped at {max_claims} claims.")
    result = ClaimExtractionResult(claims=claims, total_claims=len(claims),
        runtime_ms=int((time.perf_counter() - started) * 1000), method=EXTRACTION_METHOD, warnings=warnings)
    log.info("Claims extracted: %d candidates [%d ms]", len(claims), result.runtime_ms)
    return result
