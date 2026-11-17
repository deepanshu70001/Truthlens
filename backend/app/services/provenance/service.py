"""
TruthLens — Image provenance / web detection service.

Uses Google Cloud Vision API Web Detection to find:
- Matching pages (full match)
- Matching images (exact copies)
- Partial matches
- Visually similar images
- Web entities

Behavior:
- Configured + success  → returns real results
- Not configured        → returns status "unavailable"
- API failure           → returns status "failed"

NEVER claims that finding a match proves manipulation.
Provenance is supporting evidence, not proof.
"""

from __future__ import annotations

import base64
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from backend.app.config import settings
from backend.app.core.logging import get_logger

log = get_logger("provenance")

VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"


@dataclass
class WebEntity:
    """A web entity associated with the image."""
    entity_id: str = ""
    description: str = ""
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "description": self.description,
            "score": round(self.score, 4),
        }


@dataclass
class WebPage:
    """A web page containing or referencing the image."""
    url: str = ""
    page_title: str = ""
    match_type: str = ""  # full_match | partial_match
    publisher: str = ""
    retrieved_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "page_title": self.page_title,
            "match_type": self.match_type,
            "publisher": self.publisher,
            "retrieved_at": self.retrieved_at,
        }


@dataclass
class SimilarImage:
    """A visually similar image found on the web."""
    url: str = ""
    match_type: str = ""  # exact | similar
    retrieved_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "match_type": self.match_type,
            "retrieved_at": self.retrieved_at,
        }


@dataclass
class ProvenanceResult:
    """Complete provenance/web detection result."""
    status: str = "unavailable"  # unavailable | success | no_results | failed
    web_entities: List[WebEntity] = field(default_factory=list)
    matching_pages: List[WebPage] = field(default_factory=list)
    matching_images: List[SimilarImage] = field(default_factory=list)
    similar_images: List[SimilarImage] = field(default_factory=list)
    best_guess_labels: List[str] = field(default_factory=list)
    total_matches: int = 0
    runtime_ms: int = 0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    provider: str = "google_vision_web_detection"
    retrieved_at: Optional[str] = None
    content_credentials: Dict[str, Any] = field(default_factory=lambda: {
        "status": "unavailable",
        "reason": "No C2PA/Content Credentials parser is configured; manifest presence and validity were not assessed.",
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "web_entities": [e.to_dict() for e in self.web_entities],
            "matching_pages": [p.to_dict() for p in self.matching_pages],
            "matching_images": [i.to_dict() for i in self.matching_images],
            "similar_images": [i.to_dict() for i in self.similar_images],
            "best_guess_labels": self.best_guess_labels,
            "total_matches": self.total_matches,
            "runtime_ms": self.runtime_ms,
            "error": self.error,
            "warnings": self.warnings,
            "provider": self.provider,
            "retrieved_at": self.retrieved_at,
            "content_credentials": self.content_credentials,
            "note": (
                "Provenance results show where this image (or similar images) "
                "appear online. A match does NOT prove manipulation — it provides "
                "context about the image's web presence."
            ),
        }


def _is_configured() -> bool:
    """Check if Google Cloud Vision API is configured."""
    return bool(settings.GOOGLE_API_KEY)


def search_provenance(image_data: bytes) -> ProvenanceResult:
    """
    Search for image provenance using Google Cloud Vision Web Detection.

    Args:
        image_data: Raw image bytes

    Returns:
        ProvenanceResult with web entities, pages, and similar images
    """
    t0 = time.time()

    if not _is_configured():
        return ProvenanceResult(
            status="unavailable",
            warnings=["Google API key not configured. "
                      "Set GOOGLE_API_KEY in .env for web detection."],
            runtime_ms=int((time.time() - t0) * 1000),
        )

    # Build request
    image_b64 = base64.b64encode(image_data).decode("utf-8")

    request_body = {
        "requests": [
            {
                "image": {"content": image_b64},
                "features": [
                    {"type": "WEB_DETECTION", "maxResults": 20},
                ],
            }
        ]
    }

    try:
        with httpx.Client(timeout=settings.PROVENANCE_API_TIMEOUT_SECONDS) as client:
            response = client.post(
                VISION_API_URL,
                params={"key": settings.GOOGLE_API_KEY},
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Malformed Vision API response")

    except httpx.HTTPStatusError as e:
        log.error("Vision API HTTP error (%s)", e.response.status_code)
        return ProvenanceResult(
            status="failed",
            error=f"Vision API error: HTTP {e.response.status_code}",
            runtime_ms=int((time.time() - t0) * 1000),
        )
    except httpx.RequestError as e:
        log.error("Vision API request failed (%s)", type(e).__name__)
        return ProvenanceResult(
            status="failed",
            error="Vision API request failed.",
            runtime_ms=int((time.time() - t0) * 1000),
        )
    except Exception as e:
        log.error("Vision API unexpected error (%s)", type(e).__name__)
        return ProvenanceResult(
            status="failed",
            error="Vision API processing failed.",
            runtime_ms=int((time.time() - t0) * 1000),
        )

    # Parse response
    responses = data.get("responses", [])
    if not isinstance(responses, list) or (responses and not isinstance(responses[0], dict)):
        return ProvenanceResult(status="failed", error="Vision API returned a malformed response.", runtime_ms=int((time.time() - t0) * 1000))
    if responses and responses[0].get("error"):
        return ProvenanceResult(status="failed", error="Vision API could not process this image.", runtime_ms=int((time.time() - t0) * 1000))
    if not responses:
        return ProvenanceResult(
            status="no_results",
            runtime_ms=int((time.time() - t0) * 1000),
        )

    web_detection = responses[0].get("webDetection", {})
    if not isinstance(web_detection, dict):
        return ProvenanceResult(status="failed", error="Vision API returned malformed web-detection data.", runtime_ms=int((time.time() - t0) * 1000))
    retrieved_at = datetime.now(timezone.utc).isoformat()

    def external_https(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        parsed = urlparse(value)
        return value if parsed.scheme == "https" and parsed.hostname and not parsed.username and not parsed.password else ""

    # Web entities
    web_entities = []
    for entity in web_detection.get("webEntities", []):
        if not isinstance(entity, dict):
            continue
        try:
            entity_score = float(entity.get("score", 0.0))
        except (TypeError, ValueError):
            entity_score = 0.0
        web_entities.append(WebEntity(
            entity_id=entity.get("entityId", ""),
            description=entity.get("description", ""),
            score=entity_score if 0 <= entity_score <= 1 else 0.0,
        ))

    # Full matching images
    matching_images = []
    for match in web_detection.get("fullMatchingImages", []):
        if not isinstance(match, dict) or not external_https(match.get("url")):
            continue
        matching_images.append(SimilarImage(
            url=external_https(match.get("url")),
            match_type="exact",
            retrieved_at=retrieved_at,
        ))

    # Partial matching images
    for match in web_detection.get("partialMatchingImages", []):
        if not isinstance(match, dict) or not external_https(match.get("url")):
            continue
        matching_images.append(SimilarImage(
            url=external_https(match.get("url")),
            match_type="partial",
            retrieved_at=retrieved_at,
        ))

    # Pages with matching images
    matching_pages = []
    for page in web_detection.get("pagesWithMatchingImages", []):
        if not isinstance(page, dict) or not external_https(page.get("url")):
            continue
        page_url = external_https(page.get("url"))
        matching_pages.append(WebPage(
            url=page_url,
            page_title=str(page.get("pageTitle", ""))[:500],
            match_type="full_match" if page.get("fullMatchingImages") else "partial_match",
            publisher=urlparse(page_url).hostname or "",
            retrieved_at=retrieved_at,
        ))

    # Visually similar images
    similar_images = []
    for img in web_detection.get("visuallySimilarImages", []):
        if not isinstance(img, dict) or not external_https(img.get("url")):
            continue
        similar_images.append(SimilarImage(
            url=external_https(img.get("url")),
            match_type="similar",
            retrieved_at=retrieved_at,
        ))

    # Best guess labels
    best_guess_labels = []
    for label in web_detection.get("bestGuessLabels", []):
        if not isinstance(label, dict):
            continue
        lbl = label.get("label", "")
        if lbl:
            best_guess_labels.append(lbl)

    total = len(matching_images) + len(matching_pages) + len(similar_images)
    status = "success" if total > 0 or web_entities else "no_results"

    runtime_ms = int((time.time() - t0) * 1000)

    result = ProvenanceResult(
        status=status,
        web_entities=web_entities,
        matching_pages=matching_pages[:10],
        matching_images=matching_images[:10],
        similar_images=similar_images[:10],
        best_guess_labels=best_guess_labels,
        total_matches=total,
        runtime_ms=runtime_ms,
        retrieved_at=retrieved_at,
    )

    log.info(
        "Provenance: %d entities, %d pages, %d matches, %d similar [%d ms]",
        len(web_entities), len(matching_pages),
        len(matching_images), len(similar_images), runtime_ms,
    )

    return result
