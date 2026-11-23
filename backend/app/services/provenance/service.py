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
import hashlib
import json
import re
import threading
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx

from backend.app.config import settings
from backend.app.core.logging import get_logger
from backend.app.services.provenance.c2pa_verifier import verify_content_credentials

log = get_logger("provenance")

VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"
_WEB_DETECTION_SEMAPHORE = threading.BoundedSemaphore(4)


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
    status: str = "unavailable"  # unavailable | success | no_results | partial | failed
    outcome: Optional[str] = None
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
        "reason": "C2PA verification is unavailable; manifest presence and validity were not assessed.",
    })
    image_sha256: Optional[str] = None
    metadata_status: str = "unavailable"
    metadata: Dict[str, Any] = field(default_factory=dict)
    web_detection_status: str = "unavailable"
    web_detection_outcome: Optional[str] = "unavailable"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "outcome": self.outcome or ("success" if self.status == "success" else "no_results" if self.status == "no_results" else self.status),
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
            "image_integrity": {"algorithm": "SHA-256", "value": self.image_sha256} if self.image_sha256 else None,
            "metadata_status": self.metadata_status,
            "metadata": self.metadata,
            "web_detection": {"status": self.web_detection_status, "outcome": self.web_detection_outcome},
            "note": (
                "Provenance results show where this image (or similar images) "
                "appear online. A match does NOT prove manipulation — it provides "
                "context about the image's web presence."
            ),
        }


def _is_configured() -> bool:
    """Check if Google Cloud Vision API is configured."""
    for attr in ("VISION_API_KEY", "GOOGLE_API_KEY"):
        val = getattr(settings, attr, "")
        if isinstance(val, str) and val.strip():
            return True
    return False


def _get_vision_api_key() -> str:
    for attr in ("VISION_API_KEY", "GOOGLE_API_KEY"):
        val = getattr(settings, attr, "")
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _safe_external_url(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 2048:
        return ""
    try:
        parsed = urlparse(value)
        hostname = parsed.hostname
        # Accessing .port also validates malformed/non-numeric port syntax.
        parsed.port
    except (TypeError, ValueError):
        return ""
    if (parsed.scheme.lower() != "https" or not hostname or any(ch.isspace() for ch in hostname)
            or parsed.username or parsed.password):
        return ""
    secret_parameters = {
        "key", "api_key", "access_key", "secret", "client_secret", "token",
        "access_token", "refresh_token", "signature", "sig", "authorization",
        "credential", "password", "auth",
    }
    safe_query = [
        (key, item) for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if key.casefold().replace("-", "_") not in secret_parameters
    ]
    return urlunparse(("https", parsed.netloc, parsed.path, "", urlencode(safe_query), ""))


def _safe_provider_text(value: Any, limit: int) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]", " ", str(value))
    text = re.sub(r"(?i)(?:[a-z]:\\|/)(?:[^\s:/\\]+[\\/])+[^\s:]*", "[redacted path]", text)
    text = re.sub(r"(?i)(api[_ -]?key|key|token|secret|password)\s*[:=]\s*\S+", r"\1=[redacted]", text)
    return text[:limit]


def _bounded_setting(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(getattr(settings, name, default))
        return min(maximum, max(minimum, value)) if value == value else default
    except (TypeError, ValueError):
        return default


def _request_web_detection(image_data: bytes) -> tuple[str, Dict[str, Any], Optional[str], List[str]]:
    if not _is_configured():
        return "unavailable", {}, None, ["Google Vision Web Detection credentials are not configured."]
    timeout = _bounded_setting("PROVENANCE_API_TIMEOUT_SECONDS", 30.0, 0.1, 60.0)
    if not _WEB_DETECTION_SEMAPHORE.acquire(timeout=timeout):
        return "failed", {}, "PROVIDER_CAPACITY", ["Web Detection could not start within the configured concurrency bound."]
    api_key = _get_vision_api_key()
    body = {"requests": [{"image": {"content": base64.b64encode(image_data).decode("ascii")},
        "features": [{"type": "WEB_DETECTION", "maxResults": 20}]}]}
    try:
        retry_count = int(_bounded_setting("PROVENANCE_RETRY_ATTEMPTS", 1, 0, 2))
        deadline = time.monotonic() + _bounded_setting("PROVENANCE_MAX_TOTAL_SECONDS", 35.0, 0.1, 120.0)
        last_category = "PROVIDER_REQUEST_ERROR"
        for attempt in range(retry_count + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "failed", {}, "PROVIDER_TIMEOUT", ["Web Detection exceeded its configured total time limit."]
            try:
                with httpx.Client(timeout=min(timeout, remaining)) as client:
                    if type(client).__module__.startswith("httpx") and callable(getattr(client, "stream", None)):
                        with client.stream("POST", VISION_API_URL, params={"key": api_key}, json=body) as response:
                            response.raise_for_status()
                            raw = bytearray()
                            for chunk in response.iter_bytes():
                                raw.extend(chunk)
                                if len(raw) > 2 * 1024 * 1024:
                                    return "failed", {}, "PROVIDER_RESPONSE_TOO_LARGE", ["Web Detection response exceeded the configured size limit."]
                            data = json.loads(raw.decode("utf-8"))
                    else:  # small provider mocks used by deterministic unit tests
                        response = client.post(VISION_API_URL, params={"key": api_key}, json=body)
                        response.raise_for_status()
                        content = getattr(response, "content", b"")
                        if isinstance(content, bytes) and len(content) > 2 * 1024 * 1024:
                            return "failed", {}, "PROVIDER_RESPONSE_TOO_LARGE", ["Web Detection response exceeded the configured size limit."]
                        data = response.json()
                if not isinstance(data, dict):
                    return "failed", {}, "MALFORMED_PROVIDER_RESPONSE", ["Web Detection returned a malformed response."]
                return "success", data, None, []
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code not in {429, 500, 502, 503, 504} or attempt >= retry_count or time.monotonic() >= deadline:
                    return "failed", {}, "PROVIDER_HTTP_ERROR", [f"Web Detection request failed with HTTP {status_code}."]
                last_category = "PROVIDER_HTTP_ERROR"
            except httpx.TimeoutException:
                last_category = "PROVIDER_TIMEOUT"
                if attempt >= retry_count or time.monotonic() >= deadline:
                    return "failed", {}, last_category, ["Web Detection request timed out."]
            except httpx.RequestError:
                last_category = "PROVIDER_REQUEST_ERROR"
                if attempt >= retry_count or time.monotonic() >= deadline:
                    return "failed", {}, last_category, ["Web Detection provider could not be reached."]
            except Exception as exc:
                log.warning("Web Detection request failed (%s)", type(exc).__name__)
                return "failed", {}, "PROVIDER_RESPONSE_ERROR", ["Web Detection response could not be processed."]
            time.sleep(min(0.2 * (2 ** attempt), max(0.0, deadline - time.monotonic())))
        return "failed", {}, last_category, ["Web Detection request could not be completed."]
    finally:
        _WEB_DETECTION_SEMAPHORE.release()


def _parse_web_detection(data: Dict[str, Any]) -> tuple[str, Dict[str, Any], Optional[str], List[str]]:
    responses = data.get("responses", [])
    if not isinstance(responses, list):
        return "failed", {}, "MALFORMED_PROVIDER_RESPONSE", ["Web Detection returned a malformed response."]
    if not responses:
        return "no_results", {}, None, []
    if not isinstance(responses[0], dict) or responses[0].get("error"):
        return "failed", {}, "MALFORMED_PROVIDER_RESPONSE", ["Web Detection could not process the image response."]
    web = responses[0].get("webDetection", {})
    if not isinstance(web, dict):
        return "failed", {}, "MALFORMED_PROVIDER_RESPONSE", ["Web Detection returned malformed result data."]
    retrieved = datetime.now(timezone.utc).isoformat()
    entities = []
    for item in web.get("webEntities", []) if isinstance(web.get("webEntities", []), list) else []:
        if not isinstance(item, dict):
            continue
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        if not 0 <= score <= 1:
            continue
        entities.append(WebEntity(_safe_provider_text(item.get("entityId", ""), 200), _safe_provider_text(item.get("description", ""), 300), score))
    images = []
    for collection, match_type in (("fullMatchingImages", "exact"), ("partialMatchingImages", "partial")):
        values = web.get(collection, [])
        if isinstance(values, list):
            images.extend(SimilarImage(_safe_external_url(item.get("url")), match_type, retrieved)
                for item in values[:100] if isinstance(item, dict) and _safe_external_url(item.get("url")))
    pages = []
    page_values = web.get("pagesWithMatchingImages", [])
    if isinstance(page_values, list):
        for item in page_values[:100]:
            if not isinstance(item, dict):
                continue
            url = _safe_external_url(item.get("url"))
            if url:
                pages.append(WebPage(url, _safe_provider_text(item.get("pageTitle", ""), 500),
                    "full_match" if item.get("fullMatchingImages") else "partial_match", urlparse(url).hostname or "", retrieved))
    similar = []
    similar_values = web.get("visuallySimilarImages", [])
    if isinstance(similar_values, list):
        similar = [SimilarImage(_safe_external_url(item.get("url")), "similar", retrieved)
            for item in similar_values[:100] if isinstance(item, dict) and _safe_external_url(item.get("url"))]
    labels = [_safe_provider_text(item.get("label", ""), 200) for item in web.get("bestGuessLabels", [])[:20]
        if isinstance(item, dict) and isinstance(item.get("label"), str)] if isinstance(web.get("bestGuessLabels", []), list) else []
    values = {"web_entities": entities, "matching_pages": pages, "matching_images": images,
        "similar_images": similar, "best_guess_labels": labels, "retrieved_at": retrieved}
    return ("success" if pages or images or similar or entities else "no_results"), values, None, []


def _extract_provenance_metadata(image_data: bytes) -> tuple[str, Dict[str, Any]]:
    try:
        from backend.app.services.metadata.service import extract_metadata
        value = extract_metadata(image_data)
        if value.image_width is None or value.image_height is None:
            return "failed", {"error_category": "METADATA_EXTRACTION_ERROR"}
        return ("success" if value.has_exif else "no_results"), {
            "has_exif": value.has_exif,
            "camera_make": _safe_provider_text(value.camera_make, 200) if value.camera_make else None,
            "camera_model": _safe_provider_text(value.camera_model, 200) if value.camera_model else None,
            "software": _safe_provider_text(value.software, 200) if value.software else None,
            "datetime_original": _safe_provider_text(value.datetime_original, 80) if value.datetime_original else None,
            "datetime_digitized": _safe_provider_text(value.datetime_digitized, 80) if value.datetime_digitized else None,
            "datetime_modified": _safe_provider_text(value.datetime_modified, 80) if value.datetime_modified else None,
            "image_width": value.image_width,
            "image_height": value.image_height,
            "gps_available": value.gps_available,
            "source": "EXIF metadata; not cryptographically verified",
        }
    except Exception as exc:
        log.warning("EXIF metadata extraction failed (%s)", type(exc).__name__)
        return "failed", {"error_category": "METADATA_EXTRACTION_ERROR"}


def search_provenance(image_data: bytes) -> ProvenanceResult:
    """Check original bytes with C2PA and optionally search web matches."""
    started = time.perf_counter()
    if not isinstance(image_data, bytes) or not image_data or len(image_data) > settings.max_upload_bytes:
        return ProvenanceResult(status="failed", outcome="failed", error="Original image bytes failed validation.",
            warnings=["The provenance service requires a non-empty image within the configured upload limit."])
    c2pa_result = verify_content_credentials(image_data)
    metadata_status, metadata = _extract_provenance_metadata(image_data)
    digest = hashlib.sha256(image_data).hexdigest()
    web_status, payload, error_category, warnings = _request_web_detection(image_data)
    web_outcome = "success" if web_status == "success" else "no_results" if web_status == "no_results" else web_status
    if web_status == "success":
        web_status, payload, error_category, parse_warnings = _parse_web_detection(payload)
        warnings.extend(parse_warnings)
        web_outcome = "success" if web_status == "success" else "no_results" if web_status == "no_results" else web_status
    completed_methods = int(c2pa_result.status == "completed") + int(web_status in {"success", "no_results"})
    failed_methods = int(c2pa_result.status == "failed" or web_status == "failed")
    unavailable_methods = int(c2pa_result.status == "unavailable" or web_status == "unavailable")
    web_values = payload if isinstance(payload, dict) and web_status in {"success", "no_results"} else {}
    matching_pages = web_values.get("matching_pages", [])
    matching_images = web_values.get("matching_images", [])
    similar_images = web_values.get("similar_images", [])
    entities = web_values.get("web_entities", [])
    total = len(matching_pages) + len(matching_images) + len(similar_images)
    if completed_methods and (failed_methods or unavailable_methods):
        overall_status = "partial"
    elif completed_methods:
        overall_status = "success" if total or c2pa_result.manifest_present else "no_results"
    elif failed_methods:
        overall_status = "failed"
    else:
        overall_status = "unavailable"
    warnings.extend(c2pa_result.warnings)
    if metadata_status == "failed":
        warnings.append("EXIF metadata extraction could not complete.")
    return ProvenanceResult(
        status=overall_status, outcome="success" if overall_status == "success" else "no_results" if overall_status == "no_results" else overall_status,
        web_entities=entities[:50], matching_pages=matching_pages[:10],
        matching_images=matching_images[:10], similar_images=similar_images[:10],
        best_guess_labels=web_values.get("best_guess_labels", [])[:20], total_matches=total,
        runtime_ms=int((time.perf_counter()-started)*1000),
        error="Web Detection request could not complete." if web_status == "failed" else None,
        warnings=warnings, retrieved_at=web_values.get("retrieved_at"),
        content_credentials=c2pa_result.to_dict(), image_sha256=digest,
        metadata_status=metadata_status, metadata=metadata,
        web_detection_status=web_status, web_detection_outcome=web_outcome,
    )
