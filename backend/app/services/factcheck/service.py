"""
TruthLens — Fact-check API integration.

Uses the official Google Fact Check Tools API to search
for existing fact-checks related to extracted claims.

Behavior:
- Configured + success  → returns real results
- Not configured        → returns status "unavailable"
- API failure           → returns status "failed"
- No results found      → returns status "no_results"

NEVER fabricates fact-check results. All data comes directly
from the API response.
"""

from __future__ import annotations

import time
import hashlib
import json
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from backend.app.config import settings
from backend.app.core.logging import get_logger

log = get_logger("factcheck")

FACTCHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


class ProviderResponseError(ValueError):
    """The provider returned a response that cannot be safely normalized."""


class ProviderResponseTooLarge(ProviderResponseError):
    pass


class ParsedReviews(list):
    def __init__(self, values=(), malformed_items: int = 0):
        super().__init__(values)
        self.malformed_items = malformed_items


_SENSITIVE_QUERY_KEYS = {
    "key", "api_key", "access_token", "token", "authorization", "signature", "sig",
    "client_secret", "password", "secret", "auth", "x_api_key", "refresh_token", "oauth_token",
}
_TRACKING_QUERY_KEYS = {"gclid", "fbclid", "mc_cid", "mc_eid", "ref", "ref_src"}


def _sanitize_review_url(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value or len(value) > 2048:
        return None
    if any(ord(char) < 0x20 or ord(char) == 0x7F or char.isspace() for char in value):
        return None
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        # Accessing port forces urllib to reject malformed port strings.
        _ = parsed.port
    except ValueError:
        return None
    if parsed.scheme.lower() != "https" or not hostname or parsed.username or parsed.password:
        return None
    try:
        parsed_query = parse_qsl(parsed.query, keep_blank_values=True, max_num_fields=100)
    except ValueError:
        return None
    query = []
    for key, val in parsed_query:
        normalized_key = re.sub(r"[-.]", "_", key.casefold())
        if normalized_key in _SENSITIVE_QUERY_KEYS or normalized_key in _TRACKING_QUERY_KEYS:
            continue
        if normalized_key.startswith("utm_"):
            continue
        query.append((key, val))
    return urlunsplit(("https", parsed.netloc, parsed.path, urlencode(query, doseq=True), ""))


def _read_bounded_body(response: Any, max_bytes: int) -> bytes:
    headers = getattr(response, "headers", {}) or {}
    content_length = headers.get("content-length") if hasattr(headers, "get") else None
    try:
        parsed_length = int(content_length) if content_length is not None else None
    except (TypeError, ValueError):
        parsed_length = None
    if parsed_length is not None and parsed_length > max_bytes:
        raise ProviderResponseTooLarge("Provider response exceeded the configured size limit")
    body = bytearray()
    for chunk in response.iter_bytes():
        if not isinstance(chunk, (bytes, bytearray)):
            raise ProviderResponseError("Provider returned malformed response bytes")
        if len(body) + len(chunk) > max_bytes:
            raise ProviderResponseTooLarge("Provider response exceeded the configured size limit")
        body.extend(chunk)
    return bytes(body)


@dataclass
class FactCheckClaim:
    """A single fact-check result from the API."""
    query: str = ""
    claimed_text: str = ""
    claimant: str = ""
    claim_date: str = ""
    publisher_name: str = ""
    publisher_site: str = ""
    url: str = ""
    title: str = ""
    review_date: str = ""
    textual_rating: str = ""
    language_code: str = ""
    claim_id: Optional[str] = None
    evidence_id: str = ""
    retrieved_at: str = ""
    source_type: str = "third_party_fact_check"
    claim_match: str = "unassessed"

    def __post_init__(self) -> None:
        if not self.evidence_id:
            identity = f"{self.url}|{self.claim_id or self.query}|{self.title}"
            self.evidence_id = "fc_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "claimed_text": self.claimed_text,
            "claimant": self.claimant,
            "claim_date": self.claim_date,
            "publisher_name": self.publisher_name,
            "publisher_site": self.publisher_site,
            "url": self.url,
            "title": self.title,
            "review_date": self.review_date,
            "textual_rating": self.textual_rating,
            "language_code": self.language_code,
            "claim_id": self.claim_id,
            "evidence_id": self.evidence_id,
            "retrieved_at": self.retrieved_at,
            "source_type": self.source_type,
            "claim_match": self.claim_match,
        }


@dataclass
class FactCheckResult:
    """Complete fact-check search result."""
    status: str = "unavailable"  # success | no_results | partial | unavailable | failed | skipped
    provider: str = "google_fact_check_tools"
    claims: List[FactCheckClaim] = field(default_factory=list)
    total_results: int = 0
    queries_performed: int = 0
    runtime_ms: int = 0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    claim_verifications: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "provider": self.provider,
            "claims": [c.to_dict() for c in self.claims],
            "total_results": self.total_results,
            "queries_performed": self.queries_performed,
            "runtime_ms": self.runtime_ms,
            "error": self.error,
            "warnings": self.warnings,
            "claim_verifications": self.claim_verifications,
            "note": (
                "Fact-check results come from the Google Fact Check Tools API. "
                "They reflect third-party reviews, not TruthLens assessments."
            ),
        }


def _is_configured() -> bool:
    """Check if the Fact Check API is configured."""
    for attr in ("GOOGLE_FACT_CHECK_API_KEY", "FACTCHECK_API_KEY", "GOOGLE_API_KEY"):
        val = getattr(settings, attr, "")
        if isinstance(val, str) and val.strip():
            return True
    return False


def _get_api_key() -> str:
    """Get the API key for the Fact Check API."""
    for attr in ("GOOGLE_FACT_CHECK_API_KEY", "FACTCHECK_API_KEY", "GOOGLE_API_KEY"):
        val = getattr(settings, attr, "")
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _bounded_setting(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(getattr(settings, name, default))
        if value != value:  # NaN
            return default
        return min(maximum, max(minimum, value))
    except (TypeError, ValueError):
        return default


def _search_single_claim(
    query: str,
    api_key: str,
    language_code: Optional[str] = None,
    max_age_days: Optional[int] = None,
    review_publisher_site_filter: Optional[str] = None,
    claim_id: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
) -> List[FactCheckClaim]:
    """
    Search for fact-checks related to a single claim.

    Returns a list of FactCheckClaim objects from the API response.
    """
    params = {
        "key": api_key,
        "query": query,
    }
    if language_code and re.fullmatch(r"[a-z]{2,3}(?:-[A-Z]{2})?", language_code):
        params["languageCode"] = language_code
    if max_age_days and isinstance(max_age_days, int) and max_age_days > 0:
        params["maxAgeDays"] = str(max_age_days)
    if review_publisher_site_filter and isinstance(review_publisher_site_filter, str):
        cleaned_filter = review_publisher_site_filter.strip().lower()
        if re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}", cleaned_filter):
            params["reviewPublisherSiteFilter"] = cleaned_filter

    try:
        timeout = timeout_seconds or _bounded_setting("FACTCHECK_API_TIMEOUT_SECONDS", 10.0, 0.1, 60.0)
        max_bytes = int(_bounded_setting("FACTCHECK_MAX_RESPONSE_BYTES", 1_000_000, 1024, 10_000_000))
        with httpx.Client(timeout=timeout) as client:
            with client.stream("GET", FACTCHECK_API_URL, params=params) as response:
                response.raise_for_status()
                body = _read_bounded_body(response, max_bytes)
        try:
            data = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderResponseError("Provider returned invalid JSON") from exc
        if not isinstance(data, dict) or ("claims" in data and not isinstance(data["claims"], list)):
            raise ProviderResponseError("Provider returned a malformed response")
    except httpx.HTTPStatusError as e:
        log.error("Fact Check API HTTP error (%s)", e.response.status_code)
        raise
    except httpx.RequestError as e:
        log.error("Fact Check API request failed (%s)", type(e).__name__)
        raise
    except Exception as e:
        log.error("Fact Check API unexpected error (%s)", type(e).__name__)
        raise

    claims = []
    malformed_items = 0
    retrieved_at = datetime.now(timezone.utc).isoformat()
    raw_claims = data.get("claims", [])
    if len(raw_claims) > 100:
        malformed_items += len(raw_claims) - 100
        raw_claims = raw_claims[:100]
    for item in raw_claims:
        if not isinstance(item, dict):
            malformed_items += 1
            continue
        claim_text = item.get("text", "")
        claimant = item.get("claimant", "")
        claim_date = item.get("claimDate", "")
        reviews = item.get("claimReview", [])
        if (not isinstance(claim_text, str) or not claim_text.strip() or len(claim_text) > 2000
                or not isinstance(claimant, str) or len(claimant) > 300
                or not isinstance(claim_date, str) or len(claim_date) > 40
                or not isinstance(reviews, list)):
            malformed_items += 1
            continue
        if len(reviews) > 100:
            malformed_items += len(reviews) - 100
            reviews = reviews[:100]

        for review in reviews:
            if not isinstance(review, dict):
                malformed_items += 1
                continue
            publisher = review.get("publisher", {})
            if not isinstance(publisher, dict):
                malformed_items += 1
                continue
            publisher_name = publisher.get("name", "")
            publisher_site = publisher.get("site", "")
            title = review.get("title", "")
            review_date = review.get("reviewDate", "")
            rating = review.get("textualRating", "")
            review_language = review.get("languageCode", language_code or "")
            if (not isinstance(publisher_name, str) or len(publisher_name) > 300
                    or not isinstance(publisher_site, str) or len(publisher_site) > 300
                    or not isinstance(title, str) or len(title) > 500
                    or not isinstance(review_date, str) or len(review_date) > 40
                    or not isinstance(rating, str) or len(rating) > 160
                    or not isinstance(review_language, str) or len(review_language) > 20):
                malformed_items += 1
                continue
            url = _sanitize_review_url(review.get("url"))
            if url is None:
                malformed_items += 1
                continue
            identity = f"{url}|{claim_id or query}|{title}"
            claims.append(FactCheckClaim(
                query=query,
                claimed_text=claim_text,
                claimant=claimant,
                claim_date=claim_date,
                publisher_name=publisher_name,
                publisher_site=publisher_site,
                url=url,
                title=title,
                review_date=review_date,
                textual_rating=rating,
                language_code=review_language,
                claim_id=claim_id,
                evidence_id="fc_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
                retrieved_at=retrieved_at,
                claim_match="strong_match" if _claim_match(query, claim_text) else "related_unconfirmed",
            ))

    if malformed_items and not claims and raw_claims:
        raise ProviderResponseError("Provider response contained no valid review records")
    return ParsedReviews(claims, malformed_items=malformed_items)


def search_fact_checks(
    claim_texts: List[str],
    language_code: Optional[str] = None,
    max_results_per_claim: int = 5,
    claim_ids: Optional[List[str]] = None,
    max_age_days: Optional[int] = None,
    review_publisher_site_filter: Optional[str] = None,
) -> FactCheckResult:
    """
    Search for fact-checks across multiple claims.

    Args:
        claim_texts: List of claim strings to search
        language_code: Language code for the search
        max_results_per_claim: Max results to return per claim
        claim_ids: Optional list of claim identifier UUIDs
        max_age_days: Optional maximum age of claims in days
        review_publisher_site_filter: Optional review publisher domain filter

    Returns:
        FactCheckResult with all found reviews
    """
    t0 = time.time()
    # Reject malformed/oversized input and collapse repeated normalized queries.
    inputs = []
    seen_queries = set()
    for index, text in enumerate(claim_texts or []):
        query = re.sub(r"\s+", " ", text).strip() if isinstance(text, str) else ""
        if not query or len(query) > 2000:
            continue
        key = query.casefold()
        if key in seen_queries:
            continue
        seen_queries.add(key)
        inputs.append((query, claim_ids[index] if claim_ids and index < len(claim_ids) else None))
    claim_texts = [item[0] for item in inputs]
    claim_ids = [item[1] for item in inputs]
    if language_code and not re.fullmatch(r"[a-z]{2,3}(?:-[A-Z]{2})?", language_code):
        language_code = None

    claim_ids = claim_ids or [None] * len(claim_texts)
    claim_ids = (claim_ids + [None] * len(claim_texts))[:len(claim_texts)]
    claim_limit = int(_bounded_setting("FACTCHECK_MAX_CLAIMS", 5, 1, 20))
    if not _is_configured():
        return FactCheckResult(
            status="unavailable",
            warnings=["Google Fact Check API key not configured. "
                      "Set GOOGLE_API_KEY or FACTCHECK_API_KEY in .env."],
            runtime_ms=int((time.time() - t0) * 1000),
            claim_verifications=[_verification(cid, text, "RETRIEVAL_UNAVAILABLE", [], "Fact-check provider is not configured.") for text, cid in zip(claim_texts[:claim_limit], claim_ids[:claim_limit])],
        )

    if not claim_texts:
        return FactCheckResult(
            status="no_results",
            warnings=["No claims provided for fact-checking."],
            runtime_ms=int((time.time() - t0) * 1000),
        )

    api_key = _get_api_key()
    all_claims: List[FactCheckClaim] = []
    queries_performed = 0
    warnings: List[str] = []
    if len(claim_texts) > claim_limit:
        warnings.append(f"Claim retrieval capped at {claim_limit} queries.")

    attempted = 0
    succeeded = 0
    failed_queries = 0
    timed_out = False
    partial_data = False
    per_query_status: Dict[str, str] = {}
    per_query_errors: Dict[str, str] = {}
    result_limit = min(10, max(1, int(max_results_per_claim)))
    retry_attempts = int(_bounded_setting("FACTCHECK_RETRY_ATTEMPTS", 0, 0, 2))
    deadline = time.monotonic() + _bounded_setting("FACTCHECK_MAX_TOTAL_SECONDS", 25, 1, 120)
    for claim_text, claim_id in list(zip(claim_texts, claim_ids))[:claim_limit]:
        if not claim_text or len(claim_text.strip()) < 5:
            continue

        try:
            if time.monotonic() >= deadline:
                warnings.append("Total fact-check retrieval time limit reached.")
                break
            attempted += 1
            results = None
            for attempt in range(retry_attempts + 1):
                try:
                    results = _search_single_claim(
                        query=claim_text.strip(), api_key=api_key,
                        language_code=language_code,
                        max_age_days=max_age_days,
                        review_publisher_site_filter=review_publisher_site_filter,
                        claim_id=claim_id,
                        timeout_seconds=max(0.1, min(
                            _bounded_setting("FACTCHECK_API_TIMEOUT_SECONDS", 10.0, 0.1, 60.0),
                            deadline - time.monotonic(),
                        )),
                    )
                    break
                except httpx.HTTPStatusError as retry_error:
                    if retry_error.response.status_code not in {429, 500, 502, 503, 504} or attempt >= retry_attempts or time.monotonic() >= deadline:
                        if retry_error.response.status_code in {429, 500, 502, 503, 504}:
                            timed_out = timed_out or retry_error.response.status_code == 429
                        raise
                except httpx.RequestError:
                    if attempt >= retry_attempts or time.monotonic() >= deadline:
                        raise
                time.sleep(min(0.25 * (2 ** attempt), 1.0))
            if results is None:
                raise RuntimeError("Fact-check retrieval timed out before a request completed")
            if getattr(results, "malformed_items", 0):
                partial_data = True
                warnings.append("Some malformed provider records were ignored.")
            for review in results:
                if claim_id and review.claim_id is None:
                    review.claim_id = claim_id
                    identity = f"{review.url}|{claim_id}|{review.title}"
                    review.evidence_id = "fc_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
                if review.claimed_text:
                    review.claim_match = "strong_match" if _claim_match(claim_text.strip(), review.claimed_text) else "related_unconfirmed"
            all_claims.extend(results[:result_limit])
            queries_performed += 1
            succeeded += 1
            per_query_status[claim_id or claim_text] = "success"
        except ProviderResponseTooLarge:
            failed_queries += 1
            per_query_errors[claim_id or claim_text] = "PROVIDER_RESPONSE_TOO_LARGE"
            warnings.append("Fact-check provider response exceeded the configured size limit.")
        except ProviderResponseError:
            failed_queries += 1
            per_query_errors[claim_id or claim_text] = "PROVIDER_INVALID_RESPONSE"
            warnings.append("Fact-check provider returned an invalid response.")
        except httpx.HTTPStatusError as e:
            per_query_errors[claim_id or claim_text] = (
                "PROVIDER_RATE_LIMIT" if e.response.status_code == 429
                else "PROVIDER_AUTH_OR_QUOTA" if e.response.status_code == 403
                else "PROVIDER_HTTP_ERROR"
            )
            if e.response.status_code == 403:
                failed_queries += 1
                warnings.append("Fact-check provider rejected credentials or quota for a query.")
            elif e.response.status_code == 429:
                failed_queries += 1
                timed_out = True
                warnings.append("Rate limit reached — some claims not checked.")
                break
            else:
                failed_queries += 1
                warnings.append(f"Fact-check request failed with HTTP {e.response.status_code}.")
        except Exception as e:
            failed_queries += 1
            per_query_errors[claim_id or claim_text] = "PROVIDER_TIMEOUT" if isinstance(e, httpx.TimeoutException) else "PROVIDER_REQUEST_ERROR"
            warnings.append(f"Fact-check request failed ({type(e).__name__}).")

    runtime_ms = int((time.time() - t0) * 1000)

    truncated = len(claim_texts) > claim_limit
    if succeeded == attempted and not truncated and not partial_data:
        status = "success" if all_claims else "no_results"
    elif succeeded or truncated or partial_data or failed_queries and (succeeded or attempted > failed_queries):
        status = "partial"
    elif attempted:
        status = "failed"
    else:
        status = "no_results"
    verifications = []
    for index, (text, claim_id) in enumerate(zip(claim_texts, claim_ids)):
        if index >= claim_limit:
            verifications.append({
                "claim_id": claim_id, "original_claim": text,
                "verdict": "INSUFFICIENT_EVIDENCE",
                "reasoning_summary": "Configured per-image claim retrieval limit reached.",
                "supporting_evidence_ids": [], "contradicting_evidence_ids": [],
                "conflicting_evidence_ids": [], "verification_status": "COMPLETED",
                "confidence": None, "limitations": ["External lookup was not attempted because the configured claim cap was reached."],
            })
            continue
        if not text or len(text.strip()) < 5:
            verifications.append({
                "claim_id": claim_id, "original_claim": text,
                "verdict": "UNVERIFIABLE", "reasoning_summary": "The text is too short to form a useful fact-check query.",
                "supporting_evidence_ids": [], "contradicting_evidence_ids": [],
                "conflicting_evidence_ids": [], "verification_status": "COMPLETED",
                "confidence": None, "limitations": ["No external lookup was performed for this short text."],
            })
            continue
        related = [item for item in all_claims if (claim_id and item.claim_id == claim_id) or (not claim_id and item.query == text)]
        retrieval_status = per_query_status.get(claim_id or text)
        verdict, supporting, contradicting, conflicting, reason = _assess_ratings(text, related)
        if retrieval_status is None:
            verdict, reason = "RETRIEVAL_UNAVAILABLE", "Fact-check retrieval did not complete for this claim."
        verifications.append({
            "claim_id": claim_id, "original_claim": text, "verdict": verdict,
            "publisher_rating_summary": verdict,
            "verdict_semantics": "summary_of_third_party_publisher_ratings_not_truthlens_finding",
            "reasoning_summary": reason, "supporting_evidence_ids": supporting,
            "contradicting_evidence_ids": contradicting, "conflicting_evidence_ids": conflicting,
            "verification_status": "COMPLETED" if retrieval_status else "UNAVAILABLE",
            "retrieval_error_category": per_query_errors.get(claim_id or text),
            "confidence": None,
            "limitations": ["Verdict summarizes third-party fact-check ratings; it is not an independent TruthLens investigation."],
        })

    result = FactCheckResult(
        status=status,
        claims=all_claims,
        total_results=len(all_claims),
        queries_performed=queries_performed,
        runtime_ms=runtime_ms,
        warnings=warnings,
        error="Fact-check provider rate limit reached." if timed_out and succeeded == 0 else None,
        claim_verifications=verifications,
    )

    log.info(
        "Fact-check: %d results from %d queries, status=%s [%d ms]",
        len(all_claims), queries_performed, status, runtime_ms,
    )

    return result


def _verification(claim_id: Optional[str], text: str, verdict: str, evidence: List[str], reason: str) -> Dict[str, Any]:
    return {"claim_id": claim_id, "original_claim": text, "verdict": verdict,
            "reasoning_summary": reason, "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [], "conflicting_evidence_ids": evidence,
            "verification_status": "UNAVAILABLE", "confidence": None,
            "limitations": ["External fact-check retrieval was unavailable."]}


def _normalized_words(text: str) -> set[str]:
    return set(re.findall(r"[\w]+", text.casefold()))


def _claim_match(query: str, reviewed_claim: str) -> bool:
    def canonical(value: str) -> str:
        return " ".join(re.findall(r"[\w]+", value.casefold()))

    if canonical(query) and canonical(query) == canonical(reviewed_claim):
        return True
    query_words = _normalized_words(query)
    reviewed_words = _normalized_words(reviewed_claim)
    query_numbers = set(re.findall(r"\d[\d,.%]*", query))
    reviewed_numbers = set(re.findall(r"\d[\d,.%]*", reviewed_claim))
    overlap = len(query_words & reviewed_words) / max(1, len(query_words))
    tokens = re.findall(r"[A-Z][\w-]*", query)
    proper_names = {token.casefold() for token in tokens[1:] if token.casefold() not in {"the", "a", "an", "on", "in", "at", "from", "according", "government"}}
    shared_names = reviewed_words
    has_anchor = bool(query_numbers or proper_names)
    anchor_match = (not query_numbers or query_numbers.issubset(reviewed_numbers)) and (not proper_names or proper_names.issubset(shared_names))
    return bool(reviewed_claim) and overlap >= 0.7 and has_anchor and anchor_match


def _assess_ratings(query: str, reviews: List[FactCheckClaim]):
    """Conservative aggregation of actual third-party ratings.

    Require strong lexical overlap and matching numbers before using a review;
    this gate avoids treating a merely related search hit as a verdict.
    """
    relevant = []
    for review in reviews:
        if not _claim_match(query, review.claimed_text):
            continue
        rating = " ".join(re.sub(r"[^a-z ]", " ", review.textual_rating.casefold()).split())
        if rating in {"false", "mostly false", "fake", "incorrect", "misleading", "pants on fire", "out of context", "incorrect context"}:
            label = "CONTRADICTED"
        elif rating in {"mixed", "half true", "partly true", "partially true", "mixture"}:
            label = "MIXED"
        elif rating in {"true", "mostly true", "accurate", "correct", "correct context"}:
            label = "SUPPORTED"
        else:
            label = "INSUFFICIENT_EVIDENCE"
        relevant.append((review, label))
    if not relevant:
        return "INSUFFICIENT_EVIDENCE", [], [], [], "No sufficiently claim-matched fact-check review with an interpretable rating was returned."
    labels = {label for _, label in relevant if label != "INSUFFICIENT_EVIDENCE"}
    supporting = [r.evidence_id for r, label in relevant if label == "SUPPORTED"]
    contradicting = [r.evidence_id for r, label in relevant if label == "CONTRADICTED"]
    mixed = [r.evidence_id for r, label in relevant if label == "MIXED"]
    if "SUPPORTED" in labels and "CONTRADICTED" in labels:
        return "MIXED", supporting, contradicting, [r.evidence_id for r, _ in relevant], "Relevant third-party reviews contain conflicting ratings."
    if "MIXED" in labels:
        return "MIXED", supporting, contradicting, mixed, "Relevant third-party review rates the claim as mixed or partly true."
    if "SUPPORTED" in labels:
        return "SUPPORTED", supporting, [], [], "Relevant third-party fact-check review(s) rate the claim as true or accurate."
    if "CONTRADICTED" in labels:
        return "CONTRADICTED", [], contradicting, [], "Relevant third-party fact-check review(s) rate the claim as false or misleading."
    return "INSUFFICIENT_EVIDENCE", [], [], [], "Relevant reviews were returned, but their ratings do not map to a supported verdict."
