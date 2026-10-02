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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from backend.app.config import settings
from backend.app.core.logging import get_logger

log = get_logger("factcheck")

FACTCHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


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
        }


@dataclass
class FactCheckResult:
    """Complete fact-check search result."""
    status: str = "unavailable"  # unavailable | success | no_results | failed
    claims: List[FactCheckClaim] = field(default_factory=list)
    total_results: int = 0
    queries_performed: int = 0
    runtime_ms: int = 0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "claims": [c.to_dict() for c in self.claims],
            "total_results": self.total_results,
            "queries_performed": self.queries_performed,
            "runtime_ms": self.runtime_ms,
            "error": self.error,
            "warnings": self.warnings,
            "note": (
                "Fact-check results come from the Google Fact Check Tools API. "
                "They reflect third-party reviews, not TruthLens assessments."
            ),
        }


def _is_configured() -> bool:
    """Check if the Fact Check API is configured."""
    return bool(settings.GOOGLE_API_KEY or settings.FACTCHECK_API_KEY)


def _get_api_key() -> str:
    """Get the API key for the Fact Check API."""
    return settings.FACTCHECK_API_KEY or settings.GOOGLE_API_KEY


def _search_single_claim(
    query: str,
    api_key: str,
    language_code: str = "en",
    max_age_days: Optional[int] = None,
) -> List[FactCheckClaim]:
    """
    Search for fact-checks related to a single claim.

    Returns a list of FactCheckClaim objects from the API response.
    """
    params = {
        "key": api_key,
        "query": query,
        "languageCode": language_code,
    }
    if max_age_days:
        params["maxAgeDays"] = str(max_age_days)

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(FACTCHECK_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as e:
        log.error("Fact Check API HTTP error: %s", e)
        raise
    except httpx.RequestError as e:
        log.error("Fact Check API request error: %s", e)
        raise
    except Exception as e:
        log.error("Fact Check API unexpected error: %s", e)
        raise

    claims = []
    for item in data.get("claims", []):
        claim_text = item.get("text", "")
        claimant = item.get("claimant", "")
        claim_date = item.get("claimDate", "")

        for review in item.get("claimReview", []):
            publisher = review.get("publisher", {})
            claims.append(FactCheckClaim(
                query=query,
                claimed_text=claim_text,
                claimant=claimant,
                claim_date=claim_date,
                publisher_name=publisher.get("name", ""),
                publisher_site=publisher.get("site", ""),
                url=review.get("url", ""),
                title=review.get("title", ""),
                review_date=review.get("reviewDate", ""),
                textual_rating=review.get("textualRating", ""),
                language_code=review.get("languageCode", language_code),
            ))

    return claims


def search_fact_checks(
    claim_texts: List[str],
    language_code: str = "en",
    max_results_per_claim: int = 5,
) -> FactCheckResult:
    """
    Search for fact-checks across multiple claims.

    Args:
        claim_texts: List of claim strings to search
        language_code: Language code for the search
        max_results_per_claim: Max results to return per claim

    Returns:
        FactCheckResult with all found reviews
    """
    t0 = time.time()

    if not _is_configured():
        return FactCheckResult(
            status="unavailable",
            warnings=["Google Fact Check API key not configured. "
                      "Set GOOGLE_API_KEY or FACTCHECK_API_KEY in .env."],
            runtime_ms=int((time.time() - t0) * 1000),
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

    for claim_text in claim_texts[:10]:  # Cap at 10 queries
        if not claim_text or len(claim_text.strip()) < 5:
            continue

        try:
            results = _search_single_claim(
                query=claim_text.strip(),
                api_key=api_key,
                language_code=language_code,
            )
            all_claims.extend(results[:max_results_per_claim])
            queries_performed += 1
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                return FactCheckResult(
                    status="failed",
                    error="API key invalid or quota exceeded.",
                    runtime_ms=int((time.time() - t0) * 1000),
                )
            elif e.response.status_code == 429:
                warnings.append("Rate limit reached — some claims not checked.")
                break
            else:
                warnings.append(f"API error for claim: {e}")
        except Exception as e:
            warnings.append(f"Failed to check claim: {e}")

    runtime_ms = int((time.time() - t0) * 1000)

    status = "success" if all_claims else "no_results"

    result = FactCheckResult(
        status=status,
        claims=all_claims,
        total_results=len(all_claims),
        queries_performed=queries_performed,
        runtime_ms=runtime_ms,
        warnings=warnings,
    )

    log.info(
        "Fact-check: %d results from %d queries, status=%s [%d ms]",
        len(all_claims), queries_performed, status, runtime_ms,
    )

    return result
