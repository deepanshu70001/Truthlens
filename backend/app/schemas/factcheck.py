"""
TruthLens — Pydantic schemas for the Standalone Fact-Check endpoint.

POST /api/v1/fact-check
"""

from __future__ import annotations

from typing import Annotated, Any, List, Optional
from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator


class FactCheckRequest(BaseModel):
    """Request payload for POST /api/v1/fact-check."""

    claims: List[
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=5, max_length=2000),
        ]
    ] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of candidate claim strings to search against third-party fact-check reviews (1 to 20 items)",
    )
    language_code: Optional[str] = Field(
        None,
        min_length=2,
        max_length=5,
        pattern=r"^[a-z]{2,3}(-[A-Z]{2})?$",
        description="Optional ISO 639-1 language code (e.g. 'en', 'hi')",
    )
    max_results_per_claim: int = Field(
        5,
        ge=1,
        le=10,
        description="Maximum number of review matches to return per claim (clamped between 1 and 10)",
    )

    @field_validator("language_code")
    @classmethod
    def validate_language_code(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        from backend.app.services.nlp.language_detector import LANGUAGE_NAMES

        if value not in LANGUAGE_NAMES:
            supported = ", ".join(sorted(LANGUAGE_NAMES.keys()))
            raise ValueError(
                f"Unsupported language code '{value}'. Supported language codes: {supported}"
            )
        return value


class FactCheckClaimItem(BaseModel):
    """A matched third-party fact-check review from the external provider."""

    query: str = Field(..., description="Query string used for the search")
    claimed_text: str = Field(..., description="Exact claim text reviewed by the fact-checker")
    claimant: str = Field("", description="Person or entity that made the original claim, if identified")
    claim_date: str = Field("", description="Date the original claim was made, if recorded")
    publisher_name: str = Field(..., description="Name of the fact-checking organization")
    publisher: Optional[str] = Field(None, description="Alias for publisher_name")
    publisher_site: str = Field("", description="Domain or website of the publisher")
    url: str = Field(..., description="Sanitized HTTPS URL to the full published fact-check review")
    title: str = Field("", description="Headline or title of the published fact-check review")
    review_date: str = Field("", description="Date the review was published, if supplied")
    textual_rating: str = Field(..., description="Rating or verdict exactly as stated by the third-party publisher")
    rating_category: Optional[str] = Field(None, description="Categorized rating indicator (e.g. true, false, mixture)")
    language_code: str = Field("", description="Language code of the review")
    language: Optional[str] = Field(None, description="Alias for language_code")
    claim_id: Optional[str] = Field(None, description="Associated claim identifier")
    evidence_id: str = Field(..., description="Deterministic unique identifier for this evidence item")
    retrieved_at: str = Field(..., description="ISO-8601 UTC timestamp when this review was retrieved")
    source_type: str = Field("third_party_fact_check", description="Source modality identifier")
    claim_match: str = Field(
        "unassessed",
        description="Match degree between query and reviewed claim ('strong_match', 'related_unconfirmed')",
    )

    @model_validator(mode="before")
    @classmethod
    def harmonize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "publisher_name" not in data and "publisher" in data:
                data["publisher_name"] = data["publisher"]
            if "publisher" not in data and "publisher_name" in data:
                data["publisher"] = data["publisher_name"]
            if "language_code" not in data and "language" in data:
                data["language_code"] = data["language"]
            if "language" not in data and "language_code" in data:
                data["language"] = data["language_code"]
        return data


class ClaimVerificationItem(BaseModel):
    """Per-claim verification summary and eligibility assessment."""

    request_index: Optional[int] = Field(None, description="0-based index of claim in request list")
    claim_id: Optional[str] = Field(None, description="Deterministic claim identifier")
    original_claim: str = Field(..., description="Original claim text")
    normalized_claim: Optional[str] = Field(None, description="Normalized claim text")
    eligibility_status: str = Field(
        ...,
        description="Eligibility status: 'ELIGIBLE' or 'INELIGIBLE'",
    )
    eligibility_reason: str = Field(
        "",
        description="Reason explaining eligibility or why external search was skipped",
    )
    verification_status: str = Field(
        ...,
        description="Verification lifecycle status: 'COMPLETED', 'SKIPPED', or 'UNAVAILABLE'",
    )
    verdict: str = Field(
        ...,
        description="Summary verdict: publisher consensus, 'INSUFFICIENT_EVIDENCE', 'UNVERIFIABLE', or 'RETRIEVAL_UNAVAILABLE'",
    )
    publisher_rating_summary: Optional[str] = Field(
        None,
        description="Consensus summary of third-party publisher ratings",
    )
    verdict_semantics: Optional[str] = Field(
        None,
        description="Explicit caveat that verdict reflects third-party publisher ratings, not TruthLens ground truth",
    )
    reasoning_summary: Optional[str] = Field(
        None,
        description="Human-readable explanation of verification outcome",
    )
    confidence: Optional[float] = Field(
        None,
        description="Confidence score if applicable (null for third-party summaries)",
    )
    cache_hit: bool = Field(
        False,
        description="Whether this claim result was served from persistent cache",
    )
    supporting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="Evidence IDs supporting the claim",
    )
    contradicting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="Evidence IDs contradicting the claim",
    )
    conflicting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="Evidence IDs showing conflicting reviews",
    )
    retrieval_error_category: Optional[str] = Field(
        None,
        description="Specific error category if retrieval failed (e.g. 'PROVIDER_TIMEOUT', 'PROVIDER_RATE_LIMIT')",
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Caveats and limitations applicable to this verification",
    )


class FactCheckResponse(BaseModel):
    """Response payload for POST /api/v1/fact-check."""

    status: str = Field(
        ...,
        description="Overall execution status: 'success', 'no_results', 'partial', 'unavailable', 'failed', or 'skipped'",
    )
    provider: str = Field(
        "google_fact_check_tools",
        description="Primary fact-check data provider",
    )
    claims: List[FactCheckClaimItem] = Field(
        default_factory=list,
        description="List of matched review records retrieved from the provider",
    )
    total_results: int = Field(0, description="Total review records returned")
    queries_performed: int = Field(0, description="Number of distinct provider API calls made")
    cached_queries_count: int = Field(0, description="Number of queries served from persistent cache")
    runtime_ms: int = Field(0, description="Total execution duration in milliseconds")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    warnings: List[str] = Field(
        default_factory=list,
        description="Operational warnings or caveats",
    )
    claim_verifications: List[ClaimVerificationItem] = Field(
        default_factory=list,
        description="Structured per-claim verification results preserving original request order",
    )
    note: Optional[str] = Field(
        None,
        description="Standard attribution and disclaimer note regarding third-party fact-check data",
    )
