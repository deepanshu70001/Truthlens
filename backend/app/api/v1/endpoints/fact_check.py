"""
TruthLens — Dedicated Standalone Fact-Checking API endpoint.

POST /api/v1/fact-check
"""

from __future__ import annotations

import asyncio
import time
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Request

from backend.app.config import settings
from backend.app.core.auth import Principal, get_current_principal
from backend.app.core.logging import get_logger
from backend.app.core.rate_limit import check_rate_limit
from backend.app.schemas.factcheck import (
    ClaimVerificationItem,
    FactCheckClaimItem,
    FactCheckRequest,
    FactCheckResponse,
)
from backend.app.services.claims.extractor import extract_claims
from backend.app.services.factcheck import service as factcheck_service
from backend.app.services.factcheck.cache import (
    compute_factcheck_cache_key,
    get_factcheck_cache,
)
from backend.app.services.factcheck.service import (
    FactCheckClaim,
    FactCheckResult,
    _assess_ratings,
)

log = get_logger("api.factcheck")
router = APIRouter()


@router.post(
    "/fact-check",
    response_model=FactCheckResponse,
    summary="Query third-party fact-check reviews for verifiable claims",
    description=(
        "Retrieves published third-party fact-check reviews matching submitted factual claims "
        "using the Google Fact Check Tools API and local persistent cache. Claims are gated "
        "through deterministic eligibility screening (excluding questions, opinions, and slogans). "
        "Results retain full publisher attribution and review URLs. "
        "Third-party reviews reflect independent publisher ratings, not TruthLens assessments."
    ),
)
async def factcheck_endpoint(
    request: FactCheckRequest,
    http_request: Request,
    principal: Principal = Depends(get_current_principal),
) -> FactCheckResponse:
    # 1. Rate limiting
    await check_rate_limit(
        http_request,
        scope="fact-check",
        limit=settings.RATE_LIMIT_FACTCHECK_PER_MINUTE,
        principal=principal,
    )

    t0 = time.time()
    cache = get_factcheck_cache()

    # 2. Gate incoming claims through deterministic eligibility filtering
    prepared = []
    eligible = []
    for request_index, raw_claim in enumerate(request.claims):
        # Extract claim proposition while preserving punctuation and numbers
        extraction = extract_claims([raw_claim], max_claims=1)
        claim = extraction.claims[0] if extraction.claims else None
        if claim is None:
            prepared.append({
                "request_index": request_index,
                "original_claim": raw_claim,
                "normalized_claim": " ".join(raw_claim.split()),
                "verification_status": "SKIPPED",
                "verdict": "UNVERIFIABLE",
                "eligibility_status": "INELIGIBLE",
                "eligibility_reason": "No usable claim text was produced.",
                "confidence": None,
                "cache_hit": False,
            })
            continue

        entry = {"request_index": request_index, "claim": claim}
        prepared.append(entry)
        if claim.verification_eligibility == "ELIGIBLE":
            eligible.append(entry)
        else:
            entry["ineligible_result"] = {
                "request_index": request_index,
                "claim_id": claim.claim_id,
                "original_claim": claim.original_text,
                "normalized_claim": claim.normalized_text,
                "verification_status": "SKIPPED",
                "verdict": "UNVERIFIABLE",
                "eligibility_status": "INELIGIBLE",
                "eligibility_reason": claim.eligibility_reason,
                "confidence": None,
                "cache_hit": False,
                "limitations": ["No provider lookup was performed for this ineligible claim."],
            }

    # 3. If no claims are eligible, return clean skipped response without querying provider or cache
    if not eligible:
        runtime_ms = int((time.time() - t0) * 1000)
        ordered_verifications: List[ClaimVerificationItem] = []
        for entry in prepared:
            if "ineligible_result" in entry:
                ordered_verifications.append(ClaimVerificationItem(**entry["ineligible_result"]))
            elif "claim" not in entry:
                ordered_verifications.append(ClaimVerificationItem(**entry))

        return FactCheckResponse(
            status="skipped",
            provider="google_fact_check_tools",
            claims=[],
            total_results=0,
            queries_performed=0,
            cached_queries_count=0,
            runtime_ms=runtime_ms,
            warnings=["No eligible claims were submitted; the provider was not queried."],
            claim_verifications=ordered_verifications,
            note=(
                "Fact-check results come from the Google Fact Check Tools API. "
                "They reflect third-party reviews, not TruthLens assessments."
            ),
        )

    # 4. Process eligible claims with cache-aside and in-flight deduplication
    all_claims: List[FactCheckClaim] = []
    queries_performed = 0
    cached_queries_count = 0
    warnings: List[str] = []
    claim_verifications_map: Dict[str, Dict[str, Any]] = {}

    # Check cache first for each eligible claim
    uncached_entries = []
    for entry in eligible:
        claim = entry["claim"]
        query_text = claim.normalized_text
        claim_id = claim.claim_id
        cache_key = compute_factcheck_cache_key(query_text, request.language_code)

        cached = cache.get(cache_key)
        if cached is not None:
            cached_claims, cached_status = cached
            cached_queries_count += 1
            for rc in cached_claims:
                if not rc.claim_id and claim_id:
                    rc.claim_id = claim_id
            all_claims.extend(cached_claims[:request.max_results_per_claim])

            verdict, supporting, contradicting, conflicting, reason = _assess_ratings(query_text, cached_claims)
            claim_verifications_map[claim_id] = {
                "request_index": entry["request_index"],
                "claim_id": claim_id,
                "original_claim": claim.original_text,
                "normalized_claim": claim.normalized_text,
                "eligibility_status": "ELIGIBLE",
                "eligibility_reason": claim.eligibility_reason,
                "verification_status": "COMPLETED",
                "verdict": verdict,
                "publisher_rating_summary": verdict,
                "verdict_semantics": "summary_of_third_party_publisher_ratings_not_truthlens_finding",
                "reasoning_summary": reason,
                "supporting_evidence_ids": supporting,
                "contradicting_evidence_ids": contradicting,
                "conflicting_evidence_ids": conflicting,
                "cache_hit": True,
                "confidence": None,
                "limitations": ["Verdict summarizes third-party fact-check ratings; it is not an independent TruthLens investigation."],
            }
        else:
            uncached_entries.append((entry, cache_key))

    provider_status: Optional[str] = None
    if uncached_entries:
        unique_keys = list(dict.fromkeys(k for _, k in uncached_entries))
        async with AsyncExitStack() as stack:
            for k in unique_keys:
                lock = await cache.get_in_flight_lock(k)
                await stack.enter_async_context(lock)

            still_uncached = []
            for entry, cache_key in uncached_entries:
                claim = entry["claim"]
                query_text = claim.normalized_text
                claim_id = claim.claim_id
                re_cached = cache.get(cache_key)
                if re_cached is not None:
                    cached_claims, cached_status = re_cached
                    cached_queries_count += 1
                    for rc in cached_claims:
                        if not rc.claim_id and claim_id:
                            rc.claim_id = claim_id
                    all_claims.extend(cached_claims[:request.max_results_per_claim])
                    verdict, supporting, contradicting, conflicting, reason = _assess_ratings(query_text, cached_claims)
                    claim_verifications_map[claim_id] = {
                        "request_index": entry["request_index"],
                        "claim_id": claim_id,
                        "original_claim": claim.original_text,
                        "normalized_claim": claim.normalized_text,
                        "eligibility_status": "ELIGIBLE",
                        "eligibility_reason": claim.eligibility_reason,
                        "verification_status": "COMPLETED",
                        "verdict": verdict,
                        "publisher_rating_summary": verdict,
                        "verdict_semantics": "summary_of_third_party_publisher_ratings_not_truthlens_finding",
                        "reasoning_summary": reason,
                        "supporting_evidence_ids": supporting,
                        "contradicting_evidence_ids": contradicting,
                        "conflicting_evidence_ids": conflicting,
                        "cache_hit": True,
                        "confidence": None,
                        "limitations": ["Verdict summarizes third-party fact-check ratings; it is not an independent TruthLens investigation."],
                    }
                else:
                    still_uncached.append((entry, cache_key))

            if still_uncached:
                queries_to_send = [e["claim"].normalized_text for e, _ in still_uncached]
                ids_to_send = [e["claim"].claim_id for e, _ in still_uncached]

                def _fetch():
                    return factcheck_service.search_fact_checks(
                        queries_to_send,
                        language_code=request.language_code,
                        max_results_per_claim=request.max_results_per_claim,
                        claim_ids=ids_to_send,
                        max_age_days=request.max_age_days,
                        review_publisher_site_filter=request.review_publisher_site_filter,
                    )

                try:
                    sub_result: FactCheckResult = await asyncio.to_thread(_fetch)
                    provider_status = sub_result.status
                    queries_performed += sub_result.queries_performed
                    warnings.extend(sub_result.warnings)
                    all_claims.extend(sub_result.claims)

                    v_by_id = {v["claim_id"]: v for v in sub_result.claim_verifications if "claim_id" in v}
                    for entry, cache_key in still_uncached:
                        cid = entry["claim"].claim_id
                        q_text = entry["claim"].normalized_text
                        q_claims = [c for c in sub_result.claims if c.claim_id == cid or c.query == q_text]

                        if sub_result.status in ("success", "no_results"):
                            cache.set(
                                cache_key=cache_key,
                                query=q_text,
                                claims=q_claims,
                                status=sub_result.status,
                                language_code=request.language_code,
                                provider="google_fact_check_tools",
                            )

                        if cid in v_by_id:
                            v_item = dict(v_by_id[cid])
                            v_item.update({
                                "request_index": entry["request_index"],
                                "cache_hit": False,
                                "eligibility_status": "ELIGIBLE",
                                "eligibility_reason": entry["claim"].eligibility_reason,
                            })
                            claim_verifications_map[cid] = v_item
                        else:
                            claim_verifications_map[cid] = {
                                "request_index": entry["request_index"],
                                "claim_id": cid,
                                "original_claim": entry["claim"].original_text,
                                "normalized_claim": q_text,
                                "eligibility_status": "ELIGIBLE",
                                "eligibility_reason": entry["claim"].eligibility_reason,
                                "verification_status": "COMPLETED" if sub_result.status in ("success", "no_results") else "UNAVAILABLE",
                                "verdict": "INSUFFICIENT_EVIDENCE" if sub_result.status in ("success", "no_results") else "RETRIEVAL_UNAVAILABLE",
                                "cache_hit": False,
                                "confidence": None,
                            }
                except Exception as exc:
                    log.error("Provider query failed for claims (%s): %s", ids_to_send, exc)
                    provider_status = "failed"
                    for entry, _ in still_uncached:
                        cid = entry["claim"].claim_id
                        claim_verifications_map[cid] = {
                            "request_index": entry["request_index"],
                            "claim_id": cid,
                            "original_claim": entry["claim"].original_text,
                            "normalized_claim": entry["claim"].normalized_text,
                            "eligibility_status": "ELIGIBLE",
                            "eligibility_reason": entry["claim"].eligibility_reason,
                            "verification_status": "UNAVAILABLE",
                            "verdict": "RETRIEVAL_UNAVAILABLE",
                            "cache_hit": False,
                            "confidence": None,
                            "retrieval_error_category": "PROVIDER_REQUEST_ERROR",
                            "reasoning_summary": "Provider request failed.",
                        }

    # 5. Assemble ordered verifications matching original request indices
    ordered_verifications = []
    for entry in prepared:
        if "ineligible_result" in entry:
            ordered_verifications.append(ClaimVerificationItem(**entry["ineligible_result"]))
        elif "claim" not in entry:
            ordered_verifications.append(ClaimVerificationItem(**entry))
        else:
            cid = entry["claim"].claim_id
            v_dict = claim_verifications_map.get(cid, {})
            ordered_verifications.append(ClaimVerificationItem(**v_dict))

    # 6. Determine overall execution status
    ineligible_count = sum("ineligible_result" in entry or "claim" not in entry for entry in prepared)
    if ineligible_count:
        warnings.append(f"{ineligible_count} ineligible claim(s) were skipped before provider submission.")

    # Deduplicate review items by evidence_id
    seen_evidence = set()
    deduped_claims: List[FactCheckClaimItem] = []
    for c in all_claims:
        if c.evidence_id not in seen_evidence:
            seen_evidence.add(c.evidence_id)
            deduped_claims.append(FactCheckClaimItem(**c.to_dict()))

    completed_eligible = sum(1 for v in ordered_verifications if v.eligibility_status == "ELIGIBLE" and v.verification_status == "COMPLETED")
    has_failed_eligible = any(v.eligibility_status == "ELIGIBLE" and (v.verification_status == "UNAVAILABLE" or v.retrieval_error_category is not None) for v in ordered_verifications)
    total_eligible = len(eligible)

    if provider_status == "unavailable" and cached_queries_count == 0:
        overall_status = "unavailable"
    elif provider_status == "failed" and cached_queries_count == 0:
        overall_status = "failed"
    elif total_eligible > 0 and completed_eligible == total_eligible:
        overall_status = "success" if deduped_claims else "no_results"
    elif completed_eligible > 0:
        overall_status = "partial"
    elif provider_status is not None:
        overall_status = provider_status
    elif has_failed_eligible:
        overall_status = "failed"
    else:
        overall_status = "no_results"

    runtime_ms = int((time.time() - t0) * 1000)

    return FactCheckResponse(
        status=overall_status,
        provider="google_fact_check_tools",
        claims=deduped_claims,
        total_results=len(deduped_claims),
        queries_performed=queries_performed,
        cached_queries_count=cached_queries_count,
        runtime_ms=runtime_ms,
        warnings=list(dict.fromkeys(warnings)),
        claim_verifications=ordered_verifications,
        note=(
            "Fact-check results come from the Google Fact Check Tools API. "
            "They reflect third-party reviews, not TruthLens assessments."
        ),
    )
