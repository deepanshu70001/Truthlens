"""
TruthLens — Core API Integration & Workflow Tests.

Verifies the end-to-end chaining of the four core APIs:
1. GET /api/v1/models (Registry introspection)
2. POST /api/v1/ocr (Text extraction from image upload)
3. POST /api/v1/claims (Candidate claim extraction from OCR text)
4. POST /api/v1/fact-check (Third-party review retrieval for eligible claims)
"""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest
from PIL import Image

from backend.app.schemas.claims import ClaimExtractionResponse
from backend.app.schemas.factcheck import FactCheckResponse
from backend.app.schemas.models import ModelRegistryResponse
from backend.app.schemas.ocr import OCRResponse
from backend.app.services.factcheck.service import FactCheckClaim, FactCheckResult
from backend.app.services.ocr.service import OCRBlock, OCRResult


def _create_test_image() -> bytes:
    """Generate minimal valid PNG bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_four_core_apis_integrated_workflow(client, mock_optional_analysis_modules):
    """Step-by-step verification of the 4 core APIs in sequence:
    Models -> OCR -> Claims -> Fact-Check.
    """
    # -------------------------------------------------------------
    # Step 1: Model Registry introspection (GET /api/v1/models)
    # -------------------------------------------------------------
    resp_models = await client.get("/api/v1/models")
    assert resp_models.status_code == 200
    models_data = ModelRegistryResponse.model_validate(resp_models.json())
    assert models_data.status == "ok"
    assert models_data.total_models == 5
    registered_ids = {m.id for m in models_data.models}
    assert "ocr_easyocr" in registered_ids
    assert "authenticity" in registered_ids

    # -------------------------------------------------------------
    # Step 2: Standalone OCR (POST /api/v1/ocr)
    # -------------------------------------------------------------
    image_bytes = _create_test_image()
    mock_optional_analysis_modules["ocr"].extract = lambda _img: OCRResult(
        status="completed",
        text="The Ministry of Health opened 50 hospitals in 2025. Is this wonderful?",
        normalized_text="The Ministry of Health opened 50 hospitals in 2025. Is this wonderful?",
        engine="EasyOCR",
        language="en",
        has_text=True,
        total_confidence=0.92,
        word_count=12,
        blocks=[
            OCRBlock(
                text="The Ministry of Health opened 50 hospitals in 2025. Is this wonderful?",
                confidence=0.92,
                bbox=[[0, 0], [100, 0], [100, 50], [0, 50]],
            )
        ],
        language_configuration=["en", "hi"],
        warnings=[],
    )

    resp_ocr = await client.post(
        "/api/v1/ocr",
        files={"image": ("sample.png", image_bytes, "image/png")},
    )
    assert resp_ocr.status_code == 200
    ocr_data = OCRResponse.model_validate(resp_ocr.json())
    assert ocr_data.has_text is True
    assert "50 hospitals" in ocr_data.text

    # -------------------------------------------------------------
    # Step 3: Standalone Claim Extraction (POST /api/v1/claims)
    # -------------------------------------------------------------
    resp_claims = await client.post(
        "/api/v1/claims",
        json={
            "text": ocr_data.text,
            "language": ocr_data.language,
            "source": "ocr",
            "ocr_confidence": ocr_data.confidence,
        },
    )
    assert resp_claims.status_code == 200
    claims_data = ClaimExtractionResponse.model_validate(resp_claims.json())
    assert claims_data.total_claims >= 1

    eligible_claims = [c for c in claims_data.claims if c.verification_eligibility == "ELIGIBLE"]
    ineligible_claims = [c for c in claims_data.claims if c.verification_eligibility != "ELIGIBLE"]

    assert len(eligible_claims) >= 1
    target_claim = eligible_claims[0]
    assert "50 hospitals" in target_claim.claim
    assert target_claim.source_start is not None
    assert target_claim.source_end is not None

    # -------------------------------------------------------------
    # Step 4: Standalone Fact-Checking (POST /api/v1/fact-check)
    # -------------------------------------------------------------
    all_extracted_texts = [c.claim for c in claims_data.claims]
    mock_fc_review = FactCheckClaim(
        evidence_id="fc_integ_test_001",
        query=target_claim.normalized_text,
        url="https://factchecker.org/review/hospitals-2025",
        title="Fact-check: Health ministry hospital claims in 2025",
        publisher_name="FactCheck Org",
        publisher_site="factchecker.org",
        textual_rating="Mostly False",
        claimed_text=target_claim.original_text,
        claimant="Ministry Spokesperson",
        claim_date="2025-05-01",
        review_date="2025-05-03",
        language_code="en",
    )
    mock_sub_result = FactCheckResult(
        status="success",
        claims=[mock_fc_review],
        queries_performed=1,
        claim_verifications=[{
            "claim_id": target_claim.claim_id,
            "original_claim": target_claim.original_text,
            "verdict": "CONTRADICTED",
            "verification_status": "COMPLETED",
            "publisher_rating_summary": "CONTRADICTED",
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": ["fc_integ_test_001"],
            "conflicting_evidence_ids": [],
            "reasoning_summary": "Relevant reviews rate claim false.",
        }],
    )

    with patch("backend.app.services.factcheck.service.search_fact_checks", return_value=mock_sub_result):
        resp_fc = await client.post(
            "/api/v1/fact-check",
            json={
                "claims": all_extracted_texts,
                "language_code": "en",
            },
        )
    assert resp_fc.status_code == 200
    fc_data = FactCheckResponse.model_validate(resp_fc.json())

    # Verify attribution and gating:
    # 1. Total claims match input count
    assert len(fc_data.claim_verifications) == len(all_extracted_texts)

    # 2. Eligible claim has completed verification and review attribution
    eligible_verif = [v for v in fc_data.claim_verifications if v.eligibility_status == "ELIGIBLE"][0]
    assert eligible_verif.verification_status == "COMPLETED"
    assert eligible_verif.verdict == "CONTRADICTED"
    assert len(fc_data.claims) == 1
    assert fc_data.claims[0].publisher == "FactCheck Org"
    assert fc_data.claims[0].url == "https://factchecker.org/review/hospitals-2025"

    # 3. Ineligible claims (e.g. questions) are skipped
    ineligible_verifs = [v for v in fc_data.claim_verifications if v.eligibility_status == "INELIGIBLE"]
    for inv in ineligible_verifs:
        assert inv.verification_status == "SKIPPED"
        assert inv.verdict == "UNVERIFIABLE"

    # -------------------------------------------------------------
    # Step 5: Verification of Cache Hit on repeated workflow request
    # -------------------------------------------------------------
    resp_cached = await client.post(
        "/api/v1/fact-check",
        json={
            "claims": [target_claim.claim],
            "language_code": "en",
        },
    )
    assert resp_cached.status_code == 200
    cached_fc_data = FactCheckResponse.model_validate(resp_cached.json())
    assert cached_fc_data.status == "success"
    assert cached_fc_data.cached_queries_count == 1
    assert cached_fc_data.queries_performed == 0
    assert cached_fc_data.claim_verifications[0].cache_hit is True


@pytest.mark.asyncio
async def test_integrated_workflow_no_text_in_image(client, mock_optional_analysis_modules):
    """When an image contains no detectable text, OCR gracefully completes with has_text=False."""
    image_bytes = _create_test_image()
    mock_optional_analysis_modules["ocr"].extract = lambda _img: OCRResult(
        status="completed",
        text="",
        normalized_text="",
        engine="EasyOCR",
        language="unknown",
        has_text=False,
        total_confidence=0.0,
        word_count=0,
        blocks=[],
        no_text_reason="NO_TEXT_DETECTED",
        warnings=[],
    )

    resp_ocr = await client.post(
        "/api/v1/ocr",
        files={"image": ("blank.png", image_bytes, "image/png")},
    )
    assert resp_ocr.status_code == 200
    ocr_data = OCRResponse.model_validate(resp_ocr.json())
    assert ocr_data.has_text is False
    assert ocr_data.text == ""

    # Submitting empty text to claims endpoint is rejected with 422
    resp_claims = await client.post("/api/v1/claims", json={"text": ocr_data.text})
    assert resp_claims.status_code == 422


@pytest.mark.asyncio
async def test_integrated_workflow_unconfigured_factcheck(client, monkeypatch):
    """When fact-check provider is unconfigured, fact-check gracefully returns unavailable."""
    from backend.app.config import settings

    monkeypatch.setattr(settings, "FACTCHECK_API_KEY", "")
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "")

    resp_fc = await client.post(
        "/api/v1/fact-check",
        json={"claims": ["The government passed the legislation in 2025."]},
    )
    assert resp_fc.status_code == 200
    fc_data = FactCheckResponse.model_validate(resp_fc.json())
    assert fc_data.status == "unavailable"
    assert any("not configured" in w.lower() for w in fc_data.warnings)
    assert fc_data.claim_verifications[0].verification_status == "UNAVAILABLE"

