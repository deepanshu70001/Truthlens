"""
TruthLens — Stage 4 tests: Evidence & verification layer.

Tests cover:
- Text normalization (with/without text)
- Claim extraction (various inputs)
- Fact-check API (configured, not configured, failure)
- Provenance (configured, not configured, failure)
- Image hashing
- Evidence schema (all builders)
- Fusion engine (authentic, synthetic, conflicting, no evidence)
- Explanation generator
- Report builder
"""

import io
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image, ImageDraw

FIXTURE_DIR = Path(__file__).parent / "fixtures"
TEST_IMAGE_PATH = FIXTURE_DIR / "test_photo.jpg"


# ── Text Normalization ───────────────────────────────────────────

class TestTextNormalization:
    def test_normalize_basic(self):
        from backend.app.services.text.normalizer import normalize_text

        result = normalize_text("  Hello   World!   This is   a test.  ")
        assert result.normalized_text == "Hello World! This is a test."
        assert result.raw_text == "  Hello   World!   This is   a test.  "
        assert result.word_count == 6

    def test_normalize_empty(self):
        from backend.app.services.text.normalizer import normalize_text

        result = normalize_text("")
        assert result.normalized_text == ""
        assert result.word_count == 0

    def test_normalize_none(self):
        from backend.app.services.text.normalizer import normalize_text

        result = normalize_text(None)
        assert result.normalized_text == ""

    def test_extract_urls(self):
        from backend.app.services.text.normalizer import normalize_text

        result = normalize_text("Visit https://example.com for more info.")
        assert result.has_urls
        assert "https://example.com" in result.extracted_urls

    def test_extract_hashtags(self):
        from backend.app.services.text.normalizer import normalize_text

        result = normalize_text("This is #Breaking #News today.")
        assert result.has_hashtags
        assert "#Breaking" in result.extracted_hashtags
        assert "#News" in result.extracted_hashtags

    def test_extract_mentions(self):
        from backend.app.services.text.normalizer import normalize_text

        result = normalize_text("Reported by @IndiaTV today.")
        assert result.has_mentions
        assert "@IndiaTV" in result.extracted_mentions

    def test_sentence_splitting(self):
        from backend.app.services.text.normalizer import normalize_text

        result = normalize_text("First sentence. Second sentence! Third sentence?")
        assert len(result.sentences) == 3

    def test_to_dict_serializable(self):
        from backend.app.services.text.normalizer import normalize_text

        result = normalize_text("Test text with 1000 people affected.")
        d = result.to_dict()
        json.dumps(d)  # Should not raise
        assert "raw_text" in d
        assert "normalized_text" in d


# ── Claim Extraction ─────────────────────────────────────────────

class TestClaimExtraction:
    def test_extract_factual_claim(self):
        from backend.app.services.claims.extractor import extract_claims

        sentences = [
            "According to officials, 500 people were killed in the disaster.",
            "The weather is nice today.",
        ]
        result = extract_claims(sentences)
        assert result.total_claims >= 1
        # The factual claim should score higher
        if result.claims:
            assert result.claims[0].claim_type in ("attribution", "statistical", "factual", "temporal", "unclassified")

    def test_extract_no_sentences(self):
        from backend.app.services.claims.extractor import extract_claims

        result = extract_claims([])
        assert result.total_claims == 0
        assert len(result.warnings) > 0

    def test_extract_short_text(self):
        from backend.app.services.claims.extractor import extract_claims

        result = extract_claims(["Hi.", "Ok."])
        assert result.total_claims == 0

    def test_entities_extraction(self):
        from backend.app.services.claims.extractor import extract_claims

        sentences = [
            "Prime Minister Narendra Modi announced new policy in Delhi yesterday."
        ]
        result = extract_claims(sentences)
        if result.claims:
            claim = result.claims[0]
            # Should find entities
            assert len(claim.entities) > 0 or len(claim.locations) > 0

    def test_dates_extraction(self):
        from backend.app.services.claims.extractor import extract_claims

        sentences = [
            "On January 15, 2026 the government confirmed new regulations."
        ]
        result = extract_claims(sentences)
        if result.claims:
            assert len(result.claims[0].dates) > 0

    def test_claim_to_dict(self):
        from backend.app.services.claims.extractor import extract_claims

        sentences = ["Breaking news: 1000 cases reported today in Mumbai."]
        result = extract_claims(sentences)
        d = result.to_dict()
        json.dumps(d)  # Should not raise
        assert "claims" in d
        assert "total_claims" in d


# ── Fact-Check API ───────────────────────────────────────────────

class TestFactCheck:
    def test_not_configured(self):
        """Should return 'unavailable' when no API key set."""
        from backend.app.services.factcheck.service import search_fact_checks

        with patch("backend.app.services.factcheck.service.settings") as mock_settings:
            mock_settings.GOOGLE_API_KEY = ""
            mock_settings.FACTCHECK_API_KEY = ""
            result = search_fact_checks(["test claim"])

        assert result.status == "unavailable"
        assert len(result.warnings) > 0

    def test_no_claims(self):
        """Should return 'no_results' for empty claims list."""
        from backend.app.services.factcheck.service import search_fact_checks

        with patch("backend.app.services.factcheck.service.settings") as mock_settings:
            mock_settings.GOOGLE_API_KEY = "fake-key"
            mock_settings.FACTCHECK_API_KEY = ""
            result = search_fact_checks([])

        assert result.status == "no_results"

    def test_api_failure(self):
        """Should return 'failed' or have warnings on API error."""
        from backend.app.services.factcheck.service import search_fact_checks
        import httpx

        with patch("backend.app.services.factcheck.service.settings") as mock_settings:
            mock_settings.GOOGLE_API_KEY = "fake-key"
            mock_settings.FACTCHECK_API_KEY = ""

            with patch("backend.app.services.factcheck.service._search_single_claim") as mock_search:
                mock_search.side_effect = httpx.RequestError("Connection failed")
                result = search_fact_checks(["test claim"])

        assert len(result.warnings) > 0 or result.status == "no_results"

    def test_successful_result(self):
        """Should correctly parse a mocked API response."""
        from backend.app.services.factcheck.service import search_fact_checks, FactCheckClaim

        mock_claim = FactCheckClaim(
            query="test claim",
            claimed_text="Test claim text",
            claimant="Test Source",
            publisher_name="PolitiFact",
            publisher_site="politifact.com",
            url="https://politifact.com/test",
            title="Fact Check: Test",
            textual_rating="False",
        )

        with patch("backend.app.services.factcheck.service.settings") as mock_settings:
            mock_settings.GOOGLE_API_KEY = "fake-key"
            mock_settings.FACTCHECK_API_KEY = ""

            with patch("backend.app.services.factcheck.service._search_single_claim") as mock_search:
                mock_search.return_value = [mock_claim]
                result = search_fact_checks(["test claim"])

        assert result.status == "success"
        assert result.total_results == 1
        assert result.claims[0].publisher_name == "PolitiFact"
        assert result.claims[0].textual_rating == "False"

    def test_result_to_dict(self):
        """Result dict should be JSON-serializable and include a note."""
        from backend.app.services.factcheck.service import FactCheckResult

        result = FactCheckResult(status="unavailable")
        d = result.to_dict()
        json.dumps(d)  # Should not raise
        assert "note" in d


# ── Provenance ───────────────────────────────────────────────────

class TestProvenance:
    def test_not_configured(self):
        from backend.app.services.provenance.service import search_provenance

        with patch("backend.app.services.provenance.service.settings") as mock_settings:
            mock_settings.GOOGLE_API_KEY = ""
            mock_settings.max_upload_bytes = 5 * 1024 * 1024
            with patch("backend.app.services.provenance.service.verify_content_credentials") as mock_c2pa:
                from backend.app.services.provenance.c2pa_verifier import C2PAResult
                mock_c2pa.return_value = C2PAResult(status="unavailable", outcome="unavailable")
                result = search_provenance(b"fake image data")

        assert result.status == "unavailable"
        assert len(result.warnings) > 0

    def test_api_failure(self):
        from backend.app.services.provenance.service import search_provenance
        import httpx

        with patch("backend.app.services.provenance.service.settings") as mock_settings:
            mock_settings.GOOGLE_API_KEY = "fake-key"
            mock_settings.max_upload_bytes = 5 * 1024 * 1024
            mock_settings.PROVENANCE_API_TIMEOUT_SECONDS = 1
            mock_settings.PROVENANCE_RETRY_ATTEMPTS = 0

            with patch("backend.app.services.provenance.service.verify_content_credentials") as mock_c2pa, patch("httpx.Client") as mock_client_cls:
                from backend.app.services.provenance.c2pa_verifier import C2PAResult
                mock_c2pa.return_value = C2PAResult(status="unavailable", outcome="unavailable")
                mock_client = MagicMock()
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=False)
                mock_client.post.side_effect = httpx.RequestError("Connection failed")
                mock_client_cls.return_value = mock_client
                result = search_provenance(b"fake image data")

        assert result.status == "failed"

    def test_no_results(self):
        from backend.app.services.provenance.service import search_provenance

        with patch("backend.app.services.provenance.service.settings") as mock_settings:
            mock_settings.GOOGLE_API_KEY = "fake-key"
            mock_settings.max_upload_bytes = 5 * 1024 * 1024
            mock_settings.PROVENANCE_API_TIMEOUT_SECONDS = 1
            mock_settings.PROVENANCE_RETRY_ATTEMPTS = 0

            with patch("backend.app.services.provenance.service.verify_content_credentials") as mock_c2pa, patch("httpx.Client") as mock_client_cls:
                from backend.app.services.provenance.c2pa_verifier import C2PAResult
                mock_c2pa.return_value = C2PAResult(status="unavailable", outcome="unavailable")
                mock_client = MagicMock()
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=False)
                mock_response = MagicMock()
                mock_response.json.return_value = {"responses": [{"webDetection": {}}]}
                mock_response.raise_for_status = MagicMock()
                mock_client.post.return_value = mock_response
                mock_client_cls.return_value = mock_client
                result = search_provenance(b"fake image data")

        assert result.status == "partial"
        assert result.web_detection_status == "no_results"

    def test_result_to_dict(self):
        from backend.app.services.provenance.service import ProvenanceResult

        result = ProvenanceResult(status="unavailable")
        d = result.to_dict()
        json.dumps(d)
        assert "note" in d


# ── Image Hashing ────────────────────────────────────────────────

class TestHashing:
    def test_compute_hashes(self):
        from backend.app.services.hashing.service import compute_hashes

        # Create a simple test image
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        data = buf.getvalue()

        result = compute_hashes(data)
        assert result.sha256
        assert len(result.sha256) == 64
        assert result.phash
        assert result.ahash
        assert result.dhash

    def test_same_image_same_hash(self):
        from backend.app.services.hashing.service import compute_hashes

        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        data = buf.getvalue()

        r1 = compute_hashes(data)
        r2 = compute_hashes(data)

        assert r1.sha256 == r2.sha256
        assert r1.phash == r2.phash

    def test_hamming_distance(self):
        from backend.app.services.hashing.service import compute_hamming_distance

        dist = compute_hamming_distance("0000000000000000", "0000000000000000")
        assert dist == 0

    def test_find_matches_exact(self):
        from backend.app.services.hashing.service import find_matches, compute_hashes

        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        data = buf.getvalue()

        hashes = compute_hashes(data)
        known = [{"sha256": hashes.sha256, "phash": hashes.phash, "analysis_id": "test-123"}]
        result = find_matches(data, known)

        assert result.has_exact_match
        assert len(result.matches) >= 1

    def test_hash_result_to_dict(self):
        from backend.app.services.hashing.service import compute_hashes

        img = Image.new("RGB", (50, 50), color=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = compute_hashes(buf.getvalue())
        d = result.to_dict()
        json.dumps(d)
        assert "sha256" in d
        assert "phash" in d


# ── Evidence Schema ──────────────────────────────────────────────

class TestEvidenceSchema:
    def test_evidence_item_creation(self):
        from backend.app.services.evidence.schema import EvidenceItem

        item = EvidenceItem(
            type="model_evidence",
            source="test_model",
            strength="high",
            direction="synthetic",
            description="Test evidence",
            confidence=0.95,
            calibrated=False,
        )
        d = item.to_dict()
        assert d["type"] == "model_evidence"
        assert "confidence_note" in d  # Uncalibrated

    def test_evidence_collection(self):
        from backend.app.services.evidence.schema import EvidenceCollection, EvidenceItem

        col = EvidenceCollection()
        col.add(EvidenceItem(
            type="model_evidence", source="m1", strength="high",
            direction="authentic", description="Model says authentic",
        ))
        col.add(EvidenceItem(
            type="forensic_evidence", source="f1", strength="low",
            direction="neutral", description="No anomalies",
        ))

        assert col.count == 2
        assert len(col.get_by_type("model_evidence")) == 1
        assert len(col.get_strong_evidence()) == 1
        assert not col.has_conflicting_evidence

    def test_informational_module_directions_do_not_create_authenticity_conflict(self):
        from backend.app.services.evidence.schema import EvidenceCollection, EvidenceItem

        col = EvidenceCollection()
        col.add(EvidenceItem(
            type="model_evidence", source="m1", strength="high",
            direction="authentic", description="Model says authentic",
        ))
        col.add(EvidenceItem(
            type="forensic_evidence", source="f1", strength="high",
            direction="synthetic", description="Forensics says synthetic",
        ))

        assert not col.has_conflicting_evidence

    def test_build_evidence_from_authenticity(self):
        from backend.app.services.evidence.schema import build_evidence_from_authenticity

        class MockAuth:
            label = "likely_authentic"
            confidence = 0.85
            ai_generated_score = 0.15
            authentic_score = 0.85
            model_name = "test-model"
            model_prediction = "likely_authentic"
            calibrated_ai_generated_probability = 0.15
            calibration_status = "validated_temperature_scaling"
            domain_validation_status = "validated_for_input_domain"

        items = build_evidence_from_authenticity(MockAuth())
        assert len(items) == 1
        assert items[0].direction == "authentic"
        assert items[0].strength == "medium"

    def test_build_evidence_from_none(self):
        from backend.app.services.evidence.schema import (
            build_evidence_from_authenticity,
            build_evidence_from_forensics,
            build_evidence_from_ocr,
            build_evidence_from_factcheck,
            build_evidence_from_provenance,
        )

        assert build_evidence_from_authenticity(None) == []
        assert build_evidence_from_forensics(None) == []
        assert build_evidence_from_ocr(None) == []
        assert build_evidence_from_factcheck(None) == []
        assert build_evidence_from_provenance(None) == []


# ── Fusion Engine ────────────────────────────────────────────────

def _eligible_fusion_item(direction, description):
    from backend.app.services.evidence.schema import EvidenceItem

    labels = {"authentic": "likely_authentic", "synthetic": "likely_ai_generated", "manipulated": "likely_manipulated"}
    probability = {"authentic": 0.1, "synthetic": 0.9, "manipulated": 0.7}[direction]
    return EvidenceItem(
        type="model_evidence", source="authenticity_detector", stage_name="authenticity",
        strength="high", direction=direction, description=description,
        data={"label": labels[direction], "calibrated_candidate_label": labels[direction], "model": "test-model",
              "calibration_status": "validated_temperature_scaling",
              "domain_validation_status": "validated_for_input_domain",
              "calibrated_ai_generated_probability": probability},
        calibrated=True, confidence=probability,
    )


class TestFusion:
    def test_no_evidence(self):
        from backend.app.services.fusion.engine import fuse_evidence
        from backend.app.services.evidence.schema import EvidenceCollection

        result = fuse_evidence(EvidenceCollection())
        assert result.label == "inconclusive"
        assert result.confidence is None
        assert result.overall_score is None
        assert not result.calibrated

    def test_strong_authentic(self):
        from backend.app.services.fusion.engine import fuse_evidence
        from backend.app.services.evidence.schema import EvidenceCollection, EvidenceItem

        col = EvidenceCollection()
        col.add(_eligible_fusion_item("authentic", "Model says authentic"))
        col.add(EvidenceItem(
            type="metadata_evidence", source="exif", strength="low",
            direction="authentic", description="EXIF present",
        ))

        result = fuse_evidence(col)
        assert result.label == "likely_authentic"
        assert result.confidence > 0

    def test_strong_synthetic(self):
        from backend.app.services.fusion.engine import fuse_evidence
        from backend.app.services.evidence.schema import EvidenceCollection, EvidenceItem

        col = EvidenceCollection()
        col.add(_eligible_fusion_item("synthetic", "Model says AI-generated"))

        result = fuse_evidence(col)
        assert result.label == "likely_ai_generated"

    def test_conflicting_evidence_produces_inconclusive(self):
        from backend.app.services.fusion.engine import fuse_evidence
        from backend.app.services.evidence.schema import EvidenceCollection, EvidenceItem

        col = EvidenceCollection()
        col.add(_eligible_fusion_item("authentic", "Model says authentic"))
        col.add(_eligible_fusion_item("synthetic", "Conflicting model says synthetic"))

        result = fuse_evidence(col)
        assert result.label == "inconclusive"
        assert result.conflicting

    def test_uncalibrated_confidence(self):
        from backend.app.services.fusion.engine import fuse_evidence
        from backend.app.services.evidence.schema import EvidenceCollection, EvidenceItem

        col = EvidenceCollection()
        col.add(EvidenceItem(
            type="model_evidence", source="m1", strength="high",
            direction="authentic", description="Test",
        ))

        result = fuse_evidence(col)
        assert not result.calibrated
        d = result.to_dict()
        assert "confidence_note" in d
        assert "uncalibrated" in d["confidence_note"].lower()

    def test_fusion_always_has_caveat(self):
        from backend.app.services.fusion.engine import fuse_evidence
        from backend.app.services.evidence.schema import EvidenceCollection, EvidenceItem

        col = EvidenceCollection()
        col.add(EvidenceItem(
            type="model_evidence", source="m1", strength="low",
            direction="authentic", description="Test",
        ))

        result = fuse_evidence(col)
        d = result.to_dict()
        assert "caveat" in d


# ── Explanation Generator ────────────────────────────────────────

class TestExplanation:
    def test_generate_explanation(self):
        from backend.app.services.explanation.generator import generate_explanation
        from backend.app.services.evidence.schema import EvidenceCollection, EvidenceItem

        col = EvidenceCollection()
        col.add(EvidenceItem(
            type="model_evidence", source="m1", strength="high",
            direction="authentic", description="Model says authentic with high confidence.",
        ))
        col.add(EvidenceItem(
            type="ocr_evidence", source="easyocr", strength="medium",
            direction="informational", description="OCR extracted 20 words.",
        ))

        result = generate_explanation(col, "likely_authentic")
        assert result.summary
        assert len(result.sections) > 0
        assert len(result.limitations) > 0

    def test_explanation_empty_evidence(self):
        from backend.app.services.explanation.generator import generate_explanation
        from backend.app.services.evidence.schema import EvidenceCollection

        result = generate_explanation(EvidenceCollection(), "inconclusive")
        assert result.summary
        assert "inconclusive" in result.summary.lower() or "insufficient" in result.summary.lower()

    def test_explanation_to_dict(self):
        from backend.app.services.explanation.generator import generate_explanation
        from backend.app.services.evidence.schema import EvidenceCollection, EvidenceItem

        col = EvidenceCollection()
        col.add(EvidenceItem(
            type="model_evidence", source="m1", strength="high",
            direction="synthetic", description="Model predicts AI-generated.",
        ))

        result = generate_explanation(col, "likely_ai_generated")
        d = result.to_dict()
        json.dumps(d)
        assert "summary" in d
        assert "sections" in d
        assert "limitations" in d


# ── Report Builder ───────────────────────────────────────────────

class TestReportBuilder:
    def test_build_minimal_report(self):
        from backend.app.services.reporting.builder import build_report

        report = build_report(
            analysis_id="test-123",
            status="completed",
        )
        assert report["analysis_id"] == "test-123"
        assert report["status"] == "completed"
        assert report["disclaimer"]
        assert report["limitations"]

    def test_build_full_report(self):
        from backend.app.services.reporting.builder import build_report

        report = build_report(
            analysis_id="test-456",
            status="completed",
            image_info={"width": 640, "height": 480},
            authenticity={"label": "likely_authentic"},
            forensics={"signals": []},
            ocr={"text": "Hello"},
            claims={"claims": []},
            fact_checks={"status": "unavailable"},
            provenance={"status": "unavailable"},
            assessment={"label": "likely_authentic", "limitations": ["Test limitation"]},
            explanation={"summary": "Test", "limitations": ["Another limitation"]},
        )

        assert report["analysis_id"] == "test-456"
        assert report["image"] is not None
        assert report["authenticity"] is not None
        assert report["disclaimer"]
        # Limitations should include both assessment and standard ones
        assert len(report["limitations"]) >= 2

    def test_report_serializable(self):
        from backend.app.services.reporting.builder import build_report

        report = build_report(analysis_id="test", status="completed")
        json_str = json.dumps(report)
        assert json_str

    def test_report_always_has_disclaimer(self):
        from backend.app.services.reporting.builder import build_report

        report = build_report(analysis_id="test", status="completed")
        assert "DISCLAIMER" in report["disclaimer"]
        assert "proof" in report["disclaimer"].lower()
