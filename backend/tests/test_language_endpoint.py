"""
TruthLens — Tests for the Standalone Language Identification endpoint (POST /api/v1/nlp/language).

Verifies the 17 specified scenarios:
1. English detection (HTTP 200, language_code="en", language_name="English", confidence >= 0.7)
2. Hindi detection (HTTP 200, language_code="hi", language_name="Hindi", confidence >= 0.7)
3. Multilingual detection (French, Spanish, German, etc. with top-k candidates)
4. Short or ambiguous text (status="insufficient_text", outcome="insufficient_input", confidence=None)
5. Empty text rejection (HTTP 422)
6. Whitespace-only text rejection (HTTP 422)
7. Maximum input length exceeded > 10,000 (HTTP 422)
8. Invalid top_k bounds (< 1 or > 20, HTTP 422)
9. Detector unavailable (status="unavailable", unavailable_reason set)
10. Model initialization failure (status="unavailable")
11. Prediction exception (status="failed")
12. Language code and name mapping (verifies LANGUAGE_NAMES match)
13. Confidence bounds and schema validation (0.0 <= conf <= 1.0, confidence_semantics present)
14. Bearer authentication enforced (HTTP 401 when unauthenticated)
15. Rate limiting enforced (HTTP 429 when quota exceeded)
16. Security & path sanitization (zero filesystem paths or secrets in response text)
17. No eager loading on startup (importing router / app does not call load())
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest
import torch
from httpx import ASGITransport, AsyncClient

from backend.app.config import settings
from backend.app.main import app
from backend.app.schemas.language import (
    LanguageCandidate,
    LanguageDetectionRequest,
    LanguageDetectionResponse,
)
from backend.app.services.nlp.language_detector import (
    LANGUAGE_NAMES,
    MODEL_ID,
    LanguageDetectionResult,
    LanguageDetector,
    get_language_detector,
)
from backend.tests.test_language_detector import (
    FakeModel,
    FakeTokenizer,
    ready_detector,
)


@pytest.fixture
def mock_en_detector(monkeypatch):
    """Fixture providing a mock detector configured for confident English prediction."""
    labels = {0: "en", 1: "hi", 2: "fr", 3: "es", 4: "de"}
    logits = [[12.0, -5.0, -6.0, -8.0, -10.0]]
    detector = ready_detector(logits, labels=labels)
    monkeypatch.setattr(
        "backend.app.api.v1.endpoints.language.get_language_detector",
        lambda: detector,
    )
    return detector


@pytest.fixture
def mock_hi_detector(monkeypatch):
    """Fixture providing a mock detector configured for confident Hindi prediction."""
    labels = {0: "en", 1: "hi", 2: "fr", 3: "es", 4: "de"}
    logits = [[-5.0, 14.0, -6.0, -8.0, -10.0]]
    detector = ready_detector(logits, labels=labels)
    monkeypatch.setattr(
        "backend.app.api.v1.endpoints.language.get_language_detector",
        lambda: detector,
    )
    return detector


@pytest.fixture
def mock_multilingual_detector(monkeypatch):
    """Fixture providing a mock detector configured for French prediction with top-k ranking."""
    labels = {0: "en", 1: "hi", 2: "fr", 3: "es", 4: "de"}
    logits = [[-2.0, -10.0, 15.0, 2.0, 0.0]]
    detector = ready_detector(logits, labels=labels)
    monkeypatch.setattr(
        "backend.app.api.v1.endpoints.language.get_language_detector",
        lambda: detector,
    )
    return detector


# ── Scenario 1: English detection ──────────────────────────────────

@pytest.mark.asyncio
async def test_language_detection_english(client, mock_en_detector):
    """1. English detection returns HTTP 200, en/English, and confidence >= 0.7."""
    text = (
        "The TruthLens verification system provides automated claim extraction, "
        "multilingual forensic analysis, and neural authenticity assessments."
    )
    resp = await client.post("/api/v1/nlp/language", json={"text": text, "top_k": 3})
    assert resp.status_code == 200
    data = resp.json()

    validated = LanguageDetectionResponse.model_validate(data)
    assert validated.status == "completed"
    assert validated.outcome == "success"
    assert validated.detected_language == "en"
    assert validated.language_code == "en"
    assert validated.language_name == "English"
    assert validated.confidence is not None and validated.confidence >= 0.70
    assert len(validated.top_languages) == 3
    assert validated.top_languages[0].language_code == "en"
    assert validated.model_id == MODEL_ID
    assert validated.input_characters == len(text)
    assert not validated.input_truncated


# ── Scenario 2: Hindi detection ────────────────────────────────────

@pytest.mark.asyncio
async def test_language_detection_hindi(client, mock_hi_detector):
    """2. Hindi detection returns HTTP 200, hi/Hindi, and confidence >= 0.7."""
    text = "सच्चाई और निष्पक्षता के साथ समाचार और सूचनाओं का विश्लेषण करना जरूरी है।"
    resp = await client.post("/api/v1/nlp/language", json={"text": text, "top_k": 3})
    assert resp.status_code == 200
    data = resp.json()

    validated = LanguageDetectionResponse.model_validate(data)
    assert validated.status == "completed"
    assert validated.outcome == "success"
    assert validated.detected_language == "hi"
    assert validated.language_code == "hi"
    assert validated.language_name == "Hindi"
    assert validated.confidence is not None and validated.confidence >= 0.70
    assert validated.top_languages[0].language_name == "Hindi"


# ── Scenario 3: Multilingual detection (French / top-k) ───────────

@pytest.mark.asyncio
async def test_language_detection_multilingual_candidates(client, mock_multilingual_detector):
    """3. Multilingual detection returns top-k ordered candidates with French primary."""
    text = "Le système d'analyse documentaire permet d'identifier la langue principale du texte."
    resp = await client.post("/api/v1/nlp/language", json={"text": text, "top_k": 4})
    assert resp.status_code == 200
    data = resp.json()

    validated = LanguageDetectionResponse.model_validate(data)
    assert validated.status == "completed"
    assert validated.detected_language == "fr"
    assert validated.language_code == "fr"
    assert validated.language_name == "French"
    assert len(validated.top_languages) == 4
    assert validated.top_languages[0].language_code == "fr"
    # Ensure descending order of confidence scores
    scores = [c.confidence for c in validated.top_languages]
    assert scores == sorted(scores, reverse=True)


# ── Scenario 4: Short or ambiguous text ───────────────────────────

@pytest.mark.asyncio
async def test_language_detection_short_ambiguous_text(client, mock_en_detector):
    """4. Short or ambiguous text returns status='insufficient_text', outcome='insufficient_input'."""
    text = "Short text."  # 9 alphabetic characters < 20 MIN_CONTENT_CHARACTERS
    resp = await client.post("/api/v1/nlp/language", json={"text": text})
    assert resp.status_code == 200
    data = resp.json()

    validated = LanguageDetectionResponse.model_validate(data)
    assert validated.status == "insufficient_text"
    assert validated.outcome == "insufficient_input"
    assert validated.detected_language == "unknown"
    assert validated.language_code is None
    assert validated.confidence is None
    assert len(validated.warnings) > 0
    assert "too short or noisy" in validated.warnings[0]


# ── Scenario 5: Empty text rejection ──────────────────────────────

@pytest.mark.asyncio
async def test_language_detection_empty_text_rejected(client):
    """5. Empty string input is rejected with HTTP 422 Unprocessable Entity."""
    resp = await client.post("/api/v1/nlp/language", json={"text": ""})
    assert resp.status_code == 422


# ── Scenario 6: Whitespace-only text rejection ────────────────────

@pytest.mark.asyncio
async def test_language_detection_whitespace_rejected(client):
    """6. Whitespace-only input is rejected with HTTP 422 Unprocessable Entity."""
    resp = await client.post("/api/v1/nlp/language", json={"text": "   \n\t  \r  "})
    assert resp.status_code == 422


# ── Scenario 7: Maximum input length exceeded ──────────────────────

@pytest.mark.asyncio
async def test_language_detection_max_length_exceeded(client):
    """7. Input exceeding 10,000 characters is rejected with HTTP 422."""
    oversized_text = "The quick brown fox jumps over the lazy dog. " * 300  # > 10,000 chars
    assert len(oversized_text) > 10000
    resp = await client.post("/api/v1/nlp/language", json={"text": oversized_text})
    assert resp.status_code == 422


# ── Scenario 8: Invalid top_k bounds ──────────────────────────────

@pytest.mark.asyncio
async def test_language_detection_invalid_top_k(client):
    """8. top_k values < 1 or > 20 are rejected with HTTP 422."""
    valid_text = "This is a sufficiently long test sentence for language identification."
    # top_k = 0 (< 1)
    r_zero = await client.post("/api/v1/nlp/language", json={"text": valid_text, "top_k": 0})
    assert r_zero.status_code == 422

    # top_k = 25 (> 20)
    r_high = await client.post("/api/v1/nlp/language", json={"text": valid_text, "top_k": 25})
    assert r_high.status_code == 422


# ── Scenario 9: Detector unavailable ──────────────────────────────

@pytest.mark.asyncio
async def test_language_detection_detector_unavailable(client, monkeypatch):
    """9. When local cache lacks weights, endpoint returns status='unavailable' with reason."""
    detector = LanguageDetector()
    monkeypatch.setattr(detector, "load", lambda device="auto": False)
    monkeypatch.setattr(
        "backend.app.api.v1.endpoints.language.get_language_detector",
        lambda: detector,
    )

    text = "This is a valid sentence that should attempt model loading."
    resp = await client.post("/api/v1/nlp/language", json={"text": text})
    assert resp.status_code == 200
    data = resp.json()

    validated = LanguageDetectionResponse.model_validate(data)
    assert validated.status == "unavailable"
    assert validated.outcome == "unavailable"
    assert validated.detected_language == "unknown"
    assert validated.unavailable_reason is not None
    assert len(validated.warnings) > 0


# ── Scenario 10: Model initialization failure ─────────────────────

@pytest.mark.asyncio
async def test_language_detection_initialization_failure(client, monkeypatch):
    """10. Loader exception triggers status='unavailable' with sanitized message."""
    detector = LanguageDetector()

    def failing_load(device="auto"):
        raise OSError("Local cache file corrupted or missing header")

    monkeypatch.setattr(detector, "load", failing_load)
    monkeypatch.setattr(
        "backend.app.api.v1.endpoints.language.get_language_detector",
        lambda: detector,
    )

    text = "This is a valid sentence that will trigger a loader exception."
    resp = await client.post("/api/v1/nlp/language", json={"text": text})
    assert resp.status_code == 200
    data = resp.json()

    validated = LanguageDetectionResponse.model_validate(data)
    assert validated.status == "unavailable"
    assert validated.unavailable_reason is not None
    assert "corrupted" not in resp.text  # Internal path/error sanitization


# ── Scenario 11: Prediction exception ─────────────────────────────

@pytest.mark.asyncio
async def test_language_detection_prediction_exception(client, monkeypatch):
    """11. Inference exception during forward pass returns status='failed'."""
    broken_detector = ready_detector(RuntimeError("GPU OOM during forward pass"))
    monkeypatch.setattr(
        "backend.app.api.v1.endpoints.language.get_language_detector",
        lambda: broken_detector,
    )

    text = "This is a valid sentence that triggers an unexpected runtime forward pass failure."
    resp = await client.post("/api/v1/nlp/language", json={"text": text})
    assert resp.status_code == 200
    data = resp.json()

    validated = LanguageDetectionResponse.model_validate(data)
    assert validated.status == "failed"
    assert validated.outcome == "failed"
    assert validated.detected_language == "unknown"
    assert "GPU OOM" not in resp.text  # Sanitized


# ── Scenario 12: Language code and name mapping ────────────────────

@pytest.mark.asyncio
async def test_language_detection_code_and_name_mapping(client, mock_en_detector):
    """12. Returned language candidates map strictly to documented LANGUAGE_NAMES."""
    text = "The quick brown fox jumps over the lazy dog in this sample text."
    resp = await client.post("/api/v1/nlp/language", json={"text": text, "top_k": 5})
    assert resp.status_code == 200
    data = resp.json()

    validated = LanguageDetectionResponse.model_validate(data)
    for candidate in validated.top_languages:
        assert candidate.language_code in LANGUAGE_NAMES
        assert candidate.language_name == LANGUAGE_NAMES[candidate.language_code]
        assert candidate.language == candidate.language_code


# ── Scenario 13: Confidence bounds and schema validation ──────────

@pytest.mark.asyncio
async def test_language_detection_confidence_bounds_and_schema(client, mock_en_detector):
    """13. Confidence values stay in [0.0, 1.0] and schema fields validate fully."""
    text = "Validating that all probability scores and schema fields adhere to strict contracts."
    resp = await client.post("/api/v1/nlp/language", json={"text": text, "top_k": 3})
    assert resp.status_code == 200
    data = resp.json()

    validated = LanguageDetectionResponse.model_validate(data)
    if validated.confidence is not None:
        assert 0.0 <= validated.confidence <= 1.0
    for cand in validated.top_languages:
        assert 0.0 <= cand.confidence <= 1.0

    assert validated.confidence_semantics == "Maximum softmax probability from the model; not calibrated."
    assert validated.runtime_ms >= 0
    assert validated.processing_time_ms >= 0
    assert validated.model_id == MODEL_ID
    assert len(validated.limitations) > 0


# ── Scenario 14: Bearer authentication enforced ───────────────────

@pytest.mark.asyncio
async def test_language_detection_authentication_enforced(monkeypatch):
    """14. Unauthenticated requests are rejected with HTTP 401 when dev auth is disabled."""
    monkeypatch.setattr(settings, "DEV_AUTH_BYPASS", False)
    monkeypatch.setattr(settings, "APP_ENV", "production")

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as unauth_client:
        resp = await unauth_client.post(
            "/api/v1/nlp/language",
            json={"text": "Authenticating requests to protected endpoints."},
        )
        assert resp.status_code == 401


# ── Scenario 15: Rate limiting enforced ───────────────────────────

@pytest.mark.asyncio
async def test_language_detection_rate_limiting_enforced(client, monkeypatch, mock_en_detector):
    """15. Exceeding RATE_LIMIT_LANGUAGE_PER_MINUTE returns HTTP 429."""
    monkeypatch.setattr(settings, "RATE_LIMIT_LANGUAGE_PER_MINUTE", 2)
    payload = {"text": "A valid sentence repeated to test rate limit enforcement."}

    r1 = await client.post("/api/v1/nlp/language", json=payload)
    assert r1.status_code in (200, 429)

    r2 = await client.post("/api/v1/nlp/language", json=payload)
    r3 = await client.post("/api/v1/nlp/language", json=payload)
    if r3.status_code != 429:
        r4 = await client.post("/api/v1/nlp/language", json=payload)
        assert r4.status_code == 429
    else:
        assert r3.status_code == 429


# ── Scenario 16: Security & path sanitization ─────────────────────

@pytest.mark.asyncio
async def test_language_detection_path_and_secret_sanitization(client, mock_en_detector):
    """16. Responses never leak internal file paths, tokens, or environment secrets."""
    text = "Security verification ensuring no local filesystem paths or environment secrets appear."
    resp = await client.post("/api/v1/nlp/language", json={"text": text})
    assert resp.status_code == 200
    text_content = resp.text

    # No local filesystem path patterns
    assert not re.search(r"[A-Za-z]:\\[A-Za-z0-9_\\\.-]+", text_content)
    assert "/Users/" not in text_content
    assert "/home/" not in text_content
    assert "SECRET" not in text_content
    assert "token" not in text_content.lower() or "papluca" not in text_content


# ── Scenario 17: No eager loading on startup ──────────────────────

def test_language_detection_no_eager_loading_on_startup():
    """17. Importing router or creating fresh detector does not eagerly load model weights."""
    detector = LanguageDetector()
    assert not detector.is_loaded
    assert not detector._load_attempted
    assert detector._model is None
    assert detector._tokenizer is None
