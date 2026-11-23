"""Tests for Google Cloud Vision Web Detection integration."""
from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest
from PIL import Image

from backend.app.config import settings
from backend.app.services.provenance.service import (
    _is_configured,
    _parse_web_detection,
    _safe_external_url,
    search_provenance,
)


def _make_test_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color="red").save(buf, format="JPEG")
    return buf.getvalue()


def test_vision_web_detection_configured_detection(monkeypatch):
    monkeypatch.setattr(settings, "VISION_API_KEY", "")
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "")
    assert _is_configured() is False

    monkeypatch.setattr(settings, "VISION_API_KEY", "vision-key-test")
    assert _is_configured() is True

    monkeypatch.setattr(settings, "VISION_API_KEY", "")
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "google-key-test")
    assert _is_configured() is True


def test_vision_web_detection_url_sanitization():
    # Strips sensitive query keys
    dirty_url = "https://example.com/photo.jpg?key=secret123&token=abc&view=true"
    sanitized = _safe_external_url(dirty_url)
    assert "key=" not in sanitized
    assert "token=" not in sanitized
    assert "view=true" in sanitized

    # Rejects non-https
    assert _safe_external_url("http://example.com/photo.jpg") == ""
    assert _safe_external_url("ftp://example.com/photo.jpg") == ""
    assert _safe_external_url("") == ""


def test_vision_web_detection_parse_success():
    mock_provider_data = {
        "responses": [{
            "webDetection": {
                "webEntities": [
                    {"entityId": "/m/0199g", "description": "Bicycle", "score": 0.85},
                    {"entityId": "/m/0k4j", "description": "Car", "score": 0.65},
                ],
                "fullMatchingImages": [{"url": "https://example.com/exact.jpg"}],
                "partialMatchingImages": [{"url": "https://example.com/partial.jpg"}],
                "pagesWithMatchingImages": [{
                    "url": "https://news.example.com/story.html",
                    "pageTitle": "Breaking News Story",
                    "fullMatchingImages": [{"url": "https://example.com/exact.jpg"}],
                }],
                "visuallySimilarImages": [{"url": "https://example.com/similar.jpg"}],
                "bestGuessLabels": [{"label": "City Street Bicycle"}],
            }
        }]
    }
    status, payload, error, warnings = _parse_web_detection(mock_provider_data)
    assert status == "success"
    assert len(payload["web_entities"]) == 2
    assert payload["web_entities"][0].description == "Bicycle"
    assert len(payload["matching_pages"]) == 1
    assert payload["matching_pages"][0].page_title == "Breaking News Story"
    assert payload["matching_pages"][0].match_type == "full_match"
    assert len(payload["matching_images"]) == 2  # exact + partial
    assert len(payload["similar_images"]) == 1
    assert payload["best_guess_labels"] == ["City Street Bicycle"]


def test_vision_web_detection_parse_no_results():
    empty_data = {"responses": [{"webDetection": {}}]}
    status, payload, error, warnings = _parse_web_detection(empty_data)
    assert status == "no_results"
    assert payload.get("matching_pages") == []


def test_search_provenance_missing_credentials(monkeypatch):
    monkeypatch.setattr(settings, "VISION_API_KEY", "")
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "")
    jpeg = _make_test_jpeg()
    res = search_provenance(jpeg)
    assert res.web_detection_status == "unavailable"
    assert any("not configured" in w.lower() for w in res.warnings)


def test_search_provenance_mock_success(monkeypatch):
    monkeypatch.setattr(settings, "VISION_API_KEY", "test-vision-key")

    mock_resp = {
        "responses": [{
            "webDetection": {
                "webEntities": [{"description": "Historic Monument", "score": 0.92}],
                "fullMatchingImages": [{"url": "https://example.org/monument.jpg"}],
                "pagesWithMatchingImages": [{
                    "url": "https://archive.example.org/monument",
                    "pageTitle": "Monument Archive",
                }],
            }
        }]
    }

    class _MockResponse:
        status_code = 200
        content = json.dumps(mock_resp).encode("utf-8")
        def raise_for_status(self): pass
        def json(self): return mock_resp

    class _MockClient:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, params=None, json=None, **kwargs):
            return _MockResponse()

    with patch("backend.app.services.provenance.service.httpx.Client", _MockClient):
        res = search_provenance(_make_test_jpeg())
        assert res.web_detection_status == "success"
        assert len(res.web_entities) >= 1
        assert res.web_entities[0].description == "Historic Monument"
        assert len(res.matching_pages) == 1
        assert res.total_matches >= 1


def test_search_provenance_http_403_permission_denied(monkeypatch):
    monkeypatch.setattr(settings, "VISION_API_KEY", "test-vision-key")

    import httpx

    class _MockClient:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, params=None, json=None, **kwargs):
            req = httpx.Request("POST", url)
            resp = httpx.Response(403, request=req)
            resp.raise_for_status()

    with patch("backend.app.services.provenance.service.httpx.Client", _MockClient):
        res = search_provenance(_make_test_jpeg())
        assert res.web_detection_status == "failed"
        assert any("403" in w for w in res.warnings)
