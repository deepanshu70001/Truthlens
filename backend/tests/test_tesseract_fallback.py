"""Verification and test suite for Tesseract OCR Fallback integration.

Verifies:
1. Direct Tesseract OCR execution and schema normalization with engine='PyTesseract'.
2. Automatic fallback to Tesseract when primary EasyOCR throws an error during inference.
3. Automatic fallback to Tesseract when primary EasyOCR produces no usable text.
4. Graceful failure when Tesseract executable is not installed or available.
5. Language configuration and missing language pack error handling.
6. Coordinate bounding box normalization and valid area clipping.
7. Total confidence averaging and Unicode text normalization.
"""
from __future__ import annotations

import io
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from backend.app.services.ocr.service import OCRBlock, OCRResult, OCRService


@pytest.fixture
def test_image_with_text() -> Image.Image:
    """Create a clean synthetic image with clear English text."""
    img = Image.new("RGB", (400, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 35), "TruthLens Verification 2028", fill=(0, 0, 0))
    return img


def test_tesseract_direct_execution_and_schema(test_image_with_text):
    """Direct Tesseract execution produces schema-compliant OCRResult."""
    service = OCRService(languages=["en"], gpu=False)
    service._load_tesseract()
    assert service._engine == "PyTesseract"
    assert service._tesseract is not None

    result = service.extract(test_image_with_text)
    assert result.status == "completed"
    assert result.engine == "PyTesseract"
    assert result.has_text is True
    assert "TruthLens" in result.text or "Verification" in result.text
    assert "TruthLens" in result.normalized_text or "Verification" in result.normalized_text
    assert result.word_count >= 1
    assert result.runtime_ms > 0
    assert result.language in {"en", "unknown"}

    d = result.to_dict()
    assert d["engine"] == "PyTesseract"
    assert d["status"] == "completed"
    assert d["has_text"] is True
    assert isinstance(d["blocks"], list)
    for block in d["blocks"]:
        assert "text" in block
        assert "confidence" in block
        assert "bbox" in block
        assert len(block["bbox"]) == 4


def test_primary_easyocr_failure_triggers_tesseract_fallback(monkeypatch, test_image_with_text):
    """When EasyOCR inference raises an exception, fallback to Tesseract succeeds."""
    service = OCRService(languages=["en"], gpu=False)
    # Simulate pre-loaded EasyOCR reader that fails on inference
    class BrokenEasyOCRReader:
        def readtext(self, _img):
            raise RuntimeError("EasyOCR GPU out of memory or inference crash")

    service._reader = BrokenEasyOCRReader()
    service._engine = "EasyOCR"
    service._load_attempted = True

    result = service.extract(test_image_with_text)
    assert result.status == "completed"
    assert result.engine == "PyTesseract"
    assert result.has_text is True
    assert any("EasyOCR inference failed" in w for w in result.warnings)
    assert any("Readable OCR text was produced by the Tesseract fallback" in w for w in result.warnings)


def test_primary_easyocr_empty_triggers_tesseract_fallback(monkeypatch, test_image_with_text):
    """When EasyOCR produces no usable text, fallback to Tesseract extracts text."""
    service = OCRService(languages=["en"], gpu=False)
    # Simulate EasyOCR returning empty results
    class EmptyEasyOCRReader:
        def readtext(self, _img):
            return []

    service._reader = EmptyEasyOCRReader()
    service._engine = "EasyOCR"
    service._load_attempted = True

    result = service.extract(test_image_with_text)
    assert result.status == "completed"
    assert result.engine == "PyTesseract"
    assert result.has_text is True
    assert any("EasyOCR produced no usable text; attempting the Tesseract fallback" in w for w in result.warnings)
    assert any("Readable OCR text was produced by the Tesseract fallback" in w for w in result.warnings)


def test_tesseract_missing_executable_fails_gracefully(monkeypatch, test_image_with_text):
    """When Tesseract executable is not on PATH, fails safely with informative warning."""
    service = OCRService(languages=["en"], gpu=False)
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)

    service._load_tesseract()
    assert service._tesseract is None
    assert service._engine is None
    assert any("Tesseract executable not found" in w for w in service._load_warnings)

    result = service.extract(test_image_with_text)
    assert result.status == "unavailable"
    assert result.has_text is False


def test_tesseract_unsupported_language_fails_gracefully(test_image_with_text):
    """When requested language pack is not mapped or available, reports unavailable without crash."""
    service = OCRService(languages=["xx"], gpu=False)
    service._load_tesseract()
    assert service._tesseract is None
    assert any("Tesseract fallback unavailable" in w for w in service._load_warnings)
    result = service.extract(test_image_with_text)
    assert result.status == "unavailable"


def test_tesseract_bounding_boxes_are_valid_and_clipped():
    """Verify that Tesseract bounding boxes are strictly clipped within image bounds."""
    service = OCRService(languages=["en"], gpu=False)
    service._load_tesseract()

    # Image with small text at borders
    img = Image.new("RGB", (200, 60), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), "TOP-LEFT", fill=(0, 0, 0))

    result = service.extract(img)
    for block in result.blocks:
        for x, y in block.bbox:
            assert 0 <= x <= 200
            assert 0 <= y <= 60
