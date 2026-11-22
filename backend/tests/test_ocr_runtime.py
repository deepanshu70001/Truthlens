"""Offline readiness, failure-path and coordinate tests for the OCR adapter."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from backend.app.services.ocr.service import OCRBlock, OCRService


def test_easyocr_initialization_is_cache_only(monkeypatch, tmp_path):
    captured = {}

    class Reader:
        def __init__(self, languages, **kwargs):
            captured["languages"] = languages
            captured.update(kwargs)

    monkeypatch.setattr("backend.app.services.ocr.service.settings.OCR_MODEL_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(Reader=Reader))
    service = OCRService(languages=["en", "hi"], gpu=False)
    service.load()

    assert service.is_loaded
    assert service._engine == "EasyOCR"
    assert captured["languages"] == ["en", "hi"]
    assert captured["gpu"] is False
    assert captured["download_enabled"] is False


@pytest.mark.parametrize("failure", [ImportError("missing package"), FileNotFoundError("missing weights")])
def test_easyocr_package_or_weights_missing_is_unavailable_without_network(monkeypatch, tmp_path, failure):
    class Reader:
        def __init__(self, *_args, **kwargs):
            assert kwargs["download_enabled"] is False
            raise failure

    monkeypatch.setattr("backend.app.services.ocr.service.settings.OCR_MODEL_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(Reader=Reader))
    service = OCRService(languages=["en"], gpu=False)
    monkeypatch.setattr(service, "_load_tesseract", lambda: None)

    service.load()
    result = service.extract(Image.new("RGB", (64, 32), "white"))
    assert result.status == "unavailable"
    assert not service.is_loaded
    assert service._load_warnings


def test_easyocr_reader_initialization_exception_attempts_tesseract(monkeypatch, tmp_path):
    class Reader:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("reader init failed")

    monkeypatch.setattr("backend.app.services.ocr.service.settings.OCR_MODEL_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(Reader=Reader))
    service = OCRService(languages=["en"], gpu=False)
    attempted = []
    monkeypatch.setattr(service, "_load_tesseract", lambda: attempted.append(True))

    service.load()
    assert attempted == [True]
    assert not service.is_loaded


def test_easyocr_import_error_is_unavailable(monkeypatch, tmp_path):
    original_import = __import__

    def no_easyocr(name, *args, **kwargs):
        if name == "easyocr":
            raise ImportError("EasyOCR missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", no_easyocr)
    monkeypatch.setattr("backend.app.services.ocr.service.settings.OCR_MODEL_DIR", str(tmp_path))
    service = OCRService(languages=["en"], gpu=False)
    monkeypatch.setattr(service, "_load_tesseract", lambda: None)

    service.load()
    assert service.extract(Image.new("RGB", (32, 32), "white")).status == "unavailable"
    assert any("ImportError" in warning for warning in service._load_warnings)


def test_tesseract_executable_unavailable_is_reported(monkeypatch):
    wrapper = SimpleNamespace(pytesseract=SimpleNamespace(tesseract_cmd="tesseract"))
    monkeypatch.setitem(sys.modules, "pytesseract", wrapper)
    monkeypatch.setattr("backend.app.services.ocr.service.shutil.which", lambda _command: None)
    service = OCRService(languages=["en"], gpu=False)

    service._load_tesseract()
    assert service._tesseract is None
    assert any("FileNotFoundError" in warning for warning in service._load_warnings)


def test_tesseract_missing_requested_language_pack_is_reported(monkeypatch):
    wrapper = SimpleNamespace(
        pytesseract=SimpleNamespace(tesseract_cmd="tesseract"),
        get_tesseract_version=lambda: "5.0-test",
        get_languages=lambda config="": ["eng"],
    )
    monkeypatch.setitem(sys.modules, "pytesseract", wrapper)
    monkeypatch.setattr("backend.app.services.ocr.service.shutil.which", lambda _command: "C:/fake/tesseract.exe")
    service = OCRService(languages=["en", "hi"], gpu=False)

    service._load_tesseract()
    assert service._tesseract is None
    assert any("hin" in warning for warning in service._load_warnings)


def test_tesseract_boxes_are_clipped_to_original_image_and_bad_confidence_is_unknown():
    data = {
        "text": ["outside", "overlap", "edge"],
        "left": [-100, -5, 95], "top": [4, -10, 20],
        "width": [20, 30, 15], "height": [10, 25, 15],
        "conf": [90, float("nan"), 75],
    }
    fake = SimpleNamespace(
        Output=SimpleNamespace(DICT=object()),
        image_to_data=lambda *_args, **_kwargs: data,
    )
    service = OCRService(languages=["en"], gpu=False)
    service._tesseract, service._engine = fake, "PyTesseract"

    blocks = service._run_tesseract(Image.new("RGB", (100, 50), "white"))
    assert [block.text for block in blocks] == ["overlap", "edge"]
    assert blocks[0].bbox == [[0, 0], [25, 0], [25, 15], [0, 15]]
    assert blocks[0].confidence is None
    assert blocks[1].bbox == [[95, 20], [100, 20], [100, 35], [95, 35]]
    for block in blocks:
        assert all(0 <= x <= 100 and 0 <= y <= 50 for x, y in block.bbox)


def test_tesseract_inference_exception_is_failed():
    fake = SimpleNamespace(
        Output=SimpleNamespace(DICT=object()),
        image_to_data=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("inference error")),
    )
    service = OCRService(languages=["en"], gpu=False)
    service._tesseract, service._engine = fake, "PyTesseract"

    result = service.extract(Image.new("RGB", (32, 32), "white"))
    assert result.status == "failed"
    assert any("inference failed" in warning for warning in result.warnings)


def test_easyocr_empty_output_is_completed_and_does_not_try_tesseract(monkeypatch):
    class EmptyReader:
        def readtext(self, _image):
            return []

    service = OCRService(languages=["en"], gpu=False)
    service._reader, service._engine = EmptyReader(), "EasyOCR"
    fallback_calls = []
    monkeypatch.setattr(service, "_load_tesseract", lambda: fallback_calls.append(True))

    result = service.extract(Image.new("RGB", (64, 32), "white"))
    assert result.status == "completed"
    assert not result.has_text
    assert result.no_text_reason
    assert fallback_calls == []


def test_easyocr_discards_malformed_boxes_and_nonfinite_confidence():
    class InvalidReader:
        def readtext(self, _image):
            return [
                ([[1, 1], [20, 1], [20, 10], [1, 10]], "bad confidence", float("nan")),
                ([[1, 1], [70, 1], [70, 10], [1, 10]], "out of bounds", 0.9),
                ([[1, 1], [20, 1], [20, 10]], "wrong shape", 0.9),
            ]

    service = OCRService(languages=["en"], gpu=False)
    service._reader, service._engine = InvalidReader(), "EasyOCR"
    result = service.extract(Image.new("RGB", (64, 32), "white"))

    assert result.status == "partial"
    assert result.blocks == []
    assert result.total_confidence is None
    assert result.warnings


def test_synthetic_benchmark_error_metrics_use_character_and_word_units():
    from experiments.ocr_runtime_smoke import _errors

    scores = _errors("one two", "one three")
    assert scores["cer"] == pytest.approx(4 / 7)
    assert scores["wer"] == pytest.approx(0.5)
    assert _errors(None, "") == {"cer": None, "wer": None}


def test_multiple_unavailable_engines_are_distinguished_from_empty_output(monkeypatch, tmp_path):
    def no_easyocr(name, *args, **kwargs):
        if name == "easyocr":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", no_easyocr)
    monkeypatch.setattr("backend.app.services.ocr.service.settings.OCR_MODEL_DIR", str(tmp_path))
    service = OCRService(languages=["en", "hi"], gpu=False)
    monkeypatch.setattr("backend.app.services.ocr.service.shutil.which", lambda _command: None)

    service.load()
    result = service.extract(Image.new("RGB", (64, 32), "white"))
    assert result.status == "unavailable"
    assert not result.has_text
    assert any("EasyOCR unavailable" in warning for warning in result.warnings)
    assert any("Tesseract fallback unavailable" in warning for warning in result.warnings)


@pytest.mark.skipif(os.environ.get("TRUTHLENS_REAL_OCR_TESTS") != "1", reason="Set TRUTHLENS_REAL_OCR_TESTS=1 for cache-only real OCR inference")
@pytest.mark.parametrize(
    ("language", "weight_name", "reference", "font_candidates"),
    [
        ("en", "english_g2.pth", "THE CITY OPENED 125 NEW SCHOOLS IN 2026", [r"C:\Windows\Fonts\arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]),
        ("hi", "devanagari.pth", "दिल्ली में 125 नए विद्यालय 2026 में खुले", [r"C:\Windows\Fonts\Nirmala.ttc", "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"]),
    ],
)
def test_real_easyocr_inference_is_cache_only(language, weight_name, reference, font_candidates):
    from PIL import ImageDraw, ImageFont
    from backend.app.config import settings

    model_path = Path(settings.OCR_MODEL_DIR) / weight_name
    if not model_path.is_file():
        pytest.skip(f"Local EasyOCR weight missing: {weight_name}; downloads are disabled")
    font_path = next((Path(path) for path in font_candidates if Path(path).is_file()), None)
    if font_path is None:
        pytest.skip(f"No local font available for {language} synthetic fixture")

    image = Image.new("RGB", (1600, 280), "white")
    font = ImageFont.truetype(str(font_path), 68)
    ImageDraw.Draw(image).text((40, 80), reference, font=font, fill="black")
    service = OCRService(languages=[language], gpu=False)
    service.load()
    if service._reader is None:
        pytest.skip("EasyOCR could not initialize from local weights")
    try:
        result = service.extract(image)
        assert result.status in {"completed", "partial"}
        assert result.has_text, f"Real {language} OCR returned no text: {result.warnings}"
        assert result.engine == "EasyOCR"
        assert result.blocks
        for block in result.blocks:
            assert len(block.bbox) == 4
            assert all(0 <= x <= image.width and 0 <= y <= image.height for x, y in block.bbox)
    finally:
        service.unload()
