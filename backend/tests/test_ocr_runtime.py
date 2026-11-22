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


def test_unavailable_tesseract_is_not_reinitialized_for_each_image(monkeypatch):
    wrapper = SimpleNamespace(pytesseract=SimpleNamespace(tesseract_cmd="tesseract"))
    monkeypatch.setitem(sys.modules, "pytesseract", wrapper)
    calls = []
    monkeypatch.setattr("backend.app.services.ocr.service.shutil.which", lambda command: calls.append(command) or None)

    class EmptyReader:
        def readtext(self, _image): return []

    service = OCRService(languages=["en"], gpu=False)
    service._reader, service._engine = EmptyReader(), "EasyOCR"
    for _ in range(3):
        result = service.extract(Image.new("RGB", (32, 16), "white"))
        assert result.status == "completed" and not result.has_text
    assert calls == ["tesseract"]


def test_tesseract_boxes_are_clipped_to_original_image_and_bad_confidence_is_unknown():
    data = {
        "text": ["outside", "overlap", "edge"],
        "left": [-100, -5, 95], "top": [4, -10, 20],
        "width": [20, 30, 15], "height": [10, 25, 15],
        "conf": [90, "not-a-confidence", 75],
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


def test_tesseract_missing_confidence_keeps_text_and_coordinates():
    fake = SimpleNamespace(
        Output=SimpleNamespace(DICT=object()),
        image_to_data=lambda *_args, **_kwargs: {
            "text": ["visible text"], "left": [2], "top": [3], "width": [10], "height": [5],
        },
    )
    service = OCRService(languages=["en"], gpu=False)
    service._tesseract, service._engine = fake, "PyTesseract"
    warnings = []
    blocks = service._run_tesseract(Image.new("RGB", (32, 16)), warnings)
    assert len(blocks) == 1 and blocks[0].text == "visible text"
    assert blocks[0].confidence is None
    assert warnings


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


def test_easyocr_empty_output_attempts_tesseract_and_preserves_fallback_result(monkeypatch):
    class EmptyReader:
        def readtext(self, _image):
            return []

    service = OCRService(languages=["en"], gpu=False)
    service._reader, service._engine = EmptyReader(), "EasyOCR"
    fallback_calls = []
    service._load_tesseract = lambda: setattr(service, "_tesseract", object())
    service._run_tesseract = lambda _image, _warnings=None: (fallback_calls.append(True) or [OCRBlock("fallback text", .8, [[0,0],[20,0],[20,10],[0,10]])])

    result = service.extract(Image.new("RGB", (64, 32), "white"))
    assert result.status == "completed"
    assert result.has_text and result.text == "fallback text"
    assert result.engine == "PyTesseract"
    assert fallback_calls == [True]
    assert any("fallback" in warning.lower() for warning in result.warnings)


def test_easyocr_successful_text_does_not_invoke_fallback():
    class Reader:
        def readtext(self, _image):
            return [([[1, 1], [20, 1], [20, 10], [1, 10]], "read me", 0.8)]

    service = OCRService(languages=["en"], gpu=False)
    service._reader, service._engine = Reader(), "EasyOCR"
    service._tesseract_attempted = True
    result = service.extract(Image.new("RGB", (64, 32), "white"))
    assert result.status == "completed" and result.has_text
    assert result.engine == "EasyOCR"
    assert not any("fallback" in warning.lower() for warning in result.warnings)


def test_empty_result_from_both_engines_is_completed_no_text():
    class EmptyReader:
        def readtext(self, _image): return []

    service = OCRService(languages=["en"], gpu=False)
    service._reader, service._engine = EmptyReader(), "EasyOCR"
    service._load_tesseract = lambda: setattr(service, "_tesseract", object())
    service._run_tesseract = lambda _image, _warnings=None: []
    result = service.extract(Image.new("RGB", (64, 32), "white"))
    assert result.status == "completed" and not result.has_text
    assert result.no_text_reason
    assert "Tesseract completed" in " ".join(result.warnings)


def test_primary_exception_then_fallback_success_is_completed_with_warning():
    class BrokenReader:
        def readtext(self, _image): raise RuntimeError("secret path C:/private/model")

    service = OCRService(languages=["en"], gpu=False)
    service._reader, service._engine = BrokenReader(), "EasyOCR"
    service._load_tesseract = lambda: setattr(service, "_tesseract", object())
    service._run_tesseract = lambda _image, _warnings=None: [OCRBlock("fallback", .7, [[0,0],[10,0],[10,8],[0,8]])]
    result = service.extract(Image.new("RGB", (32, 16), "white"))
    assert result.status == "completed" and result.text == "fallback"
    assert result.engine == "PyTesseract"
    serialized = str(result.to_dict())
    assert "secret path" not in serialized and "C:/private" not in serialized


def test_primary_empty_then_fallback_failure_keeps_successful_no_text():
    class EmptyReader:
        def readtext(self, _image): return []

    service = OCRService(languages=["en"], gpu=False)
    service._reader, service._engine = EmptyReader(), "EasyOCR"
    service._load_tesseract = lambda: setattr(service, "_tesseract", object())
    def fail(_image, _warnings=None): raise TimeoutError("private provider detail")
    service._run_tesseract = fail
    result = service.extract(Image.new("RGB", (32, 16), "white"))
    assert result.status == "completed" and not result.has_text
    assert "private provider detail" not in str(result.to_dict())


def test_empty_and_non_image_inputs_return_safe_failure():
    service = OCRService(languages=["en"], gpu=False)
    for invalid in (None, b"", object()):
        result = service.extract(invalid)
        assert result.status == "failed"
        assert result.no_text_reason


def test_corrupt_decoded_image_is_reported_without_crashing(monkeypatch):
    image = Image.new("RGB", (16, 16))
    monkeypatch.setattr(image, "load", lambda: (_ for _ in ()).throw(OSError("corrupt input")))
    service = OCRService(languages=["en"], gpu=False)
    service._reader = SimpleNamespace(readtext=lambda _image: [])
    result = service.extract(image)
    assert result.status == "failed"
    assert "corrupt input" not in str(result.to_dict())


def test_unsupported_image_format_is_rejected():
    image = Image.new("RGB", (16, 16))
    image.format = "PDF"
    service = OCRService(languages=["en"], gpu=False)
    service._reader = SimpleNamespace(readtext=lambda _image: [])
    assert service.extract(image).status == "failed"


def test_easyocr_bbox_clipping_and_confidence_validation_keep_text():
    class Reader:
        def readtext(self, _image):
            return [([[-4, 2], [20, 2], [20, 12], [-4, 12]], "clipped", float("nan"))]

    image = Image.new("RGB", (16, 16), "white")
    before = image.tobytes()
    service = OCRService(languages=["en"], gpu=False)
    service._reader, service._engine = Reader(), "EasyOCR"
    result = service.extract(image)
    assert result.status == "partial" and result.text == "clipped"
    assert result.blocks[0].bbox == [[0, 2], [16, 2], [16, 12], [0, 12]]
    assert result.blocks[0].confidence is None and result.total_confidence is None
    assert image.tobytes() == before


@pytest.mark.asyncio
async def test_ocr_stage_timeout_maps_to_e1_failed_status():
    import time
    from backend.app.services.orchestrator import _run_stage

    class DB:
        async def commit(self): pass
        async def rollback(self): pass
        def add(self, _event): pass

    analysis = SimpleNamespace(id="ocr-timeout", current_step=None, completed_steps="[]")
    stages = {}
    result = await _run_stage(
        "ocr", lambda: time.sleep(.05), analysis, DB(), stages, timeout_seconds=.001
    )
    assert result is None
    assert stages["ocr"].status == "failed"
    assert stages["ocr"].error_category == "STAGE_TIMEOUT"


@pytest.mark.asyncio
async def test_stage_warning_serialization_redacts_local_path_and_secret():
    from backend.app.services.orchestrator import _run_stage

    class DB:
        async def commit(self): pass
        async def rollback(self): pass
        def add(self, _event): pass

    analysis = SimpleNamespace(id="ocr-safe", current_step=None, completed_steps="[]")
    stages = {}
    await _run_stage(
        "ocr",
        lambda: SimpleNamespace(status="completed", has_text=True,
                                warnings=[r"Model at C:\private\weights; api_key=secret-value"]),
        analysis, DB(), stages,
    )
    serialized = str(stages["ocr"].to_dict())
    assert "C:\\private" not in serialized
    assert "secret-value" not in serialized
    assert "[redacted]" in serialized


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
    assert [block.text for block in result.blocks] == ["bad confidence", "out of bounds"]
    assert result.blocks[0].confidence is None
    assert result.blocks[1].bbox == [[1, 1], [64, 1], [64, 10], [1, 10]]
    assert result.total_confidence == pytest.approx(.9)
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
