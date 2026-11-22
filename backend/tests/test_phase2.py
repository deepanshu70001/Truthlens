"""Offline regression tests for Phase 2 image-analysis contracts."""
import hashlib
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image


def image_bytes(fmt="JPEG", size=(96, 80)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (80, 120, 160)).save(buffer, fmt)
    return buffer.getvalue()


def test_forensics_contract_format_scope_and_hashes():
    from backend.app.services.forensics.analyzer import analyze_forensics
    from backend.app.services.hashing.service import compute_hashes

    jpeg = image_bytes()
    result = analyze_forensics(jpeg, "JPEG")
    data = result.to_dict()
    assert result.status == "completed"
    assert len(data["signals"]) == 6
    assert data["overall_anomaly_score"] is None
    assert all(s["status"] in {"completed", "skipped"} for s in data["signals"])
    assert compute_hashes(jpeg).sha256 == hashlib.sha256(jpeg).hexdigest() == compute_hashes(jpeg).sha256
    assert compute_hashes(jpeg).phash != compute_hashes(jpeg).sha256
    png = analyze_forensics(image_bytes("PNG"), "PNG").to_dict()
    jpeg_signal = next(s for s in png["signals"] if s["name"] == "jpeg_artifacts")
    assert jpeg_signal["status"] == "skipped" and jpeg_signal["score"] is None
    ela = next(s for s in png["signals"] if s["name"] == "error_level_analysis")
    assert "not directly comparable" in ela["limitations"]
    from backend.app.services.metadata.service import extract_metadata
    assert extract_metadata(image_bytes("PNG")).has_exif is False


def test_forensics_corrupt_data_and_artifact_failure(monkeypatch, tmp_path):
    from backend.app.services.forensics import analyzer
    from backend.app.config import settings

    failed = analyzer.analyze_forensics(b"not an image")
    assert failed.status == "failed" and failed.overall_anomaly_score is None
    monkeypatch.setattr(settings, "STORAGE_PATH", str(tmp_path))
    monkeypatch.setattr(analyzer.cv2, "imwrite", lambda *args, **kwargs: False)
    result = analyzer.analyze_forensics(image_bytes(), "JPEG", analysis_id="abc")
    assert result.status == "partial" and result.signals[0].status == "partial"
    assert result.signals[0].score is not None
    assert result.signals[0].details.get("artifact_url") is None
    assert any("artifact could not be saved" in warning for warning in result.warnings)


def test_ocr_preserves_text_and_structured_confidence():
    from backend.app.services.ocr.service import OCRService

    class Reader:
        def readtext(self, _image):
            return [([[1, 2], [30, 2], [30, 12], [1, 12]], "No. 1: ₹500!", 0.91)]

    service = OCRService(languages=["en", "hi"], gpu=False)
    service._reader, service._engine = Reader(), "EasyOCR"
    result = service.extract(Image.new("RGB", (40, 20), "white"))
    assert result.status == "completed" and result.engine == "EasyOCR"
    assert result.text == "No. 1: ₹500!"
    assert result.normalized_text == result.text
    assert result.blocks[0].bbox == [[1,2],[30,2],[30,12],[1,12]]
    assert result.total_confidence == pytest.approx(0.91)
    assert json.loads(json.dumps(result.to_dict()))["raw_text"] == result.text


def test_ocr_empty_invalid_unavailable_and_fallback(monkeypatch):
    from backend.app.services.ocr.service import OCRService, OCRBlock

    class Empty:
        def readtext(self, _image): return []
    service = OCRService(languages=["en"], gpu=False)
    service._reader, service._engine = Empty(), "EasyOCR"
    empty = service.extract(Image.new("RGB", (16, 16)))
    assert empty.status == "completed" and not empty.has_text and empty.blocks == []

    class Invalid:
        def readtext(self, _image): return [(np.zeros((3, 2)), "bad", float("nan"))]
    service._reader = Invalid()
    invalid = service.extract(Image.new("RGB", (16, 16)))
    assert invalid.status == "partial" and not invalid.blocks

    service.unload()
    unavailable = service.extract(Image.new("RGB", (16, 16)))
    assert unavailable.status == "unavailable"

    class Broken:
        def readtext(self, _image): raise RuntimeError("failed")
    service._reader, service._engine = Broken(), "EasyOCR"
    def fallback_available():
        service._tesseract = object()
        service._engine = "PyTesseract"
    monkeypatch.setattr(service, "_load_tesseract", fallback_available)
    monkeypatch.setattr(service, "_run_tesseract", lambda _image, _warnings=None: [OCRBlock("fallback", None, [[0,0],[1,0],[1,1],[0,1]])])
    fallback = service.extract(Image.new("RGB", (16, 16)))
    assert fallback.status == "completed" and fallback.text == "fallback"
    assert fallback.engine == "PyTesseract"

    bad_config = OCRService(languages=[], gpu=False)
    bad_config.load()
    assert bad_config.extract(Image.new("RGB", (4, 4))).status == "unavailable"


def test_forensic_signals_are_informational_not_fusion_votes():
    from backend.app.services.forensics.analyzer import ForensicsResult, ForensicSignal
    from backend.app.services.evidence.schema import EvidenceCollection, build_evidence_from_forensics
    from backend.app.services.fusion.engine import fuse_evidence

    result = ForensicsResult(signals=[ForensicSignal("ela", 0.99, "high indicator", "encoding-dependent")])
    evidence = EvidenceCollection(analysis_id="example")
    for item in build_evidence_from_forensics(result):
        evidence.add(item)
    assert evidence.items[0].direction == "informational"
    assert evidence.items[0].data["indicator_score"] == 0.99
    assert fuse_evidence(evidence).label == "inconclusive"


class _FakeBox:
    cls = np.array([0])
    conf = np.array([0.875])
    xyxy = np.array([[10.0, 12.0, 40.0, 50.0]])


def test_yolo_real_output_and_empty_output_are_distinct():
    from backend.app.services.vision.detector import ObjectDetector

    detector = ObjectDetector()
    detector._loaded = True
    detector._model = SimpleNamespace(predict=lambda **kw: [SimpleNamespace(boxes=[_FakeBox()], names={0:"person"})])
    result = detector.detect(Image.new("RGB", (64, 64)))
    assert result.status == "completed" and result.object_count == 1
    assert result.objects[0].confidence == pytest.approx(0.875)
    assert result.objects[0].bbox == [10,12,40,50]
    assert result.image_width == 64 and result.image_height == 64
    detector._model = SimpleNamespace(predict=lambda **kw: [SimpleNamespace(boxes=[], names={})])
    empty = detector.detect(Image.new("RGB", (64, 64)))
    assert empty.status == "completed" and empty.object_count == 0


def test_yolo_unavailable_and_invalid_outputs():
    from backend.app.services.vision.detector import ObjectDetector

    detector = ObjectDetector()
    assert detector.detect(Image.new("RGB", (20, 20))).status == "unavailable"
    detector._loaded = True
    detector._model = SimpleNamespace(predict=lambda **kw: [SimpleNamespace(boxes=[SimpleNamespace(cls=[0], conf=[float("nan")], xyxy=[[0,0,1,1]])], names={0:"person"})])
    assert detector.detect(Image.new("RGB", (20, 20))).status == "failed"

    detector._model = SimpleNamespace(predict=lambda **kw: (_ for _ in ()).throw(RuntimeError("inference failed")))
    assert detector.detect(Image.new("RGB", (20, 20))).status == "failed"


def test_yolo_missing_dependency_is_unavailable(monkeypatch):
    import builtins
    from backend.app.services.vision.detector import ObjectDetector

    original_import = builtins.__import__
    def no_ultralytics(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError("test dependency unavailable")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_ultralytics)
    detector = ObjectDetector()
    detector.load()
    assert detector.detect(Image.new("RGB", (20, 20))).status == "unavailable"


class _ToySiglip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        norm = torch.nn.LayerNorm(3)
        self.vision_model = SimpleNamespace(encoder=SimpleNamespace(layers=[SimpleNamespace(layer_norm2=norm)]))
        self.layer = norm
        self.classifier = torch.nn.Linear(3, 2)
        with torch.no_grad():
            self.classifier.weight.zero_()
            self.classifier.bias.zero_()
            self.classifier.weight[1, 0] = 1.0
        self.config = SimpleNamespace(id2label={0:"Fake", 1:"Real"}, _name_or_path="toy-siglip")

    def forward(self, pixel_values):
        tokens = pixel_values.flatten(2).transpose(1,2)
        features = self.layer(tokens)
        return SimpleNamespace(logits=self.classifier(features.mean(dim=1)))


class _ToyProcessor:
    def __call__(self, images, return_tensors):
        return {"pixel_values": torch.tensor([[[[1,2],[3,4]], [[0,0],[0,0]], [[0,0],[0,0]]]], dtype=torch.float32)}


def test_gradcam_creates_real_attribution_artifacts(monkeypatch, tmp_path):
    from backend.app.services.xai.gradcam import generate_heatmap
    from backend.app.config import settings

    monkeypatch.setattr(settings, "STORAGE_PATH", str(tmp_path))
    model = _ToySiglip()
    result = generate_heatmap(model, _ToyProcessor(), Image.new("RGB", (20, 16), (20,40,60)), "123e4567-e89b-12d3-a456-426614174000", torch.device("cpu"), target_class=1)
    assert result.status == "completed"
    assert result.target_class == "Real" and result.target_score is not None
    assert Image.open(result.heatmap_path).size == (20,16)
    assert Image.open(result.overlay_path).size == (20,16)
    assert result.to_dict()["overlay_url"].endswith("variant=overlay")


def test_gradcam_incompatible_model_is_explicit_failure():
    from backend.app.services.xai.gradcam import generate_heatmap

    result = generate_heatmap(torch.nn.Linear(2, 2), _ToyProcessor(), Image.new("RGB", (4,4)), "analysis-id", torch.device("cpu"))
    assert result.status == "failed" and result.heatmap_path is None
