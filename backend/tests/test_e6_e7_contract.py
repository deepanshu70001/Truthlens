"""Deterministic E6/E7 contracts; model and network dependencies are mocked."""

import hashlib
import io
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image


def image_bytes(fmt="JPEG", size=(96, 80), color=(83, 129, 177)):
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def grayscale_image_bytes(pixels, fmt="JPEG", *, quality=95):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(buffer, format=fmt, quality=quality)
    return buffer.getvalue()


def test_forensics_jpeg_and_png_applicability_and_heuristic_contract():
    from backend.app.services.forensics.analyzer import analyze_forensics

    jpeg = analyze_forensics(image_bytes("JPEG"), generate_artifacts=False)
    assert jpeg.status == "completed"
    assert {signal.name for signal in jpeg.signals} == {
        "error_level_analysis", "noise_consistency", "jpeg_artifacts",
        "edge_consistency", "texture_consistency", "resampling_clues",
    }
    assert all(signal.status == "completed" for signal in jpeg.signals)
    assert all(signal.output_reference for signal in jpeg.signals)
    assert all(signal.runtime_ms >= 0 for signal in jpeg.signals)
    serialized = jpeg.to_dict()
    assert all(
        item["score_type"] in (None, "uncalibrated_heuristic_indicator_0_to_1_not_probability")
        for item in serialized["signals"]
    )

    png = analyze_forensics(image_bytes("PNG"), generate_artifacts=False)
    for name in ("error_level_analysis", "jpeg_artifacts"):
        signal = next(value for value in png.signals if value.name == name)
        assert signal.status == "completed" and signal.outcome == "not_applicable"
        assert signal.score is None


def test_forensics_rejects_bad_bytes_pixel_limit_and_handles_small_images(monkeypatch):
    from backend.app.config import settings
    from backend.app.services.forensics.analyzer import analyze_forensics

    assert analyze_forensics(b"").status == "failed"
    assert analyze_forensics(b"not an image").status == "failed"
    assert analyze_forensics(image_bytes("JPEG", size=(16, 16)), generate_artifacts=False).status == "completed"
    small = analyze_forensics(image_bytes("JPEG", size=(2, 2)), generate_artifacts=False)
    assert small.status == "completed"
    assert next(item for item in small.signals if item.name == "edge_consistency").outcome == "not_applicable"
    assert next(item for item in small.signals if item.name == "jpeg_artifacts").outcome == "not_applicable"
    monkeypatch.setattr(settings, "MAX_IMAGE_PIXELS", 4)
    assert analyze_forensics(image_bytes("PNG", size=(8, 8)), generate_artifacts=False).status == "failed"


def test_resampling_abstains_on_flat_and_near_uniform_jpegs():
    from backend.app.services.forensics.analyzer import analyze_forensics

    flat_gray = grayscale_image_bytes(np.full((80, 96), 128, dtype=np.uint8))
    flat_rgb = image_bytes("JPEG", size=(96, 80), color=(128, 128, 128))
    rng = np.random.default_rng(20261123)
    near_uniform = grayscale_image_bytes(
        np.clip(128 + rng.integers(-1, 2, size=(80, 96)), 0, 255)
    )

    for payload in (flat_gray, flat_rgb, near_uniform):
        result = analyze_forensics(payload, generate_artifacts=False)
        signal = next(
            item for item in result.signals
            if item.name == "resampling_clues"
        )
        assert signal.status == "completed"
        assert signal.outcome == "not_applicable"
        assert signal.score is None
        assert signal.details["reason_code"] == "INSUFFICIENT_TEXTURE_FOR_RESAMPLING"
        assert signal.details["second_difference_rms"] < signal.details["minimum_second_difference_rms"]
        serialized = next(
            item for item in result.to_dict()["signals"]
            if item["name"] == "resampling_clues"
        )
        assert serialized["outcome"] == "not_applicable"
        assert serialized["score"] is None
        assert serialized["details"]["reason_code"] == "INSUFFICIENT_TEXTURE_FOR_RESAMPLING"


def test_resampling_handles_textured_grid_small_corrupt_and_non_jpeg_inputs():
    from pathlib import Path
    from backend.app.services.forensics.analyzer import analyze_forensics

    photo = (Path(__file__).parent / "fixtures" / "test_photo.jpg").read_bytes()
    photo_signal = next(
        item for item in analyze_forensics(photo, generate_artifacts=False).signals
        if item.name == "resampling_clues"
    )
    assert photo_signal.status == "completed" and photo_signal.outcome == "success"
    assert photo_signal.score is not None

    yy, xx = np.indices((96, 96))
    grid = np.where(((xx // 8 + yy // 8) % 2) == 0, 40, 220).astype(np.uint8)
    grid_signal = next(
        item for item in analyze_forensics(grayscale_image_bytes(grid), generate_artifacts=False).signals
        if item.name == "resampling_clues"
    )
    assert grid_signal.status == "completed" and grid_signal.outcome == "success"
    assert grid_signal.details["second_difference_rms"] >= 2.0

    small_signal = next(
        item for item in analyze_forensics(
            image_bytes("JPEG", size=(3, 16)), generate_artifacts=False
        ).signals if item.name == "resampling_clues"
    )
    assert small_signal.outcome == "not_applicable" and small_signal.score is None
    assert small_signal.details["reason_code"] == "IMAGE_TOO_SMALL_FOR_RESAMPLING"

    corrupt = analyze_forensics(b"not an image", generate_artifacts=False)
    assert corrupt.status == "failed" and corrupt.signals == []

    png_signal = next(
        item for item in analyze_forensics(
            grayscale_image_bytes(grid, fmt="PNG"), generate_artifacts=False
        ).signals if item.name == "resampling_clues"
    )
    assert png_signal.status == "completed" and png_signal.outcome == "success"


def test_forensics_keeps_good_signals_when_one_engine_fails(monkeypatch):
    from backend.app.services.forensics import analyzer

    monkeypatch.setattr(analyzer.cv2, "Canny", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("private diagnostic")))
    result = analyzer.analyze_forensics(image_bytes("JPEG"), generate_artifacts=False)
    assert result.status == "partial"
    assert next(item for item in result.signals if item.name == "edge_consistency").status == "failed"
    assert any(item.status == "completed" for item in result.signals)
    assert "private diagnostic" not in json.dumps(result.to_dict())


def test_forensic_invalid_score_is_safe_failure():
    from backend.app.services.forensics.analyzer import _signal

    signal = _signal("test", lambda: (float("nan"), "bad", {}), "test limitation")
    assert signal.status == "failed" and signal.score is None
    assert signal.error_category == "SIGNAL_EXECUTION_ERROR"


def test_yolo_local_initialization_identity_and_coordinate_contract(monkeypatch, tmp_path):
    from backend.app.services import vision
    from backend.app.services.vision.detector import ObjectDetector

    weights = tmp_path / "local-yolo.pt"
    weights.write_bytes(b"test-only model fixture")
    seen = {}

    class MockYOLO:
        def __init__(self, path):
            seen["path"] = path

        def predict(self, **kwargs):
            seen["predict"] = kwargs
            box = SimpleNamespace(cls=[0], conf=[0.875], xyxy=[[10.0, 12.0, 40.0, 50.0]])
            return [SimpleNamespace(boxes=[box], names={0: "person"})]

    monkeypatch.setattr(vision.detector, "YOLO_MODEL_NAME", str(weights))
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=MockYOLO))
    detector = ObjectDetector()
    detector.load(device="cpu")
    result = detector.detect(Image.new("RGB", (64, 64)))
    assert detector.is_loaded and result.status == "completed"
    assert result.model_sha256 == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert result.objects[0].bbox == [10.0, 12.0, 40.0, 50.0]
    assert result.coordinate_space == "exif_corrected_input_pixels"
    assert result.objects[0].confidence == pytest.approx(0.875)
    assert seen["predict"]["imgsz"] == 640 and seen["predict"]["max_det"] == 100


def test_yolo_empty_unavailable_and_malformed_outputs(monkeypatch, tmp_path):
    from backend.app.services import vision
    from backend.app.services.vision.detector import ObjectDetector

    detector = ObjectDetector()
    assert detector.detect(Image.new("RGB", (20, 20))).status == "unavailable"
    missing = tmp_path / "missing.pt"
    monkeypatch.setattr(vision.detector, "YOLO_MODEL_NAME", str(missing))
    detector.load()
    assert not detector.is_loaded
    assert detector.detect(Image.new("RGB", (20, 20))).status == "unavailable"

    detector._loaded = True
    detector._model = SimpleNamespace(predict=lambda **_: [SimpleNamespace(boxes=[], names={})])
    empty = detector.detect(Image.new("RGB", (20, 20)))
    assert empty.status == "completed" and empty.outcome == "no_results"
    detector._model = SimpleNamespace(predict=lambda **_: [SimpleNamespace(
        boxes=[SimpleNamespace(cls=[0], conf=[0.7], xyxy=[[-1, 0, 2, 3]])], names={0: "person"}
    )])
    assert detector.detect(Image.new("RGB", (20, 20))).status == "failed"


def test_yolo_failure_modes_and_bounds_verification(monkeypatch, tmp_path):
    import builtins
    from backend.app.config import settings
    from backend.app.services import vision
    from backend.app.services.vision.detector import ObjectDetector

    # 1. Missing Ultralytics dependency
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError("No module named 'ultralytics'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    detector = ObjectDetector()
    detector.load()
    assert not detector.is_loaded
    unavail = detector.detect(Image.new("RGB", (20, 20)))
    assert unavail.status == "unavailable" and unavail.outcome == "unavailable"
    assert "Ultralytics is not installed" in detector._load_warning
    monkeypatch.setattr(builtins, "__import__", real_import)

    # 2. Missing local weights & no automatic model download
    yolo_called = []

    class SpyYOLO:
        def __init__(self, path):
            yolo_called.append(path)

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=SpyYOLO))
    missing_weights = tmp_path / "nonexistent_weights.pt"
    monkeypatch.setattr(vision.detector, "YOLO_MODEL_NAME", str(missing_weights))
    detector = ObjectDetector()
    detector.load()
    assert not detector.is_loaded
    assert len(yolo_called) == 0, "YOLO constructor must not be called when weights are absent (prevents auto-download)"
    assert "Configured YOLO weights are not present locally; automatic model downloads are disabled." in detector._load_warning
    assert detector.detect(Image.new("RGB", (20, 20))).status == "unavailable"

    # 3. Invalid model initialization (corrupted weights raising exception)
    class FailingYOLO:
        def __init__(self, path):
            raise RuntimeError("Corrupted checkpoint header")

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FailingYOLO))
    valid_file = tmp_path / "corrupt_weights.pt"
    valid_file.write_bytes(b"invalid data")
    monkeypatch.setattr(vision.detector, "YOLO_MODEL_NAME", str(valid_file))
    detector = ObjectDetector()
    detector.load()
    assert not detector.is_loaded
    assert "RuntimeError" in detector._load_warning
    assert detector.detect(Image.new("RGB", (20, 20))).status == "unavailable"

    # 4. Invalid or non-finite confidence values
    detector._loaded = True
    failed_conf = None
    for bad_conf in [float("nan"), float("inf"), float("-inf"), -0.01, 1.05]:
        detector._model = SimpleNamespace(predict=lambda **_: [SimpleNamespace(
            boxes=[SimpleNamespace(cls=[0], conf=[bad_conf], xyxy=[[5.0, 5.0, 15.0, 15.0]])], names={0: "person"}
        )])
        failed_conf = detector.detect(Image.new("RGB", (20, 20)))
        assert failed_conf.status == "failed" and failed_conf.outcome == "failed"

    # 5. Out-of-bounds bounding boxes (on a 50x50 image)
    for bad_box in [
        [-1.0, 5.0, 20.0, 20.0],
        [5.0, -1.0, 20.0, 20.0],
        [5.0, 5.0, 55.0, 20.0],
        [5.0, 5.0, 20.0, 55.0],
        [20.0, 5.0, 10.0, 20.0],
        [5.0, 20.0, 20.0, 10.0],
        [float("nan"), 5.0, 20.0, 20.0],
        [5.0, 5.0, 20.0],
    ]:
        detector._model = SimpleNamespace(predict=lambda **_: [SimpleNamespace(
            boxes=[SimpleNamespace(cls=[0], conf=[0.9], xyxy=[bad_box])], names={0: "person"}
        )])
        failed_box = detector.detect(Image.new("RGB", (50, 50)))
        assert failed_box.status == "failed" and failed_box.outcome == "failed"

    # 6. Malformed model output
    detector._model = SimpleNamespace(predict=lambda **_: [])
    assert detector.detect(Image.new("RGB", (20, 20))).outcome == "no_results"

    detector._model = SimpleNamespace(predict=lambda **_: [None])
    assert detector.detect(Image.new("RGB", (20, 20))).status == "failed"

    detector._model = SimpleNamespace(predict=lambda **_: [SimpleNamespace(boxes=[SimpleNamespace(cls=[], conf=[], xyxy=[])], names={})])
    assert detector.detect(Image.new("RGB", (20, 20))).status == "failed"

    # 7. Image-size limit enforcement
    monkeypatch.setattr(settings, "MAX_IMAGE_PIXELS", 500)
    oversized = Image.new("RGB", (30, 30))  # 900 > 500
    failed_size = detector.detect(oversized)
    assert failed_size.status == "failed" and failed_size.outcome == "failed"
    monkeypatch.setattr(settings, "MAX_IMAGE_PIXELS", 40000000)

    # 8. Safe error serialization (no path leakage, valid contract)
    unavail_dict = unavail.to_dict()
    assert "objects" in unavail_dict
    assert unavail_dict["status"] == "unavailable"
    assert unavail_dict["outcome"] == "unavailable"
    assert "YOLO detection confidence; not an authenticity probability." in unavail_dict["confidence_semantics"]
    assert "Object presence or absence is not proof of authenticity" in unavail_dict["limitations"][0]
    assert "/" not in unavail_dict["model_name"] and "\\" not in unavail_dict["model_name"]
    json.dumps(unavail_dict)
    assert failed_conf is not None
    json.dumps(failed_conf.to_dict())


def test_yolo_real_runtime_integration_and_fusion_contract():
    """Verify actual local YOLOv8n checkpoint discovery, loading, inference, and fusion exclusion."""
    from pathlib import Path
    from backend.app.config import settings
    from backend.app.services.vision.detector import ObjectDetector, _resolve_model_path
    from backend.app.services.evidence.schema import build_evidence_from_vision, EvidenceCollection
    from backend.app.services.fusion.engine import fuse_evidence

    resolved = _resolve_model_path(settings.VISION_MODEL)
    if not resolved.is_file():
        pytest.skip("Approved local YOLOv8n checkpoint is not present on disk")

    try:
        import ultralytics
    except ImportError:
        pytest.skip("Ultralytics package is not installed")

    detector = ObjectDetector()
    detector.load(device=settings.DEVICE)
    assert detector.is_loaded
    assert detector._model_sha256 == "31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95"

    fixture_path = Path(__file__).parent / "fixtures" / "test_photo.jpg"
    assert fixture_path.is_file()
    img = Image.open(fixture_path)

    result = detector.detect(img)
    assert result.status == "completed"
    assert result.outcome == "no_results"
    assert result.object_count == 0
    assert result.objects == []
    assert result.model_sha256 == "31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95"
    assert result.model_name == "yolov8n.pt"

    serialized = result.to_dict()
    assert serialized["status"] == "completed"
    assert serialized["outcome"] == "no_results"
    assert "YOLO detection confidence; not an authenticity probability." in serialized["confidence_semantics"]
    assert "Object presence or absence is not proof of authenticity" in serialized["limitations"][0]

    # Verify fusion contract: vision evidence must be informational only
    evidence = build_evidence_from_vision(result)
    assert len(evidence) == 1
    assert evidence[0].type == "contextual_evidence"
    assert evidence[0].direction == "informational"
    assert evidence[0].source == "yolov8_object_detection"

    fusion = fuse_evidence(EvidenceCollection(items=evidence))
    assert fusion.evidence_summary["synthetic_signals"] == 0
    assert fusion.evidence_summary["authentic_signals"] == 0
    assert fusion.label == "inconclusive"


class _ToySiglip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer = torch.nn.LayerNorm(3)
        self.vision_model = SimpleNamespace(encoder=SimpleNamespace(layers=[SimpleNamespace(layer_norm2=layer)]))
        self.layer = layer
        self.classifier = torch.nn.Linear(3, 2)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
            self.classifier.bias.zero_()
        self.config = SimpleNamespace(id2label={0: "Authentic", 1: "Synthetic"}, _name_or_path="test-siglip")

    def forward(self, pixel_values):
        tokens = pixel_values.flatten(2).transpose(1, 2)
        return SimpleNamespace(logits=self.classifier(self.layer(tokens).mean(dim=1)))


class _ToyProcessor:
    def __call__(self, images, return_tensors):
        values = torch.tensor([[[[0.0, 1.0], [3.0, 6.0]], [[2.0, 1.0], [0.0, 2.0]], [[1.0, 2.0], [5.0, 1.0]]]])
        return {"pixel_values": values}


def test_gradcam_artifacts_align_and_leave_classifier_prediction_unchanged(monkeypatch, tmp_path):
    from backend.app.config import settings
    from backend.app.services.xai.gradcam import generate_heatmap

    monkeypatch.setattr(settings, "STORAGE_PATH", str(tmp_path))
    model = _ToySiglip().eval()
    source = Image.new("RGB", (26, 18), (40, 90, 120))
    pixel_values = _ToyProcessor()(source, return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        before = model(pixel_values).logits.clone()
    weights_before = [parameter.detach().clone() for parameter in model.parameters()]
    result = generate_heatmap(model, _ToyProcessor(), source, "analysis-test", torch.device("cpu"), target_class=0)
    with torch.no_grad():
        after = model(pixel_values).logits.clone()
    assert result.status == "completed" and result.target_class == "Authentic"
    assert Image.open(result.heatmap_path).size == source.size
    assert Image.open(result.overlay_path).size == source.size
    assert torch.equal(before, after)
    assert all(torch.equal(old, new) for old, new in zip(weights_before, model.parameters()))


def test_gradcam_missing_target_layer_is_unavailable():
    from backend.app.services.xai.gradcam import generate_heatmap

    result = generate_heatmap(torch.nn.Linear(2, 2), _ToyProcessor(), Image.new("RGB", (8, 8)), "x", torch.device("cpu"))
    assert result.status == "unavailable" and result.outcome == "unavailable"


def test_gradcam_finds_nested_siglip_vision_model_and_cleans_up():
    from backend.app.services.xai.gradcam import GradCAM

    target = torch.nn.LayerNorm(4)
    nested = SimpleNamespace(
        vision_model=SimpleNamespace(
            vision_model=SimpleNamespace(
                encoder=SimpleNamespace(
                    layers=[SimpleNamespace(layer_norm2=target)]
                )
            )
        )
    )
    gc = GradCAM(nested)
    assert gc.target_layer is target
    gc.cleanup()
    assert len(gc._hooks) == 0
    assert gc.activations is None and gc.gradients is None


class _FakeC2PAContext:
    def __init__(self, _settings):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _FakeC2PAReader:
    store = {}

    def __init__(self, _mime, _stream, context=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def json(self):
        return json.dumps(self.store)

    def resource_to_stream(self, _identifier):
        return io.BytesIO(b"thumbnail")


def fake_c2pa(store):
    class Settings:
        @classmethod
        def from_dict(cls, value):
            return value
    Reader = type("Reader", (_FakeC2PAReader,), {"store": store})
    return SimpleNamespace(Settings=Settings, Context=_FakeC2PAContext, Reader=Reader, __version__="mock-sdk")


def test_c2pa_absent_valid_invalid_and_untrusted_are_separate():
    from backend.app.services.provenance.c2pa_verifier import verify_content_credentials

    payload = image_bytes()
    absent = verify_content_credentials(payload, fake_c2pa({"active_manifest": None, "manifests": {}}))
    assert absent.status == "completed" and absent.outcome == "no_results"
    assert absent.verification_outcome == "no_manifest" and absent.manifest_present is False

    store = {
        "active_manifest": "active",
        "manifests": {"active": {
            "assertions": [{"label": "c2pa.actions"}, {"label": "c2pa.ingredient"}],
            "ingredients": [{"title": "source"}],
            "signature_info": {"common_name": "C:\\Users\\private\\signer api_key=secret"},
            "thumbnail": {"identifier": "thumb"},
        }},
        "validation_results": {"activeManifest": {"success": [
            {"code": "claimSignature.validated"}, {"code": "assertion.dataHash.match"},
        ], "failure": []}},
        "validation_state": "Valid",
    }
    valid = verify_content_credentials(payload, fake_c2pa(store))
    assert valid.status == "completed" and valid.verification_outcome == "verified"
    assert valid.signature_status == "valid" and valid.content_binding_status == "valid"
    assert valid.signer_trust_status == "unverified" and not valid.trust_check_configured
    assert valid.ingredient_count == 1 and valid.referenced_resources_available == 1
    assert valid.signer_identity is not None
    assert "C:\\Users\\private" not in valid.signer_identity["common_name"]
    assert "secret" not in valid.signer_identity["common_name"]

    trusted = json.loads(json.dumps(store))
    trusted["validation_state"] = "Trusted"
    from backend.app.config import settings
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        anchors = Path(directory) / "anchors.pem"
        anchors.write_text("test trust anchor", encoding="utf-8")
        previous = settings.C2PA_TRUST_ANCHORS_PATH
        settings.C2PA_TRUST_ANCHORS_PATH = str(anchors)
        try:
            trust_result = verify_content_credentials(payload, fake_c2pa(trusted))
        finally:
            settings.C2PA_TRUST_ANCHORS_PATH = previous
    assert trust_result.trust_check_configured and trust_result.signer_trust_status == "trusted"

    broken = json.loads(json.dumps(store))
    broken["validation_results"]["activeManifest"]["success"] = []
    broken["validation_results"]["activeManifest"]["failure"] = [
        {"code": "claimSignature.mismatch"}, {"code": "assertion.dataHash.mismatch"},
        {"code": "signingCredential.untrusted"},
    ]
    invalid = verify_content_credentials(payload, fake_c2pa(broken))
    assert invalid.status == "completed" and invalid.verification_outcome == "verification_failed"
    assert invalid.signature_status == "invalid" and invalid.content_binding_status == "mismatch"
    assert invalid.signer_trust_status == "unverified"


def test_c2pa_untrusted_signer_requires_enabled_trust_check(monkeypatch, tmp_path):
    from backend.app.config import settings
    from backend.app.services.provenance.c2pa_verifier import verify_content_credentials

    anchors = tmp_path / "anchors.pem"
    anchors.write_text("test trust anchor", encoding="utf-8")
    monkeypatch.setattr(settings, "C2PA_TRUST_ANCHORS_PATH", str(anchors))
    store = {
        "active_manifest": "active",
        "manifests": {"active": {"assertions": [], "signature_info": {"common_name": "Signer"}}},
        "validation_results": {"activeManifest": {
            "success": [{"code": "claimSignature.validated"}, {"code": "assertion.dataHash.match"}],
            "failure": [{"code": "signingCredential.untrusted"}],
        }},
        "validation_state": "Invalid",
    }
    result = verify_content_credentials(image_bytes(), fake_c2pa(store))
    assert result.trust_check_configured and result.signer_trust_status == "untrusted"


def test_c2pa_malformed_store_and_missing_dependency_are_safe(monkeypatch):
    from backend.app.services.provenance import c2pa_verifier

    malformed = c2pa_verifier.verify_content_credentials(image_bytes(), fake_c2pa({"active_manifest": "bad", "manifests": []}))
    assert malformed.status == "completed" and malformed.verification_outcome == "verification_failed"
    monkeypatch.setattr(c2pa_verifier.importlib, "import_module", lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("private path")))
    missing = c2pa_verifier.verify_content_credentials(image_bytes())
    assert missing.status == "unavailable"
    assert "private path" not in json.dumps(missing.to_dict())


def test_c2pa_real_runtime_verification():
    """Verify real C2PA SDK against genuine signed fixture, trust anchors, tampering, and fusion exclusion."""
    from pathlib import Path
    import pytest
    from backend.app.config import settings
    from backend.app.services.provenance.c2pa_verifier import verify_content_credentials
    from backend.app.services.evidence.schema import build_evidence_from_provenance, EvidenceCollection
    from backend.app.services.fusion.engine import fuse_evidence
    from backend.app.services.provenance.service import search_provenance

    try:
        import c2pa
    except ImportError:
        pytest.skip("c2pa-python is not installed")

    fixture_path = Path(__file__).parent / "fixtures" / "c2pa" / "C.jpg"
    if not fixture_path.is_file():
        pytest.skip("C2PA signed fixture C.jpg is not present on disk")

    signed_bytes = fixture_path.read_bytes()

    # 1. Real cryptographic verification without trust anchors
    old_anchors = settings.C2PA_TRUST_ANCHORS_PATH
    settings.C2PA_TRUST_ANCHORS_PATH = ""
    try:
        res = verify_content_credentials(signed_bytes)
        assert res.status == "completed"
        assert res.outcome == "success"
        assert res.verification_outcome == "verified"
        assert res.manifest_present is True
        assert res.manifest_readable is True
        assert res.signature_status == "valid"
        assert res.content_binding_status == "valid"
        assert res.signer_trust_status == "unverified"
        assert res.trust_check_configured is False
        assert res.signer_identity is not None
        assert res.signer_identity.get("common_name") == "C2PA Signer"
        assert res.signer_identity.get("issuer") == "C2PA Test Signing Cert"
        assert res.referenced_resources_status == "available"
        assert res.referenced_resources_available is not None and res.referenced_resources_available >= 1

        # 2. Tampered payload (data hash mismatch)
        tampered_bytes = bytearray(signed_bytes)
        tampered_bytes[50000] ^= 0xFF
        res_tampered = verify_content_credentials(bytes(tampered_bytes))
        assert res_tampered.status == "completed"
        assert res_tampered.verification_outcome == "verification_failed"
        assert res_tampered.signature_status == "valid"
        assert res_tampered.content_binding_status == "mismatch"
        assert "assertion.dataHash.mismatch" in res_tampered.validation_codes

        # 3. Trust anchor verification: production trust list (untrusted test cert)
        prod_tl = Path(__file__).parent / "fixtures" / "c2pa" / "C2PA-TRUST-LIST.pem"
        if prod_tl.is_file():
            settings.C2PA_TRUST_ANCHORS_PATH = str(prod_tl)
            res_prod = verify_content_credentials(signed_bytes)
            assert res_prod.trust_check_configured is True
            assert res_prod.signer_trust_status == "untrusted"
            assert "signingCredential.untrusted" in res_prod.validation_codes

        # 4. Trust anchor verification: test cert root bundle (trusted)
        test_bundle = Path(__file__).parent / "fixtures" / "c2pa" / "test_cert_root_bundle.pem"
        if test_bundle.is_file():
            settings.C2PA_TRUST_ANCHORS_PATH = str(test_bundle)
            res_test = verify_content_credentials(signed_bytes)
            assert res_test.trust_check_configured is True
            assert res_test.signer_trust_status == "trusted"
            assert "signingCredential.trusted" in res_test.validation_codes
    finally:
        settings.C2PA_TRUST_ANCHORS_PATH = old_anchors

    # 5. Negative check: real unsigned image
    unsigned_path = Path(__file__).parent / "fixtures" / "test_photo.jpg"
    assert unsigned_path.is_file()
    unsigned_bytes = unsigned_path.read_bytes()
    res_unsigned = verify_content_credentials(unsigned_bytes)
    assert res_unsigned.status == "completed"
    assert res_unsigned.outcome == "no_results"
    assert res_unsigned.verification_outcome == "no_manifest"
    assert res_unsigned.manifest_present is False
    assert res_unsigned.manifest_readable is None

    # 6. Negative check: corrupted and empty inputs
    res_corrupt = verify_content_credentials(b"invalid raw bytes not an image")
    assert res_corrupt.status == "failed"
    assert res_corrupt.error_category in {"C2PA_VERIFICATION_ERROR", "INVALID_IMAGE_BYTES"}

    res_empty = verify_content_credentials(b"")
    assert res_empty.status == "failed"
    assert res_empty.error_category == "INVALID_IMAGE_BYTES"

    # 7. Evidence builder and fusion non-voting contract
    prov_res = search_provenance(signed_bytes)
    evidence_items = build_evidence_from_provenance(prov_res)
    assert any(e.source == "c2pa_content_credentials" and e.direction == "informational" for e in evidence_items)
    fusion = fuse_evidence(EvidenceCollection(items=evidence_items))
    assert fusion.evidence_summary["synthetic_signals"] == 0
    assert fusion.evidence_summary["authentic_signals"] == 0
    assert fusion.label == "inconclusive"


def test_provenance_original_hash_metadata_and_partial_status(monkeypatch):
    from backend.app.services.provenance import service

    original = image_bytes("JPEG")
    monkeypatch.setattr(service, "verify_content_credentials", lambda _data: SimpleNamespace(
        status="completed", outcome="no_results", verification_outcome="no_manifest",
        manifest_present=False,
        warnings=[], to_dict=lambda: {"status": "completed", "verification_outcome": "no_manifest", "manifest_present": False},
    ))
    monkeypatch.setattr(service, "_request_web_detection", lambda _data: ("failed", {}, "PROVIDER_TIMEOUT", ["Web Detection request timed out."]))
    result = service.search_provenance(original)
    data = result.to_dict()
    assert result.status == "partial" and data["web_detection"]["status"] == "failed"
    assert data["image_integrity"] == {"algorithm": "SHA-256", "value": hashlib.sha256(original).hexdigest()}
    assert data["metadata_status"] == "no_results"
    assert "api_key" not in json.dumps(data).lower()


def test_web_detection_parsing_empty_and_unsafe_urls():
    from backend.app.services.provenance.service import _parse_web_detection

    status, values, _, _ = _parse_web_detection({"responses": [{"webDetection": {
        "pagesWithMatchingImages": [
            {"url": "https://publisher.example/story?token=private&section=public", "pageTitle": "Report C:\\Users\\private\\file.jpg api_key=secret"},
            {"url": "http://127.0.0.1/private", "pageTitle": "unsafe"},
            {"url": "javascript:alert(1)"},
        ],
        "visuallySimilarImages": [{"url": "https://images.example/photo.jpg"}],
    }}]})
    assert status == "success" and len(values["matching_pages"]) == 1
    assert values["matching_pages"][0].url.endswith("?section=public")
    assert "C:\\Users\\private" not in values["matching_pages"][0].page_title
    assert "secret" not in values["matching_pages"][0].page_title
    assert len(values["similar_images"]) == 1
    assert _parse_web_detection({"responses": [{"webDetection": {}}]})[0] == "no_results"
    assert _parse_web_detection({"responses": "malformed"})[0] == "failed"


@pytest.mark.parametrize(
    ("url", "expected_safe_query", "secret_values"),
    [
        ("https://source.example/p?key=key-secret&section=public", "section=public", ("key-secret",)),
        ("https://source.example/p?signature=sig-secret", "", ("sig-secret",)),
        ("https://source.example/p?API_KEY=api-secret", "", ("api-secret",)),
        ("https://source.example/p?%61ccess_token=access-secret&%53IG=sig-secret", "", ("access-secret", "sig-secret")),
        ("https://source.example/p?key=first-secret&KEY=second-secret&x=1", "x=1", ("first-secret", "second-secret")),
        ("https://source.example/p?section=public&lang=en", "section=public&lang=en", ()),
        ("https://source.example/p", "", ()),
        ("https://source.example/p?credential=cred-secret#token=fragment-secret", "", ("cred-secret", "fragment-secret")),
    ],
)
def test_external_url_removes_secret_parameters_and_fragment(url, expected_safe_query, secret_values):
    from backend.app.services.provenance.service import _safe_external_url

    safe = _safe_external_url(url)
    assert safe.startswith("https://source.example/p")
    assert safe.partition("?")[2] == expected_safe_query
    assert "#" not in safe
    assert all(secret not in safe for secret in secret_values)


@pytest.mark.parametrize(
    "url",
    [
        "http://source.example/p?key=http-secret",
        "https://[malformed/p?key=bad-secret",
        "https://source.example:invalid/p?key=bad-secret",
    ],
)
def test_external_url_rejects_http_and_malformed_urls(url):
    from backend.app.services.provenance.service import _safe_external_url

    assert _safe_external_url(url) == ""


@pytest.mark.parametrize(
    "parameter",
    [
        "key", "api_key", "access_key", "secret", "client_secret", "token",
        "access_token", "refresh_token", "signature", "sig", "authorization",
        "credential", "password", "auth", "api-key",
    ],
)
def test_external_url_removes_every_supported_secret_parameter(parameter):
    from backend.app.services.provenance.service import _safe_external_url

    secret_value = f"value-for-{parameter}"
    safe = _safe_external_url(f"https://source.example/p?{parameter.upper()}={secret_value}&topic=public")
    assert safe == "https://source.example/p?topic=public"
    assert secret_value not in safe


def test_sanitized_provider_urls_are_safe_in_serialized_parse_results():
    from backend.app.services.provenance.service import _parse_web_detection

    status, values, _, _ = _parse_web_detection({"responses": [{"webDetection": {
        "pagesWithMatchingImages": [{
            "url": "https://source.example/report?key=private-key&signature=private-signature&section=public",
            "pageTitle": "Report",
        }],
    }}]})
    serialized = json.dumps([item.to_dict() for item in values["matching_pages"]])
    assert status == "success"
    assert "private-key" not in serialized and "private-signature" not in serialized
    assert "section=public" in serialized


def test_web_detection_http_timeout_is_bounded_and_safe(monkeypatch):
    import httpx
    from backend.app.services.provenance import service

    monkeypatch.setattr(service.settings, "GOOGLE_API_KEY", "provider-secret")
    monkeypatch.setattr(service.settings, "PROVENANCE_RETRY_ATTEMPTS", 0)
    monkeypatch.setattr(service.settings, "PROVENANCE_API_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(service.settings, "PROVENANCE_MAX_TOTAL_SECONDS", 1)

    class TimeoutClient:
        def __init__(self, timeout):
            assert timeout <= 1
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def post(self, *_args, **_kwargs): raise httpx.TimeoutException("private timeout details")

    monkeypatch.setattr(service.httpx, "Client", TimeoutClient)
    status, _, category, warnings = service._request_web_detection(image_bytes())
    assert status == "failed" and category == "PROVIDER_TIMEOUT"
    assert "private timeout details" not in json.dumps(warnings)
    assert "provider-secret" not in json.dumps(warnings)


def test_web_success_with_unavailable_c2pa_stays_partial(monkeypatch):
    from backend.app.services.provenance import service

    payload = image_bytes()
    monkeypatch.setattr(service, "verify_content_credentials", lambda _data: SimpleNamespace(
        status="unavailable", outcome="unavailable", verification_outcome="unavailable",
        warnings=[], to_dict=lambda: {"status": "unavailable", "verification_outcome": "unavailable"},
    ))
    monkeypatch.setattr(service, "_request_web_detection", lambda _data: ("success", {"responses": [{"webDetection": {
        "pagesWithMatchingImages": [{"url": "https://source.example/report", "pageTitle": "Report"}],
    }}]}, None, []))
    result = service.search_provenance(payload)
    assert result.status == "partial"
    assert result.web_detection_status == "success"
    assert result.matching_pages[0].url == "https://source.example/report"
    assert result.image_sha256 == hashlib.sha256(payload).hexdigest()


def test_metadata_extraction_failure_does_not_hide_provider_result(monkeypatch):
    from backend.app.services.provenance import service

    monkeypatch.setattr(service, "verify_content_credentials", lambda _data: SimpleNamespace(
        status="completed", outcome="no_results", verification_outcome="no_manifest",
        manifest_present=False,
        warnings=[], to_dict=lambda: {"status": "completed", "verification_outcome": "no_manifest"},
    ))
    monkeypatch.setattr(service, "_extract_provenance_metadata", lambda _data: ("failed", {"error_category": "METADATA_EXTRACTION_ERROR"}))
    monkeypatch.setattr(service, "_request_web_detection", lambda _data: ("no_results", {}, None, []))
    result = service.search_provenance(image_bytes())
    assert result.status == "no_results" and result.metadata_status == "failed"
    assert any("EXIF metadata extraction" in warning for warning in result.warnings)


def test_provenance_missing_credentials_is_unavailable_for_web_check(monkeypatch):
    from backend.app.services.provenance import service

    monkeypatch.setattr(service.settings, "GOOGLE_API_KEY", "")
    status, _, category, warnings = service._request_web_detection(image_bytes())
    assert status == "unavailable" and category is None
    assert warnings and "credentials" in warnings[0]


def test_provenance_endpoint_reuses_validation_and_response_service(monkeypatch):
    from fastapi import HTTPException, UploadFile, Request
    from backend.app.api.v1.endpoints import stubs
    from backend.app.core.auth import Principal
    from backend.app.services.provenance import service

    monkeypatch.setattr(service, "search_provenance", lambda _data: SimpleNamespace(to_dict=lambda: {"status": "unavailable"}))
    async def allow_rate_limit(*_args, **_kwargs):
        return None
    monkeypatch.setattr(stubs, "check_rate_limit", allow_rate_limit)
    request = Request({
        "type": "http", "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": "/api/v1/provenance", "raw_path": b"/api/v1/provenance",
        "query_string": b"", "headers": [], "client": ("test", 1234), "server": ("test", 80),
    })

    async def invoke(data, name="image.jpg"):
        return await stubs.provenance_endpoint(
            http_request=request,
            image=UploadFile(filename=name, file=io.BytesIO(data)),
            principal=Principal(subject="e6e7-test"),
        )

    import asyncio
    valid = asyncio.run(invoke(image_bytes()))
    assert valid["status"] == "unavailable"
    assert stubs.ProvenanceResponse.model_validate(valid).status == "unavailable"
    with pytest.raises(HTTPException) as error:
        asyncio.run(invoke(b"not-an-image"))
    assert error.value.status_code == 422


def test_provenance_evidence_is_informational_and_old_reports_still_serialize():
    from backend.app.services.evidence.schema import EvidenceCollection, build_evidence_from_provenance
    from backend.app.services.fusion.engine import fuse_evidence
    from backend.app.services.provenance.service import ProvenanceResult
    from backend.app.services.reporting.builder import build_report

    result = ProvenanceResult(status="partial", content_credentials={
        "status": "completed", "verification_outcome": "verification_failed",
        "signature_status": "invalid", "content_binding_status": "mismatch",
    }, web_detection_status="unavailable")
    items = build_evidence_from_provenance(result)
    assert len(items) == 2
    assert all(item.direction == "informational" for item in items)
    assert items[0].data["verification_outcome"] == "verification_failed"
    collection = EvidenceCollection(analysis_id="e6e7")
    for item in items:
        collection.add(item)
    assessment = fuse_evidence(collection)
    assert assessment.label == "inconclusive"
    assert assessment.evidence_summary["authentic_signals"] == 0
    report = build_report(analysis_id="old-report", status="completed", provenance={"status": "no_results"})
    json.dumps(report)


def test_e6_e7_integrated_report_keeps_partial_results_and_stage_outcomes(monkeypatch):
    from pathlib import Path
    from backend.app.services.evidence.schema import (
        EvidenceCollection, build_evidence_from_forensics, build_evidence_from_provenance,
        build_evidence_from_vision, build_evidence_from_xai,
    )
    from backend.app.services.forensics.analyzer import analyze_forensics
    from backend.app.services.fusion.engine import fuse_evidence
    from backend.app.services.provenance import service
    from backend.app.services.provenance.c2pa_verifier import C2PAResult
    from backend.app.services.reporting.builder import build_report
    from backend.app.services.vision.detector import ObjectDetector
    from backend.app.services.xai.gradcam import generate_heatmap

    original = (Path(__file__).parent / "fixtures" / "test_photo.jpg").read_bytes()
    forensic = analyze_forensics(original, generate_artifacts=False)
    with Image.open(io.BytesIO(original)) as image:
        detector = ObjectDetector()
        detector._loaded = True
        detector._model = SimpleNamespace(predict=lambda **_: [SimpleNamespace(boxes=[], names={})])
        vision = detector.detect(image)
        attribution = generate_heatmap(torch.nn.Linear(2, 2), _ToyProcessor(), image, "e6-e7-test", torch.device("cpu"))

    monkeypatch.setattr(service, "verify_content_credentials", lambda _data: C2PAResult(
        status="unavailable", outcome="unavailable", verification_outcome="unavailable",
        warnings=["C2PA verifier unavailable."],
    ))
    monkeypatch.setattr(service, "_request_web_detection", lambda _data: (
        "failed", {}, "PROVIDER_TIMEOUT", ["Web Detection request timed out."]
    ))
    provenance = service.search_provenance(original)

    collection = EvidenceCollection(analysis_id="e6-e7-test")
    for item in (
        build_evidence_from_forensics(forensic)
        + build_evidence_from_vision(vision)
        + build_evidence_from_xai(attribution)
        + build_evidence_from_provenance(provenance)
    ):
        collection.add(item)
    assessment = fuse_evidence(collection)
    report = build_report(
        analysis_id="e6-e7-test", status="completed",
        forensics=forensic.to_dict(), objects=vision.to_dict(), xai=attribution.to_dict(),
        provenance=provenance.to_dict(), evidence=collection.to_dict(),
        assessment=assessment.to_dict(),
        stages={
            "forensics": {"status": forensic.status, "outcome": forensic.outcome},
            "vision": {"status": vision.status, "outcome": vision.outcome},
            "xai": {"status": attribution.status, "outcome": attribution.outcome},
                "provenance": {"status": provenance.status, "outcome": provenance.outcome},
        },
    )
    assert forensic.status == "completed"
    assert vision.status == "completed" and vision.outcome == "no_results"
    assert attribution.status == "unavailable"
    assert provenance.status == "failed"
    assert report["status"] == "partial"
    assert report["assessment"]["label"] == "inconclusive"
    assert all(item.direction == "informational" for item in collection.items)
    assert report["provenance"]["image_integrity"]["value"] == hashlib.sha256(original).hexdigest()
    json.dumps(report)

