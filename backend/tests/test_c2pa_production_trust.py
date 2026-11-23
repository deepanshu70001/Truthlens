"""Tests for C2PA production trust outcomes across all 8 defined lifecycle states:
1. NO_MANIFEST
2. MANIFEST_PRESENT_VALID
3. SIGNATURE_VALID_TRUSTED
4. SIGNATURE_VALID_UNTRUSTED
5. SIGNATURE_INVALID
6. MANIFEST_INVALID
7. UNSUPPORTED_FORMAT
8. VALIDATION_ERROR
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from backend.app.config import settings
from backend.app.services.provenance.c2pa_verifier import (
    C2PAResult,
    verify_content_credentials,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "c2pa"
C_JPG = FIXTURE_DIR / "C.jpg"
TEST_ROOT_BUNDLE = FIXTURE_DIR / "test_cert_root_bundle.pem"
PROD_TRUST_LIST = Path(__file__).resolve().parent.parent.parent / "credentials" / "C2PA-TRUST-LIST.pem"


def _fake_sdk_with_store(store: dict):
    class FakeSettings:
        @classmethod
        def from_dict(cls, data):
            return data

    class FakeContext:
        def __init__(self, s):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    class FakeReader:
        def __init__(self, mime, stream, context=None):
            self.mime = mime
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def json(self):
            return json.dumps(store)
        def resource_to_stream(self, identifier, out=None):
            if out is not None:
                out.write(b"thumb")
                return 5
            return io.BytesIO(b"thumb")

    return SimpleNamespace(
        Settings=FakeSettings,
        Context=FakeContext,
        Reader=FakeReader,
        __version__="mock-c2pa-0.39.0",
    )


def test_outcome_1_no_manifest():
    """Unsigned JPEG image yields NO_MANIFEST."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(100, 150, 200)).save(buf, format="JPEG")
    unsigned_bytes = buf.getvalue()

    result = verify_content_credentials(unsigned_bytes)
    assert result.status == "completed"
    assert result.outcome == "no_results"
    assert result.verification_outcome == "no_manifest"
    assert result.trust_outcome == "NO_MANIFEST"
    assert result.manifest_present is False
    assert result.to_dict()["trust_outcome"] == "NO_MANIFEST"


def test_outcome_2_manifest_present_valid_without_trust_anchors(monkeypatch):
    """Signed image with valid signature and binding, without trust anchors, yields MANIFEST_PRESENT_VALID."""
    if not C_JPG.is_file():
        pytest.skip("Fixture C.jpg not found")

    monkeypatch.setattr(settings, "C2PA_TRUST_ANCHORS_PATH", "")
    signed_bytes = C_JPG.read_bytes()

    result = verify_content_credentials(signed_bytes)
    assert result.status == "completed"
    assert result.verification_outcome == "verified"
    assert result.signature_status == "valid"
    assert result.content_binding_status == "valid"
    assert result.trust_check_configured is False
    assert result.signer_trust_status == "unverified"
    assert result.trust_outcome == "MANIFEST_PRESENT_VALID"
    assert result.to_dict()["trust_outcome"] == "MANIFEST_PRESENT_VALID"


def test_outcome_3_signature_valid_trusted(monkeypatch):
    """Signed image with test cert root bundle configured yields SIGNATURE_VALID_TRUSTED."""
    if not C_JPG.is_file() or not TEST_ROOT_BUNDLE.is_file():
        pytest.skip("Fixture C.jpg or test_cert_root_bundle.pem not found")

    monkeypatch.setattr(settings, "C2PA_TRUST_ANCHORS_PATH", str(TEST_ROOT_BUNDLE))
    signed_bytes = C_JPG.read_bytes()

    result = verify_content_credentials(signed_bytes)
    assert result.status == "completed"
    assert result.verification_outcome == "verified"
    assert result.signature_status == "valid"
    assert result.content_binding_status == "valid"
    assert result.trust_check_configured is True
    assert result.signer_trust_status == "trusted"
    assert result.trust_outcome == "SIGNATURE_VALID_TRUSTED"
    assert result.to_dict()["trust_outcome"] == "SIGNATURE_VALID_TRUSTED"


def test_outcome_4_signature_valid_untrusted(monkeypatch):
    """Signed image against production trust list yields SIGNATURE_VALID_UNTRUSTED."""
    if not C_JPG.is_file() or not PROD_TRUST_LIST.is_file():
        pytest.skip("Fixture C.jpg or credentials/C2PA-TRUST-LIST.pem not found")

    monkeypatch.setattr(settings, "C2PA_TRUST_ANCHORS_PATH", str(PROD_TRUST_LIST))
    signed_bytes = C_JPG.read_bytes()

    result = verify_content_credentials(signed_bytes)
    assert result.status == "completed"
    assert result.verification_outcome == "verified"
    assert result.signature_status == "valid"
    assert result.content_binding_status == "valid"
    assert result.trust_check_configured is True
    assert result.signer_trust_status == "untrusted"
    assert result.trust_outcome == "SIGNATURE_VALID_UNTRUSTED"
    assert result.to_dict()["trust_outcome"] == "SIGNATURE_VALID_UNTRUSTED"


def test_outcome_5_signature_invalid():
    """Manifest with invalid signature yields SIGNATURE_INVALID."""
    store = {
        "active_manifest": "m1",
        "manifests": {
            "m1": {
                "assertions": [{"label": "c2pa.actions"}],
                "signature_info": {"common_name": "Test Signer"},
            }
        },
        "validation_results": {
            "activeManifest": {
                "success": [],
                "failure": [{"code": "claimSignature.mismatch"}],
            }
        },
        "validation_state": "Invalid",
    }
    fake_sdk = _fake_sdk_with_store(store)
    buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, format="PNG")

    result = verify_content_credentials(buf.getvalue(), sdk_module=fake_sdk)
    assert result.signature_status == "invalid"
    assert result.trust_outcome == "SIGNATURE_INVALID"
    assert result.to_dict()["trust_outcome"] == "SIGNATURE_INVALID"


def test_outcome_6_manifest_invalid_content_hash_mismatch():
    """Tampered image bytes cause content binding mismatch, yielding MANIFEST_INVALID."""
    if not C_JPG.is_file():
        pytest.skip("Fixture C.jpg not found")

    signed_bytes = C_JPG.read_bytes()
    tampered_bytes = bytearray(signed_bytes)
    tampered_bytes[50000] ^= 0xFF

    result = verify_content_credentials(bytes(tampered_bytes))
    assert result.status == "completed"
    assert result.verification_outcome == "verification_failed"
    assert result.content_binding_status == "mismatch"
    assert result.trust_outcome == "MANIFEST_INVALID"
    assert result.to_dict()["trust_outcome"] == "MANIFEST_INVALID"


def test_outcome_6_manifest_invalid_malformed_store():
    """Manifest store missing active manifest yields MANIFEST_INVALID."""
    store = {"active_manifest": "missing", "manifests": {}}
    fake_sdk = _fake_sdk_with_store(store)
    buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, format="PNG")

    result = verify_content_credentials(buf.getvalue(), sdk_module=fake_sdk)
    assert result.manifest_readable is False
    assert result.trust_outcome == "MANIFEST_INVALID"


def test_outcome_7_unsupported_format():
    """Unsupported image format (e.g. BMP) yields UNSUPPORTED_FORMAT."""
    buf = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buf, format="BMP")
    bmp_bytes = buf.getvalue()

    result = verify_content_credentials(bmp_bytes)
    assert result.status == "failed"
    assert result.trust_outcome == "UNSUPPORTED_FORMAT"
    assert result.error_category == "UNSUPPORTED_FORMAT"
    assert result.to_dict()["trust_outcome"] == "UNSUPPORTED_FORMAT"


def test_outcome_8_validation_error_empty_bytes():
    """Empty or invalid raw bytes yield VALIDATION_ERROR."""
    result = verify_content_credentials(b"")
    assert result.status == "failed"
    assert result.trust_outcome == "VALIDATION_ERROR"
    assert result.error_category == "INVALID_IMAGE_BYTES"
    assert result.to_dict()["trust_outcome"] == "VALIDATION_ERROR"


def test_outcome_8_validation_error_missing_sdk(monkeypatch):
    """Missing c2pa SDK dependency yields VALIDATION_ERROR."""
    from backend.app.services.provenance import c2pa_verifier

    monkeypatch.setattr(
        c2pa_verifier.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("c2pa missing")),
    )
    result = c2pa_verifier.verify_content_credentials(b"some-bytes")
    assert result.status == "unavailable"
    assert result.trust_outcome == "VALIDATION_ERROR"
    assert result.error_category == "VERIFIER_DEPENDENCY_MISSING"
