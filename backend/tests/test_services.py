"""
TruthLens — Unit tests for security, hashing, and service modules.
"""

import io
import tempfile
from pathlib import Path

import pytest

from backend.app.core.security import (
    compute_sha256,
    detect_mime_from_bytes,
    sanitize_filename,
    validate_magic_bytes,
    generate_storage_filename,
)
from backend.app.services.metadata.service import extract_metadata
from backend.app.services.quality.service import analyze_quality


# ── Filename sanitization ────────────────────────────────────────

class TestSanitizeFilename:
    def test_normal_filename(self):
        assert sanitize_filename("photo.jpg") == "photo.jpg"

    def test_path_traversal(self):
        result = sanitize_filename("../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_special_characters(self):
        result = sanitize_filename("my photo (1) [final].jpg")
        assert "(" not in result  # Non-alphanumeric stripped
        assert ")" not in result

    def test_windows_path(self):
        result = sanitize_filename("C:\\Users\\test\\photo.jpg")
        assert "\\" not in result
        assert "photo.jpg" in result

    def test_empty_becomes_unnamed(self):
        assert sanitize_filename("") == "unnamed"

    def test_long_filename_truncated(self):
        long_name = "a" * 300 + ".jpg"
        result = sanitize_filename(long_name)
        assert len(result) <= 200


# ── Magic-byte validation ────────────────────────────────────────

class TestMagicBytes:
    def test_jpeg_magic(self):
        assert validate_magic_bytes(b"\xff\xd8\xff\xe0", "image/jpeg")

    def test_png_magic(self):
        assert validate_magic_bytes(b"\x89PNG\r\n\x1a\n", "image/png")

    def test_wrong_magic(self):
        assert not validate_magic_bytes(b"\x89PNG\r\n\x1a\n", "image/jpeg")


# ── SHA-256 hashing ──────────────────────────────────────────────

class TestHashing:
    def test_sha256_consistency(self):
        data = b"test data for hashing"
        hash1 = compute_sha256(data)
        hash2 = compute_sha256(data)
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_sha256_different_data(self):
        assert compute_sha256(b"data1") != compute_sha256(b"data2")


# ── MIME detection ───────────────────────────────────────────────

class TestMimeDetection:
    def test_detect_jpeg(self):
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="JPEG")
        assert detect_mime_from_bytes(buf.getvalue()) == "image/jpeg"

    def test_detect_png(self):
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="PNG")
        assert detect_mime_from_bytes(buf.getvalue()) == "image/png"

    def test_detect_text_returns_none(self):
        assert detect_mime_from_bytes(b"Hello world") is None


# ── Storage filename generation ──────────────────────────────────

class TestStorageFilename:
    def test_preserves_jpg_extension(self):
        result = generate_storage_filename("photo.jpg")
        assert result.endswith(".jpg")

    def test_preserves_png_extension(self):
        result = generate_storage_filename("photo.png")
        assert result.endswith(".png")

    def test_unknown_extension_becomes_bin(self):
        result = generate_storage_filename("file.exe")
        assert result.endswith(".bin")

    def test_unique_filenames(self):
        f1 = generate_storage_filename("photo.jpg")
        f2 = generate_storage_filename("photo.jpg")
        assert f1 != f2  # UUID-based, always unique


# ── Metadata extraction ─────────────────────────────────────────

class TestMetadataExtraction:
    def test_basic_metadata(self):
        from PIL import Image
        buf = io.BytesIO()
        img = Image.new("RGB", (640, 480))
        img.save(buf, format="JPEG")
        result = extract_metadata(buf.getvalue())
        assert result.image_width == 640
        assert result.image_height == 480

    def test_no_exif_anomaly(self):
        from PIL import Image
        buf = io.BytesIO()
        img = Image.new("RGB", (100, 100))
        img.save(buf, format="PNG")
        result = extract_metadata(buf.getvalue())
        assert result.has_exif is False
        assert len(result.anomalies) > 0  # Should flag missing EXIF


# ── Quality analysis ─────────────────────────────────────────────

class TestQualityAnalysis:
    def test_basic_quality(self):
        from PIL import Image
        buf = io.BytesIO()
        img = Image.new("RGB", (200, 150), color=(100, 150, 200))
        img.save(buf, format="JPEG")
        result = analyze_quality(buf.getvalue())
        assert result.width == 200
        assert result.height == 150
        assert result.megapixels == 0.03
        assert result.brightness > 0
        assert result.contrast >= 0
        assert result.blur_score >= 0
        assert result.format == "JPEG"

    def test_resolution_category(self):
        from PIL import Image
        # Low resolution
        buf = io.BytesIO()
        Image.new("RGB", (100, 100)).save(buf, format="JPEG")
        result = analyze_quality(buf.getvalue())
        assert result.resolution_category == "low"
