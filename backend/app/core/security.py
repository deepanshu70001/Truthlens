"""
TruthLens — File upload security utilities.

Validates MIME types, magic bytes, file size, and sanitizes filenames.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path, PurePosixPath
from typing import Tuple

import filetype
from PIL import Image

from backend.app.config import settings

# ── Magic-byte signatures for allowed image types ────────────────
_MAGIC_SIGNATURES: dict[str, list[bytes]] = {
    "image/jpeg": [b"\xff\xd8\xff"],
    "image/png": [b"\x89PNG\r\n\x1a\n"],
    "image/gif": [b"GIF87a", b"GIF89a"],
    "image/webp": [b"RIFF"],  # Full check: RIFF....WEBP
    "image/bmp": [b"BM"],
    "image/tiff": [b"II\x2a\x00", b"MM\x00\x2a"],
}


def validate_magic_bytes(header: bytes, claimed_mime: str) -> bool:
    """Return True if the file header matches the claimed MIME type."""
    sigs = _MAGIC_SIGNATURES.get(claimed_mime, [])
    if not sigs:
        # If we don't have a signature for this MIME, use filetype lib
        kind = filetype.match(header)
        return kind is not None and kind.mime == claimed_mime
    return any(header.startswith(sig) for sig in sigs)


def detect_mime_from_bytes(header: bytes) -> str | None:
    """Detect MIME type from file header bytes using filetype lib."""
    kind = filetype.match(header)
    return kind.mime if kind else None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a user-supplied filename:
    - Strip path components (prevent path traversal)
    - Remove non-alphanumeric characters except dot, hyphen, underscore
    - Collapse multiple dots
    - Limit length
    """
    # Take only the basename (strip any directory info)
    name = PurePosixPath(filename).name
    name = Path(name).name  # Also handle Windows paths

    # Remove any path traversal attempts
    name = name.replace("..", "").replace("/", "").replace("\\", "")

    # Keep only safe characters
    name = re.sub(r"[^\w.\-]", "_", name)

    # Collapse multiple dots/underscores
    name = re.sub(r"\.{2,}", ".", name)
    name = re.sub(r"_{2,}", "_", name)

    # Limit length
    if len(name) > 200:
        ext = Path(name).suffix
        name = name[: 200 - len(ext)] + ext

    return name or "unnamed"


def generate_storage_filename(original_filename: str) -> str:
    """Generate a UUID-based storage filename preserving the original extension."""
    ext = Path(original_filename).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"):
        ext = ".bin"
    return f"{uuid.uuid4().hex}{ext}"


def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hash of binary data."""
    return hashlib.sha256(data).hexdigest()


def compute_sha256_file(filepath: Path) -> str:
    """Compute SHA-256 hash of a file on disk."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_image_file(
    data: bytes,
    filename: str,
    content_type: str | None,
) -> Tuple[str, str]:
    """
    Full validation pipeline for an uploaded image file.

    Returns (safe_filename, detected_mime).
    Raises ValueError with a user-safe message on failure.
    """
    # 1. Size check
    if len(data) > settings.max_upload_bytes:
        raise ValueError(
            f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE_MB} MB."
        )

    if len(data) < 8:
        raise ValueError("File is too small to be a valid image.")

    # 2. Detect actual MIME from bytes
    detected_mime = detect_mime_from_bytes(data[:2048])
    if detected_mime is None:
        raise ValueError("Unable to determine file type. Please upload a valid image.")

    # 3. Check detected MIME against allowed list
    if detected_mime not in settings.allowed_mime_list:
        raise ValueError(
            f"File type '{detected_mime}' is not allowed. "
            f"Accepted types: {', '.join(settings.allowed_mime_list)}"
        )

    # Content-Type is user supplied, so use magic-byte detection as the source
    # of truth and reject a contradictory declaration instead of silently
    # treating a spoofed MIME type as an image.
    claimed_mime = (content_type or "").split(";", 1)[0].strip().lower()
    claimed_mime = {"image/jpg": "image/jpeg", "image/x-png": "image/png"}.get(
        claimed_mime, claimed_mime
    )
    if claimed_mime and claimed_mime != "application/octet-stream" and claimed_mime != detected_mime:
        raise ValueError("Declared MIME type does not match the uploaded image content.")

    expected_extensions = {
        "image/jpeg": {".jpg", ".jpeg"},
        "image/png": {".png"},
        "image/webp": {".webp"},
        "image/gif": {".gif"},
        "image/bmp": {".bmp"},
        "image/tiff": {".tif", ".tiff"},
    }
    supplied_extension = Path(sanitize_filename(filename)).suffix.lower()
    if supplied_extension and supplied_extension not in expected_extensions.get(detected_mime, set()):
        raise ValueError("Filename extension does not match the uploaded image content.")

    # 4. Magic-byte validation
    if not validate_magic_bytes(data[:16], detected_mime):
        raise ValueError("File header does not match its type. Possible corruption.")

    # 5. Try opening with Pillow to verify it's a valid image
    from io import BytesIO

    try:
        img = Image.open(BytesIO(data))
    except Exception:
        raise ValueError("File could not be decoded as a valid image.")
    width, height = img.size
    if width <= 0 or height <= 0 or width * height > settings.MAX_IMAGE_PIXELS:
        raise ValueError("Image dimensions exceed the configured safety limit.")
    try:
        img.verify()
    except Exception:
        raise ValueError("File could not be decoded as a valid image.")

    # 6. Sanitize filename
    safe_name = sanitize_filename(filename)

    return safe_name, detected_mime
