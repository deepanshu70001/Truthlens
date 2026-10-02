"""
TruthLens — Image preprocessing service.

Handles EXIF orientation correction, RGB conversion, thumbnail creation,
model-input resizing, and basic image-quality statistics.

NEVER modifies the original file.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ExifTags

from backend.app.core.logging import get_logger

log = get_logger("preprocessing")

# Standard model input sizes
MODEL_INPUT_SIZES: dict[str, int] = {
    "small": 224,
    "medium": 384,
    "large": 640,
}

THUMBNAIL_SIZE = (256, 256)


@dataclass
class PreprocessingResult:
    """Result of the preprocessing pipeline."""
    original_width: int
    original_height: int
    format: str
    mode: str
    has_exif: bool
    orientation_corrected: bool
    thumbnail_path: Optional[str] = None
    resized_paths: Dict[str, str] = field(default_factory=dict)
    quality_stats: Dict[str, float] = field(default_factory=dict)


def _correct_exif_orientation(img: Image.Image) -> Tuple[Image.Image, bool]:
    """Apply EXIF orientation tag and return (corrected_image, was_corrected)."""
    try:
        exif = img.getexif()
        if not exif:
            return img, False

        orientation_key = None
        for k, v in ExifTags.TAGS.items():
            if v == "Orientation":
                orientation_key = k
                break

        if orientation_key is None or orientation_key not in exif:
            return img, False

        orientation = exif[orientation_key]
        transforms = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: lambda im: im.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270),
            6: Image.ROTATE_270,
            7: lambda im: im.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90),
            8: Image.ROTATE_90,
        }

        if orientation in transforms:
            t = transforms[orientation]
            if callable(t) and not isinstance(t, int):
                img = t(img)
            else:
                img = img.transpose(t)
            return img, True

    except Exception as e:
        log.warning("EXIF orientation correction failed: %s", e)

    return img, False


def _ensure_rgb(img: Image.Image) -> Image.Image:
    """Convert image to RGB mode if necessary (handles RGBA, L, P, etc.)."""
    if img.mode == "RGB":
        return img
    if img.mode == "RGBA":
        # Composite onto white background
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        return background
    return img.convert("RGB")


def _compute_basic_stats(img_array: np.ndarray) -> Dict[str, float]:
    """Compute basic image quality statistics from a NumPy array (BGR or RGB)."""
    stats: Dict[str, float] = {}

    # Convert to grayscale for some metrics
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Brightness (mean intensity)
    stats["brightness"] = round(float(np.mean(gray)), 2)

    # Contrast (std dev of intensity)
    stats["contrast"] = round(float(np.std(gray)), 2)

    # Blur estimate (Laplacian variance — lower = blurrier)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    stats["blur_score"] = round(float(laplacian.var()), 2)

    # Noise estimate (std dev of Laplacian)
    stats["noise_estimate"] = round(float(np.std(laplacian)), 2)

    return stats


def preprocess_image(
    image_data: bytes,
    analysis_id: str,
    thumbnails_dir: Path,
    processed_dir: Path,
) -> PreprocessingResult:
    """
    Run the full preprocessing pipeline on raw image bytes.

    Creates:
    - EXIF-corrected + RGB-normalized working copy
    - Thumbnail
    - Resized versions for model input
    - Basic quality statistics

    Never modifies the original file.
    """
    img = Image.open(io.BytesIO(image_data))

    original_width, original_height = img.size
    fmt = img.format or "UNKNOWN"
    mode = img.mode
    has_exif = bool(img.getexif())

    # Step 1: EXIF orientation correction
    img, orientation_corrected = _correct_exif_orientation(img)

    # Step 2: Ensure RGB
    img = _ensure_rgb(img)

    # Step 3: Create thumbnail
    thumb = img.copy()
    thumb.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
    thumb_filename = f"{analysis_id}_thumb.jpg"
    thumb_path = thumbnails_dir / thumb_filename
    thumb.save(thumb_path, "JPEG", quality=85)

    # Step 4: Create resized versions for model inputs
    resized_paths: Dict[str, str] = {}
    for label, size in MODEL_INPUT_SIZES.items():
        resized = img.copy()
        resized.thumbnail((size, size), Image.LANCZOS)
        resized_filename = f"{analysis_id}_{label}.jpg"
        resized_path = processed_dir / resized_filename
        resized.save(resized_path, "JPEG", quality=95)
        resized_paths[label] = str(resized_path)

    # Step 5: Compute quality statistics
    img_array = np.array(img)
    quality_stats = _compute_basic_stats(img_array)

    result = PreprocessingResult(
        original_width=original_width,
        original_height=original_height,
        format=fmt,
        mode=mode,
        has_exif=has_exif,
        orientation_corrected=orientation_corrected,
        thumbnail_path=str(thumb_path),
        resized_paths=resized_paths,
        quality_stats=quality_stats,
    )

    log.info(
        "Preprocessing complete for analysis=%s dims=%dx%d fmt=%s",
        analysis_id, original_width, original_height, fmt,
    )

    return result
