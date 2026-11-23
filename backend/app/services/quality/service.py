"""
TruthLens — Image quality analysis service.

Extracts useful signals: resolution, blur estimate, brightness, contrast,
compression info. These are informational signals — never call them proof.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from backend.app.core.logging import get_logger

log = get_logger("quality")


@dataclass
class QualityResult:
    """Image quality analysis signals."""
    width: int
    height: int
    megapixels: float
    aspect_ratio: str
    file_size_bytes: int
    bits_per_pixel: float

    # Quality metrics
    brightness: float        # Mean intensity (0–255)
    contrast: float          # Std dev of intensity
    blur_score: float        # Laplacian variance (lower → blurrier)
    noise_estimate: float    # Estimated noise level
    sharpness: float         # Edge density (higher → sharper)
    saturation_mean: float   # Mean saturation in HSV
    dynamic_range: float     # Max–min intensity (0–255)

    # Compression info
    estimated_quality: Optional[int] = None  # Estimated JPEG quality (if JPEG)
    format: str = "UNKNOWN"

    @property
    def resolution_category(self) -> str:
        mp = self.megapixels
        if mp >= 8:
            return "high"
        elif mp >= 2:
            return "medium"
        else:
            return "low"


def _estimate_jpeg_quality(image_data: bytes) -> Optional[int]:
    """
    Rough estimate of JPEG quality by file-size / pixel ratio.
    This is an approximation, not exact.
    """
    try:
        img = Image.open(io.BytesIO(image_data))
        if img.format != "JPEG":
            return None
        w, h = img.size
        pixels = w * h
        if pixels == 0:
            return None
        bpp = (len(image_data) * 8) / pixels
        # Very rough mapping: bpp → quality
        if bpp > 4:
            return 95
        elif bpp > 2:
            return 85
        elif bpp > 1:
            return 75
        elif bpp > 0.5:
            return 60
        else:
            return 40
    except Exception:
        return None


def analyze_quality(image_data: bytes) -> QualityResult:
    """
    Analyze image quality from raw bytes.

    Returns informational quality signals.
    These are NOT proof of manipulation.
    """
    img = Image.open(io.BytesIO(image_data))
    width, height = img.size
    fmt = img.format or "UNKNOWN"

    # Convert to numpy (RGB)
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb)

    # Grayscale for intensity metrics
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # HSV for saturation
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    # Brightness
    brightness = float(np.mean(gray))

    # Contrast
    contrast = float(np.std(gray))

    # Blur (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())

    # Noise estimate
    noise_estimate = float(np.std(laplacian))

    # Sharpness (edge density via Canny)
    edges = cv2.Canny(gray, 50, 150)
    sharpness = float(np.count_nonzero(edges)) / (width * height) * 100

    # Saturation
    saturation_mean = float(np.mean(hsv[:, :, 1]))

    # Dynamic range
    dynamic_range = float(np.max(gray)) - float(np.min(gray))

    # Bits per pixel
    bpp = (len(image_data) * 8) / (width * height) if width * height > 0 else 0.0

    # JPEG quality estimate
    estimated_quality = _estimate_jpeg_quality(image_data)

    # Aspect ratio
    from math import gcd
    g = gcd(width, height)
    aspect_ratio = f"{width // g}:{height // g}"

    megapixels = round((width * height) / 1_000_000, 2)

    result = QualityResult(
        width=width,
        height=height,
        megapixels=megapixels,
        aspect_ratio=aspect_ratio,
        file_size_bytes=len(image_data),
        bits_per_pixel=round(bpp, 2),
        brightness=round(brightness, 2),
        contrast=round(contrast, 2),
        blur_score=round(blur_score, 2),
        noise_estimate=round(noise_estimate, 2),
        sharpness=round(sharpness, 2),
        saturation_mean=round(saturation_mean, 2),
        dynamic_range=round(dynamic_range, 2),
        estimated_quality=estimated_quality,
        format=fmt,
    )

    log.info(
        "Quality analysis: %dx%d blur=%.1f brightness=%.1f contrast=%.1f",
        width, height, blur_score, brightness, contrast,
    )

    return result
