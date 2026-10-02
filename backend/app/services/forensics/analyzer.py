"""
TruthLens — Forensic analysis service.

Provides signal-based forensic analysis using classical image processing.
These are informational signals — NEVER describe them as definitive proof.

Signals implemented:
- Error Level Analysis (ELA)
- Noise inconsistency analysis
- JPEG compression artifact analysis
- Edge anomaly detection
- Texture consistency analysis
- Resampling detection clues
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image

from backend.app.core.logging import get_logger

log = get_logger("forensics")


@dataclass
class ForensicSignal:
    """A single forensic signal with interpretation and limitations."""
    name: str
    score: float  # 0.0 = no anomaly, 1.0 = maximum anomaly
    interpretation: str
    limitations: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForensicsResult:
    """Complete forensic analysis result."""
    signals: List[ForensicSignal] = field(default_factory=list)
    overall_anomaly_score: float = 0.0
    runtime_ms: int = 0
    caveat: str = (
        "Forensic signals are informational indicators derived from classical "
        "image processing. They are NOT proof of manipulation. Many legitimate "
        "image processing operations (resizing, compression, filters) produce "
        "similar patterns."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signals": [
                {
                    "name": s.name,
                    "score": round(s.score, 4),
                    "interpretation": s.interpretation,
                    "limitations": s.limitations,
                    "details": s.details,
                }
                for s in self.signals
            ],
            "overall_anomaly_score": round(self.overall_anomaly_score, 4),
            "runtime_ms": self.runtime_ms,
            "caveat": self.caveat,
        }


def _error_level_analysis(img_array: np.ndarray) -> ForensicSignal:
    """
    Error Level Analysis (ELA).

    Re-saves the image at a known JPEG quality and measures the difference.
    Manipulated regions often show different error levels.
    """
    try:
        # Encode at quality 90
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
        _, enc = cv2.imencode(".jpg", img_array, encode_params)
        recompressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        # Compute absolute difference
        diff = cv2.absdiff(img_array, recompressed).astype(np.float32)
        ela_map = diff.mean(axis=2)  # Average across channels

        # Metrics
        mean_ela = float(np.mean(ela_map))
        std_ela = float(np.std(ela_map))
        max_ela = float(np.max(ela_map))

        # Normalized score (0-1): higher std relative to mean suggests inconsistency
        if mean_ela > 0:
            cv_ratio = std_ela / mean_ela  # Coefficient of variation
            score = min(cv_ratio / 3.0, 1.0)  # Normalize
        else:
            score = 0.0

        interpretation = (
            f"Mean ELA difference: {mean_ela:.2f}, Std: {std_ela:.2f}. "
            f"{'Relatively uniform error levels (normal).' if score < 0.3 else ''}"
            f"{'Some variation in error levels detected.' if 0.3 <= score < 0.6 else ''}"
            f"{'Notable variation in error levels.' if score >= 0.6 else ''}"
        )

        return ForensicSignal(
            name="error_level_analysis",
            score=score,
            interpretation=interpretation,
            limitations=(
                "ELA is a rough heuristic. Normal image processing (resizing, "
                "brightness, cropping) also affects error levels. Not reliable "
                "as a standalone indicator."
            ),
            details={"mean": round(mean_ela, 2), "std": round(std_ela, 2), "max": round(max_ela, 2)},
        )
    except Exception as e:
        return ForensicSignal(
            name="error_level_analysis",
            score=0.0,
            interpretation=f"ELA analysis failed: {e}",
            limitations="Analysis could not be performed.",
        )


def _noise_analysis(img_array: np.ndarray) -> ForensicSignal:
    """
    Noise consistency analysis.

    Compares noise levels across image regions. AI-generated images often
    have unnaturally uniform noise patterns.
    """
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # Estimate noise via Laplacian
        noise = cv2.Laplacian(gray, cv2.CV_64F)
        global_noise = float(np.std(noise))

        # Split into quadrants and measure noise per quadrant
        h, w = gray.shape
        quadrants = [
            noise[: h // 2, : w // 2],
            noise[: h // 2, w // 2:],
            noise[h // 2:, : w // 2],
            noise[h // 2:, w // 2:],
        ]
        quad_stds = [float(np.std(q)) for q in quadrants]

        # Check consistency: large variance among quadrants suggests edits
        if global_noise > 0:
            quad_cv = float(np.std(quad_stds)) / float(np.mean(quad_stds)) if np.mean(quad_stds) > 0 else 0
        else:
            quad_cv = 0.0

        score = min(quad_cv / 0.5, 1.0)  # Normalize

        interpretation = (
            f"Global noise level: {global_noise:.2f}. "
            f"Quadrant noise std variation: {quad_cv:.3f}. "
            f"{'Noise appears consistent across regions.' if score < 0.3 else ''}"
            f"{'Some noise variation detected between regions.' if 0.3 <= score < 0.6 else ''}"
            f"{'Significant noise inconsistency between regions.' if score >= 0.6 else ''}"
        )

        return ForensicSignal(
            name="noise_consistency",
            score=score,
            interpretation=interpretation,
            limitations=(
                "Noise patterns vary naturally with image content (sky vs texture). "
                "This signal alone cannot distinguish editing from natural variation."
            ),
            details={
                "global_noise": round(global_noise, 2),
                "quadrant_stds": [round(q, 2) for q in quad_stds],
                "quadrant_cv": round(quad_cv, 4),
            },
        )
    except Exception as e:
        return ForensicSignal(
            name="noise_consistency", score=0.0,
            interpretation=f"Noise analysis failed: {e}",
            limitations="Analysis could not be performed.",
        )


def _jpeg_artifact_analysis(img_array: np.ndarray, image_format: str) -> ForensicSignal:
    """
    JPEG compression artifact analysis.

    Checks for blocking artifacts and quantization patterns.
    """
    try:
        if image_format.upper() not in ("JPEG", "JPG"):
            return ForensicSignal(
                name="jpeg_artifacts",
                score=0.0,
                interpretation="Image is not JPEG — JPEG artifact analysis not applicable.",
                limitations="Only applicable to JPEG images.",
            )

        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY).astype(np.float64)
        h, w = gray.shape

        # Detect 8x8 block boundaries (JPEG uses 8x8 DCT blocks)
        block_diffs_h = []
        block_diffs_v = []

        for y in range(8, h - 8, 8):
            diff = np.abs(gray[y, :] - gray[y - 1, :]).mean()
            block_diffs_h.append(diff)

        for x in range(8, w - 8, 8):
            diff = np.abs(gray[:, x] - gray[:, x - 1]).mean()
            block_diffs_v.append(diff)

        mean_block_diff = float(np.mean(block_diffs_h + block_diffs_v))
        non_block_diffs = []
        for y in range(1, min(h - 1, 100)):
            if y % 8 != 0:
                diff = np.abs(gray[y, :w // 2] - gray[y - 1, :w // 2]).mean()
                non_block_diffs.append(diff)

        mean_non_block = float(np.mean(non_block_diffs)) if non_block_diffs else 1.0

        # Ratio: if block boundaries are more different than non-block, artifacts present
        if mean_non_block > 0:
            artifact_ratio = mean_block_diff / mean_non_block
        else:
            artifact_ratio = 1.0

        score = min(max(artifact_ratio - 1.0, 0.0) / 0.5, 1.0)

        interpretation = (
            f"Block boundary difference: {mean_block_diff:.2f}, "
            f"Non-block difference: {mean_non_block:.2f}. "
            f"{'Minimal JPEG blocking artifacts.' if score < 0.3 else ''}"
            f"{'Moderate JPEG compression detected.' if 0.3 <= score < 0.6 else ''}"
            f"{'Strong JPEG blocking artifacts visible.' if score >= 0.6 else ''}"
        )

        return ForensicSignal(
            name="jpeg_artifacts",
            score=score,
            interpretation=interpretation,
            limitations=(
                "JPEG artifacts depend on compression quality settings and are "
                "normal for any JPEG image. They do not indicate manipulation."
            ),
            details={
                "block_boundary_diff": round(mean_block_diff, 2),
                "non_block_diff": round(mean_non_block, 2),
                "artifact_ratio": round(artifact_ratio, 4),
            },
        )
    except Exception as e:
        return ForensicSignal(
            name="jpeg_artifacts", score=0.0,
            interpretation=f"JPEG analysis failed: {e}",
            limitations="Analysis could not be performed.",
        )


def _edge_analysis(img_array: np.ndarray) -> ForensicSignal:
    """
    Edge anomaly detection.

    Compares edge density and consistency across image regions.
    Spliced regions may show different edge characteristics.
    """
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        h, w = edges.shape
        # Split into a 3x3 grid
        grid_densities = []
        step_h, step_w = h // 3, w // 3
        for gy in range(3):
            for gx in range(3):
                region = edges[gy * step_h:(gy + 1) * step_h, gx * step_w:(gx + 1) * step_w]
                density = float(np.count_nonzero(region)) / region.size
                grid_densities.append(density)

        mean_density = float(np.mean(grid_densities))
        std_density = float(np.std(grid_densities))

        if mean_density > 0:
            cv = std_density / mean_density
        else:
            cv = 0.0

        # High CV in edge density can indicate splicing, but also just varied content
        score = min(cv / 1.5, 1.0)

        interpretation = (
            f"Mean edge density: {mean_density:.4f}, Std: {std_density:.4f}. "
            f"{'Edge distribution appears natural.' if score < 0.3 else ''}"
            f"{'Some variation in edge distribution.' if 0.3 <= score < 0.6 else ''}"
            f"{'Notable variation in edge density across regions.' if score >= 0.6 else ''}"
        )

        return ForensicSignal(
            name="edge_consistency",
            score=score,
            interpretation=interpretation,
            limitations=(
                "Edge density naturally varies with image content (sky vs detailed textures). "
                "This cannot distinguish editing from natural composition."
            ),
            details={
                "mean_density": round(mean_density, 4),
                "std_density": round(std_density, 4),
                "grid_densities": [round(d, 4) for d in grid_densities],
            },
        )
    except Exception as e:
        return ForensicSignal(
            name="edge_consistency", score=0.0,
            interpretation=f"Edge analysis failed: {e}",
            limitations="Analysis could not be performed.",
        )


def _texture_analysis(img_array: np.ndarray) -> ForensicSignal:
    """
    Texture consistency analysis using local binary patterns (simplified).

    AI-generated images sometimes exhibit unnatural texture regularity.
    """
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Compute texture energy using Gabor-like responses
        ksize = 31
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        responses = []
        for theta in angles:
            kernel = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_64F)
            filtered = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, kernel)
            responses.append(float(np.std(filtered)))

        # Compare texture energy across orientations
        mean_response = float(np.mean(responses))
        std_response = float(np.std(responses))

        if mean_response > 0:
            regularity = 1.0 - (std_response / mean_response)  # Higher = more regular
        else:
            regularity = 0.5

        # Extremely regular or extremely irregular textures can be suspicious
        # But we keep score moderate since texture varies widely
        score = max(0.0, regularity - 0.7) / 0.3  # Only flag very high regularity
        score = min(score, 1.0)

        interpretation = (
            f"Texture regularity: {regularity:.3f}. "
            f"{'Normal texture variation.' if score < 0.3 else ''}"
            f"{'Somewhat uniform texture pattern.' if 0.3 <= score < 0.6 else ''}"
            f"{'Unusually uniform texture (may indicate synthetic origin).' if score >= 0.6 else ''}"
        )

        return ForensicSignal(
            name="texture_consistency",
            score=score,
            interpretation=interpretation,
            limitations=(
                "Texture regularity depends heavily on image content. Photos of "
                "uniform surfaces (walls, sky) naturally show regular texture patterns."
            ),
            details={
                "gabor_responses": [round(r, 2) for r in responses],
                "regularity": round(regularity, 4),
            },
        )
    except Exception as e:
        return ForensicSignal(
            name="texture_consistency", score=0.0,
            interpretation=f"Texture analysis failed: {e}",
            limitations="Analysis could not be performed.",
        )


def _resampling_detection(img_array: np.ndarray) -> ForensicSignal:
    """
    Resampling detection clues.

    Looks for periodic patterns in pixel differences that might indicate
    scaling/rotation interpolation.
    """
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # Compute second-order differences
        dx2 = np.diff(gray, n=2, axis=1)
        dy2 = np.diff(gray, n=2, axis=0)

        # Zero-crossing rate in second-order differences
        zc_x = float(np.mean(np.abs(np.diff(np.sign(dx2), axis=1)) > 0))
        zc_y = float(np.mean(np.abs(np.diff(np.sign(dy2), axis=0)) > 0))

        # Natural images typically have higher zero-crossing rates
        # Resampled images may show periodic patterns (lower ZC rate)
        avg_zc = (zc_x + zc_y) / 2

        # Lower zero-crossing rate may suggest interpolation artifacts
        # But this is very weak evidence
        score = max(0.0, 0.5 - avg_zc) / 0.5 if avg_zc < 0.5 else 0.0
        score = min(score, 1.0)

        interpretation = (
            f"Second-order zero-crossing rate: {avg_zc:.4f}. "
            f"{'No strong resampling indicators.' if score < 0.3 else ''}"
            f"{'Possible interpolation artifacts detected.' if score >= 0.3 else ''}"
        )

        return ForensicSignal(
            name="resampling_clues",
            score=score,
            interpretation=interpretation,
            limitations=(
                "Resampling detection through zero-crossing analysis is a weak signal. "
                "Many normal operations (camera processing, social media upload) involve "
                "resampling. Cannot distinguish intentional manipulation from normal processing."
            ),
            details={
                "zero_crossing_x": round(zc_x, 4),
                "zero_crossing_y": round(zc_y, 4),
                "average_zc": round(avg_zc, 4),
            },
        )
    except Exception as e:
        return ForensicSignal(
            name="resampling_clues", score=0.0,
            interpretation=f"Resampling analysis failed: {e}",
            limitations="Analysis could not be performed.",
        )


def analyze_forensics(image_data: bytes, image_format: str = "JPEG") -> ForensicsResult:
    """
    Run the complete forensic analysis pipeline on raw image bytes.

    Returns ForensicsResult with multiple informational signals.
    These are NOT proof of manipulation.
    """
    t0 = time.time()

    # Decode image
    img_array = cv2.imdecode(
        np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR
    )
    if img_array is None:
        return ForensicsResult(
            signals=[],
            overall_anomaly_score=0.0,
            runtime_ms=0,
        )

    # Run all analyses
    signals = [
        _error_level_analysis(img_array),
        _noise_analysis(img_array),
        _jpeg_artifact_analysis(img_array, image_format),
        _edge_analysis(img_array),
        _texture_analysis(img_array),
        _resampling_detection(img_array),
    ]

    # Overall score: weighted average of individual signals
    weights = {
        "error_level_analysis": 0.25,
        "noise_consistency": 0.20,
        "jpeg_artifacts": 0.15,
        "edge_consistency": 0.15,
        "texture_consistency": 0.15,
        "resampling_clues": 0.10,
    }
    total_weight = sum(weights.get(s.name, 0.1) for s in signals)
    overall = sum(s.score * weights.get(s.name, 0.1) for s in signals) / total_weight if total_weight > 0 else 0.0

    runtime_ms = int((time.time() - t0) * 1000)

    result = ForensicsResult(
        signals=signals,
        overall_anomaly_score=overall,
        runtime_ms=runtime_ms,
    )

    log.info(
        "Forensics analysis: %d signals, overall=%.3f [%d ms]",
        len(signals), overall, runtime_ms,
    )

    return result
