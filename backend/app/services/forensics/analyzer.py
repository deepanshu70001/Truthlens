"""Classical image indicators. These computations are informational, not proof."""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from backend.app.core.logging import get_logger

log = get_logger("forensics")


@dataclass
class ForensicSignal:
    name: str
    score: Optional[float]
    interpretation: str
    limitations: str
    details: Dict[str, Any] = field(default_factory=dict)
    status: str = "completed"


@dataclass
class ForensicsResult:
    signals: List[ForensicSignal] = field(default_factory=list)
    # Kept for report compatibility. No aggregate is computed from unrelated heuristics.
    overall_anomaly_score: Optional[float] = None
    runtime_ms: int = 0
    status: str = "completed"
    warnings: List[str] = field(default_factory=list)
    caveat: str = (
        "Classical forensic indicators are content- and encoding-dependent heuristics. "
        "They are not calibrated probabilities and are not proof of manipulation."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "signals": [{
                "name": s.name, "status": s.status,
                "score": round(float(s.score), 4) if s.score is not None else None,
                "score_type": "heuristic_indicator_not_probability" if s.score is not None else None,
                "interpretation": s.interpretation, "limitations": s.limitations,
                "details": s.details,
            } for s in self.signals],
            "overall_anomaly_score": None,
            "runtime_ms": self.runtime_ms,
            "warnings": self.warnings,
            "caveat": self.caveat,
        }


def _signal(name: str, compute, limitations: str, not_applicable: bool = False) -> ForensicSignal:
    if not_applicable:
        return ForensicSignal(name, None, "Not applicable to this image format.", limitations, status="skipped")
    try:
        score, interpretation, details = compute()
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("computed indicator is not finite or outside [0, 1]")
        return ForensicSignal(name, float(score), interpretation, limitations, details)
    except Exception as exc:
        log.warning("Forensic signal %s failed (%s)", name, type(exc).__name__)
        return ForensicSignal(name, None, f"{name} computation failed.", limitations, status="failed")


def analyze_forensics(
    image_data: bytes,
    image_format: str = "JPEG",
    analysis_id: str | None = None,
    generate_artifacts: bool = True,
) -> ForensicsResult:
    """Run the six existing forensic indicators on an independently decoded image."""
    started = time.perf_counter()
    warnings: List[str] = []
    try:
        with Image.open(io.BytesIO(image_data)) as source:
            source_format = (source.format or image_format or "UNKNOWN").upper()
            source.load()
            rgb = np.asarray(source.convert("RGB"))
        if rgb.ndim != 3 or rgb.shape[0] < 3 or rgb.shape[1] < 3:
            raise ValueError("image dimensions are too small")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    except Exception as exc:
        return ForensicsResult(status="failed", warnings=[f"Image decode failed ({type(exc).__name__})."])

    limitations = {
        "error_level_analysis": "Compares decoded pixels with a JPEG quality-90 re-encode; highly encoding-dependent. For non-JPEG sources it is not directly comparable to native JPEG ELA. Recompression and ordinary editing affect it.",
        "noise_consistency": "Laplacian residual variation depends on scene texture, sharpening, denoising and camera processing; it is not a sensor-noise provenance test.",
        "jpeg_artifacts": "Boundary discontinuities are affected by JPEG quality, crop alignment and later processing; they do not identify editing.",
        "edge_consistency": "Edge density varies naturally with scene content and composition.",
        "texture_consistency": "Gabor response variation depends strongly on subject matter, focus and image processing.",
        "resampling_clues": "Second-difference zero crossings are a weak clue and can be changed by normal resizing, camera processing or social-media recompression.",
    }
    h, w = gray.shape
    signals: List[ForensicSignal] = []

    def ela():
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("JPEG re-encoding failed")
        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if recompressed is None or recompressed.shape != bgr.shape:
            raise ValueError("JPEG re-encode did not decode to the expected shape")
        diff = cv2.absdiff(bgr, recompressed).astype(np.float32).mean(axis=2)
        mean, std, maximum = float(diff.mean()), float(diff.std()), float(diff.max())
        score = min(std / mean / 3.0, 1.0) if mean > 0 else 0.0
        interpretation = f"JPEG quality-90 difference: mean {mean:.2f}, standard deviation {std:.2f}, maximum {maximum:.2f}."
        if source_format not in {"JPEG", "JPG"}:
            interpretation += f" Source format is {source_format}; treat this JPEG comparison as encoding-dependent."
        return score, interpretation, {
            "mean_difference": round(mean, 3), "std_difference": round(std, 3),
            "max_difference": round(maximum, 3), "reencode_quality": 90,
            "source_format": source_format,
        }

    signals.append(_signal("error_level_analysis", ela, limitations["error_level_analysis"]))

    def noise():
        residual = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
        levels = [float(np.std(residual[y0:y1, x0:x1])) for y0, y1 in ((0, h//2), (h//2, h)) for x0, x1 in ((0, w//2), (w//2, w))]
        mean = float(np.mean(levels))
        cv = float(np.std(levels) / mean) if mean > 0 else 0.0
        return min(cv / 0.5, 1.0), f"Laplacian residual standard deviation varies across quadrants (coefficient of variation {cv:.4f}).", {"global_residual_std": round(float(residual.std()), 3), "quadrant_residual_stds": [round(x, 3) for x in levels], "quadrant_cv": round(cv, 5)}

    signals.append(_signal("noise_consistency", noise, limitations["noise_consistency"]))

    def jpeg():
        if source_format not in {"JPEG", "JPG"}:
            raise NotImplementedError
        if h < 16 or w < 16:
            raise ValueError("image is too small to evaluate multiple JPEG blocks")
        horizontal = [float(np.mean(np.abs(gray[y, :].astype(float) - gray[y-1, :].astype(float)))) for y in range(8, h, 8)]
        vertical = [float(np.mean(np.abs(gray[:, x].astype(float) - gray[:, x-1].astype(float)))) for x in range(8, w, 8)]
        block = horizontal + vertical
        nonblock = [float(np.mean(np.abs(gray[y, :].astype(float) - gray[y-1, :].astype(float)))) for y in range(1, h) if y % 8]
        if not block or not nonblock:
            raise ValueError("insufficient JPEG block boundaries")
        block_mean, other_mean = float(np.mean(block)), float(np.mean(nonblock))
        ratio = block_mean / other_mean if other_mean > 0 else (1.0 if block_mean == 0 else float("inf"))
        if not np.isfinite(ratio):
            raise ValueError("boundary ratio is not finite")
        score = min(max(ratio - 1.0, 0.0) / 0.5, 1.0)
        return score, f"8-pixel boundary mean difference {block_mean:.3f}; other-boundary mean difference {other_mean:.3f}.", {"block_boundary_difference": round(block_mean, 4), "non_block_boundary_difference": round(other_mean, 4), "ratio": round(ratio, 5)}

    if source_format not in {"JPEG", "JPG"}:
        signals.append(ForensicSignal("jpeg_artifacts", None, f"JPEG artifact analysis is not applicable to {source_format} input (skipped).", limitations["jpeg_artifacts"], status="skipped"))
    else:
        signals.append(_signal("jpeg_artifacts", jpeg, limitations["jpeg_artifacts"]))

    def edge():
        edges = cv2.Canny(gray, 50, 150)
        sh, sw = h // 3, w // 3
        densities = [float(np.count_nonzero(edges[y*sh:(y+1)*sh if y < 2 else h, x*sw:(x+1)*sw if x < 2 else w])) / max(1, ((sh if y < 2 else h-y*sh) * (sw if x < 2 else w-x*sw))) for y in range(3) for x in range(3)]
        mean, std = float(np.mean(densities)), float(np.std(densities))
        cv = std / mean if mean > 0 else 0.0
        return min(cv / 1.5, 1.0), f"Canny edge density varies across a 3x3 grid (coefficient of variation {cv:.4f}).", {"mean_density": round(mean, 6), "std_density": round(std, 6), "grid_densities": [round(x, 6) for x in densities]}

    signals.append(_signal("edge_consistency", edge, limitations["edge_consistency"]))

    def texture():
        responses = []
        for angle in (0, np.pi/4, np.pi/2, 3*np.pi/4):
            kernel = cv2.getGaborKernel((31,31), 4.0, angle, 10.0, 0.5, 0, ktype=cv2.CV_64F)
            responses.append(float(np.std(cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, kernel))))
        mean, std = float(np.mean(responses)), float(np.std(responses))
        regularity = 1.0 - std / mean if mean > 0 else 0.5
        score = min(max(regularity - 0.7, 0.0) / 0.3, 1.0)
        return score, f"Gabor response regularity indicator is {regularity:.4f}.", {"gabor_responses": [round(x, 4) for x in responses], "regularity": round(regularity, 5)}

    signals.append(_signal("texture_consistency", texture, limitations["texture_consistency"]))

    def resampling():
        dx2, dy2 = np.diff(gray.astype(np.float64), n=2, axis=1), np.diff(gray.astype(np.float64), n=2, axis=0)
        if dx2.shape[1] < 2 or dy2.shape[0] < 2:
            raise ValueError("image is too small for second-difference analysis")
        zcx = float(np.mean(np.diff(np.sign(dx2), axis=1) != 0))
        zcy = float(np.mean(np.diff(np.sign(dy2), axis=0) != 0))
        avg = (zcx + zcy) / 2
        score = max(0.0, 0.5 - avg) / 0.5
        return min(score, 1.0), f"Second-difference zero-crossing rate is {avg:.5f}.", {"zero_crossing_x": round(zcx, 6), "zero_crossing_y": round(zcy, 6), "average_zero_crossing": round(avg, 6)}

    signals.append(_signal("resampling_clues", resampling, limitations["resampling_clues"]))

    ela_signal = signals[0]
    if generate_artifacts and analysis_id and ela_signal.score is not None:
        try:
            from backend.app.config import settings
            settings.ensure_storage_dirs()
            ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else None
            if recompressed is None:
                raise OSError("ELA recompression could not be decoded")
            difference = cv2.cvtColor(cv2.absdiff(bgr, recompressed), cv2.COLOR_BGR2GRAY)
            scaled = cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX)
            output = settings.artifacts_dir / "forensics"
            output.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output / f"{analysis_id}_ela.png"), scaled):
                raise OSError("ELA artifact could not be written")
            ela_signal.details["artifact_url"] = f"/api/v1/analysis/{analysis_id}/forensics/ela"
        except Exception as exc:
            warnings.append(f"ELA artifact could not be saved ({type(exc).__name__}).")
            ela_signal.status = "partial"

    for signal in signals:
        if signal.status in {"failed", "unavailable"}:
            warnings.append(f"{signal.name}: {signal.interpretation}")
    status = "partial" if warnings or any(s.status in {"failed", "unavailable", "partial"} for s in signals) else "completed"
    result = ForensicsResult(signals=signals, runtime_ms=int((time.perf_counter()-started)*1000), status=status, warnings=warnings)
    log.info("Forensics complete: %d signals (%s) in %dms", len(signals), status, result.runtime_ms)
    return result
