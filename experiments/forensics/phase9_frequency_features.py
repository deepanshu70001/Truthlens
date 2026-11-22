"""Deterministic, source-agnostic forensic image feature extraction for Phase 9.

The extractor consumes decoded pixels only. It never reads a path, label, or
dataset identifier. Features are computed in memory after the canonical
Phase 7 resize-short-side/center-crop preprocessing contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from scipy.fft import dctn

FEATURE_NAMES = (
    "fft_log_energy_total",
    "fft_log_energy_low",
    "fft_log_energy_mid",
    "fft_log_energy_high",
    "fft_high_low_ratio",
    "fft_spectral_centroid",
    "fft_spectral_flatness",
    "fft_axis_anisotropy",
    "dct_log_energy_dc",
    "dct_log_energy_low",
    "dct_log_energy_mid",
    "dct_log_energy_high",
    "dct_high_low_ratio",
    "dct_ac_sparsity",
    "dct_block_energy_cv",
    "residual_mean_abs",
    "residual_std",
    "residual_mad",
    "residual_skewness",
    "residual_kurtosis",
    "residual_zero_fraction",
    "residual_horizontal_corr",
    "residual_vertical_corr",
    "edge_density",
    "edge_strength_mean",
    "edge_strength_std",
    "laplacian_variance",
    "gradient_energy",
    "local_contrast_mean",
    "local_contrast_std",
    "resample_periodicity_x",
    "resample_periodicity_y",
    "resample_peak_x",
    "resample_peak_y",
)


def _canonical_gray(image: Image.Image, size: int = 224) -> np.ndarray:
    """Apply documented Phase 7 decode and geometry in deterministic RGB/L mode."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    if min(width, height) < 1:
        raise ValueError("image has an empty dimension")
    scale = size / min(width, height)
    resized = rgb.resize(
        (max(size, round(width * scale)), max(size, round(height * scale))),
        resample=Image.Resampling.BILINEAR,
    )
    left = (resized.width - size) // 2
    top = (resized.height - size) // 2
    cropped = resized.crop((left, top, left + size, top + size)).convert("L")
    return np.asarray(cropped, dtype=np.float32) / 255.0


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract_frequency_features(image: Image.Image) -> dict[str, Any]:
    """Return named deterministic features and explicit status for a PIL image.

    Images smaller than 8 pixels on either side are marked unsupported. Decode
    and numerical errors are captured in the result so batch jobs can continue.
    """
    result: dict[str, Any] = {
        "status": "error",
        "failure": None,
        "feature_names": list(FEATURE_NAMES),
        "features": None,
    }
    try:
        if not isinstance(image, Image.Image):
            raise TypeError("input must be a PIL.Image.Image")
        if image.width < 8 or image.height < 8:
            result.update(status="unsupported", failure="image dimension below 8 pixels")
            return result

        gray = _canonical_gray(image)
        centered = gray - float(gray.mean())
        eps = 1e-12

        # FFT radial energy and directional anisotropy.
        spectrum = np.fft.fftshift(np.fft.fft2(centered))
        power = np.abs(spectrum) ** 2
        yy, xx = np.indices(power.shape, dtype=np.float32)
        radius = np.sqrt((xx - 111.5) ** 2 + (yy - 111.5) ** 2) / (np.sqrt(2) * 112)
        masks = (radius < 0.15, (radius >= 0.15) & (radius < 0.50), radius >= 0.50)
        fft_bands = [float(power[m].mean()) for m in masks]
        total_fft = float(power.mean())
        magnitude = np.sqrt(power + eps)
        normalized_power = power / (power.sum() + eps)
        spectral_centroid = float((radius * normalized_power).sum())
        spectral_flatness = float(np.exp(np.log(magnitude + eps).mean()) / (magnitude.mean() + eps))
        row_power = power.sum(axis=1)
        col_power = power.sum(axis=0)
        axis_anisotropy = float(abs(row_power.std() - col_power.std()) / (row_power.std() + col_power.std() + eps))

        # Non-overlapping 8x8 orthonormal DCT blocks. Exclude each block's DC
        # term from AC bands; report DC separately.
        blocks = gray[: (gray.shape[0] // 8) * 8, : (gray.shape[1] // 8) * 8].reshape(28, 8, 28, 8)
        blocks = blocks.transpose(0, 2, 1, 3)
        coeff = dctn(blocks, axes=(-2, -1), norm="ortho")
        dct_power = coeff**2
        cy, cx = np.indices((8, 8))
        freq = np.sqrt(cx**2 + cy**2)
        dct_low = (freq > 0) & (freq <= 2)
        dct_mid = (freq > 2) & (freq <= 5)
        dct_high = freq > 5
        dct_dc = float(dct_power[..., 0, 0].mean())
        dct_bands = [float(dct_power[..., mask].mean()) for mask in (dct_low, dct_mid, dct_high)]
        ac = np.abs(coeff[..., 1:, :]).ravel()
        block_energy = dct_power.sum(axis=(-2, -1)).ravel()
        dct_sparsity = float(np.mean(ac <= np.quantile(ac, 0.5)))
        block_cv = float(block_energy.std() / (block_energy.mean() + eps))

        # High-pass residual against a small Gaussian low-pass.
        gray_image = Image.fromarray(np.uint8(np.clip(gray * 255, 0, 255)))
        lowpass = np.asarray(gray_image.filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.float32) / 255.0
        residual = gray - lowpass
        r = residual.astype(np.float64)
        r_centered = r - r.mean()
        r_std = float(r.std())
        r_mad = float(np.median(np.abs(r_centered)))
        r_skew = float(np.mean(r_centered**3) / (r_std**3 + eps))
        r_kurt = float(np.mean(r_centered**4) / (r_std**4 + eps))
        residual_hcorr = _safe_corr(r[:, :-1], r[:, 1:])
        residual_vcorr = _safe_corr(r[:-1, :], r[1:, :])

        # Edge/texture measures.
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
        gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
        grad = np.hypot(gx, gy)
        lap = -4 * gray + np.roll(gray, 1, 0) + np.roll(gray, -1, 0) + np.roll(gray, 1, 1) + np.roll(gray, -1, 1)
        local_mean = np.asarray(gray_image.filter(ImageFilter.GaussianBlur(radius=3.0)), dtype=np.float32) / 255.0
        local_contrast = np.abs(gray - local_mean)
        edge_strength = grad[2:-2, 2:-2]
        edge_density = float(np.mean(edge_strength > 0.08))

        # Resampling periodicity proxy: peaks in first-difference autocorrelation
        # at lags 2..16 after removing the mean, averaged over rows/columns.
        dx = np.diff(gray, axis=1)
        dy = np.diff(gray, axis=0)

        def periodicity(values: np.ndarray, spatial_axis: int) -> tuple[float, int]:
            # Average across the orthogonal axis first. This keeps the statistic
            # deterministic and avoids tens of millions of redundant per-pixel
            # correlations for a single image.
            signal = values.mean(axis=0 if spatial_axis == 1 else 1).ravel()
            scores = []
            for lag in range(2, 17):
                score = _safe_corr(signal[:-lag], signal[lag:])
                scores.append(score)
            index = int(np.argmax(scores))
            return float(scores[index]), index + 2

        px, peak_x = periodicity(dx, 1)
        py, peak_y = periodicity(dy, 0)

        values = [
            np.log1p(total_fft), np.log1p(fft_bands[0]), np.log1p(fft_bands[1]), np.log1p(fft_bands[2]),
            np.log1p(fft_bands[2] / (fft_bands[0] + eps)), spectral_centroid, spectral_flatness, axis_anisotropy,
            np.log1p(dct_dc), *(np.log1p(v) for v in dct_bands), np.log1p(dct_bands[2] / (dct_bands[0] + eps)),
            dct_sparsity, block_cv, float(np.mean(np.abs(r))), r_std, r_mad, r_skew, r_kurt,
            float(np.mean(r == 0)), residual_hcorr, residual_vcorr, edge_density, float(edge_strength.mean()),
            float(edge_strength.std()), float(np.var(lap)), float(np.mean(grad**2)), float(local_contrast.mean()),
            float(local_contrast.std()), px, py, float(peak_x), float(peak_y),
        ]
        vector = np.asarray(values, dtype=np.float64)
        if vector.size != len(FEATURE_NAMES) or not np.isfinite(vector).all():
            raise FloatingPointError("feature vector has invalid shape or non-finite values")
        result.update(status="ok", failure=None, features=vector.tolist())
        return result
    except Exception as exc:  # batch extraction records failures instead of aborting
        result["failure"] = f"{type(exc).__name__}: {exc}"
        return result
