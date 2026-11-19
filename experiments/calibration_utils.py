"""Deterministic helpers for binary temperature scaling and diagnostics."""

from __future__ import annotations

import math

import numpy as np


def probabilities_from_margins(margins: list[float], temperature: float) -> list[float]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    values = np.asarray(margins, dtype=float) / temperature
    return (1.0 / (1.0 + np.exp(-np.clip(values, -700, 700)))).tolist()


def binary_diagnostics(labels: list[int], probabilities: list[float], bins: int = 10) -> dict:
    if not labels or len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must be non-empty and have equal length")
    y = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1 - 1e-12)
    if not np.isfinite(y).all() or not set(y.tolist()).issubset({0.0, 1.0}) or not np.isfinite(p).all():
        raise ValueError("labels or probabilities contain invalid values")
    nll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    brier = float(np.mean((p - y) ** 2))
    ece = 0.0
    rows = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (p >= low) & ((p < high) if index < bins - 1 else (p <= high))
        count = int(mask.sum())
        if count:
            confidence = float(p[mask].mean())
            frequency = float(y[mask].mean())
            gap = abs(confidence - frequency)
            ece += count / len(p) * gap
            rows.append({"low": low, "high": high, "count": count,
                         "mean_prediction": confidence, "observed_frequency": frequency, "gap": gap})
    return {"n": len(labels), "nll": nll, "brier": brier, "ece": float(ece), "bins": rows,
            "warning": "Validation-set diagnostics describe calibration fit data; they do not estimate held-out test accuracy or establish domain generalization."}


def fit_temperature(margins: list[float], labels: list[int]) -> float:
    """Minimize binary validation NLL over log(T) with bounded golden search."""
    if len(margins) != len(labels) or not margins or set(labels) != {0, 1}:
        raise ValueError("temperature fitting requires equal non-empty arrays containing both classes")
    m = np.asarray(margins, dtype=float)
    y = np.asarray(labels, dtype=float)
    if not np.isfinite(m).all():
        raise ValueError("logits must be finite")

    def objective(log_temperature: float) -> float:
        p = np.clip(1 / (1 + np.exp(-np.clip(m / math.exp(log_temperature), -700, 700))), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    left, right = -5.0, 5.0
    ratio = (math.sqrt(5) - 1) / 2
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    fc, fd = objective(c), objective(d)
    for _ in range(120):
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = objective(c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = objective(d)
    return float(math.exp((left + right) / 2))


def select_abstention_thresholds(labels: list[int], probabilities: list[float], max_class_error: float = 0.10) -> dict:
    """Choose the highest-coverage low/high pair within class-wise error caps."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0 or len(y) != len(p) or set(y.tolist()) != {0, 1}:
        raise ValueError("threshold selection requires both classes and aligned arrays")
    best = None
    grid = np.linspace(0.0, 1.0, 101)
    real_count, ai_count = int((y == 0).sum()), int((y == 1).sum())
    for auth_max in grid:
        predicted_real = p <= auth_max
        false_negative_rate = float(((y == 1) & predicted_real).sum() / ai_count)
        if false_negative_rate > max_class_error:
            continue
        for ai_min in grid[grid > auth_max]:
            predicted_ai = p >= ai_min
            false_positive_rate = float(((y == 0) & predicted_ai).sum() / real_count)
            if false_positive_rate > max_class_error:
                continue
            coverage = float((predicted_real | predicted_ai).mean())
            key = (coverage, -(false_negative_rate + false_positive_rate), -(ai_min - auth_max))
            if best is None or key > best[0]:
                best = (key, float(auth_max), float(ai_min), false_negative_rate, false_positive_rate)
    if best is None:
        return {"authentic_max": 0.0, "ai_generated_min": 1.0, "coverage": 0.0,
                "false_negative_rate": 0.0, "false_positive_rate": 0.0,
                "selection_status": "no_pair_met_error_caps_abstain_all"}
    key, auth_max, ai_min, fnr, fpr = best
    return {"authentic_max": auth_max, "ai_generated_min": ai_min, "coverage": key[0],
            "false_negative_rate": fnr, "false_positive_rate": fpr,
            "selection_status": "validation_selected_with_class_error_caps"}
