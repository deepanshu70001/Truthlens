"""Temperature-scaling support for authenticity logits.

Calibration is opt-in: a candidate artifact is accepted only when it records a
separate validation split, matching model ID, documented provenance, and the
diagnostics required by the project. No artifact is shipped with TruthLens.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class CalibrationArtifactError(ValueError):
    """Raised when a calibration artifact is incomplete or incompatible."""


def load_calibration_artifact(path: str | Path, model_id: str, model_revision: str | None = None) -> dict[str, Any]:
    """Load a validated two-class temperature-scaling artifact."""
    artifact_path = Path(path)
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationArtifactError("Calibration artifact could not be read.") from exc

    required = {
        "schema_version", "method", "model_id", "validation_manifest_sha256",
        "validation_split", "sample_count", "class_counts", "temperature",
        "thresholds", "metrics_before", "metrics_after", "validated", "model_revision",
        "ai_generation_families",
    }
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise CalibrationArtifactError("Calibration artifact is missing required validation metadata.")
    if artifact["schema_version"] != 1 or artifact["method"] != "temperature_scaling":
        raise CalibrationArtifactError("Unsupported calibration artifact schema or method.")
    if artifact["model_id"] != model_id:
        raise CalibrationArtifactError("Calibration artifact belongs to a different model.")
    if not model_revision or artifact["model_revision"] != model_revision:
        raise CalibrationArtifactError("Calibration artifact does not match the exact model checkpoint revision.")
    if artifact["validation_split"] != "validation" or artifact["validated"] is not True:
        raise CalibrationArtifactError("Only explicitly validated validation-split artifacts are accepted.")
    if not isinstance(artifact["validation_manifest_sha256"], str) or len(artifact["validation_manifest_sha256"]) != 64:
        raise CalibrationArtifactError("Calibration artifact has no valid manifest fingerprint.")
    counts = artifact["class_counts"]
    if not isinstance(counts, dict) or any(not isinstance(counts.get(k), int) or counts[k] < 30 for k in ("real", "ai_generated")):
        raise CalibrationArtifactError("At least 30 reviewed examples per class are required for calibration use.")
    if artifact["sample_count"] != counts["real"] + counts["ai_generated"]:
        raise CalibrationArtifactError("Calibration sample counts are inconsistent.")
    if not isinstance(artifact["ai_generation_families"], list) or len(set(artifact["ai_generation_families"])) < 2:
        raise CalibrationArtifactError("Calibration data must document multiple AI-generation families.")
    try:
        temperature = float(artifact["temperature"])
        ai_threshold = float(artifact["thresholds"]["ai_generated_min"])
        authentic_threshold = float(artifact["thresholds"]["authentic_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationArtifactError("Calibration parameters are invalid.") from exc
    if not math.isfinite(temperature) or temperature <= 0:
        raise CalibrationArtifactError("Temperature must be finite and greater than zero.")
    if not (0 <= authentic_threshold < ai_threshold <= 1):
        raise CalibrationArtifactError("Decision thresholds must define a non-empty uncertainty band.")
    for key in ("metrics_before", "metrics_after"):
        metrics = artifact[key]
        if not isinstance(metrics, dict) or any(k not in metrics for k in ("nll", "brier", "ece")):
            raise CalibrationArtifactError("Calibration diagnostics must include NLL, Brier score, and ECE.")
        if any(not isinstance(metrics[k], (float, int)) or not math.isfinite(metrics[k]) for k in ("nll", "brier", "ece")):
            raise CalibrationArtifactError("Calibration diagnostics must be finite numeric values.")
    return artifact


def calibrated_ai_probability(fake_logit: float, real_logit: float, temperature: float) -> float:
    """Return binary temperature-scaled probability for the AI-generated class."""
    difference = (float(fake_logit) - float(real_logit)) / float(temperature)
    if not math.isfinite(difference) or temperature <= 0:
        raise ValueError("Logits and temperature must be finite; temperature must be positive.")
    # Numerically stable sigmoid.
    if difference >= 0:
        return 1.0 / (1.0 + math.exp(-difference))
    exp_value = math.exp(difference)
    return exp_value / (1.0 + exp_value)
