"""Temperature-scaling support for authenticity logits.

Calibration is opt-in: an artifact is accepted only when it records a
separate validation split, matching canonical model identity (Schema Version 2),
documented provenance, and the diagnostics required by the project.
No artifact is shipped with TruthLens.

Legacy schema version 1 artifacts are safely rejected because they cannot
prove complete model identity (backbone revision, head hash, preprocessing
version, and label mapping). All calibration artifacts must conform to
Schema Version 2.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Optional


class CalibrationArtifactError(ValueError):
    """Raised when a calibration artifact is incomplete or incompatible."""


_REQUIRED_V1 = frozenset({
    "schema_version", "method", "model_id", "validation_manifest_sha256",
    "validation_split", "sample_count", "class_counts", "temperature",
    "thresholds", "metrics_before", "metrics_after", "validated", "model_revision",
    "ai_generation_families",
})

_IDENTITY_FIELDS_V2 = frozenset({
    "backbone_model", "backbone_revision", "head_sha256",
    "preprocessing_version", "label_mapping", "inference_config",
})


def load_calibration_artifact(
    path: str | Path,
    model_id: str,
    model_revision: str | None = None,
    model_identity: Optional[Dict[str, Any]] = None,
) -> dict[str, Any]:
    """Load a validated two-class temperature-scaling artifact.

    Parameters
    ----------
    path : str | Path
        Path to the JSON calibration artifact.
    model_id : str
        Runtime model identifier (must match ``artifact["model_id"]``).
    model_revision : str | None
        Runtime model revision string.
    model_identity : dict | None
        Full model identity dict from ``AuthenticityDetector.model_identity()``.
        When provided, the artifact is checked against every identity field.
    """
    artifact_path = Path(path)

    # ── Read & parse ─────────────────────────────────────────────
    try:
        raw = artifact_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CalibrationArtifactError("Calibration artifact could not be read.") from exc

    try:
        artifact = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise CalibrationArtifactError("Calibration artifact is not valid JSON.") from exc

    if not isinstance(artifact, dict):
        raise CalibrationArtifactError("Calibration artifact must be a JSON object.")

    # ── Schema version & method ──────────────────────────────────
    schema_version = artifact.get("schema_version")
    if schema_version == 1:
        raise CalibrationArtifactError(
            "Legacy schema version 1 calibration artifacts cannot prove complete model identity "
            "(backbone revision, classification head hash, preprocessing version, and label mapping) "
            "and are rejected for safety. Re-calibrate with calibrate_authenticity.py to generate a schema version 2 artifact."
        )
    if schema_version != 2:
        raise CalibrationArtifactError("Unsupported calibration artifact schema version.")
    if artifact.get("method") != "temperature_scaling":
        raise CalibrationArtifactError("Unsupported calibration method.")

    # ── Required keys ────────────────────────────────────────────
    required = set(_REQUIRED_V1) | _IDENTITY_FIELDS_V2
    missing = required - set(artifact)
    if missing:
        raise CalibrationArtifactError(
            f"Calibration artifact is missing required fields: {', '.join(sorted(missing))}."
        )

    # ── Model identity — primary fields ──────────────────────────
    if artifact["model_id"] != model_id:
        raise CalibrationArtifactError("Calibration artifact belongs to a different model.")
    if not model_revision or artifact["model_revision"] != model_revision:
        raise CalibrationArtifactError(
            "Calibration artifact does not match the exact model checkpoint revision."
        )

    # ── Model identity — extended V2 fields ──────────────────────
    if model_identity is not None and schema_version == 2:
        if artifact.get("model_id") != model_identity.get("model_id"):
            raise CalibrationArtifactError(
                "Calibration artifact belongs to a different model."
            )
        if artifact.get("model_revision") != model_identity.get("model_revision"):
            raise CalibrationArtifactError(
                "Calibration artifact does not match the exact model checkpoint revision."
            )
        if artifact.get("backbone_model") != model_identity.get("backbone_model"):
            raise CalibrationArtifactError(
                "Calibration artifact was produced with a different backbone model."
            )
        if artifact.get("backbone_revision") != model_identity.get("backbone_revision"):
            raise CalibrationArtifactError(
                "Calibration artifact was produced with a different backbone revision."
            )
        if artifact.get("head_sha256") != model_identity.get("head_sha256"):
            raise CalibrationArtifactError(
                "Calibration artifact was produced with a different classification head."
            )
        if artifact.get("preprocessing_version") != model_identity.get("preprocessing_version"):
            raise CalibrationArtifactError(
                "Calibration artifact was produced with a different preprocessing version."
            )
        artifact_labels = artifact.get("label_mapping")
        runtime_labels = model_identity.get("label_mapping")
        if artifact_labels is not None and runtime_labels is not None:
            normalized_artifact = {str(k): v for k, v in artifact_labels.items()}
            normalized_runtime = {str(k): v for k, v in runtime_labels.items()}
            if normalized_artifact != normalized_runtime:
                raise CalibrationArtifactError(
                    "Calibration artifact was produced with a different label mapping."
                )
        artifact_infer = artifact.get("inference_config")
        runtime_infer = model_identity.get("inference_config")
        if artifact_infer is not None and runtime_infer is not None:
            if artifact_infer != runtime_infer:
                raise CalibrationArtifactError(
                    "Calibration artifact was produced with different inference configuration."
                )

    # ── Validation status ────────────────────────────────────────
    if artifact["validation_split"] != "validation" or artifact["validated"] is not True:
        raise CalibrationArtifactError(
            "Only explicitly validated validation-split artifacts are accepted."
        )

    # ── Manifest fingerprint ─────────────────────────────────────
    manifest = artifact["validation_manifest_sha256"]
    if not isinstance(manifest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", manifest):
        raise CalibrationArtifactError("Calibration artifact has no valid manifest fingerprint.")

    # ── Class counts ─────────────────────────────────────────────
    counts = artifact["class_counts"]
    if not isinstance(counts, dict) or any(
        not isinstance(counts.get(k), int) or counts[k] < 30
        for k in ("real", "ai_generated")
    ):
        raise CalibrationArtifactError(
            "At least 30 reviewed examples per class are required for calibration use."
        )
    if not isinstance(artifact["sample_count"], int) or artifact["sample_count"] != counts["real"] + counts["ai_generated"]:
        raise CalibrationArtifactError("Calibration sample counts are inconsistent.")

    # ── Generation families ──────────────────────────────────────
    families = artifact["ai_generation_families"]
    if not isinstance(families, list) or len(set(families)) < 2:
        raise CalibrationArtifactError(
            "Calibration data must document multiple AI-generation families."
        )

    # ── Temperature & thresholds ─────────────────────────────────
    try:
        temperature = float(artifact["temperature"])
        ai_threshold = float(artifact["thresholds"]["ai_generated_min"])
        authentic_threshold = float(artifact["thresholds"]["authentic_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationArtifactError("Calibration parameters are invalid.") from exc

    if not math.isfinite(temperature) or temperature <= 0:
        raise CalibrationArtifactError("Temperature must be finite and greater than zero.")
    if not (0 <= authentic_threshold < ai_threshold <= 1):
        raise CalibrationArtifactError(
            "Decision thresholds must define a non-empty uncertainty band."
        )

    # ── Diagnostic metrics ───────────────────────────────────────
    for key in ("metrics_before", "metrics_after"):
        metrics = artifact[key]
        if not isinstance(metrics, dict) or any(
            k not in metrics for k in ("nll", "brier", "ece")
        ):
            raise CalibrationArtifactError(
                "Calibration diagnostics must include NLL, Brier score, and ECE."
            )
        if any(
            not isinstance(metrics[k], (float, int)) or not math.isfinite(metrics[k])
            for k in ("nll", "brier", "ece")
        ):
            raise CalibrationArtifactError(
                "Calibration diagnostics must be finite numeric values."
            )

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
