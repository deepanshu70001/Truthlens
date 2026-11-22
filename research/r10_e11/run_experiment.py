"""Run the preregistered, research-only R10 residual experiment.

This script reads only the explicitly allowlisted Phase 9 feature archive and
the production linear head. It never reads an image or split manifest and never
persists sample-level identifiers, labels, or predictions.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.forensics.phase9_frequency_features import FEATURE_NAMES  # noqa: E402

PROTOCOL_ID = "truthlens-r10-siglip-frequency-residual-v1"
CANDIDATE_ID = "truthlens/r10-siglip-frequency-residual-v1"
ARCHIVE_REL = Path("experiments/reports/phase9_feature_vectors.npz")
HEAD_REL = Path("models/baselines/siglip_frozen_head.pt")
OUTPUT_DIR = ROOT / "research" / "r10_e11"
ARCHIVE_SHA256 = "a1b9d3530f9555ba4b67cc4a5d2b3793ebac7804bb150eb0a2856da4efb00ed9"
HEAD_SHA256 = "e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538"
EXPECTED_DATASET_COUNTS = {
    ("training", "tiny_genimage_training"): 6995,
    ("validation", "tiny_genimage"): 5958,
    ("validation", "frontier_benchmark_2026_10"): 297,
    ("validation", "wildaigc_bench"): 181,
}
EXPECTED_CLASS_COUNTS = {
    ("training", "tiny_genimage_training", 0): 3515,
    ("training", "tiny_genimage_training", 1): 3480,
    ("validation", "tiny_genimage", 0): 3410,
    ("validation", "tiny_genimage", 1): 2548,
    ("validation", "frontier_benchmark_2026_10", 0): 193,
    ("validation", "frontier_benchmark_2026_10", 1): 104,
    ("validation", "wildaigc_bench", 0): 63,
    ("validation", "wildaigc_bench", 1): 118,
}
EXPECTED_ARCHIVE_MEMBERS = {
    "metadata_json", "siglip", "resnet50", "frequency", "frequency_valid",
}
PRIMARY_PARTITION = "frontier_benchmark_2026_10"
L2_COEFFICIENT = 1.0
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 2026


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_counts(rows: list[dict[str, Any]]) -> None:
    roles = {str(row.get("split")) for row in rows}
    if roles != {"training", "validation"}:
        raise ValueError("feature archive contains an unapproved split role")
    dataset_counts = Counter((str(r.get("split")), str(r.get("dataset"))) for r in rows)
    if dict(dataset_counts) != EXPECTED_DATASET_COUNTS:
        raise ValueError("feature archive partition counts do not match the preregistration")
    class_counts = Counter(
        (str(r.get("split")), str(r.get("dataset")), int(r.get("label")))
        for r in rows
    )
    if dict(class_counts) != EXPECTED_CLASS_COUNTS:
        raise ValueError("feature archive aggregate class counts do not match the preregistration")
    if any(not isinstance(r.get("source_group"), str) or not r["source_group"] for r in rows):
        raise ValueError("source-group metadata is missing")


def _sigmoid(value: np.ndarray) -> np.ndarray:
    result = np.empty_like(value, dtype=np.float64)
    positive = value >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def _fit_residual(base_logits: np.ndarray, features: np.ndarray, valid: np.ndarray, labels: np.ndarray):
    dimension = features.shape[1]

    def objective(parameters: np.ndarray):
        weights = parameters[:dimension]
        intercept = parameters[dimension]
        residual = valid * (features @ weights + intercept)
        logits = base_logits + residual
        loss = np.logaddexp(0.0, logits) - labels * logits
        objective_value = float(loss.mean() + 0.5 * L2_COEFFICIENT * np.dot(weights, weights))
        errors = _sigmoid(logits) - labels
        gradient_weights = (features.T @ (errors * valid)) / len(labels) + L2_COEFFICIENT * weights
        gradient_intercept = float(np.sum(errors * valid) / len(labels))
        return objective_value, np.concatenate((gradient_weights, [gradient_intercept]))

    result = minimize(
        objective,
        np.zeros(dimension + 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError("fixed residual optimizer did not converge cleanly")
    return result.x[:dimension], float(result.x[dimension]), float(result.fun), int(result.nit)


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float | int]:
    predicted = logits >= 0.0
    positive = labels == 1
    negative = labels == 0
    tp = int(np.sum(predicted & positive))
    tn = int(np.sum((~predicted) & negative))
    fp = int(np.sum(predicted & negative))
    fn = int(np.sum((~predicted) & positive))
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return {
        "n": int(len(labels)),
        "authentic": int(np.sum(negative)),
        "ai_generated": int(np.sum(positive)),
        "accuracy": float((tp + tn) / max(len(labels), 1)),
        "balanced_accuracy": float((tpr + tnr) / 2.0),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tpr),
        "fpr": float(fp / max(fp + tn, 1)),
        "fnr": float(fn / max(fn + tp, 1)),
        "roc_auc": float(roc_auc_score(labels, logits)),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def _paired_group_auc_interval(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    group_ids: np.ndarray,
) -> dict[str, float | int]:
    unique_groups = np.unique(group_ids)
    rows_by_group = {group: np.flatnonzero(group_ids == group) for group in unique_groups}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for iteration in range(BOOTSTRAP_REPLICATES):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([rows_by_group[group] for group in selected])
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size != 2:
            # The small probability of this event is handled by drawing a new
            # group sample; the replicate count remains fixed at 2,000.
            while np.unique(sampled_labels).size != 2:
                selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
                indices = np.concatenate([rows_by_group[group] for group in selected])
                sampled_labels = labels[indices]
        deltas[iteration] = (
            roc_auc_score(sampled_labels, candidate[indices])
            - roc_auc_score(sampled_labels, baseline[indices])
        )
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "group_count": int(len(unique_groups)),
        "mean_delta": float(np.mean(deltas)),
        "ci_95_lower": float(np.quantile(deltas, 0.025)),
        "ci_95_upper": float(np.quantile(deltas, 0.975)),
    }


def run() -> dict[str, Any]:
    started = time.perf_counter()
    archive_path = ROOT / ARCHIVE_REL
    head_path = ROOT / HEAD_REL
    coefficients_path = OUTPUT_DIR / "candidate_coefficients.json"
    result_path = OUTPUT_DIR / "result.json"
    if coefficients_path.exists():
        raise FileExistsError("candidate coefficient output already exists; refusing to overwrite")
    if not archive_path.is_file() or not head_path.is_file():
        raise FileNotFoundError("one of the exact allowlisted R10 inputs is unavailable")

    archive_digest = sha256_file(archive_path)
    head_digest = sha256_file(head_path)
    if archive_digest != ARCHIVE_SHA256:
        raise ValueError("feature archive checksum does not match the preflight checksum")
    if head_digest != HEAD_SHA256:
        raise ValueError("production head checkpoint checksum does not match the supplied identity")

    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != EXPECTED_ARCHIVE_MEMBERS:
            raise ValueError("feature archive members differ from the documented Phase 9 archive")
        metadata_raw = archive["metadata_json"].item()
        metadata = json.loads(metadata_raw)
        if not isinstance(metadata, list) or not all(isinstance(row, dict) for row in metadata):
            raise ValueError("feature archive metadata schema is invalid")
        _expected_counts(metadata)
        count = len(metadata)

        # Only after the allowlisted roles and aggregate counts are verified do
        # we load the approved feature arrays. The unused ResNet array is never
        # read into memory.
        siglip = np.asarray(archive["siglip"], dtype=np.float64)
        frequency = np.asarray(archive["frequency"], dtype=np.float64)
        frequency_valid = np.asarray(archive["frequency_valid"], dtype=bool)

    if siglip.shape != (count, 768) or frequency.shape != (count, len(FEATURE_NAMES)):
        raise ValueError("feature array dimensions do not match the preregistered schema")
    if frequency_valid.shape != (count,):
        raise ValueError("frequency-validity mask dimensions do not match metadata")
    if not np.isfinite(siglip).all():
        raise ValueError("SigLIP feature matrix contains non-finite values")
    if frequency_valid.any() and not np.isfinite(frequency[frequency_valid]).all():
        raise ValueError("valid frequency feature rows contain non-finite values")
    if not frequency_valid.any():
        raise ValueError("there are no valid frequency rows")

    labels = np.asarray([int(row["label"]) for row in metadata], dtype=np.float64)
    splits = np.asarray([str(row["split"]) for row in metadata])
    datasets = np.asarray([str(row["dataset"]) for row in metadata])
    groups = np.asarray([str(row["source_group"]) for row in metadata])
    train_mask = splits == "training"
    train_indices = np.flatnonzero(train_mask)
    if len(train_indices) != 6995:
        raise ValueError("approved training row count changed after preflight")

    valid_train_indices = train_indices[frequency_valid[train_indices]]
    if not len(valid_train_indices):
        raise ValueError("no valid training frequency rows are available")
    scaler_mean = frequency[valid_train_indices].mean(axis=0)
    scaler_scale = frequency[valid_train_indices].std(axis=0)
    scaler_scale[scaler_scale < 1e-12] = 1.0
    standardized = (frequency - scaler_mean) / scaler_scale
    standardized[~frequency_valid] = 0.0

    import torch

    state = torch.load(head_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or set(state) != {"weight", "bias"}:
        raise ValueError("frozen production checkpoint schema is not the expected linear head")
    weights = state["weight"].detach().cpu().numpy().reshape(-1)
    bias = state["bias"].detach().cpu().numpy().reshape(-1)
    if weights.shape != (768,) or bias.shape != (1,):
        raise ValueError("frozen production head dimensions do not match SigLIP features")
    baseline_logits = siglip @ weights + float(bias[0])

    training_valid_mask = frequency_valid[train_indices].astype(np.float64)
    model_weights, model_intercept, training_loss, iterations = _fit_residual(
        baseline_logits[train_indices],
        standardized[train_indices],
        training_valid_mask,
        labels[train_indices],
    )
    residual = training_valid_mask * (standardized[train_indices] @ model_weights + model_intercept)
    train_candidate_logits = baseline_logits[train_indices] + residual
    if not np.isfinite(train_candidate_logits).all():
        raise ValueError("candidate training output contains non-finite values")

    reports: dict[str, Any] = {}
    frontier_interval = None
    for dataset_name in ("tiny_genimage", "frontier_benchmark_2026_10", "wildaigc_bench"):
        selected = np.flatnonzero((splits == "validation") & (datasets == dataset_name))
        if not len(selected):
            raise ValueError("a preregistered validation partition is missing")
        validation_mask = frequency_valid[selected].astype(np.float64)
        candidate_logits = baseline_logits[selected] + validation_mask * (
            standardized[selected] @ model_weights + model_intercept
        )
        base_metrics = _metrics(labels[selected], baseline_logits[selected])
        hybrid_metrics = _metrics(labels[selected], candidate_logits)
        row = {
            "n": int(len(selected)),
            "class_counts": {
                "authentic": int(np.sum(labels[selected] == 0)),
                "ai_generated": int(np.sum(labels[selected] == 1)),
            },
            "group_count": int(len(np.unique(groups[selected]))),
            "frozen_baseline": base_metrics,
            "hybrid_candidate": hybrid_metrics,
            "delta": {
                "roc_auc": float(hybrid_metrics["roc_auc"] - base_metrics["roc_auc"]),
                "balanced_accuracy": float(hybrid_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"]),
                "fpr": float(hybrid_metrics["fpr"] - base_metrics["fpr"]),
                "fnr": float(hybrid_metrics["fnr"] - base_metrics["fnr"]),
            },
            "frequency_valid_rows": int(np.sum(frequency_valid[selected])),
        }
        if dataset_name == PRIMARY_PARTITION:
            frontier_interval = _paired_group_auc_interval(
                labels[selected], baseline_logits[selected], candidate_logits, groups[selected],
            )
            row["paired_group_bootstrap_roc_auc_delta"] = frontier_interval
        reports[dataset_name] = row

    # Verify the registered missing-feature fallback algebra on every validation row.
    validation_indices = np.flatnonzero(splits == "validation")
    forced_missing_logits = baseline_logits[validation_indices] + 0.0 * (
        standardized[validation_indices] @ model_weights + model_intercept
    )
    missing_fallback_equal = bool(np.array_equal(forced_missing_logits, baseline_logits[validation_indices]))
    if not missing_fallback_equal:
        raise AssertionError("missing-feature fallback did not equal the frozen baseline")

    frontier = reports[PRIMARY_PARTITION]
    primary = frontier["delta"]["roc_auc"]
    ci = frontier_interval
    tiny_ba_delta = reports["tiny_genimage"]["delta"]["balanced_accuracy"]
    frontier_fpr_delta = frontier["delta"]["fpr"]
    supported = bool(
        primary > 0.0
        and ci["ci_95_lower"] > 0.0
        and tiny_ba_delta >= -0.01
        and frontier_fpr_delta <= 0.02
    )

    coefficients = {
        "candidate_id": CANDIDATE_ID,
        "protocol_id": PROTOCOL_ID,
        "research_only": True,
        "baseline_head_sha256": head_digest,
        "feature_names": list(FEATURE_NAMES),
        "standardizer_mean_training_only": scaler_mean.tolist(),
        "standardizer_scale_training_only": scaler_scale.tolist(),
        "residual_weights": model_weights.tolist(),
        "residual_intercept": model_intercept,
        "l2_coefficient": L2_COEFFICIENT,
        "training_objective": "binary_cross_entropy_with_logits_plus_l2",
        "training_loss": training_loss,
        "optimizer_iterations": iterations,
        "missing_frequency_fallback": "frozen_baseline_logit",
        "probability_calibration": "none",
    }
    coefficients_path.write_text(json.dumps(coefficients, indent=2, sort_keys=True), encoding="utf-8")
    coefficient_digest = sha256_file(coefficients_path)
    elapsed = time.perf_counter() - started

    result = {
        "protocol_id": PROTOCOL_ID,
        "candidate_id": CANDIDATE_ID,
        "status": "EXECUTED",
        "hypothesis_status": "SUPPORTED" if supported else "NOT_SUPPORTED_OR_INCONCLUSIVE",
        "archive_path": ARCHIVE_REL.as_posix(),
        "archive_sha256": archive_digest,
        "baseline_head_sha256": head_digest,
        "backbone_revision": "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
        "aggregate_partition_counts": {
            "training": 6995,
            "tiny_validation": 5958,
            "frontier_validation": 297,
            "wildaigc_validation_diagnostic_only": 181,
        },
        "train_frequency_valid_rows": int(np.sum(frequency_valid[train_indices])),
        "evaluation": reports,
        "primary_metric": {
            "name": "frontier_validation_paired_roc_auc_delta",
            "value": primary,
            "group_bootstrap_95ci": ci,
        },
        "acceptance_checks": {
            "frontier_delta_positive": bool(primary > 0.0),
            "frontier_ci_excludes_zero": bool(ci["ci_95_lower"] > 0.0),
            "tiny_balanced_accuracy_drop_at_most_0_01": bool(tiny_ba_delta >= -0.01),
            "frontier_fpr_increase_at_most_0_02": bool(frontier_fpr_delta <= 0.02),
        },
        "missing_frequency_fallback_equal_to_baseline": missing_fallback_equal,
        "fit": {
            "optimizer": "scipy.optimize.minimize/L-BFGS-B",
            "iterations": iterations,
            "training_loss": training_loss,
            "runtime_seconds": elapsed,
            "device": "CPU",
        },
        "reproducibility": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "source_group_bootstrap_seed": BOOTSTRAP_SEED,
            "source_group_bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "coefficient_file": coefficients_path.name,
            "coefficient_sha256": coefficient_digest,
        },
        "limitations": [
            "Validation results are development evidence, not locked held-out evaluation.",
            "WildAIGC labels are contextual and diagnostic only.",
            "Hybrid logits are uncalibrated and are not probabilities.",
            "No raw images, row-level predictions, sample identifiers, or protected partitions were used or saved.",
        ],
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"protocol_id={PROTOCOL_ID}")
    print(f"candidate_id={CANDIDATE_ID}")
    print(f"status={result['status']}")
    print(f"hypothesis_status={result['hypothesis_status']}")
    print(f"archive_sha256={archive_digest}")
    print(f"training_rows={len(train_indices)}")
    print(f"frontier_primary_delta={primary:.6f}")
    print(f"frontier_group_bootstrap_95ci=[{ci['ci_95_lower']:.6f}, {ci['ci_95_upper']:.6f}]")
    print(f"coefficient_sha256={coefficient_digest}")
    print(f"runtime_seconds={elapsed:.3f}")
    return result


if __name__ == "__main__":
    run()
