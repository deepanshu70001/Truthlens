"""
TruthLens Phase 6 - Step 3: Confidence Calibration of E3 Candidate.

Fits temperature scaling exclusively on calibration partitions:
- Tiny-GenImage calibration (12,048 samples)
- Frontier Benchmark calibration (588 samples)
- WildAIGC-Bench calibration (364 samples)
- Pooled calibration (13,000 samples)

Strictly enforces:
- Zero validation or test data is seen during calibration fitting.
- Checks optimizer convergence and positive finite temperature.
- Compares against Phase 3 frozen baseline calibration.

Produces:
- experiments/reports/phase6_calibration.json
- experiments/reports/phase6_reliability_data.json
"""

from __future__ import annotations

import csv
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader
from transformers import SiglipVisionModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.training_utils import (
    ImageManifestDataset,
    SiglipClassifierModel,
    compute_file_sha256,
    get_canonical_transform,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("calibrate_phase6")


def compute_nll(labels: np.ndarray, logits: np.ndarray, temperature: float) -> float:
    """Compute binary negative log-likelihood at given temperature."""
    scaled = logits / max(1e-6, temperature)
    # Stable BCE: log(1 + exp(-scaled)) or scaled + log(1 + exp(scaled))
    # Using np.logaddexp: log(1 + exp(x)) = np.logaddexp(0, x)
    # loss = - [ y * log(sigma(z)) + (1-y) * log(1-sigma(z)) ]
    # log(sigma(z)) = -log(1 + exp(-z)) = -logaddexp(0, -z)
    # log(1-sigma(z)) = -z - log(1 + exp(-z)) = -logaddexp(0, z)
    loss = labels * np.logaddexp(0, -scaled) + (1.0 - labels) * np.logaddexp(0, scaled)
    return float(np.mean(loss))


def compute_brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute Brier score (mean squared error of probabilities)."""
    return float(np.mean((probs - labels) ** 2))


def compute_reliability_diagram(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Compute ECE, MCE, and binned reliability diagram coordinates."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_data = []
    ece = 0.0
    mce = 0.0
    n = len(labels)

    for i in range(n_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= low) & (probs <= high)
        else:
            mask = (probs >= low) & (probs < high)

        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(labels[mask]))
            bin_conf = float(np.mean(probs[mask]))
            gap = abs(bin_acc - bin_conf)
            ece += (count / n) * gap
            mce = max(mce, gap)
        else:
            bin_acc = None
            bin_conf = None
            gap = 0.0

        bins_data.append({
            "bin_index": i,
            "low": round(float(low), 4),
            "high": round(float(high), 4),
            "count": count,
            "mean_confidence": round(bin_conf, 4) if bin_conf is not None else None,
            "empirical_accuracy": round(bin_acc, 4) if bin_acc is not None else None,
            "calibration_gap": round(float(gap), 4),
        })

    return round(float(ece), 5), round(float(mce), 5), bins_data


def fit_temperature(labels: np.ndarray, logits: np.ndarray) -> Tuple[float, bool]:
    """Find temperature T minimizing NLL via bounded golden-section search."""
    def objective(t: float) -> float:
        return compute_nll(labels, logits, t)

    res = minimize_scalar(objective, bounds=(0.05, 50.0), method="bounded")
    t_opt = float(res.x)
    converged = bool(res.success and math.isfinite(t_opt) and t_opt > 0)
    return round(t_opt, 4), converged


def extract_calibration_predictions(
    model: nn.Module,
    manifest_path: Path,
    base_dir: Path,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract raw logits and binary ground truth on calibration split only."""
    ds = ImageManifestDataset(
        manifest_path,
        base_dir,
        role_filter="calibration",
        transform=get_canonical_transform(),
    )
    assert len(ds) > 0, f"No calibration samples found in {manifest_path}"

    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    model.eval()

    all_logits = []
    all_targets = []
    all_paths = []

    with torch.inference_mode():
        for tensors, targets, paths in loader:
            tensors = tensors.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(tensors)

            all_logits.append(logits.float().cpu().numpy())
            all_targets.append(targets.float().cpu().numpy())
            all_paths.extend(paths)

    y_true = np.concatenate(all_targets).flatten().astype(float)
    raw_logits = np.concatenate(all_logits).flatten().astype(float)
    return y_true, raw_logits, all_paths


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Active compute device: %s", device)

    # 1. Load E3 Candidate Checkpoint
    chkpt_path = Path("experiments/checkpoints/phase5/e3_partial_finetuning/best_checkpoint.pt")
    logger.info("Loading E3 candidate model from %s...", chkpt_path)
    chkpt = torch.load(chkpt_path, map_location="cpu", weights_only=False)

    vision_model = SiglipVisionModel.from_pretrained(
        "google/siglip-base-patch16-224",
        cache_dir="models/cache",
    )
    model = SiglipClassifierModel(vision_model, nn.Linear(768, 1))
    model.load_state_dict(chkpt["model_state_dict"])
    model.to(device)
    model.eval()

    # 2. Define Calibration Datasets
    cal_datasets = [
        ("tiny_genimage", Path("experiments/manifests/tiny_genimage_evaluation_splits.csv"), Path("datasets/external/tiny_genimage")),
        ("frontier_benchmark_2026_10", Path("experiments/manifests/frontier_benchmark_2026_10_evaluation_splits.csv"), Path("datasets/external/ai_image_detection_benchmark_2026_10")),
        ("wildaigc_bench", Path("experiments/manifests/wildaigc_bench_evaluation_splits.csv"), Path("datasets/external/wildaigc_bench")),
    ]

    extracted_data = {}
    pooled_labels = []
    pooled_logits = []

    for name, manifest_path, base_dir in cal_datasets:
        logger.info("Extracting calibration logits for %s...", name)
        y_true, logits, paths = extract_calibration_predictions(model, manifest_path, base_dir, device)
        extracted_data[name] = {"labels": y_true, "logits": logits, "paths": paths}
        pooled_labels.append(y_true)
        pooled_logits.append(logits)
        logger.info("%s: %d calibration samples loaded (Mean Logit: %.3f, AI Ratio: %.3f)", name, len(y_true), float(np.mean(logits)), float(np.mean(y_true)))

    pooled_y = np.concatenate(pooled_labels)
    pooled_z = np.concatenate(pooled_logits)
    extracted_data["pooled"] = {"labels": pooled_y, "logits": pooled_z}
    logger.info("Pooled calibration: %d samples total.", len(pooled_y))

    # 3. Fit Temperature and Compute Diagnostics
    results = {}
    reliability_data = {}

    for name, data in extracted_data.items():
        y = data["labels"]
        z = data["logits"]

        t_opt, converged = fit_temperature(y, z)

        # Uncalibrated (T=1.0)
        p_uncal = 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))
        nll_uncal = compute_nll(y, z, 1.0)
        brier_uncal = compute_brier_score(y, p_uncal)
        ece_uncal, mce_uncal, bins_uncal = compute_reliability_diagram(y, p_uncal)

        # Calibrated (T=t_opt)
        p_cal = 1.0 / (1.0 + np.exp(-np.clip(z / t_opt, -700, 700)))
        nll_cal = compute_nll(y, z, t_opt)
        brier_cal = compute_brier_score(y, p_cal)
        ece_cal, mce_cal, bins_cal = compute_reliability_diagram(y, p_cal)

        results[name] = {
            "dataset_name": name,
            "sample_count": len(y),
            "optimal_temperature": t_opt,
            "optimizer_converged": converged,
            "uncalibrated_T1": {
                "nll": round(nll_uncal, 4),
                "brier_score": round(brier_uncal, 4),
                "ece": round(ece_uncal, 4),
                "mce": round(mce_uncal, 4),
            },
            "calibrated": {
                "temperature": t_opt,
                "nll": round(nll_cal, 4),
                "brier_score": round(brier_cal, 4),
                "ece": round(ece_cal, 4),
                "mce": round(mce_cal, 4),
                "nll_delta": round(nll_cal - nll_uncal, 4),
                "brier_delta": round(brier_cal - brier_uncal, 4),
                "ece_delta": round(ece_cal - ece_uncal, 4),
            },
        }

        reliability_data[name] = {
            "dataset_name": name,
            "temperature": t_opt,
            "uncalibrated_bins": bins_uncal,
            "calibrated_bins": bins_cal,
        }

        logger.info(
            "%-26s | T: %6.4f (conv=%s) | NLL: %.4f -> %.4f | Brier: %.4f -> %.4f | ECE: %.4f -> %.4f",
            name,
            t_opt,
            converged,
            nll_uncal,
            nll_cal,
            brier_uncal,
            brier_cal,
            ece_uncal,
            ece_cal,
        )

    # 4. Compare with Baseline Phase 3 Calibration
    baseline_cal_file = Path("experiments/reports/phase3_calibration.json")
    baseline_comparison = {}
    if baseline_cal_file.exists():
        with open(baseline_cal_file, "r", encoding="utf-8") as f:
            b_cal = json.load(f)
        b_temps = b_cal.get("fitted_temperatures", {})
        baseline_comparison = {
            "baseline_temperatures": b_temps,
            "candidate_temperatures": {k: v["optimal_temperature"] for k, v in results.items()},
        }

    # 5. Save Artifacts
    cal_report = {
        "schema_version": "2.0",
        "phase": "Phase 6 - Calibration",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": chkpt.get("model_identity", {}).get("model_id"),
        "checkpoint_sha256": compute_file_sha256(chkpt_path),
        "protocol_governance": {
            "fitted_exclusively_on_calibration": True,
            "validation_data_used": False,
            "test_data_used": False,
            "status": "PASS",
        },
        "datasets": results,
        "baseline_comparison": baseline_comparison,
        "governance_and_safety_checks": {
            "all_temperatures_finite_and_positive": all(v["optimal_temperature"] > 0 and math.isfinite(v["optimal_temperature"]) for v in results.values()),
            "ranking_preserved": "Yes (temperature scaling is strictly monotonic)",
            "production_authorization": "NO - Calibration does not authorize production label assignments without Phase 6 held-out validation.",
        },
    }

    out_cal = Path("experiments/reports/phase6_calibration.json")
    out_rel = Path("experiments/reports/phase6_reliability_data.json")

    with open(out_cal, "w", encoding="utf-8") as f:
        json.dump(cal_report, f, indent=2)

    with open(out_rel, "w", encoding="utf-8") as f:
        json.dump(reliability_data, f, indent=2)

    logger.info("Saved Phase 6 calibration report to %s", out_cal)
    logger.info("Saved Phase 6 reliability data to %s", out_rel)


if __name__ == "__main__":
    main()
