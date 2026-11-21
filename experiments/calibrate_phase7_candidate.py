"""
TruthLens Phase 7: Confidence Calibration of Shortcut-Resistant Candidates.

Fits temperature scaling exclusively on calibration partitions:
- Tiny-GenImage calibration (12,048 samples)
- Frontier Benchmark calibration (588 samples)
- WildAIGC-Bench calibration (364 samples)
- Pooled calibration (13,000 samples)

Strictly enforces:
- Zero test data accessed.
- Flags temperature hitting search bounds [0.05, 50.0].
- Serializes to experiments/reports/phase7_calibration.json and phase7_reliability_data.json.
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
from torch.utils.data import DataLoader, Dataset
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
logger = logging.getLogger("calibrate_phase7")


def compute_nll(labels: np.ndarray, logits: np.ndarray, temperature: float) -> float:
    scaled = logits / max(1e-6, temperature)
    loss = labels * np.logaddexp(0, -scaled) + (1.0 - labels) * np.logaddexp(0, scaled)
    return float(np.mean(loss))


def compute_brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def compute_reliability_diagram(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, float, List[Dict[str, Any]]]:
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
    ds = ImageManifestDataset(
        manifest_path,
        base_dir,
        role_filter="calibration",
        transform=get_canonical_transform(),
    )
    assert len(ds) > 0, f"No calibration samples found in {manifest_path}"

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
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

    # 1. Load Primary Candidate S2 Checkpoint
    chkpt_path = Path("experiments/checkpoints/phase7/s2_siglip_partial_ft/best_checkpoint.pt")
    logger.info("Loading selected primary candidate (S2) from %s...", chkpt_path)
    chkpt = torch.load(chkpt_path, map_location="cpu", weights_only=False)

    vision_model = SiglipVisionModel.from_pretrained(
        "google/siglip-base-patch16-224",
        cache_dir="models/cache",
    )
    model = SiglipClassifierModel(vision_model, nn.Linear(768, 1))
    model.load_state_dict(chkpt["model_state_dict"])
    model.to(device)
    model.eval()

    # 2. Calibration Datasets
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
        logger.info("  %s: %d samples, %.1f%% positive", name, len(y_true), float(np.mean(y_true)) * 100)

    pooled_y = np.concatenate(pooled_labels)
    pooled_z = np.concatenate(pooled_logits)
    extracted_data["pooled"] = {"labels": pooled_y, "logits": pooled_z, "paths": []}
    logger.info("Total pooled calibration samples: %d", len(pooled_y))

    # 3. Fit Temperature on Calibration Partitions
    calibration_report = {
        "metadata": {
            "model_id": "truthlens/siglip-base-v2-s2",
            "candidate": "s2_siglip_partial_ft",
            "checkpoint_sha256": compute_file_sha256(chkpt_path),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "method": "temperature_scaling_nll_minimization",
            "bounds": [0.05, 50.0],
            "total_calibration_samples": len(pooled_y),
        },
        "datasets": {},
    }
    reliability_data = {}

    for name, data in extracted_data.items():
        y = data["labels"]
        z = data["logits"]

        t_opt, converged = fit_temperature(y, z)
        hit_lower_bound = bool(t_opt <= 0.051)
        hit_upper_bound = bool(t_opt >= 49.9)

        uncal_probs = 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))
        cal_probs = 1.0 / (1.0 + np.exp(-np.clip(z / t_opt, -700, 700)))

        uncal_nll = round(compute_nll(y, z, 1.0), 5)
        cal_nll = round(compute_nll(y, z, t_opt), 5)
        uncal_brier = round(compute_brier_score(y, uncal_probs), 5)
        cal_brier = round(compute_brier_score(y, cal_probs), 5)

        uncal_ece, uncal_mce, uncal_bins = compute_reliability_diagram(y, uncal_probs)
        cal_ece, cal_mce, cal_bins = compute_reliability_diagram(y, cal_probs)

        status = "CONVERGED" if (converged and not hit_lower_bound and not hit_upper_bound) else "FLAGGED_BOUNDS"
        if hit_upper_bound or hit_lower_bound:
            logger.warning("Dataset %s hit temperature optimization boundary: T = %.4f", name, t_opt)

        calibration_report["datasets"][name] = {
            "sample_count": len(y),
            "positive_ratio": round(float(np.mean(y)), 4),
            "fitted_temperature": t_opt,
            "converged": converged,
            "status": status,
            "hit_bounds": bool(hit_lower_bound or hit_upper_bound),
            "uncalibrated": {
                "nll": uncal_nll,
                "brier_score": uncal_brier,
                "ece": uncal_ece,
                "mce": uncal_mce,
            },
            "calibrated": {
                "nll": cal_nll,
                "brier_score": cal_brier,
                "ece": cal_ece,
                "mce": cal_mce,
                "ece_improvement": round(uncal_ece - cal_ece, 5),
            },
        }

        reliability_data[name] = {
            "sample_count": len(y),
            "fitted_temperature": t_opt,
            "uncalibrated_bins": uncal_bins,
            "calibrated_bins": cal_bins,
        }

        logger.info(
            "Dataset %-28s | T = %6.4f | ECE: %6.4f -> %6.4f | NLL: %6.4f -> %6.4f | Status: %s",
            name, t_opt, uncal_ece, cal_ece, uncal_nll, cal_nll, status,
        )

    # Save reports
    cal_rep_path = Path("experiments/reports/phase7_calibration.json")
    rel_data_path = Path("experiments/reports/phase7_reliability_data.json")
    with open(cal_rep_path, "w", encoding="utf-8") as f:
        json.dump(calibration_report, f, indent=2)
    with open(rel_data_path, "w", encoding="utf-8") as f:
        json.dump(reliability_data, f, indent=2)

    logger.info("Saved Phase 7 calibration report to %s", cal_rep_path)
    logger.info("Saved Phase 7 reliability data to %s", rel_data_path)


if __name__ == "__main__":
    main()
