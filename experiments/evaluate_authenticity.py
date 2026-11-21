"""
TruthLens Phase 3: Controlled Benchmark Evaluation.
Inference and Raw Prediction Script for Authenticity Detector.

Runs batched evaluation across the three group-aware evaluation split manifests:
1. Tiny-GenImage (experiments/manifests/tiny_genimage_evaluation_splits.csv)
2. Frontier Benchmark (experiments/manifests/frontier_benchmark_2026_10_evaluation_splits.csv)
3. WildAIGC-Bench (experiments/manifests/wildaigc_bench_evaluation_splits.csv)

Strictly verifies model identity before running and records raw logits,
calibrated candidates, and model predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import yaml

# Adjust project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.authenticity.detector import get_detector, AuthenticityDetector  # noqa: E402

# Safe limit for large images
Image.MAX_IMAGE_PIXELS = 250_000_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("evaluate_authenticity")


class BenchmarkDataset(Dataset):
    """PyTorch Dataset for evaluation manifests."""

    def __init__(
        self,
        base_dir: Path,
        manifest_rows: List[Dict[str, str]],
        transform: transforms.Compose,
    ):
        self.base_dir = base_dir
        self.rows = manifest_rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.rows[idx]
        image_path = self.base_dir / row["image_path"]
        ground_truth_binary = 1 if row["label"] == "ai_generated" else 0

        try:
            with Image.open(image_path) as img:
                rgb_img = img.convert("RGB")
                tensor = self.transform(rgb_img)
        except Exception as e:
            log.error("Failed to load image %s: %s", image_path, e)
            # Create a zero tensor fallback if corrupted (none expected)
            tensor = torch.zeros(3, 224, 224, dtype=torch.float32)

        return tensor, ground_truth_binary, row["image_path"]


def verify_model_identity(
    detector: AuthenticityDetector, config_path: Path
) -> Dict[str, Any]:
    """Verify runtime detector identity against Phase 3 evaluation config."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    expected = config.get("evaluation_target", {})
    actual = detector.model_identity()

    errors = []
    if actual.get("model_id") != expected.get("model_id"):
        errors.append(
            f"model_id mismatch: expected '{expected.get('model_id')}', got '{actual.get('model_id')}'"
        )
    if actual.get("head_sha256") != expected.get("expected_head_sha256"):
        errors.append(
            f"head_sha256 mismatch: expected '{expected.get('expected_head_sha256')}', got '{actual.get('head_sha256')}'"
        )
    if actual.get("backbone_model") != expected.get("backbone_model"):
        errors.append(
            f"backbone_model mismatch: expected '{expected.get('backbone_model')}', got '{actual.get('backbone_model')}'"
        )
    if actual.get("backbone_revision") != expected.get("backbone_revision"):
        errors.append(
            f"backbone_revision mismatch: expected '{expected.get('backbone_revision')}', got '{actual.get('backbone_revision')}'"
        )
    if actual.get("preprocessing_version") != expected.get("preprocessing_version"):
        errors.append(
            f"preprocessing_version mismatch: expected '{expected.get('preprocessing_version')}', got '{actual.get('preprocessing_version')}'"
        )

    if errors:
        msg = "Model identity verification failed:\n- " + "\n- ".join(errors)
        log.error(msg)
        raise ValueError(msg)

    log.info("Model identity verified successfully: %s", actual.get("model_id"))
    return actual


def load_manifest_records(manifest_path: Path) -> List[Dict[str, str]]:
    """Load evaluation split manifest, excluding 'excluded' reserve records."""
    records = []
    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("evaluation_role") == "excluded":
                continue
            records.append(row)
    return records


def evaluate_dataset(
    dataset_id: str,
    base_dir: Path,
    manifest_path: Path,
    detector: AuthenticityDetector,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run batched inference on one dataset."""
    records = load_manifest_records(manifest_path)
    log.info("Evaluating dataset '%s': %d records loaded from %s", dataset_id, len(records), manifest_path)

    # Preprocessing transform
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    ds = BenchmarkDataset(base_dir, records, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = detector.get_internal_model()
    model.eval()

    all_raw_logits_fake = []
    all_raw_logits_real = []
    all_raw_margins = []
    all_fake_scores = []
    all_real_scores = []
    all_predictions = []

    t0 = time.time()
    processed_count = 0

    with torch.inference_mode():
        for batch_idx, (tensors, labels_batch, paths_batch) in enumerate(loader):
            tensors = tensors.to(device)
            outputs = model(pixel_values=tensors)
            logits = outputs.logits  # Shape [B, 2]
            probs = torch.softmax(logits, dim=-1)  # Shape [B, 2]

            real_logits = logits[:, 0].cpu().numpy()
            fake_logits = logits[:, 1].cpu().numpy()
            margins = fake_logits - real_logits

            real_probs = probs[:, 0].cpu().numpy()
            fake_probs = probs[:, 1].cpu().numpy()

            for i in range(len(tensors)):
                r_logit = float(real_logits[i])
                f_logit = float(fake_logits[i])
                margin = float(margins[i])
                r_score = float(real_probs[i])
                f_score = float(fake_probs[i])

                pred = "likely_ai_generated" if f_score > r_score else "likely_authentic"

                all_raw_logits_real.append(r_logit)
                all_raw_logits_fake.append(f_logit)
                all_raw_margins.append(margin)
                all_real_scores.append(r_score)
                all_fake_scores.append(f_score)
                all_predictions.append(pred)

            processed_count += len(tensors)
            if (batch_idx + 1) % 25 == 0 or processed_count == len(records):
                elapsed = time.time() - t0
                speed = processed_count / elapsed if elapsed > 0 else 0
                log.info(
                    "[%s] Processed %d/%d images (%.1f img/s)",
                    dataset_id,
                    processed_count,
                    len(records),
                    speed,
                )

    elapsed_total = time.time() - t0
    log.info("[%s] Completed inference in %.2f seconds (%.1f img/s)", dataset_id, elapsed_total, len(records) / elapsed_total)

    # Combine records with model predictions
    results = []
    correct_count = 0
    role_counts: Dict[str, Dict[str, int]] = {}

    for i, row in enumerate(records):
        gt_binary = 1 if row["label"] == "ai_generated" else 0
        pred_label = "ai_generated" if all_predictions[i] == "likely_ai_generated" else "likely_authentic"
        is_corr = (pred_label == row["label"])
        if is_corr:
            correct_count += 1

        role = row["evaluation_role"]
        if role not in role_counts:
            role_counts[role] = {"total": 0, "correct": 0, "ai_generated": 0, "likely_authentic": 0}
        role_counts[role]["total"] += 1
        if is_corr:
            role_counts[role]["correct"] += 1
        role_counts[role][row["label"]] += 1

        res_row = {
            "image_path": row["image_path"],
            "label": row["label"],
            "ground_truth_binary": gt_binary,
            "evaluation_role": row["evaluation_role"],
            "group_id": row["group_id"],
            "generator": row["generator"],
            "source_url": row.get("source_url", ""),
            "license_ref": row.get("license_ref", ""),
            "sha256": row["sha256"],
            "raw_logit_fake": round(all_raw_logits_fake[i], 6),
            "raw_logit_real": round(all_raw_logits_real[i], 6),
            "raw_margin": round(all_raw_margins[i], 6),
            "authentic_score": round(all_real_scores[i], 6),
            "ai_generated_score": round(all_fake_scores[i], 6),
            "model_prediction": all_predictions[i],
            "is_correct": 1 if is_corr else 0,
            "annotation_notes": row.get("annotation_notes", ""),
        }
        results.append(res_row)

    dataset_summary = {
        "dataset_id": dataset_id,
        "total_evaluated": len(records),
        "overall_raw_accuracy": round(correct_count / len(records), 4) if records else 0.0,
        "total_runtime_seconds": round(elapsed_total, 2),
        "images_per_second": round(len(records) / elapsed_total, 1) if elapsed_total > 0 else 0.0,
        "role_breakdown": {
            r: {
                "total": d["total"],
                "accuracy": round(d["correct"] / d["total"], 4) if d["total"] > 0 else 0.0,
                "class_counts": {
                    "ai_generated": d["ai_generated"],
                    "likely_authentic": d["likely_authentic"],
                },
            }
            for r, d in role_counts.items()
        },
    }

    return results, dataset_summary


def save_predictions_csv(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Save prediction rows to CSV."""
    if not results:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    log.info("Saved %d predictions to %s", len(results), output_path)


def main():
    parser = argparse.ArgumentParser(description="TruthLens Phase 3 Authenticity Evaluation")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "configs" / "phase3_evaluation_config.yaml",
        help="Path to Phase 3 evaluation config YAML",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for model inference (default: 64)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device (cuda or cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "phase3",
        help="Directory to save evaluation predictions and results",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["tiny_genimage", "frontier_benchmark_2026_10", "wildaigc_bench"],
        help="Datasets to evaluate",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize detector & verify model identity contract
    detector = get_detector()
    detector.load(device=args.device)
    model_identity = verify_model_identity(detector, args.config)
    exec_device = detector.get_device()

    dataset_configs = {
        "tiny_genimage": {
            "base_dir": PROJECT_ROOT / "datasets" / "external" / "tiny_genimage",
            "manifest": PROJECT_ROOT / "experiments" / "manifests" / "tiny_genimage_evaluation_splits.csv",
            "output_csv": args.output_dir / "tiny_genimage_predictions.csv",
        },
        "frontier_benchmark_2026_10": {
            "base_dir": PROJECT_ROOT / "datasets" / "external" / "ai_image_detection_benchmark_2026_10",
            "manifest": PROJECT_ROOT / "experiments" / "manifests" / "frontier_benchmark_2026_10_evaluation_splits.csv",
            "output_csv": args.output_dir / "frontier_predictions.csv",
        },
        "wildaigc_bench": {
            "base_dir": PROJECT_ROOT / "datasets" / "external" / "wildaigc_bench",
            "manifest": PROJECT_ROOT / "experiments" / "manifests" / "wildaigc_bench_evaluation_splits.csv",
            "output_csv": args.output_dir / "wildaigc_predictions.csv",
        },
    }

    all_summaries = {}
    total_images = 0
    t_start = time.time()

    for ds_id in args.datasets:
        if ds_id not in dataset_configs:
            log.warning("Unknown dataset %s, skipping", ds_id)
            continue
        cfg = dataset_configs[ds_id]
        results, summary = evaluate_dataset(
            dataset_id=ds_id,
            base_dir=cfg["base_dir"],
            manifest_path=cfg["manifest"],
            detector=detector,
            device=exec_device,
            batch_size=args.batch_size,
        )
        save_predictions_csv(results, cfg["output_csv"])
        all_summaries[ds_id] = summary
        total_images += len(results)

    total_time = time.time() - t_start

    final_report = {
        "execution_date": datetime.now(timezone.utc).isoformat(),
        "device": str(exec_device),
        "batch_size": args.batch_size,
        "total_images_evaluated": total_images,
        "total_duration_seconds": round(total_time, 2),
        "overall_images_per_second": round(total_images / total_time, 1) if total_time > 0 else 0.0,
        "model_identity": model_identity,
        "datasets": all_summaries,
    }

    summary_json_path = args.output_dir / "inference_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    log.info("Phase 3 inference complete! Total %d images in %.1fs. Summary saved to %s", total_images, total_time, summary_json_path)


if __name__ == "__main__":
    main()
