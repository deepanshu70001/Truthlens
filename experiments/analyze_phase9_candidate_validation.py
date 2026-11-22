"""Validation-only error analysis for existing Phase 7 checkpoints.

This script never selects or trains a candidate and loads only the validation
roles. It must not be pointed at locked test rows.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from transformers import SiglipVisionModel

from experiments.analyze_phase9 import (
    BOOTSTRAPS,
    SPECS,
    AuthorizedImageDataset,
    canonical_transform,
    grouped_bootstrap_ci,
    load_authorized_records,
    metric_values,
    per_group_metrics,
    sha256,
    write_json,
)
from experiments.training_utils import SiglipClassifierModel


def make_candidate(name: str, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if name in {"s1_siglip_linear_probe", "s2_siglip_partial_ft", "a1_siglip_no_aug_ablation"}:
        checkpoint_path = ROOT / "experiments/checkpoints/phase7" / name / "best_checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        backbone = SiglipVisionModel.from_pretrained(
            "google/siglip-base-patch16-224",
            revision="7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
            cache_dir=str(ROOT / "models/cache"),
            local_files_only=True,
        )
        model = SiglipClassifierModel(backbone, nn.Linear(768, 1))
        model.load_state_dict(checkpoint["model_state_dict"])
        model_id = checkpoint["model_id"]
        backbone_name = "siglip"
    else:
        checkpoint_path = ROOT / "experiments/checkpoints/phase7" / name / "best_checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = tv_models.resnet50(weights=None)
        model.fc = nn.Linear(2048, 1)
        model.load_state_dict(checkpoint["model_state_dict"])
        model_id = checkpoint["model_id"]
        backbone_name = "resnet50"
    model.eval().to(device)
    identity = {"model_id": model_id, "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha256(checkpoint_path), "backbone": backbone_name}
    return model, identity


def predict(model: torch.nn.Module, records: list[dict[str, Any]], backbone: str, device: torch.device) -> tuple[np.ndarray, float]:
    loader = torch.utils.data.DataLoader(
        AuthorizedImageDataset(records, canonical_transform(backbone)), batch_size=32, shuffle=False, num_workers=0
    )
    output = []
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for tensors, _ in loader:
            tensors = tensors.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(pixel_values=tensors) if backbone == "siglip" else model(tensors)
            output.append(torch.sigmoid(logits.float().reshape(-1)).cpu().numpy())
    return np.concatenate(output), time.perf_counter() - start


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = [
        "s1_siglip_linear_probe", "s2_siglip_partial_ft", "r1_resnet50_full_ft",
        "r2_resnet50_last_stage", "a1_siglip_no_aug_ablation",
    ]
    all_records = {spec["name"]: load_authorized_records(spec) for spec in SPECS}
    predictions: dict[str, dict[str, np.ndarray]] = {name: {} for name in all_records}
    runtimes = {}
    identities = {}
    for candidate in candidates:
        model, identity = make_candidate(candidate, device)
        identities[candidate] = identity
        total_seconds = 0.0
        for dataset_name, records in all_records.items():
            scores, elapsed = predict(model, records, identity["backbone"], device)
            predictions[dataset_name][candidate] = scores
            total_seconds += elapsed
        runtimes[candidate] = {"validation_inference_seconds": total_seconds, "seconds_per_image": total_seconds / sum(map(len, all_records.values()))}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report: dict[str, Any] = {
        "scope": "existing Phase 7 research checkpoints evaluated on validation rows only; no training or model selection",
        "validation_counts": {name: len(records) for name, records in all_records.items()},
        "candidates": {},
        "runtime": {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0, "by_candidate": runtimes},
        "wild_aigc_caveat": "WildAIGC labels are contextual and are not objective provenance labels.",
    }
    for candidate in candidates:
        report["candidates"][candidate] = {"identity": identities[candidate], "validation": {}}
        for dataset_name, records in all_records.items():
            scores = predictions[dataset_name][candidate]
            y = np.asarray([row["label"] for row in records], dtype=int)
            groups = np.asarray([row["source_group"] for row in records])
            report["candidates"][candidate]["validation"][dataset_name] = {
                "sample_count": len(records), "class_counts": {"authentic": int(np.sum(y == 0)), "ai_generated": int(np.sum(y == 1))},
                "metrics": metric_values(y, scores),
                "confidence_intervals": grouped_bootstrap_ci(y, scores, groups, 900 + candidates.index(candidate) * 10),
                "per_generator": per_group_metrics(records, scores, "generator"),
                "per_source": per_group_metrics(records, scores, "source"),
            }
    output_json = ROOT / "experiments/reports/phase9_candidate_validation_errors.json"
    write_json(output_json, report)

    # Merge exact checkpoint errors into the Phase 9 failure-case artifact while
    # retaining the frozen-feature probe records already generated there.
    failure_path = ROOT / "experiments/reports/phase9_failure_cases.csv"
    existing = list(csv.DictReader(failure_path.open("r", encoding="utf-8", newline="")))
    for row in existing:
        row["model_id"] = "phase9/siglip_frozen_feature_probe"
    fields = ["model_id", "sample_id", "dataset", "split", "label", "prediction", "score_ai", "error_type", "generator_or_source", "source_group", "width", "height", "aspect_ratio", "color_mode", "encoding", "file_size_bytes", "load_failure"]
    with failure_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)
        for dataset_name, records in all_records.items():
            for candidate in candidates:
                scores = predictions[dataset_name][candidate]
                for record, score in zip(records, scores):
                    label, prediction = int(record["label"]), int(score >= 0.5)
                    if label == prediction:
                        continue
                    writer.writerow({
                        "model_id": identities[candidate]["model_id"], "sample_id": record["sample_id"], "dataset": dataset_name,
                        "split": "validation", "label": label, "prediction": prediction, "score_ai": f"{score:.8f}",
                        "error_type": "false_positive" if label == 0 else "false_negative", "generator_or_source": record["generator"],
                        "source_group": record["source_group"], "width": record.get("width", 0), "height": record.get("height", 0),
                        "aspect_ratio": f"{record.get('width', 0) / max(record.get('height', 0), 1):.6f}",
                        "color_mode": record.get("color_mode", "unknown"), "encoding": record.get("encoding", "unknown"),
                        "file_size_bytes": record.get("file_size_bytes", 0), "load_failure": record.get("load_failure", ""),
                    })

    generator_path = ROOT / "experiments/reports/phase9_generator_error_analysis.json"
    generator_report = json.loads(generator_path.read_text(encoding="utf-8"))
    generator_report["existing_phase7_candidate_validation_diagnostics"] = report["candidates"]
    generator_report["candidate_inference_runtime"] = report["runtime"]
    write_json(generator_path, generator_report)
    print(json.dumps({"candidate_validation_errors": output_json.relative_to(ROOT).as_posix(), "candidate_count": len(candidates), "validation_counts": report["validation_counts"], "runtime": report["runtime"]}, indent=2))


if __name__ == "__main__":
    main()
