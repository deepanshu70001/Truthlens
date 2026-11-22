"""Apply the unchanged production linear head to cached validation embeddings."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_phase9 import ROOT, SPECS, TRAIN, grouped_bootstrap_ci, load_authorized_records, metric_values, per_group_metrics, sha256, write_json


def main() -> None:
    started = time.perf_counter()
    archive_path = ROOT / "experiments/reports/phase9_feature_vectors.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        embeddings = archive["siglip"]
    records = [load_authorized_records(spec) for spec in SPECS]
    flat_records = [row for dataset in records for row in dataset]
    offset = len(load_authorized_records(TRAIN))
    if len(flat_records) != len(metadata) - offset or any(row["sample_id"] != meta["sample_id"] for row, meta in zip(flat_records, metadata[offset:])):
        raise RuntimeError("Cached SigLIP vectors do not align with the authorized validation records")
    weights = torch.load(ROOT / "models/baselines/siglip_frozen_head.pt", map_location="cpu", weights_only=True)
    head = nn.Linear(768, 1)
    head.load_state_dict(weights)
    scores = torch.sigmoid(torch.from_numpy(embeddings[offset:]) @ head.weight.T + head.bias).detach().numpy().reshape(-1)
    pointer = 0
    metadata_pointer = offset
    validation = {}
    for spec, dataset_records in zip(SPECS, records):
        size = len(dataset_records)
        y = np.asarray([row["label"] for row in dataset_records], dtype=int)
        group = np.asarray([row["source_group"] for row in dataset_records])
        local_scores = scores[pointer : pointer + size]
        pointer += size
        validation[spec["name"]] = {
            "sample_count": size,
            "class_counts": {"authentic": int(np.sum(y == 0)), "ai_generated": int(np.sum(y == 1))},
            "metrics": metric_values(y, local_scores),
            "confidence_intervals": grouped_bootstrap_ci(y, local_scores, group, 950),
            "per_generator": per_group_metrics(dataset_records, local_scores, "generator"),
            "per_source": per_group_metrics(dataset_records, local_scores, "source"),
        }
        for row, score, meta in zip(dataset_records, local_scores, metadata[metadata_pointer : metadata_pointer + size]):
            row["score"] = float(score)
            for key in ("width", "height", "aspect_ratio", "color_mode", "encoding"):
                row[key] = meta.get(key, "unknown")
            row["file_size_bytes"] = 0
        metadata_pointer += size

    candidate_path = ROOT / "experiments/reports/phase9_candidate_validation_errors.json"
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    base_id = "truthlens/siglip-cifake-frozen-head"
    candidates["candidates"]["production_baseline"] = {
        "identity": {"model_id": base_id, "head": "models/baselines/siglip_frozen_head.pt", "head_sha256": sha256(ROOT / "models/baselines/siglip_frozen_head.pt"), "backbone": "SigLIP-base frozen at pinned revision"},
        "validation": validation,
    }
    candidates["runtime"]["baseline_head_application_seconds"] = time.perf_counter() - started
    write_json(candidate_path, candidates)

    failure_path = ROOT / "experiments/reports/phase9_failure_cases.csv"
    existing = list(csv.DictReader(failure_path.open("r", encoding="utf-8", newline="")))
    fields = ["model_id", "sample_id", "dataset", "split", "label", "prediction", "score_ai", "error_type", "generator_or_source", "source_group", "width", "height", "aspect_ratio", "color_mode", "encoding", "file_size_bytes", "load_failure"]
    with failure_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)
        for spec, dataset_records in zip(SPECS, records):
            for row in dataset_records:
                score = row["score"]
                prediction = int(score >= 0.5)
                if prediction == row["label"]:
                    continue
                writer.writerow({
                    "model_id": base_id, "sample_id": row["sample_id"], "dataset": spec["name"], "split": "validation",
                    "label": row["label"], "prediction": prediction, "score_ai": f"{score:.8f}",
                    "error_type": "false_positive" if row["label"] == 0 else "false_negative",
                    "generator_or_source": row["generator"], "source_group": row["source_group"],
                    "width": row.get("width", 0), "height": row.get("height", 0), "aspect_ratio": row.get("aspect_ratio", ""), "color_mode": row.get("color_mode", ""), "encoding": row.get("encoding", ""), "file_size_bytes": row.get("file_size_bytes", 0), "load_failure": "",
                })
    generator_path = ROOT / "experiments/reports/phase9_generator_error_analysis.json"
    generator_report = json.loads(generator_path.read_text(encoding="utf-8"))
    generator_report["production_baseline_validation_diagnostics"] = validation
    write_json(generator_path, generator_report)
    print(json.dumps({"baseline_model_id": base_id, "validation": {name: item["metrics"] for name, item in validation.items()}}, indent=2))


if __name__ == "__main__":
    main()
