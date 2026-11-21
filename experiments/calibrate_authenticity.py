"""Fit an opt-in SigLIP2 temperature artifact from a reviewed validation split.

This command never evaluates or tunes on the reserved test split. The generated
artifact is not accepted by the application unless the operator confirms that
labels, rights, generator/source metadata, and split isolation were reviewed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.calibration_utils import (  # noqa: E402
    binary_diagnostics,
    fit_temperature,
    probabilities_from_margins,
    select_abstention_thresholds,
)


def read_reviewed_manifest(path: Path) -> tuple[list[dict], list[str], str]:
    errors: list[str] = []
    rows: list[dict] = []
    required = {"image_path", "label", "split", "group_id", "generator", "source_url", "license_ref"}
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            return [], [f"manifest requires columns: {', '.join(sorted(required))}"], digest
        for line, row in enumerate(reader, 2):
            label = row["label"].strip().lower()
            split = row["split"].strip().lower()
            if label not in {"real", "ai_generated"}:
                errors.append(f"line {line}: label must be real or ai_generated")
                continue
            if split not in {"train", "validation", "test"}:
                errors.append(f"line {line}: split must be train, validation, or test")
                continue
            if not all(row[name].strip() for name in ("group_id", "source_url", "license_ref", "generator")):
                errors.append(f"line {line}: group, generator, source and rights references are required")
                continue
            image_path = (path.parent / row["image_path"].strip()).resolve()
            if not image_path.is_file():
                errors.append(f"line {line}: image file is missing")
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
                image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
            except Exception:
                errors.append(f"line {line}: image cannot be decoded")
                continue
            rows.append({**row, "label": label, "split": split, "_path": image_path, "_sha256": image_hash})
    return rows, errors, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-reviewed-validation-data", action="store_true",
                        help="mark the artifact production-eligible after human confirmation of labels, source rights, and split isolation")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()

    rows, errors, manifest_digest = read_reviewed_manifest(args.manifest)
    validation = [row for row in rows if row["split"] == "validation"]
    test = [row for row in rows if row["split"] == "test"]
    train = [row for row in rows if row["split"] == "train"]
    if not train or not validation or not test:
        errors.append("separate non-empty train, calibration/validation, and reserved test splits are required")
    group_splits: dict[str, set[str]] = defaultdict(set)
    hashes_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[row["group_id"].strip()].add(row["split"])
        hashes_by_split[row["split"]].add(row["_sha256"])
    if any(len(splits) > 1 for splits in group_splits.values()):
        errors.append("a group_id crosses dataset splits")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if hashes_by_split[left] & hashes_by_split[right]:
            errors.append(f"exact duplicate image bytes cross {left}/{right} splits")
    counts = Counter(row["label"] for row in validation)
    if counts["real"] < 30 or counts["ai_generated"] < 30:
        errors.append("validation calibration requires at least 30 examples per class")
    generator_families = sorted({row["generator"].strip() for row in validation if row["label"] == "ai_generated"})
    if len(generator_families) < 2:
        errors.append("validation calibration requires at least two documented AI-generation families")
    if errors:
        raise SystemExit("Calibration aborted:\n- " + "\n- ".join(errors))

    from backend.app.config import settings
    from backend.app.services.authenticity.detector import AuthenticityDetector

    detector = AuthenticityDetector()
    detector.load(args.device)
    margins: list[float] = []
    labels: list[int] = []
    for row in validation:
        with Image.open(row["_path"]) as image:
            prediction = detector.detect(image.convert("RGB"))
        margins.append(prediction.raw_logits["fake"] - prediction.raw_logits["real"])
        labels.append(int(row["label"] == "ai_generated"))
    model_identity = detector.model_identity()
    model_revision = model_identity.get("model_revision")
    if not model_revision or model_revision == "unknown":
        raise SystemExit("Calibration aborted: the exact checkpoint revision could not be identified.")

    temperature = fit_temperature(margins, labels)
    before = probabilities_from_margins(margins, 1.0)
    after = probabilities_from_margins(margins, temperature)
    thresholds = select_abstention_thresholds(labels, after)
    artifact = {
        "schema_version": 2,
        "method": "temperature_scaling",
        "model_id": model_identity["model_id"],
        "model_revision": model_identity["model_revision"],
        "backbone_model": model_identity.get("backbone_model"),
        "backbone_revision": model_identity.get("backbone_revision"),
        "head_sha256": model_identity.get("head_sha256"),
        "preprocessing_version": model_identity.get("preprocessing_version"),
        "label_mapping": model_identity.get("label_mapping"),
        "inference_config": model_identity.get("inference_config", {}),
        "validation_manifest_sha256": manifest_digest,
        "validation_split": "validation",
        "sample_count": len(validation),
        "class_counts": {"real": counts["real"], "ai_generated": counts["ai_generated"]},
        "ai_generation_families": generator_families,
        "temperature": temperature,
        "thresholds": {"authentic_max": thresholds["authentic_max"], "ai_generated_min": thresholds["ai_generated_min"]},
        "threshold_selection": {key: value for key, value in thresholds.items() if key not in {"authentic_max", "ai_generated_min"}},
        "metrics_before": binary_diagnostics(labels, before),
        "metrics_after": binary_diagnostics(labels, after),
        "reserved_test_count": len(test),
        "validated": bool(args.confirm_reviewed_validation_data),
        "status": "eligible_for_controlled_use" if args.confirm_reviewed_validation_data else "candidate_requires_human_review",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Calibration diagnostics use validation data and are not a held-out test result.",
            "Temperature scaling does not improve feature extraction, domain coverage, or detector accuracy.",
            "Empirical accuracy must come from the reserved independent test split and is not asserted by this artifact.",
            "The artifact is only applicable to this exact model checkpoint and declared data domain.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
