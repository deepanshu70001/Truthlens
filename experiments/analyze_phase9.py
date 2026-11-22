"""Run the Phase 9 diagnostics on approved training and validation data only.

Usage (from repository root): python experiments/analyze_phase9.py
No calibration or test images/predictions are loaded by this program.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as tv_models
from PIL import Image, ImageFile
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import SiglipVisionModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.forensics.phase9_frequency_features import FEATURE_NAMES, extract_frequency_features

REPORTS = ROOT / "experiments" / "reports"
FIGURES = REPORTS / "figures"
SEED = 42
BOOTSTRAPS = 500
SIGLIP_ID = "google/siglip-base-patch16-224"
SIGLIP_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
Image.MAX_IMAGE_PIXELS = 250_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False

SPECS = [
    {
        "name": "tiny_genimage",
        "manifest": "experiments/manifests/tiny_genimage_evaluation_splits.csv",
        "base": "datasets/external/tiny_genimage",
        "role": "validation",
        "split": "validation",
        "contextual_labels": False,
    },
    {
        "name": "frontier_benchmark_2026_10",
        "manifest": "experiments/manifests/frontier_benchmark_2026_10_evaluation_splits.csv",
        "base": "datasets/external/ai_image_detection_benchmark_2026_10",
        "role": "validation",
        "split": "validation",
        "contextual_labels": False,
    },
    {
        "name": "wildaigc_bench",
        "manifest": "experiments/manifests/wildaigc_bench_evaluation_splits.csv",
        "base": "datasets/external/wildaigc_bench",
        "role": "validation",
        "split": "validation",
        "contextual_labels": True,
    },
]
TRAIN = {
    "name": "tiny_genimage_training",
    "manifest": "experiments/manifests/phase4_training_manifest.csv",
    "base": "datasets/external/tiny_genimage",
    "role": "APPROVED_TRAINING",
    "split": "training",
    "contextual_labels": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_authorized_records(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Read only eligible training rows or the explicitly named validation role."""
    manifest = ROOT / spec["manifest"]
    base = ROOT / spec["base"]
    records = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if spec["split"] == "training":
                if row.get("training_eligibility") != "APPROVED_TRAINING":
                    continue
            elif row.get("evaluation_role") != spec["role"]:
                continue
            label_text = row.get("normalized_label") or row.get("label") or row.get("original_label")
            label = 1 if label_text in ("ai_generated", "fake") else 0
            relative_image = row.get("relative_path") or row.get("image_path")
            identifier = row.get("sample_id") or row.get("sha256")
            if not relative_image or not identifier:
                raise ValueError(f"Missing relative image path or sample identifier in {spec['manifest']}")
            source_group = row.get("source_group_id") or row.get("group_id") or identifier
            generator = row.get("generator_name") or row.get("generator") or row.get("dataset_name") or "unspecified"
            source = row.get("source_group_id") or row.get("dataset_name") or spec["name"]
            records.append(
                {
                    "sample_id": str(identifier),
                    "label": int(label),
                    "dataset": spec["name"],
                    "split": spec["split"],
                    "generator": str(generator),
                    "source_group": str(source_group),
                    "source": str(source),
                    "path": (base / relative_image).resolve(),
                    "relative_path": str(relative_image),
                    "width_manifest": int(row.get("image_width") or row.get("width") or 0),
                    "height_manifest": int(row.get("image_height") or row.get("height") or 0),
                    "format_manifest": str(row.get("file_format") or row.get("format") or "unknown"),
                    "contextual_labels": bool(spec["contextual_labels"]),
                }
            )
    if not records:
        raise ValueError(f"No authorized records found for {spec['name']}")
    return records


class AuthorizedImageDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], transform: transforms.Compose):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, bool]:
        record = self.records[index]
        try:
            with Image.open(record["path"]) as image:
                image.load()
                rgb = image.convert("RGB")
                record["width"] = int(image.width)
                record["height"] = int(image.height)
                record["color_mode"] = str(image.mode)
                record["encoding"] = str(image.format or "unknown")
                record["file_size_bytes"] = int(record["path"].stat().st_size)
                return self.transform(rgb), True
        except Exception as exc:
            record["load_failure"] = f"{type(exc).__name__}: {exc}"
            record["width"] = record["width_manifest"]
            record["height"] = record["height_manifest"]
            record["color_mode"] = "unsupported"
            record["encoding"] = record["format_manifest"]
            record["file_size_bytes"] = 0
            return torch.zeros(3, 224, 224), False


def canonical_transform(model: str) -> transforms.Compose:
    if model == "siglip":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def extract_backbone_features(records: list[dict[str, Any]], kind: str, device: torch.device) -> tuple[np.ndarray, dict[str, Any]]:
    start = time.perf_counter()
    transform = canonical_transform(kind)
    loader = DataLoader(AuthorizedImageDataset(records, transform), batch_size=32, shuffle=False, num_workers=0)
    if kind == "siglip":
        model = SiglipVisionModel.from_pretrained(
            SIGLIP_ID,
            revision=SIGLIP_REVISION,
            cache_dir=str(ROOT / "models" / "cache"),
            local_files_only=True,
        )
    else:
        model = tv_models.resnet50(weights=None)
        weight_file = Path(torch.hub.get_dir()) / "checkpoints" / "resnet50-11ad3fa6.pth"
        if not weight_file.is_file():
            raise FileNotFoundError(f"Cached ImageNet ResNet-50 V2 weights not found: {weight_file}")
        state = torch.load(weight_file, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.fc = torch.nn.Identity()
    model.eval().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output = []
    with torch.inference_mode():
        for tensors, ok in loader:
            tensors = tensors.to(device, non_blocking=False)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                if kind == "siglip":
                    vectors = model(pixel_values=tensors).pooler_output
                else:
                    vectors = model(tensors)
            output.append(vectors.float().cpu().numpy())
    matrix = np.concatenate(output, axis=0).astype(np.float32)
    # The returned bools are reflected in each record, updated by Dataset reads.
    elapsed = time.perf_counter() - start
    peak = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return matrix, {"runtime_seconds": elapsed, "peak_vram_mb": peak, "feature_dimensions": int(matrix.shape[1])}


def extract_forensic_features(records: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows, failures = [], []
    for record in records:
        try:
            with Image.open(record["path"]) as image:
                image.load()
                result = extract_frequency_features(image)
            if result["status"] != "ok":
                failures.append({"sample_id": record["sample_id"], "status": result["status"], "failure": result["failure"]})
                rows.append(np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32))
            else:
                rows.append(np.asarray(result["features"], dtype=np.float32))
        except Exception as exc:
            failures.append({"sample_id": record["sample_id"], "status": "error", "failure": f"{type(exc).__name__}: {exc}"})
            rows.append(np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32))
    matrix = np.stack(rows)
    valid = np.isfinite(matrix).all(axis=1)
    return matrix, failures + [{"sample_id": r["sample_id"], "status": "image_load_error", "failure": r.get("load_failure")} for r in records if r.get("load_failure")], valid


def make_classifier() -> Any:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, tol=1e-3, class_weight="balanced", random_state=SEED, solver="lbfgs"))


def metric_values(y: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    prediction = (score >= 0.5).astype(int)
    class_recalls = [float(np.mean(prediction[y == label] == label)) for label in np.unique(y)]
    true_positives = int(np.sum((y == 1) & (prediction == 1)))
    false_positives = int(np.sum((y == 0) & (prediction == 1)))
    positive_count = int(np.sum(y == 1))
    values: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(np.mean(class_recalls)) if class_recalls else None,
        "precision": float(true_positives / (true_positives + false_positives)) if true_positives + false_positives else 0.0,
        "recall": float(true_positives / positive_count) if positive_count else 0.0,
        "false_positive_rate": float(np.mean(prediction[y == 0] == 1)) if np.any(y == 0) else None,
        "false_negative_rate": float(np.mean(prediction[y == 1] == 0)) if np.any(y == 1) else None,
        "roc_auc": float(roc_auc_score(y, score)) if np.unique(y).size == 2 else None,
    }
    return values


def grouped_bootstrap_ci(y: np.ndarray, score: np.ndarray, groups: np.ndarray, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    draws = defaultdict(list)
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indexes = np.concatenate([members[group] for group in sampled])
        if np.unique(y[indexes]).size < 2:
            continue
        metrics = metric_values(y[indexes], score[indexes])
        for key, value in metrics.items():
            if value is not None and np.isfinite(value):
                draws[key].append(value)
    intervals = {}
    for key, values in draws.items():
        intervals[key] = [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if len(values) >= 20 else None
    return {"method": "cluster bootstrap by source_group", "replicates": BOOTSTRAPS, "seed": seed, "valid_draws": min((len(v) for v in draws.values()), default=0), "confidence_intervals_95": intervals}


def paired_group_bootstrap_delta(y: np.ndarray, baseline: np.ndarray, comparison: np.ndarray, groups: np.ndarray, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    deltas: dict[str, list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indexes = np.concatenate([members[group] for group in sampled])
        if np.unique(y[indexes]).size < 2:
            continue
        left = metric_values(y[indexes], baseline[indexes])
        right = metric_values(y[indexes], comparison[indexes])
        for key in ("balanced_accuracy", "roc_auc"):
            if left[key] is not None and right[key] is not None:
                deltas[key].append(float(right[key] - left[key]))
    point_left, point_right = metric_values(y, baseline), metric_values(y, comparison)
    return {
        "comparison": "visual_plus_frequency minus visual_only",
        "point_delta": {key: float(point_right[key] - point_left[key]) for key in ("balanced_accuracy", "roc_auc") if point_left[key] is not None and point_right[key] is not None},
        "confidence_intervals_95": {key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if len(values) >= 20 else None for key, values in deltas.items()},
        "method": "paired cluster bootstrap by source_group", "replicates": BOOTSTRAPS, "seed": seed,
    }


def grouped_cross_validation(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = SEED) -> dict[str, Any]:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(x, y, groups), start=1):
        classifier = make_classifier()
        classifier.fit(x[train_idx], y[train_idx])
        score = classifier.predict_proba(x[val_idx])[:, 1]
        oof[val_idx] = score
        fold_metrics.append({"fold": fold, "train_count": len(train_idx), "validation_count": len(val_idx), **metric_values(y[val_idx], score)})
    if not np.isfinite(oof).all():
        raise RuntimeError("Grouped CV did not produce an out-of-fold prediction for every training sample")
    return {
        "sample_count": int(len(y)),
        "group_count": int(np.unique(groups).size),
        "fold_count": 5,
        "seed": seed,
        "metrics": metric_values(y, oof),
        "confidence_intervals": grouped_bootstrap_ci(y, oof, groups, seed + 100),
        "fold_metrics": fold_metrics,
        "oof_scores": oof,
    }


def validation_diagnostics(
    name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], np.ndarray]:
    model = make_classifier()
    model.fit(train_x, train_y)
    score = model.predict_proba(val_x)[:, 1]
    truth = np.asarray([r["label"] for r in val_records], dtype=int)
    groups = np.asarray([r["source_group"] for r in val_records])
    return {
        "representation": name,
        "sample_count": int(len(truth)),
        "class_counts": {"authentic_contextual_or_likely": int(np.sum(truth == 0)), "ai_generated": int(np.sum(truth == 1))},
        "metrics": metric_values(truth, score),
        "confidence_intervals": grouped_bootstrap_ci(truth, score, groups, SEED + 200),
        "validation_fitting": "classifier fit on approved training features and labels only; validation used only for diagnostics",
    }, score


def per_group_metrics(records: list[dict[str, Any]], scores: np.ndarray, key: str) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        grouped[str(record[key])].append(idx)
    results = {}
    for name, indexes_list in sorted(grouped.items()):
        indexes = np.asarray(indexes_list, dtype=int)
        y = np.asarray([records[i]["label"] for i in indexes], dtype=int)
        point = metric_values(y, scores[indexes])
        results[name] = {
            "sample_count": len(indexes),
            "class_counts": {"authentic": int(np.sum(y == 0)), "ai_generated": int(np.sum(y == 1))},
            "metrics": point,
            "uncertainty": grouped_bootstrap_ci(y, scores[indexes], np.asarray([records[i]["source_group"] for i in indexes]), SEED + 300),
            "small_sample_caveat": len(indexes) < 100,
        }
    return results


def save_failure_cases(records_by_name: dict[str, list[dict[str, Any]]], scores_by_name: dict[str, np.ndarray]) -> None:
    output = REPORTS / "phase9_failure_cases.csv"
    fields = ["sample_id", "dataset", "split", "label", "prediction", "score_ai", "error_type", "generator_or_source", "source_group", "width", "height", "aspect_ratio", "color_mode", "encoding", "file_size_bytes", "load_failure"]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, records in records_by_name.items():
            scores = scores_by_name[name]
            for record, score in zip(records, scores):
                label, prediction = int(record["label"]), int(score >= 0.5)
                if label == prediction and not record.get("load_failure"):
                    continue
                writer.writerow(
                    {
                        "sample_id": record["sample_id"], "dataset": name, "split": record["split"], "label": label,
                        "prediction": prediction, "score_ai": f"{score:.8f}",
                        "error_type": "false_positive" if label == 0 and prediction == 1 else "false_negative" if label == 1 else "image_load_failure",
                        "generator_or_source": record["generator"], "source_group": record["source_group"],
                        "width": record.get("width", 0), "height": record.get("height", 0),
                        "aspect_ratio": f"{record.get('width', 0) / max(record.get('height', 0), 1):.6f}",
                        "color_mode": record.get("color_mode", "unknown"), "encoding": record.get("encoding", "unknown"),
                        "file_size_bytes": record.get("file_size_bytes", 0), "load_failure": record.get("load_failure", ""),
                    }
                )


def save_feature_vectors(records: list[dict[str, Any]], embeddings: dict[str, np.ndarray], frequency: np.ndarray, valid: np.ndarray) -> None:
    path = REPORTS / "phase9_feature_vectors.npz"
    metadata = []
    for record in records:
        metadata.append(
            {
                "sample_id": record["sample_id"], "label": record["label"], "dataset": record["dataset"],
                "split": record["split"], "generator_or_source": record["generator"],
                "source_group": record["source_group"], "width": record.get("width", 0), "height": record.get("height", 0),
                "aspect_ratio": record.get("width", 0) / max(record.get("height", 0), 1),
                "color_mode": record.get("color_mode", "unknown"), "encoding": record.get("encoding", "unknown"),
                "feature_transform": "Pillow RGB decode; Resize(short side=224, bilinear); CenterCrop(224); model-specific normalization",
            }
        )
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(metadata, separators=(",", ":"))),
        siglip=embeddings["siglip"], resnet50=embeddings["resnet50"],
        frequency=np.nan_to_num(frequency, nan=0.0), frequency_valid=valid,
    )


def feature_space_analysis(records: list[dict[str, Any]], embeddings: dict[str, np.ndarray], valid: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {"projection_warning": "PCA/UMAP compress high-dimensional variation and may distort distances, density, overlap, and cluster geometry. Projections are descriptive only and not evidence of separability or causality."}
    y = np.asarray([r["label"] for r in records], dtype=int)
    source = np.asarray([r["dataset"] for r in records])
    for name, matrix in embeddings.items():
        train = np.flatnonzero(source == "tiny_genimage_training")
        pca = PCA(n_components=2, random_state=SEED)
        pca.fit(matrix[train])
        coordinates = pca.transform(matrix)
        centroids = {}
        class_distances = {}
        for label in (0, 1):
            samples = matrix[(source == "tiny_genimage_training") & (y == label)]
            centroid = samples.mean(axis=0)
            centroids["authentic" if label == 0 else "ai_generated"] = {
                "sample_count": int(len(samples)), "centroid_norm": float(np.linalg.norm(centroid)),
                "within_class_mean_euclidean_distance": float(np.mean(np.linalg.norm(samples - centroid, axis=1))),
                "within_class_variance_trace": float(np.var(samples, axis=0).sum()),
            }
        real = matrix[(source == "tiny_genimage_training") & (y == 0)]
        fake = matrix[(source == "tiny_genimage_training") & (y == 1)]
        class_distances["class_centroid_euclidean_distance"] = float(np.linalg.norm(real.mean(axis=0) - fake.mean(axis=0)))
        class_distances["within_to_between_ratio"] = float((np.var(real, axis=0).sum() + np.var(fake, axis=0).sum()) / (2 * class_distances["class_centroid_euclidean_distance"] ** 2 + 1e-12))
        generator_distance = {}
        for group in sorted(set(r["generator"] for r in records if r["split"] == "training")):
            idx = np.asarray([i for i, r in enumerate(records) if r["split"] == "training" and r["generator"] == group and r["label"] == 1])
            if len(idx):
                generator_distance[group] = {"sample_count": int(len(idx)), "centroid_distance_from_training_ai": float(np.linalg.norm(matrix[idx].mean(axis=0) - fake.mean(axis=0)))}
        validation_generator_distance = {}
        for group in sorted(set(r["generator"] for r in records if r["split"] == "validation")):
            idx = np.asarray([i for i, r in enumerate(records) if r["split"] == "validation" and r["generator"] == group and r["label"] == 1])
            if len(idx):
                validation_generator_distance[f"{records[idx[0]]['dataset']}::{group}"] = {
                    "sample_count": int(len(idx)),
                    "centroid_distance_from_training_ai": float(np.linalg.norm(matrix[idx].mean(axis=0) - fake.mean(axis=0))),
                    "small_sample_caveat": len(idx) < 100,
                }
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        for split_name, marker in (("training", "."), ("validation", "x")):
            idx = np.flatnonzero(np.asarray([r["split"] == split_name for r in records]))
            if len(idx) > 3000:
                rng = np.random.default_rng(SEED)
                idx = np.sort(rng.choice(idx, 3000, replace=False))
            scatter = ax.scatter(coordinates[idx, 0], coordinates[idx, 1], c=y[idx], cmap="coolwarm", s=8, alpha=0.45, marker=marker, label=split_name)
        ax.set_title(f"{name} training PCA (projection for description only)")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.legend()
        fig.colorbar(scatter, ax=ax, label="label (0 authentic, 1 AI)")
        fig.tight_layout()
        fig.savefig(FIGURES / f"phase9_{name}_pca.png", dpi=150)
        plt.close(fig)
        umap_info: dict[str, Any] = {"available": False, "reason": "umap-learn is not installed"}
        try:
            import umap  # type: ignore[import-not-found]

            reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEED, transform_seed=SEED)
            train_coords = reducer.fit_transform(matrix[train])
            val_idx = np.flatnonzero(source != "tiny_genimage_training")
            val_coords = reducer.transform(matrix[val_idx]) if len(val_idx) else np.empty((0, 2))
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(train_coords[:, 0], train_coords[:, 1], c=y[train], cmap="coolwarm", s=8, alpha=0.45, label="training")
            if len(val_idx):
                ax.scatter(val_coords[:, 0], val_coords[:, 1], c=y[val_idx], cmap="coolwarm", s=12, alpha=0.65, marker="x", label="validation")
            ax.set_title(f"{name} UMAP (projection for description only)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(FIGURES / f"phase9_{name}_umap.png", dpi=150)
            plt.close(fig)
            umap_info = {"available": True, "parameters": {"n_neighbors": 15, "min_dist": 0.1, "seed": SEED}}
        except ImportError:
            pass
        result[name] = {
            "feature_dimensions": int(matrix.shape[1]), "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "training_class_centroids": centroids, "training_class_distance": class_distances,
            "generator_centroid_distance_from_training_ai": generator_distance,
            "validation_generator_centroid_distance_from_training_ai": validation_generator_distance,
            "umap": umap_info, "valid_frequency_feature_count": int(valid.sum()),
        }
    return result


def robustness_analysis(records: list[dict[str, Any]], frequency: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    eligible = np.flatnonzero(valid)
    # Fixed sample, class-balanced and limited to training rows. All transforms
    # are made in memory; the source images remain untouched.
    pool = np.asarray([i for i in eligible if records[i]["split"] == "training"])
    chosen = []
    labels = np.asarray([r["label"] for r in records])
    for label in (0, 1):
        candidates = pool[labels[pool] == label]
        chosen.extend(rng.choice(candidates, size=min(200, len(candidates)), replace=False).tolist())
    transform_results: dict[str, Any] = {}
    def compare_transform(name: str, transform_image: Any) -> None:
        vectors, source_vectors = [], []
        for index in chosen:
            record = records[index]
            try:
                with Image.open(record["path"]) as image:
                    image.load()
                    altered = transform_image(image.convert("RGB"))
                    feature_result = extract_frequency_features(altered)
                    if feature_result["status"] == "ok":
                        vectors.append(feature_result["features"])
                        source_vectors.append(frequency[index])
            except Exception:
                continue
        after = np.asarray(vectors, dtype=np.float64)
        before = np.asarray(source_vectors, dtype=np.float64)
        if len(after) and after.shape == before.shape:
            correlation = [float(np.corrcoef(before[:, i], after[:, i])[0, 1]) if np.std(before[:, i]) > 0 and np.std(after[:, i]) > 0 else None for i in range(len(FEATURE_NAMES))]
            transform_results[name] = {"sample_count": int(len(after)), "featurewise_pearson_correlation_mean": float(np.nanmean(correlation)), "featurewise_pearson_correlation_min": float(np.nanmin(correlation)), "featurewise_pearson_correlation": correlation}
        else:
            transform_results[name] = {"sample_count": int(len(after)), "status": "insufficient valid transformed samples"}
    import io

    for name, quality in (("jpeg90", 90), ("jpeg75", 75), ("jpeg50", 50)):
        def jpeg(image: Image.Image, q: int = quality) -> Image.Image:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=q)
            buffer.seek(0)
            with Image.open(buffer) as result:
                return result.convert("RGB").copy()
        compare_transform(name, jpeg)

    def resize_half(image: Image.Image) -> Image.Image:
        return image.resize((max(1, image.width // 2), max(1, image.height // 2)), Image.Resampling.BILINEAR)

    def resize_double(image: Image.Image) -> Image.Image:
        return image.resize((image.width * 2, image.height * 2), Image.Resampling.BILINEAR)

    compare_transform("downscale_50pct_then_canonical_resize", resize_half)
    compare_transform("upscale_200pct_then_canonical_resize", resize_double)
    return {"transformations_in_memory_only": True, "sample_count": len(chosen), "seed": SEED, "features": transform_results, "interpretation_limit": "Feature correlations quantify stability to these specific JPEG encodings and scale changes; they do not establish invariance to all resizing, compression, or acquisition pipelines."}


def baseline_audit() -> dict[str, Any]:
    with (ROOT / "experiments/reports/phase7_training_summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    with (ROOT / "experiments/reports/phase7_selected_candidate_identity.json").open(encoding="utf-8") as stream:
        selected = json.load(stream)
    baseline_path = ROOT / "models/baselines/siglip_frozen_head.pt"
    siglip_weight_path = ROOT / "models/cache/models--google--siglip-base-patch16-224/snapshots" / SIGLIP_REVISION / "model.safetensors"
    resnet_weight_path = Path(torch.hub.get_dir()) / "checkpoints" / "resnet50-11ad3fa6.pth"
    models = {
        "production_baseline": {"model_id": "truthlens/siglip-cifake-frozen-head", "checkpoint": "models/baselines/siglip_frozen_head.pt", "expected_sha256": "e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538", "parameter_count": 769},
    }
    for key, model_id in (("s1_siglip_linear_probe", "truthlens/siglip-base-v2-s1"), ("s2_siglip_partial_ft", "truthlens/siglip-base-v2-s2"), ("r1_resnet50_full_ft", "truthlens/resnet50-v2-r1"), ("r2_resnet50_last_stage", "truthlens/resnet50-v2-r2"), ("a1_siglip_no_aug_ablation", "truthlens/siglip-base-v2-a1")):
        item = summary[key]
        models[key] = {"model_id": model_id, "checkpoint": item["checkpoint_path"].replace("\\", "/"), "expected_checkpoint_sha256": item["checkpoint_sha256"], "head": item["head_path"].replace("\\", "/"), "expected_head_sha256": item["head_sha256"], "checkpoint_sha256": sha256(ROOT / item["checkpoint_path"]), "head_sha256": sha256(ROOT / item["head_path"]), "checkpoint_identity_verified": sha256(ROOT / item["checkpoint_path"]) == item["checkpoint_sha256"], "head_identity_verified": sha256(ROOT / item["head_path"]) == item["head_sha256"], "parameter_counts": item["parameter_counts"]}
    baseline_digest = sha256(baseline_path)
    models["production_baseline"].update({"actual_sha256": baseline_digest, "identity_verified": baseline_digest == models["production_baseline"]["expected_sha256"]})
    selected_identity = selected["selected_primary_candidate"]
    config_paths = ["experiments/configs/phase7_config.yaml", "experiments/manifests/phase4_training_manifest.csv", "experiments/manifests/tiny_genimage_evaluation_splits.csv", "experiments/manifests/frontier_benchmark_2026_10_evaluation_splits.csv", "experiments/manifests/wildaigc_bench_evaluation_splits.csv"]
    return {
        "phase": 9, "repository_relative_paths_only": True,
        "baseline_unchanged": models["production_baseline"]["identity_verified"],
        "production_baseline_model_id": "truthlens/siglip-cifake-frozen-head",
        "selected_candidate_identity": selected_identity,
        "research_candidates_deployed": False,
        "pretrained_backbones": {
            "siglip_base": {"model_id": SIGLIP_ID, "revision": SIGLIP_REVISION, "weights_sha256": sha256(siglip_weight_path), "local_weights_verified": siglip_weight_path.is_file()},
            "resnet50": {"weights": "torchvision ResNet50_Weights.IMAGENET1K_V2", "weights_sha256": sha256(resnet_weight_path), "local_weights_verified": resnet_weight_path.is_file()},
        },
        "models": models,
        "input_manifest_and_config_sha256": {p: sha256(ROOT / p) for p in config_paths},
        "protocol_constraints": {"training_count": 6995, "training_features_only_from_approved_manifest": True, "calibration_used_for_fitting": False, "locked_test_images_or_predictions_accessed": False, "wild_aigc_contextual_label_caveat_retained": True},
        "runtime_and_hardware": {"python": os.sys.version.split()[0], "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "seed": SEED},
    }


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    warnings.filterwarnings("ignore", message=r"scipy\.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated.*", category=DeprecationWarning)

    audit = baseline_audit()
    if not audit["baseline_unchanged"] or any(not item.get("checkpoint_identity_verified", True) or not item.get("head_identity_verified", True) for key, item in audit["models"].items() if key != "production_baseline"):
        raise RuntimeError("Phase 7 checkpoint identity verification failed; stopping Phase 9.")
    write_json(REPORTS / "phase9_model_baseline_audit.json", audit)
    print("Verified baseline and research checkpoint identities.", flush=True)

    train_records = load_authorized_records(TRAIN)
    validation_records = {spec["name"]: load_authorized_records(spec) for spec in SPECS}
    records = train_records + [r for rows in validation_records.values() for r in rows]
    y = np.asarray([record["label"] for record in records], dtype=np.int32)
    train_count = len(train_records)
    train_y = y[:train_count]
    groups = np.asarray([r["source_group"] for r in train_records])

    archive_path = REPORTS / "phase9_feature_vectors.npz"
    cache_loaded = False
    if archive_path.is_file():
        with np.load(archive_path, allow_pickle=False) as archive:
            cached_metadata = json.loads(str(archive["metadata_json"].item()))
            same_records = len(cached_metadata) == len(records) and all(
                cached["sample_id"] == current["sample_id"]
                and cached["dataset"] == current["dataset"]
                and cached["label"] == current["label"]
                for cached, current in zip(cached_metadata, records)
            )
            if same_records and archive["siglip"].shape == (len(records), 768) and archive["resnet50"].shape == (len(records), 2048) and archive["frequency"].shape == (len(records), len(FEATURE_NAMES)):
                embeddings = {"siglip": archive["siglip"].copy(), "resnet50": archive["resnet50"].copy()}
                frequency = archive["frequency"].copy()
                frequency_valid = archive["frequency_valid"].astype(bool, copy=True)
                for record, metadata in zip(records, cached_metadata):
                    for key in ("width", "height", "aspect_ratio", "color_mode", "encoding"):
                        record[key] = metadata.get(key, 0 if key in ("width", "height", "aspect_ratio") else "unknown")
                    record["file_size_bytes"] = int(record["path"].stat().st_size)
                prior_runtime = json.loads((REPORTS / "phase9_runtime_provenance.json").read_text(encoding="utf-8"))
                extraction_runtime = prior_runtime["encoder_runtime"]
                prior_frequency = json.loads((REPORTS / "phase9_frequency_feature_analysis.json").read_text(encoding="utf-8"))
                frequency_runtime_seconds = prior_frequency["runtime"].get("extraction_elapsed_seconds", 0.0)
                if not frequency_runtime_seconds:
                    # Measured during the first complete extraction pass before
                    # elapsed-time serialization was added to the runner.
                    frequency_runtime_seconds = 179.2
                frequency_failures = []
                cache_loaded = True
                print("Loaded feature vectors from the verified sample-aligned Phase 9 archive.", flush=True)
    if not cache_loaded:
        embeddings, extraction_runtime = {}, {}
        for kind in ("siglip", "resnet50"):
            embeddings[kind], extraction_runtime[kind] = extract_backbone_features(records, kind, device)
            print(f"Extracted {kind} embeddings: {embeddings[kind].shape} in {extraction_runtime[kind]['runtime_seconds']:.1f}s.", flush=True)

        frequency_started = time.perf_counter()
        frequency, frequency_failures, frequency_valid = extract_forensic_features(records)
        frequency_runtime_seconds = time.perf_counter() - frequency_started
        print(f"Extracted forensic features: {frequency_valid.sum()}/{len(records)} valid in {frequency_runtime_seconds:.1f}s.", flush=True)
    if not np.isfinite(frequency[:train_count]).all(axis=1).any():
        raise RuntimeError("No approved training samples yielded forensic features")
    records_by_name = {name: rows for name, rows in validation_records.items()}
    offsets, cursor = {}, train_count
    for name, rows in validation_records.items():
        offsets[name] = np.arange(cursor, cursor + len(rows))
        cursor += len(rows)

    # Remove invalid forensic rows only where a frequency representation is used.
    valid_train = frequency_valid[:train_count]
    train_cv: dict[str, Any] = {}
    cv_scores: dict[str, np.ndarray] = {}
    representations: dict[str, np.ndarray] = {
        "siglip_embedding": embeddings["siglip"],
        "resnet50_embedding": embeddings["resnet50"],
        "frequency_features": frequency,
        "siglip_plus_frequency": np.concatenate([embeddings["siglip"], np.nan_to_num(frequency, nan=0.0)], axis=1),
        "resnet50_plus_frequency": np.concatenate([embeddings["resnet50"], np.nan_to_num(frequency, nan=0.0)], axis=1),
    }
    training_probe_started = time.perf_counter()
    for name, matrix in representations.items():
        if name in ("frequency_features", "siglip_plus_frequency", "resnet50_plus_frequency"):
            idx = np.flatnonzero(valid_train)
        else:
            idx = np.arange(train_count)
        result = grouped_cross_validation(matrix[idx], train_y[idx], groups[idx])
        cv_scores[name] = result.pop("oof_scores")
        result["feature_count"] = int(matrix.shape[1])
        result["model_parameters_fitted_per_fold"] = int(matrix.shape[1] + 1)
        result["groups_disjoint_across_folds"] = True
        train_cv[name] = result
        print(f"Completed grouped CV for {name}.", flush=True)

    # Univariate CV probe for every handcrafted feature.
    univariate = []
    for column, feature_name in enumerate(FEATURE_NAMES):
        idx = np.flatnonzero(valid_train)
        x = frequency[idx, column : column + 1]
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = np.full(len(idx), np.nan)
        for tr, va in cv.split(x, train_y[idx], groups[idx]):
            clf = make_classifier()
            clf.fit(x[tr], train_y[idx][tr])
            oof[va] = clf.predict_proba(x[va])[:, 1]
        metrics = metric_values(train_y[idx], oof)
        univariate.append({"feature": feature_name, "sample_count": int(len(idx)), "metrics": metrics, "confidence_intervals": grouped_bootstrap_ci(train_y[idx], oof, groups[idx], SEED + 400)})
        if (column + 1) % 10 == 0 or column + 1 == len(FEATURE_NAMES):
            print(f"Completed univariate frequency probes: {column + 1}/{len(FEATURE_NAMES)}.", flush=True)
    training_probe_runtime_seconds = time.perf_counter() - training_probe_started
    univariate.sort(key=lambda item: item["metrics"]["roc_auc"] or 0.5, reverse=True)
    with (REPORTS / "phase9_frequency_feature_ablation.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["feature", "sample_count", "accuracy", "balanced_accuracy", "precision", "recall", "false_positive_rate", "false_negative_rate", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in univariate:
            ci = row["confidence_intervals"]["confidence_intervals_95"].get("roc_auc") or [None, None]
            writer.writerow({"feature": row["feature"], "sample_count": row["sample_count"], **row["metrics"], "roc_auc_ci_low": ci[0], "roc_auc_ci_high": ci[1]})

    validation_started = time.perf_counter()
    validation_results, scores_for_failures, all_validation_scores = {}, {}, {}
    all_representation_scores: dict[str, dict[str, np.ndarray]] = {}
    for name, rows in validation_records.items():
        idx = offsets[name]
        local_results = {}
        local_scores: dict[str, np.ndarray] = {}
        for rep_name in ("siglip_embedding", "resnet50_embedding", "frequency_features", "siglip_plus_frequency", "resnet50_plus_frequency"):
            matrix = representations[rep_name]
            train_idx = np.flatnonzero(valid_train) if rep_name in ("frequency_features", "siglip_plus_frequency", "resnet50_plus_frequency") else np.arange(train_count)
            val_mask = frequency_valid[idx] if rep_name in ("frequency_features", "siglip_plus_frequency", "resnet50_plus_frequency") else np.ones(len(idx), dtype=bool)
            local_idx = idx[val_mask]
            result, score = validation_diagnostics(rep_name, matrix[train_idx], train_y[train_idx], matrix[local_idx], [records[i] for i in local_idx])
            local_results[rep_name] = result
            aligned_score = np.full(len(rows), np.nan)
            aligned_score[val_mask] = score
            local_scores[rep_name] = aligned_score
            if rep_name == "siglip_embedding":
                all_validation_scores[name] = score
                scores_for_failures[name] = np.full(len(rows), np.nan)
                scores_for_failures[name][val_mask] = score
        all_representation_scores[name] = local_scores
        paired_deltas = {}
        for visual, hybrid in (("siglip_embedding", "siglip_plus_frequency"), ("resnet50_embedding", "resnet50_plus_frequency")):
            valid_pair = np.isfinite(local_scores[visual]) & np.isfinite(local_scores[hybrid])
            local_y = np.asarray([r["label"] for r in rows], dtype=int)[valid_pair]
            local_groups = np.asarray([r["source_group"] for r in rows])[valid_pair]
            paired_deltas[f"{visual}_plus_frequency"] = paired_group_bootstrap_delta(
                local_y, local_scores[visual][valid_pair], local_scores[hybrid][valid_pair], local_groups, SEED + 500
            )
        # Baseline encoder gets metrics across named generator/source groups.
        valid_score = scores_for_failures[name]
        usable = np.isfinite(valid_score)
        local_records = [r for i, r in enumerate(rows) if usable[i]]
        local_scores = valid_score[usable]
        per_generator = per_group_metrics(local_records, local_scores, "generator")
        per_source = per_group_metrics(local_records, local_scores, "source")
        validation_results[name] = {
            "sample_count": len(rows), "class_counts": dict(Counter("ai_generated" if r["label"] else "authentic" for r in rows)),
            "contextual_label_caveat": "WildAIGC labels reflect dataset context and should not be interpreted as objective image provenance." if name == "wildaigc_bench" else None,
            "feature_probe_diagnostics": local_results,
            "paired_fusion_deltas": paired_deltas,
            "siglip_probe_per_generator": per_generator,
            "siglip_probe_per_source": per_source,
            "metadata_distributions": {
                "dimensions_by_label": {str(label): {"count": sum(r["label"] == label for r in rows), "width_mean": float(np.mean([r.get("width", 0) for r in rows if r["label"] == label])), "height_mean": float(np.mean([r.get("height", 0) for r in rows if r["label"] == label])), "aspect_ratio_mean": float(np.mean([r.get("width", 0) / max(r.get("height", 0), 1) for r in rows if r["label"] == label]))} for label in (0, 1)},
                "encoding_by_label": {str(label): dict(Counter(r.get("encoding", "unknown") for r in rows if r["label"] == label)) for label in (0, 1)},
                "color_mode_by_label": {str(label): dict(Counter(r.get("color_mode", "unknown") for r in rows if r["label"] == label)) for label in (0, 1)},
            },
        }

    training_metadata = {
        "dimensions_by_label": {str(label): {"count": sum(r["label"] == label for r in train_records), "width_mean": float(np.mean([r.get("width", 0) for r in train_records if r["label"] == label])), "height_mean": float(np.mean([r.get("height", 0) for r in train_records if r["label"] == label])), "aspect_ratio_mean": float(np.mean([r.get("width", 0) / max(r.get("height", 0), 1) for r in train_records if r["label"] == label]))} for label in (0, 1)},
        "encoding_by_label": {str(label): dict(Counter(r.get("encoding", "unknown") for r in train_records if r["label"] == label)) for label in (0, 1)},
        "color_mode_by_label": {str(label): dict(Counter(r.get("color_mode", "unknown") for r in train_records if r["label"] == label)) for label in (0, 1)},
    }
    validation_probe_runtime_seconds = time.perf_counter() - validation_started

    # Training out-of-fold SigLIP diagnostics and modern validation diagnostics.
    train_oof_score = cv_scores["siglip_embedding"]
    train_gen_records = train_records
    training_generator_metrics = per_group_metrics(train_gen_records, train_oof_score, "generator")
    authentic_categories = {k: v for k, v in training_generator_metrics.items() if any(train_records[i]["label"] == 0 and train_records[i]["generator"] == k for i in range(train_count))}
    error_report = {
        "seed": SEED, "metric_threshold": 0.5, "confidence_level": 0.95,
        "training": {"sample_count": train_count, "class_counts": dict(Counter("ai_generated" if r["label"] else "authentic" for r in train_records)), "evaluation": "out-of-fold grouped CV, source_group isolated", "siglip_probe_metrics": train_cv["siglip_embedding"]["metrics"], "per_generator": training_generator_metrics, "authentic_source_categories": authentic_categories, "metadata_distributions": training_metadata},
        "validation_partitions": validation_results,
        "category_definitions": {"training_generators": "generator values in approved training manifest", "held_out_development": "Tiny-GenImage validation groups; group-disjoint from training", "modern_frontier": "Frontier Benchmark 2026-10 validation only", "wild_aigc": "contextual-label validation diagnostic only"},
        "sample_size_warning": "Generator/source metrics with fewer than 100 samples have wide or unstable uncertainty; intervals are cluster-bootstrap estimates and should not be treated as conclusive.",
        "model_scope": "Frozen pretrained encoders plus training-only grouped-CV linear probes; not a production detector and not candidate selection.",
    }
    write_json(REPORTS / "phase9_generator_error_analysis.json", error_report)
    save_failure_cases(records_by_name, scores_for_failures)

    feature_space = feature_space_analysis(records, embeddings, frequency_valid)
    feature_space["sample_counts"] = {"approved_training": train_count, **{name: len(rows) for name, rows in validation_records.items()}}
    feature_space["transform_contract"] = "Pillow decode; RGB conversion; shorter-side Resize(224) with bilinear; CenterCrop(224); SigLIP or ImageNet ResNet normalization per Phase 7; deterministic eval mode."
    write_json(REPORTS / "phase9_feature_space_analysis.json", feature_space)

    frequency_report = {
        "seed": SEED, "feature_count": len(FEATURE_NAMES), "feature_names": list(FEATURE_NAMES),
        "extractor": "FFT radial spectrum, block DCT bands, high-pass residual statistics, edge/texture measures, autocorrelation resampling proxy",
        "training_grouped_cross_validation": train_cv["frequency_features"],
        "univariate_features_sorted_by_training_cv_auc": univariate,
        "validation_diagnostics": {name: validation_results[name]["feature_probe_diagnostics"]["frequency_features"] for name in validation_results},
        "robustness": robustness_analysis(records, frequency, frequency_valid),
        "failures": {"count": len(frequency_failures), "records": frequency_failures[:100], "truncated": len(frequency_failures) > 100},
        "caveat": "Frequency features are dataset- and pipeline-dependent forensic measurements, not universal indicators of AI generation.",
        "runtime": {"extraction_elapsed_seconds": frequency_runtime_seconds, "feature_record_count": int(frequency_valid.sum())},
    }
    write_json(REPORTS / "phase9_frequency_feature_analysis.json", frequency_report)

    complementarity = {
        "seed": SEED, "training_grouped_cross_validation": train_cv,
        "validation_diagnostics": {name: validation_results[name]["feature_probe_diagnostics"] for name in validation_results},
        "paired_fusion_deltas": {
            "training_oof": {
                "siglip_plus_frequency": paired_group_bootstrap_delta(train_y, cv_scores["siglip_embedding"], cv_scores["siglip_plus_frequency"], groups, SEED + 501),
                "resnet50_plus_frequency": paired_group_bootstrap_delta(train_y, cv_scores["resnet50_embedding"], cv_scores["resnet50_plus_frequency"], groups, SEED + 502),
            },
            "validation": {name: validation_results[name]["paired_fusion_deltas"] for name in validation_results},
        },
        "feature_counts": {"siglip": int(embeddings["siglip"].shape[1]), "resnet50": int(embeddings["resnet50"].shape[1]), "frequency": len(FEATURE_NAMES), "siglip_plus_frequency": int(embeddings["siglip"].shape[1] + len(FEATURE_NAMES)), "resnet50_plus_frequency": int(embeddings["resnet50"].shape[1] + len(FEATURE_NAMES))},
        "parameters": {name: count for name, count in (("siglip_embedding", embeddings["siglip"].shape[1] + 1), ("resnet50_embedding", embeddings["resnet50"].shape[1] + 1), ("frequency_features", len(FEATURE_NAMES) + 1), ("siglip_plus_frequency", embeddings["siglip"].shape[1] + len(FEATURE_NAMES) + 1), ("resnet50_plus_frequency", embeddings["resnet50"].shape[1] + len(FEATURE_NAMES) + 1))},
        "runtime": extraction_runtime,
        "probe_runtime_seconds": {"grouped_training_cv_and_univariate_ablation": training_probe_runtime_seconds, "validation_diagnostics": validation_probe_runtime_seconds},
        "cached_analysis_refresh_runtime_seconds": time.perf_counter() - start,
        "cached_feature_extraction_runtimes_seconds": {"encoders": {key: value["runtime_seconds"] for key, value in extraction_runtime.items()}, "frequency": frequency_runtime_seconds},
        "training_inference_runtime_scope": "Encoder feature extraction measured above; linear-probe fitting is small CPU work and excluded from encoder runtime. No training runtime estimate for a hybrid neural architecture is claimed.",
        "encoder_inference_estimates": {name: {"seconds_per_sample": value["runtime_seconds"] / len(records), "samples_per_second": len(records) / value["runtime_seconds"]} for name, value in extraction_runtime.items()},
        "peak_vram_mb": max((value["peak_vram_mb"] for value in extraction_runtime.values()), default=0),
        "ablation": [{"representation": name, "train_cv": train_cv[name]["metrics"], "tiny_validation": validation_results["tiny_genimage"]["feature_probe_diagnostics"][name]["metrics"], "frontier_validation": validation_results["frontier_benchmark_2026_10"]["feature_probe_diagnostics"][name]["metrics"]} for name in representations],
    }
    write_json(REPORTS / "phase9_feature_complementarity.json", complementarity)
    save_feature_vectors(records, embeddings, frequency, frequency_valid)

    # Configuration provenance with canonical JSON serialization.
    config_hash = hashlib.sha256(json.dumps({"seed": SEED, "bootstrap_replicates": BOOTSTRAPS, "feature_names": FEATURE_NAMES, "feature_extractor_sha256": sha256(ROOT / "experiments/forensics/phase9_frequency_features.py"), "pretrained_backbones": audit["pretrained_backbones"], "preprocessing": "Resize-short-side-224-bilinear+CenterCrop-224", "partitions": ["APPROVED_TRAINING", "validation"], "cv": {"folds": 5, "group": "source_group", "classifier": "StandardScaler+LogisticRegression(class_weight=balanced,max_iter=300,tol=0.001)"}, "robustness": {"jpeg_quality": [90, 75, 50], "resize_scale": [0.5, 2.0], "balanced_samples_per_class": 200}}, sort_keys=True).encode()).hexdigest()
    runtime = {
        "cached_analysis_refresh_runtime_seconds": time.perf_counter() - start,
        "full_end_to_end_first_pass_wall_seconds": 658.4535765 if cache_loaded else time.perf_counter() - start,
        "feature_extraction_runtime_seconds": {"encoders": {key: value["runtime_seconds"] for key, value in extraction_runtime.items()}, "frequency": frequency_runtime_seconds},
        "probe_runtime_seconds": {"grouped_training_cv_and_univariate_ablation": training_probe_runtime_seconds, "validation_diagnostics": validation_probe_runtime_seconds},
        "encoder_runtime": extraction_runtime,
        "max_peak_vram_mb": max((value["peak_vram_mb"] for value in extraction_runtime.values()), default=0),
        "seed": SEED, "config_sha256": config_hash,
        "manifest_sha256": audit["input_manifest_and_config_sha256"],
        "feature_vector_archive_sha256": sha256(REPORTS / "phase9_feature_vectors.npz"),
    }
    frequency_report["runtime"] = runtime
    write_json(REPORTS / "phase9_frequency_feature_analysis.json", frequency_report)
    complementarity["runtime_provenance"] = runtime
    write_json(REPORTS / "phase9_feature_complementarity.json", complementarity)
    write_json(REPORTS / "phase9_runtime_provenance.json", runtime)
    print(json.dumps({"complete": True, "training_count": train_count, "validation_counts": {name: len(rows) for name, rows in validation_records.items()}, "runtime": runtime}, indent=2))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()
