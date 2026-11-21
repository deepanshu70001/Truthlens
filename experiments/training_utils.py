"""
TruthLens Phase 5: Reusable Training & Evaluation Utilities for SigLIP Experiments.

Provides:
1. Dataset loading with canonical preprocessing.
2. Evaluation loop computing all validation metrics (loss, acc, bal_acc, prec, rec, spec, f1, auc, pr_auc, mcc).
3. Reproducibility seed setup.
4. Cryptographic model identity and checkpoint hashing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = 250_000_000
log = logging.getLogger("phase5_utils")


def set_seed(seed: int = 42) -> None:
    """Set random seeds across python, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log.info("Set random seed to %d (deterministic flags enabled)", seed)


def get_canonical_transform() -> transforms.Compose:
    """Return the exact SigLIP frozen baseline preprocessing transform."""
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


class ImageManifestDataset(Dataset):
    """PyTorch Dataset loading images from a validated manifest CSV."""

    def __init__(
        self,
        manifest_path: Path,
        base_dir: Path,
        role_filter: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
    ):
        self.base_dir = base_dir
        self.transform = transform or get_canonical_transform()

        rows = []
        with open(manifest_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if role_filter and r.get("evaluation_role") != role_filter:
                    continue
                rows.append(r)

        self.rows = rows
        log.info(
            "Loaded %d images from %s (role_filter=%s)",
            len(self.rows),
            manifest_path.name,
            role_filter,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.rows[idx]
        rel_p = row.get("relative_path") or row.get("image_path")
        full_path = self.base_dir / rel_p

        # Ground truth binary: 1 = ai_generated, 0 = likely_authentic
        lbl_str = row.get("normalized_label") or row.get("label")
        gt = 1.0 if lbl_str == "ai_generated" else 0.0

        try:
            with Image.open(full_path) as img:
                rgb = img.convert("RGB")
                tensor = self.transform(rgb)
        except Exception as e:
            log.error("Failed to read image %s: %s", full_path, e)
            tensor = torch.zeros(3, 224, 224, dtype=torch.float32)

        return tensor, torch.tensor([gt], dtype=torch.float32), rel_p


class SiglipClassifierModel(nn.Module):
    """SigLIP vision model + binary linear classification head."""

    def __init__(
        self,
        vision_model: nn.Module,
        head: Optional[nn.Linear] = None,
    ):
        super().__init__()
        self.vision_model = vision_model
        self.head = head or nn.Linear(768, 1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pooler_output shape: [B, 768]
        feats = self.vision_model(pixel_values=pixel_values).pooler_output
        logits = self.head(feats)  # shape: [B, 1]
        return logits


def evaluate_model_on_validation(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """Run full evaluation on the validation dataset and return all metrics."""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.inference_mode():
        for tensors, targets, _ in val_loader:
            tensors = tensors.to(device)
            targets = targets.to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(pixel_values=tensors)
                loss = criterion(logits, targets)

            total_loss += loss.item() * len(tensors)
            all_logits.append(logits.float().cpu().numpy())
            all_targets.append(targets.float().cpu().numpy())

    avg_loss = total_loss / len(val_loader.dataset)
    y_true = np.concatenate(all_targets).flatten().astype(int)
    raw_logits = np.concatenate(all_logits).flatten()

    probs = 1.0 / (1.0 + np.exp(-np.clip(raw_logits, -100, 100)))
    y_pred = (probs >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    try:
        auc = float(roc_auc_score(y_true, probs))
    except Exception:
        auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, probs))
    except Exception:
        pr_auc = float(np.mean(y_true))

    return {
        "val_loss": round(avg_loss, 4),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "mcc": round(mcc, 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "sample_count": len(y_true),
    }


def compute_file_sha256(path: Path) -> str:
    """Compute exact SHA-256 hex digest of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()
