"""
TruthLens Phase 7: Controlled Training of Shortcut-Resistant Model Candidates.

Trains:
- S1: SigLIP-base frozen linear probe (K=5 precomputed augmented views per image)
- S2: SigLIP-base last 2 transformer blocks (layers 10 and 11) + head fine-tuning
- R1: ResNet-50 ImageNet V2 full fine-tuning
- R2: ResNet-50 ImageNet V2 stage 4 + head fine-tuning
- A1: Ablation run (SigLIP layers 10, 11 + head WITHOUT matched augmentation)

Enforces:
- Early stopping based strictly on composite validation score:
  Score_val = 0.5 * BalAcc(Tiny-val) + 0.5 * BalAcc(Frontier-val)
- Tie-breaker: Frontier-val ROC-AUC
- Zero test data used during training.
- Checkpoints registered with SHA-256 digests.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image, ImageFilter
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from transformers import SiglipVisionModel
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.training_utils import (
    SiglipClassifierModel,
    compute_file_sha256,
    set_seed,
)

Image.MAX_IMAGE_PIXELS = 250_000_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_phase7")


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine learning rate decay with linear warmup."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_model_transforms(backbone_type: str = "siglip", matched_aug: bool = True):
    """Return appropriate train and validation transforms."""
    if backbone_type == "siglip":
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:  # resnet50
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return val_transform, mean, std


def apply_matched_augmentation(
    img: Image.Image,
    p_jpeg: float = 0.70,
    p_webp: float = 0.15,
    p_png: float = 0.15,
    p_blur: float = 0.15,
) -> Image.Image:
    """Apply shortcut-resistant matched augmentation in-memory."""
    w, h = img.size

    # 1. Random rescale shorter side [224, 340]
    short_target = random.randint(224, 340)
    if w < h:
        new_w = short_target
        new_h = int(round(short_target * h / w))
    else:
        new_h = short_target
        new_w = int(round(short_target * w / h))

    interp = random.choice([Image.Resampling.BILINEAR, Image.Resampling.BICUBIC])
    resized = img.resize((new_w, new_h), interp)

    # 2. Random crop 224x224
    left = random.randint(0, new_w - 224)
    top = random.randint(0, new_h - 224)
    cropped = resized.crop((left, top, left + 224, top + 224))

    # 3. Optional Gaussian blur
    if random.random() < p_blur:
        radius = random.uniform(0.5, 1.2)
        cropped = cropped.filter(ImageFilter.GaussianBlur(radius))

    # 4. Transcode in memory
    u = random.random()
    buf = io.BytesIO()
    if u < p_jpeg:
        q = random.randint(40, 95)
        cropped.save(buf, format="JPEG", quality=q)
    elif u < p_jpeg + p_webp:
        q = random.randint(50, 95)
        cropped.save(buf, format="WEBP", quality=q, method=1)
    else:
        cropped.save(buf, format="PNG", compress_level=1)

    buf.seek(0)
    return Image.open(buf).convert("RGB")


class Phase7Dataset(Dataset):
    """Dataset with optional on-the-fly matched augmentation."""

    def __init__(
        self,
        manifest_path: Path,
        base_dir: Path,
        role_filter: Optional[str] = None,
        matched_aug: bool = False,
        backbone_type: str = "siglip",
    ):
        self.base_dir = base_dir
        self.matched_aug = matched_aug
        self.val_transform, self.mean, self.std = get_model_transforms(backbone_type, matched_aug)
        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        rows = []
        with open(manifest_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if role_filter and r.get("evaluation_role") != role_filter:
                    continue
                rows.append(r)

        self.rows = rows
        logger.info(
            "Phase7Dataset loaded %d samples from %s (role=%s, matched_aug=%s)",
            len(self.rows),
            manifest_path.name,
            role_filter,
            matched_aug,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.rows[idx]
        rel_p = row.get("relative_path") or row.get("image_path") or row.get("original_path")
        full_p = self.base_dir / rel_p

        lbl_str = row.get("normalized_label") or row.get("label")
        gt = 1.0 if lbl_str in ("ai_generated", "fake") else 0.0

        try:
            with Image.open(full_p) as img:
                rgb = img.convert("RGB")
                if self.matched_aug:
                    aug_img = apply_matched_augmentation(rgb)
                    tensor = self.to_tensor_norm(aug_img)
                else:
                    tensor = self.val_transform(rgb)
        except Exception as e:
            logger.error("Failed to load image %s: %s", full_p, e)
            tensor = torch.zeros(3, 224, 224, dtype=torch.float32)

        return tensor, torch.tensor([gt], dtype=torch.float32), rel_p


def evaluate_on_partition(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate model on a validation split and return all metrics."""
    model.eval()
    all_logits = []
    all_targets = []

    with torch.inference_mode():
        for tensors, targets, _ in val_loader:
            tensors = tensors.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(tensors)

            all_logits.append(logits.float().cpu().numpy())
            all_targets.append(targets.float().cpu().numpy())

    y_true = np.concatenate(all_targets).flatten().astype(int)
    raw_logits = np.concatenate(all_logits).flatten()
    probs = 1.0 / (1.0 + np.exp(-np.clip(raw_logits, -700, 700)))
    y_pred = (probs >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, probs))
    except Exception:
        auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, probs))
    except Exception:
        pr_auc = float(np.mean(y_true))

    return {
        "sample_count": len(y_true),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def evaluate_robustness_probe(
    model: nn.Module,
    samples: List[Tuple[Image.Image, int]],
    device: torch.device,
    to_tensor_norm,
) -> Dict[str, Any]:
    """Evaluate in-memory JPEG probes and report prediction flip rates."""
    model.eval()
    q_settings = [90, 75, 50]
    flips = {q: 0 for q in q_settings}
    shifts = {q: [] for q in q_settings}

    with torch.inference_mode():
        for img, gt in samples:
            # Baseline uncompressed
            t_orig = to_tensor_norm(img).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                z_orig = float(model(t_orig).item())
            p_orig = 1.0 / (1.0 + math.exp(-z_orig))

            for q in q_settings:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=q)
                buf.seek(0)
                re_img = Image.open(buf).convert("RGB")
                t_re = to_tensor_norm(re_img).unsqueeze(0).to(device)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                    z_re = float(model(t_re).item())
                p_re = 1.0 / (1.0 + math.exp(-z_re))

                shifts[q].append(abs(z_re - z_orig))
                if (p_re >= 0.5) != (p_orig >= 0.5):
                    flips[q] += 1

    n = len(samples)
    res = {}
    for q in q_settings:
        res[f"jpeg{q}"] = {
            "flip_rate_pct": round((flips[q] / max(1, n)) * 100.0, 2),
            "mean_abs_logit_shift": round(float(np.mean(shifts[q])), 4),
        }
    return res


def build_experiment_model(
    exp_cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[nn.Module, torch.optim.Optimizer, Dict[str, int]]:
    """Construct model and configure optimizer based on experiment specifications."""
    backbone_type = exp_cfg.get("backbone_type", "siglip")
    mode = exp_cfg.get("mode", "partial_finetuning")
    lr_head = float(exp_cfg.get("lr_head", 1e-4))
    lr_backbone = float(exp_cfg.get("lr_backbone", 1e-5))
    weight_decay = float(exp_cfg.get("weight_decay", 0.01))

    if backbone_type == "siglip":
        backbone_id = exp_cfg.get("backbone_model_id", "google/siglip-base-patch16-224")
        cache_dir = "models/cache"
        vision_model = SiglipVisionModel.from_pretrained(backbone_id, cache_dir=cache_dir)
        head = nn.Linear(768, 1)
        nn.init.normal_(head.weight, std=0.02)
        nn.init.zeros_(head.bias)
        model = SiglipClassifierModel(vision_model, head)

        if mode == "linear_probe":
            for p in model.vision_model.parameters():
                p.requires_grad = False
            for p in model.head.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr_head, weight_decay=weight_decay)

        elif mode == "partial_finetuning":
            for p in model.vision_model.parameters():
                p.requires_grad = False

            # Unfreeze last 2 layers: layer 10 and layer 11
            layers_to_unfreeze = [
                model.vision_model.vision_model.encoder.layers[-2],
                model.vision_model.vision_model.encoder.layers[-1],
            ]
            for layer in layers_to_unfreeze:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in model.head.parameters():
                p.requires_grad = True

            backbone_params = []
            for layer in layers_to_unfreeze:
                backbone_params.extend(list(layer.parameters()))

            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
                {"params": model.head.parameters(), "lr": lr_head, "weight_decay": weight_decay},
            ])
        else:
            raise ValueError(f"Unknown mode for SigLIP: {mode}")

    elif backbone_type == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(2048, 1)
        nn.init.normal_(model.fc.weight, std=0.02)
        nn.init.zeros_(model.fc.bias)

        if mode == "full_finetuning":
            for p in model.parameters():
                p.requires_grad = True
            backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]
            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
                {"params": model.fc.parameters(), "lr": lr_head, "weight_decay": weight_decay},
            ])

        elif mode == "stage4_finetuning":
            for p in model.parameters():
                p.requires_grad = False
            for p in model.layer4.parameters():
                p.requires_grad = True
            for p in model.fc.parameters():
                p.requires_grad = True

            optimizer = torch.optim.AdamW([
                {"params": model.layer4.parameters(), "lr": lr_backbone, "weight_decay": weight_decay},
                {"params": model.fc.parameters(), "lr": lr_head, "weight_decay": weight_decay},
            ])
        else:
            raise ValueError(f"Unknown mode for ResNet: {mode}")

    else:
        raise ValueError(f"Unknown backbone_type: {backbone_type}")

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_counts = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
    }
    logger.info("Model %s: %d / %d trainable parameters (%.4f%%)", exp_cfg.get("model_id"), trainable_params, total_params, (trainable_params / total_params) * 100)
    return model, optimizer, param_counts


def run_linear_probe_s1(
    exp_name: str,
    exp_cfg: Dict[str, Any],
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Execute S1: SigLIP frozen linear probe on K=5 precomputed augmented views."""
    logger.info("=" * 70)
    logger.info("STARTING S1 LINEAR PROBE WITH K=5 PRECOMPUTED AUGMENTED VIEWS")
    logger.info("=" * 70)

    t0_start = time.time()
    backbone_id = exp_cfg.get("backbone_model_id", "google/siglip-base-patch16-224")
    cache_dir = config["global_settings"].get("cache_dir", "models/cache")
    vision_model = SiglipVisionModel.from_pretrained(backbone_id, cache_dir=cache_dir).to(device)
    vision_model.eval()
    for p in vision_model.parameters():
        p.requires_grad = False

    head = nn.Linear(768, 1).to(device)
    nn.init.normal_(head.weight, std=0.02)
    nn.init.zeros_(head.bias)

    total_params = sum(p.numel() for p in vision_model.parameters()) + sum(p.numel() for p in head.parameters())
    trainable_params = sum(p.numel() for p in head.parameters())
    param_counts = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
    }

    # Precompute K=5 augmented embeddings for training set
    train_manifest = Path(config["datasets"]["training"]["manifest"])
    base_dir = Path(config["datasets"]["training"]["base_dir"])
    k_views = int(exp_cfg.get("k_augmented_views", 5))

    rows = []
    with open(train_manifest, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    _, mean, std = get_model_transforms("siglip")
    to_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_norm = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    logger.info("Precomputing K=%d views for %d training samples...", k_views, len(rows))
    train_feats_list = []
    train_targets_list = []
    batch_tensors = []
    batch_labels = []

    with torch.inference_mode():
        for r_idx, r in enumerate(rows):
            rel_p = r.get("relative_path") or r.get("image_path")
            full_p = base_dir / rel_p
            lbl = 1.0 if (r.get("normalized_label") or r.get("label")) == "ai_generated" else 0.0

            try:
                with Image.open(full_p) as img:
                    rgb = img.convert("RGB")
                    for _ in range(k_views):
                        aug_img = apply_matched_augmentation(rgb)
                        t = to_norm(aug_img)
                        batch_tensors.append(t)
                        batch_labels.append(lbl)
            except Exception as e:
                logger.error("Failed to read %s: %s", full_p, e)

            if len(batch_tensors) >= 64 or (r_idx == len(rows) - 1 and len(batch_tensors) > 0):
                batch_t = torch.stack(batch_tensors).to(device)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                    feats = vision_model(pixel_values=batch_t).pooler_output
                train_feats_list.append(feats.float().cpu())
                train_targets_list.extend(batch_labels)
                batch_tensors = []
                batch_labels = []

    X_train = torch.cat(train_feats_list, dim=0)
    y_train = torch.tensor(train_targets_list, dtype=torch.float32).unsqueeze(1)
    logger.info("Precomputed training features: shape %s, targets %s", X_train.shape, y_train.shape)

    # Precompute validation features (canonical transform, 1 view per image)
    def extract_val_features(manifest_p: Path, base_p: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        v_rows = []
        with open(manifest_p, "r", encoding="utf-8-sig") as vf:
            for vr in csv.DictReader(vf):
                if vr.get("evaluation_role") == "validation":
                    v_rows.append(vr)
        f_list = []
        l_list = []
        b_t = []
        b_l = []
        with torch.inference_mode():
            for v_idx, vr in enumerate(v_rows):
                rel = vr.get("relative_path") or vr.get("image_path")
                fp = base_p / rel
                v_lbl = 1.0 if (vr.get("normalized_label") or vr.get("label")) == "ai_generated" else 0.0
                try:
                    with Image.open(fp) as vi:
                        b_t.append(val_norm(vi.convert("RGB")))
                        b_l.append(v_lbl)
                except Exception as ve:
                    logger.error("Val load error %s: %s", fp, ve)

                if len(b_t) >= 64 or (v_idx == len(v_rows) - 1 and len(b_t) > 0):
                    bt_tensor = torch.stack(b_t).to(device)
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                        vf = vision_model(pixel_values=bt_tensor).pooler_output
                    f_list.append(vf.float().cpu())
                    l_list.extend(b_l)
                    b_t = []
                    b_l = []
        return torch.cat(f_list, dim=0), torch.tensor(l_list, dtype=torch.float32).unsqueeze(1)

    logger.info("Precomputing validation features...")
    val_tiny_manifest = Path(config["datasets"]["validation_tiny"]["manifest"])
    val_tiny_base = Path(config["datasets"]["validation_tiny"]["base_dir"])
    val_frontier_manifest = Path(config["datasets"]["validation_frontier"]["manifest"])
    val_frontier_base = Path(config["datasets"]["validation_frontier"]["base_dir"])

    X_val_tiny, y_val_tiny = extract_val_features(val_tiny_manifest, val_tiny_base)
    X_val_frontier, y_val_frontier = extract_val_features(val_frontier_manifest, val_frontier_base)
    logger.info("Validation features precomputed: Tiny %s, Frontier %s", X_val_tiny.shape, X_val_frontier.shape)

    # Train linear head
    lr_head = float(exp_cfg.get("lr_head", 1e-4))
    weight_decay = float(exp_cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr_head, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(X_train, y_train)
    batch_size = int(exp_cfg.get("batch_size", 128))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    max_epochs = int(exp_cfg.get("epochs", 10))
    patience = int(exp_cfg.get("patience", 3))
    total_steps = max_epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    chkpt_dir = Path(config["global_settings"]["checkpoints_dir"]) / exp_name
    chkpt_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_features(feats_in: torch.Tensor, targets_in: torch.Tensor) -> Dict[str, Any]:
        head.eval()
        with torch.inference_mode():
            logits = head(feats_in.to(device)).flatten().float().cpu().numpy()
        y_true = targets_in.flatten().numpy().astype(int)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -700, 700)))
        y_pred = (probs >= 0.5).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        acc = float(accuracy_score(y_true, y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            auc = float(roc_auc_score(y_true, probs))
        except Exception:
            auc = 0.5
        try:
            pr_auc = float(average_precision_score(y_true, probs))
        except Exception:
            pr_auc = float(np.mean(y_true))

        return {
            "sample_count": len(y_true),
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "specificity": round(spec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "pr_auc": round(pr_auc, 4),
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }

    history = []
    best_score = -1.0
    best_epoch = -1
    best_tiny_metrics = {}
    best_frontier_metrics = {}
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        t0_ep = time.time()
        head.train()
        running_loss = 0.0
        n_samples = 0

        for f_batch, y_batch in train_loader:
            f_batch = f_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = head(f_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * len(y_batch)
            n_samples += len(y_batch)

        train_loss = running_loss / max(1, n_samples)
        m_tiny = evaluate_features(X_val_tiny, y_val_tiny)
        m_frontier = evaluate_features(X_val_frontier, y_val_frontier)
        bal_tiny = m_tiny["balanced_accuracy"]
        bal_frontier = m_frontier["balanced_accuracy"]
        composite_score = round(0.50 * bal_tiny + 0.50 * bal_frontier, 4)
        ep_dur = round(time.time() - t0_ep, 2)
        peak_vram = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2) if device.type == "cuda" else 0.0

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "composite_score": composite_score,
            "tiny_val_balanced_accuracy": bal_tiny,
            "tiny_val_roc_auc": m_tiny["roc_auc"],
            "frontier_val_balanced_accuracy": bal_frontier,
            "frontier_val_roc_auc": m_frontier["roc_auc"],
            "epoch_duration_sec": ep_dur,
            "peak_vram_mb": peak_vram,
        }
        history.append(record)
        logger.info(
            "Epoch %d/%d (%.1fs) | Train Loss: %.4f | Score: %.4f (Tiny BalAcc: %.4f, Frontier BalAcc: %.4f) | VRAM: %.1f MB",
            epoch, max_epochs, ep_dur, train_loss, composite_score, bal_tiny, bal_frontier, peak_vram,
        )

        is_better = False
        if composite_score > best_score:
            is_better = True
        elif composite_score == best_score and m_frontier["roc_auc"] > best_frontier_metrics.get("roc_auc", 0.0):
            is_better = True

        if is_better:
            best_score = composite_score
            best_epoch = epoch
            best_tiny_metrics = m_tiny
            best_frontier_metrics = m_frontier
            epochs_no_improve = 0

            # Assemble full model state dict
            full_model = SiglipClassifierModel(vision_model, head)
            chkpt_file = chkpt_dir / "best_checkpoint.pt"
            torch.save({
                "experiment_name": exp_name,
                "model_id": exp_cfg["model_id"],
                "backbone_type": "siglip",
                "epoch": epoch,
                "composite_score": composite_score,
                "model_state_dict": full_model.state_dict(),
                "config": exp_cfg,
                "tiny_val_metrics": m_tiny,
                "frontier_val_metrics": m_frontier,
            }, chkpt_file)

            head_file = chkpt_dir / "head.pt"
            torch.save({
                "weight": head.weight.detach().cpu(),
                "bias": head.bias.detach().cpu(),
            }, head_file)
            logger.info("★ NEW BEST MODEL at Epoch %d (Score: %.4f). Saved checkpoint to %s", epoch, composite_score, chkpt_file.name)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered at Epoch %d after %d non-improving epochs.", epoch, patience)
                break

    chkpt_path = chkpt_dir / "best_checkpoint.pt"
    head_path = chkpt_dir / "head.pt"
    chkpt_sha = compute_file_sha256(chkpt_path) if chkpt_path.exists() else ""
    head_sha = compute_file_sha256(head_path) if head_path.exists() else ""

    summary = {
        "experiment_name": exp_name,
        "model_id": exp_cfg["model_id"],
        "backbone_type": "siglip",
        "matched_augmentation": True,
        "k_augmented_views": k_views,
        "best_epoch": best_epoch,
        "best_composite_score": best_score,
        "total_training_duration_sec": round(time.time() - t0_start, 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2) if device.type == "cuda" else 0.0,
        "parameter_counts": param_counts,
        "best_tiny_val_metrics": best_tiny_metrics,
        "best_frontier_val_metrics": best_frontier_metrics,
        "checkpoint_path": str(chkpt_path),
        "checkpoint_sha256": chkpt_sha,
        "head_path": str(head_path),
        "head_sha256": head_sha,
    }

    with open(chkpt_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(chkpt_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_training_experiment(
    exp_name: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute complete training run for a single model configuration."""
    exp_cfg = config["experiments"][exp_name]
    logger.info("=" * 70)
    logger.info("STARTING PHASE 7 TRAINING: %s (%s)", exp_name, exp_cfg.get("model_id"))
    logger.info("=" * 70)

    set_seed(int(config["global_settings"].get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Route S1 to specialized linear probe flow
    if exp_name == "s1_siglip_linear_probe":
        return run_linear_probe_s1(exp_name, exp_cfg, config, device)

    backbone_type = exp_cfg.get("backbone_type", "siglip")
    matched_aug = exp_cfg.get("matched_augmentation", True)

    train_manifest = Path(config["datasets"]["training"]["manifest"])
    base_dir = Path(config["datasets"]["training"]["base_dir"])

    val_tiny_manifest = Path(config["datasets"]["validation_tiny"]["manifest"])
    val_tiny_base = Path(config["datasets"]["validation_tiny"]["base_dir"])

    val_frontier_manifest = Path(config["datasets"]["validation_frontier"]["manifest"])
    val_frontier_base = Path(config["datasets"]["validation_frontier"]["base_dir"])

    train_ds = Phase7Dataset(train_manifest, base_dir, matched_aug=matched_aug, backbone_type=backbone_type)
    val_tiny_ds = Phase7Dataset(val_tiny_manifest, val_tiny_base, role_filter="validation", matched_aug=False, backbone_type=backbone_type)
    val_frontier_ds = Phase7Dataset(val_frontier_manifest, val_frontier_base, role_filter="validation", matched_aug=False, backbone_type=backbone_type)

    batch_size = int(exp_cfg.get("batch_size", 16))
    grad_accum_steps = int(exp_cfg.get("gradient_accumulation_steps", 2))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_tiny_loader = DataLoader(val_tiny_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    val_frontier_loader = DataLoader(val_frontier_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    model, optimizer, param_counts = build_experiment_model(exp_cfg, device)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    max_epochs = int(exp_cfg.get("epochs", 10))
    patience = int(exp_cfg.get("patience", 3))

    total_update_steps = max_epochs * math.ceil(len(train_loader) / grad_accum_steps)
    warmup_steps = int(0.05 * total_update_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)

    chkpt_dir = Path(config["global_settings"]["checkpoints_dir"]) / exp_name
    chkpt_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_score = -1.0
    best_epoch = -1
    best_tiny_metrics = {}
    best_frontier_metrics = {}
    epochs_no_improve = 0
    t0_train = time.time()

    for epoch in range(1, max_epochs + 1):
        t0_epoch = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for step, (images, targets, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, targets)
                accum_loss = loss / grad_accum_steps

            scaler.scale(accum_loss).backward()
            running_loss += loss.item() * len(targets)
            n_batches += len(targets)

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        train_loss = running_loss / max(1, n_batches)

        # Validation on both splits
        m_tiny = evaluate_on_partition(model, val_tiny_loader, device)
        m_frontier = evaluate_on_partition(model, val_frontier_loader, device)

        bal_tiny = m_tiny["balanced_accuracy"]
        bal_frontier = m_frontier["balanced_accuracy"]
        composite_score = round(0.50 * bal_tiny + 0.50 * bal_frontier, 4)
        epoch_dur = round(time.time() - t0_epoch, 2)
        peak_vram = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2) if device.type == "cuda" else 0.0

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "composite_score": composite_score,
            "tiny_val_balanced_accuracy": bal_tiny,
            "tiny_val_roc_auc": m_tiny["roc_auc"],
            "frontier_val_balanced_accuracy": bal_frontier,
            "frontier_val_roc_auc": m_frontier["roc_auc"],
            "epoch_duration_sec": epoch_dur,
            "peak_vram_mb": peak_vram,
        }
        history.append(record)

        logger.info(
            "Epoch %d/%d (%.1fs) | Train Loss: %.4f | Composite Score: %.4f (Tiny BalAcc: %.4f, Frontier BalAcc: %.4f) | VRAM: %.1f MB",
            epoch,
            max_epochs,
            epoch_dur,
            train_loss,
            composite_score,
            bal_tiny,
            bal_frontier,
            peak_vram,
        )

        is_better = False
        if composite_score > best_score:
            is_better = True
        elif composite_score == best_score and m_frontier["roc_auc"] > best_frontier_metrics.get("roc_auc", 0.0):
            is_better = True

        if is_better:
            best_score = composite_score
            best_epoch = epoch
            best_tiny_metrics = m_tiny
            best_frontier_metrics = m_frontier
            epochs_no_improve = 0

            # Save best checkpoint
            chkpt_file = chkpt_dir / "best_checkpoint.pt"
            payload = {
                "experiment_name": exp_name,
                "model_id": exp_cfg["model_id"],
                "backbone_type": backbone_type,
                "epoch": epoch,
                "composite_score": composite_score,
                "model_state_dict": model.state_dict(),
                "config": exp_cfg,
                "tiny_val_metrics": m_tiny,
                "frontier_val_metrics": m_frontier,
            }
            torch.save(payload, chkpt_file)

            # Save head weights separately
            head_file = chkpt_dir / "head.pt"
            if backbone_type == "siglip":
                head_dict = {"weight": model.head.weight.detach().cpu(), "bias": model.head.bias.detach().cpu()}
            else:
                head_dict = {"weight": model.fc.weight.detach().cpu(), "bias": model.fc.bias.detach().cpu()}
            torch.save(head_dict, head_file)

            logger.info("★ NEW BEST MODEL at Epoch %d (Score: %.4f). Saved checkpoint to %s", epoch, composite_score, chkpt_file.name)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered at Epoch %d after %d non-improving epochs.", epoch, patience)
                break

    # Save summary and history
    chkpt_path = chkpt_dir / "best_checkpoint.pt"
    head_path = chkpt_dir / "head.pt"
    chkpt_sha = compute_file_sha256(chkpt_path) if chkpt_path.exists() else ""
    head_sha = compute_file_sha256(head_path) if head_path.exists() else ""

    summary = {
        "experiment_name": exp_name,
        "model_id": exp_cfg["model_id"],
        "backbone_type": backbone_type,
        "matched_augmentation": matched_aug,
        "best_epoch": best_epoch,
        "best_composite_score": best_score,
        "total_training_duration_sec": round(time.time() - t0_train, 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2) if device.type == "cuda" else 0.0,
        "parameter_counts": param_counts,
        "best_tiny_val_metrics": best_tiny_metrics,
        "best_frontier_val_metrics": best_frontier_metrics,
        "checkpoint_path": str(chkpt_path),
        "checkpoint_sha256": chkpt_sha,
        "head_path": str(head_path),
        "head_sha256": head_sha,
    }

    with open(chkpt_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(chkpt_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved experiment summary and history for %s", exp_name)
    return summary

    summary = {
        "experiment_name": exp_name,
        "model_id": exp_cfg["model_id"],
        "backbone_type": backbone_type,
        "matched_augmentation": matched_aug,
        "best_epoch": best_epoch,
        "best_composite_score": best_score,
        "total_training_duration_sec": round(time.time() - t0_train, 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2) if device.type == "cuda" else 0.0,
        "parameter_counts": param_counts,
        "best_tiny_val_metrics": best_tiny_metrics,
        "best_frontier_val_metrics": best_frontier_metrics,
        "checkpoint_path": str(chkpt_path),
        "checkpoint_sha256": chkpt_sha,
        "head_path": str(head_path),
        "head_sha256": head_sha,
    }

    with open(chkpt_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(chkpt_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved experiment summary and history for %s", exp_name)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="TruthLens Phase 7 Training Pipeline")
    parser.add_argument("--config", type=str, default="experiments/configs/phase7_config.yaml")
    parser.add_argument("--experiment", type=str, default="all", choices=["all", "s1_siglip_linear_probe", "s2_siglip_partial_ft", "r1_resnet50_full_ft", "r2_resnet50_last_stage", "a1_siglip_no_aug_ablation"])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.experiment == "all":
        exps = ["s1_siglip_linear_probe", "s2_siglip_partial_ft", "r1_resnet50_full_ft", "r2_resnet50_last_stage", "a1_siglip_no_aug_ablation"]
    else:
        exps = [args.experiment]

    all_summaries = {}
    for exp in exps:
        s = run_training_experiment(exp, config)
        all_summaries[exp] = s

    logger.info("=" * 70)
    logger.info("PHASE 7 TRAINING COMPLETE ACROSS %d EXPERIMENTS", len(all_summaries))
    for name, s in all_summaries.items():
        logger.info(
            "%-26s | Best Ep: %2d | Comp Score: %.4f | Tiny BalAcc: %.4f | Front BalAcc: %.4f | Front AUC: %.4f",
            name,
            s["best_epoch"],
            s["best_composite_score"],
            s["best_tiny_val_metrics"].get("balanced_accuracy", 0.0),
            s["best_frontier_val_metrics"].get("balanced_accuracy", 0.0),
            s["best_frontier_val_metrics"].get("roc_auc", 0.0),
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
