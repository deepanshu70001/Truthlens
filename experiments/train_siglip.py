"""
TruthLens Phase 5: Controlled SigLIP Model Training & Fine-Tuning Pipeline.

Runs controlled experiments (E1, E2, E3) strictly on the approved Phase 4 training dataset:
- E1: Frozen SigLIP backbone + new linear head (standard BCE loss).
- E2: Frozen SigLIP backbone + class-balanced loss (weighted BCE).
- E3: Partial fine-tuning (SigLIP layer 11 + head) with differential learning rates.

Validation-only model selection against Tiny-GenImage validation split (5,958 images).
LOCKED TEST SET IS NEVER EVALUATED IN THIS SCRIPT.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from transformers import SiglipVisionModel

from experiments.training_utils import (
    ImageManifestDataset,
    SiglipClassifierModel,
    compute_file_sha256,
    evaluate_model_on_validation,
    get_canonical_transform,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_siglip")


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


def build_model_and_optimizer(
    config: Dict[str, Any],
    exp_cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[SiglipClassifierModel, torch.optim.Optimizer, Dict[str, int]]:
    """Construct SigLIP classifier model, freeze appropriate layers, and configure AdamW."""
    backbone_id = config["model_architecture"]["backbone_model_id"]
    cache_dir = config["global_settings"].get("cache_dir", "models/cache")
    mode = exp_cfg.get("mode", "frozen_head")
    lr_head = float(exp_cfg.get("lr_head", 1e-4))
    lr_backbone = float(exp_cfg.get("lr_backbone", 1e-5))
    weight_decay = float(config["training_hyperparameters"].get("weight_decay", 0.01))

    logger.info("Loading pretrained SigLIP backbone %s from cache %s...", backbone_id, cache_dir)
    vision_model = SiglipVisionModel.from_pretrained(backbone_id, cache_dir=cache_dir)
    head = nn.Linear(config["model_architecture"]["embedding_dim"], 1)

    # Initialize head with normal distribution
    nn.init.normal_(head.weight, mean=0.0, std=0.02)
    nn.init.zeros_(head.bias)

    model = SiglipClassifierModel(vision_model, head)

    if mode in ("frozen_head", "balanced_head"):
        # Freeze entire vision backbone
        for param in model.vision_model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW(
            model.head.parameters(),
            lr=lr_head,
            weight_decay=weight_decay,
        )

    elif mode == "partial_finetuning":
        # Freeze vision backbone except the final encoder layer (layer 11)
        for param in model.vision_model.parameters():
            param.requires_grad = False

        # Unfreeze final transformer encoder layer (layer 11 of 0..11)
        final_layer = model.vision_model.vision_model.encoder.layers[-1]
        for param in final_layer.parameters():
            param.requires_grad = True

        # Unfreeze head
        for param in model.head.parameters():
            param.requires_grad = True

        param_groups = [
            {"params": final_layer.parameters(), "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": model.head.parameters(), "lr": lr_head, "weight_decay": weight_decay},
        ]
        optimizer = torch.optim.AdamW(param_groups)

    else:
        raise ValueError(f"Unknown experiment mode: {mode}")

    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    param_counts = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
    }

    logger.info(
        "Model setup (%s): Total params = %d, Trainable = %d (%.4f%%), Frozen = %d",
        mode,
        total_params,
        trainable_params,
        (trainable_params / total_params) * 100.0,
        frozen_params,
    )

    return model, optimizer, param_counts


def run_experiment(
    exp_name: str,
    config: Dict[str, Any],
    override_epochs: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute a single controlled training experiment."""
    exp_cfg = config["experiments"][exp_name]
    logger.info("=" * 70)
    logger.info("STARTING EXPERIMENT: %s (%s)", exp_name, exp_cfg.get("description"))
    logger.info("=" * 70)

    # Global configs
    seed = int(config["global_settings"].get("seed", 42))
    set_seed(seed)

    device_str = config["global_settings"].get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    logger.info("Active compute device: %s", device)

    use_fp16 = bool(config["global_settings"].get("fp16", True)) and (device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Dataset paths
    base_dir = Path(config["global_settings"].get("base_dir", "datasets/external/tiny_genimage"))
    train_manifest = Path(config["global_settings"]["training_manifest"])
    val_manifest = Path(config["global_settings"]["validation_manifest"])

    transform = get_canonical_transform()

    train_ds = ImageManifestDataset(train_manifest, base_dir, transform=transform)
    val_ds = ImageManifestDataset(val_manifest, base_dir, role_filter="validation", transform=transform)

    batch_size = int(config["training_hyperparameters"].get("batch_size", 16))
    grad_accum_steps = int(config["training_hyperparameters"].get("gradient_accumulation_steps", 2))
    num_workers = int(config["global_settings"].get("num_workers", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model & Optimizer
    model, optimizer, param_counts = build_model_and_optimizer(config, exp_cfg, device)

    # Loss Criterion
    loss_type = exp_cfg.get("loss_type", "bce")
    pos_weight_val = exp_cfg.get("pos_weight")
    if loss_type == "weighted_bce" and pos_weight_val:
        pos_weight_t = torch.tensor([float(pos_weight_val)], device=device)
        train_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
        logger.info("Using weighted BCEWithLogitsLoss with pos_weight = %.6f", float(pos_weight_val))
    else:
        train_criterion = nn.BCEWithLogitsLoss()
        logger.info("Using standard BCEWithLogitsLoss")

    eval_criterion = nn.BCEWithLogitsLoss()

    # Training parameters
    max_epochs = override_epochs or int(config["training_hyperparameters"].get("max_epochs", 10))
    patience = int(config["training_hyperparameters"].get("early_stopping_patience", 3))
    clip_norm = float(config["training_hyperparameters"].get("clip_grad_norm", 1.0))
    warmup_ratio = float(config["training_hyperparameters"].get("warmup_ratio", 0.05))

    total_update_steps = max_epochs * math.ceil(len(train_loader) / grad_accum_steps)
    warmup_steps = int(warmup_ratio * total_update_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    # Checkpoint dir
    chkpt_dir = Path(config["global_settings"]["checkpoints_dir"]) / exp_name
    chkpt_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    best_bal_acc = -1.0
    best_epoch = -1
    best_metrics: Dict[str, Any] = {}
    epochs_no_improve = 0

    mode = exp_cfg.get("mode", "frozen_head")
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_t0 = time.time()

        # Set training mode
        if mode in ("frozen_head", "balanced_head"):
            model.vision_model.eval()  # Keep frozen backbone strictly in eval mode
            model.head.train()
        else:
            # Partial fine-tuning: keep layers 0..10 in eval mode, layer 11 & head in train mode
            model.vision_model.eval()
            model.vision_model.vision_model.encoder.layers[-1].train()
            model.head.train()

        running_loss = 0.0
        correct_train = 0
        total_train = 0
        optimizer.zero_grad()

        for step, (images, targets, _) in enumerate(train_loader):
            if dry_run and step >= 2:
                break

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                logits = model(images)
                loss = train_criterion(logits, targets)
                accum_loss = loss / grad_accum_steps

            scaler.scale(accum_loss).backward()

            # Record batch train accuracy
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct_train += (preds == targets).sum().item()
            total_train += len(targets)
            running_loss += loss.item() * len(targets)

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                if use_fp16:
                    scaler.unscale_(optimizer)
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        train_loss = running_loss / max(1, total_train)
        train_acc = correct_train / max(1, total_train)

        # Run validation
        val_metrics = evaluate_model_on_validation(model, val_loader, eval_criterion, device)
        val_bal_acc = val_metrics["balanced_accuracy"]
        epoch_sec = time.time() - epoch_t0

        peak_vram_mb = 0.0
        if device.type == "cuda":
            peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": val_metrics["val_loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_specificity": val_metrics["specificity"],
            "val_f1": val_metrics["f1_score"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_mcc": val_metrics["mcc"],
            "epoch_duration_sec": round(epoch_sec, 2),
            "peak_vram_mb": peak_vram_mb,
            "lr": round(optimizer.param_groups[-1]["lr"], 7),
        }
        history.append(record)

        logger.info(
            "Epoch %d/%d (%.1fs) | Train Loss: %.4f, Acc: %.4f | Val Loss: %.4f, Acc: %.4f, BalAcc: %.4f, AUC: %.4f, F1: %.4f | VRAM: %.1f MB",
            epoch,
            max_epochs,
            epoch_sec,
            train_loss,
            train_acc,
            val_metrics["val_loss"],
            val_metrics["accuracy"],
            val_metrics["balanced_accuracy"],
            val_metrics["roc_auc"],
            val_metrics["f1_score"],
            peak_vram_mb,
        )

        # Check for improvement
        if val_bal_acc > best_bal_acc:
            best_bal_acc = val_bal_acc
            best_epoch = epoch
            best_metrics = val_metrics
            epochs_no_improve = 0

            # Save standalone head weights conforming to TruthLens Phase 1 contract
            head_weights_path = chkpt_dir / "head.pt"
            torch.save(
                {
                    "weight": model.head.weight.detach().cpu(),
                    "bias": model.head.bias.detach().cpu(),
                },
                head_weights_path,
            )
            head_sha256 = compute_file_sha256(head_weights_path)

            # Save full checkpoint
            chkpt_path = chkpt_dir / "best_checkpoint.pt"
            checkpoint_payload = {
                "experiment_name": exp_name,
                "epoch": epoch,
                "best_balanced_accuracy": val_bal_acc,
                "model_state_dict": model.state_dict(),
                "head_state_dict": model.head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": exp_cfg,
                "validation_metrics": val_metrics,
                "train_metrics": {"loss": round(train_loss, 4), "accuracy": round(train_acc, 4)},
                "model_identity": {
                    "model_id": f"truthlens/siglip-base-tiny-genimage-{exp_name}",
                    "backbone_model_id": config["model_architecture"]["backbone_model_id"],
                    "backbone_revision": config["model_architecture"]["backbone_revision"],
                    "head_sha256": head_sha256,
                    "preprocessing": config["model_architecture"]["preprocessing_id"],
                    "label_mapping": config["model_architecture"]["label_mapping"],
                },
            }
            torch.save(checkpoint_payload, chkpt_path)
            chkpt_sha256 = compute_file_sha256(chkpt_path)
            logger.info(
                "★ NEW BEST MODEL at Epoch %d (BalAcc: %.4f). Saved checkpoint to %s (SHA-256: %s)",
                epoch,
                val_bal_acc,
                chkpt_path.name,
                chkpt_sha256[:16],
            )
        else:
            epochs_no_improve += 1
            logger.info("No improvement in balanced accuracy for %d epoch(s) (patience: %d)", epochs_no_improve, patience)
            if epochs_no_improve >= patience and not dry_run:
                logger.info("Early stopping triggered at Epoch %d after %d epochs without improvement.", epoch, patience)
                break

    total_training_sec = round(time.time() - start_time, 2)
    peak_vram_total = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2) if device.type == "cuda" else 0.0

    # Save history and summary
    best_chkpt_file = chkpt_dir / "best_checkpoint.pt"
    best_head_file = chkpt_dir / "head.pt"
    best_chkpt_sha256 = compute_file_sha256(best_chkpt_file) if best_chkpt_file.exists() else ""
    best_head_sha256 = compute_file_sha256(best_head_file) if best_head_file.exists() else ""

    summary = {
        "experiment_name": exp_name,
        "description": exp_cfg.get("description"),
        "mode": exp_cfg.get("mode"),
        "best_epoch": best_epoch,
        "total_epochs_trained": len(history),
        "total_training_duration_sec": total_training_sec,
        "peak_vram_mb": peak_vram_total,
        "parameter_counts": param_counts,
        "best_validation_metrics": best_metrics,
        "best_checkpoint_path": str(best_chkpt_file),
        "best_checkpoint_sha256": best_chkpt_sha256,
        "best_head_path": str(best_head_file),
        "best_head_sha256": best_head_sha256,
        "model_identity": {
            "model_id": f"truthlens/siglip-base-tiny-genimage-{exp_name}",
            "backbone_model_id": config["model_architecture"]["backbone_model_id"],
            "backbone_revision": config["model_architecture"]["backbone_revision"],
            "head_sha256": best_head_sha256,
            "preprocessing": config["model_architecture"]["preprocessing_id"],
            "label_mapping": config["model_architecture"]["label_mapping"],
        },
    }

    with open(chkpt_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(chkpt_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved training history and experiment summary to %s", chkpt_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="TruthLens Phase 5 SigLIP Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/phase5_experiment_config.yaml",
        help="Path to Phase 5 YAML config",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "e1_frozen_head", "e2_balanced_training", "e3_partial_finetuning"],
        help="Specific experiment to run, or 'all'",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--dry-run", action="store_true", help="Run 2 batches for testing")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.experiment == "all":
        exps_to_run = ["e1_frozen_head", "e2_balanced_training", "e3_partial_finetuning"]
    else:
        exps_to_run = [args.experiment]

    summaries = {}
    for exp in exps_to_run:
        summary = run_experiment(exp, config, override_epochs=args.epochs, dry_run=args.dry_run)
        summaries[exp] = summary

    logger.info("=" * 70)
    logger.info("ALL REQUESTED EXPERIMENTS COMPLETE.")
    for name, s in summaries.items():
        val = s.get("best_validation_metrics", {})
        logger.info(
            "%-22s | Best Epoch: %d | BalAcc: %.4f | AUC: %.4f | F1: %.4f | Loss: %.4f",
            name,
            s.get("best_epoch", -1),
            val.get("balanced_accuracy", 0.0),
            val.get("roc_auc", 0.0),
            val.get("f1_score", 0.0),
            val.get("val_loss", 0.0),
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
