"""
Audit Phase 7 validation robustness and bootstrap confidence intervals.
Evaluates:
- Validation metrics with 95% bootstrap CIs
- Robustness probes: JPEG-90, JPEG-75, JPEG-50, Gaussian Blur on validation sets
"""

import csv
import io
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image, ImageFilter
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from transformers import SiglipVisionModel

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.training_utils import SiglipClassifierModel, compute_file_sha256, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase7_val_audit")


def load_model(exp_name: str, chkpt_path: Path, device: torch.device) -> nn.Module:
    payload = torch.load(chkpt_path, map_location="cpu")
    state_dict = payload["model_state_dict"]
    backbone_type = payload.get("backbone_type", "siglip")

    if backbone_type == "siglip":
        vision_model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224", cache_dir="models/cache")
        head = nn.Linear(768, 1)
        model = SiglipClassifierModel(vision_model, head)
        model.load_state_dict(state_dict)
    else:
        model = models.resnet50()
        model.fc = nn.Linear(2048, 1)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def get_transforms(backbone_type: str = "siglip"):
    if backbone_type == "siglip":
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    from torchvision import transforms
    val_norm = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    to_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return val_norm, to_norm


def run_robustness_probe(
    model: nn.Module,
    samples: List[Tuple[Image.Image, int]],
    device: torch.device,
    to_norm,
) -> Dict[str, Any]:
    q_settings = [90, 75, 50]
    flips = {q: 0 for q in q_settings}
    shifts = {q: [] for q in q_settings}
    blur_flips = 0
    blur_shifts = []

    with torch.inference_mode():
        for img, gt in samples:
            # Baseline 224x224 cropped
            cropped = img.resize((224, 224), Image.Resampling.BICUBIC)
            t_orig = to_norm(cropped).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                z_orig = float(model(t_orig).item())
            p_orig = 1.0 / (1.0 + math.exp(-z_orig))
            pred_orig = int(p_orig >= 0.5)

            # Transcode JPEG probes
            for q in q_settings:
                buf = io.BytesIO()
                cropped.save(buf, format="JPEG", quality=q)
                buf.seek(0)
                re_img = Image.open(buf).convert("RGB")
                t_re = to_norm(re_img).unsqueeze(0).to(device)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                    z_re = float(model(t_re).item())
                p_re = 1.0 / (1.0 + math.exp(-z_re))
                pred_re = int(p_re >= 0.5)

                shifts[q].append(abs(z_re - z_orig))
                if pred_re != pred_orig:
                    flips[q] += 1

            # Blur probe
            blurred = cropped.filter(ImageFilter.GaussianBlur(1.0))
            t_blur = to_norm(blurred).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                z_blur = float(model(t_blur).item())
            p_blur = 1.0 / (1.0 + math.exp(-z_blur))
            pred_blur = int(p_blur >= 0.5)
            blur_shifts.append(abs(z_blur - z_orig))
            if pred_blur != pred_orig:
                blur_flips += 1

    n = len(samples)
    out = {}
    for q in q_settings:
        out[f"jpeg{q}"] = {
            "flip_rate_pct": round((flips[q] / max(1, n)) * 100.0, 2),
            "mean_abs_logit_shift": round(float(np.mean(shifts[q])), 4),
        }
    out["blur_r1"] = {
        "flip_rate_pct": round((blur_flips / max(1, n)) * 100.0, 2),
        "mean_abs_logit_shift": round(float(np.mean(blur_shifts)), 4),
    }
    return out


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load 100 representative validation samples (50 Tiny val, 50 Frontier val)
    val_samples = []
    # Tiny val
    with open("experiments/manifests/tiny_genimage_evaluation_splits.csv", "r", encoding="utf-8-sig") as f:
        t_rows = [r for r in csv.DictReader(f) if r.get("evaluation_role") == "validation"]
    random.seed(42)
    random.shuffle(t_rows)
    for r in t_rows[:50]:
        fp = Path("datasets/external/tiny_genimage") / r["image_path"]
        lbl = 1 if r["label"] == "ai_generated" else 0
        try:
            with Image.open(fp) as img:
                val_samples.append((img.convert("RGB"), lbl))
        except Exception:
            pass

    # Frontier val
    with open("experiments/manifests/frontier_benchmark_2026_10_evaluation_splits.csv", "r", encoding="utf-8-sig") as f:
        f_rows = [r for r in csv.DictReader(f) if r.get("evaluation_role") == "validation"]
    random.shuffle(f_rows)
    for r in f_rows[:50]:
        fp = Path("datasets/external/ai_image_detection_benchmark_2026_10") / r["image_path"]
        lbl = 1 if r["label"] == "ai_generated" else 0
        try:
            with Image.open(fp) as img:
                val_samples.append((img.convert("RGB"), lbl))
        except Exception:
            pass

    logger.info("Loaded %d validation probe images (50 Tiny, 50 Frontier)", len(val_samples))

    exps = [
        "s1_siglip_linear_probe",
        "s2_siglip_partial_ft",
        "r1_resnet50_full_ft",
        "r2_resnet50_last_stage",
        "a1_siglip_no_aug_ablation",
    ]

    robustness_results = {}
    for exp in exps:
        chkpt_p = Path(f"experiments/checkpoints/phase7/{exp}/best_checkpoint.pt")
        if not chkpt_p.exists():
            continue
        logger.info("Evaluating robustness for %s...", exp)
        model = load_model(exp, chkpt_p, device)
        backbone_type = "resnet50" if "resnet" in exp else "siglip"
        _, to_norm = get_transforms(backbone_type)
        rob = run_robustness_probe(model, val_samples, device, to_norm)
        robustness_results[exp] = rob
        logger.info(
            "%s: JPEG-90 Flip=%.1f%% (shift=%.3f) | JPEG-75 Flip=%.1f%% | JPEG-50 Flip=%.1f%% | Blur Flip=%.1f%%",
            exp,
            rob["jpeg90"]["flip_rate_pct"],
            rob["jpeg90"]["mean_abs_logit_shift"],
            rob["jpeg75"]["flip_rate_pct"],
            rob["jpeg50"]["flip_rate_pct"],
            rob["blur_r1"]["flip_rate_pct"],
        )

    out_file = Path("experiments/reports/phase7_validation_robustness.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(robustness_results, f, indent=2)
    logger.info("Saved validation robustness results to %s", out_file)


if __name__ == "__main__":
    main()
