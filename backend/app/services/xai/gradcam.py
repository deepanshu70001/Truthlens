"""Grad-CAM attribution for the configured image classifier (SigLIP ViT)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from backend.app.config import settings
from backend.app.core.logging import get_logger

log = get_logger("xai")
ATTRIBUTION_CAVEAT = "Attribution highlights regions that influenced this model prediction; it does not prove manipulation or model correctness."


@dataclass
class XAIResult:
    heatmap_path: Optional[str] = None
    overlay_path: Optional[str] = None
    suspicious_regions: List[Dict[str, Any]] = field(default_factory=list)
    method: str = "grad_cam"
    target_class: str = ""
    target_score: Optional[float] = None
    runtime_ms: int = 0
    warnings: List[str] = field(default_factory=list)
    status: str = "unavailable"
    model_name: str = ""
    heatmap_url: Optional[str] = None
    overlay_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status, "model": self.model_name, "method": self.method,
            "target_class": self.target_class,
            "target_score": round(self.target_score, 4) if self.target_score is not None else None,
            "heatmap_path": self.heatmap_path, "overlay_path": self.overlay_path,
            "heatmap_url": self.heatmap_url,
            "overlay_url": self.overlay_url,
            "suspicious_regions": self.suspicious_regions, "runtime_ms": self.runtime_ms,
            "warnings": self.warnings + ([ATTRIBUTION_CAVEAT] if self.status == "completed" else []),
            "note": ATTRIBUTION_CAVEAT,
        }


class GradCAM:
    """Grad-CAM using the last SigLIP vision block's layer_norm2 token features."""
    def __init__(self, model, target_layer=None):
        self.model = model
        self.activations = None
        self.gradients = None
        self.target_layer = target_layer or self._find_target_layer()
        self._hooks = []
        self.target_class: Optional[int] = None
        self.target_score: Optional[float] = None
        if self.target_layer is not None:
            self._hooks.append(self.target_layer.register_forward_hook(self._save_activation))
            self._hooks.append(self.target_layer.register_full_backward_hook(self._save_gradient))

    def _find_target_layer(self):
        vision = getattr(self.model, "vision_model", None)
        encoder = getattr(vision, "encoder", None)
        layers = getattr(encoder, "layers", None)
        if layers:
            last = layers[-1]
            # SigLIP applies layer_norm2 to tokens on the path to its attention pooler.
            return getattr(last, "layer_norm2", last)
        return None

    def _save_activation(self, module, inputs, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        self.activations = value

    def _save_gradient(self, module, grad_input, grad_output):
        value = grad_output[0] if grad_output else None
        self.gradients = value

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None):
        if self.target_layer is None:
            raise ValueError("Unsupported model architecture: no compatible SigLIP/ViT target layer found")
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError("Grad-CAM expects one image tensor shaped [1,C,H,W]")
        self.model.eval()
        self.activations = self.gradients = None
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            output = self.model(pixel_values=input_tensor)
            logits = getattr(output, "logits", None)
            if logits is None or logits.ndim != 2 or logits.shape[0] != 1 or not torch.isfinite(logits).all():
                raise ValueError("Model returned invalid classification logits")
            if target_class is None:
                target_class = int(logits[0].argmax().item())
            if not 0 <= int(target_class) < logits.shape[1]:
                raise ValueError("Target class is outside the model output range")
            self.target_class = int(target_class)
            self.target_score = float(torch.softmax(logits.detach().float(), dim=1)[0, int(target_class)].item())
            logits[0, int(target_class)].backward()

        activations, gradients = self.activations, self.gradients
        if activations is None or gradients is None or activations.shape != gradients.shape or activations.ndim != 3:
            raise ValueError("Target layer did not provide compatible token activations and gradients")
        if not torch.isfinite(activations).all() or not torch.isfinite(gradients).all():
            raise ValueError("Target layer returned non-finite values")
        # SigLIP emits a square patch grid without CLS. Support ViT with CLS too.
        tokens = activations.shape[1]
        side = int(np.sqrt(tokens))
        start = 0
        if side * side != tokens:
            side = int(np.sqrt(tokens - 1))
            if side * side != tokens - 1:
                raise ValueError(f"Token sequence length {tokens} cannot be mapped to a square attribution grid")
            start = 1
        features = activations[0, start:, :]
        grads = gradients[0, start:, :]
        if features.shape[0] != side * side:
            raise ValueError("Attribution token grid shape is inconsistent")
        weights = grads.mean(dim=0)
        cam = torch.relu((features * weights).sum(dim=1)).reshape(side, side)
        cam = cam.detach().float().cpu().numpy()
        if not np.isfinite(cam).all():
            raise ValueError("Attribution map contains non-finite values")
        maximum = float(cam.max())
        if maximum <= 0:
            raise ValueError("Attribution map has no positive activation")
        cam = np.clip(cam / maximum, 0.0, 1.0).astype(np.float32)
        return cam

    def cleanup(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def generate_heatmap(model, processor, image: Image.Image, analysis_id: str, device: torch.device, target_class: Optional[int] = None, model_name: Optional[str] = None) -> XAIResult:
    started = time.perf_counter()
    grad_cam = None
    try:
        grad_cam = GradCAM(model)
        if grad_cam.target_layer is None:
            raise ValueError("Unsupported classifier architecture for Grad-CAM")
        image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        input_tensor = inputs["pixel_values"].to(device)
        input_tensor.requires_grad_(True)
        cam = grad_cam.generate(input_tensor, target_class)
        # Determine the explained class and probability using the same forward result
        # would require reusing graph outputs; target_class is selected from the detector.
        id2label = getattr(model.config, "id2label", {}) or {}
        labels = {int(k): str(v) for k, v in id2label.items()}
        target_class = grad_cam.target_class if grad_cam.target_class is not None else target_class
        target_label = labels.get(int(target_class), str(target_class))
        if not np.isfinite(cam).all():
            raise ValueError("Attribution map is invalid")
        rgb = np.asarray(image)
        height, width = rgb.shape[:2]
        cam_full = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
        cam_full = np.clip(cam_full, 0.0, 1.0)
        color = cv2.cvtColor(cv2.applyColorMap(np.uint8(cam_full * 255), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(rgb, 0.6, color, 0.4, 0)
        settings.ensure_storage_dirs()
        analysis_dir = settings.heatmaps_dir / analysis_id
        analysis_dir.mkdir(parents=True, exist_ok=True)
        raw_path = analysis_dir / "attribution.png"
        overlay_path = analysis_dir / "overlay.png"
        Image.fromarray(np.uint8(cam_full * 255)).save(raw_path)
        Image.fromarray(overlay).save(overlay_path)
        # These are attribution regions, not manipulation masks.
        mask = np.uint8(cam_full >= 0.7)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area >= width * height * 0.005:
                regions.append({"bbox": [x, y, x+w, y+h], "area_fraction": round(area/(width*height), 5), "max_activation": round(float(cam_full[y:y+h, x:x+w].max()), 5)})
        result = XAIResult(
            heatmap_path=str(raw_path), overlay_path=str(overlay_path),
            suspicious_regions=regions, target_class=target_label,
            target_score=grad_cam.target_score, status="completed", model_name=model_name or getattr(model.config, "_name_or_path", type(model).__name__),
            runtime_ms=int((time.perf_counter()-started)*1000), warnings=[],
        )
        result.heatmap_url = f"/api/v1/analysis/{analysis_id}/heatmap?variant=raw"
        result.overlay_url = f"/api/v1/analysis/{analysis_id}/heatmap?variant=overlay"
        return result
    except Exception as exc:
        log.warning("XAI generation failed (%s): %s", type(exc).__name__, exc)
        return XAIResult(status="failed", runtime_ms=int((time.perf_counter()-started)*1000), model_name=model_name or getattr(getattr(model, "config", None), "_name_or_path", type(model).__name__), warnings=[f"Attribution unavailable ({type(exc).__name__})."])
    finally:
        if grad_cam is not None:
            grad_cam.cleanup()
        if hasattr(model, "zero_grad"):
            model.zero_grad(set_to_none=True)
        if hasattr(model, "eval"):
            model.eval()
