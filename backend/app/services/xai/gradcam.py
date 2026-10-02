"""
TruthLens — XAI service (Explainable AI).

Generates Grad-CAM heatmaps from the authenticity detection model.
The heatmap shows which image regions most influenced the model's decision.

The heatmap is derived from ACTUAL model attribution, never fabricated.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from backend.app.core.logging import get_logger
from backend.app.config import settings

log = get_logger("xai")


@dataclass
class XAIResult:
    """XAI analysis result with heatmap paths."""
    heatmap_path: Optional[str] = None
    overlay_path: Optional[str] = None
    suspicious_regions: List[Dict[str, Any]] = field(default_factory=list)
    method: str = "grad_cam"
    target_class: str = ""
    target_score: float = 0.0
    runtime_ms: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "heatmap_path": self.heatmap_path,
            "overlay_path": self.overlay_path,
            "suspicious_regions": self.suspicious_regions,
            "method": self.method,
            "target_class": self.target_class,
            "target_score": round(self.target_score, 4),
            "runtime_ms": self.runtime_ms,
            "warnings": self.warnings,
            "note": (
                "This heatmap shows regions that influenced the AI model's decision. "
                "It does NOT directly indicate manipulated areas."
            ),
        }


class GradCAM:
    """
    Grad-CAM implementation compatible with ViT and SigLIP models.

    Uses hooks on the last attention/conv layer to compute gradient-weighted
    class activation maps.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.gradients = None
        self.activations = None
        self._hooks = []

        # Find target layer — for ViT/SigLIP, use the last encoder layer's layer_norm
        if target_layer is None:
            target_layer = self._find_target_layer()

        if target_layer is not None:
            self._hooks.append(
                target_layer.register_forward_hook(self._save_activation)
            )
            self._hooks.append(
                target_layer.register_full_backward_hook(self._save_gradient)
            )
            log.info("Grad-CAM target layer: %s", type(target_layer).__name__)
        else:
            log.warning("Could not find a target layer for Grad-CAM")

    def _find_target_layer(self):
        """Try to find the right target layer for attribution."""
        # For SigLIP/ViT models
        for attr_path in [
            "vision_model.encoder.layers",  # SigLIP structure
            "vit.encoder.layer",            # Standard ViT
            "encoder.layers",               # Generic
        ]:
            obj = self.model
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                # Get the last layer
                if hasattr(obj, "__getitem__"):
                    last_layer = obj[-1]
                    # Try to get layer_norm or output within it
                    for sub in ["layer_norm2", "layernorm_after", "norm2", "output"]:
                        if hasattr(last_layer, sub):
                            return getattr(last_layer, sub)
                    return last_layer
            except (AttributeError, IndexError):
                continue

        # Fallback: last module before the classifier
        modules = list(self.model.modules())
        for m in reversed(modules):
            if isinstance(m, torch.nn.LayerNorm):
                return m
        return None

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Generate a Grad-CAM heatmap.

        Args:
            input_tensor: preprocessed input tensor [1, C, H, W]
            target_class: class index to explain (None = predicted class)

        Returns:
            Heatmap as numpy array (H, W) with values 0-1, or None on failure.
        """
        self.model.eval()

        try:
            # Forward pass
            output = self.model(input_tensor)
            logits = output.logits if hasattr(output, "logits") else output

            if target_class is None:
                target_class = logits.argmax(dim=1).item()

            target_score = logits[0, target_class]

            # Backward pass
            self.model.zero_grad()
            target_score.backward(retain_graph=True)

            if self.gradients is None or self.activations is None:
                log.warning("No gradients/activations captured for Grad-CAM")
                return None

            # Compute Grad-CAM
            gradients = self.gradients
            activations = self.activations

            # Handle ViT token-based activations [1, seq_len, hidden_dim]
            if len(activations.shape) == 3:
                # Remove CLS token if present, reshape to spatial
                seq_len = activations.shape[1]
                hidden_dim = activations.shape[2]

                # For ViT: seq_len = 1 (CLS) + num_patches
                # Assume square patches
                num_patches = seq_len - 1 if seq_len > 1 else seq_len
                side = int(np.sqrt(num_patches))

                if side * side == num_patches and num_patches > 1:
                    # Remove CLS token
                    act = activations[0, 1:, :]  # [num_patches, hidden_dim]
                    grad = gradients[0, 1:, :]

                    # Global average pooling of gradients
                    weights = grad.mean(dim=0)  # [hidden_dim]

                    # Weighted sum
                    cam = (act * weights).sum(dim=1)  # [num_patches]
                    cam = cam.reshape(side, side)
                else:
                    # Fallback: just use activations directly
                    weights = gradients.mean(dim=1, keepdim=True)
                    cam = (activations * weights).sum(dim=2)[0]
                    side = int(np.sqrt(cam.shape[0]))
                    if side * side == cam.shape[0]:
                        cam = cam.reshape(side, side)
                    else:
                        cam = cam.reshape(1, -1)

            elif len(activations.shape) == 4:
                # Standard CNN: [1, C, H, W]
                weights = gradients.mean(dim=(2, 3), keepdim=True)
                cam = (activations * weights).sum(dim=1)[0]
            else:
                log.warning("Unexpected activation shape: %s", activations.shape)
                return None

            # ReLU and normalize
            cam = torch.relu(cam)
            cam = cam.cpu().numpy()
            if cam.max() > 0:
                cam = cam / cam.max()

            return cam.astype(np.float32)

        except Exception as e:
            log.error("Grad-CAM generation failed: %s", e)
            return None

    def cleanup(self):
        """Remove hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def generate_heatmap(
    model,
    processor,
    image: Image.Image,
    analysis_id: str,
    device: torch.device,
) -> XAIResult:
    """
    Generate Grad-CAM heatmap from the authenticity model.

    Saves:
    - Raw heatmap image
    - Overlay on original image

    Stores under artifacts/heatmaps/
    """
    t0 = time.time()
    warnings: List[str] = []

    settings.ensure_storage_dirs()
    heatmap_dir = settings.heatmaps_dir

    try:
        # Initialize Grad-CAM
        grad_cam = GradCAM(model)

        # Preprocess image
        inputs = processor(images=image, return_tensors="pt")
        input_tensor = inputs["pixel_values"].to(device)

        # Need gradients
        input_tensor.requires_grad_(True)

        # Generate heatmap
        cam = grad_cam.generate(input_tensor)
        grad_cam.cleanup()

        if cam is None:
            return XAIResult(
                runtime_ms=int((time.time() - t0) * 1000),
                warnings=["Grad-CAM could not generate a heatmap for this model architecture."],
            )

        # Resize heatmap to original image size
        img_array = np.array(image.convert("RGB"))
        h, w = img_array.shape[:2]
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

        # Create colormap heatmap
        heatmap_colored = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Save raw heatmap
        heatmap_path = heatmap_dir / f"{analysis_id}_heatmap.png"
        Image.fromarray(heatmap_colored).save(str(heatmap_path))

        # Create overlay
        overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
        overlay_path = heatmap_dir / f"{analysis_id}_overlay.png"
        Image.fromarray(overlay).save(str(overlay_path))

        # Find suspicious regions (areas with high activation)
        suspicious_regions = []
        threshold = 0.7
        mask = (cam_resized > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if area > (w * h * 0.005):  # Only regions > 0.5% of image
                suspicious_regions.append({
                    "bbox": [x, y, x + cw, y + ch],
                    "area_fraction": round(area / (w * h), 4),
                    "max_activation": round(float(cam_resized[y:y+ch, x:x+cw].max()), 4),
                })

        # Get target class info
        with torch.no_grad():
            outputs = model(**{k: v.to(device) for k, v in inputs.items()})
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=0)
            pred_class = probs.argmax().item()
            pred_label = model.config.id2label.get(pred_class, str(pred_class))
            pred_score = float(probs[pred_class])

        runtime_ms = int((time.time() - t0) * 1000)

        result = XAIResult(
            heatmap_path=str(heatmap_path),
            overlay_path=str(overlay_path),
            suspicious_regions=suspicious_regions,
            method="grad_cam",
            target_class=pred_label,
            target_score=pred_score,
            runtime_ms=runtime_ms,
            warnings=warnings,
        )

        log.info(
            "XAI: heatmap generated for class=%s score=%.3f regions=%d [%d ms]",
            pred_label, pred_score, len(suspicious_regions), runtime_ms,
        )

        return result

    except Exception as e:
        log.error("XAI generation failed: %s", e)
        return XAIResult(
            runtime_ms=int((time.time() - t0) * 1000),
            warnings=[f"Heatmap generation failed: {e}"],
        )
