"""
TruthLens — Authenticity detection service.

Uses SigLIP2-based image classifier (Deepfake-Real-Class-Siglip2)
as primary detector.

IMPORTANT: This is a probabilistic AI-assisted assessment.
The output is a signal, NOT proof of authenticity or fakery.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from backend.app.config import settings
from backend.app.core.logging import get_logger
from backend.app.utils.device import resolve_device, clear_gpu_memory

log = get_logger("authenticity")

# ── Model IDs ────────────────────────────────────────────────────
PRIMARY_MODEL = "prithivMLmods/Deepfake-Real-Class-Siglip2"
SECONDARY_MODEL = "buildborderless/CommunityForensics-DeepfakeDet-ViT"


@dataclass
class AuthenticityResult:
    """Result from authenticity detection."""
    label: str  # likely_ai_generated | likely_manipulated | likely_authentic | inconclusive
    ai_generated_score: float = 0.0
    manipulation_score: float = 0.0
    authentic_score: float = 0.0
    confidence: float = 0.0
    model_name: str = ""
    model_version: str = ""
    runtime_ms: int = 0
    warnings: List[str] = field(default_factory=list)
    raw_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "ai_generated_score": round(self.ai_generated_score, 4),
            "manipulation_score": round(self.manipulation_score, 4),
            "authentic_score": round(self.authentic_score, 4),
            "confidence": round(self.confidence, 4),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "runtime_ms": self.runtime_ms,
            "warnings": self.warnings,
            "raw_scores": {k: round(v, 4) for k, v in self.raw_scores.items()},
            "caveat": (
                "This is an AI-assisted probabilistic assessment. "
                "Model predictions can be wrong and should not be treated as proof."
            ),
        }


class AuthenticityDetector:
    """
    Authenticity detection using SigLIP2-based binary classifier.

    Selected model: prithivMLmods/Deepfake-Real-Class-Siglip2
    - Architecture: SiglipForImageClassification (SigLIP2-base)
    - Input: 224x224 RGB
    - Output: 2-class logits (Fake / Real)
    - Parameters: ~93M (F32)
    - VRAM: ~400 MB
    - License: Apache 2.0
    - Why selected:
      * Apache 2.0 license (commercial-friendly)
      * Standard HF Transformers integration (AutoModelForImageClassification)
      * Lightweight enough for RTX 3050 4GB
      * Modern SigLIP2 vision encoder (strong feature extraction)
    - Limitations:
      * Binary only (no manipulation-type breakdown)
      * No published evaluation benchmarks
      * Concept drift risk with newer generation techniques
      * TruthLens has NOT independently verified accuracy
    """

    def __init__(self):
        self._model = None
        self._processor = None
        self._device: Optional[torch.device] = None
        self._loaded = False
        self._model_id = PRIMARY_MODEL

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, device: str = "auto") -> None:
        """Load model and processor from HuggingFace Hub / local cache."""
        if self._loaded:
            return

        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self._device = resolve_device(device)

        log.info("Loading authenticity model: %s on %s", self._model_id, self._device)
        t0 = time.time()

        cache_dir = settings.MODEL_CACHE_DIR

        self._processor = AutoImageProcessor.from_pretrained(
            self._model_id,
            cache_dir=cache_dir,
            token=settings.HF_TOKEN or None,
        )
        self._model = AutoModelForImageClassification.from_pretrained(
            self._model_id,
            cache_dir=cache_dir,
            token=settings.HF_TOKEN or None,
        )
        self._model.to(self._device)
        self._model.eval()

        elapsed = int((time.time() - t0) * 1000)
        log.info("Authenticity model loaded in %d ms", elapsed)
        self._loaded = True

    def unload(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._loaded = False
        clear_gpu_memory()
        log.info("Authenticity model unloaded")

    @torch.no_grad()
    def detect(self, image: Image.Image) -> AuthenticityResult:
        """
        Run authenticity detection on a PIL Image.

        Returns AuthenticityResult with probabilistic scores.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        t0 = time.time()
        warnings: List[str] = []

        # Preprocess
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Forward pass
        outputs = self._model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=0).cpu().numpy()

        # Map model labels to our schema
        id2label = self._model.config.id2label
        label_map = {}
        for idx, label_name in id2label.items():
            label_map[label_name.lower()] = float(probs[int(idx)])

        # The SigLIP2 model outputs: {0: "Fake", 1: "Real"} (or similar)
        fake_score = 0.0
        real_score = 0.0
        for label_name, score in label_map.items():
            if "fake" in label_name or "deepfake" in label_name or "ai" in label_name:
                fake_score = max(fake_score, score)
            elif "real" in label_name or "authentic" in label_name:
                real_score = max(real_score, score)

        # If neither mapped, use raw position
        if fake_score == 0.0 and real_score == 0.0:
            fake_score = float(probs[0])
            real_score = float(probs[1]) if len(probs) > 1 else 1.0 - fake_score
            warnings.append(
                f"Model labels {id2label} did not map to expected Fake/Real. "
                "Using positional fallback."
            )

        # Determine label
        confidence = max(fake_score, real_score)
        if confidence < 0.55:
            label = "inconclusive"
            warnings.append("Low confidence — model is not sure about this image.")
        elif fake_score > real_score:
            label = "likely_ai_generated"
        else:
            label = "likely_authentic"

        runtime_ms = int((time.time() - t0) * 1000)

        result = AuthenticityResult(
            label=label,
            ai_generated_score=fake_score,
            manipulation_score=fake_score * 0.5,  # Proxy — no separate manipulation model
            authentic_score=real_score,
            confidence=confidence,
            model_name=self._model_id,
            model_version=self._model.config._name_or_path,
            runtime_ms=runtime_ms,
            warnings=warnings,
            raw_scores=label_map,
        )

        log.info(
            "Authenticity: label=%s confidence=%.3f fake=%.3f real=%.3f [%d ms]",
            label, confidence, fake_score, real_score, runtime_ms,
        )

        return result

    def get_internal_model(self):
        """Return the internal PyTorch model (for XAI attribution)."""
        return self._model

    def get_processor(self):
        """Return the image processor (for XAI attribution)."""
        return self._processor

    def get_device(self):
        """Return the device the model is on."""
        return self._device


# Module-level singleton (lazy-loaded)
_detector: Optional[AuthenticityDetector] = None


def get_detector() -> AuthenticityDetector:
    """Get or create the singleton authenticity detector."""
    global _detector
    if _detector is None:
        _detector = AuthenticityDetector()
    return _detector
