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
import threading
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
PRIMARY_MODEL = settings.AUTHENTICITY_MODEL
SECONDARY_MODEL = "buildborderless/CommunityForensics-DeepfakeDet-ViT"


@dataclass
class AuthenticityResult:
    """Result from authenticity detection."""
    label: str  # likely_ai_generated | likely_manipulated | likely_authentic | inconclusive
    ai_generated_score: float = 0.0
    manipulation_score: Optional[float] = None
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
            "manipulation_score": round(self.manipulation_score, 4) if self.manipulation_score is not None else None,
            "authentic_score": round(self.authentic_score, 4),
            "confidence": round(self.confidence, 4),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "runtime_ms": self.runtime_ms,
            "warnings": self.warnings,
            "raw_scores": {k: round(v, 4) for k, v in self.raw_scores.items()},
            "score_semantics": "Uncalibrated model softmax scores; not validated probabilities or accuracy estimates.",
            "manipulation_score_status": "UNAVAILABLE: this detector does not independently classify editing/manipulation.",
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
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, device: str = "auto") -> None:
        """Load model and processor from HuggingFace Hub / local cache."""
        with self._lock:
            if self._loaded:
                return
            self._load_model(device)

    def _load_model(self, device: str) -> None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self._device = resolve_device(device)

        log.info("Loading authenticity model: %s on %s", self._model_id, self._device)
        t0 = time.time()

        cache_dir = settings.MODEL_CACHE_DIR

        processor = AutoImageProcessor.from_pretrained(
            self._model_id,
            cache_dir=cache_dir,
            token=settings.HF_TOKEN or None,
        )
        model = AutoModelForImageClassification.from_pretrained(
            self._model_id,
            cache_dir=cache_dir,
            token=settings.HF_TOKEN or None,
        )
        model.to(self._device)
        model.eval()
        self._processor = processor
        self._model = model

        elapsed = int((time.time() - t0) * 1000)
        log.info("Authenticity model loaded in %d ms", elapsed)
        self._loaded = True

    def unload(self) -> None:
        """Unload model to free memory."""
        with self._lock:
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
        with self._lock:
            return self._detect(image)

    def _detect(self, image: Image.Image) -> AuthenticityResult:
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
        probs = torch.softmax(logits, dim=0).float().cpu().numpy()
        if probs.ndim != 1 or len(probs) != len(self._model.config.id2label):
            raise ValueError("Detector returned an unexpected number of class scores.")
        if not np.isfinite(probs).all():
            raise ValueError("Detector returned non-finite class scores.")

        # Map model labels to our schema
        id2label = self._model.config.id2label
        label_map = {}
        for idx, label_name in id2label.items():
            label_map[str(label_name).lower()] = float(probs[int(idx)])

        # Class meaning comes from the loaded checkpoint configuration, not index order.
        fake_score = None
        real_score = None
        for label_name, score in label_map.items():
            if "fake" in label_name or "deepfake" in label_name or "ai" in label_name:
                fake_score = max(fake_score or 0.0, score)
            elif "real" in label_name or "authentic" in label_name:
                real_score = max(real_score or 0.0, score)

        if fake_score is None or real_score is None:
            raise ValueError("Detector class labels do not identify both fake and real classes.")

        # Determine label
        confidence = max(fake_score, real_score)
        if settings.AI_GENERATED_THRESHOLD <= settings.AUTHENTIC_THRESHOLD:
            raise ValueError("Detector decision thresholds are misconfigured.")
        if fake_score >= settings.AI_GENERATED_THRESHOLD - 1e-6:
            label = "likely_ai_generated"
        elif fake_score <= settings.AUTHENTIC_THRESHOLD + 1e-6:
            label = "likely_authentic"
        else:
            label = "inconclusive"
            warnings.append("Fake-class score falls in the configured uncertainty band.")

        runtime_ms = int((time.time() - t0) * 1000)

        result = AuthenticityResult(
            label=label,
            ai_generated_score=fake_score,
            manipulation_score=None,
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
