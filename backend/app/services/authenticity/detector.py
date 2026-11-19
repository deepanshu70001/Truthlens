"""
TruthLens — Authenticity detection service.

Uses SigLIP2-based image classifier (Deepfake-Real-Class-Siglip2)
as primary detector.

IMPORTANT: This is a probabilistic AI-assisted assessment.
The output is a signal, NOT proof of authenticity or fakery.
"""

from __future__ import annotations

import io
import re
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
from backend.app.services.authenticity.calibration import (
    CalibrationArtifactError,
    calibrated_ai_probability,
    load_calibration_artifact,
)

log = get_logger("authenticity")

# ── Model IDs ────────────────────────────────────────────────────
PRIMARY_MODEL = settings.AUTHENTICITY_MODEL
SECONDARY_MODEL = "buildborderless/CommunityForensics-DeepfakeDet-ViT"


def _resolve_checkpoint_revision(model_id: str, cache_dir: str | Path, config: Any, requested_revision: str = "") -> str:
    """Resolve the exact cached Hub commit when Transformers omits _commit_hash."""
    candidates = [getattr(config, "_commit_hash", None), requested_revision]
    for candidate in candidates:
        if isinstance(candidate, str) and re.fullmatch(r"[0-9a-f]{40}", candidate):
            return candidate

    model_cache = Path(cache_dir) / f"models--{model_id.replace('/', '--')}"
    refs = model_cache / "refs"
    ref_file = refs / (requested_revision or "main")
    try:
        revision = ref_file.read_text(encoding="utf-8").strip()
    except OSError:
        revision = ""
    if re.fullmatch(r"[0-9a-f]{40}", revision) and (model_cache / "snapshots" / revision).is_dir():
        return revision
    if requested_revision:
        return "unknown"

    snapshots = model_cache / "snapshots"
    if snapshots.is_dir():
        revisions = [item.name for item in snapshots.iterdir() if item.is_dir() and re.fullmatch(r"[0-9a-f]{40}", item.name)]
        if len(revisions) == 1:
            return revisions[0]
    return "unknown"


@dataclass
class AuthenticityResult:
    """Result from authenticity detection."""
    label: str  # likely_ai_generated | likely_manipulated | likely_authentic | inconclusive
    model_prediction: str = "inconclusive"
    ai_generated_score: float = 0.0
    manipulation_score: Optional[float] = None
    authentic_score: float = 0.0
    confidence: Optional[float] = None
    raw_max_softmax_score: Optional[float] = None
    model_name: str = ""
    model_version: str = ""
    runtime_ms: int = 0
    warnings: List[str] = field(default_factory=list)
    raw_scores: Dict[str, float] = field(default_factory=dict)
    raw_logits: Dict[str, float] = field(default_factory=dict)
    calibrated_ai_generated_probability: Optional[float] = None
    calibrated_candidate_label: str = "inconclusive"
    calibration_status: str = "not_calibrated"
    calibration_method: Optional[str] = None
    domain_validation_status: str = "not_evaluated"
    empirical_accuracy: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "model_prediction": self.model_prediction,
            "ai_generated_score": round(self.ai_generated_score, 4),
            "raw_logits": {k: round(v, 6) for k, v in self.raw_logits.items()},
            "calibrated_ai_generated_probability": round(self.calibrated_ai_generated_probability, 4) if self.calibrated_ai_generated_probability is not None else None,
            "calibrated_candidate_label": self.calibrated_candidate_label,
            "calibration_status": self.calibration_status,
            "calibration_method": self.calibration_method,
            "domain_validation_status": self.domain_validation_status,
            "empirical_accuracy": self.empirical_accuracy,
            "manipulation_score": round(self.manipulation_score, 4) if self.manipulation_score is not None else None,
            "authentic_score": round(self.authentic_score, 4),
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "raw_max_softmax_score": round(self.raw_max_softmax_score, 4) if self.raw_max_softmax_score is not None else None,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "runtime_ms": self.runtime_ms,
            "warnings": self.warnings,
            "raw_scores": {k: round(v, 4) for k, v in self.raw_scores.items()},
            "score_semantics": "ai_generated_score and authentic_score are raw softmax scores, not calibrated probabilities or estimates of real-world accuracy.",
            "probability_semantics": "calibrated_ai_generated_probability is populated only when an independently reviewed, validation-split calibration artifact is configured.",
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
        self._model_revision: Optional[str] = None
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
            revision=settings.AUTHENTICITY_MODEL_REVISION or None,
        )
        model = AutoModelForImageClassification.from_pretrained(
            self._model_id,
            cache_dir=cache_dir,
            token=settings.HF_TOKEN or None,
            revision=settings.AUTHENTICITY_MODEL_REVISION or None,
        )
        model.to(self._device)
        model.eval()
        self._processor = processor
        self._model = model
        self._model_revision = _resolve_checkpoint_revision(
            self._model_id, cache_dir, model.config, settings.AUTHENTICITY_MODEL_REVISION
        )

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
        # The checkpoint's processor uses RGB input and its stored resize and
        # normalization parameters. Convert explicitly to avoid mode-dependent
        # behavior for grayscale, palette, or CMYK uploads.
        inputs = self._processor(images=image.convert("RGB"), return_tensors="pt")
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
        fake_logit = None
        real_logit = None
        for label_name, score in label_map.items():
            if "fake" in label_name or "deepfake" in label_name or "ai" in label_name:
                fake_score = max(fake_score or 0.0, score)
                class_index = next(int(idx) for idx, name in id2label.items() if str(name).lower() == label_name)
                fake_logit = float(logits[class_index].item())
            elif "real" in label_name or "authentic" in label_name:
                real_score = max(real_score or 0.0, score)
                class_index = next(int(idx) for idx, name in id2label.items() if str(name).lower() == label_name)
                real_logit = float(logits[class_index].item())

        if fake_score is None or real_score is None or fake_logit is None or real_logit is None:
            raise ValueError("Detector class labels do not identify both fake and real classes.")

        # Never turn raw softmax confidence into a real-world confidence claim.
        # Without reviewed calibration evidence, expose the raw argmax as a
        # model prediction and abstain from the application-level label.
        raw_max_softmax_score = max(fake_score, real_score)
        model_prediction = (
            "likely_ai_generated" if fake_logit > real_logit
            else "likely_authentic" if real_logit > fake_logit
            else "inconclusive"
        )
        label = "inconclusive"
        calibrated_probability = None
        calibrated_candidate_label = "inconclusive"
        calibration_status = "not_calibrated"
        calibration_method = None
        domain_validation_status = "not_evaluated"

        if settings.AUTHENTICITY_CALIBRATION_PATH:
            try:
                artifact = load_calibration_artifact(settings.AUTHENTICITY_CALIBRATION_PATH, self._model_id, self._model_revision)
                calibrated_probability = calibrated_ai_probability(fake_logit, real_logit, artifact["temperature"])
                calibration_status = "validated_temperature_scaling"
                calibration_method = "temperature_scaling"
                if calibrated_probability >= artifact["thresholds"]["ai_generated_min"]:
                    calibrated_candidate_label = "likely_ai_generated"
                elif calibrated_probability <= artifact["thresholds"]["authentic_max"]:
                    calibrated_candidate_label = "likely_authentic"
                else:
                    warnings.append("Calibrated score falls in the validation-selected uncertainty band.")
                warnings.append("Input-domain membership is not assessed; the application classification remains inconclusive.")
            except CalibrationArtifactError as exc:
                warnings.append(f"Configured calibration was rejected: {exc}")
                calibration_status = "invalid_artifact"
        if calibration_status == "not_calibrated":
            warnings.append("No independently reviewed calibration artifact is configured; overall detector classification is inconclusive.")

        runtime_ms = int((time.time() - t0) * 1000)

        result = AuthenticityResult(
            label=label,
            model_prediction=model_prediction,
            ai_generated_score=fake_score,
            manipulation_score=None,
            authentic_score=real_score,
            confidence=(
                max(calibrated_probability, 1.0 - calibrated_probability)
                if calibrated_probability is not None and domain_validation_status == "validated_for_input_domain"
                else None
            ),
            raw_max_softmax_score=raw_max_softmax_score,
            model_name=self._model_id,
            model_version=self._model_revision or self._model.config._name_or_path,
            runtime_ms=runtime_ms,
            warnings=warnings,
            raw_scores=label_map,
            raw_logits={"fake": fake_logit, "real": real_logit},
            calibrated_ai_generated_probability=calibrated_probability,
            calibrated_candidate_label=calibrated_candidate_label,
            calibration_status=calibration_status,
            calibration_method=calibration_method,
            domain_validation_status=domain_validation_status,
            empirical_accuracy=None,
        )

        log.info(
            "Authenticity: label=%s raw_max_softmax=%.3f fake_raw=%.3f real_raw=%.3f [%d ms]",
            label, raw_max_softmax_score, fake_score, real_score, runtime_ms,
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
