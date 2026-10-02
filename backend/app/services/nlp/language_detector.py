"""
TruthLens — Language detection service.

Uses XLM-RoBERTa for language detection on OCR-extracted text.
Supplements OCR's script-based detection with ML-based classification.
Does NOT overwrite raw OCR output — adds a parallel signal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from backend.app.core.logging import get_logger
from backend.app.config import settings
from backend.app.utils.device import resolve_device, clear_gpu_memory

log = get_logger("langdetect")

MODEL_ID = "papluca/xlm-roberta-base-language-detection"


@dataclass
class LanguageDetectionResult:
    """Language detection result."""
    detected_language: str = "unknown"
    confidence: float = 0.0
    top_languages: List[Dict[str, float]] = field(default_factory=list)
    model_name: str = MODEL_ID
    runtime_ms: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected_language": self.detected_language,
            "confidence": round(self.confidence, 4),
            "top_languages": self.top_languages,
            "model_name": self.model_name,
            "runtime_ms": self.runtime_ms,
            "warnings": self.warnings,
        }


class LanguageDetector:
    """
    XLM-RoBERTa based language detection.

    Used only when OCR text is available. Does not overwrite
    OCR's own language evidence.
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device: Optional[torch.device] = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, device: str = "auto") -> None:
        """Load language detection model."""
        if self._loaded:
            return

        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self._device = resolve_device(device)

        log.info("Loading language detection model: %s", MODEL_ID)
        t0 = time.time()

        cache_dir = settings.MODEL_CACHE_DIR

        self._tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, cache_dir=cache_dir,
            token=settings.HF_TOKEN or None,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, cache_dir=cache_dir,
            token=settings.HF_TOKEN or None,
        )
        self._model.to(self._device)
        self._model.eval()

        elapsed = int((time.time() - t0) * 1000)
        log.info("Language detection model loaded in %d ms", elapsed)
        self._loaded = True

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False
        clear_gpu_memory()
        log.info("Language detection model unloaded")

    @torch.no_grad()
    def detect(self, text: str, top_k: int = 3) -> LanguageDetectionResult:
        """
        Detect language of the given text.

        Returns top-k language predictions with confidence scores.
        """
        if not self._loaded:
            return LanguageDetectionResult(
                warnings=["Language detection model not loaded."],
            )

        if not text or len(text.strip()) < 3:
            return LanguageDetectionResult(
                detected_language="unknown",
                warnings=["Text too short for reliable language detection."],
            )

        t0 = time.time()

        # Truncate long text
        text_input = text[:512]

        inputs = self._tokenizer(
            text_input, return_tensors="pt",
            truncation=True, max_length=128, padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        probs = torch.softmax(outputs.logits[0], dim=0)

        # Get top-k predictions
        topk_probs, topk_indices = torch.topk(probs, min(top_k, len(probs)))

        id2label = self._model.config.id2label
        top_langs = []
        for prob, idx in zip(topk_probs, topk_indices):
            lang = id2label.get(idx.item(), f"lang_{idx.item()}")
            top_langs.append({
                "language": lang,
                "confidence": round(float(prob), 4),
            })

        best_lang = top_langs[0]["language"] if top_langs else "unknown"
        best_conf = top_langs[0]["confidence"] if top_langs else 0.0

        runtime_ms = int((time.time() - t0) * 1000)

        result = LanguageDetectionResult(
            detected_language=best_lang,
            confidence=best_conf,
            top_languages=top_langs,
            runtime_ms=runtime_ms,
        )

        log.info("Language detected: %s (%.3f) [%d ms]", best_lang, best_conf, runtime_ms)

        return result


# Module-level singleton
_detector: Optional[LanguageDetector] = None


def get_language_detector() -> LanguageDetector:
    global _detector
    if _detector is None:
        _detector = LanguageDetector()
    return _detector
