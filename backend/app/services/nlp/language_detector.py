"""Local-only document-level language identification for OCR text."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import torch

from backend.app.config import settings
from backend.app.core.logging import get_logger
from backend.app.utils.device import clear_gpu_memory, resolve_device

log = get_logger("langdetect")

MODEL_ID = "papluca/xlm-roberta-base-language-detection"
MAX_INPUT_CHARS = 4096
MAX_TOKEN_LENGTH = 256
MIN_CONTENT_CHARACTERS = 20
MIN_ALPHABETIC_RATIO = 0.35
MIN_CONFIDENCE = 0.70

# These are the model card's declared labels, not script-to-language guesses.
LANGUAGE_NAMES = {
    "ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fr": "French", "hi": "Hindi",
    "it": "Italian", "ja": "Japanese", "nl": "Dutch", "pl": "Polish",
    "pt": "Portuguese", "ru": "Russian", "sw": "Swahili", "th": "Thai",
    "tr": "Turkish", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese",
}


@dataclass
class LanguageDetectionResult:
    """Prediction payload; ``confidence`` is the model's max softmax score."""

    status: str = "completed"
    outcome: str = "undetermined"
    detected_language: str = "unknown"
    language_name: Optional[str] = None
    confidence: Optional[float] = None
    top_languages: List[Dict[str, Any]] = field(default_factory=list)
    model_name: str = MODEL_ID
    model_version: Optional[str] = None
    model_revision: Optional[str] = None
    runtime_ms: int = 0
    input_characters: int = 0
    input_truncated: bool = False
    confidence_semantics: str = "Maximum softmax probability from the model; not calibrated."
    warnings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        # Keep the previous field names for consumers of older service payloads.
        return {
            "status": self.status,
            "outcome": self.outcome,
            "detected_language": self.detected_language,
            "language_code": self.detected_language if self.detected_language != "unknown" else None,
            "language_name": self.language_name,
            "confidence": round(self.confidence, 6) if self.confidence is not None else None,
            "top_languages": self.top_languages,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_revision": self.model_revision,
            "runtime_ms": self.runtime_ms,
            "input_characters": self.input_characters,
            "input_truncated": self.input_truncated,
            "confidence_semantics": self.confidence_semantics,
            "warnings": self.warnings,
            "limitations": self.limitations,
        }


class LanguageDetector:
    """Thread-safe, cache-only wrapper around the fine-tuned PAPLUCA classifier."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = None
        self._loaded = False
        self._load_attempted = False
        self._load_failure: Optional[str] = None
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_revision(self) -> Optional[str]:
        config = getattr(self._model, "config", None)
        return getattr(config, "_commit_hash", None) if config is not None else None

    def load(self, device: str = "auto") -> bool:
        """Load only from the configured local Hugging Face cache; never download."""
        with self._lock:
            if self._loaded:
                return True
            if self._load_attempted:
                return False
            self._load_attempted = True
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self._device = resolve_device(device)
                common = {
                    "cache_dir": settings.MODEL_CACHE_DIR,
                    "local_files_only": True,
                }
                self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **common)
                self._model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, **common)
                self._model.to(self._device)
                self._model.eval()
                self._validate_label_mapping(self._model.config.id2label)
                self._loaded = True
                log.info("Language identification model loaded from configured local cache")
                return True
            except Exception as exc:
                self._model = None
                self._tokenizer = None
                self._load_failure = type(exc).__name__
                # Do not log exception text: loader errors can contain local paths.
                log.warning("Language identification model unavailable (%s)", self._load_failure)
                return False

    @staticmethod
    def _validate_label_mapping(id2label: Mapping[Any, Any]) -> None:
        if not isinstance(id2label, Mapping) or not id2label:
            raise ValueError("The model has no usable label mapping")
        for index, label in id2label.items():
            try:
                int(index)
            except (TypeError, ValueError) as exc:
                raise ValueError("The model label mapping has a malformed class index") from exc
            if not isinstance(label, str) or not label.strip():
                raise ValueError("The model label mapping has a malformed label")

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._device = None
            self._loaded = False
            self._load_attempted = False
            self._load_failure = None
            clear_gpu_memory()

    @staticmethod
    def _base_result(status: str, outcome: str, **kwargs: Any) -> LanguageDetectionResult:
        return LanguageDetectionResult(status=status, outcome=outcome, **kwargs)

    @staticmethod
    def _id_to_label(id2label: Mapping[Any, Any], index: int) -> Optional[str]:
        label = id2label.get(index, id2label.get(str(index)))
        if not isinstance(label, str):
            return None
        return label.strip().lower()

    def detect(self, text: str, top_k: int = 3, device: str = "auto") -> LanguageDetectionResult:
        """Predict one language for a whole OCR document without changing OCR text."""
        started = time.perf_counter()
        if not isinstance(text, str):
            return self._base_result(
                "completed", "insufficient_input",
                runtime_ms=max(0, int((time.perf_counter() - started) * 1000)),
                warnings=["Language identification requires text input."],
                limitations=["No prediction was attempted for malformed input."],
            )
        if not text.strip():
            return self._base_result(
                "completed", "no_text", input_characters=len(text),
                runtime_ms=max(0, int((time.perf_counter() - started) * 1000)),
                warnings=["No OCR text was available for language identification."],
            )

        model_input = text[:MAX_INPUT_CHARS]
        non_space = [char for char in model_input if not char.isspace()]
        content_chars = sum(char.isalpha() for char in non_space)
        ratio = content_chars / len(non_space) if non_space else 0.0
        if content_chars < MIN_CONTENT_CHARACTERS or ratio < MIN_ALPHABETIC_RATIO:
            return self._base_result(
                "completed", "insufficient_input",
                runtime_ms=max(0, int((time.perf_counter() - started) * 1000)),
                input_characters=len(text), input_truncated=len(text) > MAX_INPUT_CHARS,
                warnings=["OCR text is too short or noisy for reliable language identification."],
                limitations=["No script-based fallback is used as a language prediction."],
            )

        if not self.load(device=device):
            return self._base_result(
                "unavailable", "unavailable", input_characters=len(text),
                input_truncated=len(text) > MAX_INPUT_CHARS,
                runtime_ms=max(0, int((time.perf_counter() - started) * 1000)),
                warnings=["The local language identification model or its dependencies are unavailable."],
                limitations=["Model loading is cache-only; no model weights were downloaded."],
            )

        with self._lock:
            try:
                encoded = self._tokenizer(
                    model_input,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_TOKEN_LENGTH,
                    padding=False,
                )
                token_count = int(encoded["input_ids"].shape[-1])
                was_truncated = len(text) > MAX_INPUT_CHARS or token_count >= MAX_TOKEN_LENGTH
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                with torch.inference_mode():
                    output = self._model(**encoded)
                logits = output.logits
                if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[0] != 1:
                    raise ValueError("The model returned malformed classification logits")
                labels = self._model.config.id2label
                self._validate_label_mapping(labels)
                if logits.shape[1] != len(labels) or logits.shape[1] == 0:
                    raise ValueError("The model output does not match its label mapping")
                probabilities = torch.softmax(logits[0].float(), dim=0)
                if not bool(torch.isfinite(probabilities).all()):
                    raise ValueError("The model returned non-finite probabilities")
                count = min(max(int(top_k), 1), len(probabilities))
                scores, indices = torch.topk(probabilities, count)
                top_languages = []
                unknown_label = False
                for score, class_index in zip(scores.tolist(), indices.tolist()):
                    label = self._id_to_label(labels, int(class_index))
                    if label not in LANGUAGE_NAMES:
                        unknown_label = True
                        continue
                    top_languages.append({
                        "language": label,
                        "language_name": LANGUAGE_NAMES[label],
                        "confidence": round(float(score), 6),
                    })

                elapsed_ms = max(0, int((time.perf_counter() - started) * 1000))
                warnings = []
                limitations = [
                    "One document-level prediction is returned; mixed-language regions are not separately identified.",
                    "Softmax confidence is not calibrated and does not measure truthfulness or authenticity.",
                ]
                if was_truncated:
                    warnings.append("Input was truncated to the documented character/token limit for inference.")
                if unknown_label or not top_languages:
                    warnings.append("The model returned an unsupported or unknown language label.")
                    return self._base_result(
                        "completed", "undetermined", input_characters=len(text),
                        input_truncated=was_truncated, top_languages=top_languages,
                        model_version=getattr(self._model.config, "_commit_hash", None),
                        model_revision=self.model_revision, runtime_ms=elapsed_ms,
                        warnings=warnings, limitations=limitations,
                    )

                best = top_languages[0]
                confident = best["confidence"] >= MIN_CONFIDENCE
                return self._base_result(
                    "completed", "success" if confident else "undetermined",
                    detected_language=best["language"] if confident else "unknown",
                    language_name=best["language_name"] if confident else None,
                    confidence=best["confidence"], top_languages=top_languages,
                    model_version=getattr(self._model.config, "_commit_hash", None),
                    model_revision=self.model_revision, runtime_ms=elapsed_ms,
                    input_characters=len(text), input_truncated=was_truncated,
                    warnings=warnings, limitations=limitations,
                )
            except Exception as exc:
                elapsed_ms = max(0, int((time.perf_counter() - started) * 1000))
                log.warning("Language identification inference failed (%s)", type(exc).__name__)
                return self._base_result(
                    "failed", "failed", input_characters=len(text),
                    input_truncated=len(text) > MAX_INPUT_CHARS,
                    runtime_ms=elapsed_ms,
                    warnings=["Language identification inference could not complete."],
                    limitations=["OCR text is preserved; no prediction was substituted."],
                )


_detector: Optional[LanguageDetector] = None
_detector_lock = threading.RLock()


def get_language_detector() -> LanguageDetector:
    global _detector
    with _detector_lock:
        if _detector is None:
            _detector = LanguageDetector()
        return _detector
