"""Lazy OCR with EasyOCR primary and optional, opportunistic Tesseract fallback."""
from __future__ import annotations

import re
import shutil
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from backend.app.config import settings
from backend.app.core.logging import get_logger

log = get_logger("ocr")
DEFAULT_LANGUAGES = ["en", "hi"]
INDIAN_LANGUAGES = {"en": "English", "hi": "Hindi (Devanagari)"}


@dataclass
class OCRBlock:
    text: str
    confidence: Optional[float]
    bbox: List[List[int]]


@dataclass
class OCRResult:
    text: str = ""
    normalized_text: str = ""
    language: str = "unknown"
    blocks: List[OCRBlock] = field(default_factory=list)
    total_confidence: Optional[float] = None
    word_count: int = 0
    runtime_ms: int = 0
    has_text: bool = False
    no_text_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    status: str = "completed"
    engine: Optional[str] = None
    language_configuration: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status, "engine": self.engine,
            "language_configuration": self.language_configuration,
            "text": self.text or "", "raw_text": self.text or "",
            "no_text_reason": self.no_text_reason,
            "normalized_text": self.normalized_text, "language": self.language,
            "blocks": [{"text": b.text, "confidence": round(b.confidence, 4) if b.confidence is not None else None, "bbox": b.bbox} for b in self.blocks],
            "total_confidence": round(self.total_confidence, 4) if self.total_confidence is not None else None,
            "word_count": self.word_count, "runtime_ms": self.runtime_ms,
            "has_text": self.has_text, "detected": self.has_text, "warnings": self.warnings,
        }


def _normalize_text(text: str) -> str:
    """Canonicalize Unicode and whitespace only; preserve punctuation and numbers."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def _detect_primary_script(text: str) -> str:
    counts: Dict[str, int] = {}
    for char in text:
        name = unicodedata.name(char, "")
        for marker, code in (("DEVANAGARI", "hi"), ("LATIN", "en"), ("TAMIL", "ta"), ("TELUGU", "te"), ("BENGALI", "bn"), ("GURMUKHI", "pa"), ("GUJARATI", "gu"), ("KANNADA", "kn")):
            if marker in name:
                counts[code] = counts.get(code, 0) + 1
                break
    return max(counts, key=counts.get) if counts else "unknown"


def _tesseract_languages(languages: List[str]) -> str:
    mapping = {"en": "eng", "hi": "hin", "mr": "mar", "ne": "nep", "ta": "tam", "te": "tel", "kn": "kan", "gu": "guj", "pa": "pan", "bn": "ben"}
    resolved = [mapping[lang] for lang in languages if lang in mapping]
    if not resolved:
        raise ValueError("No configured language has a Tesseract language mapping")
    return "+".join(resolved)


class OCRService:
    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = True):
        self._languages = list(DEFAULT_LANGUAGES if languages is None else languages)
        self._gpu = gpu
        self._reader = None
        self._tesseract = None
        self._engine: Optional[str] = None
        self._load_warnings: List[str] = []
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._reader is not None or self._tesseract is not None

    def load(self) -> None:
        with self._lock:
            self._load()

    def _load(self) -> None:
        if self.is_loaded:
            return
        if not self._languages or any(not isinstance(code, str) or not code.strip() for code in self._languages):
            self._load_warnings.append("OCR language configuration is invalid.")
            return
        try:
            import easyocr
            model_dir = settings.OCR_MODEL_DIR
            # EasyOCR otherwise writes to the user's home directory, which can
            # be read-only for the app process or deployment container.
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            self._reader = easyocr.Reader(
                self._languages,
                gpu=self._gpu,
                verbose=False,
                model_storage_directory=model_dir,
            )
            self._engine = "EasyOCR"
            return
        except Exception as exc:
            self._load_warnings.append(f"EasyOCR unavailable ({type(exc).__name__}).")
            log.warning("EasyOCR unavailable (%s): %s", type(exc).__name__, exc)
        self._load_tesseract()

    def _load_tesseract(self) -> None:
        try:
            import pytesseract
            if not shutil.which("tesseract") and not getattr(pytesseract.pytesseract, "tesseract_cmd", None):
                raise RuntimeError("Tesseract executable not found")
            pytesseract.get_tesseract_version()
            self._tesseract = pytesseract
            self._engine = "PyTesseract"
        except Exception as exc:
            self._load_warnings.append(f"Tesseract fallback unavailable ({type(exc).__name__}).")

    def unload(self) -> None:
        with self._lock:
            self._reader = self._tesseract = None
            self._engine = None

    def _run_tesseract(self, image: Image.Image) -> List[OCRBlock]:
        data = self._tesseract.image_to_data(image.convert("RGB"), lang=_tesseract_languages(self._languages), output_type=self._tesseract.Output.DICT)
        blocks: List[OCRBlock] = []
        for i, text in enumerate(data.get("text", [])):
            text = str(text).strip()
            if not text:
                continue
            try:
                x, y = int(data["left"][i]), int(data["top"][i])
                w, h = int(data["width"][i]), int(data["height"][i])
                conf_raw = float(data["conf"][i])
                conf = conf_raw / 100 if np.isfinite(conf_raw) and 0 <= conf_raw <= 100 else None
                if w <= 0 or h <= 0:
                    continue
                blocks.append(OCRBlock(text, conf, [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]))
            except (KeyError, ValueError, TypeError, IndexError):
                continue
        return blocks

    def extract(self, image: Image.Image) -> OCRResult:
        with self._lock:
            return self._extract(image)

    def _extract(self, image: Image.Image) -> OCRResult:
        started = time.perf_counter()
        warnings = list(self._load_warnings)
        if not self.is_loaded:
            reason = "OCR could not run because no configured OCR engine is available."
            return OCRResult(status="unavailable", runtime_ms=int((time.perf_counter()-started)*1000), warnings=warnings or [reason], no_text_reason=reason, language_configuration=self._languages)
        blocks: List[OCRBlock] = []
        engine = self._engine
        fallback_succeeded = False
        try:
            if self._reader is not None:
                raw_results = self._reader.readtext(np.asarray(image.convert("RGB")))
                for item in raw_results or []:
                    try:
                        bbox, text, confidence = item
                        text = str(text).strip()
                        points = np.asarray(bbox, dtype=float)
                        conf = float(confidence)
                        width, height = image.size
                        if not text or points.shape != (4, 2) or not np.isfinite(points).all() or np.any(points < 0) or np.any(points[:, 0] > width) or np.any(points[:, 1] > height) or not np.isfinite(conf) or not 0 <= conf <= 1:
                            warnings.append("An invalid EasyOCR block was discarded.")
                            continue
                        blocks.append(OCRBlock(text, conf, [[int(x), int(y)] for x,y in points]))
                    except (TypeError, ValueError, IndexError):
                        warnings.append("An invalid EasyOCR block was discarded.")
            else:
                blocks = self._run_tesseract(image)
        except Exception as exc:
            warnings.append(f"{engine} inference failed ({type(exc).__name__}).")
            if self._reader is not None:
                self._reader = None
                self._load_tesseract()
                if self._tesseract is not None:
                    engine = self._engine
                    try:
                        blocks = self._run_tesseract(image)
                        fallback_succeeded = True
                    except Exception as fallback_exc:
                        warnings.append(f"Tesseract inference failed ({type(fallback_exc).__name__}).")
                        blocks = []
                    else:
                        warnings.append("OCR output was produced by the Tesseract fallback.")
                else:
                    engine = "EasyOCR"
            if not blocks:
                if fallback_succeeded:
                    reason = f"{engine} ran for {', '.join(self._languages) or 'the configured languages'} but found no readable text regions. The image may contain no text, or text may be too small, blurry, or low-contrast to recognize."
                    return OCRResult(status="completed", engine=engine, runtime_ms=int((time.perf_counter()-started)*1000), warnings=warnings, no_text_reason=reason, language_configuration=self._languages)
                reason = f"OCR could not complete using {engine or 'an available engine'}; check the OCR warnings for details."
                return OCRResult(status="failed" if engine else "unavailable", engine=engine, runtime_ms=int((time.perf_counter()-started)*1000), warnings=warnings, no_text_reason=reason, language_configuration=self._languages)

        raw_text = "\n".join(block.text for block in blocks)
        normalized = _normalize_text(raw_text)
        confidences = [b.confidence for b in blocks if b.confidence is not None]
        mean_confidence = float(np.mean(confidences)) if confidences else None
        no_text_reason = None if normalized else (
            f"{engine or 'OCR'} ran for {', '.join(self._languages) or 'the configured languages'} "
            "but found no readable text regions. The image may contain no text, or text may be too small, blurry, or low-contrast to recognize."
        )
        return OCRResult(
            text=raw_text, normalized_text=normalized, language=_detect_primary_script(raw_text),
            blocks=blocks, total_confidence=mean_confidence,
            word_count=len(normalized.split()) if normalized else 0,
            runtime_ms=int((time.perf_counter()-started)*1000), has_text=bool(normalized),
            warnings=warnings, status="partial" if any("discarded" in w for w in warnings) else "completed",
            engine=engine, no_text_reason=no_text_reason, language_configuration=self._languages,
        )


_ocr_service: Optional[OCRService] = None


def get_ocr_service(gpu: bool = True, languages: Optional[List[str]] = None) -> OCRService:
    global _ocr_service
    if _ocr_service is None or _ocr_service._gpu != gpu or (languages is not None and _ocr_service._languages != languages):
        _ocr_service = OCRService(languages=languages, gpu=gpu)
    return _ocr_service
