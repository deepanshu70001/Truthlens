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
        self._load_attempted = False
        self._tesseract_attempted = False
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._reader is not None or self._tesseract is not None

    def load(self) -> None:
        with self._lock:
            self._load()

    def _load(self) -> None:
        if self.is_loaded or self._load_attempted:
            return
        self._load_attempted = True
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
                # Inference must be reproducible and must never make an
                # implicit network request to fetch model weights.
                download_enabled=False,
            )
            self._engine = "EasyOCR"
            return
        except Exception as exc:
            self._load_warnings.append(f"EasyOCR unavailable ({type(exc).__name__}).")
            log.warning("EasyOCR unavailable (%s): %s", type(exc).__name__, exc)
        self._load_tesseract()

    def _load_tesseract(self) -> None:
        if self._tesseract_attempted or self._tesseract is not None:
            return
        self._tesseract_attempted = True
        try:
            import os
            import subprocess
            import pytesseract
            command = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract") or "tesseract"
            resolved_cmd = shutil.which(command) or (command if Path(command).is_file() else None)
            if not resolved_cmd:
                raise FileNotFoundError("Tesseract executable not found")
            prefix = os.environ.get("TESSDATA_PREFIX", "")
            if prefix and Path(prefix).name != "tessdata" and (Path(prefix) / "tessdata").is_dir():
                os.environ["TESSDATA_PREFIX"] = str(Path(prefix) / "tessdata")
            elif not prefix:
                cmd_dir = Path(resolved_cmd).parent
                candidate_tessdata = cmd_dir / "tessdata"
                if candidate_tessdata.is_dir():
                    os.environ["TESSDATA_PREFIX"] = str(candidate_tessdata)
            pytesseract.get_tesseract_version()
            raw_available = pytesseract.get_languages(config="")
            available_languages = {
                Path(lang.replace("\\", "/")).name for lang in raw_available
            }
            if not available_languages:
                try:
                    out = subprocess.check_output([resolved_cmd, "--list-langs"], text=True, stderr=subprocess.STDOUT)
                    for line in out.splitlines()[1:]:
                        clean = Path(line.strip().replace("\\", "/")).name
                        if clean and clean != "osd":
                            available_languages.add(clean)
                except Exception:
                    pass
            requested_languages = set(_tesseract_languages(self._languages).split("+"))
            missing_languages = sorted(requested_languages - available_languages)
            if missing_languages:
                raise RuntimeError(
                    "Tesseract language packs unavailable: " + ", ".join(missing_languages)
                )
            self._tesseract = pytesseract
            self._engine = "PyTesseract"
        except Exception as exc:
            if str(exc).startswith("Tesseract language packs unavailable:"):
                detail = f"{type(exc).__name__}: {exc}"
            elif isinstance(exc, FileNotFoundError):
                detail = "FileNotFoundError: Tesseract executable not found"
            else:
                detail = type(exc).__name__
            self._load_warnings.append(f"Tesseract fallback unavailable ({detail}).")

    def unload(self) -> None:
        with self._lock:
            self._reader = self._tesseract = None
            self._engine = None
            self._load_attempted = False
            self._tesseract_attempted = False
            self._load_warnings.clear()

    def _run_easyocr(self, image: Image.Image, warnings: List[str]) -> tuple[List[OCRBlock], bool]:
        blocks: List[OCRBlock] = []
        had_invalid_output = False
        width, height = image.size
        raw_results = self._reader.readtext(np.asarray(image))
        for item in raw_results or []:
            try:
                bbox, raw_text, raw_confidence = item
                text = str(raw_text).strip()
                points = np.asarray(bbox, dtype=float)
                if not text:
                    continue
                if points.shape != (4, 2) or not np.isfinite(points).all():
                    warnings.append("An OCR region with invalid coordinates was discarded.")
                    had_invalid_output = True
                    continue
                # No geometric transform is applied; clipping retains the
                # coordinate frame of the original image.
                points[:, 0] = np.clip(points[:, 0], 0, width)
                points[:, 1] = np.clip(points[:, 1], 0, height)
                if points[:, 0].max() <= points[:, 0].min() or points[:, 1].max() <= points[:, 1].min():
                    warnings.append("An OCR region with zero visible area was discarded.")
                    had_invalid_output = True
                    continue
                try:
                    confidence = float(raw_confidence)
                except (TypeError, ValueError):
                    confidence = float("nan")
                if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                    confidence = None
                    warnings.append("An OCR region had invalid confidence; its text was retained without confidence.")
                    had_invalid_output = True
                blocks.append(OCRBlock(
                    text, confidence,
                    [[int(round(x)), int(round(y))] for x, y in points],
                ))
            except (TypeError, ValueError, IndexError):
                warnings.append("An OCR region with malformed output was discarded.")
                had_invalid_output = True
        return blocks, had_invalid_output

    def _result_from_blocks(
        self,
        blocks: List[OCRBlock],
        engine: Optional[str],
        warnings: List[str],
        started: float,
        partial: bool = False,
    ) -> OCRResult:
        raw_text = "\n".join(block.text for block in blocks)
        normalized = _normalize_text(raw_text)
        confidences = [b.confidence for b in blocks if b.confidence is not None]
        reason = None if normalized else (
            f"{engine or 'OCR'} ran for {', '.join(self._languages) or 'the configured languages'} "
            "but found no readable text regions. The image may contain no text, or text may be too small, blurry, or low-contrast to recognize."
        )
        return OCRResult(
            text=raw_text, normalized_text=normalized, language=_detect_primary_script(raw_text),
            blocks=blocks,
            total_confidence=float(np.mean(confidences)) if confidences else None,
            word_count=len(normalized.split()) if normalized else 0,
            runtime_ms=max(0, int((time.perf_counter() - started) * 1000)),
            has_text=bool(normalized), warnings=warnings,
            status="partial" if partial else "completed", engine=engine,
            no_text_reason=reason, language_configuration=self._languages,
        )

    def _run_tesseract(self, image: Image.Image, warnings: Optional[List[str]] = None) -> List[OCRBlock]:
        data = self._tesseract.image_to_data(image.convert("RGB"), lang=_tesseract_languages(self._languages), output_type=self._tesseract.Output.DICT)
        blocks: List[OCRBlock] = []
        image_width, image_height = image.size
        for i, text in enumerate(data.get("text", [])):
            text = str(text).strip()
            if not text:
                continue
            try:
                x, y = int(data["left"][i]), int(data["top"][i])
                w, h = int(data["width"][i]), int(data["height"][i])
                try:
                    conf_raw = float(data.get("conf", [])[i])
                    conf = conf_raw / 100 if np.isfinite(conf_raw) and 0 <= conf_raw <= 100 else None
                except (ValueError, TypeError, IndexError, KeyError):
                    conf = None
                if conf is None and warnings is not None:
                    warnings.append("A Tesseract OCR region had invalid confidence; its text was retained without confidence.")
                if w <= 0 or h <= 0:
                    continue
                # Tesseract returns xywh boxes. Clip them to the EXIF-corrected
                # original image frame, then discard boxes with no visible area.
                left, top = max(0, x), max(0, y)
                right, bottom = min(image_width, x + w), min(image_height, y + h)
                if right <= left or bottom <= top:
                    continue
                blocks.append(OCRBlock(text, conf, [[left,top],[right,top],[right,bottom],[left,bottom]]))
            except (KeyError, ValueError, TypeError, IndexError):
                continue
        return blocks

    def extract(self, image: Image.Image) -> OCRResult:
        with self._lock:
            return self._extract(image)

    def _extract(self, image: Image.Image) -> OCRResult:
        started = time.perf_counter()
        warnings = list(self._load_warnings)
        if not isinstance(image, Image.Image):
            reason = "OCR input is not a decoded image."
            return OCRResult(status="failed", runtime_ms=max(0, int((time.perf_counter() - started) * 1000)), warnings=[reason], no_text_reason=reason, language_configuration=self._languages)
        try:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > settings.MAX_IMAGE_PIXELS:
                reason = "Image dimensions exceed the configured OCR safety limit."
                return OCRResult(status="failed", runtime_ms=max(0, int((time.perf_counter() - started) * 1000)), warnings=[reason], no_text_reason=reason, language_configuration=self._languages)
            supported_formats = {None, "JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF"}
            if image.format and image.format.upper() not in supported_formats:
                reason = "Image format is not supported for OCR."
                return OCRResult(status="failed", runtime_ms=max(0, int((time.perf_counter() - started) * 1000)), warnings=[reason], no_text_reason=reason, language_configuration=self._languages)
            image.load()
            # Work on an OCR-local copy; leave the caller's decoded image intact.
            ocr_image = image.copy().convert("RGB")
        except Exception:
            reason = "Image could not be decoded for OCR."
            return OCRResult(status="failed", runtime_ms=max(0, int((time.perf_counter() - started) * 1000)), warnings=[reason], no_text_reason=reason, language_configuration=self._languages)
        if not self.is_loaded:
            reason = "OCR could not run because no configured OCR engine is available."
            return OCRResult(status="unavailable", runtime_ms=int((time.perf_counter()-started)*1000), warnings=warnings or [reason], no_text_reason=reason, language_configuration=self._languages)
        if self._reader is None:
            try:
                blocks = self._run_tesseract(ocr_image, warnings)
            except Exception as exc:
                warnings.append(f"Tesseract inference failed ({type(exc).__name__}).")
                reason = "OCR could not complete using the configured engine."
                return OCRResult(status="failed", engine="PyTesseract", runtime_ms=max(0, int((time.perf_counter() - started) * 1000)), warnings=warnings, no_text_reason=reason, language_configuration=self._languages)
            return self._result_from_blocks(blocks, "PyTesseract", warnings, started)

        primary_engine = "EasyOCR"
        primary_failed = False
        invalid_output = False
        blocks: List[OCRBlock] = []
        try:
            blocks, invalid_output = self._run_easyocr(ocr_image, warnings)
        except Exception as exc:
            primary_failed = True
            warnings.append(f"EasyOCR inference failed ({type(exc).__name__}).")
            # Do not retry a broken primary model on every subsequent image.
            self._reader = None
        if blocks:
            return self._result_from_blocks(blocks, primary_engine, warnings, started, partial=invalid_output)

        warnings.append("EasyOCR produced no usable text; attempting the Tesseract fallback.")
        self._load_tesseract()
        if self._tesseract is not None:
            try:
                fallback_blocks = self._run_tesseract(ocr_image, warnings)
            except Exception as exc:
                warnings.append(f"Tesseract inference failed ({type(exc).__name__}).")
                if primary_failed:
                    reason = "All available OCR engines failed during inference."
                    return OCRResult(status="failed", engine="PyTesseract", runtime_ms=max(0, int((time.perf_counter() - started) * 1000)), warnings=warnings, no_text_reason=reason, language_configuration=self._languages)
                # EasyOCR completed, so its no-text result remains valid even
                # though the optional second opinion failed.
                warnings.append("The primary OCR engine completed without readable text.")
                return self._result_from_blocks([], primary_engine, warnings, started, partial=invalid_output)
            if fallback_blocks:
                warnings.append("Readable OCR text was produced by the Tesseract fallback.")
                return self._result_from_blocks(fallback_blocks, "PyTesseract", warnings, started)
            warnings.append("Tesseract completed but found no readable text.")

        if primary_failed and self._tesseract is None:
            reason = "OCR could not complete using the available engine."
            return OCRResult(status="failed", engine=primary_engine, runtime_ms=max(0, int((time.perf_counter() - started) * 1000)), warnings=warnings, no_text_reason=reason, language_configuration=self._languages)
        return self._result_from_blocks([], "PyTesseract" if self._tesseract is not None else primary_engine, warnings, started, partial=invalid_output)


_ocr_service: Optional[OCRService] = None
_ocr_service_lock = threading.RLock()


def get_ocr_service(gpu: bool = True, languages: Optional[List[str]] = None) -> OCRService:
    global _ocr_service
    with _ocr_service_lock:
        if _ocr_service is None or _ocr_service._gpu != gpu or (languages is not None and _ocr_service._languages != languages):
            _ocr_service = OCRService(languages=languages, gpu=gpu)
        return _ocr_service
