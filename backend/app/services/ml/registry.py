"""
TruthLens — Model Registry Service.

Provides non-eager inspection and operational status reporting for all configured
machine learning models and OCR engines.

CRITICAL SAFETY AND SEMANTIC GUARANTEES:
1. Non-eager: This service NEVER calls `.load()` or triggers weight loading.
2. No inference: Readiness is verified via package inspection, disk checks, and
   singleton state inspection; no GPU allocations or inference passes occur.
3. Privacy: Absolute local filesystem paths, environment secrets, and stack traces
   are strictly sanitized. Only public filenames or repository IDs are reported.
4. Status semantics:
   - 'loaded': Resident in memory, ready for immediate inference.
   - 'available_not_loaded': Dependencies and weights verified on disk, not resident in memory.
   - 'cached_not_loaded': Weights exist in local model cache, not resident in memory.
   - 'checkpoint_missing': Expected checkpoint file was not found on disk.
   - 'dependency_unavailable': Required Python package or system binary is not installed.
   - 'download_required': Model weights are not present in local cache.
   - 'failed': Model initialization previously failed.
   - 'disabled': Explicitly disabled by application settings.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.config import settings
from backend.app.core.logging import get_logger
from backend.app.schemas.models import ModelInfo, ModelRegistryResponse

log = get_logger("model_registry")


def _is_package_available(package_name: str) -> bool:
    """Check if a Python package is importable without loading it into memory."""
    try:
        return importlib.util.find_spec(package_name) is not None
    except Exception:
        return False


def _inspect_authenticity_detector() -> ModelInfo:
    """Inspect production SigLIP authenticity detector status."""
    finetuned = settings.AUTHENTICITY_BACKEND.strip().lower() == "finetuned_siglip"
    transformers_avail = _is_package_available("transformers")
    torch_avail = _is_package_available("torch")

    if finetuned:
        model_id = "truthlens/siglip-cifake-frozen-head"
        backbone_id = settings.AUTHENTICITY_BACKBONE_MODEL
        architecture = "SigLIP (Frozen ViT-B/16 Backbone + Linear Classifier Head)"
        checkpoint_path = Path(settings.AUTHENTICITY_CHECKPOINT_PATH)
        checkpoint_name = checkpoint_path.name
        checkpoint_present = checkpoint_path.is_file()
        version_summary = "v1.0-frozen-head" if checkpoint_present else None
    else:
        model_id = settings.AUTHENTICITY_MODEL
        backbone_id = None
        architecture = "SigLIP (Vision Transformer Classifier)"
        checkpoint_name = None
        checkpoint_present = None
        version_summary = settings.AUTHENTICITY_MODEL_REVISION or "main"

    model_root = Path(settings.MODEL_CACHE_DIR)
    backbone_dir = (
        model_root / f"models--{backbone_id.replace('/', '--')}"
        if backbone_id
        else model_root / f"models--{settings.AUTHENTICITY_MODEL.replace('/', '--')}"
    )
    is_cached = backbone_dir.is_dir()

    # Query existing singleton without calling load()
    from backend.app.services.authenticity.detector import get_detector

    detector = get_detector()
    is_loaded = bool(getattr(detector, "is_loaded", False))

    device: Optional[str] = None
    if is_loaded:
        try:
            dev = detector.get_device()
            device = str(dev) if dev is not None else None
        except Exception:
            device = "unknown"

    # Status determination
    if is_loaded:
        status = "loaded"
        readiness = "ready"
        reason = "Model is resident in memory and ready for inference."
    elif not (transformers_avail and torch_avail):
        status = "dependency_unavailable"
        readiness = "unavailable"
        missing = []
        if not transformers_avail:
            missing.append("transformers")
        if not torch_avail:
            missing.append("torch")
        reason = f"Required dependency missing: {', '.join(missing)}."
    elif finetuned and not checkpoint_present:
        status = "checkpoint_missing"
        readiness = "unavailable"
        reason = f"Checkpoint file '{checkpoint_name}' was not found on disk."
    elif is_cached:
        status = "cached_not_loaded"
        readiness = "available"
        reason = "Model weights are cached locally; ready for on-demand loading."
    else:
        status = "download_required"
        readiness = "unavailable"
        reason = "Model backbone weights are not present in local cache."

    details: Dict[str, Any] = {
        "backend": settings.AUTHENTICITY_BACKEND,
        "backbone_model_id": backbone_id,
        "checkpoint_present": checkpoint_present,
        "checkpoint_filename": checkpoint_name,
        "cached": is_cached,
    }

    return ModelInfo(
        id="authenticity",
        name="SigLIP Authenticity Detector",
        task="authenticity_classification",
        provider="Google / TruthLens",
        architecture=architecture,
        model_id=model_id,
        version=version_summary,
        configured=bool(settings.AUTHENTICITY_BACKEND),
        loaded=is_loaded,
        readiness=readiness,
        status=status,
        device=device,
        capabilities=["authenticity_prediction", "gradcam_explainability", "calibrated_probabilities"],
        details=details,
        reason=reason,
    )


def _inspect_object_detector() -> ModelInfo:
    """Inspect YOLOv8 object detector status."""
    ultralytics_avail = _is_package_available("ultralytics")
    from backend.app.services.vision.detector import _resolve_model_path, get_object_detector

    configured_model = str(settings.VISION_MODEL)
    checkpoint_name = Path(configured_model).name
    resolved_path = _resolve_model_path(configured_model)
    checkpoint_present = resolved_path.is_file()

    vision = get_object_detector()
    is_loaded = bool(getattr(vision, "is_loaded", False))

    device: Optional[str] = None
    version_summary: Optional[str] = getattr(vision, "_model_sha256", None)
    if is_loaded:
        if hasattr(vision, "_model") and vision._model is not None:
            model_device = getattr(vision._model, "device", None)
            device = str(model_device) if model_device is not None else "cpu"
    elif checkpoint_present:
        version_summary = "yolov8n-official"

    if is_loaded:
        status = "loaded"
        readiness = "ready"
        reason = "YOLO object detector is resident in memory and ready for inference."
    elif not ultralytics_avail:
        status = "dependency_unavailable"
        readiness = "unavailable"
        reason = "Ultralytics package is not installed."
    elif not checkpoint_present:
        status = "checkpoint_missing"
        readiness = "unavailable"
        reason = f"YOLO weights file '{checkpoint_name}' was not found on disk."
    else:
        status = "available_not_loaded"
        readiness = "available"
        reason = "YOLO weights are available locally; ready for on-demand loading."

    details: Dict[str, Any] = {
        "checkpoint_filename": checkpoint_name,
        "checkpoint_present": checkpoint_present,
        "ultralytics_available": ultralytics_avail,
    }

    return ModelInfo(
        id="object_detection",
        name="YOLOv8 Object Detector",
        task="object_detection",
        provider="Ultralytics",
        architecture="YOLOv8n (Object Detection CNN)",
        model_id=checkpoint_name,
        version=version_summary,
        configured=bool(settings.VISION_MODEL),
        loaded=is_loaded,
        readiness=readiness,
        status=status,
        device=device,
        capabilities=["bounding_box_detection", "scene_context"],
        details=details,
        reason=reason,
    )


def _inspect_ocr_easyocr() -> ModelInfo:
    """Inspect EasyOCR primary engine status."""
    easyocr_avail = _is_package_available("easyocr")
    languages = [code.strip() for code in settings.OCR_LANGUAGES.split(",") if code.strip()]

    from backend.app.services.ocr.service import get_ocr_service

    ocr_service = get_ocr_service(gpu=settings.DEVICE != "cpu", languages=languages)
    is_loaded = bool(getattr(ocr_service, "_reader", None) is not None)

    device: Optional[str] = None
    if is_loaded:
        device = "cuda" if getattr(ocr_service, "_gpu", False) else "cpu"

    if is_loaded:
        status = "loaded"
        readiness = "ready"
        reason = "EasyOCR reader is resident in memory and ready for inference."
    elif getattr(ocr_service, "_load_attempted", False) and ocr_service._reader is None:
        status = "failed"
        readiness = "failed"
        reason = "EasyOCR initialization was attempted and failed."
    elif not easyocr_avail:
        status = "dependency_unavailable"
        readiness = "unavailable"
        reason = "EasyOCR package is not installed."
    else:
        status = "available_not_loaded"
        readiness = "available"
        reason = "EasyOCR package is installed; ready for on-demand initialization."

    details: Dict[str, Any] = {
        "configured_languages": languages,
        "gpu_configured": settings.DEVICE != "cpu",
        "easyocr_available": easyocr_avail,
    }

    return ModelInfo(
        id="ocr_easyocr",
        name="EasyOCR Text Recognition",
        task="optical_character_recognition",
        provider="JaidedAI / EasyOCR",
        architecture="CRAFT (Detection) + CRNN (Recognition)",
        model_id="easyocr",
        version=getattr(ocr_service, "_engine", None) or "EasyOCR",
        configured=bool(settings.OCR_LANGUAGES),
        loaded=is_loaded,
        readiness=readiness,
        status=status,
        device=device,
        capabilities=["multilingual_text_detection", "bounding_box_extraction"],
        details=details,
        reason=reason,
    )


def _inspect_ocr_tesseract() -> ModelInfo:
    """Inspect Tesseract OCR fallback status."""
    pytesseract_avail = _is_package_available("pytesseract")
    binary_found = False
    binary_name = "tesseract"

    if pytesseract_avail:
        try:
            import pytesseract

            cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract") or "tesseract"
            binary_name = Path(cmd).name
            if shutil.which(cmd) or Path(cmd).is_file():
                binary_found = True
        except Exception:
            binary_found = False

    from backend.app.services.ocr.service import get_ocr_service

    ocr_service = get_ocr_service()
    is_loaded = bool(getattr(ocr_service, "_tesseract", None) is not None)

    device: Optional[str] = "cpu" if is_loaded else None

    if is_loaded:
        status = "loaded"
        readiness = "ready"
        reason = "Tesseract wrapper is active in OCR service."
    elif getattr(ocr_service, "_tesseract_attempted", False) and ocr_service._tesseract is None:
        status = "failed"
        readiness = "failed"
        reason = "Tesseract fallback initialization was attempted and failed."
    elif not pytesseract_avail:
        status = "dependency_unavailable"
        readiness = "unavailable"
        reason = "pytesseract package is not installed."
    elif not binary_found:
        status = "dependency_unavailable"
        readiness = "unavailable"
        reason = f"Tesseract executable '{binary_name}' was not found on system PATH."
    else:
        status = "available_not_loaded"
        readiness = "available"
        reason = "Tesseract binary and Python wrapper are available for fallback."

    details: Dict[str, Any] = {
        "pytesseract_available": pytesseract_avail,
        "binary_present": binary_found,
    }

    return ModelInfo(
        id="ocr_tesseract",
        name="Tesseract OCR Fallback",
        task="optical_character_recognition",
        provider="Google / Tesseract",
        architecture="Tesseract LSTM OCR Engine",
        model_id="tesseract",
        version="Tesseract",
        configured=True,
        loaded=is_loaded,
        readiness=readiness,
        status=status,
        device=device,
        capabilities=["offline_text_fallback", "bounding_box_extraction"],
        details=details,
        reason=reason,
    )


def _inspect_language_detector() -> ModelInfo:
    """Inspect XLM-RoBERTa language identification model status."""
    from backend.app.services.nlp.language_detector import (
        MODEL_ID,
        get_language_detector,
    )

    transformers_avail = _is_package_available("transformers")
    model_root = Path(settings.MODEL_CACHE_DIR)
    cache_dir = model_root / f"models--{MODEL_ID.replace('/', '--')}"
    is_cached = cache_dir.is_dir()

    lang_detector = get_language_detector()
    is_loaded = bool(getattr(lang_detector, "is_loaded", False))

    device: Optional[str] = None
    if is_loaded and getattr(lang_detector, "_device", None):
        device = str(lang_detector._device)

    pinned_revision = "9865598389ca9d95637462f743f683b51d75b87b"
    version_summary = getattr(lang_detector, "model_revision", None) or pinned_revision

    load_failure = getattr(lang_detector, "_load_failure", None)

    if is_loaded:
        status = "loaded"
        readiness = "ready"
        reason = "Language identification model is resident in memory and ready for inference."
    elif load_failure:
        status = "failed"
        readiness = "failed"
        reason = f"Language detector initialization failed ({load_failure})."
    elif not transformers_avail:
        status = "dependency_unavailable"
        readiness = "unavailable"
        reason = "transformers package is not installed."
    elif is_cached:
        status = "cached_not_loaded"
        readiness = "available"
        reason = "Model weights are cached locally; ready for on-demand loading."
    else:
        status = "download_required"
        readiness = "unavailable"
        reason = "Language detector weights were not found in local cache."

    details: Dict[str, Any] = {
        "supported_languages_count": 20,
        "pinned_revision": pinned_revision,
        "cached": is_cached,
    }

    return ModelInfo(
        id="language_detection",
        name="XLM-RoBERTa Language Identification",
        task="language_identification",
        provider="papluca / Hugging Face",
        architecture="XLM-RoBERTa Sequence Classification (20 Languages)",
        model_id=MODEL_ID,
        version=version_summary,
        configured=True,
        loaded=is_loaded,
        readiness=readiness,
        status=status,
        device=device,
        capabilities=["20_language_classification", "offline_cache_only"],
        details=details,
        reason=reason,
    )


def get_model_registry() -> ModelRegistryResponse:
    """Build and return the complete model registry.

    Inspects all configured models without triggering any eager loading or inference.
    """
    models: List[ModelInfo] = [
        _inspect_authenticity_detector(),
        _inspect_object_detector(),
        _inspect_ocr_easyocr(),
        _inspect_ocr_tesseract(),
        _inspect_language_detector(),
    ]

    loaded_count = sum(1 for m in models if m.loaded)
    available_count = sum(1 for m in models if m.readiness == "available")
    ready_count = sum(1 for m in models if m.readiness in ("ready", "available"))

    return ModelRegistryResponse(
        status="ok",
        models=models,
        total_models=len(models),
        loaded_count=loaded_count,
        available_count=available_count,
        ready_count=ready_count,
        environment=settings.APP_ENV,
        message="Model registry reports configured services without eager loading.",
    )
