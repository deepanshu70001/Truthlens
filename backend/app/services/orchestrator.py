"""
TruthLens — Analysis orchestrator (Stage 3 — real pipeline).

Coordinates the full analysis pipeline:
upload → preprocessing → metadata → quality → authenticity →
forensics → OCR → vision → XAI

Each stage has: status, runtime, result, error.
One failing optional module does NOT crash the entire analysis.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageOps
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.app.core.logging import get_logger
from backend.app.models import Analysis, AnalysisStatus, ImageAsset, ModelRun, SystemEvent
from backend.app.services.stage_contract import (
    normalize_service_status,
)

log = get_logger("orchestrator")

PIPELINE_STAGES = (
    "preprocessing", "metadata", "quality", "authenticity", "forensics", "ocr",
    "vision", "xai", "text_normalization", "language_identification", "claim_extraction", "fact_check",
    "provenance", "hashing", "evidence_collection", "fusion", "explanation",
    "virality", "regional_spread", "social_impact", "report_assembly",
)


class StageTimeoutError(Exception):
    """Raised when the orchestrator's bounded wait expires for a stage."""


def _safe_message(value: Any) -> str:
    """Keep public diagnostics useful without exposing paths or exception text."""
    value = str(value)
    value = re.sub(r"(?i)(?:[a-z]:\\|/)(?:[^\s:]+[\\/])+[^\s:]*", "[redacted path]", value)
    value = re.sub(r"(?i)(api[_ -]?key|key|token|secret|password)\s*[:=]\s*\S+", r"\1=[redacted]", value)
    return value[:500]


def _safe_messages(values: Any) -> List[str]:
    if not isinstance(values, (list, tuple)):
        return []
    return [_safe_message(value) for value in values if isinstance(value, str)][:50]


def _store_progress(analysis: Analysis, stage_results: Dict[str, "StageResult"], current: str | None) -> None:
    statuses = {name: "pending" for name in PIPELINE_STAGES}
    statuses.update({name: stage.status for name, stage in stage_results.items() if name in statuses})
    if current in statuses:
        statuses[current] = "running"
    completed = [name for name in PIPELINE_STAGES if statuses[name] == "completed"]
    analysis.current_step = current
    # Keep the API-facing completed_steps list stable while retaining detailed
    # terminal statuses for accurate progress recovery.
    analysis.completed_steps = json.dumps({"completed_steps": completed, "stage_statuses": statuses})


class StageResult:
    """Result of a single pipeline stage."""
    def __init__(self, name: str):
        self.name = name
        self.status: str = "pending"
        self.outcome: Optional[str] = None
        self.runtime_ms: int = 0
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.error_message: Optional[str] = None
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.stage_id: Optional[str] = None
        self.correlation_id: Optional[str] = None
        self.error_category: Optional[str] = None
        self.output_reference: Optional[str] = None
        self.module_identity: Optional[Dict[str, Any]] = None
        self.warnings: List[str] = []
        self.limitations: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "outcome": self.outcome,
            "runtime_ms": self.runtime_ms,
            "error": self.error,
            "error_message": self.error_message,
            "result": self.result,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "stage_id": self.stage_id,
            "correlation_id": self.correlation_id,
            "error_category": self.error_category,
            "output_reference": self.output_reference,
            "module_identity": self.module_identity,
            "warnings": self.warnings,
            "limitations": self.limitations,
        }


def _record_skipped_stage(stage_results: Dict[str, StageResult], name: str, reason: str, analysis_id: str) -> None:
    stage = StageResult(name)
    stage.stage_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"truthlens:{analysis_id}:{name}"))
    stage.correlation_id = analysis_id
    stage.status = "skipped"
    stage.outcome = "not_applicable"
    stage.warnings = [_safe_message(reason)]
    now = datetime.now(timezone.utc).isoformat()
    stage.started_at = now
    stage.completed_at = now
    stage_results[name] = stage


def _record_unavailable_stage(stage_results: Dict[str, StageResult], name: str, reason: str, analysis_id: str) -> None:
    stage = StageResult(name)
    stage.stage_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"truthlens:{analysis_id}:{name}"))
    stage.correlation_id = analysis_id
    stage.status = "unavailable"
    stage.outcome = "unavailable"
    stage.error_category = "OPTIONAL_DEPENDENCY_OR_MODEL_UNAVAILABLE"
    stage.error = "The optional stage could not run because a required model or resource was unavailable."
    stage.error_message = stage.error
    stage.warnings = [_safe_message(reason)]
    now = datetime.now(timezone.utc).isoformat()
    stage.started_at = now
    stage.completed_at = now
    stage.output_reference = _output_reference(name)
    stage_results[name] = stage


async def _run_stage(
    stage_name: str,
    func,
    analysis: Analysis,
    db: AsyncSession,
    stage_results: Dict[str, StageResult],
    required: bool = False,
    timeout_seconds: Optional[float] = None,
) -> Optional[Any]:
    """
    Run a single pipeline stage with error handling.

    If required=False, a failure is logged but does not crash the pipeline.
    If required=True, a failure raises an exception to abort.
    """
    stage = StageResult(stage_name)
    stage.stage_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"truthlens:{analysis.id}:{stage_name}"))
    stage.correlation_id = str(analysis.id)
    stage.status = "running"
    stage.started_at = datetime.now(timezone.utc).isoformat()
    stage_results[stage_name] = stage
    log.info("Analysis stage started", extra={"analysis_id": str(analysis.id), "stage_id": stage.stage_id, "stage": stage_name, "status": "running"})

    # Update analysis progress
    _store_progress(analysis, stage_results, stage_name)
    await db.commit()

    t0 = time.perf_counter()
    try:
        # Run the stage function in a thread pool to avoid blocking the event loop
        work = asyncio.to_thread(func)
        if timeout_seconds:
            try:
                result = await asyncio.wait_for(work, timeout=timeout_seconds)
            except asyncio.TimeoutError as exc:
                raise StageTimeoutError from exc
        else:
            result = await work
        stage.status, stage.outcome, invalid_result = normalize_service_status(result)
        if invalid_result:
            stage.error_category = "INVALID_MODULE_RESULT"
            stage.error = invalid_result
            stage.error_message = invalid_result
        elif stage.status in {"failed", "unavailable"}:
            stage.error_category = "MODULE_REPORTED_FAILURE" if stage.status == "failed" else "DEPENDENCY_UNAVAILABLE"
            stage.error = "The stage reported that it could not complete."
            stage.error_message = stage.error
        elif stage.status == "skipped":
            stage.outcome = stage.outcome or "not_applicable"
        if stage.status == "failed":
            stage.error_category = stage.error_category or "MODULE_REPORTED_FAILURE"
        elif stage.status == "unavailable":
            stage.error_category = "OPTIONAL_DEPENDENCY_OR_SERVICE_UNAVAILABLE"
        elif stage.status == "skipped":
            stage.error_category = "PREREQUISITE_UNAVAILABLE"
        stage.runtime_ms = max(0, int((time.perf_counter() - t0) * 1000))
        stage.completed_at = datetime.now(timezone.utc).isoformat()

        # Convert dataclass results to dict
        if invalid_result is None and hasattr(result, "to_dict"):
            result_dict = result.to_dict()
            stage.result = {"available": True, "type": type(result).__name__}
            stage.output_reference = _output_reference(stage_name)
            stage.warnings = _safe_messages(result_dict.get("warnings") if isinstance(result_dict, dict) else [])
            stage.limitations = _safe_messages(result_dict.get("limitations") if isinstance(result_dict, dict) else [])
        elif invalid_result is None and isinstance(result, dict):
            stage.result = {"available": True, "type": "dict"}
            stage.output_reference = _output_reference(stage_name)
            stage.warnings = _safe_messages(result.get("warnings"))
            stage.limitations = _safe_messages(result.get("limitations"))
        elif invalid_result is None and hasattr(result, "__dict__"):
            stage.result = {"available": True, "type": type(result).__name__}
            stage.output_reference = _output_reference(stage_name)
            stage.warnings = _safe_messages(getattr(result, "warnings", []))
            stage.limitations = _safe_messages(getattr(result, "limitations", []))
        stage.module_identity = _module_identity(result)
        # Persist terminal status and progress at the stage boundary.
        _store_progress(analysis, stage_results, None)
        await db.commit()
        log.info("Analysis stage finished", extra={"analysis_id": str(analysis.id), "stage_id": stage.stage_id, "stage": stage_name, "status": stage.status, "duration_ms": stage.runtime_ms})

        if invalid_result:
            return None

        return result

    except Exception as e:
        stage.status = "unavailable" if isinstance(e, ImportError) else "failed"
        stage.error_category = (
            "DEPENDENCY_UNAVAILABLE" if isinstance(e, ImportError)
            else "STAGE_TIMEOUT" if isinstance(e, StageTimeoutError)
            else "STAGE_EXECUTION_ERROR"
        )
        stage.runtime_ms = max(0, int((time.perf_counter() - t0) * 1000))
        stage.completed_at = datetime.now(timezone.utc).isoformat()
        stage.error = "The stage could not complete."
        stage.error_message = stage.error

        log.error("Analysis stage failed", extra={"analysis_id": str(analysis.id), "stage_id": stage.stage_id, "stage": stage_name, "status": stage.status, "duration_ms": stage.runtime_ms, "error_category": stage.error_category})

        # Log system event
        await _rollback_safely(db)
        event = SystemEvent(
            analysis_id=analysis.id,
            event_type="stage_failed",
            severity="warning" if not required else "error",
            source=f"orchestrator.{stage_name}",
            message=f"Stage '{stage_name}' failed ({type(e).__name__}).",
        )
        try:
            db.add(event)
            _store_progress(analysis, stage_results, None)
            await db.commit()
        except Exception as persist_exc:
            await _rollback_safely(db)
            log.error("Could not persist stage failure state for %s (%s)", stage_name, type(persist_exc).__name__)

        if required:
            raise

        return None


def _module_identity(result: Any) -> Optional[Dict[str, Any]]:
    if result is None:
        return None
    keys = (
        "model_name", "model_version", "model_id", "model_sha256", "provider",
        "engine", "model_revision", "implementation", "implementation_version",
        "method", "method_version",
    )
    identity = {
        key: _safe_message(getattr(result, key))
        for key in keys if getattr(result, key, None) is not None
    }
    return identity or None


def _output_reference(stage_name: str) -> str:
    report_fields = {
        "vision": "objects", "claim_extraction": "claims",
        "fact_check": "fact_checks", "evidence_collection": "evidence",
        "fusion": "fusion", "language_identification": "language_identification",
    }
    return f"report.{report_fields.get(stage_name, stage_name)}"


async def _rollback_safely(db: AsyncSession) -> None:
    try:
        await db.rollback()
    except Exception as exc:
        log.error("Database rollback failed while recording terminal analysis state (%s)", type(exc).__name__)


async def _persist_terminal_failure(
    analysis_id: str,
    db: AsyncSession,
    stage_results: Dict[str, StageResult],
    user_message: str = "Analysis could not be completed due to an internal pipeline error.",
) -> Exception | None:
    """Persist a safe terminal report, returning an error if persistence failed.

    The worker needs to distinguish an analysis failure that was durably
    recorded from a database outage that prevented terminal persistence. In
    the latter case it can apply its bounded retry and lease recovery policy.
    """
    await _rollback_safely(db)
    for name in PIPELINE_STAGES:
        stage = stage_results.get(name)
        if stage is None or stage.status == "pending":
            _record_skipped_stage(
                stage_results, name, "Pipeline terminated before this stage was run.", analysis_id
            )
        elif stage.status == "running":
            stage.status = "failed"
            stage.error_category = "PIPELINE_ABORTED"
            stage.error = "The pipeline terminated before this stage completed."
            stage.error_message = stage.error
            stage.completed_at = datetime.now(timezone.utc).isoformat()

    failed = StageResult("pipeline")
    failed.stage_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"truthlens:{analysis_id}:pipeline"))
    failed.correlation_id = analysis_id
    failed.status = "failed"
    failed.error_category = "PIPELINE_EXECUTION_ERROR"
    failed.error = user_message
    failed.error_message = user_message
    failed.started_at = failed.completed_at = datetime.now(timezone.utc).isoformat()
    stage_results["pipeline"] = failed
    try:
        analysis = await db.get(Analysis, analysis_id)
        if analysis is None:
            log.error("Cannot persist terminal failure: analysis %s was not found", analysis_id)
            return RuntimeError("Analysis record is unavailable for terminal persistence.")

        stages = {name: stage.to_dict() for name, stage in stage_results.items()}
        try:
            from backend.app.services.reporting.builder import build_report
            asset = analysis.image_asset
            image_info = None
            if asset:
                image_info = {
                    "original_filename": asset.original_filename,
                    "sha256_hash": asset.sha256_hash,
                    "perceptual_hash": asset.perceptual_hash,
                    "mime_type": asset.mime_type,
                    "file_size_bytes": asset.file_size_bytes,
                    "width": asset.width,
                    "height": asset.height,
                    "format": asset.format,
                }
            report = build_report(
                analysis_id=analysis_id, status="failed", image_info=image_info,
                assessment=None, stages=stages,
                submitted_at=analysis.created_at.isoformat() if analysis.created_at else None,
            )
        except Exception as report_exc:
            log.error("Report builder failed during terminal error recovery (%s)", type(report_exc).__name__)
            report = {
                "analysis_id": analysis_id, "status": "failed", "stages": stages,
                "warnings": ["The analysis pipeline could not complete."],
            }

        analysis.report_json = json.dumps(report, default=str)
        analysis.status = AnalysisStatus.FAILED
        analysis.error_message = user_message
        analysis.completed_at = datetime.now(timezone.utc)
        _store_progress(analysis, stage_results, None)
        analysis.processing_time_ms = analysis.processing_time_ms or 0
        await db.commit()
        return None
    except Exception as commit_exc:
        log.error("Initial terminal failure persistence failed (%s)", type(commit_exc).__name__)
        await _rollback_safely(db)
        # A transient commit/report serialization failure gets one bounded
        # retry with a minimal report so the row does not remain processing.
        try:
            analysis = await db.get(Analysis, analysis_id)
            if analysis is None:
                return
            analysis.status = AnalysisStatus.FAILED
            analysis.error_message = user_message
            analysis.completed_at = datetime.now(timezone.utc)
            analysis.report_json = json.dumps({
                "analysis_id": analysis_id,
                "status": "failed",
                "stages": {name: stage.to_dict() for name, stage in stage_results.items()},
                "warnings": ["The analysis pipeline could not complete."],
            }, default=str)
            _store_progress(analysis, stage_results, None)
            analysis.processing_time_ms = analysis.processing_time_ms or 0
            await db.commit()
            return None
        except Exception as persist_exc:
            await _rollback_safely(db)
            log.error("Could not persist terminal failure for analysis %s (%s)", analysis_id, type(persist_exc).__name__)
            return persist_exc


async def run_analysis_pipeline(analysis_id: str, db: AsyncSession) -> None:
    """
    Run the full analysis pipeline on an uploaded image.

    Stages:
    1. preprocessing (required) — already done during upload, verify
    2. metadata (optional) — EXIF extraction
    3. quality (optional) — image quality signals
    4. authenticity (optional) — AI-generated detection
    5. forensics (optional) — classical forensic signals
    6. ocr (optional) — text extraction
    7. vision (optional) — object detection
    8. xai (optional) — heatmap generation (depends on authenticity)
    """
    stage_results: Dict[str, StageResult] = {}

    try:
        # Fetch analysis and image asset
        analysis = await db.get(Analysis, analysis_id)
        if not analysis:
            log.error("Analysis %s not found", analysis_id)
            return

        preprocessing_event = next((event for event in (analysis.system_events or []) if event.event_type == "preprocessing"), None)
        if preprocessing_event:
            try:
                preprocessing_result = json.loads(preprocessing_event.metadata_json or "{}")
            except (json.JSONDecodeError, TypeError):
                preprocessing_result = {"status": "unavailable", "warnings": ["Preprocessing details could not be loaded."]}
            preprocessing_stage = StageResult("preprocessing")
            preprocessing_stage.stage_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"truthlens:{analysis_id}:preprocessing"))
            preprocessing_stage.correlation_id = analysis_id
            preprocessing_stage.status = preprocessing_result.get("status") if preprocessing_result.get("status") in {"completed", "failed", "unavailable"} else "unavailable"
            preprocessing_stage.runtime_ms = int(preprocessing_result.get("runtime_ms") or 0)
            preprocessing_stage.completed_at = preprocessing_event.created_at.isoformat()
            preprocessing_stage.started_at = (preprocessing_event.created_at - timedelta(milliseconds=preprocessing_stage.runtime_ms)).isoformat()
            preprocessing_stage.result = {key: value for key, value in preprocessing_result.items() if key != "status"}
            preprocessing_stage.output_reference = "report.preprocessing"
            preprocessing_stage.warnings = _safe_messages(preprocessing_result.get("warnings"))
            preprocessing_stage.error_category = None if preprocessing_stage.status == "completed" else "PREPROCESSING_FAILED"
            if preprocessing_stage.status in {"failed", "unavailable"}:
                preprocessing_stage.error = "Preprocessing could not be completed."
                preprocessing_stage.error_message = preprocessing_stage.error
            stage_results["preprocessing"] = preprocessing_stage
        else:
            preprocessing_result = {"status": "unavailable", "warnings": ["No preprocessing record exists for this analysis."]}
            preprocessing_stage = StageResult("preprocessing")
            preprocessing_stage.stage_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"truthlens:{analysis_id}:preprocessing"))
            preprocessing_stage.correlation_id = analysis_id
            preprocessing_stage.status = "unavailable"
            preprocessing_stage.error_category = "PREPROCESSING_RECORD_UNAVAILABLE"
            preprocessing_stage.error = "No preprocessing record exists for this analysis."
            stage_results["preprocessing"] = preprocessing_stage

        # Get image asset
        if not analysis.image_asset:
            log.error("No image asset for analysis %s", analysis_id)
            persistence_error = await _persist_terminal_failure(
                analysis_id, db, stage_results,
                "No image file was available for analysis.",
            )
            if persistence_error is not None:
                raise persistence_error
            return

        image_path = Path(analysis.image_asset.storage_path)
        if not image_path.exists():
            log.error("Image file not found: %s", image_path)
            persistence_error = await _persist_terminal_failure(
                analysis_id, db, stage_results,
                "The uploaded image is no longer available in storage.",
            )
            if persistence_error is not None:
                raise persistence_error
            return

        # Read image data
        image_data = image_path.read_bytes()
        pil_image = ImageOps.exif_transpose(
            Image.open(io.BytesIO(image_data))
        ).convert("RGB")
        image_format = analysis.image_asset.format or "JPEG"

        # Mark as processing
        analysis.status = AnalysisStatus.PROCESSING
        await db.commit()

        pipeline_start = time.time()

        # ── Stage 1: Metadata ────────────────────────────────────
        from backend.app.services.metadata.service import extract_metadata

        metadata_result = await _run_stage(
            "metadata",
            lambda: extract_metadata(image_data),
            analysis, db, stage_results,
            required=False,
        )

        # ── Stage 2: Quality ─────────────────────────────────────
        from backend.app.services.quality.service import analyze_quality

        quality_result = await _run_stage(
            "quality",
            lambda: analyze_quality(image_data),
            analysis, db, stage_results,
            required=False,
        )

        # ── Stage 3: Authenticity Detection ──────────────────────
        from backend.app.services.authenticity.detector import get_detector

        detector = get_detector()

        def run_authenticity():
            if not detector.is_loaded:
                detector.load(device=settings.DEVICE)
            return detector.detect(pil_image)

        auth_result = await _run_stage(
            "authenticity",
            run_authenticity,
            analysis, db, stage_results,
            required=False,
        )

        # Record model run for authenticity
        if auth_result is not None:
            model_run = ModelRun(
                analysis_id=analysis_id,
                model_id=auth_result.model_name,
                model_version=auth_result.model_version,
                task="authenticity_detection",
                device=str(detector.get_device()) if detector.get_device() else "unknown",
                input_hash=analysis.image_asset.sha256_hash,
                result_json=json.dumps(auth_result.to_dict()),
                confidence=auth_result.confidence,
                runtime_ms=auth_result.runtime_ms,
            )
            db.add(model_run)
            await db.commit()

        # ── Stage 4: Forensic Analysis ───────────────────────────
        from backend.app.services.forensics.analyzer import analyze_forensics

        forensics_result = await _run_stage(
            "forensics",
            lambda: analyze_forensics(image_data, image_format, analysis_id=analysis_id),
            analysis, db, stage_results,
            required=False,
            timeout_seconds=settings.FORENSIC_TIMEOUT_SECONDS,
        )
        if forensics_result is not None:
            stage_results["forensics"].outcome = getattr(forensics_result, "outcome", None)

        # ── Stage 5: OCR ─────────────────────────────────────────
        from backend.app.services.ocr.service import OCRResult, get_ocr_service

        ocr_languages = [language.strip() for language in settings.OCR_LANGUAGES.split(",") if language.strip()]
        ocr_service = get_ocr_service(gpu=settings.DEVICE != "cpu", languages=ocr_languages)

        def run_ocr():
            if not ocr_service.is_loaded:
                ocr_service.load()
            return ocr_service.extract(pil_image)

        ocr_result = await _run_stage(
            "ocr",
            run_ocr,
            analysis, db, stage_results,
            required=False,
            timeout_seconds=settings.OCR_TIMEOUT_SECONDS,
        )
        ocr_stage = stage_results.get("ocr")
        if ocr_stage and ocr_result is not None and ocr_stage.status in {"completed", "partial"}:
            ocr_stage.outcome = "success" if getattr(ocr_result, "has_text", False) else "no_text"
        if ocr_result is None:
            ocr_stage = stage_results.get("ocr")
            failure_reason = getattr(ocr_stage, "error", None) or "The OCR stage returned no result."
            ocr_result = OCRResult(
                status=getattr(ocr_stage, "status", "failed"),
                no_text_reason=f"OCR could not complete: {failure_reason}",
                warnings=[f"OCR stage did not produce text ({failure_reason})."],
                language_configuration=ocr_languages,
            )

        # ── Stage 6: Vision (Object Detection) ───────────────────
        from backend.app.services.vision.detector import get_object_detector

        obj_detector = get_object_detector()

        def run_vision():
            if not obj_detector.is_loaded:
                obj_detector.load(device=settings.DEVICE)
            return obj_detector.detect(pil_image)

        vision_result = await _run_stage(
            "vision",
            run_vision,
            analysis, db, stage_results,
            required=False,
            timeout_seconds=settings.VISION_TIMEOUT_SECONDS,
        )
        if vision_result is not None:
            stage_results["vision"].outcome = getattr(vision_result, "outcome", None)

        # ── Stage 7: XAI (Heatmap) ──────────────────────────────
        from backend.app.services.xai.gradcam import generate_heatmap

        xai_result = None
        if auth_result is not None and stage_results.get("authenticity") is not None and stage_results["authenticity"].status == "completed" and detector.is_loaded:

            def run_xai():
                model = detector.get_internal_model()
                processor = detector.get_processor()
                device = detector.get_device()
                labels = getattr(getattr(model, "config", None), "id2label", {}) or {}
                scores = auth_result.raw_scores or {}
                target_class = None
                if labels and scores:
                    normalized_scores = {str(name).strip().lower(): value for name, value in scores.items()}
                    candidates = [(int(class_id), normalized_scores[str(class_label).strip().lower()]) for class_id, class_label in labels.items() if str(class_label).strip().lower() in normalized_scores]
                    if candidates:
                        target_class = max(candidates, key=lambda candidate: candidate[1])[0]
                return generate_heatmap(model, processor, pil_image, analysis_id, device, target_class=target_class, model_name=auth_result.model_name)

            xai_result = await _run_stage(
                "xai",
                run_xai,
                analysis, db, stage_results,
                required=False,
                timeout_seconds=settings.XAI_TIMEOUT_SECONDS,
            )
            if xai_result is not None:
                stage_results["xai"].outcome = getattr(xai_result, "outcome", None)
        else:
            _record_unavailable_stage(stage_results, "xai", "The authenticity model output or compatible target layer was unavailable.", analysis_id)

        # ── Stage 8: Text Normalization ──────────────────────────
        from backend.app.services.text.normalizer import normalize_text

        text_result = None
        if ocr_result is not None and ocr_result.has_text:
            text_result = await _run_stage(
                "text_normalization",
                lambda: normalize_text(ocr_result.text),
                analysis, db, stage_results,
                required=False,
            )
        else:
            _record_skipped_stage(stage_results, "text_normalization", "No OCR text was available.", analysis_id)

        from backend.app.services.nlp.language_detector import get_language_detector
        language_detector = get_language_detector()
        language_text = text_result.normalized_text if text_result is not None else ""
        language_result = await _run_stage(
            "language_identification",
            lambda: language_detector.detect(language_text, device=settings.DEVICE),
            analysis, db, stage_results, required=False,
            timeout_seconds=settings.LANGUAGE_DETECTION_TIMEOUT_SECONDS,
        )
        if language_result is not None:
            stage_results["language_identification"].outcome = getattr(language_result, "outcome", None)

        # ── Stage 9: Claim Extraction ────────────────────────────
        from backend.app.services.claims.extractor import extract_claims

        claims_result = None
        if text_result is not None:
            source_regions = []
            source_cursor = 0
            for region_index, block in enumerate(getattr(ocr_result, "blocks", []) or []):
                block_text = getattr(block, "text", "")
                if not isinstance(block_text, str) or not block_text.strip():
                    continue
                region_start = text_result.normalized_text.find(block_text.strip(), source_cursor)
                if region_start < 0:
                    continue
                region_end = region_start + len(block_text.strip())
                source_regions.append({
                    "region_id": f"ocr_region_{region_index}",
                    "text_start": region_start,
                    "text_end": region_end,
                    "bbox": getattr(block, "bbox", None),
                })
                source_cursor = region_end
            claims_result = await _run_stage(
                "claim_extraction",
                lambda: extract_claims(
                    text_result.normalized_text,
                    max_claims=settings.FACTCHECK_MAX_CLAIMS,
                    language_result=language_result,
                    sentence_spans=text_result.sentence_spans,
                    source_regions=source_regions,
                    ocr_confidence=getattr(ocr_result, "total_confidence", None) if ocr_result else None,
                ),
                analysis, db, stage_results,
                required=False,
            )
        else:
            _record_skipped_stage(stage_results, "claim_extraction", "No sentences were available for claim extraction.", analysis_id)

        # ── Stage 10: Fact-Check API ─────────────────────────────
        from backend.app.services.factcheck.service import search_fact_checks

        factcheck_result = None
        eligible_claims = [c for c in (claims_result.claims if claims_result else []) if c.verification_eligibility == "ELIGIBLE"]
        if eligible_claims:
            claim_texts = [c.normalized_text or c.claim for c in eligible_claims]
            claim_ids = [c.claim_id for c in eligible_claims]
            language_code = (
                getattr(language_result, "detected_language", None)
                if language_result is not None and getattr(language_result, "outcome", None) == "success"
                else None
            )

            factcheck_result = await _run_stage(
                "fact_check",
                lambda: search_fact_checks(claim_texts, language_code=language_code,
                    max_results_per_claim=settings.FACTCHECK_MAX_RESULTS_PER_CLAIM,
                    claim_ids=claim_ids),
                analysis, db, stage_results,
                required=False,
            )
            if factcheck_result is not None:
                factcheck_result.claim_verifications.extend({
                    "claim_id": c.claim_id, "original_claim": c.original_text,
                    "verdict": "UNVERIFIABLE",
                    "reasoning_summary": c.eligibility_reason,
                    "supporting_evidence_ids": [], "contradicting_evidence_ids": [],
                    "conflicting_evidence_ids": [], "verification_status": "COMPLETED",
                    "confidence": None,
                    "limitations": ["This text was not sent for factual retrieval because it was not classified as an eligible factual claim."],
                } for c in (claims_result.claims if claims_result else []) if c.verification_eligibility != "ELIGIBLE")
        else:
            from backend.app.services.factcheck.service import FactCheckResult
            factcheck_result = FactCheckResult(
                status="no_results",
                warnings=["No eligible factual claims were identified; external lookup was not run."],
                claim_verifications=[{
                    "claim_id": c.claim_id,
                    "original_claim": c.original_text,
                    "verdict": "UNVERIFIABLE" if c.verification_eligibility != "ELIGIBLE" else "INSUFFICIENT_EVIDENCE",
                    "reasoning_summary": c.eligibility_reason or "No searchable claim was available.",
                    "supporting_evidence_ids": [], "contradicting_evidence_ids": [],
                    "conflicting_evidence_ids": [], "verification_status": "COMPLETED",
                    "confidence": None,
                    "limitations": ["No external fact-check lookup was performed."],
                } for c in (claims_result.claims if claims_result else [])],
            )
            _record_skipped_stage(stage_results, "fact_check", "No eligible factual claims were available for external lookup.", analysis_id)

        # ── Stage 11: Provenance (Web Detection) ─────────────────
        from backend.app.services.provenance.service import search_provenance

        provenance_result = await _run_stage(
            "provenance",
            lambda: search_provenance(image_data),
            analysis, db, stage_results,
            required=False,
            timeout_seconds=settings.PROVENANCE_API_TIMEOUT_SECONDS + 5.0,
        )
        if provenance_result is not None:
            stage_results["provenance"].outcome = getattr(provenance_result, "outcome", None)

        # ── Stage 12: Image Hashing ──────────────────────────────
        from backend.app.services.hashing.service import compute_hashes

        hash_result = await _run_stage(
            "hashing",
            lambda: compute_hashes(image_data),
            analysis, db, stage_results,
            required=False,
        )

        # ── Stage 13: Evidence Collection ────────────────────────
        from backend.app.services.evidence.schema import (
            EvidenceCollection,
            build_evidence_from_authenticity,
            build_evidence_from_forensics,
            build_evidence_from_ocr,
            build_evidence_from_metadata,
            build_evidence_from_factcheck,
            build_evidence_from_provenance,
            build_evidence_from_xai,
            build_evidence_from_vision,
        )

        evidence = EvidenceCollection(analysis_id=analysis_id)
        collection_stage = StageResult("evidence_collection")
        collection_stage.stage_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"truthlens:{analysis_id}:evidence_collection"))
        collection_stage.correlation_id = analysis_id
        collection_t0 = time.time()
        collection_stage.status = "running"
        collection_stage.started_at = datetime.now(timezone.utc).isoformat()
        stage_results["evidence_collection"] = collection_stage

        # Collect evidence from all modules
        for item in build_evidence_from_authenticity(auth_result):
            evidence.add(item)
        for item in build_evidence_from_forensics(forensics_result):
            evidence.add(item)
        for item in build_evidence_from_ocr(ocr_result, text_result):
            evidence.add(item)
        for item in build_evidence_from_metadata(metadata_result):
            evidence.add(item)
        for item in build_evidence_from_factcheck(factcheck_result):
            evidence.add(item)
        for item in build_evidence_from_provenance(provenance_result):
            evidence.add(item)
        for item in build_evidence_from_xai(xai_result):
            evidence.add(item)
        for item in build_evidence_from_vision(vision_result):
            evidence.add(item)
        collection_stage.status = "completed"
        collection_stage.runtime_ms = int((time.time() - collection_t0) * 1000)
        collection_stage.completed_at = datetime.now(timezone.utc).isoformat()
        collection_stage.result = {"evidence_count": evidence.count}
        collection_stage.output_reference = "report.evidence"

        # ── Stage 14: Evidence Fusion ────────────────────────────
        from backend.app.services.fusion.engine import fuse_evidence

        assessment = await _run_stage(
            "fusion",
            lambda: fuse_evidence(
                evidence,
                stage_results={name: stage for name, stage in stage_results.items() if name != "fusion"},
                analysis_id=analysis_id,
                correlation_id=analysis_id,
            ),
            analysis, db, stage_results,
            required=False,
        )

        # ── Stage 15: Explanation Generation ─────────────────────
        from backend.app.services.explanation.generator import generate_explanation

        explanation_result = None
        if assessment is not None:
            explanation_result = await _run_stage(
                "explanation",
                lambda: generate_explanation(
                    evidence, assessment.label, assessment_conflicting=assessment.conflicting,
                ),
                analysis, db, stage_results,
                required=False,
            )
        else:
            _record_skipped_stage(
                stage_results, "explanation", "Evidence fusion did not produce an assessment.", analysis_id
            )

        # ── Stage 16: Virality Estimation ────────────────────────
        from backend.app.services.virality.engine import estimate_virality

        # Prepare inputs for virality engine
        _ocr_text = ocr_result.text if ocr_result and ocr_result.has_text else None
        _claims_list = claims_result.to_dict().get("claims") if claims_result else None
        _text_norm_dict = text_result.to_dict() if text_result else None
        _vision_dict = vision_result.to_dict() if vision_result else None

        virality_result = await _run_stage(
            "virality",
            lambda: estimate_virality(
                ocr_text=_ocr_text,
                claims=_claims_list,
                text_normalized=_text_norm_dict,
                vision_result=_vision_dict,
            ),
            analysis, db, stage_results,
            required=False,
        )

        # ── Stage 17: Regional Spread Estimation ─────────────────
        from backend.app.services.regional.engine import estimate_regional_spread

        # Determine content language
        _content_lang = None
        if ocr_result and hasattr(ocr_result, "language") and ocr_result.language:
            _content_lang = ocr_result.language

        regional_result = await _run_stage(
            "regional_spread",
            lambda: estimate_regional_spread(
                content_language=_content_lang,
                ocr_text=_ocr_text,
            ),
            analysis, db, stage_results,
            required=False,
        )

        # ── Stage 18: Social Impact Assessment ───────────────────
        from backend.app.services.social_impact.engine import assess_social_impact

        _claims_verification = factcheck_result.to_dict().get("claim_verifications", {}) if factcheck_result else {}

        social_impact_result = await _run_stage(
            "social_impact",
            lambda: assess_social_impact(
                ocr_text=_ocr_text,
                content_language=_content_lang,
                claims=_claims_list,
                claim_verifications=_claims_verification,
            ),
            analysis, db, stage_results,
            required=False,
        )

        # ── Stage 19: Report Assembly ────────────────────────────
        from backend.app.services.reporting.builder import build_report
        report_stage = StageResult("report_assembly")
        report_stage.stage_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"truthlens:{analysis_id}:report_assembly"))
        report_stage.correlation_id = analysis_id
        report_t0 = time.time()
        report_stage.status = "running"
        report_stage.started_at = datetime.now(timezone.utc).isoformat()
        stage_results["report_assembly"] = report_stage

        # Build image info dict
        img_info = None
        if analysis.image_asset:
            ia = analysis.image_asset
            img_info = {
                "original_filename": ia.original_filename,
                "sha256_hash": ia.sha256_hash,
                "perceptual_hash": ia.perceptual_hash,
                "mime_type": ia.mime_type,
                "file_size_bytes": ia.file_size_bytes,
                "width": ia.width,
                "height": ia.height,
                "format": ia.format,
            }

        report = build_report(
            analysis_id=analysis_id,
            status="completed",
            image_info=img_info,
            authenticity=auth_result.to_dict() if auth_result else None,
            metadata=metadata_result.__dict__ if metadata_result else None,
            quality=(quality_result.to_dict() if hasattr(quality_result, "to_dict") else quality_result.__dict__) if quality_result else None,
            preprocessing=preprocessing_result,
            forensics=forensics_result.to_dict() if forensics_result else None,
            objects=vision_result.to_dict() if vision_result else None,
            ocr=ocr_result.to_dict() if ocr_result else None,
            language_identification=language_result.to_dict() if language_result else None,
            text_normalized=text_result.to_dict() if text_result else None,
            claims=claims_result.to_dict() if claims_result else None,
            fact_checks=factcheck_result.to_dict() if factcheck_result else None,
            provenance=provenance_result.to_dict() if provenance_result else None,
            xai=xai_result.to_dict() if xai_result else None,
            hashes=hash_result.to_dict() if hash_result else None,
            evidence=evidence.to_dict(),
            assessment=assessment.to_dict() if assessment else None,
            explanation=explanation_result.to_dict() if explanation_result else None,
            virality=virality_result.to_dict() if virality_result else None,
            regional_spread=regional_result.to_dict() if regional_result else None,
            social_impact=social_impact_result.to_dict() if social_impact_result else None,
            processing_time_ms=int((time.time() - pipeline_start) * 1000),
            stages={name: stage.to_dict() for name, stage in stage_results.items()},
            submitted_at=analysis.created_at.isoformat() if analysis.created_at else None,
        )
        report_stage.status = "completed"
        report_stage.runtime_ms = int((time.time() - report_t0) * 1000)
        report_stage.completed_at = datetime.now(timezone.utc).isoformat()
        report_stage.result = {"report_created": True}
        report_stage.output_reference = "report"
        report["stages"]["report_assembly"] = report_stage.to_dict()

        # Store report as JSON in the analysis
        analysis.report_json = json.dumps(report, default=str)

        # ── Compile results ──────────────────────────────────────
        total_time_ms = int((time.time() - pipeline_start) * 1000)

        # Update analysis with fusion results (overrides raw model output)
        if assessment is not None:
            analysis.truth_score = assessment.overall_score
            analysis.truth_confidence = assessment.confidence
            analysis.truth_label = assessment.label

        analysis.processing_time_ms = total_time_ms
        has_degraded_stage = any(
            stage.status in {"partial", "failed", "unavailable"}
            for stage in stage_results.values()
        )
        # The persisted enum remains backward compatible. The report and GET
        # status endpoint expose `partial` when optional stages degraded.
        analysis.status = AnalysisStatus.COMPLETED
        analysis.current_step = None
        analysis.completed_at = datetime.now(timezone.utc)

        completed_steps = [s for s, r in stage_results.items() if r.status == "completed"]
        _store_progress(analysis, stage_results, None)
        if has_degraded_stage and report.get("status") == "completed":
            report["status"] = "partial"
            analysis.report_json = json.dumps(report, default=str)

        await db.commit()

        # Log completion event
        event = SystemEvent(
            analysis_id=analysis_id,
            event_type="analysis_completed",
            severity="info",
            source="orchestrator",
            message=(
                f"Analysis {analysis_id} completed in {total_time_ms}ms. "
                f"Stages: {len(completed_steps)} completed, "
                f"{sum(1 for r in stage_results.values() if r.status == 'failed')} failed, "
                f"{sum(1 for r in stage_results.values() if r.status == 'skipped')} skipped."
            ),
            metadata_json=json.dumps({
                "stages": {k: v.to_dict() for k, v in stage_results.items()},
                "total_time_ms": total_time_ms,
            }, default=str),
        )
        db.add(event)
        await db.commit()

        log.info("Analysis %s terminal: report_status=%s, %d/%d stages completed, %d ms total",
                 analysis_id, report.get("status"), len(completed_steps), len(stage_results), total_time_ms)

    except Exception as e:
        log.error("Pipeline failed for analysis %s (%s)", analysis_id, type(e).__name__)
        persistence_error = await _persist_terminal_failure(analysis_id, db, stage_results)
        if persistence_error is not None:
            # Surface the persistence error so the durable worker can classify
            # transient database failures and retry within the configured cap.
            raise persistence_error from e
