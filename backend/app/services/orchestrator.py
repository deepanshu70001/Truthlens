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

log = get_logger("orchestrator")


class StageResult:
    """Result of a single pipeline stage."""
    def __init__(self, name: str):
        self.name = name
        self.status: str = "pending"  # pending | running | completed | failed | skipped
        self.runtime_ms: int = 0
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.stage_id: Optional[str] = None
        self.correlation_id: Optional[str] = None
        self.error_category: Optional[str] = None
        self.output_reference: Optional[str] = None
        self.warnings: List[str] = []
        self.limitations: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "runtime_ms": self.runtime_ms,
            "error": self.error,
            "result": self.result,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "stage_id": self.stage_id,
            "correlation_id": self.correlation_id,
            "error_category": self.error_category,
            "output_reference": self.output_reference,
            "warnings": self.warnings,
            "limitations": self.limitations,
        }


def _record_skipped_stage(stage_results: Dict[str, StageResult], name: str, reason: str, analysis_id: str) -> None:
    stage = StageResult(name)
    stage.stage_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"truthlens:{analysis_id}:{name}"))
    stage.correlation_id = analysis_id
    stage.status = "skipped"
    stage.error = reason
    stage.error_category = "PREREQUISITE_UNAVAILABLE"
    stage.started_at = datetime.now(timezone.utc).isoformat()
    stage.completed_at = stage.started_at
    stage_results[name] = stage


async def _run_stage(
    stage_name: str,
    func,
    analysis: Analysis,
    db: AsyncSession,
    stage_results: Dict[str, StageResult],
    required: bool = False,
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
    analysis.current_step = stage_name
    completed = [s for s, r in stage_results.items() if r.status == "completed"]
    analysis.completed_steps = json.dumps(completed)
    await db.commit()

    t0 = time.time()
    try:
        # Run the stage function in a thread pool to avoid blocking the event loop
        result = await asyncio.to_thread(func)
        result_status = getattr(result, "status", None)
        stage.status = result_status if result_status in {"partial", "failed", "unavailable", "skipped"} else "completed"
        if stage.status == "failed":
            stage.error_category = "MODULE_REPORTED_FAILURE"
        elif stage.status == "unavailable":
            stage.error_category = "OPTIONAL_DEPENDENCY_OR_SERVICE_UNAVAILABLE"
        elif stage.status == "skipped":
            stage.error_category = "PREREQUISITE_UNAVAILABLE"
        stage.runtime_ms = int((time.time() - t0) * 1000)
        stage.completed_at = datetime.now(timezone.utc).isoformat()

        # Convert dataclass results to dict
        if hasattr(result, "to_dict"):
            result_dict = result.to_dict()
            stage.result = {"available": True, "type": type(result).__name__}
            stage.output_reference = f"report.{stage_name}"
            stage.warnings = list(result_dict.get("warnings") or []) if isinstance(result_dict, dict) else []
            stage.limitations = list(result_dict.get("limitations") or []) if isinstance(result_dict, dict) else []
        elif hasattr(result, "__dict__"):
            stage.result = {"available": True, "type": type(result).__name__}
            stage.output_reference = f"report.{stage_name}"
        stage.warnings = list(getattr(result, "warnings", []) or [])
        stage.limitations = list(getattr(result, "limitations", []) or [])
        log.info("Analysis stage finished", extra={"analysis_id": str(analysis.id), "stage_id": stage.stage_id, "stage": stage_name, "status": stage.status, "duration_ms": stage.runtime_ms})

        return result

    except Exception as e:
        stage.status = "unavailable" if isinstance(e, ImportError) else "failed"
        stage.error_category = "DEPENDENCY_UNAVAILABLE" if isinstance(e, ImportError) else "STAGE_EXECUTION_ERROR"
        stage.runtime_ms = int((time.time() - t0) * 1000)
        stage.completed_at = datetime.now(timezone.utc).isoformat()
        stage.error = f"{type(e).__name__}: stage execution failed"

        log.exception("Stage '%s' failed for analysis %s", stage_name, analysis.id)
        log.error("Analysis stage failed", extra={"analysis_id": str(analysis.id), "stage_id": stage.stage_id, "stage": stage_name, "status": stage.status, "duration_ms": stage.runtime_ms, "error_category": stage.error_category})

        # Log system event
        event = SystemEvent(
            analysis_id=analysis.id,
            event_type="stage_failed",
            severity="warning" if not required else "error",
            source=f"orchestrator.{stage_name}",
            message=f"Stage '{stage_name}' failed ({type(e).__name__}).",
        )
        db.add(event)
        await db.commit()

        if required:
            raise

        return None


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
            preprocessing_stage.warnings = list(preprocessing_result.get("warnings") or [])
            preprocessing_stage.error_category = None if preprocessing_stage.status == "completed" else "PREPROCESSING_FAILED"
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
            analysis.status = AnalysisStatus.FAILED
            analysis.error_message = "No image file found for this analysis."
            await db.commit()
            return

        image_path = Path(analysis.image_asset.storage_path)
        if not image_path.exists():
            log.error("Image file not found: %s", image_path)
            analysis.status = AnalysisStatus.FAILED
            analysis.error_message = "Image file missing from storage."
            await db.commit()
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
        )

        # ── Stage 5: OCR ─────────────────────────────────────────
        from backend.app.services.ocr.service import get_ocr_service

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
        )

        # ── Stage 7: XAI (Heatmap) ──────────────────────────────
        from backend.app.services.xai.gradcam import generate_heatmap

        xai_result = None
        if auth_result is not None and detector.is_loaded:

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
            )
        else:
            _record_skipped_stage(stage_results, "xai", "Authenticity model unavailable; heatmap skipped.", analysis_id)

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

        # ── Stage 9: Claim Extraction ────────────────────────────
        from backend.app.services.claims.extractor import extract_claims

        claims_result = None
        if text_result is not None:
            claims_result = await _run_stage(
                "claim_extraction",
                lambda: extract_claims(
                    text_result.sentences,
                    max_claims=settings.FACTCHECK_MAX_CLAIMS,
                    language=getattr(ocr_result, "language", "unknown") if ocr_result else "unknown",
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
            language_code = getattr(ocr_result, "language", "en") if ocr_result else "en"
            if not isinstance(language_code, str) or len(language_code) != 2:
                language_code = "en"

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
        )

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

        # ── Stage 14: Evidence Fusion ────────────────────────────
        from backend.app.services.fusion.engine import fuse_evidence

        assessment = await _run_stage(
            "fusion",
            lambda: fuse_evidence(evidence),
            analysis, db, stage_results,
            required=False,
        )

        # ── Stage 15: Explanation Generation ─────────────────────
        from backend.app.services.explanation.generator import generate_explanation

        explanation_result = None
        if assessment is not None:
            explanation_result = await _run_stage(
                "explanation",
                lambda: generate_explanation(evidence, assessment.label),
                analysis, db, stage_results,
                required=False,
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
        analysis.status = AnalysisStatus.COMPLETED
        analysis.current_step = None
        analysis.completed_at = datetime.now(timezone.utc)

        completed_steps = [
            s for s, r in stage_results.items() if r.status == "completed"
        ]
        analysis.completed_steps = json.dumps(completed_steps)

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

        log.info(
            "Analysis %s completed: %d/%d stages OK, %d ms total",
            analysis_id, len(completed_steps), len(stage_results), total_time_ms,
        )

    except Exception as e:
        log.exception("Pipeline failed for analysis %s", analysis_id)
        try:
            analysis = await db.get(Analysis, analysis_id)
            if analysis:
                failed_stage = StageResult("pipeline")
                failed_stage.stage_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"truthlens:{analysis_id}:pipeline"))
                failed_stage.correlation_id = analysis_id
                failed_stage.status = "failed"
                failed_stage.error = "Internal pipeline error."
                failed_stage.completed_at = datetime.now(timezone.utc).isoformat()
                stage_results["pipeline"] = failed_stage
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
                analysis.report_json = json.dumps(build_report(
                    analysis_id=analysis_id,
                    status="failed",
                    image_info=image_info,
                    assessment=None,
                    stages={name: stage.to_dict() for name, stage in stage_results.items()},
                    submitted_at=analysis.created_at.isoformat() if analysis.created_at else None,
                ), default=str)
                analysis.status = AnalysisStatus.FAILED
                analysis.error_message = "Analysis could not be completed due to an internal pipeline error."
                analysis.completed_steps = json.dumps(
                    [s for s, r in stage_results.items() if r.status == "completed"]
                )
                await db.commit()
        except Exception:
            log.exception("Failed to persist failure report for analysis %s", analysis_id)
            pass
