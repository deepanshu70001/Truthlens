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
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "runtime_ms": self.runtime_ms,
            "error": self.error,
        }


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
    stage.status = "running"
    stage_results[stage_name] = stage

    # Update analysis progress
    analysis.current_step = stage_name
    completed = [s for s, r in stage_results.items() if r.status == "completed"]
    analysis.completed_steps = json.dumps(completed)
    await db.commit()

    t0 = time.time()
    try:
        # Run the stage function in a thread pool to avoid blocking the event loop
        result = await asyncio.to_thread(func)
        stage.status = "completed"
        stage.runtime_ms = int((time.time() - t0) * 1000)

        # Convert dataclass results to dict
        if hasattr(result, "to_dict"):
            stage.result = result.to_dict()
        elif hasattr(result, "__dict__"):
            stage.result = result.__dict__

        return result

    except Exception as e:
        stage.status = "failed"
        stage.runtime_ms = int((time.time() - t0) * 1000)
        stage.error = str(e)

        log.error("Stage '%s' failed: %s", stage_name, e)

        # Log system event
        event = SystemEvent(
            analysis_id=analysis.id,
            event_type="stage_failed",
            severity="warning" if not required else "error",
            source=f"orchestrator.{stage_name}",
            message=f"Stage '{stage_name}' failed: {e}",
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
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
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
            lambda: analyze_forensics(image_data, image_format),
            analysis, db, stage_results,
            required=False,
        )

        # ── Stage 5: OCR ─────────────────────────────────────────
        from backend.app.services.ocr.service import get_ocr_service

        ocr_service = get_ocr_service(gpu=settings.DEVICE != "cpu")

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
                return generate_heatmap(model, processor, pil_image, analysis_id, device)

            xai_result = await _run_stage(
                "xai",
                run_xai,
                analysis, db, stage_results,
                required=False,
            )
        else:
            skip_stage = StageResult("xai")
            skip_stage.status = "skipped"
            skip_stage.error = "Authenticity model not available — skipping heatmap."
            stage_results["xai"] = skip_stage

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
            skip_stage = StageResult("text_normalization")
            skip_stage.status = "skipped"
            skip_stage.error = "No OCR text available."
            stage_results["text_normalization"] = skip_stage

        # ── Stage 9: Claim Extraction ────────────────────────────
        from backend.app.services.claims.extractor import extract_claims

        claims_result = None
        if text_result is not None and text_result.sentences:
            claims_result = await _run_stage(
                "claim_extraction",
                lambda: extract_claims(text_result.sentences),
                analysis, db, stage_results,
                required=False,
            )
        else:
            skip_stage = StageResult("claim_extraction")
            skip_stage.status = "skipped"
            skip_stage.error = "No sentences available for claim extraction."
            stage_results["claim_extraction"] = skip_stage

        # ── Stage 10: Fact-Check API ─────────────────────────────
        from backend.app.services.factcheck.service import search_fact_checks

        factcheck_result = None
        if claims_result is not None and claims_result.claims:
            claim_texts = [c.claim for c in claims_result.claims[:5]]

            factcheck_result = await _run_stage(
                "fact_check",
                lambda: search_fact_checks(claim_texts),
                analysis, db, stage_results,
                required=False,
            )
        else:
            # Still try with raw text if no structured claims
            if text_result is not None and text_result.normalized_text:
                factcheck_result = await _run_stage(
                    "fact_check",
                    lambda: search_fact_checks([text_result.normalized_text[:200]]),
                    analysis, db, stage_results,
                    required=False,
                )
            else:
                skip_stage = StageResult("fact_check")
                skip_stage.status = "skipped"
                skip_stage.error = "No claims or text for fact-checking."
                stage_results["fact_check"] = skip_stage

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

        evidence = EvidenceCollection()

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
        _content_lang = "en"
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

        _auth_label = assessment.label if assessment else (auth_result.label if auth_result else None)
        _fc_status = factcheck_result.status if factcheck_result else None
        _fc_rating = None
        if factcheck_result and factcheck_result.claims:
            _fc_rating = factcheck_result.claims[0].textual_rating
        _regional_top = len(regional_result.top_risk_regions) if regional_result else 0
        _virality_score = virality_result.virality_score if virality_result else 0.0

        social_impact_result = await _run_stage(
            "social_impact",
            lambda: assess_social_impact(
                ocr_text=_ocr_text,
                authenticity_label=_auth_label,
                factcheck_status=_fc_status,
                factcheck_rating=_fc_rating,
                content_language=_content_lang,
                regional_top_count=_regional_top,
                virality_score=_virality_score,
            ),
            analysis, db, stage_results,
            required=False,
        )

        # ── Stage 19: Report Assembly ────────────────────────────
        from backend.app.services.reporting.builder import build_report

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
        )

        # Store report as JSON in the analysis
        analysis.report_json = json.dumps(report)

        # ── Compile results ──────────────────────────────────────
        total_time_ms = int((time.time() - pipeline_start) * 1000)

        # Update analysis with fusion results (overrides raw model output)
        if assessment is not None:
            analysis.truth_score = assessment.overall_score
            analysis.truth_confidence = assessment.confidence
            analysis.truth_label = assessment.label
        elif auth_result is not None:
            analysis.truth_score = auth_result.authentic_score
            analysis.truth_confidence = auth_result.confidence
            analysis.truth_label = auth_result.label

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
            }),
        )
        db.add(event)
        await db.commit()

        log.info(
            "Analysis %s completed: %d/%d stages OK, %d ms total",
            analysis_id, len(completed_steps), len(stage_results), total_time_ms,
        )

    except Exception as e:
        log.error("Pipeline failed for analysis %s: %s\n%s", analysis_id, e, traceback.format_exc())
        try:
            analysis = await db.get(Analysis, analysis_id)
            if analysis:
                analysis.status = AnalysisStatus.FAILED
                analysis.error_message = f"Pipeline error: {e}"
                analysis.completed_steps = json.dumps(
                    [s for s, r in stage_results.items() if r.status == "completed"]
                )
                await db.commit()
        except Exception:
            pass
