"""
TruthLens — API endpoint: Analysis status and results.

GET /api/v1/analysis/{analysis_id}
GET /api/v1/analysis/{analysis_id}/report
GET /api/v1/analysis/{analysis_id}/heatmap
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.exceptions import AnalysisNotFoundError
from backend.app.config import settings
from backend.app.db.session import get_db
from backend.app.models import Analysis
from backend.app.schemas import (
    AnalysisOut,
    AnalysisProgress,
    ImageAssetOut,
    ModelRunOut,
)

router = APIRouter()


@router.get(
    "/analysis/{analysis_id}/image",
    summary="Retrieve uploaded analysis image",
    description="Serves the validated original image for this analysis without exposing its storage path.",
)
async def get_analysis_image(analysis_id: str, db: AsyncSession = Depends(get_db)):
    analysis = await db.get(Analysis, analysis_id)
    if not analysis or not analysis.image_asset:
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        UUID(analysis_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Image not found")
    asset_path = Path(analysis.image_asset.storage_path).resolve()
    originals_root = settings.originals_dir.resolve()
    if not asset_path.is_relative_to(originals_root) or not asset_path.is_file():
        raise HTTPException(status_code=404, detail="Image is not available")
    return FileResponse(asset_path, media_type=analysis.image_asset.mime_type, filename=analysis.image_asset.original_filename)


def _parse_completed_steps(raw: str | None) -> List[str]:
    """Parse the current progress object or the legacy JSON list."""
    if not raw:
        return []
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            value = value.get("completed_steps", [])
        return value if isinstance(value, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


@router.get(
    "/analysis/{analysis_id}",
    response_model=AnalysisOut,
    summary="Get analysis status and results",
    description="Retrieve the current status, progress, and results of an analysis.",
)
async def get_analysis(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
):
    analysis = await db.get(Analysis, analysis_id)
    if not analysis:
        raise AnalysisNotFoundError(analysis_id)

    completed_steps = _parse_completed_steps(analysis.completed_steps)
    total_steps = 20
    try:
        progress_state = json.loads(analysis.completed_steps or "[]")
    except (json.JSONDecodeError, TypeError):
        progress_state = {}
    stage_statuses = progress_state.get("stage_statuses", {}) if isinstance(progress_state, dict) else {}
    if not isinstance(stage_statuses, dict):
        stage_statuses = {}
    else:
        stage_statuses = dict(stage_statuses)
    if not stage_statuses:
        stage_statuses = {name: "completed" for name in completed_steps}
    if analysis.current_step:
        stage_statuses.setdefault(analysis.current_step, "running")
    percentage = round(sum(1 for value in stage_statuses.values() if value in {
        "completed", "partial", "failed", "unavailable", "skipped"
    }) / total_steps * 100, 1)
    response_status = analysis.status.value if hasattr(analysis.status, 'value') else analysis.status
    if response_status == "completed" and analysis.report_json:
        try:
            stored_report = json.loads(analysis.report_json)
            if stored_report.get("status") == "partial":
                response_status = "partial"
        except (json.JSONDecodeError, TypeError):
            pass

    progress = AnalysisProgress(
        current_step=analysis.current_step,
        completed_steps=completed_steps,
        total_steps=total_steps,
        stage_statuses=stage_statuses,
        percentage=percentage,
    )

    image_out = None
    if analysis.image_asset:
        image_out = ImageAssetOut.model_validate(analysis.image_asset)

    model_runs_out = [
        ModelRunOut.model_validate(mr) for mr in (analysis.model_runs or [])
    ]

    return AnalysisOut(
        analysis_id=analysis.id,
        status=response_status,
        progress=progress,
        truth_score=analysis.truth_score,
        truth_confidence=analysis.truth_confidence,
        truth_label=analysis.truth_label,
        image=image_out,
        model_runs=model_runs_out,
        processing_time_ms=analysis.processing_time_ms,
        error_message=analysis.error_message,
        created_at=analysis.created_at,
        updated_at=analysis.updated_at,
        completed_at=analysis.completed_at,
    )


@router.get(
    "/analysis/{analysis_id}/report",
    summary="Get full analysis report",
    description=(
        "Returns the complete analysis report including authenticity assessment, "
        "forensic signals, OCR, claims, fact-checks, provenance, evidence, "
        "fusion assessment, and explanation."
    ),
)
async def get_analysis_report(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
):
    analysis = await db.get(Analysis, analysis_id)
    if not analysis:
        raise AnalysisNotFoundError(analysis_id)

    # If report is stored, return it directly
    if analysis.report_json:
        try:
            return json.loads(analysis.report_json)
        except (json.JSONDecodeError, TypeError):
            pass

    # Otherwise build a minimal response
    status_val = analysis.status.value if hasattr(analysis.status, "value") else analysis.status

    if status_val in ("queued", "processing"):
        return {
            "analysis_id": analysis_id,
            "status": status_val,
            "message": "Analysis is still in progress. Report will be available upon completion.",
            "progress": {
                "current_step": analysis.current_step,
                "completed_steps": _parse_completed_steps(analysis.completed_steps),
            },
        }

    if status_val == "failed":
        return {
            "analysis_id": analysis_id,
            "status": "failed",
            "error": analysis.error_message,
            "message": "Analysis failed. No report available.",
        }

    # Completed but no stored report (legacy analyses)
    return {
        "analysis_id": analysis_id,
        "status": status_val,
        "truth_label": analysis.truth_label,
        "truth_score": analysis.truth_score,
        "truth_confidence": analysis.truth_confidence,
        "processing_time_ms": analysis.processing_time_ms,
        "message": "Report data not available for this analysis (pre-report-generation).",
        "disclaimer": (
            "TruthLens is an AI-assisted verification tool. Its outputs represent "
            "probabilistic assessments, not definitive truth."
        ),
    }


@router.get(
    "/analysis/{analysis_id}/heatmap",
    summary="Get heatmap overlay",
    description="Returns the suspicious-region heatmap. (Stub — will be implemented in Stage 3.)",
)
async def get_analysis_heatmap(
    analysis_id: str,
    variant: str = "overlay",
    db: AsyncSession = Depends(get_db),
):
    analysis = await db.get(Analysis, analysis_id)
    if not analysis:
        raise AnalysisNotFoundError(analysis_id)

    if variant not in {"overlay", "raw"}:
        raise HTTPException(status_code=422, detail="variant must be 'overlay' or 'raw'")
    # The database lookup confirms the analysis; UUID validation prevents traversal.
    try:
        safe_id = str(UUID(analysis_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="Heatmap not found")
    filename = "overlay.png" if variant == "overlay" else "attribution.png"
    path = (settings.heatmaps_dir / safe_id / filename).resolve()
    if not path.is_relative_to(settings.heatmaps_dir.resolve()) or not path.is_file():
        raise HTTPException(status_code=404, detail="Heatmap is not available for this analysis")
    return FileResponse(path, media_type="image/png", filename=filename)


@router.get(
    "/analysis/{analysis_id}/forensics/ela",
    summary="Get Error Level Analysis visualization",
)
async def get_forensics_ela_artifact(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
):
    analysis = await db.get(Analysis, analysis_id)
    if not analysis:
        raise AnalysisNotFoundError(analysis_id)
    try:
        safe_id = str(UUID(analysis_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="ELA artifact not found")
    base = (settings.artifacts_dir / "forensics").resolve()
    path = (base / f"{safe_id}_ela.png").resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise HTTPException(status_code=404, detail="ELA artifact is not available for this analysis")
    return FileResponse(path, media_type="image/png", filename="ela.png")
