"""
TruthLens — API endpoint: Analysis status and results.

GET /api/v1/analysis/{analysis_id}
GET /api/v1/analysis/{analysis_id}/report
GET /api/v1/analysis/{analysis_id}/heatmap
"""

from __future__ import annotations

import json
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.exceptions import AnalysisNotFoundError
from backend.app.db.session import get_db
from backend.app.models import Analysis
from backend.app.schemas import (
    AnalysisOut,
    AnalysisProgress,
    ImageAssetOut,
    ModelRunOut,
)

router = APIRouter()


def _parse_completed_steps(raw: str | None) -> List[str]:
    """Parse completed_steps JSON string to list."""
    if not raw:
        return []
    try:
        return json.loads(raw)
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
    total_steps = 18  # Total pipeline steps (will grow with ML integration)

    progress = AnalysisProgress(
        current_step=analysis.current_step,
        completed_steps=completed_steps,
        total_steps=total_steps,
        percentage=round(len(completed_steps) / total_steps * 100, 1)
        if total_steps > 0
        else 0.0,
    )

    image_out = None
    if analysis.image_asset:
        image_out = ImageAssetOut.model_validate(analysis.image_asset)

    model_runs_out = [
        ModelRunOut.model_validate(mr) for mr in (analysis.model_runs or [])
    ]

    return AnalysisOut(
        analysis_id=analysis.id,
        status=analysis.status.value if hasattr(analysis.status, 'value') else analysis.status,
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
    db: AsyncSession = Depends(get_db),
):
    analysis = await db.get(Analysis, analysis_id)
    if not analysis:
        raise AnalysisNotFoundError(analysis_id)

    return {
        "analysis_id": analysis_id,
        "message": "Heatmap generation will be available in Stage 3.",
    }
