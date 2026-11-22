"""
TruthLens — API endpoint: Analysis status and results.

GET /api/v1/analysis/{analysis_id}
GET /api/v1/analysis/{analysis_id}/report
GET /api/v1/analysis/{analysis_id}/heatmap
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from sqlalchemy import delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.config import settings
from backend.app.core.auth import Principal, get_current_principal
from backend.app.db.session import get_db
from backend.app.core.ownership import require_analysis_owner
from backend.app.core.public_output import safe_public_text, sanitize_public_payload
from backend.app.core.rate_limit import check_rate_limit
from backend.app.models import Analysis, AnalysisJob, ImageAsset, ModelRun, SystemEvent
from backend.app.services.job_queue import release_job_capacity
from backend.app.schemas import (
    AnalysisOut,
    AnalysisProgress,
    ImageAssetOut,
    JobProgress,
    ModelRunOut,
)

router = APIRouter()


@router.get(
    "/analysis/{analysis_id}/image",
    summary="Retrieve uploaded analysis image",
    description="Serves the validated original image for this analysis without exposing its storage path.",
)
async def get_analysis_image(
    analysis_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    principal: Principal = Depends(get_current_principal),
):
    await check_rate_limit(request, scope="analysis-retrieval", limit=settings.RATE_LIMIT_RETRIEVAL_PER_MINUTE, principal=principal)
    analysis = await require_analysis_owner(analysis_id, principal, db)
    if not analysis.image_asset:
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
    request: Request,
    db: AsyncSession = Depends(get_db),
    principal: Principal = Depends(get_current_principal),
):
    await check_rate_limit(request, scope="analysis-retrieval", limit=settings.RATE_LIMIT_RETRIEVAL_PER_MINUTE, principal=principal)
    analysis = await require_analysis_owner(analysis_id, principal, db)

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

    model_runs_out = []
    for model_run in analysis.model_runs or []:
        public_run = ModelRunOut.model_validate(model_run)
        if public_run.error_message:
            public_run.error_message = safe_public_text(public_run.error_message)
        model_runs_out.append(public_run)

    job = await db.get(AnalysisJob, analysis_id)
    job_out = JobProgress(
        job_id=job.analysis_id,
        state=job.state,
        attempt_count=job.attempt_count,
        max_attempts=job.max_attempts,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_category=job.error_category,
        error_message=safe_public_text(job.error_message) if job.error_message else None,
    ) if job else None

    return AnalysisOut(
        analysis_id=analysis.id,
        job=job_out,
        status=response_status,
        progress=progress,
        truth_score=analysis.truth_score,
        truth_confidence=analysis.truth_confidence,
        truth_label=analysis.truth_label,
        image=image_out,
        model_runs=model_runs_out,
        processing_time_ms=analysis.processing_time_ms,
        error_message=safe_public_text(analysis.error_message) if analysis.error_message else None,
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
    request: Request,
    db: AsyncSession = Depends(get_db),
    principal: Principal = Depends(get_current_principal),
):
    await check_rate_limit(request, scope="analysis-retrieval", limit=settings.RATE_LIMIT_RETRIEVAL_PER_MINUTE, principal=principal)
    analysis = await require_analysis_owner(analysis_id, principal, db)

    # If report is stored, return it directly
    if analysis.report_json:
        try:
            return sanitize_public_payload(json.loads(analysis.report_json))
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
            "error": safe_public_text(analysis.error_message) if analysis.error_message else "Analysis failed.",
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
    request: Request,
    variant: str = "overlay",
    db: AsyncSession = Depends(get_db),
    principal: Principal = Depends(get_current_principal),
):
    await check_rate_limit(request, scope="analysis-retrieval", limit=settings.RATE_LIMIT_RETRIEVAL_PER_MINUTE, principal=principal)
    analysis = await require_analysis_owner(analysis_id, principal, db)

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
    request: Request,
    db: AsyncSession = Depends(get_db),
    principal: Principal = Depends(get_current_principal),
):
    await check_rate_limit(request, scope="analysis-retrieval", limit=settings.RATE_LIMIT_RETRIEVAL_PER_MINUTE, principal=principal)
    await require_analysis_owner(analysis_id, principal, db)
    try:
        safe_id = str(UUID(analysis_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="ELA artifact not found")
    base = (settings.artifacts_dir / "forensics").resolve()
    path = (base / f"{safe_id}_ela.png").resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise HTTPException(status_code=404, detail="ELA artifact is not available for this analysis")
    return FileResponse(path, media_type="image/png", filename="ela.png")


@router.get("/analysis")
async def list_analyses(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0, le=100_000),
    db: AsyncSession = Depends(get_db),
    principal: Principal = Depends(get_current_principal),
):
    await check_rate_limit(request, scope="analysis-retrieval", limit=settings.RATE_LIMIT_RETRIEVAL_PER_MINUTE, principal=principal)
    query = select(Analysis).options(selectinload(Analysis.image_asset)).order_by(Analysis.created_at.desc()).limit(limit).offset(offset)
    if principal.development_bypass:
        query = query.where(or_(Analysis.owner_id == principal.subject, Analysis.owner_id.is_(None)))
    else:
        query = query.where(Analysis.owner_id == principal.subject)
    rows = (await db.scalars(query)).all()
    return [{
        "analysis_id": row.id,
        "status": _response_status(row),
        "created_at": row.created_at,
        "completed_at": row.completed_at,
        "original_filename": row.image_asset.original_filename if row.image_asset else None,
    } for row in rows]


def _response_status(analysis: Analysis) -> str:
    value = analysis.status.value if hasattr(analysis.status, "value") else analysis.status
    if value == "completed" and analysis.report_json:
        try:
            if json.loads(analysis.report_json).get("status") == "partial":
                return "partial"
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
    return value


def _analysis_storage_targets(analysis_id: str, image_asset: ImageAsset | None) -> list[Path]:
    """Build deletion targets only from known per-analysis storage roots/names."""
    try:
        safe_id = str(UUID(analysis_id))
    except ValueError:
        return []
    roots = [settings.originals_dir, settings.thumbnails_dir, settings.processed_dir,
             settings.heatmaps_dir, settings.artifacts_dir / "forensics"]
    targets: set[Path] = set()
    if image_asset is not None:
        for raw in (image_asset.storage_path, image_asset.thumbnail_path):
            if raw:
                path = Path(raw).resolve()
                if any(path.is_relative_to(root.resolve()) for root in roots):
                    targets.add(path)
    for root in roots:
        resolved_root = root.resolve()
        if not resolved_root.exists():
            continue
        if root == settings.heatmaps_dir:
            candidate = (root / safe_id).resolve()
            if candidate.is_relative_to(resolved_root) and candidate.exists():
                targets.add(candidate)
        else:
            for candidate in root.glob(f"{safe_id}*"):
                resolved = candidate.resolve()
                if resolved.is_relative_to(resolved_root):
                    targets.add(resolved)
    return list(targets)


@router.delete("/analysis/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_analysis(
    analysis_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    principal: Principal = Depends(get_current_principal),
):
    await check_rate_limit(request, scope="analysis-delete", limit=10, principal=principal)
    analysis = await require_analysis_owner(analysis_id, principal, db)
    job = await db.get(AnalysisJob, analysis_id)
    if job is not None and job.state == "running":
        raise HTTPException(status_code=409, detail="A running analysis cannot be deleted.")
    if job is not None and job.state in {"queued", "retry_wait"}:
        await release_job_capacity(db)
    targets = _analysis_storage_targets(analysis_id, analysis.image_asset)
    # Delete dependent rows explicitly; the legacy schema has no ON DELETE CASCADE.
    await db.execute(delete(AnalysisJob).where(AnalysisJob.analysis_id == analysis_id))
    await db.execute(delete(ModelRun).where(ModelRun.analysis_id == analysis_id))
    await db.execute(delete(SystemEvent).where(SystemEvent.analysis_id == analysis_id))
    await db.execute(delete(ImageAsset).where(ImageAsset.analysis_id == analysis_id))
    await db.delete(analysis)
    await db.commit()
    if targets:
        def remove_owned_targets():
            for path in targets:
                try:
                    if path.is_dir():
                        shutil.rmtree(path)
                    elif path.is_file():
                        path.unlink()
                except OSError:
                    # DB deletion is authoritative. Orphan reporting is dry-run only.
                    pass
        await asyncio.to_thread(remove_owned_targets)
    return None
