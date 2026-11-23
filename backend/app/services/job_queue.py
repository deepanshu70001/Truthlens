"""Durable job admission and single-slot analysis worker lifecycle."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import socket
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from sqlalchemy import and_, case, or_, select, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.config import settings
from backend.app.models import Analysis, AnalysisJob, AnalysisStatus, QueueCapacity, SystemEvent, WorkerLease

log = logging.getLogger("truthlens.jobs")
WORKER_LEASE_NAME = "analysis-pipeline"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _analysis_status(value) -> str:
    return value.value if hasattr(value, "value") else str(value)


async def _record_system_event(
    db: AsyncSession,
    event_type: str,
    severity: str,
    message: str,
    *,
    analysis_id: str | None = None,
    source: str = "job_queue",
    metadata: dict | None = None,
) -> None:
    """Best-effort audit logging into system_events table and structured application log."""
    try:
        event = SystemEvent(
            analysis_id=analysis_id,
            event_type=event_type,
            severity=severity,
            source=source,
            message=message,
            metadata_json=json.dumps(metadata) if metadata else None,
            created_at=utcnow(),
        )
        db.add(event)
    except Exception as exc:
        log.warning("Could not record system event %s (%s)", event_type, type(exc).__name__)
    if severity == "error":
        log.error("%s: %s (analysis_id=%s)", event_type, message, analysis_id)
    elif severity == "warning":
        log.warning("%s: %s (analysis_id=%s)", event_type, message, analysis_id)
    else:
        log.info("%s: %s (analysis_id=%s)", event_type, message, analysis_id)


async def reserve_job_capacity(db: AsyncSession) -> bool:
    result = await db.execute(
        update(QueueCapacity)
        .where(
            QueueCapacity.id == 1,
            QueueCapacity.outstanding_count < settings.ANALYSIS_QUEUE_MAX_OUTSTANDING,
        )
        .values(outstanding_count=QueueCapacity.outstanding_count + 1, updated_at=utcnow())
    )
    return result.rowcount == 1


async def release_job_capacity(db: AsyncSession) -> None:
    await db.execute(
        update(QueueCapacity)
        .where(QueueCapacity.id == 1)
        .values(
            outstanding_count=case(
                (QueueCapacity.outstanding_count > 0, QueueCapacity.outstanding_count - 1),
                else_=0,
            ),
            updated_at=utcnow(),
        )
    )


def _reset_interrupted_progress(raw: str | None) -> str | None:
    if not raw:
        return raw
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if not isinstance(value, dict):
        return raw
    statuses = value.get("stage_statuses")
    if isinstance(statuses, dict):
        value["stage_statuses"] = {
            name: "pending" if state == "running" else state
            for name, state in statuses.items()
        }
    value["completed_steps"] = [
        name for name, state in (value.get("stage_statuses") or {}).items()
        if state == "completed"
    ]
    return json.dumps(value)


@dataclass(frozen=True)
class JobClaim:
    analysis_id: str
    lease_token: str
    worker_id: str
    attempt_count: int


def _mark_analysis_failed(analysis: Analysis, category: str, safe_message: str) -> None:
    analysis.status = AnalysisStatus.FAILED
    analysis.current_step = None
    analysis.completed_at = utcnow()
    analysis.error_message = safe_message
    if not analysis.report_json:
        analysis.report_json = json.dumps({
            "analysis_id": analysis.id,
            "status": "failed",
            "error_category": category,
            "message": safe_message,
            "stages": {},
        })


async def _finish_stale_terminal_job(db: AsyncSession, job: AnalysisJob, analysis: Analysis) -> None:
    analysis_status = _analysis_status(analysis.status)
    if analysis.report_json and analysis_status in {"completed", "failed"}:
        job.state = "completed" if analysis_status == "completed" else "failed"
        job.error_category = None if job.state == "completed" else (job.error_category or "ANALYSIS_FAILED")
        job.error_message = None if job.state == "completed" else (job.error_message or "Analysis ended with a safe terminal failure.")
        job.completed_at = analysis.completed_at or utcnow()
        job.lease_token = None
        job.lease_expires_at = None
        job.claimed_by = None
        await _record_system_event(
            db, "job_recovered", "info",
            f"Stale job for analysis {job.analysis_id} resolved to terminal {job.state}.",
            analysis_id=job.analysis_id,
            metadata={"state": job.state},
        )
        await release_job_capacity(db)
        return
    if job.attempt_count >= job.max_attempts:
        category = "WORKER_LEASE_EXPIRED"
        message = "The analysis worker stopped before completing this job after the retry limit."
        job.state = "failed"
        job.error_category = category
        job.error_message = message
        job.completed_at = utcnow()
        _mark_analysis_failed(analysis, category, message)
        await _record_system_event(
            db, "job_failed", "error",
            f"Job {job.analysis_id} failed after lease expiration and exhausting {job.max_attempts} attempts.",
            analysis_id=job.analysis_id,
            metadata={"attempts": job.attempt_count, "category": category},
        )
        await release_job_capacity(db)
        return
    job.state = "queued"
    job.available_at = utcnow()
    job.lease_token = None
    job.lease_expires_at = None
    job.claimed_by = None
    job.error_category = "WORKER_RESTARTED"
    job.error_message = "An interrupted analysis was safely returned to the durable queue."
    analysis.status = AnalysisStatus.QUEUED
    analysis.current_step = None
    analysis.completed_steps = _reset_interrupted_progress(analysis.completed_steps)
    await _record_system_event(
        db, "job_recovered", "warning",
        f"Interrupted job for analysis {job.analysis_id} reset to queued for retry (attempt {job.attempt_count}/{job.max_attempts}).",
        analysis_id=job.analysis_id,
        metadata={"attempt_count": job.attempt_count, "max_attempts": job.max_attempts},
    )


async def claim_next_job(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    worker_id: str | None = None,
) -> JobClaim | None:
    """Atomically acquire the global GPU slot and claim one queued or recovered job."""
    worker_id = worker_id or f"{socket.gethostname()}:{secrets.token_hex(6)}"
    lease_token = secrets.token_hex(32)
    now = utcnow()
    async with session_factory() as db:
        lease_update = await db.execute(
            update(WorkerLease)
            .where(
                WorkerLease.name == WORKER_LEASE_NAME,
                or_(WorkerLease.owner_token.is_(None), WorkerLease.lease_expires_at <= now),
            )
            .values(
                owner_token=lease_token,
                lease_expires_at=now + timedelta(seconds=settings.ANALYSIS_JOB_LEASE_SECONDS),
                heartbeat_at=now,
            )
        )
        if lease_update.rowcount != 1:
            await db.rollback()
            return None

        await _record_system_event(
            db, "lease_acquired", "info",
            f"Worker {worker_id} acquired global lease '{WORKER_LEASE_NAME}'.",
            metadata={"worker_id": worker_id},
        )

        stale = await db.scalar(
            select(AnalysisJob)
            .where(AnalysisJob.state == "running", AnalysisJob.lease_expires_at <= now)
            .order_by(AnalysisJob.created_at)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if stale is not None:
            analysis = await db.get(Analysis, stale.analysis_id)
            if analysis is not None:
                await _finish_stale_terminal_job(db, stale, analysis)
            else:
                stale.state = "failed"
                stale.error_category = "ANALYSIS_RECORD_MISSING"
                stale.error_message = "The analysis record for this job is unavailable."
                stale.completed_at = now
                await release_job_capacity(db)

        job = await db.scalar(
            select(AnalysisJob)
            .where(
                AnalysisJob.state.in_(("queued", "retry_wait")),
                # Recovery may have just set available_at a few microseconds
                # after the initial lease-acquisition timestamp.
                AnalysisJob.available_at <= utcnow(),
            )
            .order_by(AnalysisJob.created_at, AnalysisJob.analysis_id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if job is None:
            await db.execute(
                update(WorkerLease)
                .where(WorkerLease.name == WORKER_LEASE_NAME, WorkerLease.owner_token == lease_token)
                .values(owner_token=None, lease_expires_at=None, heartbeat_at=now)
            )
            await db.commit()
            return None

        analysis = await db.get(Analysis, job.analysis_id)
        if analysis is None:
            job.state = "failed"
            job.error_category = "ANALYSIS_RECORD_MISSING"
            job.error_message = "The analysis record for this job is unavailable."
            job.completed_at = now
            await release_job_capacity(db)
            await db.execute(
                update(WorkerLease)
                .where(WorkerLease.name == WORKER_LEASE_NAME, WorkerLease.owner_token == lease_token)
                .values(owner_token=None, lease_expires_at=None, heartbeat_at=now)
            )
            await db.commit()
            return None

        # If a crash happened after the terminal report commit, acknowledge it
        # instead of invoking the pipeline again or duplicating its artifacts.
        if analysis.report_json and _analysis_status(analysis.status) in {"completed", "failed"}:
            job.state = "completed" if _analysis_status(analysis.status) == "completed" else "failed"
            job.completed_at = analysis.completed_at or now
            job.error_category = None if job.state == "completed" else "ANALYSIS_FAILED"
            job.error_message = None if job.state == "completed" else "Analysis ended with a safe terminal failure."
            await release_job_capacity(db)
            await db.execute(
                update(WorkerLease)
                .where(WorkerLease.name == WORKER_LEASE_NAME, WorkerLease.owner_token == lease_token)
                .values(owner_token=None, lease_expires_at=None, heartbeat_at=now)
            )
            await db.commit()
            return None

        job.state = "running"
        job.attempt_count += 1
        job.claimed_by = worker_id[:128]
        job.lease_token = lease_token
        job.lease_expires_at = now + timedelta(seconds=settings.ANALYSIS_JOB_LEASE_SECONDS)
        job.heartbeat_at = now
        job.started_at = job.started_at or now
        job.updated_at = now
        job.error_category = None
        job.error_message = None
        analysis.status = AnalysisStatus.PROCESSING
        analysis.current_step = None
        await _record_system_event(
            db, "job_claimed", "info",
            f"Worker {worker_id} claimed analysis {job.analysis_id} (attempt {job.attempt_count}/{job.max_attempts}).",
            analysis_id=job.analysis_id,
            metadata={"worker_id": worker_id, "attempt": job.attempt_count},
        )
        await db.commit()
        return JobClaim(job.analysis_id, lease_token, worker_id, job.attempt_count)


async def heartbeat_job(
    session_factory: async_sessionmaker[AsyncSession], claim: JobClaim
) -> bool:
    now = utcnow()
    expires = now + timedelta(seconds=settings.ANALYSIS_JOB_LEASE_SECONDS)
    async with session_factory() as db:
        job_result = await db.execute(
            update(AnalysisJob)
            .where(
                AnalysisJob.analysis_id == claim.analysis_id,
                AnalysisJob.state == "running",
                AnalysisJob.lease_token == claim.lease_token,
            )
            .values(heartbeat_at=now, lease_expires_at=expires, updated_at=now)
        )
        lease_result = await db.execute(
            update(WorkerLease)
            .where(
                WorkerLease.name == WORKER_LEASE_NAME,
                WorkerLease.owner_token == claim.lease_token,
            )
            .values(heartbeat_at=now, lease_expires_at=expires)
        )
        await db.commit()
        return job_result.rowcount == 1 and lease_result.rowcount == 1


def classify_worker_exception(exc: BaseException) -> tuple[bool, str, str]:
    """Only known transient failures are retried; all diagnostics are user-safe."""
    if isinstance(exc, (DBAPIError, TimeoutError, ConnectionError, OSError)):
        return True, "TRANSIENT_WORKER_FAILURE", "A temporary worker or storage failure interrupted this attempt."
    if isinstance(exc, asyncio.CancelledError):
        return True, "WORKER_SHUTDOWN", "The worker stopped safely and will retry this analysis."
    return False, "UNEXPECTED_WORKER_FAILURE", "The analysis worker could not complete this request."


async def record_job_failure(
    session_factory: async_sessionmaker[AsyncSession],
    claim: JobClaim,
    exc: BaseException,
) -> None:
    retryable, category, message = classify_worker_exception(exc)
    now = utcnow()
    async with session_factory() as db:
        job = await db.scalar(
            select(AnalysisJob).where(
                AnalysisJob.analysis_id == claim.analysis_id,
                AnalysisJob.state == "running",
                AnalysisJob.lease_token == claim.lease_token,
            )
        )
        analysis = await db.get(Analysis, claim.analysis_id)
        if job is None or analysis is None:
            await db.rollback()
            return
        can_retry = retryable and job.attempt_count < job.max_attempts
        if can_retry:
            delay = min(
                settings.ANALYSIS_RETRY_MAX_SECONDS,
                settings.ANALYSIS_RETRY_BASE_SECONDS * (2 ** max(0, job.attempt_count - 1)),
            )
            job.state = "retry_wait"
            job.available_at = now + timedelta(seconds=delay)
            job.error_category = category
            job.error_message = message
            job.lease_token = None
            job.lease_expires_at = None
            job.claimed_by = None
            analysis.status = AnalysisStatus.QUEUED
            analysis.current_step = None
            analysis.completed_steps = _reset_interrupted_progress(analysis.completed_steps)
            await _record_system_event(
                db, "retry_scheduled", "warning",
                f"Retry scheduled for analysis {claim.analysis_id} after {delay}s backoff: {message}",
                analysis_id=claim.analysis_id,
                metadata={"delay_seconds": delay, "category": category, "attempt": job.attempt_count},
            )
        else:
            final_category = category if retryable else category
            final_message = message if retryable else message
            job.state = "failed"
            job.error_category = final_category
            job.error_message = final_message
            job.completed_at = now
            job.lease_token = None
            job.lease_expires_at = None
            job.claimed_by = None
            _mark_analysis_failed(analysis, final_category, final_message)
            await _record_system_event(
                db, "job_failed", "error",
                f"Analysis {claim.analysis_id} permanently failed: {final_message}",
                analysis_id=claim.analysis_id,
                metadata={"category": final_category, "attempt": job.attempt_count},
            )
            await release_job_capacity(db)
        await db.execute(
            update(WorkerLease)
            .where(WorkerLease.name == WORKER_LEASE_NAME, WorkerLease.owner_token == claim.lease_token)
            .values(owner_token=None, lease_expires_at=None, heartbeat_at=now)
        )
        await db.commit()


async def complete_job(
    session_factory: async_sessionmaker[AsyncSession], claim: JobClaim
) -> None:
    now = utcnow()
    async with session_factory() as db:
        job = await db.scalar(
            select(AnalysisJob).where(
                AnalysisJob.analysis_id == claim.analysis_id,
                AnalysisJob.state == "running",
                AnalysisJob.lease_token == claim.lease_token,
            )
        )
        if job is not None:
            analysis = await db.get(Analysis, claim.analysis_id)
            if analysis is not None and _analysis_status(analysis.status) in {"completed", "failed"}:
                job.state = "completed" if _analysis_status(analysis.status) == "completed" else "failed"
                job.error_category = None if job.state == "completed" else "ANALYSIS_FAILED"
                job.error_message = None if job.state == "completed" else "Analysis ended with a safe terminal failure."
            else:
                job.state = "failed"
                job.error_category = "PIPELINE_DID_NOT_TERMINATE"
                job.error_message = "The analysis pipeline ended without a terminal report."
                if analysis is not None:
                    _mark_analysis_failed(analysis, job.error_category, job.error_message)
            job.completed_at = now
            job.lease_token = None
            job.lease_expires_at = None
            job.claimed_by = None
            evt = "job_completed" if job.state == "completed" else "job_failed"
            sev = "info" if job.state == "completed" else "error"
            await _record_system_event(
                db, evt, sev,
                f"Analysis {claim.analysis_id} reached terminal state {job.state}.",
                analysis_id=claim.analysis_id,
                metadata={"state": job.state},
            )
            await release_job_capacity(db)
        await db.execute(
            update(WorkerLease)
            .where(WorkerLease.name == WORKER_LEASE_NAME, WorkerLease.owner_token == claim.lease_token)
            .values(owner_token=None, lease_expires_at=None, heartbeat_at=now)
        )
        await db.commit()


async def process_one_job(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    run_pipeline: Callable[[str, AsyncSession], Awaitable[None]] | None = None,
    worker_id: str | None = None,
) -> bool:
    """Claim and run one job. Returns False when no work was available."""
    claim = await claim_next_job(session_factory, worker_id=worker_id)
    if claim is None:
        return False
    if run_pipeline is None:
        from backend.app.services.orchestrator import run_analysis_pipeline
        run_pipeline = run_analysis_pipeline

    stop_heartbeat = asyncio.Event()
    lost_lease = asyncio.Event()

    async def heartbeat_loop():
        while not stop_heartbeat.is_set():
            try:
                await asyncio.wait_for(
                    stop_heartbeat.wait(), timeout=settings.ANALYSIS_JOB_HEARTBEAT_SECONDS
                )
                return
            except asyncio.TimeoutError:
                try:
                    if not await heartbeat_job(session_factory, claim):
                        lost_lease.set()
                        return
                except Exception as exc:
                    log.error("Job heartbeat could not be persisted (%s)", type(exc).__name__)

    heartbeat_task = asyncio.create_task(heartbeat_loop())
    try:
        async with session_factory() as db:
            await run_pipeline(claim.analysis_id, db)
        if lost_lease.is_set():
            return True
        await complete_job(session_factory, claim)
    except BaseException as exc:
        if not isinstance(exc, asyncio.CancelledError):
            log.error("Analysis worker attempt failed (%s)", type(exc).__name__, extra={"analysis_id": claim.analysis_id})
        try:
            await record_job_failure(session_factory, claim, exc)
        except Exception as persist_exc:
            log.error("Could not persist safe job failure (%s)", type(persist_exc).__name__, extra={"analysis_id": claim.analysis_id})
        if isinstance(exc, asyncio.CancelledError):
            raise
    finally:
        stop_heartbeat.set()
        await asyncio.gather(heartbeat_task, return_exceptions=True)
    return True


async def run_worker(
    session_factory: async_sessionmaker[AsyncSession], *, stop_event: asyncio.Event
) -> None:
    """Poll the durable queue. A single global lease serializes model/GPU jobs."""
    while not stop_event.is_set():
        try:
            worked = await process_one_job(session_factory)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A temporary queue/database error must not permanently kill the
            # worker process. The next poll retries after the normal backoff.
            log.error("Worker queue poll failed (%s)", type(exc).__name__)
            worked = False
        if worked:
            continue
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=settings.ANALYSIS_JOB_POLL_SECONDS)
        except asyncio.TimeoutError:
            pass
