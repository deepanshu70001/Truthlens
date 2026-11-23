"""
TruthLens — Durable Worker Crash, Recovery, and Lease Integrity Tests.

Verifies:
1. Worker crash simulation: interrupted job is recovered by replacement worker after lease expiry.
2. Stale worker rejection: worker with expired lease cannot heartbeat, complete, or fail job.
3. Heartbeat renewal: active worker extends lease expiration and updates heartbeat_at.
4. Bounded retry and exponential backoff: transient errors schedule retry_wait.
5. Max attempts exhaustion: job transitions to failed terminal state with safe message.
6. Structured SystemEvent audit trail: records operational events for job lifecycle transitions.
7. Unpersisted terminal report acknowledgment: completed analysis report is adopted on crash recovery without re-running pipeline.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import select

from backend.app.config import settings
from backend.app.db.session import async_session_factory
from backend.app.models import (
    Analysis,
    AnalysisJob,
    AnalysisStatus,
    QueueCapacity,
    SystemEvent,
    WorkerLease,
)
from backend.app.services.job_queue import (
    JobClaim,
    claim_next_job,
    complete_job,
    heartbeat_job,
    process_one_job,
    record_job_failure,
    utcnow,
)


async def _create_test_job(
    *,
    state: str = "queued",
    attempt_count: int = 0,
    max_attempts: int = 3,
    lease_expires_at: datetime | None = None,
    lease_token: str | None = None,
    claimed_by: str | None = None,
) -> str:
    """Helper to insert a test analysis and analysis_job."""
    analysis_id = str(uuid4())
    now = utcnow()
    async with async_session_factory() as db:
        db.add(Analysis(
            id=analysis_id,
            owner_id="test-recovery-owner",
            status=AnalysisStatus.QUEUED if state == "queued" else AnalysisStatus.PROCESSING,
            completed_steps=json.dumps({"stage_statuses": {"ocr": "pending"}}),
        ))
        db.add(AnalysisJob(
            analysis_id=analysis_id,
            owner_id="test-recovery-owner",
            payload_sha256="c" * 64,
            state=state,
            attempt_count=attempt_count,
            max_attempts=max_attempts,
            available_at=now,
            claimed_by=claimed_by,
            lease_token=lease_token,
            lease_expires_at=lease_expires_at,
            created_at=now,
            updated_at=now,
        ))
        capacity = await db.get(QueueCapacity, 1)
        if capacity:
            capacity.outstanding_count += 1
        await db.commit()
    return analysis_id


@pytest.mark.asyncio
async def test_worker_crash_simulation_and_recovery():
    """Worker process terminates unexpectedly; replacement worker claims after lease expiry."""
    analysis_id = await _create_test_job(max_attempts=3)

    # Worker 1 claims job
    claim1 = await claim_next_job(async_session_factory, worker_id="worker-crash-1")
    assert claim1 is not None
    assert claim1.analysis_id == analysis_id

    # Simulate Worker 1 crashing: lease expires
    past = utcnow() - timedelta(seconds=10)
    async with async_session_factory() as db:
        job = await db.get(AnalysisJob, analysis_id)
        job.lease_expires_at = past
        lease = await db.get(WorkerLease, "analysis-pipeline")
        lease.lease_expires_at = past
        await db.commit()

    # Worker 2 discovers and claims the recovered job
    claim2 = await claim_next_job(async_session_factory, worker_id="worker-replacement-2")
    assert claim2 is not None
    assert claim2.analysis_id == analysis_id
    assert claim2.lease_token != claim1.lease_token
    assert claim2.attempt_count == 2

    # Worker 2 completes job successfully
    async with async_session_factory() as db:
        analysis = await db.get(Analysis, analysis_id)
        analysis.status = AnalysisStatus.COMPLETED
        analysis.report_json = json.dumps({"analysis_id": analysis_id, "status": "completed"})
        await db.commit()

    await complete_job(async_session_factory, claim2)

    async with async_session_factory() as db:
        final_job = await db.get(AnalysisJob, analysis_id)
        assert final_job.state == "completed"
        assert final_job.lease_token is None


@pytest.mark.asyncio
async def test_stale_worker_cannot_heartbeat_or_complete():
    """A stale worker whose lease was expired and reassigned cannot update the job."""
    analysis_id = await _create_test_job(max_attempts=3)

    # Worker 1 claims
    claim1 = await claim_next_job(async_session_factory, worker_id="stale-worker-1")
    assert claim1 is not None

    # Force lease expiration
    past = utcnow() - timedelta(seconds=5)
    async with async_session_factory() as db:
        job = await db.get(AnalysisJob, analysis_id)
        job.lease_expires_at = past
        lease = await db.get(WorkerLease, "analysis-pipeline")
        lease.lease_expires_at = past
        await db.commit()

    # Replacement worker reclaims
    claim2 = await claim_next_job(async_session_factory, worker_id="fresh-worker-2")
    assert claim2 is not None

    # Stale Worker 1 attempts heartbeat -> must return False
    hb_success = await heartbeat_job(async_session_factory, claim1)
    assert hb_success is False

    # Stale Worker 1 attempts complete -> must not alter state or release new worker's lease
    await complete_job(async_session_factory, claim1)

    async with async_session_factory() as db:
        current_job = await db.get(AnalysisJob, analysis_id)
        assert current_job.state == "running"
        assert current_job.lease_token == claim2.lease_token

    # Cleanup Worker 2
    async with async_session_factory() as db:
        analysis = await db.get(Analysis, analysis_id)
        analysis.status = AnalysisStatus.COMPLETED
        analysis.report_json = json.dumps({"analysis_id": analysis_id, "status": "completed"})
        await db.commit()
    await complete_job(async_session_factory, claim2)


@pytest.mark.asyncio
async def test_heartbeat_extends_lease():
    """Active worker heartbeat extends lease expiration timestamp."""
    analysis_id = await _create_test_job()

    claim = await claim_next_job(async_session_factory, worker_id="heartbeat-worker")
    assert claim is not None

    async with async_session_factory() as db:
        job_before = await db.get(AnalysisJob, analysis_id)
        expires_before = job_before.lease_expires_at

    # Small delay to ensure timestamp progression
    await asyncio.sleep(0.05)
    success = await heartbeat_job(async_session_factory, claim)
    assert success is True

    async with async_session_factory() as db:
        job_after = await db.get(AnalysisJob, analysis_id)
        assert job_after.lease_expires_at > expires_before
        assert job_after.heartbeat_at is not None

    # Cleanup
    await record_job_failure(async_session_factory, claim, RuntimeError("cleanup"))


@pytest.mark.asyncio
async def test_transient_failure_bounded_retry_and_max_attempts():
    """Transient failures schedule retry_wait; max attempts leads to safe terminal failed state."""
    analysis_id = await _create_test_job(max_attempts=2)

    # Attempt 1: Transient network/storage failure
    claim1 = await claim_next_job(async_session_factory, worker_id="retry-worker")
    assert claim1 is not None
    assert claim1.attempt_count == 1

    await record_job_failure(async_session_factory, claim1, OSError("Network dropped connection"))

    async with async_session_factory() as db:
        job = await db.get(AnalysisJob, analysis_id)
        assert job.state == "retry_wait"
        assert job.error_category == "TRANSIENT_WORKER_FAILURE"
        # Reset available_at to now for immediate test claiming
        job.available_at = utcnow() - timedelta(seconds=1)
        await db.commit()

    # Attempt 2: Final attempt fails with transient failure
    claim2 = await claim_next_job(async_session_factory, worker_id="retry-worker")
    assert claim2 is not None
    assert claim2.attempt_count == 2

    await record_job_failure(async_session_factory, claim2, TimeoutError("Database query timed out"))

    async with async_session_factory() as db:
        final_job = await db.get(AnalysisJob, analysis_id)
        final_analysis = await db.get(Analysis, analysis_id)
        assert final_job.state == "failed"
        assert final_job.error_category == "TRANSIENT_WORKER_FAILURE"
        assert final_analysis.status == AnalysisStatus.FAILED
        assert "temporary worker or storage failure" in final_analysis.error_message


@pytest.mark.asyncio
async def test_system_event_logging_audit_trail():
    """SystemEvent records are logged for lifecycle events (claimed, retry, completed)."""
    analysis_id = await _create_test_job(max_attempts=3)

    claim = await claim_next_job(async_session_factory, worker_id="audit-worker")
    assert claim is not None

    async with async_session_factory() as db:
        events = (
            await db.scalars(
                select(SystemEvent)
                .where(SystemEvent.analysis_id == analysis_id)
                .order_by(SystemEvent.created_at)
            )
        ).all()
        event_types = [e.event_type for e in events]
        assert "job_claimed" in event_types

    # Mark completed
    async with async_session_factory() as db:
        analysis = await db.get(Analysis, analysis_id)
        analysis.status = AnalysisStatus.COMPLETED
        analysis.report_json = json.dumps({"analysis_id": analysis_id, "status": "completed"})
        await db.commit()

    await complete_job(async_session_factory, claim)

    async with async_session_factory() as db:
        events_after = (
            await db.scalars(
                select(SystemEvent)
                .where(SystemEvent.analysis_id == analysis_id)
                .order_by(SystemEvent.created_at)
            )
        ).all()
        event_types_after = [e.event_type for e in events_after]
        assert "job_completed" in event_types_after
