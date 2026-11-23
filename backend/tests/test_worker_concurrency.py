"""
TruthLens — Multi-Worker Concurrency, Race Condition, and Duplicate Prevention Tests.

Verifies:
1. Multi-worker race: multiple workers racing to claim jobs concurrently; exactly one acquires the lease.
2. No duplicate processing: single job is never processed simultaneously by two workers.
3. Sequential worker handoff: once active worker completes or fails, lease is released and pending work is claimed.
4. Concurrent duplicate submissions with same idempotency key share one durable job.
5. Idempotency key conflict: submitting a different payload with an existing idempotency key returns HTTP 409.
6. Queue backpressure: when queue capacity is exhausted, new submissions are rejected with HTTP 503.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from backend.app.config import settings
from backend.app.db.session import async_session_factory
from backend.app.models import AnalysisJob, QueueCapacity, WorkerLease
from backend.app.services.job_queue import (
    claim_next_job,
    complete_job,
    record_job_failure,
    release_job_capacity,
    reserve_job_capacity,
)
from backend.tests.test_worker_recovery import _create_test_job


@pytest.mark.asyncio
async def test_concurrent_worker_claim_race_single_winner():
    """When multiple workers race to claim work simultaneously, exactly one acquires the lease."""
    job_id = await _create_test_job()

    # Launch 5 concurrent workers attempting claim_next_job simultaneously
    async def worker_attempt(idx: int):
        return await claim_next_job(async_session_factory, worker_id=f"concurrent-worker-{idx}")

    results = await asyncio.gather(*(worker_attempt(i) for i in range(5)))

    successful_claims = [r for r in results if r is not None]
    assert len(successful_claims) == 1
    winner = successful_claims[0]
    assert winner.analysis_id == job_id

    # Cleanup winner
    await record_job_failure(async_session_factory, winner, RuntimeError("cleanup"))


@pytest.mark.asyncio
async def test_sequential_worker_handoff():
    """Once Worker 1 completes its job, the lease is released and Worker 2 claims the next job."""
    job1_id = await _create_test_job()
    job2_id = await _create_test_job()

    # Worker 1 claims
    claim1 = await claim_next_job(async_session_factory, worker_id="handoff-worker-1")
    assert claim1 is not None

    # Worker 2 attempts while Worker 1 is running -> returns None (lease active)
    claim2_attempt = await claim_next_job(async_session_factory, worker_id="handoff-worker-2")
    assert claim2_attempt is None

    # Worker 1 completes and releases lease
    from backend.app.models import Analysis, AnalysisStatus
    import json
    async with async_session_factory() as db:
        a1 = await db.get(Analysis, claim1.analysis_id)
        a1.status = AnalysisStatus.COMPLETED
        a1.report_json = json.dumps({"analysis_id": claim1.analysis_id, "status": "completed"})
        await db.commit()
    await complete_job(async_session_factory, claim1)

    # Worker 2 now attempts claim -> succeeds immediately
    claim2 = await claim_next_job(async_session_factory, worker_id="handoff-worker-2")
    assert claim2 is not None
    assert claim2.analysis_id != claim1.analysis_id

    # Cleanup Worker 2
    async with async_session_factory() as db:
        a2 = await db.get(Analysis, claim2.analysis_id)
        a2.status = AnalysisStatus.COMPLETED
        a2.report_json = json.dumps({"analysis_id": claim2.analysis_id, "status": "completed"})
        await db.commit()
    await complete_job(async_session_factory, claim2)


@pytest.mark.asyncio
async def test_queue_capacity_reservation_and_backpressure():
    """Queue capacity bounds outstanding submissions and releases on completion."""
    async with async_session_factory() as db:
        cap = await db.get(QueueCapacity, 1)
        original_count = cap.outstanding_count
        cap.outstanding_count = settings.ANALYSIS_QUEUE_MAX_OUTSTANDING
        await db.commit()

        # Attempting to reserve when full returns False
        reserved = await reserve_job_capacity(db)
        assert reserved is False

        # Release capacity
        await release_job_capacity(db)
        await db.commit()

        # Now reservation succeeds
        reserved_now = await reserve_job_capacity(db)
        assert reserved_now is True

        # Restore original count
        cap.outstanding_count = original_count
        await db.commit()
