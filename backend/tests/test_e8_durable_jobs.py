"""Deterministic durable-job tests; all pipeline behavior here is mocked."""

from __future__ import annotations

import json
import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy.exc import OperationalError
from backend.app.config import settings
from backend.app.db.session import async_session_factory
from backend.app.models import (
    Analysis,
    AnalysisJob,
    AnalysisStatus,
    QueueCapacity,
    WorkerLease,
)
from backend.app.services.job_queue import (
    claim_next_job,
    process_one_job,
    utcnow,
)


async def _new_job(*, max_attempts: int = 3) -> str:
    analysis_id = str(uuid4())
    now = utcnow()
    async with async_session_factory() as db:
        db.add(Analysis(
            id=analysis_id,
            owner_id="job-test-user",
            status=AnalysisStatus.QUEUED,
            completed_steps=json.dumps({"stage_statuses": {"ocr": "pending"}}),
        ))
        db.add(AnalysisJob(
            analysis_id=analysis_id,
            owner_id="job-test-user",
            payload_sha256="a" * 64,
            state="queued",
            attempt_count=0,
            max_attempts=max_attempts,
            available_at=now,
            created_at=now,
            updated_at=now,
        ))
        capacity = await db.get(QueueCapacity, 1)
        capacity.outstanding_count += 1
        await db.commit()
    return analysis_id


async def _records(analysis_id: str):
    async with async_session_factory() as db:
        return await db.get(Analysis, analysis_id), await db.get(AnalysisJob, analysis_id)


@pytest.mark.asyncio
async def test_durable_submission_persists_before_worker_and_idempotency(client_without_worker):
    from backend.tests.conftest import create_test_jpeg

    key = f"e8-{uuid4()}"
    image = create_test_jpeg()
    first = await client_without_worker.post(
        "/api/v1/analyze",
        files={"image": ("durable.jpg", image, "image/jpeg")},
        headers={"Idempotency-Key": key},
    )
    assert first.status_code == 201
    payload = first.json()
    assert payload["job_id"] == payload["analysis_id"]
    analysis_id = payload["analysis_id"]

    analysis, job = await _records(analysis_id)
    assert analysis.status == AnalysisStatus.QUEUED
    assert job.state == "queued"
    assert job.attempt_count == 0

    duplicate = await client_without_worker.post(
        "/api/v1/analyze",
        files={"image": ("durable.jpg", image, "image/jpeg")},
        headers={"Idempotency-Key": key},
    )
    assert duplicate.status_code == 201
    assert duplicate.json()["analysis_id"] == analysis_id

    changed_payload = await client_without_worker.post(
        "/api/v1/analyze",
        files={"image": ("durable.jpg", create_test_jpeg(101, 100), "image/jpeg")},
        headers={"Idempotency-Key": key},
    )
    assert changed_payload.status_code == 409
    deleted = await client_without_worker.delete(f"/api/v1/analysis/{analysis_id}")
    assert deleted.status_code == 204


@pytest.mark.asyncio
async def test_concurrent_duplicate_submissions_share_one_durable_job(client_without_worker):
    from backend.app.models import AnalysisJob
    from backend.tests.conftest import create_test_jpeg
    from sqlalchemy import select, func

    key = f"concurrent-{uuid4()}"
    image = create_test_jpeg()

    async def submit():
        return await client_without_worker.post(
            "/api/v1/analyze",
            files={"image": ("same.jpg", image, "image/jpeg")},
            headers={"Idempotency-Key": key},
        )

    responses = await asyncio.gather(submit(), submit())
    assert [response.status_code for response in responses] == [201, 201]
    analysis_ids = {response.json()["analysis_id"] for response in responses}
    assert len(analysis_ids) == 1
    analysis_id = analysis_ids.pop()
    async with async_session_factory() as db:
        jobs = await db.scalar(
            select(func.count()).select_from(AnalysisJob).where(
                AnalysisJob.owner_id == "local-development",
                AnalysisJob.idempotency_key == key,
            )
        )
    assert jobs == 1
    deleted = await client_without_worker.delete(f"/api/v1/analysis/{analysis_id}")
    assert deleted.status_code == 204


@pytest.mark.asyncio
async def test_submission_returns_persisted_progress_and_backpressure(client_without_worker, monkeypatch):
    from backend.app.config import settings
    from backend.tests.conftest import create_test_jpeg

    response = await client_without_worker.post(
        "/api/v1/analyze", files={"image": ("queued.jpg", create_test_jpeg(), "image/jpeg")}
    )
    assert response.status_code == 201
    analysis_id = response.json()["analysis_id"]
    status_response = await client_without_worker.get(f"/api/v1/analysis/{analysis_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "queued"
    assert status_response.json()["job"]["state"] == "queued"

    monkeypatch.setattr(settings, "ANALYSIS_QUEUE_MAX_OUTSTANDING", 0)
    rejected = await client_without_worker.post(
        "/api/v1/analyze", files={"image": ("busy.jpg", create_test_jpeg(), "image/jpeg")}
    )
    assert rejected.status_code == 429
    assert rejected.headers["Retry-After"] == "5"
    deleted = await client_without_worker.delete(f"/api/v1/analysis/{analysis_id}")
    assert deleted.status_code == 204


@pytest.mark.asyncio
async def test_success_and_partial_report_finish_delivery_without_losing_partial(client_without_worker):
    analysis_id = await _new_job()

    async def partial_pipeline(target_id, db):
        analysis = await db.get(Analysis, target_id)
        analysis.status = AnalysisStatus.COMPLETED
        analysis.report_json = json.dumps({
            "analysis_id": target_id,
            "status": "partial",
            "stages": {"ocr": {"status": "completed"}, "provenance": {"status": "unavailable"}},
        })
        analysis.completed_steps = json.dumps({"stage_statuses": {"ocr": "completed"}})
        analysis.completed_at = utcnow()
        await db.commit()

    assert await process_one_job(async_session_factory, run_pipeline=partial_pipeline)
    analysis, job = await _records(analysis_id)
    assert analysis.status == AnalysisStatus.COMPLETED
    assert json.loads(analysis.report_json)["status"] == "partial"
    assert job.state == "completed"

    # The terminal report and job state remain durable after the worker call.
    assert job.completed_at is not None


@pytest.mark.asyncio
async def test_unexpected_failure_is_terminal_and_user_safe():
    analysis_id = await _new_job()

    async def bad_pipeline(_target_id, _db):
        raise ValueError("secret token=abc C:\\private\\trace.py")

    assert await process_one_job(async_session_factory, run_pipeline=bad_pipeline)
    analysis, job = await _records(analysis_id)
    assert analysis.status == AnalysisStatus.FAILED
    assert job.state == "failed"
    assert job.error_category == "UNEXPECTED_WORKER_FAILURE"
    assert "abc" not in job.error_message
    assert "trace.py" not in job.error_message
    assert "Traceback" not in (analysis.report_json or "")


@pytest.mark.asyncio
async def test_retryable_failure_is_bounded_and_then_succeeds(monkeypatch):
    analysis_id = await _new_job(max_attempts=2)
    monkeypatch.setattr(settings, "ANALYSIS_RETRY_BASE_SECONDS", 0)
    calls = 0

    async def transient_once(target_id, db):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("provider timeout")
        analysis = await db.get(Analysis, target_id)
        analysis.status = AnalysisStatus.COMPLETED
        analysis.report_json = json.dumps({"analysis_id": target_id, "status": "completed"})
        analysis.completed_at = utcnow()
        await db.commit()

    assert await process_one_job(async_session_factory, run_pipeline=transient_once)
    _analysis, waiting_job = await _records(analysis_id)
    assert waiting_job.state == "retry_wait"
    assert waiting_job.attempt_count == 1

    assert await process_one_job(async_session_factory, run_pipeline=transient_once)
    analysis, finished_job = await _records(analysis_id)
    assert calls == 2
    assert analysis.status == AnalysisStatus.COMPLETED
    assert finished_job.state == "completed"
    assert finished_job.attempt_count == 2


@pytest.mark.asyncio
async def test_unpersisted_terminal_failure_reaches_worker_retry_policy(monkeypatch):
    from backend.app.services import orchestrator

    analysis_id = await _new_job(max_attempts=2)
    monkeypatch.setattr(settings, "ANALYSIS_RETRY_BASE_SECONDS", 0)
    db_error = OperationalError("COMMIT", {}, RuntimeError("database unavailable"))

    async def failed_persistence(*_args, **_kwargs):
        return db_error

    monkeypatch.setattr(orchestrator, "_persist_terminal_failure", failed_persistence)

    assert await process_one_job(
        async_session_factory,
        run_pipeline=orchestrator.run_analysis_pipeline,
    )
    analysis, job = await _records(analysis_id)
    assert analysis.status == AnalysisStatus.QUEUED
    assert job.state == "retry_wait"
    assert job.attempt_count == 1
    assert job.error_category == "TRANSIENT_WORKER_FAILURE"

    assert await process_one_job(
        async_session_factory,
        run_pipeline=orchestrator.run_analysis_pipeline,
    )
    analysis, job = await _records(analysis_id)
    assert analysis.status == AnalysisStatus.FAILED
    assert job.state == "failed"
    assert job.attempt_count == 2


@pytest.mark.asyncio
async def test_expired_lease_recovers_job_after_worker_restart():
    analysis_id = await _new_job(max_attempts=3)
    expired = utcnow() - timedelta(minutes=2)
    async with async_session_factory() as db:
        job = await db.get(AnalysisJob, analysis_id)
        job.state = "running"
        job.attempt_count = 1
        job.lease_token = "old-job-token"
        job.lease_expires_at = expired
        job.claimed_by = "crashed-worker"
        analysis = await db.get(Analysis, analysis_id)
        analysis.status = AnalysisStatus.PROCESSING
        lease = await db.get(WorkerLease, "analysis-pipeline")
        lease.owner_token = "old-job-token"
        lease.lease_expires_at = expired
        await db.commit()
    claim = await claim_next_job(async_session_factory, worker_id="recovery-worker")
    assert claim is not None
    assert claim.analysis_id == analysis_id
    assert claim.attempt_count == 2
    analysis, job = await _records(analysis_id)
    assert analysis.status == AnalysisStatus.PROCESSING
    assert job.state == "running"
    assert job.claimed_by == "recovery-worker"

    # Release the test lease through the normal failure path and keep queue
    # accounting consistent for the remaining suite.
    from backend.app.services.job_queue import record_job_failure
    await record_job_failure(async_session_factory, claim, ValueError("test cleanup"))


@pytest.mark.asyncio
async def test_terminal_report_after_crash_is_acknowledged_without_reexecution():
    analysis_id = await _new_job()
    expired = utcnow() - timedelta(minutes=2)
    async with async_session_factory() as db:
        job = await db.get(AnalysisJob, analysis_id)
        job.state = "running"
        job.attempt_count = 1
        job.lease_token = "expired-token"
        job.lease_expires_at = expired
        analysis = await db.get(Analysis, analysis_id)
        analysis.status = AnalysisStatus.COMPLETED
        analysis.report_json = json.dumps({"analysis_id": analysis_id, "status": "partial"})
        analysis.completed_at = utcnow()
        lease = await db.get(WorkerLease, "analysis-pipeline")
        lease.owner_token = "expired-token"
        lease.lease_expires_at = expired
        await db.commit()

    claim = await claim_next_job(async_session_factory, worker_id="replacement-worker")
    assert claim is None
    analysis, job = await _records(analysis_id)
    assert analysis.status == AnalysisStatus.COMPLETED
    assert job.state == "completed"
    assert json.loads(analysis.report_json)["status"] == "partial"


@pytest.mark.asyncio
async def test_global_worker_lease_serializes_model_jobs():
    first_id = await _new_job()
    second_id = await _new_job()
    first_claim = await claim_next_job(async_session_factory, worker_id="worker-one")
    second_claim = await claim_next_job(async_session_factory, worker_id="worker-two")
    assert first_claim is not None
    assert first_claim.analysis_id in {first_id, second_id}
    assert second_claim is None

    from backend.app.services.job_queue import record_job_failure
    await record_job_failure(async_session_factory, first_claim, ValueError("test cleanup"))
    # The second queued job remains available, and one worker can claim it.
    final_claim = await claim_next_job(async_session_factory, worker_id="worker-three")
    assert final_claim is not None
    assert final_claim.analysis_id != first_claim.analysis_id
    await record_job_failure(async_session_factory, final_claim, ValueError("test cleanup"))

