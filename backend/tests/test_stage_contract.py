"""Failure-injection tests for the shared stage execution contract."""

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.app.services.evidence.schema import EvidenceCollection, EvidenceItem
from backend.app.services.fusion.engine import fuse_evidence
from backend.app.services.orchestrator import (
    PIPELINE_STAGES,
    StageResult,
    _persist_terminal_failure,
    _record_skipped_stage,
    _run_stage,
    run_analysis_pipeline,
)


class FakeDB:
    def __init__(self, analysis=None, fail_commit_numbers=()):
        self.analysis = analysis
        self.commits = 0
        self.rollbacks = 0
        self.fail_commit_numbers = set(fail_commit_numbers)
        self.events = []

    async def commit(self):
        self.commits += 1
        if self.commits in self.fail_commit_numbers:
            raise RuntimeError("database commit failed")

    async def rollback(self):
        self.rollbacks += 1

    def add(self, value):
        self.events.append(value)

    async def get(self, _model, _analysis_id):
        return self.analysis


def analysis_row(analysis_id="stage-test"):
    return SimpleNamespace(
        id=analysis_id,
        current_step=None,
        completed_steps="[]",
        status="processing",
        error_message=None,
        report_json=None,
        completed_at=None,
        processing_time_ms=None,
        created_at=datetime.now(timezone.utc),
        image_asset=None,
        system_events=[],
    )


@pytest.mark.asyncio
async def test_successful_stage_has_canonical_status_identity_and_progress():
    row, db, results = analysis_row(), FakeDB(), {}
    result = await _run_stage(
        "metadata", lambda: SimpleNamespace(status="success", model_name="metadata-v1", warnings=[]),
        row, db, results,
    )
    stage = results["metadata"]
    assert result is not None
    assert stage.status == "completed" and stage.outcome == "success"
    assert stage.stage_id and stage.correlation_id == row.id
    assert stage.started_at and stage.completed_at and stage.runtime_ms >= 0
    assert stage.module_identity == {"model_name": "metadata-v1"}
    progress = json.loads(row.completed_steps)
    assert progress["completed_steps"] == ["metadata"]
    assert progress["stage_statuses"]["metadata"] == "completed"
    assert set(progress["stage_statuses"]) == set(PIPELINE_STAGES)


@pytest.mark.asyncio
async def test_no_results_is_successful_execution_not_failure():
    row, results = analysis_row(), {}
    await _run_stage("fact_check", lambda: SimpleNamespace(status="no_results"), row, FakeDB(), results)
    assert results["fact_check"].status == "completed"
    assert results["fact_check"].outcome == "no_results"


@pytest.mark.asyncio
async def test_ocr_without_text_is_completed_with_no_text_outcome():
    row, results = analysis_row(), {}
    await _run_stage("ocr", lambda: SimpleNamespace(status="completed", has_text=False), row, FakeDB(), results)
    assert results["ocr"].status == "completed"
    assert results["ocr"].outcome == "no_text"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["unavailable", "failed", "partial"])
async def test_module_reported_degraded_statuses_are_preserved(status):
    row, results = analysis_row(), {}
    await _run_stage("optional", lambda: SimpleNamespace(status=status), row, FakeDB(), results)
    assert results["optional"].status == status


@pytest.mark.asyncio
async def test_optional_stage_exception_is_safe_and_terminal():
    row, results = analysis_row(), {}

    def fail():
        raise TimeoutError("https://provider.invalid?api_key=secret")

    await _run_stage("provider", fail, row, FakeDB(), results)
    stage = results["provider"]
    assert stage.status == "failed"
    assert stage.error_category == "STAGE_EXECUTION_ERROR"
    assert stage.error_message == "The stage could not complete."
    assert "secret" not in json.dumps(stage.to_dict())
    assert "provider.invalid" not in json.dumps(stage.to_dict())


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [None, SimpleNamespace(status="running")])
async def test_malformed_module_result_fails_without_becoming_success(result):
    row, results = analysis_row(), {}
    returned = await _run_stage("broken", lambda: result, row, FakeDB(), results)
    assert returned is None
    assert results["broken"].status == "failed"
    assert results["broken"].error_category == "INVALID_MODULE_RESULT"


def test_disabled_stage_is_skipped_without_error_category():
    results = {}
    _record_skipped_stage(results, "xai", "Authenticity model is unavailable.", "analysis-1")
    assert results["xai"].status == "skipped"
    assert results["xai"].outcome == "not_applicable"
    assert results["xai"].error_category is None
    assert results["xai"].started_at == results["xai"].completed_at


@pytest.mark.asyncio
async def test_stage_commit_failure_rolls_back_and_records_failed_stage():
    row, results = analysis_row(), {}
    db = FakeDB(fail_commit_numbers={2})
    await _run_stage("quality", lambda: SimpleNamespace(status="completed"), row, db, results)
    assert db.rollbacks == 1
    assert results["quality"].status == "failed"
    assert results["quality"].error_category == "STAGE_EXECUTION_ERROR"


@pytest.mark.asyncio
async def test_report_persistence_failure_uses_safe_fallback_report(monkeypatch):
    import backend.app.services.reporting.builder as report_builder

    def fail_report(**_kwargs):
        raise RuntimeError("report persistence failure")

    monkeypatch.setattr(report_builder, "build_report", fail_report)
    row = analysis_row("report-failure")
    db = FakeDB(analysis=row)
    await _persist_terminal_failure(row.id, db, {})
    assert row.status == "failed"
    report = json.loads(row.report_json)
    assert report["status"] == "failed"
    assert report["stages"]["pipeline"]["error_message"]


@pytest.mark.asyncio
async def test_terminal_report_commit_failure_retries_after_rollback():
    row = analysis_row("terminal-commit-retry")
    db = FakeDB(analysis=row, fail_commit_numbers={1})
    await _persist_terminal_failure(row.id, db, {})
    assert db.rollbacks == 2  # initial cleanup plus rollback after the injected commit error
    assert db.commits == 2
    assert row.status == "failed"
    assert json.loads(row.report_json)["status"] == "failed"


@pytest.mark.asyncio
async def test_unexpected_orchestrator_exception_persists_failed_terminal_state():
    row = analysis_row("unexpected")
    row.image_asset = SimpleNamespace(storage_path=str(Path.cwd()), format="PNG")
    db = FakeDB(analysis=row)
    await run_analysis_pipeline(row.id, db)
    assert row.status.value == "failed"
    assert row.completed_at is not None
    assert json.loads(row.report_json)["status"] == "failed"
    progress = json.loads(row.completed_steps)
    assert all(status in {"completed", "partial", "failed", "unavailable", "skipped"}
               for status in progress["stage_statuses"].values())


def test_failed_directional_evidence_does_not_vote_in_fusion():
    collection = EvidenceCollection(items=[EvidenceItem(
        type="model_evidence", source="failed-detector", strength="high",
        direction="authentic", description="Must not vote.", calibrated=True,
        confidence=0.99, status="failed",
    )])
    assessment = fuse_evidence(collection)
    assert assessment.label == "inconclusive"
    assert assessment.overall_score is None
    assert any("Ignored failed evidence" in reason for reason in assessment.reasoning)


def test_report_status_is_partial_when_an_optional_stage_fails():
    from backend.app.services.reporting.builder import build_report

    report = build_report(
        "partial-test", "completed",
        stages={"fact_check": {"status": "failed", "outcome": None}},
    )
    assert report["status"] == "partial"

