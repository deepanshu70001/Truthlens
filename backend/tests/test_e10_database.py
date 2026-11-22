"""Migration and database lifecycle tests use isolated temporary databases."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select

from backend.app.config import settings
from backend.app.db.session import async_session_factory
from backend.app.models import Analysis, AnalysisJob, QueueCapacity, WorkerLease
from backend.app.services.storage_lifecycle import (
    cleanup_abandoned_temp_uploads,
    find_orphan_storage_files,
)


def _alembic_config() -> Config:
    root = Path(__file__).resolve().parents[2]
    return Config(str(root / "alembic.ini"))


def _db_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path.as_posix()}"


def test_fresh_database_migrates_to_head(tmp_path, monkeypatch):
    db_path = tmp_path / "fresh.sqlite3"
    monkeypatch.setenv("DATABASE_URL", _db_url(db_path))
    config = _alembic_config()
    command.upgrade(config, "head")

    with sqlite3.connect(db_path) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"analyses", "analysis_jobs", "queue_capacity", "worker_leases", "rate_limit_buckets"} <= tables
        revision = db.execute("SELECT version_num FROM alembic_version").fetchone()[0]
        assert revision == "20261122_02"
        assert db.execute("SELECT outstanding_count FROM queue_capacity WHERE id=1").fetchone()[0] == 0
        assert db.execute("SELECT name FROM worker_leases WHERE name='analysis-pipeline'").fetchone()
    command.check(config)


def test_existing_legacy_database_upgrade_preserves_report_and_recovers_queued_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.sqlite3"
    monkeypatch.setenv("DATABASE_URL", _db_url(db_path))
    config = _alembic_config()
    command.upgrade(config, "20261122_01")
    analysis_id = str(uuid.uuid4())
    report_json = json.dumps({"analysis_id": analysis_id, "status": "partial", "sentinel": "preserve-me"})
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO analyses (id,status,completed_steps,created_at,updated_at,report_json) "
            "VALUES (?, 'QUEUED', ?, ?, ?, ?)",
            (analysis_id, json.dumps({"stage_statuses": {"ocr": "running"}}),
             datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat(), report_json),
        )
        db.commit()

    command.upgrade(config, "head")
    with sqlite3.connect(db_path) as db:
        row = db.execute(
            "SELECT status, report_json, owner_id, completed_steps FROM analyses WHERE id=?", (analysis_id,)
        ).fetchone()
        assert row[0] == "QUEUED"
        assert json.loads(row[1])["sentinel"] == "preserve-me"
        assert row[2] is None  # legacy records remain private until ownership is provisioned
        assert json.loads(row[3])["stage_statuses"]["ocr"] == "pending"
        job = db.execute(
            "SELECT state, attempt_count FROM analysis_jobs WHERE analysis_id=?", (analysis_id,)
        ).fetchone()
        assert job == ("queued", 0)
        assert db.execute("SELECT outstanding_count FROM queue_capacity WHERE id=1").fetchone()[0] == 1

    # Downgrade refuses to erase durable work or ownership/report history.
    with pytest.raises(RuntimeError, match="job history"):
        command.downgrade(config, "20261122_01")
    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT report_json FROM analyses WHERE id=?", (analysis_id,)).fetchone()[0] == report_json


@pytest.mark.asyncio
async def test_analysis_deletion_removes_only_its_storage_and_database_rows(client_without_worker):
    from backend.tests.conftest import create_test_jpeg

    response = await client_without_worker.post(
        "/api/v1/analyze", files={"image": ("delete-me.jpg", create_test_jpeg(), "image/jpeg")}
    )
    assert response.status_code == 201
    analysis_id = response.json()["analysis_id"]
    async with async_session_factory() as db:
        analysis = await db.get(Analysis, analysis_id)
        original = Path(analysis.image_asset.storage_path)
    heatmap_dir = settings.heatmaps_dir / analysis_id
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    (heatmap_dir / "overlay.png").write_bytes(b"test-only artifact")
    ela = settings.artifacts_dir / "forensics" / f"{analysis_id}_ela.png"
    ela.parent.mkdir(parents=True, exist_ok=True)
    ela.write_bytes(b"test-only artifact")

    deleted = await client_without_worker.delete(f"/api/v1/analysis/{analysis_id}")
    assert deleted.status_code == 204
    assert not original.exists()
    assert not heatmap_dir.exists()
    assert not ela.exists()
    async with async_session_factory() as db:
        assert await db.get(Analysis, analysis_id) is None
        assert await db.get(AnalysisJob, analysis_id) is None


@pytest.mark.asyncio
async def test_orphan_scanner_is_report_only_and_temp_cleanup_is_name_and_age_bounded(monkeypatch):
    settings.ensure_storage_dirs()
    orphan_id = str(uuid.uuid4())
    orphan = settings.originals_dir / f"{orphan_id}.jpg"
    orphan.write_bytes(b"test orphan")
    async with async_session_factory() as db:
        found = await find_orphan_storage_files(db)
    assert orphan.relative_to(settings.storage_dir).as_posix() in found
    assert orphan.exists()
    orphan.unlink()

    monkeypatch.setattr(settings, "TEMP_UPLOAD_MAX_AGE_SECONDS", 1)
    generated_name = f"{uuid.uuid4().hex}.uploading"
    generated = settings.temp_uploads_dir / generated_name
    unrelated = settings.temp_uploads_dir / "keep-me.tmp"
    generated.write_bytes(b"expired")
    unrelated.write_bytes(b"keep")
    old_time = (datetime.now(timezone.utc) - timedelta(seconds=30)).timestamp()
    os.utime(generated, (old_time, old_time))
    os.utime(unrelated, (old_time, old_time))
    assert cleanup_abandoned_temp_uploads() == 1
    assert not generated.exists()
    assert unrelated.exists()
    unrelated.unlink()


@pytest.mark.asyncio
async def test_database_foreign_keys_reject_orphaned_job_rows():
    from sqlalchemy.exc import IntegrityError

    async with async_session_factory() as db:
        db.add(AnalysisJob(
            analysis_id=str(uuid.uuid4()),
            payload_sha256="b" * 64,
            state="queued",
            attempt_count=0,
            max_attempts=1,
            available_at=datetime.now(timezone.utc),
        ))
        with pytest.raises(IntegrityError):
            await db.commit()
        await db.rollback()

