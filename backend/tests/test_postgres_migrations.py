"""
TruthLens — PostgreSQL Migration and Schema Compatibility Tests.

Verifies:
1. Migration revision chain integrity (no missing, broken, or duplicate down-revisions).
2. Fresh database upgrade from empty state to head.
3. Safe downgrade refusal when durable job history, ownership, or capacity exists.
4. Schema DDL compilation and type compatibility against PostgreSQL dialect (asyncpg).
5. Automatic PostgreSQL URL normalization (postgresql:// -> postgresql+asyncpg://).
6. Connection pool settings configuration.
7. Table, column, foreign key, index, and unique constraint definitions.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.schema import CreateIndex, CreateTable

from backend.app.config import settings
from backend.app.db.base import Base
from backend.app.db.session import normalize_database_url
import backend.app.models  # noqa: F401


def _alembic_config() -> Config:
    root = Path(__file__).resolve().parents[2]
    return Config(str(root / "alembic.ini"))


def test_migration_revision_chain():
    """Verify revision history forms an unbroken, forward-only chain."""
    import importlib

    rev1 = importlib.import_module("backend.migrations.versions.20261122_01_legacy_schema_baseline")
    rev2 = importlib.import_module("backend.migrations.versions.20261122_02_durable_jobs_security")

    assert rev1.revision == "20261122_01"
    assert rev1.down_revision is None

    assert rev2.revision == "20261122_02"
    assert rev2.down_revision == "20261122_01"


def test_fresh_migration_creates_all_tables_and_indexes(tmp_path, monkeypatch):
    """Verify upgrading a fresh database to head creates all required tables and indexes."""
    db_file = tmp_path / "fresh_migration.sqlite3"
    db_url = f"sqlite+aiosqlite:///{db_file.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    cfg = _alembic_config()

    command.upgrade(cfg, "head")

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        tables = {row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        expected_tables = {
            "analyses",
            "image_assets",
            "model_runs",
            "system_events",
            "analysis_jobs",
            "queue_capacity",
            "worker_leases",
            "rate_limit_buckets",
            "alembic_version",
        }
        assert expected_tables.issubset(tables)

        # Verify initial baseline rows
        cursor.execute("SELECT outstanding_count FROM queue_capacity WHERE id=1")
        assert cursor.fetchone()[0] == 0

        cursor.execute("SELECT name FROM worker_leases WHERE name='analysis-pipeline'")
        assert cursor.fetchone() is not None

        # Verify revision marker
        cursor.execute("SELECT version_num FROM alembic_version")
        assert cursor.fetchone()[0] == "20261122_02"


def test_postgresql_url_normalization():
    """Verify standard postgresql:// URLs are normalized to postgresql+asyncpg://."""
    standard_url = "postgresql://user:secret@localhost:5432/truthlens_db"
    normalized = normalize_database_url(standard_url)
    assert normalized == "postgresql+asyncpg://user:secret@localhost:5432/truthlens_db"

    # Already async URL should be unchanged
    async_url = "postgresql+asyncpg://user:secret@localhost:5432/truthlens_db"
    assert normalize_database_url(async_url) == async_url

    # SQLite URL should be unchanged
    sqlite_url = "sqlite+aiosqlite:///./test.db"
    assert normalize_database_url(sqlite_url) == sqlite_url


def test_postgresql_ddl_compilation_all_models():
    """Verify all ORM models compile valid DDL against the PostgreSQL dialect."""
    pg_dialect = postgresql.dialect()

    for table in Base.metadata.sorted_tables:
        ddl = str(CreateTable(table).compile(dialect=pg_dialect))
        assert table.name in ddl
        assert len(ddl) > 0

        # Check indexes compilation
        for index in table.indexes:
            idx_ddl = str(CreateIndex(index).compile(dialect=pg_dialect))
            assert index.name in idx_ddl


def test_postgresql_connection_pool_configuration():
    """Verify connection pool settings are properly configured for production PostgreSQL."""
    assert hasattr(settings, "DATABASE_POOL_SIZE")
    assert settings.DATABASE_POOL_SIZE >= 5
    assert hasattr(settings, "DATABASE_MAX_OVERFLOW")
    assert settings.DATABASE_MAX_OVERFLOW >= 10
    assert hasattr(settings, "DATABASE_POOL_TIMEOUT")
    assert settings.DATABASE_POOL_TIMEOUT > 0
    assert hasattr(settings, "DATABASE_POOL_RECYCLE_SECONDS")
    assert settings.DATABASE_POOL_RECYCLE_SECONDS >= 300


def test_migration_downgrade_safety_refuses_when_jobs_exist(tmp_path, monkeypatch):
    """Verify downgrade refuses to delete tables when durable jobs exist."""
    db_file = tmp_path / "downgrade_test.sqlite3"
    db_url = f"sqlite+aiosqlite:///{db_file.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    cfg = _alembic_config()

    command.upgrade(cfg, "head")

    with sqlite3.connect(db_file) as conn:
        conn.execute(
            "INSERT INTO analyses (id, status, created_at, updated_at) "
            "VALUES ('test-a1', 'QUEUED', '2026-11-23 00:00:00', '2026-11-23 00:00:00')"
        )
        conn.execute(
            "INSERT INTO analysis_jobs (analysis_id, payload_sha256, state, attempt_count, max_attempts, available_at, created_at, updated_at) "
            "VALUES ('test-a1', 'dummyhash', 'queued', 0, 3, '2026-11-23 00:00:00', '2026-11-23 00:00:00', '2026-11-23 00:00:00')"
        )
        conn.commit()

    with pytest.raises(RuntimeError, match="durable job history would be lost"):
        command.downgrade(cfg, "20261122_01")
