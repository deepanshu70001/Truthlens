"""Add durable jobs, ownership, cross-process worker lease and rate limits."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "20261122_02"
down_revision = "20261122_01"
branch_labels = None
depends_on = None


def _has_column(table: str, column: str) -> bool:
    return column in {item["name"] for item in inspect(op.get_bind()).get_columns(table)}


def _has_table(table: str) -> bool:
    return table in inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    bind = op.get_bind()
    if not _has_column("analyses", "owner_id"):
        op.add_column("analyses", sa.Column("owner_id", sa.String(128), nullable=True))
    existing_indexes = {item["name"] for item in inspect(bind).get_indexes("analyses")}
    if "ix_analyses_owner_id" not in existing_indexes:
        op.create_index("ix_analyses_owner_id", "analyses", ["owner_id"])
    if "ix_analyses_created_at" not in existing_indexes:
        op.create_index("ix_analyses_created_at", "analyses", ["created_at"])

    if not _has_table("analysis_jobs"):
        op.create_table(
            "analysis_jobs",
            sa.Column("analysis_id", sa.String(36), sa.ForeignKey("analyses.id"), primary_key=True),
            sa.Column("owner_id", sa.String(128), nullable=True),
            sa.Column("idempotency_key", sa.String(128), nullable=True),
            sa.Column("payload_sha256", sa.String(64), nullable=False),
            sa.Column("state", sa.String(16), nullable=False),
            sa.Column("attempt_count", sa.Integer(), nullable=False),
            sa.Column("max_attempts", sa.Integer(), nullable=False),
            sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("claimed_by", sa.String(128)),
            sa.Column("lease_token", sa.String(64)),
            sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
            sa.Column("heartbeat_at", sa.DateTime(timezone=True)),
            sa.Column("error_category", sa.String(64)),
            sa.Column("error_message", sa.String(500)),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True)),
            sa.Column("completed_at", sa.DateTime(timezone=True)),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("owner_id", "idempotency_key", name="uq_analysis_jobs_owner_idempotency"),
        )
        op.create_index("ix_analysis_jobs_state_available", "analysis_jobs", ["state", "available_at"])
        op.create_index("ix_analysis_jobs_state_lease", "analysis_jobs", ["state", "lease_expires_at"])
        op.create_index("ix_analysis_jobs_owner_created", "analysis_jobs", ["owner_id", "created_at"])

    if not _has_table("queue_capacity"):
        op.create_table(
            "queue_capacity",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("outstanding_count", sa.Integer(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
    if not _has_table("worker_leases"):
        op.create_table(
            "worker_leases",
            sa.Column("name", sa.String(64), primary_key=True),
            sa.Column("owner_token", sa.String(64)),
            sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
            sa.Column("heartbeat_at", sa.DateTime(timezone=True)),
        )
    if not _has_table("rate_limit_buckets"):
        op.create_table(
            "rate_limit_buckets",
            sa.Column("scope", sa.String(64), primary_key=True),
            sa.Column("identity_key", sa.String(64), primary_key=True),
            sa.Column("window_start", sa.BigInteger(), nullable=False),
            sa.Column("request_count", sa.Integer(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )

    now = datetime.now(timezone.utc)
    if not bind.execute(sa.text("SELECT 1 FROM queue_capacity WHERE id=1")).first():
        bind.execute(sa.text(
            "INSERT INTO queue_capacity (id, outstanding_count, updated_at) VALUES (1, 0, :now)"
        ), {"now": now})
    if not bind.execute(sa.text("SELECT 1 FROM worker_leases WHERE name=:name"), {"name": "analysis-pipeline"}).first():
        bind.execute(sa.text(
            "INSERT INTO worker_leases (name, owner_token, lease_expires_at, heartbeat_at) "
            "VALUES (:name, NULL, NULL, NULL)"
        ), {"name": "analysis-pipeline"})

    # Adopt pre-Alembic queued/processing analyses so the first worker can
    # recover them. No report, image, model identity, or evidence fields change.
    analyses = bind.execute(sa.text(
        "SELECT id, status, completed_steps FROM analyses"
    )).mappings().all()
    outstanding = 0
    for row in analyses:
        status_value = str(row["status"] or "").lower()
        if status_value not in {"queued", "processing"}:
            continue
        existing_job = bind.execute(
            sa.text("SELECT analysis_id FROM analysis_jobs WHERE analysis_id=:analysis_id"),
            {"analysis_id": row["id"]},
        ).first()
        if existing_job:
            continue
        image = bind.execute(
            sa.text("SELECT sha256_hash FROM image_assets WHERE analysis_id=:analysis_id"),
            {"analysis_id": row["id"]},
        ).first()
        digest = image[0] if image and image[0] else "0" * 64
        bind.execute(sa.text(
            "INSERT INTO analysis_jobs (analysis_id, owner_id, idempotency_key, payload_sha256, state, "
            "attempt_count, max_attempts, available_at, claimed_by, lease_token, lease_expires_at, heartbeat_at, "
            "error_category, error_message, created_at, started_at, completed_at, updated_at) "
            "VALUES (:id, NULL, NULL, :digest, 'queued', 0, 3, :now, NULL, NULL, NULL, NULL, NULL, NULL, :now, NULL, NULL, :now)"
        ), {"id": row["id"], "digest": digest, "now": now})
        completed_steps = row["completed_steps"]
        try:
            progress = json.loads(completed_steps or "[]")
            if isinstance(progress, dict) and isinstance(progress.get("stage_statuses"), dict):
                progress["stage_statuses"] = {
                    name: "pending" if state == "running" else state
                    for name, state in progress["stage_statuses"].items()
                }
                progress["completed_steps"] = [
                    name for name, state in progress["stage_statuses"].items()
                    if state == "completed"
                ]
                completed_steps = json.dumps(progress)
        except (json.JSONDecodeError, TypeError):
            pass
        bind.execute(sa.text(
            "UPDATE analyses SET status='QUEUED', current_step=NULL, completed_steps=:steps WHERE id=:id"
        ), {"steps": completed_steps, "id": row["id"]})
        outstanding += 1
    if outstanding:
        bind.execute(sa.text(
            "UPDATE queue_capacity SET outstanding_count=outstanding_count+:count, updated_at=:now WHERE id=1"
        ), {"count": outstanding, "now": now})

    if "ix_model_runs_analysis_id" not in {item["name"] for item in inspect(bind).get_indexes("model_runs")}:
        op.create_index("ix_model_runs_analysis_id", "model_runs", ["analysis_id"])
    if "ix_system_events_analysis_id" not in {item["name"] for item in inspect(bind).get_indexes("system_events")}:
        op.create_index("ix_system_events_analysis_id", "system_events", ["analysis_id"])


def downgrade() -> None:
    bind = op.get_bind()
    if _has_table("analysis_jobs") and bind.execute(sa.text("SELECT COUNT(*) FROM analysis_jobs")).scalar_one():
        raise RuntimeError("Refusing downgrade because durable job history would be lost.")
    if _has_column("analyses", "owner_id") and bind.execute(
        sa.text("SELECT COUNT(*) FROM analyses WHERE owner_id IS NOT NULL")
    ).scalar_one():
        raise RuntimeError("Refusing downgrade because analysis ownership data would be lost.")
    if _has_table("queue_capacity") and bind.execute(
        sa.text("SELECT outstanding_count FROM queue_capacity WHERE id=1")
    ).scalar_one_or_none():
        outstanding = bind.execute(sa.text("SELECT outstanding_count FROM queue_capacity WHERE id=1")).scalar_one()
        if outstanding:
            raise RuntimeError("Refusing downgrade while analysis jobs remain outstanding.")
    for name in ("ix_model_runs_analysis_id",):
        if name in {item["name"] for item in inspect(bind).get_indexes("model_runs")}:
            op.drop_index(name, table_name="model_runs")
    for name in ("ix_system_events_analysis_id",):
        if name in {item["name"] for item in inspect(bind).get_indexes("system_events")}:
            op.drop_index(name, table_name="system_events")
    for table in ("rate_limit_buckets", "worker_leases", "queue_capacity", "analysis_jobs"):
        if _has_table(table):
            op.drop_table(table)
    for name in ("ix_analyses_owner_id", "ix_analyses_created_at"):
        if name in {item["name"] for item in inspect(bind).get_indexes("analyses")}:
            op.drop_index(name, table_name="analyses")
    if _has_column("analyses", "owner_id"):
        with op.batch_alter_table("analyses") as batch:
            batch.drop_column("owner_id")
