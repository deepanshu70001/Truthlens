"""Adopt or create the pre-E8/E9/E10 TruthLens schema.

Existing tables are validated and preserved. This baseline is idempotent for
the repository's deployed pre-Alembic SQLite/PostgreSQL schema.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "20261122_01"
down_revision = None
branch_labels = None
depends_on = None


def _existing_columns(bind, name: str) -> set[str]:
    return {column["name"] for column in inspect(bind).get_columns(name)}


def _create_or_validate(bind, name: str, required: set[str], create) -> None:
    if name in inspect(bind).get_table_names():
        missing = required - _existing_columns(bind, name)
        if missing:
            raise RuntimeError(
                f"Existing {name} table is missing required columns; baseline will not rewrite it."
            )
        return
    create()


def upgrade() -> None:
    bind = op.get_bind()
    _create_or_validate(bind, "analyses", {
        "id", "status", "truth_score", "truth_confidence", "truth_label", "current_step",
        "completed_steps", "processing_time_ms", "error_message", "created_at", "updated_at",
        "completed_at", "report_json",
    }, lambda: op.create_table(
        "analyses",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("status", sa.Enum("QUEUED", "PROCESSING", "COMPLETED", "FAILED", name="analysisstatus"), nullable=False),
        sa.Column("truth_score", sa.Float()),
        sa.Column("truth_confidence", sa.Float()),
        sa.Column("truth_label", sa.String(64)),
        sa.Column("current_step", sa.String(128)),
        sa.Column("completed_steps", sa.Text()),
        sa.Column("processing_time_ms", sa.Integer()),
        sa.Column("error_message", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("report_json", sa.Text()),
    ))
    _create_or_validate(bind, "image_assets", {
        "id", "analysis_id", "original_filename", "storage_filename", "storage_path", "sha256_hash",
        "perceptual_hash", "mime_type", "file_size_bytes", "width", "height", "format", "thumbnail_path", "created_at",
    }, lambda: op.create_table(
        "image_assets",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("analysis_id", sa.String(36), sa.ForeignKey("analyses.id"), nullable=False, unique=True),
        sa.Column("original_filename", sa.String(255), nullable=False),
        sa.Column("storage_filename", sa.String(255), nullable=False),
        sa.Column("storage_path", sa.String(512), nullable=False),
        sa.Column("sha256_hash", sa.String(64), nullable=False),
        sa.Column("perceptual_hash", sa.String(64)),
        sa.Column("mime_type", sa.String(64), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer()),
        sa.Column("height", sa.Integer()),
        sa.Column("format", sa.String(16)),
        sa.Column("thumbnail_path", sa.String(512)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    ))
    _create_or_validate(bind, "model_runs", {
        "id", "analysis_id", "model_id", "model_version", "task", "device", "input_hash",
        "result_json", "confidence", "runtime_ms", "error_message", "created_at",
    }, lambda: op.create_table(
        "model_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("analysis_id", sa.String(36), sa.ForeignKey("analyses.id"), nullable=False),
        sa.Column("model_id", sa.String(256), nullable=False),
        sa.Column("model_version", sa.String(128)),
        sa.Column("task", sa.String(64), nullable=False),
        sa.Column("device", sa.String(32), nullable=False),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("result_json", sa.Text()),
        sa.Column("confidence", sa.Float()),
        sa.Column("runtime_ms", sa.Integer()),
        sa.Column("error_message", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    ))
    _create_or_validate(bind, "system_events", {
        "id", "analysis_id", "event_type", "severity", "source", "message", "metadata_json", "created_at",
    }, lambda: op.create_table(
        "system_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("analysis_id", sa.String(36), sa.ForeignKey("analyses.id")),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("source", sa.String(128), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    ))


def downgrade() -> None:
    # The initial tables predate Alembic and may contain user reports. Never
    # drop them as part of a rollback; later migrations can be rolled back first.
    pass
