"""
TruthLens ORM model — ModelRun

Every ML model inference is recorded here for auditability.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    analysis_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analyses.id"), nullable=False
    )
    model_id: Mapped[str] = mapped_column(String(256), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    task: Mapped[str] = mapped_column(String(64), nullable=False)
    device: Mapped[str] = mapped_column(String(32), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    runtime_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # ── Relationships ────────────────────────────────────────────
    analysis: Mapped["Analysis"] = relationship(  # noqa: F821
        "Analysis", back_populates="model_runs"
    )


Index("ix_model_runs_analysis_id", ModelRun.analysis_id)
