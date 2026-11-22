"""
TruthLens ORM model — Analysis

Represents a single image analysis lifecycle.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base


class AnalysisStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # Null for records created before authentication/ownership was introduced.
    # Such legacy records are never exposed to authenticated users by default.
    owner_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    status: Mapped[str] = mapped_column(
        Enum(AnalysisStatus), default=AnalysisStatus.QUEUED, nullable=False
    )
    truth_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    truth_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    truth_label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    current_step: Mapped[str | None] = mapped_column(String(128), nullable=True)
    completed_steps: Mapped[str | None] = mapped_column(
        Text, nullable=True, default=""
    )  # JSON list stored as text
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # Full JSON report

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ── Relationships ────────────────────────────────────────────
    image_asset: Mapped["ImageAsset"] = relationship(  # noqa: F821
        "ImageAsset", back_populates="analysis", uselist=False, lazy="selectin"
    )
    model_runs: Mapped[list["ModelRun"]] = relationship(  # noqa: F821
        "ModelRun", back_populates="analysis", lazy="selectin"
    )
    system_events: Mapped[list["SystemEvent"]] = relationship(  # noqa: F821
        "SystemEvent", back_populates="analysis", lazy="selectin"
    )


Index("ix_analyses_created_at", Analysis.created_at)
