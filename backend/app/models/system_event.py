"""
TruthLens ORM model — SystemEvent

Audit log for system-wide events: model loads, errors, warnings.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base


class SystemEvent(Base):
    __tablename__ = "system_events"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    analysis_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("analyses.id"), nullable=True
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="info")
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # ── Relationships ────────────────────────────────────────────
    analysis: Mapped["Analysis | None"] = relationship(  # noqa: F821
        "Analysis", back_populates="system_events"
    )


Index("ix_system_events_analysis_id", SystemEvent.analysis_id)
