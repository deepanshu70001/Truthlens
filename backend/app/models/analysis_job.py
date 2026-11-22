"""Durable analysis queue, worker lease, admission and rate-limit records."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AnalysisJob(Base):
    """Internal delivery state for one analysis; public stage semantics remain E1."""

    __tablename__ = "analysis_jobs"
    __table_args__ = (
        UniqueConstraint("owner_id", "idempotency_key", name="uq_analysis_jobs_owner_idempotency"),
        Index("ix_analysis_jobs_state_available", "state", "available_at"),
        Index("ix_analysis_jobs_state_lease", "state", "lease_expires_at"),
        Index("ix_analysis_jobs_owner_created", "owner_id", "created_at"),
    )

    # The existing analysis UUID is also the stable job UUID for API compatibility.
    analysis_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analyses.id"), primary_key=True
    )
    owner_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    payload_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    claimed_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class QueueCapacity(Base):
    """One row bounds durable outstanding jobs across all API processes."""

    __tablename__ = "queue_capacity"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    outstanding_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class WorkerLease(Base):
    """Global analysis lease serializes model/GPU work across worker processes."""

    __tablename__ = "worker_leases"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RateLimitBucket(Base):
    """A single fixed-window counter per scope and keyed client identity."""

    __tablename__ = "rate_limit_buckets"

    scope: Mapped[str] = mapped_column(String(64), primary_key=True)
    identity_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    window_start: Mapped[int] = mapped_column(BigInteger, nullable=False)
    request_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)
