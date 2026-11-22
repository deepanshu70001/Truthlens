"""
TruthLens — Database engine & session management.

Supports PostgreSQL (primary) and SQLite (fallback).
Uses SQLAlchemy 2.0 async engine.
"""

from __future__ import annotations

import ssl

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import event

from backend.app.config import settings

# ── Engine ───────────────────────────────────────────────────────
_connect_args: dict = {}
if settings.DATABASE_URL.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}
elif settings.APP_ENV.lower() == "production" and settings.DATABASE_URL.startswith("postgresql+asyncpg://"):
    _connect_args = {"ssl": ssl.create_default_context()}

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DATABASE_ECHO,
    connect_args=_connect_args,
    pool_pre_ping=True,
)

if settings.DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_integrity(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

# ── Session factory ──────────────────────────────────────────────
async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:  # type: ignore[misc]
    """FastAPI dependency that yields an async DB session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
