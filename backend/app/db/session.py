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
def normalize_database_url(url: str) -> str:
    """Ensure standard PostgreSQL URLs use the required asyncpg driver."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


_db_url = normalize_database_url(settings.DATABASE_URL)
_connect_args: dict = {}
_engine_kwargs: dict = {
    "echo": settings.DATABASE_ECHO,
    "pool_pre_ping": True,
}

if _db_url.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}
else:
    if settings.APP_ENV.lower() == "production" and _db_url.startswith("postgresql+asyncpg://"):
        _connect_args = {"ssl": ssl.create_default_context()}
    _engine_kwargs["pool_size"] = getattr(settings, "DATABASE_POOL_SIZE", 10)
    _engine_kwargs["max_overflow"] = getattr(settings, "DATABASE_MAX_OVERFLOW", 20)
    _engine_kwargs["pool_timeout"] = getattr(settings, "DATABASE_POOL_TIMEOUT", 30.0)
    _engine_kwargs["pool_recycle"] = getattr(settings, "DATABASE_POOL_RECYCLE_SECONDS", 1800)

_engine_kwargs["connect_args"] = _connect_args

engine = create_async_engine(
    _db_url,
    **_engine_kwargs,
)

if _db_url.startswith("sqlite"):
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
