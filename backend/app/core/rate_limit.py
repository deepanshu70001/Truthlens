"""Database-backed fixed-window limits shared by API worker processes."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone

from fastapi import HTTPException, Request
from sqlalchemy import case
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from backend.app.config import settings
from backend.app.core.auth import Principal
from backend.app.db.session import async_session_factory
from backend.app.models.analysis_job import RateLimitBucket


async def check_rate_limit(
    request: Request,
    *,
    scope: str,
    limit: int,
    principal: Principal | None = None,
    window_seconds: int = 60,
) -> None:
    """Consume one shared quota unit, failing closed when the DB is unavailable."""
    if limit < 1 or window_seconds < 1:
        raise RuntimeError("Rate limit settings must be positive.")
    if principal is not None:
        identity = f"user:{principal.subject}"
    else:
        # Do not trust arbitrary X-Forwarded-For values. Uvicorn only rewrites
        # client.host for configured trusted proxy peers.
        peer = request.client.host if request.client else "unknown"
        identity = f"ip:{peer}"
    identity_key = hashlib.sha256(
        f"{settings.SECRET_KEY}\0{scope}\0{identity}".encode("utf-8")
    ).hexdigest()
    now_seconds = int(time.time())
    window_start = now_seconds // window_seconds * window_seconds
    dialect = async_session_factory.kw.get("bind").dialect.name
    insert = pg_insert if dialect == "postgresql" else sqlite_insert if dialect == "sqlite" else None
    if insert is None:
        raise HTTPException(status_code=503, detail="Shared rate-limit storage is unavailable.")

    try:
        async with async_session_factory() as session:
            statement = insert(RateLimitBucket).values(
                scope=scope,
                identity_key=identity_key,
                window_start=window_start,
                request_count=1,
                updated_at=datetime.now(timezone.utc),
            )
            statement = statement.on_conflict_do_update(
                index_elements=["scope", "identity_key"],
                set_={
                    "window_start": window_start,
                    "request_count": case(
                        (RateLimitBucket.window_start == window_start, RateLimitBucket.request_count + 1),
                        else_=1,
                    ),
                    "updated_at": datetime.now(timezone.utc),
                },
            ).returning(RateLimitBucket.request_count)
            count = (await session.execute(statement)).scalar_one()
            await session.commit()
    except HTTPException:
        raise
    except Exception:
        # Expensive work never silently falls back to unshared in-memory limits.
        raise HTTPException(status_code=503, detail="Request protection is temporarily unavailable.") from None

    if count > limit:
        retry_after = max(1, window_start + window_seconds - now_seconds)
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Retry after the current limit window.",
            headers={"Retry-After": str(retry_after)},
        )

