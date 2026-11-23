"""
TruthLens — Persistent Cache for Third-Party Fact-Check Queries.

Stores cached Google Fact Check Tools responses using a local persistent SQLite
database located under settings.STORAGE_PATH / "cache" / "factcheck_cache.sqlite3".

Key Features:
- Deterministic cache key derived from SHA-256 of provider, language_code, and normalized query.
- Preserves full publisher attribution, review URL, review dates, ratings, and evidence IDs.
- Never stores API keys, credentials, or secrets.
- In-flight request deduplication via asyncio locks to avoid concurrent duplicate provider queries.
- Explicit TTL enforcement with automatic purge of expired entries.
- Eviction when capacity limit (FACTCHECK_CACHE_MAX_ENTRIES) is reached.
- Never caches provider failures, timeouts, 429 quota exhaustion, or unconfigured states.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.app.config import settings
from backend.app.core.logging import get_logger
from backend.app.services.factcheck.service import FactCheckClaim

log = get_logger("factcheck.cache")


def compute_factcheck_cache_key(
    query: str,
    language_code: Optional[str] = None,
    provider: str = "google_fact_check_tools",
) -> str:
    """Compute a deterministic 64-character hex cache key."""
    norm_query = re.sub(r"\s+", " ", (query or "").strip()).casefold()
    norm_lang = (language_code or "").strip().lower()
    norm_provider = provider.strip().lower()
    key_material = f"{norm_provider}\0{norm_lang}\0{norm_query}"
    return hashlib.sha256(key_material.encode("utf-8")).hexdigest()


class FactCheckCache:
    """Persistent, thread-safe SQLite cache for third-party fact-check queries."""

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        if db_path is not None:
            self._db_path = Path(db_path) if str(db_path) != ":memory:" else ":memory:"
        else:
            storage_root = Path(settings.STORAGE_PATH)
            cache_dir = storage_root / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = cache_dir / "factcheck_cache.sqlite3"

        self._in_flight_locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._db_path == ":memory:":
            if not hasattr(self, "_memory_conn") or self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            return self._memory_conn

        conn = sqlite3.connect(str(self._db_path), timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS factcheck_cache (
                        cache_key TEXT PRIMARY KEY,
                        query_hash TEXT NOT NULL,
                        normalized_query TEXT NOT NULL,
                        language_code TEXT,
                        provider TEXT NOT NULL,
                        status TEXT NOT NULL,
                        claims_json TEXT NOT NULL,
                        total_results INTEGER NOT NULL,
                        retrieved_at TEXT NOT NULL,
                        expires_at REAL NOT NULL,
                        created_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS ix_factcheck_cache_expires_at ON factcheck_cache(expires_at)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS ix_factcheck_cache_query_hash ON factcheck_cache(query_hash)"
                )
        finally:
            if self._db_path != ":memory:":
                conn.close()

    async def get_in_flight_lock(self, cache_key: str) -> asyncio.Lock:
        """Obtain a per-cache-key lock to prevent concurrent duplicate provider calls."""
        async with self._global_lock:
            if cache_key not in self._in_flight_locks:
                self._in_flight_locks[cache_key] = asyncio.Lock()
            return self._in_flight_locks[cache_key]

    def get(self, cache_key: str) -> Optional[Tuple[List[FactCheckClaim], str]]:
        """
        Retrieve cached fact-check results.
        Returns (claims, status) tuple if valid cache hit; None on miss or expired.
        """
        if not getattr(settings, "FACTCHECK_CACHE_ENABLED", True):
            return None

        now = time.time()
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT status, claims_json, expires_at
                FROM factcheck_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            status, claims_json, expires_at = row
            if now >= expires_at:
                # Expired entry: purge lazily
                with conn:
                    conn.execute("DELETE FROM factcheck_cache WHERE cache_key = ?", (cache_key,))
                return None

            try:
                raw_claims = json.loads(claims_json)
                if not isinstance(raw_claims, list):
                    raise ValueError("Malformed claims_json")
                claims = [FactCheckClaim(**item) for item in raw_claims]
                return claims, status
            except Exception as exc:
                log.warning("Purging malformed fact-check cache entry (%s): %s", cache_key, exc)
                with conn:
                    conn.execute("DELETE FROM factcheck_cache WHERE cache_key = ?", (cache_key,))
                return None
        finally:
            if self._db_path != ":memory:":
                conn.close()

    def set(
        self,
        cache_key: str,
        query: str,
        claims: List[FactCheckClaim],
        status: str,
        language_code: Optional[str] = None,
        provider: str = "google_fact_check_tools",
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Store successful or no_results queries in the cache.
        Provider failures, timeouts, and rate limits are explicitly rejected.
        """
        if not getattr(settings, "FACTCHECK_CACHE_ENABLED", True):
            return

        if status not in ("success", "no_results"):
            # Never cache failures or unavailable states
            return

        ttl = ttl_seconds if ttl_seconds is not None else getattr(
            settings, "FACTCHECK_CACHE_TTL_SECONDS", 86400
        )
        now = time.time()
        expires_at = now + max(1, ttl)
        retrieved_at = datetime.now(timezone.utc).isoformat()
        normalized_query = re.sub(r"\s+", " ", (query or "").strip())
        query_hash = hashlib.sha256(normalized_query.casefold().encode("utf-8")).hexdigest()
        claims_json = json.dumps([c.to_dict() for c in claims])

        conn = self._get_connection()
        try:
            with conn:
                # 1. Bounded capacity enforcement
                max_entries = getattr(settings, "FACTCHECK_CACHE_MAX_ENTRIES", 5000)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM factcheck_cache")
                count = cursor.fetchone()[0]

                if count >= max_entries:
                    # Purge expired entries first
                    conn.execute("DELETE FROM factcheck_cache WHERE expires_at <= ?", (now,))
                    cursor.execute("SELECT COUNT(*) FROM factcheck_cache")
                    count = cursor.fetchone()[0]

                    # If still over capacity, evict oldest entries by created_at
                    if count >= max_entries:
                        to_evict = count - max_entries + 1
                        conn.execute(
                            """
                            DELETE FROM factcheck_cache WHERE cache_key IN (
                                SELECT cache_key FROM factcheck_cache
                                ORDER BY created_at ASC LIMIT ?
                            )
                            """,
                            (to_evict,),
                        )

                # 2. Insert or replace cache entry
                conn.execute(
                    """
                    INSERT OR REPLACE INTO factcheck_cache (
                        cache_key, query_hash, normalized_query, language_code, provider,
                        status, claims_json, total_results, retrieved_at, expires_at, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_key,
                        query_hash,
                        normalized_query,
                        language_code,
                        provider,
                        status,
                        claims_json,
                        len(claims),
                        retrieved_at,
                        expires_at,
                        now,
                    ),
                )
        finally:
            if self._db_path != ":memory:":
                conn.close()

    def clear(self) -> None:
        """Purge all cache entries."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("DELETE FROM factcheck_cache")
        finally:
            if self._db_path != ":memory:":
                conn.close()


# Module-level singleton
_factcheck_cache: Optional[FactCheckCache] = None


def get_factcheck_cache() -> FactCheckCache:
    """Return the shared FactCheckCache singleton."""
    global _factcheck_cache
    if _factcheck_cache is None:
        _factcheck_cache = FactCheckCache()
    return _factcheck_cache
