"""
TruthLens — Structured, privacy-conscious logging.

No PII, no raw image data in logs.
"""

from __future__ import annotations

import logging
import re
import sys

from backend.app.config import settings


class SensitiveDataFilter(logging.Filter):
    """Redacts tokens, credentials, and secrets from formatted log records."""

    _PATTERNS = [
        re.compile(r"(?i)\b(bearer\s+)[a-zA-Z0-9_\-\.]{10,}"),
        re.compile(r"(?i)\b(api[_-]?key|secret|password|access_token)\s*[:=]\s*[^\s,;]+"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for pattern in self._PATTERNS:
                record.msg = pattern.sub(r"\1[redacted]", record.msg)
        return True


def setup_logging() -> None:
    """Configure application-wide logging."""
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(SensitiveDataFilter())

    root = logging.getLogger("truthlens")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # Quieten noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if settings.DATABASE_ECHO else logging.WARNING
    )


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the 'truthlens' namespace."""
    return logging.getLogger(f"truthlens.{name}")
