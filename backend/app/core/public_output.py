"""Defensive serialization for reports written before current E1/E9 contracts."""

from __future__ import annotations

import re
from typing import Any

_PRIVATE_KEYS = {
    "storage_path", "local_path", "absolute_path", "api_key", "access_token",
    "refresh_token", "authorization", "password", "secret_key", "credentials_path",
}
_WINDOWS_PATH = re.compile(r"(?i)\b[A-Z]:\\[^\s\"'<>|]+")
_UNIX_PATH = re.compile(r"(?<![:\w])/(?:home|Users|tmp|mnt|var|root|workspace|app|opt)/[^\s\"'<>]+", re.IGNORECASE)
_SECRET_VALUE = re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]+")


def safe_public_text(value: str) -> str:
    if "Traceback (most recent call last)" in value:
        return "Internal error details are hidden."
    value = _WINDOWS_PATH.sub("[redacted path]", value)
    value = _UNIX_PATH.sub("[redacted path]", value)
    value = _SECRET_VALUE.sub(r"\1=[redacted]", value)
    return value[:20_000]


def sanitize_public_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: sanitize_public_payload(item)
            for key, item in value.items()
            if not _is_private_key(str(key))
        }
    if isinstance(value, list):
        return [sanitize_public_payload(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_public_payload(item) for item in value]
    if isinstance(value, str):
        return safe_public_text(value)
    return value


def _is_private_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
    exact = {re.sub(r"[^a-z0-9]", "", item) for item in _PRIVATE_KEYS}
    sensitive_fragments = (
        "apikey", "accesstoken", "refreshtoken", "authorization", "password",
        "secret", "credentialspath", "storagepath", "localpath", "absolutepath",
    )
    return normalized in exact or any(fragment in normalized for fragment in sensitive_fragments)
