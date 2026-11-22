"""Shared execution-status rules for analysis stages."""

from __future__ import annotations

from typing import Any

STAGE_STATUSES = frozenset({
    "pending", "running", "completed", "partial", "failed", "unavailable", "skipped"
})
TERMINAL_STAGE_STATUSES = frozenset({
    "completed", "partial", "failed", "unavailable", "skipped"
})
SERVICE_OUTCOMES = frozenset({"success", "no_results", "no_text", "not_applicable"})


def normalize_service_status(result: Any) -> tuple[str, str | None, str | None]:
    """Return (execution status, outcome, validation error) for a module result."""
    if result is None or not (isinstance(result, dict) or hasattr(result, "__dict__")):
        return "failed", None, "The stage returned an invalid result."

    raw_status = result.get("status") if isinstance(result, dict) else getattr(result, "status", None)
    if raw_status is None:
        raw_status = "completed"
    if not isinstance(raw_status, str):
        return "failed", None, "The stage returned an invalid status."
    raw_status = raw_status.strip().lower()

    if raw_status in {"completed", "partial", "failed", "unavailable", "skipped"}:
        status = raw_status
        outcome = None
    elif raw_status in SERVICE_OUTCOMES:
        status = "completed"
        outcome = raw_status
    else:
        return "failed", None, "The stage returned an unsupported status."

    has_text = result.get("has_text") if isinstance(result, dict) else getattr(result, "has_text", None)
    if status == "completed" and has_text is False:
        outcome = "no_text"
    return status, outcome, None
