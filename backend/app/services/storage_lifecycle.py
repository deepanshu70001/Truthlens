"""Constrained temporary-upload cleanup and report-only orphan detection."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.app.models import Analysis, ImageAsset

_TEMP_NAME = re.compile(r"^[0-9a-f]{32}\.uploading$")


def cleanup_abandoned_temp_uploads(*, now: datetime | None = None) -> int:
    """Delete only expired, app-generated upload staging files from the temp root."""
    now = now or datetime.now(timezone.utc)
    root = settings.temp_uploads_dir.resolve()
    if not root.exists():
        return 0
    cutoff = now.timestamp() - settings.TEMP_UPLOAD_MAX_AGE_SECONDS
    removed = 0
    for candidate in root.iterdir():
        try:
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(root) or not candidate.is_file() or candidate.is_symlink():
                continue
            if not _TEMP_NAME.fullmatch(candidate.name) or candidate.stat().st_mtime > cutoff:
                continue
            candidate.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def cleanup_uncommitted_upload(analysis_id: str, staged_path: Path | None = None) -> None:
    """Remove only files generated for one not-yet-committed analysis ID."""
    try:
        safe_id = str(UUID(analysis_id))
    except ValueError:
        return
    roots = [settings.originals_dir, settings.thumbnails_dir, settings.processed_dir,
             settings.heatmaps_dir, settings.artifacts_dir / "forensics"]
    if staged_path is not None:
        stage_root = settings.temp_uploads_dir.resolve()
        path = staged_path.resolve()
        if path.is_relative_to(stage_root) and path.is_file():
            path.unlink(missing_ok=True)
    for root in roots:
        try:
            root_resolved = root.resolve()
            for candidate in root.glob(f"{safe_id}*"):
                resolved = candidate.resolve()
                if not resolved.is_relative_to(root_resolved):
                    continue
                if resolved.is_dir():
                    # Only the per-analysis heatmap folder is a generated directory.
                    if root == settings.heatmaps_dir and resolved.name == safe_id:
                        import shutil
                        shutil.rmtree(resolved, ignore_errors=True)
                elif resolved.is_file():
                    resolved.unlink(missing_ok=True)
        except OSError:
            continue


async def find_orphan_storage_files(db: AsyncSession) -> list[str]:
    """List generated storage files without DB parents; never deletes anything."""
    analysis_ids = set((await db.scalars(select(Analysis.id))).all())
    asset_rows = (await db.execute(select(ImageAsset.storage_path, ImageAsset.thumbnail_path))).all()
    known_asset_paths = {
        Path(raw).resolve()
        for row in asset_rows
        for raw in row if raw
    }
    roots = [settings.originals_dir, settings.thumbnails_dir, settings.processed_dir,
             settings.heatmaps_dir, settings.artifacts_dir / "forensics"]
    orphans: list[str] = []
    storage_root = settings.storage_dir.resolve()
    for root in roots:
        root_resolved = root.resolve()
        if not root_resolved.exists():
            continue
        for candidate in root_resolved.rglob("*"):
            try:
                resolved = candidate.resolve(strict=True)
                if not resolved.is_relative_to(root_resolved) or candidate.is_symlink():
                    continue
                relative = resolved.relative_to(storage_root).as_posix()
                if root == settings.originals_dir:
                    orphan = resolved not in known_asset_paths
                else:
                    name = resolved.parent.name if root == settings.heatmaps_dir else resolved.name
                    match = re.match(r"^([0-9a-fA-F-]{36})(?:_|$)", name)
                    orphan = bool(match and match.group(1) not in analysis_ids)
                if orphan:
                    orphans.append(relative)
            except (OSError, ValueError):
                continue
    return sorted(orphans)
