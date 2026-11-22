"""Analysis ownership checks used by every report and artifact retrieval path."""

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.app.core.auth import Principal
from backend.app.models import Analysis


async def require_analysis_owner(
    analysis_id: str,
    principal: Principal,
    db: AsyncSession,
) -> Analysis:
    analysis = await db.get(Analysis, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    owner_id = analysis.owner_id
    if principal.development_bypass and settings.APP_ENV.lower() in {"development", "test"}:
        if owner_id in {None, principal.subject}:
            return analysis
    if owner_id is None or owner_id != principal.subject:
        # Deliberately do not reveal that another user's analysis exists.
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return analysis
