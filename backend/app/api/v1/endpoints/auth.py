"""Access-token issuance for pre-provisioned TruthLens users."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from backend.app.config import settings
from backend.app.core.auth import authenticate_user, issue_access_token
from backend.app.core.rate_limit import check_rate_limit

router = APIRouter()


class TokenRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=1024)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    expires_at: datetime


@router.get("/auth/mode")
async def auth_mode():
    return {"auth_required": not (
        settings.DEV_AUTH_BYPASS and settings.APP_ENV.lower() in {"development", "test"}
    )}


@router.post("/auth/token", response_model=TokenResponse)
async def create_access_token(payload: TokenRequest, request: Request):
    await check_rate_limit(
        request,
        scope="auth-token",
        limit=settings.RATE_LIMIT_AUTH_PER_MINUTE,
    )
    if not authenticate_user(payload.username, payload.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        token, expires_at = issue_access_token(payload.username)
    except ValueError:
        raise HTTPException(status_code=503, detail="Authentication is not configured.") from None
    current = int(datetime.now(timezone.utc).timestamp())
    return TokenResponse(
        access_token=token,
        expires_in=max(0, expires_at - current),
        expires_at=datetime.fromtimestamp(expires_at, tz=timezone.utc),
    )
