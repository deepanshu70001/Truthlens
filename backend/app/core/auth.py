"""Small pre-provisioned-user authentication boundary for the current product."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.app.config import settings

_bearer = HTTPBearer(auto_error=False)
_ISSUER = "truthlens"
_AUDIENCE = "truthlens-api"


@dataclass(frozen=True)
class Principal:
    subject: str
    development_bypass: bool = False


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """Create a salted scrypt hash suitable for AUTH_USERS_JSON provisioning."""
    if len(password) < 12:
        raise ValueError("Provisioned passwords must contain at least 12 characters.")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt$16384$8$1${_b64encode(salt)}${_b64encode(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, n_raw, r_raw, p_raw, salt_raw, digest_raw = encoded.split("$")
        n, r, p = int(n_raw), int(r_raw), int(p_raw)
        if scheme != "scrypt" or not (2**14 <= n <= 2**18 and 8 <= r <= 16 and 1 <= p <= 4):
            return False
        expected = _b64decode(digest_raw)
        actual = hashlib.scrypt(
            password.encode("utf-8"), salt=_b64decode(salt_raw), n=n, r=r, p=p, dklen=len(expected)
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError, UnicodeError):
        return False


def issue_access_token(subject: str, *, now: int | None = None) -> tuple[str, int]:
    if len(settings.SECRET_KEY.encode("utf-8")) < 32:
        raise ValueError("Token signing is not configured.")
    issued = int(time.time()) if now is None else now
    expires = issued + settings.ACCESS_TOKEN_TTL_SECONDS
    header = {"alg": "HS256", "typ": "JWT"}
    claims = {
        "iss": _ISSUER,
        "aud": _AUDIENCE,
        "sub": subject,
        "iat": issued,
        "exp": expires,
        "jti": secrets.token_urlsafe(18),
    }
    signing_input = f"{_b64encode(json.dumps(header, separators=(',', ':')).encode())}.{_b64encode(json.dumps(claims, separators=(',', ':')).encode())}"
    signature = hmac.new(settings.SECRET_KEY.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64encode(signature)}", expires


def decode_access_token(token: str, *, now: int | None = None) -> Principal:
    try:
        header_raw, claims_raw, signature_raw = token.split(".")
        signing_input = f"{header_raw}.{claims_raw}"
        expected = hmac.new(settings.SECRET_KEY.encode(), signing_input.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64decode(signature_raw)):
            raise ValueError("signature")
        header = json.loads(_b64decode(header_raw))
        claims: dict[str, Any] = json.loads(_b64decode(claims_raw))
        current = int(time.time()) if now is None else now
        if header != {"alg": "HS256", "typ": "JWT"}:
            raise ValueError("algorithm")
        if claims.get("iss") != _ISSUER or claims.get("aud") != _AUDIENCE:
            raise ValueError("issuer")
        if not isinstance(claims.get("sub"), str) or not claims["sub"]:
            raise ValueError("subject")
        if not isinstance(claims.get("exp"), int) or claims["exp"] <= current:
            raise ValueError("expired")
        if not isinstance(claims.get("iat"), int) or claims["iat"] > current + 30:
            raise ValueError("issued-at")
        if not isinstance(claims.get("jti"), str):
            raise ValueError("token-id")
        return Principal(subject=claims["sub"])
    except (ValueError, TypeError, KeyError, json.JSONDecodeError, UnicodeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None


def authenticate_user(user_id: str, password: str) -> bool:
    users = settings.auth_users
    # Perform a scrypt check even when the account name is absent to reduce enumeration timing.
    encoded = users.get(user_id)
    if encoded is None:
        encoded = _DUMMY_PASSWORD_HASH
    valid = verify_password(password, encoded)
    return user_id in users and valid


async def get_current_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal:
    if settings.DEV_AUTH_BYPASS and settings.APP_ENV.lower() in {"development", "test"}:
        if credentials is None:
            return Principal(subject="local-development", development_bypass=True)
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_access_token(credentials.credentials)


_DUMMY_PASSWORD_HASH = hash_password(
    "truthlens-invalid-user-timing-placeholder", salt=b"TruthLensDummySalt"
)
