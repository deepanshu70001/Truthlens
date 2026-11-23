"""
TruthLens Backend — Centralized Configuration

All settings are loaded from environment variables (.env file).
Never hard-code secrets.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings


def _project_root() -> Path:
    """Return the project root (two levels up from this file)."""
    return Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env."""

    # ── Application ──────────────────────────────────────────────
    APP_NAME: str = "TruthLens"
    APP_ENV: str = "development"
    APP_DEBUG: bool = False
    APP_HOST: str = "127.0.0.1"
    APP_PORT: int = 8000
    APP_VERSION: str = "0.1.0"
    SECRET_KEY: str = ""
    DEV_AUTH_BYPASS: bool = False
    AUTH_USERS_JSON: str = "{}"
    ACCESS_TOKEN_TTL_SECONDS: int = 3600
    TRUSTED_HOSTS: str = "localhost,127.0.0.1,testserver"

    # ── Database ─────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./truthlens.db"
    DATABASE_ECHO: bool = False
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_POOL_TIMEOUT: float = 30.0
    DATABASE_POOL_RECYCLE_SECONDS: int = 1800

    # ── Storage ──────────────────────────────────────────────────
    STORAGE_PATH: str = str(_project_root() / "data")
    TEMP_UPLOAD_MAX_AGE_SECONDS: int = 86400
    MAX_UPLOAD_SIZE_MB: int = 20
    MAX_IMAGE_PIXELS: int = 40_000_000
    ALLOWED_MIME_TYPES: str = "image/jpeg,image/png,image/webp,image/gif,image/bmp,image/tiff"

    # ── ML / Models ──────────────────────────────────────────────
    MODEL_CACHE_DIR: str = str(_project_root() / "models" / "cache")
    MODEL_CONFIG_PATH: str = str(_project_root() / "models" / "configs")
    DEVICE: str = "auto"
    HALF_PRECISION: bool = False
    AUTHENTICITY_BACKEND: str = "finetuned_siglip"
    AUTHENTICITY_BACKBONE_MODEL: str = "google/siglip-base-patch16-224"
    AUTHENTICITY_BACKBONE_REVISION: str = ""
    AUTHENTICITY_CHECKPOINT_PATH: str = str(_project_root() / "models" / "baselines" / "siglip_frozen_head.pt")
    AUTHENTICITY_MODEL: str = "prithivMLmods/Deepfake-Real-Class-Siglip2"
    AUTHENTICITY_MODEL_REVISION: str = ""
    AUTHENTICITY_CALIBRATION_PATH: str = ""
    VISION_MODEL: str = "models/baselines/yolov8n.pt"
    VISION_TIMEOUT_SECONDS: float = 30.0
    XAI_TIMEOUT_SECONDS: float = 30.0
    OCR_LANGUAGES: str = "en,hi"
    OCR_MODEL_DIR: str = str(_project_root() / "models" / "ocr")
    OCR_TIMEOUT_SECONDS: float = 60.0
    LANGUAGE_DETECTION_TIMEOUT_SECONDS: float = 30.0
    AI_GENERATED_THRESHOLD: float = 0.65
    AUTHENTIC_THRESHOLD: float = 0.35

    # ── Hugging Face ─────────────────────────────────────────────
    HF_TOKEN: str = ""

    # ── External APIs ────────────────────────────────────────────
    GOOGLE_API_KEY: str = ""
    GOOGLE_CSE_ID: str = ""
    GOOGLE_APPLICATION_CREDENTIALS: str = ""
    FACTCHECK_API_KEY: str = ""
    CLAIMBUSTER_API_KEY: str = ""
    FACTCHECK_API_TIMEOUT_SECONDS: float = 10.0
    FACTCHECK_MAX_RESPONSE_BYTES: int = 1000000
    FACTCHECK_CACHE_ENABLED: bool = True
    FACTCHECK_CACHE_TTL_SECONDS: int = 86400
    FACTCHECK_CACHE_MAX_ENTRIES: int = 5000
    PROVENANCE_API_TIMEOUT_SECONDS: float = 30.0
    PROVENANCE_RETRY_ATTEMPTS: int = 1
    PROVENANCE_MAX_TOTAL_SECONDS: float = 35.0
    C2PA_TRUST_ANCHORS_PATH: str = ""
    FORENSIC_TIMEOUT_SECONDS: float = 30.0
    FACTCHECK_MAX_CLAIMS: int = 5
    FACTCHECK_MAX_RESULTS_PER_CLAIM: int = 5
    FACTCHECK_MAX_TOTAL_SECONDS: float = 25.0
    FACTCHECK_RETRY_ATTEMPTS: int = 0

    # ── CORS ─────────────────────────────────────────────────────
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # ── Logging ──────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    # ── Rate Limiting ────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 30
    RATE_LIMIT_ANALYSIS_PER_MINUTE: int = 4
    RATE_LIMIT_FACTCHECK_PER_MINUTE: int = 10
    RATE_LIMIT_PROVENANCE_PER_MINUTE: int = 4
    RATE_LIMIT_AUTH_PER_MINUTE: int = 10
    RATE_LIMIT_RETRIEVAL_PER_MINUTE: int = 120
    RATE_LIMIT_MODELS_PER_MINUTE: int = 60
    RATE_LIMIT_OCR_PER_MINUTE: int = 20
    RATE_LIMIT_CLAIMS_PER_MINUTE: int = 30
    RATE_LIMIT_LANGUAGE_PER_MINUTE: int = 60
    MAX_CLAIM_TEXT_LENGTH: int = 10000

    # Durable job lifecycle. A single global worker lease serializes model/GPU work.
    ANALYSIS_QUEUE_MAX_OUTSTANDING: int = 25
    ANALYSIS_JOB_MAX_ATTEMPTS: int = 3
    ANALYSIS_JOB_LEASE_SECONDS: int = 120
    ANALYSIS_JOB_HEARTBEAT_SECONDS: int = 20
    ANALYSIS_JOB_POLL_SECONDS: float = 1.0
    ANALYSIS_RETRY_BASE_SECONDS: int = 2
    ANALYSIS_RETRY_MAX_SECONDS: int = 30

    @field_validator("AI_GENERATED_THRESHOLD", "AUTHENTIC_THRESHOLD")
    @classmethod
    def validate_detector_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Detector decision thresholds must be between 0 and 1.")
        return value

    # ── Helpers ──────────────────────────────────────────────────

    @property
    def allowed_mime_list(self) -> List[str]:
        return [m.strip() for m in self.ALLOWED_MIME_TYPES.split(",") if m.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def trusted_host_list(self) -> List[str]:
        return [host.strip() for host in self.TRUSTED_HOSTS.split(",") if host.strip()]

    @property
    def auth_users(self) -> dict[str, str]:
        try:
            users = json.loads(self.AUTH_USERS_JSON)
        except (TypeError, json.JSONDecodeError):
            return {}
        if not isinstance(users, dict):
            return {}
        return {
            str(user_id): str(password_hash)
            for user_id, password_hash in users.items()
            if isinstance(user_id, str) and isinstance(password_hash, str)
        }

    def validate_runtime_security(self) -> None:
        """Fail closed on unsafe production auth, host, CORS, or DB settings."""
        env = self.APP_ENV.strip().lower()
        if self.DEV_AUTH_BYPASS and env not in {"development", "test"}:
            raise ValueError("Development authentication bypass is forbidden outside development/test.")
        if not self.trusted_host_list or any(host == "*" for host in self.trusted_host_list):
            raise ValueError("TRUSTED_HOSTS must contain explicit host names.")
        if env != "production":
            return
        if self.APP_DEBUG:
            raise ValueError("APP_DEBUG must be false in production.")
        if len(self.SECRET_KEY.encode("utf-8")) < 32 or self.SECRET_KEY.strip().lower() in {
            "change_me", "change_me_to_a_random_secret", "secret", ""
        }:
            raise ValueError("Production requires a non-placeholder SECRET_KEY of at least 32 bytes.")
        if self.DEV_AUTH_BYPASS:
            raise ValueError("DEV_AUTH_BYPASS must be disabled in production.")
        if not self.auth_users or any(not _valid_password_hash(value) for value in self.auth_users.values()):
            raise ValueError("Production AUTH_USERS_JSON must contain pre-provisioned scrypt password hashes.")
        if not self.cors_origin_list or "*" in self.cors_origin_list:
            raise ValueError("Production CORS_ORIGINS must explicitly list trusted frontend origins.")
        if not self.DATABASE_URL.startswith("postgresql+asyncpg://"):
            raise ValueError("Production requires the shared PostgreSQL async driver for durable workers and rate limits.")
        if self.DATABASE_ECHO:
            raise ValueError("DATABASE_ECHO must be disabled in production.")

    @property
    def is_debug(self) -> bool:
        return self.APP_DEBUG

    @property
    def storage_dir(self) -> Path:
        return Path(self.STORAGE_PATH)

    @property
    def originals_dir(self) -> Path:
        return self.storage_dir / "originals"

    @property
    def thumbnails_dir(self) -> Path:
        return self.storage_dir / "thumbnails"

    @property
    def processed_dir(self) -> Path:
        return self.storage_dir / "processed"

    @property
    def heatmaps_dir(self) -> Path:
        return self.storage_dir / "heatmaps"

    @property
    def reports_dir(self) -> Path:
        return self.storage_dir / "reports"

    @property
    def artifacts_dir(self) -> Path:
        return self.storage_dir / "artifacts"

    @property
    def temp_uploads_dir(self) -> Path:
        return self.storage_dir / "tmp_uploads"

    def ensure_storage_dirs(self) -> None:
        """Create all storage sub-directories if they don't exist."""
        for d in (
            self.originals_dir,
            self.thumbnails_dir,
            self.processed_dir,
            self.heatmaps_dir,
            self.reports_dir,
            self.artifacts_dir,
            self.temp_uploads_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    model_config = {
        "env_file": str(_project_root() / ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }


# Singleton instance
settings = Settings()


def _valid_password_hash(value: str) -> bool:
    parts = value.split("$")
    if len(parts) != 6 or parts[0] != "scrypt":
        return False
    try:
        n, r, p = (int(item) for item in parts[1:4])
        return n >= 2**14 and r >= 8 and p >= 1 and len(parts[4]) >= 16 and len(parts[5]) >= 32
    except ValueError:
        return False
