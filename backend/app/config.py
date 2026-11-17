"""
TruthLens Backend — Centralized Configuration

All settings are loaded from environment variables (.env file).
Never hard-code secrets.
"""

from __future__ import annotations

import os
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
    APP_DEBUG: bool = True
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_VERSION: str = "0.1.0"
    SECRET_KEY: str = "CHANGE_ME_TO_A_RANDOM_SECRET"

    # ── Database ─────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./truthlens.db"
    DATABASE_ECHO: bool = False

    # ── Storage ──────────────────────────────────────────────────
    STORAGE_PATH: str = str(_project_root() / "data")
    MAX_UPLOAD_SIZE_MB: int = 20
    MAX_IMAGE_PIXELS: int = 40_000_000
    ALLOWED_MIME_TYPES: str = "image/jpeg,image/png,image/webp,image/gif,image/bmp,image/tiff"

    # ── ML / Models ──────────────────────────────────────────────
    MODEL_CACHE_DIR: str = str(_project_root() / "models" / "cache")
    MODEL_CONFIG_PATH: str = str(_project_root() / "models" / "configs")
    DEVICE: str = "auto"
    HALF_PRECISION: bool = False
    AUTHENTICITY_MODEL: str = "prithivMLmods/Deepfake-Real-Class-Siglip2"
    VISION_MODEL: str = "yolov8n.pt"
    OCR_LANGUAGES: str = "en,hi"
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
    PROVENANCE_API_TIMEOUT_SECONDS: float = 30.0
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

    def ensure_storage_dirs(self) -> None:
        """Create all storage sub-directories if they don't exist."""
        for d in (
            self.originals_dir,
            self.thumbnails_dir,
            self.processed_dir,
            self.heatmaps_dir,
            self.reports_dir,
            self.artifacts_dir,
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
