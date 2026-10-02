"""
TruthLens — VLM context analysis service.

Uses a Vision-Language Model (Qwen2.5-VL-3B-Instruct) for scene description
and contextual observations.

IMPORTANT: The VLM only describes observable evidence.
It does NOT determine the final fake/real result.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PIL import Image

from backend.app.core.logging import get_logger
from backend.app.config import settings

log = get_logger("vlm")

VLM_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# The prompt instructs the VLM to describe, not judge
VLM_PROMPT = """Analyze this image as a forensic image analyst. Describe ONLY what you can observe:

1. **Scene description**: What does this image depict?
2. **Visible objects**: What objects, people, or text are visible?
3. **Visual quality**: Note any visible artifacts, unusual patterns, or quality issues.
4. **Consistency observations**: Are there any visible inconsistencies in lighting, shadows, perspective, or proportions?
5. **Text content**: If there is text visible, what does it say?

IMPORTANT RULES:
- ONLY describe what you can directly observe
- Do NOT make claims about whether the image is fake or real
- Do NOT speculate beyond what is visible
- Note observations with uncertainty where appropriate
- Be specific and factual"""


@dataclass
class VLMResult:
    """VLM context analysis result."""
    scene_description: str = ""
    observations: List[str] = field(default_factory=list)
    raw_response: str = ""
    model_name: str = ""
    runtime_ms: int = 0
    available: bool = False
    warnings: List[str] = field(default_factory=list)
    note: str = (
        "VLM observations describe what the model sees in the image. "
        "The VLM does not determine authenticity."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_description": self.scene_description,
            "observations": self.observations,
            "raw_response": self.raw_response,
            "model_name": self.model_name,
            "runtime_ms": self.runtime_ms,
            "available": self.available,
            "warnings": self.warnings,
            "note": self.note,
        }


class VLMService:
    """
    Vision-Language Model service for contextual image analysis.

    The VLM is a heavy model (~3B params, ~1.9GB VRAM quantized).
    It is loaded on-demand and unloaded after use to manage VRAM.

    Currently a stub that will be fully implemented when Qwen2.5-VL
    integration is confirmed to work within 4GB VRAM.
    """

    def __init__(self):
        self._model = None
        self._processor = None
        self._loaded = False
        self._model_id = VLM_MODEL_ID

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, device: str = "auto") -> None:
        """
        Load VLM model.

        NOTE: VLM loading is deferred to Stage 5 as it requires
        significant VRAM management and quantization setup.
        """
        log.info(
            "VLM service: model loading deferred to Stage 5. "
            "Requires quantization setup for 4GB VRAM constraint."
        )
        # Full implementation in Stage 5
        self._loaded = False

    def unload(self) -> None:
        """Unload VLM to free VRAM."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._loaded = False
        log.info("VLM unloaded")

    def analyze(self, image: Image.Image) -> VLMResult:
        """
        Analyze an image with the VLM.

        Returns VLM observations as contextual evidence.
        The VLM does NOT determine authenticity.
        """
        if not self._loaded:
            return VLMResult(
                available=False,
                model_name=self._model_id,
                warnings=[
                    "VLM context analysis is not yet available. "
                    "Full implementation requires Stage 5 (quantized Qwen2.5-VL)."
                ],
            )

        # Full implementation in Stage 5
        return VLMResult(
            available=False,
            model_name=self._model_id,
            warnings=["VLM not loaded."],
        )


# Module-level singleton
_vlm_service: Optional[VLMService] = None


def get_vlm_service() -> VLMService:
    global _vlm_service
    if _vlm_service is None:
        _vlm_service = VLMService()
    return _vlm_service
