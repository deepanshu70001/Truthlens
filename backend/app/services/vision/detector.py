"""
TruthLens — Vision service (Object Detection).

Uses Ultralytics YOLO for contextual object detection.
This is for scene understanding — NOT for fake detection.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from backend.app.core.logging import get_logger
from backend.app.config import settings

log = get_logger("vision")

YOLO_MODEL_NAME = "yolov8n.pt"  # Nano model — fast, ~6MB, good enough for context


@dataclass
class DetectedObject:
    """A single detected object."""
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]


@dataclass
class VisionResult:
    """Object detection result."""
    objects: List[DetectedObject] = field(default_factory=list)
    object_count: int = 0
    unique_classes: List[str] = field(default_factory=list)
    has_people: bool = False
    has_text_objects: bool = False
    runtime_ms: int = 0
    model_name: str = ""
    warnings: List[str] = field(default_factory=list)
    note: str = "Object detection provides scene context, not manipulation evidence."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objects": [
                {
                    "class_name": o.class_name,
                    "confidence": round(o.confidence, 4),
                    "bbox": o.bbox,
                }
                for o in self.objects
            ],
            "object_count": self.object_count,
            "unique_classes": self.unique_classes,
            "has_people": self.has_people,
            "has_text_objects": self.has_text_objects,
            "runtime_ms": self.runtime_ms,
            "model_name": self.model_name,
            "note": self.note,
            "warnings": self.warnings,
        }


class ObjectDetector:
    """YOLO-based object detector for scene understanding."""

    def __init__(self):
        self._model = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, device: str = "auto") -> None:
        """Load YOLO model."""
        if self._loaded:
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            log.warning("ultralytics not installed. Object detection unavailable.")
            return

        log.info("Loading YOLO model: %s", YOLO_MODEL_NAME)
        t0 = time.time()

        self._model = YOLO(YOLO_MODEL_NAME)

        elapsed = int((time.time() - t0) * 1000)
        log.info("YOLO model loaded in %d ms", elapsed)
        self._loaded = True

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False
        log.info("YOLO model unloaded")

    def detect(self, image: Image.Image, conf_threshold: float = 0.3) -> VisionResult:
        """Run object detection on a PIL Image."""
        if not self._loaded or self._model is None:
            return VisionResult(
                warnings=["Object detection model not loaded."],
            )

        t0 = time.time()
        warnings: List[str] = []

        try:
            results = self._model(image, verbose=False, conf=conf_threshold)
        except Exception as e:
            log.error("Object detection failed: %s", e)
            return VisionResult(
                runtime_ms=int((time.time() - t0) * 1000),
                warnings=[f"Detection failed: {e}"],
            )

        objects: List[DetectedObject] = []
        class_set = set()
        has_people = False

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                class_name = result.names.get(cls_id, f"class_{cls_id}")

                objects.append(DetectedObject(
                    class_name=class_name,
                    confidence=conf,
                    bbox=[int(c) for c in xyxy],
                ))
                class_set.add(class_name)
                if class_name == "person":
                    has_people = True

        runtime_ms = int((time.time() - t0) * 1000)

        return VisionResult(
            objects=objects,
            object_count=len(objects),
            unique_classes=sorted(class_set),
            has_people=has_people,
            runtime_ms=runtime_ms,
            model_name=YOLO_MODEL_NAME,
            warnings=warnings,
        )


# Module-level singleton
_detector: Optional[ObjectDetector] = None


def get_object_detector() -> ObjectDetector:
    global _detector
    if _detector is None:
        _detector = ObjectDetector()
    return _detector
