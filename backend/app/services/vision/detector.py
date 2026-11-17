"""
TruthLens — Vision service (Object Detection).

Uses Ultralytics YOLO for contextual object detection.
This is for scene understanding — NOT for fake detection.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from backend.app.core.logging import get_logger
from backend.app.config import settings

log = get_logger("vision")

YOLO_MODEL_NAME = settings.VISION_MODEL


@dataclass
class DetectedObject:
    """A single detected object."""
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] in original image pixel coordinates


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
    status: str = "completed"
    image_width: Optional[int] = None
    image_height: Optional[int] = None

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
            "status": self.status,
            "image_width": self.image_width,
            "image_height": self.image_height,
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
        self._device: Optional[str] = None
        self._load_warning = "Object detection model is not loaded."
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, device: str = "auto") -> None:
        """Load YOLO model."""
        with self._lock:
            self._load(device)

    def _load(self, device: str = "auto") -> None:
        if self._loaded:
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            log.warning("ultralytics not installed. Object detection unavailable.")
            self._load_warning = "Ultralytics is not installed."
            return

        log.info("Loading YOLO model: %s", YOLO_MODEL_NAME)
        t0 = time.time()

        try:
            self._model = YOLO(YOLO_MODEL_NAME)
            self._device = None if device == "auto" else device
        except Exception as exc:
            self._model = None
            self._load_warning = f"Object detector could not load ({type(exc).__name__})."
            log.warning("Object detector load failed (%s)", type(exc).__name__)
            return

        elapsed = int((time.time() - t0) * 1000)
        log.info("YOLO model loaded in %d ms", elapsed)
        self._loaded = True

    def unload(self) -> None:
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
            self._loaded = False
            self._device = None
        log.info("YOLO model unloaded")

    def detect(self, image: Image.Image, conf_threshold: float = 0.3) -> VisionResult:
        """Run object detection on a PIL Image."""
        with self._lock:
            return self._detect(image, conf_threshold)

    def _detect(self, image: Image.Image, conf_threshold: float = 0.3) -> VisionResult:
        if not self._loaded or self._model is None:
            return VisionResult(
                status="unavailable",
                model_name=YOLO_MODEL_NAME,
                warnings=[self._load_warning],
            )

        t0 = time.time()
        warnings: List[str] = []

        try:
            if not 0 <= conf_threshold <= 1:
                raise ValueError("Confidence threshold must be between 0 and 1")
            # Ultralytics maps returned xyxy boxes back into this original PIL image's coordinates.
            results = self._model.predict(source=image.convert("RGB"), verbose=False, conf=conf_threshold, device=self._device)
        except Exception as e:
            log.error("Object detection failed: %s", e)
            return VisionResult(
                status="failed",
                runtime_ms=int((time.time() - t0) * 1000),
                warnings=[f"Detection failed: {e}"],
            )

        objects: List[DetectedObject] = []
        class_set = set()
        has_people = False

        try:
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    class_name = result.names.get(cls_id, f"class_{cls_id}")
                    if not class_name or not np.isfinite(conf) or not 0 <= conf <= 1 or len(xyxy) != 4 or not np.isfinite(xyxy).all():
                        raise ValueError("Object detector returned invalid class, confidence, or coordinates")
                    width, height = image.size
                    x1, y1, x2, y2 = xyxy
                    if x1 < -1 or y1 < -1 or x2 > width + 1 or y2 > height + 1 or x2 < x1 or y2 < y1:
                        raise ValueError("Object detector returned a box outside original image coordinates")

                    objects.append(DetectedObject(
                        class_name=class_name,
                        confidence=conf,
                    bbox=[round(float(c), 2) for c in xyxy],
                    ))
                    class_set.add(class_name)
                    if class_name == "person":
                        has_people = True
        except Exception as e:
            log.warning("Object detector output invalid (%s)", type(e).__name__)
            return VisionResult(status="failed", runtime_ms=int((time.time()-t0)*1000), model_name=YOLO_MODEL_NAME, warnings=[f"Detection output invalid ({type(e).__name__})."])

        runtime_ms = int((time.time() - t0) * 1000)

        return VisionResult(
            objects=objects,
            object_count=len(objects),
            image_width=image.width,
            image_height=image.height,
            unique_classes=sorted(class_set),
            has_people=has_people,
            runtime_ms=runtime_ms,
            model_name=YOLO_MODEL_NAME,
            warnings=warnings,
            status="completed",
        )


# Module-level singleton
_detector: Optional[ObjectDetector] = None


def get_object_detector() -> ObjectDetector:
    global _detector
    if _detector is None:
        _detector = ObjectDetector()
    return _detector
