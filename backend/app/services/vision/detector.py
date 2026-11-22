"""
TruthLens — Vision service (Object Detection).

Uses Ultralytics YOLO for contextual object detection.
This is for scene understanding — NOT for fake detection.
"""

from __future__ import annotations

import time
import threading
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from backend.app.core.logging import get_logger
from backend.app.config import settings

log = get_logger("vision")

YOLO_MODEL_NAME = settings.VISION_MODEL


def _public_model_identity() -> str:
    """Expose the configured model filename, never a local filesystem path."""
    return Path(str(YOLO_MODEL_NAME)).name[:160] or "configured-yolo"


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
    outcome: Optional[str] = None
    model_sha256: Optional[str] = None
    coordinate_space: str = "exif_corrected_input_pixels"
    confidence_semantics: str = "YOLO detection confidence; not an authenticity probability."
    limitations: List[str] = field(default_factory=lambda: ["Object presence or absence is not proof of authenticity, generation, or manipulation."])

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
            "outcome": self.outcome or ("no_results" if self.status == "completed" and not self.objects else "success" if self.status == "completed" else self.status),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "unique_classes": self.unique_classes,
            "has_people": self.has_people,
            "has_text_objects": self.has_text_objects,
            "runtime_ms": self.runtime_ms,
            "model_name": self.model_name,
            "model_sha256": self.model_sha256,
            "coordinate_space": self.coordinate_space,
            "confidence_semantics": self.confidence_semantics,
            "limitations": self.limitations,
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
        self._model_sha256: Optional[str] = None
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

        model_path = Path(YOLO_MODEL_NAME).expanduser()
        if not model_path.is_file():
            self._load_warning = "Configured YOLO weights are not present locally; automatic model downloads are disabled."
            log.warning("YOLO weights are unavailable locally for configured model identity")
            return

        log.info("Loading configured local YOLO weights (%s)", _public_model_identity())
        t0 = time.time()

        try:
            self._model = YOLO(str(model_path.resolve()))
            self._device = None if device == "auto" else device
            digest = hashlib.sha256()
            with model_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            self._model_sha256 = digest.hexdigest()
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
                model_name=_public_model_identity(),
                outcome="unavailable",
                warnings=[self._load_warning],
            )

        t0 = time.time()
        warnings: List[str] = []

        try:
            if image.width <= 0 or image.height <= 0 or image.width * image.height > settings.MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions exceed configured detector limits")
            if not 0 <= conf_threshold <= 1:
                raise ValueError("Confidence threshold must be between 0 and 1")
            # Ultralytics maps returned xyxy boxes back into this original PIL image's coordinates.
            results = self._model.predict(source=image.convert("RGB"), verbose=False,
                conf=conf_threshold, device=self._device, imgsz=640, max_det=100)
        except Exception as e:
            log.error("Object detection inference failed (%s)", type(e).__name__)
            return VisionResult(
                status="failed",
                outcome="failed",
                runtime_ms=int((time.time() - t0) * 1000),
                model_name=_public_model_identity(), model_sha256=self._model_sha256,
                warnings=[f"Detection failed ({type(e).__name__})."],
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
                    raw_xyxy = box.xyxy[0]
                    xyxy = raw_xyxy.tolist() if hasattr(raw_xyxy, "tolist") else list(raw_xyxy)
                    names = result.names
                    class_name = names.get(cls_id, f"class_{cls_id}") if isinstance(names, dict) else names[cls_id] if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names) else f"class_{cls_id}"
                    if cls_id < 0 or not isinstance(class_name, str) or not class_name.strip() or len(class_name) > 100 or not np.isfinite(conf) or not 0 <= conf <= 1 or len(xyxy) != 4 or not np.isfinite(xyxy).all():
                        raise ValueError("Object detector returned invalid class, confidence, or coordinates")
                    width, height = image.size
                    x1, y1, x2, y2 = xyxy
                    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
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
            return VisionResult(status="failed", outcome="failed", runtime_ms=int((time.time()-t0)*1000), model_name=_public_model_identity(), model_sha256=self._model_sha256, warnings=[f"Detection output invalid ({type(e).__name__})."])

        runtime_ms = int((time.time() - t0) * 1000)

        return VisionResult(
            objects=objects,
            object_count=len(objects),
            image_width=image.width,
            image_height=image.height,
            unique_classes=sorted(class_set),
            has_people=has_people,
            runtime_ms=runtime_ms,
            model_name=_public_model_identity(),
            model_sha256=self._model_sha256,
            warnings=warnings,
            status="completed",
            outcome="success" if objects else "no_results",
        )


# Module-level singleton
_detector: Optional[ObjectDetector] = None


def get_object_detector() -> ObjectDetector:
    global _detector
    if _detector is None:
        _detector = ObjectDetector()
    return _detector
