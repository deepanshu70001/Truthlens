"""
TruthLens — Metadata extraction service.

Extracts EXIF, camera info, software/editor, timestamps, GPS availability.
GPS coordinates are never exposed publicly by default (privacy).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PIL import Image, ExifTags
from PIL.ExifTags import GPSTAGS

from backend.app.core.logging import get_logger

log = get_logger("metadata")


@dataclass
class MetadataResult:
    """Structured metadata extracted from an image."""
    has_exif: bool = False
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    software: Optional[str] = None
    datetime_original: Optional[str] = None
    datetime_digitized: Optional[str] = None
    datetime_modified: Optional[str] = None
    gps_available: bool = False
    # GPS stored internally but not exposed by default
    _gps_data: Dict[str, Any] = field(default_factory=dict, repr=False)
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    color_space: Optional[str] = None
    bits_per_sample: Optional[int] = None
    compression: Optional[str] = None
    orientation: Optional[int] = None
    x_resolution: Optional[float] = None
    y_resolution: Optional[float] = None
    all_tags: Dict[str, str] = field(default_factory=dict)
    anomalies: List[str] = field(default_factory=list)


def _safe_str(val: Any) -> str:
    """Convert an EXIF value to a safe string representation."""
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8", errors="replace").strip("\x00")
        except Exception:
            return repr(val)
    return str(val)


def _extract_gps(exif_data: dict) -> Dict[str, Any]:
    """Extract GPS IFD data if present."""
    gps_info = exif_data.get("GPSInfo")
    if not gps_info:
        return {}
    result = {}
    for key, val in gps_info.items():
        tag_name = GPSTAGS.get(key, str(key))
        result[tag_name] = val
    return result


def extract_metadata(image_data: bytes) -> MetadataResult:
    """
    Extract metadata from image bytes.

    Returns structured MetadataResult. GPS data is extracted but flagged
    as available without exposing exact coordinates publicly.
    """
    result = MetadataResult()

    try:
        img = Image.open(io.BytesIO(image_data))
    except Exception as e:
        log.warning("Cannot open image for metadata extraction: %s", e)
        return result

    result.image_width, result.image_height = img.size

    # Get EXIF
    exif = img.getexif()
    if not exif:
        result.has_exif = False
        result.anomalies.append("No EXIF metadata is present; this is common after many image-sharing or editing workflows and is not evidence of authenticity or manipulation.")
        return result

    result.has_exif = True

    # Build human-readable tag map
    tag_map: Dict[str, str] = {}
    for tag_id, value in exif.items():
        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
        tag_map[tag_name] = _safe_str(value)

    result.all_tags = tag_map

    # Extract specific fields
    result.camera_make = tag_map.get("Make")
    result.camera_model = tag_map.get("Model")
    result.software = tag_map.get("Software")
    result.datetime_original = tag_map.get("DateTimeOriginal")
    result.datetime_digitized = tag_map.get("DateTimeDigitized")
    result.datetime_modified = tag_map.get("DateTime")
    result.color_space = tag_map.get("ColorSpace")
    result.compression = tag_map.get("Compression")

    if "Orientation" in tag_map:
        try:
            result.orientation = int(tag_map["Orientation"])
        except (ValueError, TypeError):
            pass

    if "XResolution" in tag_map:
        try:
            result.x_resolution = float(tag_map["XResolution"])
        except (ValueError, TypeError):
            pass

    if "YResolution" in tag_map:
        try:
            result.y_resolution = float(tag_map["YResolution"])
        except (ValueError, TypeError):
            pass

    # GPS
    gps_data = _extract_gps({"GPSInfo": exif.get_ifd(ExifTags.IFD.GPSInfo)})
    if gps_data:
        result.gps_available = True
        result._gps_data = gps_data

    # ── Anomaly detection (informational, NOT proof) ──────────

    # Check for editing software
    editing_software = [
        "photoshop", "gimp", "lightroom", "snapseed", "pixlr",
        "canva", "paint.net", "affinity", "corel",
    ]
    if result.software:
        sw_lower = result.software.lower()
        for editor in editing_software:
            if editor in sw_lower:
                result.anomalies.append(
                    f"Image appears to have been processed with '{result.software}' "
                    f"(this is informational, not proof of manipulation)"
                )
                break

    # Missing camera info when EXIF is present
    if result.has_exif and not result.camera_make and not result.camera_model:
        result.anomalies.append(
            "EXIF data present but camera make/model is missing "
            "(may indicate re-processing or synthetic origin)"
        )

    # Timestamp inconsistencies
    if (
        result.datetime_original
        and result.datetime_modified
        and result.datetime_original != result.datetime_modified
    ):
        result.anomalies.append(
            "Original and modified timestamps differ "
            "(may indicate post-processing)"
        )

    log.info(
        "Metadata inspected: exif=%s camera_present=%s software_present=%s gps_present=%s anomalies=%d",
        result.has_exif,
        bool(result.camera_model),
        bool(result.software),
        result.gps_available,
        len(result.anomalies),
    )

    return result
