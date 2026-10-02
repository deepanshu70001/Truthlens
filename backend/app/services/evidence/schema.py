"""
TruthLens — Evidence schema.

Defines a consistent evidence structure used across the entire system.
Every piece of evidence — from any source — is wrapped in this format.

Evidence categories:
- model_evidence      (ML model predictions)
- forensic_evidence   (classical image forensic signals)
- ocr_evidence        (text extracted from image)
- contextual_evidence (VLM/object detection observations)
- metadata_evidence   (EXIF/metadata signals)
- factcheck_evidence  (external fact-check API results)
- provenance_evidence (web detection / reverse image search)
- xai_evidence        (explainable AI attribution maps)
- hash_evidence       (image hash / duplicate detection)

Strength levels:
- low:    Weak signal, may be noise, should not influence assessment alone
- medium: Moderate signal, relevant but not conclusive
- high:   Strong signal, significant influence on assessment

Evidence items do NOT determine the final assessment. They are inputs
to the fusion engine, which makes the final determination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Valid evidence types
EVIDENCE_TYPES = {
    "model_evidence",
    "forensic_evidence",
    "ocr_evidence",
    "contextual_evidence",
    "metadata_evidence",
    "factcheck_evidence",
    "provenance_evidence",
    "xai_evidence",
    "hash_evidence",
}

# Valid strength levels
STRENGTH_LEVELS = {"low", "medium", "high"}


@dataclass
class EvidenceItem:
    """
    A single piece of evidence from any source.

    Every evidence item has:
    - type: category of evidence
    - source: which module produced it
    - strength: how significant this signal is
    - direction: what this evidence suggests
    - description: human-readable explanation
    - data: raw data backing the evidence
    """
    type: str  # one of EVIDENCE_TYPES
    source: str  # e.g., "authenticity_detector", "forensics.ela", "google_factcheck"
    strength: str  # low | medium | high
    direction: str  # authentic | synthetic | manipulated | neutral | informational
    description: str
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None  # Model confidence if available
    calibrated: bool = False  # Whether the confidence is calibrated

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": self.type,
            "source": self.source,
            "strength": self.strength,
            "direction": self.direction,
            "description": self.description,
            "data": self.data,
        }
        if self.confidence is not None:
            result["confidence"] = round(self.confidence, 4)
            result["calibrated"] = self.calibrated
            if not self.calibrated:
                result["confidence_note"] = (
                    "This confidence score is uncalibrated — it reflects "
                    "the model's raw output, not a scientifically validated probability."
                )
        return result


@dataclass
class EvidenceCollection:
    """
    A collection of all evidence items for an analysis.
    Organized by category for easy access.
    """
    items: List[EvidenceItem] = field(default_factory=list)

    def add(self, item: EvidenceItem) -> None:
        """Add an evidence item to the collection."""
        self.items.append(item)

    def get_by_type(self, evidence_type: str) -> List[EvidenceItem]:
        """Get all evidence items of a given type."""
        return [i for i in self.items if i.type == evidence_type]

    def get_by_direction(self, direction: str) -> List[EvidenceItem]:
        """Get all evidence items pointing in a given direction."""
        return [i for i in self.items if i.direction == direction]

    def get_strong_evidence(self) -> List[EvidenceItem]:
        """Get all high-strength evidence items."""
        return [i for i in self.items if i.strength == "high"]

    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def has_conflicting_evidence(self) -> bool:
        """Check if there is conflicting evidence (both authentic and synthetic/manipulated)."""
        directions = {i.direction for i in self.items if i.strength in ("medium", "high")}
        has_authentic = "authentic" in directions
        has_synthetic = "synthetic" in directions or "manipulated" in directions
        return has_authentic and has_synthetic

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [i.to_dict() for i in self.items],
            "total_count": self.count,
            "by_type": {
                t: len(self.get_by_type(t))
                for t in EVIDENCE_TYPES
                if self.get_by_type(t)
            },
            "has_conflicting_evidence": self.has_conflicting_evidence,
        }


def build_evidence_from_authenticity(auth_result) -> List[EvidenceItem]:
    """Convert authenticity detection result into evidence items."""
    if auth_result is None:
        return []

    items = []

    # Primary model prediction
    if auth_result.confidence >= 0.7:
        strength = "high"
    elif auth_result.confidence >= 0.5:
        strength = "medium"
    else:
        strength = "low"

    direction = "authentic" if auth_result.label == "likely_authentic" else "synthetic"
    if auth_result.label == "inconclusive":
        direction = "neutral"

    items.append(EvidenceItem(
        type="model_evidence",
        source="authenticity_detector",
        strength=strength,
        direction=direction,
        description=(
            f"Primary authenticity model predicts '{auth_result.label}' "
            f"with {auth_result.confidence:.1%} confidence."
        ),
        data={
            "label": auth_result.label,
            "ai_generated_score": auth_result.ai_generated_score,
            "authentic_score": auth_result.authentic_score,
            "model": auth_result.model_name,
        },
        confidence=auth_result.confidence,
        calibrated=False,
    ))

    return items


def build_evidence_from_forensics(forensics_result) -> List[EvidenceItem]:
    """Convert forensic analysis result into evidence items."""
    if forensics_result is None:
        return []

    items = []

    for signal in forensics_result.signals:
        if signal.score >= 0.6:
            strength = "medium"
            direction = "manipulated"
        elif signal.score >= 0.3:
            strength = "low"
            direction = "neutral"
        else:
            strength = "low"
            direction = "authentic"

        items.append(EvidenceItem(
            type="forensic_evidence",
            source=f"forensics.{signal.name}",
            strength=strength,
            direction=direction,
            description=signal.interpretation,
            data=signal.details,
        ))

    return items


def build_evidence_from_ocr(ocr_result, text_result=None) -> List[EvidenceItem]:
    """Convert OCR result into evidence items."""
    if ocr_result is None:
        return []

    items = []

    if ocr_result.has_text:
        items.append(EvidenceItem(
            type="ocr_evidence",
            source="easyocr",
            strength="medium" if ocr_result.word_count >= 5 else "low",
            direction="informational",
            description=(
                f"OCR extracted {ocr_result.word_count} words "
                f"({ocr_result.language} detected) with "
                f"{ocr_result.total_confidence:.1%} average confidence."
            ),
            data={
                "word_count": ocr_result.word_count,
                "language": ocr_result.language,
                "confidence": ocr_result.total_confidence,
                "text_preview": ocr_result.normalized_text[:200] if ocr_result.normalized_text else "",
            },
        ))
    else:
        items.append(EvidenceItem(
            type="ocr_evidence",
            source="easyocr",
            strength="low",
            direction="informational",
            description="No text detected in the image.",
            data={"word_count": 0},
        ))

    return items


def build_evidence_from_metadata(metadata_result) -> List[EvidenceItem]:
    """Convert metadata result into evidence items."""
    if metadata_result is None:
        return []

    items = []

    if not metadata_result.has_exif:
        items.append(EvidenceItem(
            type="metadata_evidence",
            source="exif_extractor",
            strength="low",
            direction="neutral",
            description=(
                "No EXIF metadata found. This is common in AI-generated images "
                "and images shared through social media, but is not conclusive."
            ),
            data={"has_exif": False},
        ))
    else:
        for anomaly in metadata_result.anomalies:
            items.append(EvidenceItem(
                type="metadata_evidence",
                source="exif_extractor",
                strength="low",
                direction="neutral",
                description=anomaly,
                data={
                    "has_exif": True,
                    "camera": metadata_result.camera_model,
                    "software": metadata_result.software,
                },
            ))

        if not metadata_result.anomalies:
            items.append(EvidenceItem(
                type="metadata_evidence",
                source="exif_extractor",
                strength="low",
                direction="authentic",
                description=(
                    f"EXIF metadata present with camera "
                    f"{metadata_result.camera_model or 'unknown'}. "
                    f"No anomalies detected."
                ),
                data={
                    "has_exif": True,
                    "camera": metadata_result.camera_model,
                    "software": metadata_result.software,
                    "gps_available": metadata_result.gps_available,
                },
            ))

    return items


def build_evidence_from_factcheck(factcheck_result) -> List[EvidenceItem]:
    """Convert fact-check result into evidence items."""
    if factcheck_result is None:
        return []

    items = []

    if factcheck_result.status == "unavailable":
        items.append(EvidenceItem(
            type="factcheck_evidence",
            source="google_factcheck_api",
            strength="low",
            direction="informational",
            description="Fact-check API not configured. No fact-check data available.",
            data={"status": "unavailable"},
        ))
    elif factcheck_result.status == "failed":
        items.append(EvidenceItem(
            type="factcheck_evidence",
            source="google_factcheck_api",
            strength="low",
            direction="informational",
            description=f"Fact-check API failed: {factcheck_result.error}",
            data={"status": "failed", "error": factcheck_result.error},
        ))
    elif factcheck_result.status == "no_results":
        items.append(EvidenceItem(
            type="factcheck_evidence",
            source="google_factcheck_api",
            strength="low",
            direction="informational",
            description="No matching fact-checks found for extracted claims.",
            data={"status": "no_results", "queries": factcheck_result.queries_performed},
        ))
    elif factcheck_result.status == "success":
        for claim in factcheck_result.claims:
            rating_lower = claim.textual_rating.lower()
            if any(w in rating_lower for w in ["false", "fake", "incorrect", "misleading", "pants on fire"]):
                direction = "manipulated"
                strength = "high"
            elif any(w in rating_lower for w in ["true", "correct", "accurate"]):
                direction = "authentic"
                strength = "high"
            else:
                direction = "neutral"
                strength = "medium"

            items.append(EvidenceItem(
                type="factcheck_evidence",
                source=f"factcheck.{claim.publisher_name or 'unknown'}",
                strength=strength,
                direction=direction,
                description=(
                    f"Fact-check by {claim.publisher_name}: "
                    f"'{claim.claimed_text[:100]}' rated '{claim.textual_rating}'."
                ),
                data=claim.to_dict(),
            ))

    return items


def build_evidence_from_provenance(provenance_result) -> List[EvidenceItem]:
    """Convert provenance result into evidence items."""
    if provenance_result is None:
        return []

    items = []

    if provenance_result.status == "unavailable":
        items.append(EvidenceItem(
            type="provenance_evidence",
            source="google_vision_web_detection",
            strength="low",
            direction="informational",
            description="Web detection API not configured. No provenance data available.",
            data={"status": "unavailable"},
        ))
    elif provenance_result.status == "failed":
        items.append(EvidenceItem(
            type="provenance_evidence",
            source="google_vision_web_detection",
            strength="low",
            direction="informational",
            description=f"Web detection failed: {provenance_result.error}",
            data={"status": "failed"},
        ))
    elif provenance_result.status == "success":
        if provenance_result.matching_images:
            exact_count = sum(1 for i in provenance_result.matching_images if i.match_type == "exact")
            partial_count = sum(1 for i in provenance_result.matching_images if i.match_type == "partial")
            items.append(EvidenceItem(
                type="provenance_evidence",
                source="google_vision_web_detection",
                strength="medium",
                direction="informational",
                description=(
                    f"Found {exact_count} exact and {partial_count} partial "
                    f"image matches across the web."
                ),
                data={
                    "exact_matches": exact_count,
                    "partial_matches": partial_count,
                    "pages": len(provenance_result.matching_pages),
                },
            ))

        if provenance_result.best_guess_labels:
            items.append(EvidenceItem(
                type="provenance_evidence",
                source="google_vision_web_detection",
                strength="low",
                direction="informational",
                description=(
                    f"Web detection suggests: "
                    f"{', '.join(provenance_result.best_guess_labels)}"
                ),
                data={"labels": provenance_result.best_guess_labels},
            ))
    elif provenance_result.status == "no_results":
        items.append(EvidenceItem(
            type="provenance_evidence",
            source="google_vision_web_detection",
            strength="low",
            direction="informational",
            description="No web matches found for this image.",
            data={"status": "no_results"},
        ))

    return items


def build_evidence_from_xai(xai_result) -> List[EvidenceItem]:
    """Convert XAI result into evidence items."""
    if xai_result is None:
        return []

    items = []

    if xai_result.heatmap_path:
        items.append(EvidenceItem(
            type="xai_evidence",
            source="grad_cam",
            strength="medium",
            direction="informational",
            description=(
                f"Grad-CAM attribution map generated for class '{xai_result.target_class}'. "
                f"{len(xai_result.suspicious_regions)} high-activation regions identified."
            ),
            data={
                "method": xai_result.method,
                "target_class": xai_result.target_class,
                "target_score": xai_result.target_score,
                "suspicious_regions": xai_result.suspicious_regions,
                "heatmap_path": xai_result.heatmap_path,
                "overlay_path": xai_result.overlay_path,
            },
        ))

    return items


def build_evidence_from_vision(vision_result) -> List[EvidenceItem]:
    """Convert vision/object detection result into evidence items."""
    if vision_result is None:
        return []

    items = []

    if vision_result.objects:
        items.append(EvidenceItem(
            type="contextual_evidence",
            source="yolov8_object_detection",
            strength="low",
            direction="informational",
            description=(
                f"Detected {vision_result.object_count} objects: "
                f"{', '.join(vision_result.unique_classes[:10])}."
            ),
            data={
                "object_count": vision_result.object_count,
                "unique_classes": vision_result.unique_classes,
                "has_people": vision_result.has_people,
            },
        ))

    return items
