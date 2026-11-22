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
import math
from typing import Any, Dict, List, Optional
from uuid import uuid4


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
    evidence_id: str = field(default_factory=lambda: str(uuid4()))
    analysis_id: Optional[str] = None
    stage_name: Optional[str] = None
    status: str = "completed"
    outcome: Optional[str] = None
    provenance: Optional[str] = None

    def __post_init__(self) -> None:
        if self.stage_name is None:
            self.stage_name = self.type.removesuffix("_evidence")
        if self.provenance is None:
            self.provenance = self.source

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": self.type,
            "source": self.source,
            "strength": self.strength,
            "direction": self.direction,
            "description": self.description,
            "data": self.data,
            "evidence_id": self.evidence_id,
            "analysis_id": self.analysis_id,
            "stage_name": self.stage_name,
            "status": self.status,
            "outcome": self.outcome,
            "provenance": self.provenance,
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
    analysis_id: Optional[str] = None

    def add(self, item: EvidenceItem) -> None:
        """Add an evidence item to the collection."""
        if item.analysis_id is None:
            item.analysis_id = self.analysis_id
        if any(
            existing.evidence_id == item.evidence_id
            or (existing.type, existing.source, existing.direction, existing.description)
            == (item.type, item.source, item.direction, item.description)
            for existing in self.items
        ):
            return
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
        """Only validated classifier outputs can create directional conflict."""
        directions = {
            i.direction for i in self.items
            if i.type == "model_evidence"
            and i.source == "authenticity_detector"
            and i.stage_name == "authenticity"
            and isinstance(i.status, str) and i.status in {"completed", "partial"}
            and i.calibrated
            and isinstance(i.data, dict)
            and i.data.get("calibration_status") == "validated_temperature_scaling"
            and i.data.get("domain_validation_status") == "validated_for_input_domain"
            and isinstance(i.direction, str)
            and i.direction in {"authentic", "synthetic", "manipulated"}
        }
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

    # Raw softmax magnitude is not empirical reliability and cannot determine
    # evidence strength or cast a directional vote. Direction is permitted only
    # when a reviewed calibration artifact produced an application-level label.
    calibrated_probability = getattr(auth_result, "calibrated_ai_generated_probability", None)
    calibrated = (
        getattr(auth_result, "calibration_status", "not_calibrated") == "validated_temperature_scaling"
        and getattr(auth_result, "domain_validation_status", "not_evaluated") == "validated_for_input_domain"
    )
    strength = "medium" if calibrated else "low"
    if calibrated and auth_result.label == "likely_authentic":
        direction = "authentic"
    elif calibrated and auth_result.label == "likely_ai_generated":
        direction = "synthetic"
    elif calibrated and auth_result.label == "likely_manipulated":
        direction = "manipulated"
    else:
        direction = "neutral"

    items.append(EvidenceItem(
        type="model_evidence",
        source="authenticity_detector",
        strength=strength,
        direction=direction,
        description=(
            f"Primary classifier raw argmax is '{getattr(auth_result, 'model_prediction', auth_result.label)}'; "
            f"application classification is '{auth_result.label}'. "
            "Raw softmax output is not a calibrated probability."
        ),
        data={
            "label": auth_result.label,
            "model_prediction": getattr(auth_result, "model_prediction", auth_result.label),
            "ai_generated_score": auth_result.ai_generated_score,
            "authentic_score": auth_result.authentic_score,
            "raw_logits": getattr(auth_result, "raw_logits", {}),
            "calibration_status": getattr(auth_result, "calibration_status", "not_calibrated"),
            "calibrated_ai_generated_probability": calibrated_probability,
            "calibrated_candidate_label": getattr(auth_result, "calibrated_candidate_label", "inconclusive"),
            "domain_validation_status": getattr(auth_result, "domain_validation_status", "not_evaluated"),
            "empirical_accuracy": getattr(auth_result, "empirical_accuracy", None),
            "model": auth_result.model_name,
            "model_version": getattr(auth_result, "model_version", None),
        },
        confidence=calibrated_probability if calibrated else None,
        calibrated=calibrated,
        stage_name="authenticity",
        status="completed",
        outcome="success",
    ))

    return items


def build_evidence_from_forensics(forensics_result) -> List[EvidenceItem]:
    """Convert forensic analysis result into evidence items."""
    if forensics_result is None:
        return []

    items = []

    for signal in forensics_result.signals:
        items.append(EvidenceItem(
            type="forensic_evidence",
            source=f"forensics.{signal.name}",
            strength="low",
            direction="informational",
            description=signal.interpretation,
            data={"status": signal.status, "indicator_score": signal.score, "score_type": "heuristic_not_probability", **signal.details},
            status=signal.status,
            outcome=getattr(signal, "outcome", None),
            stage_name="forensics",
        ))

    return items


def build_evidence_from_ocr(ocr_result, text_result=None) -> List[EvidenceItem]:
    """Convert OCR result into evidence items."""
    if ocr_result is None:
        return []

    items = []

    status = getattr(ocr_result, "status", "completed")
    if status not in {"pending", "running", "completed", "partial", "failed", "unavailable", "skipped"}:
        status = "unavailable"
    if status not in {"completed", "partial"}:
        items.append(EvidenceItem(
            type="ocr_evidence", source=ocr_result.engine or "ocr",
            strength="low", direction="informational",
            description=f"OCR status: {status}.",
            data={"status": status, "engine": ocr_result.engine, "warnings": ocr_result.warnings},
            status=status,
            outcome=getattr(ocr_result, "outcome", None),
            stage_name="ocr",
        ))
        return items

    if ocr_result.has_text:
        items.append(EvidenceItem(
            type="ocr_evidence",
            source=ocr_result.engine or "ocr",
            strength="medium" if ocr_result.word_count >= 5 else "low",
            direction="informational",
            description=(
                f"OCR extracted {ocr_result.word_count} words "
                f"({ocr_result.language} script estimate)" + (
                    f" with {ocr_result.total_confidence:.1%} average engine confidence."
                    if ocr_result.total_confidence is not None else "."
                )
            ),
            data={
                "word_count": ocr_result.word_count,
                "language": ocr_result.language,
                "confidence": ocr_result.total_confidence,
                "engine": ocr_result.engine,
                "text_preview": ocr_result.normalized_text[:200] if ocr_result.normalized_text else "",
            },
            status=status,
            outcome="success",
            stage_name="ocr",
        ))
    else:
        items.append(EvidenceItem(
            type="ocr_evidence",
            source=ocr_result.engine or "ocr",
            strength="low",
            direction="informational",
            description="No text detected in the image.",
            data={"word_count": 0},
            status=status,
            outcome="no_text",
            stage_name="ocr",
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
                "No EXIF metadata was found. Metadata can be removed by sharing or editing workflows; its absence does not establish authenticity or manipulation."
            ),
            data={"has_exif": False},
            stage_name="metadata",
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
                stage_name="metadata",
            ))

        if not metadata_result.anomalies:
            items.append(EvidenceItem(
                type="metadata_evidence",
                source="exif_extractor",
                strength="low",
                    direction="neutral",
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
                stage_name="metadata",
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
            status="unavailable",
            outcome="credentials_missing",
            stage_name="fact_check",
        ))
    elif factcheck_result.status == "failed":
        items.append(EvidenceItem(
            type="factcheck_evidence",
            source="google_factcheck_api",
            strength="low",
            direction="informational",
            description="Fact-check API request failed. See module status for details.",
            data={"status": "failed"},
            status="failed",
            outcome="provider_error",
            stage_name="fact_check",
        ))
    elif factcheck_result.status == "no_results":
        items.append(EvidenceItem(
            type="factcheck_evidence",
            source="google_factcheck_api",
            strength="low",
            direction="informational",
            description="No matching fact-checks found for extracted claims.",
            data={"status": "no_results", "queries": factcheck_result.queries_performed},
            status="completed",
            outcome="no_results",
            stage_name="fact_check",
        ))
    elif factcheck_result.status in {"success", "partial"}:
        if not factcheck_result.claims:
            items.append(EvidenceItem(
                type="factcheck_evidence", source="google_factcheck_api", strength="low",
                direction="informational",
                description="Fact-check queries completed without matching reviews.",
                data={"status": "no_results", "queries": factcheck_result.queries_performed},
                status="partial" if factcheck_result.status == "partial" else "completed",
                outcome="partial_results" if factcheck_result.status == "partial" else "no_results",
                stage_name="fact_check",
            ))
        for claim in factcheck_result.claims:
            rating_lower = claim.textual_rating.lower()
            if any(w in rating_lower for w in ["false", "fake", "incorrect", "misleading", "pants on fire"]):
                direction = "informational"
                strength = "high"
            elif any(w in rating_lower for w in ["true", "correct", "accurate"]):
                direction = "informational"
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
                evidence_id=claim.evidence_id,
                status="partial" if factcheck_result.status == "partial" else "completed",
                outcome="success",
                stage_name="fact_check",
            ))

    return items


def build_evidence_from_provenance(provenance_result) -> List[EvidenceItem]:
    """Keep C2PA verification and external web matches as independent evidence."""
    if provenance_result is None:
        return []

    items = []
    credentials = getattr(provenance_result, "content_credentials", None)
    if isinstance(credentials, dict):
        c2pa_status = credentials.get("status", "unavailable")
        verification = credentials.get("verification_outcome", "unverified")
        if c2pa_status == "completed":
            if verification == "no_manifest":
                c2pa_description = "No C2PA manifest was found; absence of credentials does not establish manipulation."
            elif verification == "verification_failed":
                c2pa_description = "C2PA verification reported an invalid signature or content-binding mismatch. This does not determine event truth."
            elif verification == "verified":
                c2pa_description = "C2PA signature and content binding were reported valid; signer trust and event truth are separate."
            else:
                c2pa_description = "C2PA manifest was read, but one or more verification properties remain unverified."
        elif c2pa_status == "failed":
            c2pa_description = "C2PA verification could not complete. See the stage warnings for a safe summary."
        else:
            c2pa_description = "C2PA verification is unavailable in this runtime."
        items.append(EvidenceItem(
            type="provenance_evidence", source="c2pa_content_credentials",
            strength="low", direction="informational",
            description=c2pa_description,
            data={key: value for key, value in credentials.items() if key not in {"warnings"}},
            status=c2pa_status if c2pa_status in {"completed", "partial", "failed", "unavailable", "skipped"} else "unavailable",
            outcome=verification,
            stage_name="provenance",
        ))

    web_status = getattr(provenance_result, "web_detection_status", None)
    if web_status not in {"success", "no_results", "failed", "unavailable", "partial"}:
        web_status = provenance_result.status
    if web_status == "unavailable":
        items.append(EvidenceItem(
            type="provenance_evidence",
            source="google_vision_web_detection",
            strength="low",
            direction="informational",
            description="Web detection API not configured. No provenance data available.",
            data={"status": "unavailable"},
            status="unavailable",
            outcome="credentials_missing",
            stage_name="provenance",
        ))
    elif web_status == "failed":
        items.append(EvidenceItem(
            type="provenance_evidence",
            source="google_vision_web_detection",
            strength="low",
            direction="informational",
            description="Web provenance request failed. See module status for details.",
            data={"status": web_status},
            status=web_status,
            outcome="provider_error",
            stage_name="provenance",
        ))
    elif web_status in {"success", "partial"}:
        web_item_status = "partial" if web_status == "partial" else "completed"
        emitted_web_finding = False
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
                    "provider": provenance_result.provider,
                    "retrieved_at": provenance_result.retrieved_at,
                    "exact_matches": exact_count,
                    "partial_matches": partial_count,
                    "pages": len(provenance_result.matching_pages),
                    "matching_pages": [page.to_dict() for page in provenance_result.matching_pages[:10]],
                    "matching_images": [image.to_dict() for image in provenance_result.matching_images[:10]],
                    "outcome": getattr(provenance_result, "web_detection_outcome", "success"),
                },
                status=web_item_status,
                outcome=getattr(provenance_result, "web_detection_outcome", "success"),
                stage_name="provenance",
            ))
            emitted_web_finding = True

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
                data={"labels": provenance_result.best_guess_labels,
                      "provider": provenance_result.provider,
                      "retrieved_at": provenance_result.retrieved_at,
                      "outcome": getattr(provenance_result, "web_detection_outcome", "success")},
                status=web_item_status,
                outcome=getattr(provenance_result, "web_detection_outcome", "success"),
                stage_name="provenance",
            ))
            emitted_web_finding = True
        if not emitted_web_finding:
            items.append(EvidenceItem(
                type="provenance_evidence", source="google_vision_web_detection",
                strength="low", direction="informational",
                description=("Web detection returned partial results without image matches or labels."
                             if web_status == "partial"
                             else "Web detection completed without image matches or labels."),
                data={"status": web_item_status, "provider": provenance_result.provider},
                status=web_item_status,
                outcome=(getattr(provenance_result, "web_detection_outcome", None) or "partial_results")
                if web_status == "partial" else "no_results",
                stage_name="provenance",
            ))
    elif web_status == "no_results":
        items.append(EvidenceItem(
            type="provenance_evidence",
            source="google_vision_web_detection",
            strength="low",
            direction="informational",
            description="No web matches found for this image.",
            data={"status": "no_results", "outcome": "no_results"},
            status="completed",
            outcome="no_results",
            stage_name="provenance",
        ))

    return items


def build_evidence_from_xai(xai_result) -> List[EvidenceItem]:
    """Convert XAI result into evidence items."""
    if xai_result is None:
        return []

    items = []

    status = getattr(xai_result, "status", "unavailable")
    if status not in {"pending", "running", "completed", "partial", "failed", "unavailable", "skipped"}:
        status = "unavailable"
    if xai_result.heatmap_path and status == "completed":
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
                "heatmap_url": xai_result.heatmap_url,
                "overlay_url": xai_result.overlay_url,
                "model_name": xai_result.model_name,
            },
            status="completed",
            outcome="heatmap_generated",
            stage_name="xai",
        ))
    else:
        items.append(EvidenceItem(
            type="xai_evidence", source="grad_cam", strength="low",
            direction="informational",
            description=("XAI attribution is unavailable." if status != "completed"
                         else "No XAI attribution map was produced."),
            data={"status": status},
            status=status,
            outcome="no_results" if status == "completed" else getattr(xai_result, "outcome", status),
            stage_name="xai",
        ))

    return items


def build_evidence_from_vision(vision_result) -> List[EvidenceItem]:
    """Convert vision/object detection result into evidence items."""
    if vision_result is None:
        return []

    items = []

    status = getattr(vision_result, "status", "completed")
    if status not in {"pending", "running", "completed", "partial", "failed", "unavailable", "skipped"}:
        status = "unavailable"
    outcome = getattr(vision_result, "outcome", None)
    if status in {"completed", "partial"} and vision_result.objects:
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
                "objects": [obj.__dict__ for obj in vision_result.objects],
                "model_name": vision_result.model_name,
            },
            status=status,
            outcome=outcome or "success",
            stage_name="vision",
        ))
    else:
        items.append(EvidenceItem(
            type="contextual_evidence", source="yolov8_object_detection",
            strength="low", direction="informational",
            description=("No objects were detected; absence of detections is not evidence of authenticity or manipulation."
                         if status == "completed" else f"Object detection status: {status}."),
            data={"status": status, "object_count": 0},
            status=status,
            outcome=outcome or ("no_results" if status == "completed" else status),
            stage_name="vision",
        ))

    return items
