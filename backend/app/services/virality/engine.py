"""
TruthLens — Virality estimation engine.

Estimates the potential spread risk of an image based on observable
features. This is a HEURISTIC system, NOT a trained prediction model.

DATA STATUS: DATA_UNAVAILABLE
- No publicly available dataset provides sufficient image-level
  engagement data with Indian context to train a reliable model.
- All outputs are heuristic estimates based on known correlates
  of virality from academic literature.
- No synthetic demo score path is implemented; outputs remain content
  indicators and historical prediction stays unavailable.

Feature categories:
- Image features (faces, text, objects, visual complexity)
- Text features (language, urgency, sentiment indicators, claims)
- Context features (topic sensitivity, timeliness)

References:
- Vosoughi et al. (2018) "The spread of true and false news online" Science
- Reis et al. (2020) "Can WhatsApp Counter Misinformation?"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.app.core.logging import get_logger

log = get_logger("virality")

# Feature weights derived from literature review, NOT from training
# These are heuristic correlates, not learned parameters
_FEATURE_WEIGHTS = {
    "has_faces": 0.12,           # Images with faces get more engagement
    "has_text_overlay": 0.10,    # Text in images increases shareability
    "emotional_language": 0.15,  # Emotional/urgent language correlates with shares
    "has_claims": 0.10,          # Verifiable claims increase discussion
    "has_numbers": 0.08,         # Specific numbers increase perceived credibility
    "multiple_objects": 0.05,    # Complex scenes hold attention
    "health_topic": 0.10,       # Health misinfo has historically spread fast
    "political_topic": 0.10,    # Political content has high engagement
    "religious_topic": 0.10,    # Religious content can spread rapidly in India
    "urgency_language": 0.10,   # "Breaking", "urgent" increases forwarding
}

# Emotional/urgency keywords (conservative list from literature)
_URGENCY_WORDS = {
    "breaking", "urgent", "alert", "warning", "danger", "shocking",
    "exclusive", "just in", "happening now", "confirmed", "viral",
    "share", "forward", "must see", "must read", "wake up",
}

_EMOTIONAL_WORDS = {
    "outrageous", "disgusting", "horrifying", "incredible", "amazing",
    "heartbreaking", "devastating", "terrifying", "shameful", "expose",
    "scandal", "corruption", "betrayal", "conspiracy", "cover up",
}

# Sensitive topics (language-neutral keyword patterns)
_HEALTH_KEYWORDS = {
    "covid", "vaccine", "death", "hospital", "doctor", "cure",
    "treatment", "medicine", "virus", "epidemic", "pandemic",
    "cancer", "diabetes", "health", "disease", "infection",
}

_POLITICAL_KEYWORDS = {
    "election", "vote", "government", "minister", "party", "bjp",
    "congress", "aap", "parliament", "modi", "rahul", "opposition",
    "policy", "law", "amendment", "protest", "rally",
}

_RELIGIOUS_KEYWORDS = {
    "temple", "mosque", "church", "hindu", "muslim", "christian",
    "sikh", "prayer", "festival", "religious", "faith", "god",
    "communal", "riot",
}


@dataclass
class ViralityFeature:
    """A single virality-related feature."""
    name: str
    value: Optional[float]  # None means the feature could not be assessed
    weight: float
    present: bool
    description: str
    available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 4) if self.value is not None else None,
            "weight": round(self.weight, 4),
            "present": self.present,
            "description": self.description,
            "available": self.available,
        }


@dataclass
class ViralityResult:
    """Virality estimation result."""
    data_status: str = "DATA_UNAVAILABLE"  # DATA_UNAVAILABLE | DEMO
    virality_score: Optional[float] = None  # heuristic indicator only, not spread probability
    risk_level: str = "unknown"  # low | moderate | high | unknown
    features: List[ViralityFeature] = field(default_factory=list)
    contributing_factors: List[str] = field(default_factory=list)
    methodology: str = ""
    runtime_ms: int = 0
    warnings: List[str] = field(default_factory=list)
    synthetic_demo_score: Optional[float] = None
    synthetic_demo_band: Optional[str] = None
    synthetic_record: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "data_status": self.data_status,
            "virality_score": round(self.virality_score, 4) if self.virality_score is not None else None,
            "assessment_mode": "CONTENT_BASED_HEURISTIC" if self.data_status != "DEMO" else "SYNTHETIC_DEMO",
            "historical_prediction_status": "DATA_UNAVAILABLE",
            "risk_level": self.risk_level,
            "features": [f.to_dict() for f in self.features],
            "contributing_factors": self.contributing_factors,
            "methodology": self.methodology,
            "runtime_ms": self.runtime_ms,
            "warnings": self.warnings,
        }

        if self.synthetic_demo_score is not None:
            result["synthetic_demo_score"] = self.synthetic_demo_score
        if self.synthetic_demo_band is not None:
            result["synthetic_demo_band"] = self.synthetic_demo_band
        if self.synthetic_record is not None:
            result["synthetic_record"] = self.synthetic_record

        if self.data_status == "DATA_UNAVAILABLE":
            result["disclaimer"] = (
                "This is a content indicator score, not a prediction of sharing, reach, or probability. "
                "No validated image-level historical dissemination dataset is configured."
            )
        elif self.data_status == "DEMO":
            result["disclaimer"] = (
                "⚠️ SYNTHETIC DEMONSTRATION DATA: Generated for college project demonstration only. "
                "Not observed social metrics, does not evaluate detector accuracy, "
                "and does not influence authenticity classification."
            )

        return result


def _compute_text_features(
    ocr_text: Optional[str] = None,
    claims: Optional[List[Dict]] = None,
    text_normalized: Optional[Dict] = None,
) -> List[ViralityFeature]:
    """Compute text-based virality features."""
    features = []
    text_lower = (ocr_text or "").lower()
    text_available = ocr_text is not None

    # Has text overlay
    has_text = bool(ocr_text and len(ocr_text.strip()) > 5)
    features.append(ViralityFeature(
        name="has_text_overlay",
        value=(1.0 if has_text else 0.0) if text_available else None,
        weight=_FEATURE_WEIGHTS["has_text_overlay"],
        present=has_text,
        description=("Image contains embedded text" if has_text else "No significant text detected") if text_available else "OCR text unavailable",
        available=text_available,
    ))

    # Urgency language
    urgency_count = sum(1 for w in _URGENCY_WORDS if w in text_lower)
    urgency_present = urgency_count > 0
    urgency_score = min(urgency_count / 3.0, 1.0)
    features.append(ViralityFeature(
        name="urgency_language",
        value=urgency_score if text_available else None,
        weight=_FEATURE_WEIGHTS["urgency_language"],
        present=urgency_present,
        description=(f"{urgency_count} urgency indicators detected" if urgency_present else "No urgency language detected") if text_available else "OCR text unavailable",
        available=text_available,
    ))

    # Emotional language
    emotional_count = sum(1 for w in _EMOTIONAL_WORDS if w in text_lower)
    emotional_present = emotional_count > 0
    emotional_score = min(emotional_count / 3.0, 1.0)
    features.append(ViralityFeature(
        name="emotional_language",
        value=emotional_score if text_available else None,
        weight=_FEATURE_WEIGHTS["emotional_language"],
        present=emotional_present,
        description=(f"{emotional_count} emotional indicators detected" if emotional_present else "No emotional language detected") if text_available else "OCR text unavailable",
        available=text_available,
    ))

    # Has claims
    has_claims = bool(claims and len(claims) > 0)
    features.append(ViralityFeature(
        name="has_claims",
        value=(1.0 if has_claims else 0.0) if claims is not None else None,
        weight=_FEATURE_WEIGHTS["has_claims"],
        present=has_claims,
        description=(f"{len(claims)} claims detected" if has_claims else "No claims detected") if claims is not None else "Claim extraction unavailable",
        available=claims is not None,
    ))

    # Has numbers
    has_numbers = bool(text_normalized and text_normalized.get("detected_numbers"))
    features.append(ViralityFeature(
        name="has_numbers",
        value=(1.0 if has_numbers else 0.0) if text_normalized is not None else None,
        weight=_FEATURE_WEIGHTS["has_numbers"],
        present=has_numbers,
        description=("Specific numbers/statistics present" if has_numbers else "No numbers detected") if text_normalized is not None else "Normalized text unavailable",
        available=text_normalized is not None,
    ))

    # Topic sensitivity
    for topic, keywords, weight_key in [
        ("health", _HEALTH_KEYWORDS, "health_topic"),
        ("political", _POLITICAL_KEYWORDS, "political_topic"),
        ("religious", _RELIGIOUS_KEYWORDS, "religious_topic"),
    ]:
        topic_count = sum(1 for w in keywords if w in text_lower) if text_available else 0
        topic_present = topic_count >= 2  # Need at least 2 keywords
        topic_score = min(topic_count / 4.0, 1.0) if topic_present else 0.0
        features.append(ViralityFeature(
            name=f"{topic}_topic",
            value=topic_score if text_available else None,
            weight=_FEATURE_WEIGHTS[weight_key],
            present=topic_present,
            description=(f"{topic.title()} topic indicators detected ({topic_count} keywords)" if topic_present else f"No {topic} topic indicators") if text_available else "OCR text unavailable",
            available=text_available,
        ))

    return features


def _compute_image_features(
    vision_result: Optional[Dict] = None,
) -> List[ViralityFeature]:
    """Compute image-based virality features."""
    features = []

    # Has faces
    has_people = bool(vision_result and (vision_result.get("has_people") or vision_result.get("people_count", 0)))
    features.append(ViralityFeature(
        name="has_faces",
        value=(1.0 if has_people else 0.0) if vision_result is not None else None,
        weight=_FEATURE_WEIGHTS["has_faces"],
        present=has_people,
        description=("People/faces detected in image" if has_people else "No people detected") if vision_result is not None else "Vision analysis unavailable",
        available=vision_result is not None,
    ))

    # Multiple objects (visual complexity)
    object_count = vision_result.get("object_count", len(vision_result.get("objects", []))) if vision_result else 0
    multiple = object_count >= 3
    features.append(ViralityFeature(
        name="multiple_objects",
        value=min(object_count / 10.0, 1.0) if vision_result is not None else None,
        weight=_FEATURE_WEIGHTS["multiple_objects"],
        present=multiple,
        description=(f"{object_count} objects detected" if object_count > 0 else "Few or no objects detected") if vision_result is not None else "Vision analysis unavailable",
        available=vision_result is not None,
    ))

    return features


def estimate_virality(
    ocr_text: Optional[str] = None,
    claims: Optional[List[Dict]] = None,
    text_normalized: Optional[Dict] = None,
    vision_result: Optional[Dict] = None,
    analysis_id: Optional[str] = None,
) -> ViralityResult:
    """
    Estimate virality risk based on observable content features.

    This is a HEURISTIC system, NOT a trained model.
    Results carry DATA_UNAVAILABLE by default unless synthetic report demo data
    is explicitly enabled via configuration.
    """
    t0 = time.time()
    from backend.app.config import settings

    data_status = "DATA_UNAVAILABLE"
    synthetic_demo_score: Optional[float] = None
    synthetic_demo_band: Optional[str] = None
    synthetic_rec: Optional[Dict[str, Any]] = None

    if analysis_id and getattr(settings, "TRUTHLENS_SYNTHETIC_REPORT_DATA_ENABLED", False):
        from backend.app.services.synthetic.data_access import get_synthetic_data_access

        synthetic_rec = get_synthetic_data_access().get_virality_record(analysis_id)
        if synthetic_rec:
            data_status = "DEMO"
            synthetic_demo_score = synthetic_rec.get("synthetic_content_indicator_score")
            synthetic_demo_band = synthetic_rec.get("synthetic_demo_band")

    # Compute features
    text_features = _compute_text_features(ocr_text, claims, text_normalized)
    image_features = _compute_image_features(vision_result)
    all_features = text_features + image_features

    # Compute weighted score
    weighted_sum = 0.0
    total_weight = 0.0
    contributing_factors = []

    for feat in all_features:
        if feat.available and feat.value is not None:
            weighted_sum += feat.value * feat.weight
            total_weight += feat.weight
            if feat.present:
                contributing_factors.append(feat.description)

    # Normalize to 0-1
    heuristic_score = max(0.0, min(1.0, weighted_sum / total_weight)) if total_weight else None

    # If synthetic demo is enabled, use synthetic indicator score for demo display
    if data_status == "DEMO" and synthetic_demo_score is not None:
        virality_score = synthetic_demo_score
        band_str = str(synthetic_demo_band).lower()
        if "critical" in band_str or "high" in band_str:
            risk_level = "high"
        elif "moderate" in band_str:
            risk_level = "moderate"
        else:
            risk_level = "low"
        contributing_factors.append(
            f"Synthetic demo indicator band: {synthetic_demo_band} (score {synthetic_demo_score})"
        )
    else:
        virality_score = heuristic_score
        # Risk level
        if virality_score is None:
            risk_level = "unknown"
        elif virality_score >= 0.6:
            risk_level = "high"
        elif virality_score >= 0.3:
            risk_level = "moderate"
        else:
            risk_level = "low"

    warnings = []
    if data_status == "DEMO":
        methodology = (
            "Synthetic demonstration content indicators from validated demo dataset. "
            "Simulated indicators for academic demonstration only. Fictional metrics."
        )
        warnings.append("Synthetic demonstration data enabled: Not real-world social engagement.")
    else:
        methodology = (
            "Heuristic feature-based estimation using known correlates of "
            "content virality from academic literature (Vosoughi et al., 2018). "
            "Features include text presence, urgency/emotional language, topic "
            "sensitivity, and visual characteristics. This is NOT a trained "
            "prediction model — no suitable training dataset exists for "
            "image-level virality prediction in the Indian context."
        )

    runtime_ms = int((time.time() - t0) * 1000)

    result = ViralityResult(
        data_status=data_status,
        virality_score=virality_score,
        risk_level=risk_level,
        features=all_features,
        contributing_factors=contributing_factors,
        methodology=methodology,
        runtime_ms=runtime_ms,
        warnings=warnings,
        synthetic_demo_score=synthetic_demo_score,
        synthetic_demo_band=synthetic_demo_band,
        synthetic_record=synthetic_rec,
    )

    log.info("Virality content indicators computed: score=%s level=%s status=%s [%d ms]", virality_score, risk_level, data_status, runtime_ms)

    return result
