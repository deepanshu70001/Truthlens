"""
TruthLens — Virality estimation engine.

Estimates the potential spread risk of an image based on observable
features. This is a HEURISTIC system, NOT a trained prediction model.

DATA STATUS: DATA_UNAVAILABLE
- No publicly available dataset provides sufficient image-level
  engagement data with Indian context to train a reliable model.
- All outputs are heuristic estimates based on known correlates
  of virality from academic literature.
- When DEMO_MODE=true, synthetic scores are generated for UI
  demonstration purposes — these are NOT real predictions.

Feature categories:
- Image features (faces, text, objects, visual complexity)
- Text features (language, urgency, sentiment indicators, claims)
- Context features (topic sensitivity, timeliness)

References:
- Vosoughi et al. (2018) "The spread of true and false news online" Science
- Reis et al. (2020) "Can WhatsApp Counter Misinformation?"
"""

from __future__ import annotations

import os
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


def is_demo_mode() -> bool:
    """Check if DEMO_MODE is enabled."""
    return os.environ.get("DEMO_MODE", "").lower() in ("true", "1", "yes")


@dataclass
class ViralityFeature:
    """A single virality-related feature."""
    name: str
    value: float  # 0.0 to 1.0
    weight: float
    present: bool
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "weight": round(self.weight, 4),
            "present": self.present,
            "description": self.description,
        }


@dataclass
class ViralityResult:
    """Virality estimation result."""
    data_status: str = "DATA_UNAVAILABLE"  # DATA_UNAVAILABLE | DEMO
    virality_score: float = 0.0  # 0.0 to 1.0
    risk_level: str = "unknown"  # low | moderate | high | unknown
    features: List[ViralityFeature] = field(default_factory=list)
    contributing_factors: List[str] = field(default_factory=list)
    methodology: str = ""
    runtime_ms: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "data_status": self.data_status,
            "virality_score": round(self.virality_score, 4),
            "risk_level": self.risk_level,
            "features": [f.to_dict() for f in self.features],
            "contributing_factors": self.contributing_factors,
            "methodology": self.methodology,
            "runtime_ms": self.runtime_ms,
            "warnings": self.warnings,
        }

        if self.data_status == "DATA_UNAVAILABLE":
            result["disclaimer"] = (
                "This virality score is a HEURISTIC ESTIMATE based on observable "
                "content features. It is NOT a trained prediction model. No validated "
                "dataset exists for image-level virality prediction in the Indian context. "
                "This score should not be cited as a scientific prediction."
            )
        elif self.data_status == "DEMO":
            result["disclaimer"] = (
                "DEMO DATA — NOT A REAL-WORLD PREDICTION. "
                "This output is generated for UI demonstration purposes only. "
                "It does not reflect actual virality risk."
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

    # Has text overlay
    has_text = bool(ocr_text and len(ocr_text.strip()) > 5)
    features.append(ViralityFeature(
        name="has_text_overlay",
        value=1.0 if has_text else 0.0,
        weight=_FEATURE_WEIGHTS["has_text_overlay"],
        present=has_text,
        description="Image contains embedded text" if has_text else "No significant text detected",
    ))

    # Urgency language
    urgency_count = sum(1 for w in _URGENCY_WORDS if w in text_lower)
    urgency_present = urgency_count > 0
    urgency_score = min(urgency_count / 3.0, 1.0)
    features.append(ViralityFeature(
        name="urgency_language",
        value=urgency_score,
        weight=_FEATURE_WEIGHTS["urgency_language"],
        present=urgency_present,
        description=f"{urgency_count} urgency indicators detected" if urgency_present else "No urgency language detected",
    ))

    # Emotional language
    emotional_count = sum(1 for w in _EMOTIONAL_WORDS if w in text_lower)
    emotional_present = emotional_count > 0
    emotional_score = min(emotional_count / 3.0, 1.0)
    features.append(ViralityFeature(
        name="emotional_language",
        value=emotional_score,
        weight=_FEATURE_WEIGHTS["emotional_language"],
        present=emotional_present,
        description=f"{emotional_count} emotional indicators detected" if emotional_present else "No emotional language detected",
    ))

    # Has claims
    has_claims = bool(claims and len(claims) > 0)
    features.append(ViralityFeature(
        name="has_claims",
        value=1.0 if has_claims else 0.0,
        weight=_FEATURE_WEIGHTS["has_claims"],
        present=has_claims,
        description=f"{len(claims)} verifiable claims detected" if has_claims else "No claims detected",
    ))

    # Has numbers
    has_numbers = bool(text_normalized and text_normalized.get("detected_numbers"))
    features.append(ViralityFeature(
        name="has_numbers",
        value=1.0 if has_numbers else 0.0,
        weight=_FEATURE_WEIGHTS["has_numbers"],
        present=has_numbers,
        description="Specific numbers/statistics present" if has_numbers else "No numbers detected",
    ))

    # Topic sensitivity
    for topic, keywords, weight_key in [
        ("health", _HEALTH_KEYWORDS, "health_topic"),
        ("political", _POLITICAL_KEYWORDS, "political_topic"),
        ("religious", _RELIGIOUS_KEYWORDS, "religious_topic"),
    ]:
        topic_count = sum(1 for w in keywords if w in text_lower)
        topic_present = topic_count >= 2  # Need at least 2 keywords
        topic_score = min(topic_count / 4.0, 1.0) if topic_present else 0.0
        features.append(ViralityFeature(
            name=f"{topic}_topic",
            value=topic_score,
            weight=_FEATURE_WEIGHTS[weight_key],
            present=topic_present,
            description=f"{topic.title()} topic indicators detected ({topic_count} keywords)" if topic_present else f"No {topic} topic indicators",
        ))

    return features


def _compute_image_features(
    vision_result: Optional[Dict] = None,
) -> List[ViralityFeature]:
    """Compute image-based virality features."""
    features = []

    # Has faces
    has_people = bool(vision_result and vision_result.get("has_people"))
    features.append(ViralityFeature(
        name="has_faces",
        value=1.0 if has_people else 0.0,
        weight=_FEATURE_WEIGHTS["has_faces"],
        present=has_people,
        description="People/faces detected in image" if has_people else "No people detected",
    ))

    # Multiple objects (visual complexity)
    object_count = vision_result.get("object_count", 0) if vision_result else 0
    multiple = object_count >= 3
    features.append(ViralityFeature(
        name="multiple_objects",
        value=min(object_count / 10.0, 1.0),
        weight=_FEATURE_WEIGHTS["multiple_objects"],
        present=multiple,
        description=f"{object_count} objects detected" if object_count > 0 else "Few or no objects detected",
    ))

    return features


def estimate_virality(
    ocr_text: Optional[str] = None,
    claims: Optional[List[Dict]] = None,
    text_normalized: Optional[Dict] = None,
    vision_result: Optional[Dict] = None,
) -> ViralityResult:
    """
    Estimate virality risk based on observable content features.

    This is a HEURISTIC system, NOT a trained model.
    All results carry DATA_UNAVAILABLE or DEMO status.
    """
    t0 = time.time()

    demo_mode = is_demo_mode()
    data_status = "DEMO" if demo_mode else "DATA_UNAVAILABLE"

    # Compute features
    text_features = _compute_text_features(ocr_text, claims, text_normalized)
    image_features = _compute_image_features(vision_result)
    all_features = text_features + image_features

    # Compute weighted score
    weighted_sum = 0.0
    total_weight = 0.0
    contributing_factors = []

    for feat in all_features:
        if feat.present:
            weighted_sum += feat.value * feat.weight
            total_weight += feat.weight
            contributing_factors.append(feat.description)

    # Normalize to 0-1
    max_possible = sum(_FEATURE_WEIGHTS.values())
    virality_score = weighted_sum / max_possible if max_possible > 0 else 0.0
    virality_score = max(0.0, min(1.0, virality_score))

    # Risk level
    if virality_score >= 0.6:
        risk_level = "high"
    elif virality_score >= 0.3:
        risk_level = "moderate"
    else:
        risk_level = "low"

    warnings = []
    methodology = (
        "Heuristic feature-based estimation using known correlates of "
        "content virality from academic literature (Vosoughi et al., 2018). "
        "Features include text presence, urgency/emotional language, topic "
        "sensitivity, and visual characteristics. This is NOT a trained "
        "prediction model — no suitable training dataset exists for "
        "image-level virality prediction in the Indian context."
    )

    if demo_mode:
        warnings.append("DEMO DATA — NOT A REAL-WORLD PREDICTION")
        methodology = "DEMO MODE: " + methodology

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
    )

    log.info(
        "Virality estimate: score=%.3f risk=%s status=%s [%d ms]",
        virality_score, risk_level, data_status, runtime_ms,
    )

    return result
