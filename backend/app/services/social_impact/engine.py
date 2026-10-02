"""
TruthLens — Social impact assessment engine.

Assesses measurable content characteristics that may indicate
potential social harm categories.

DATA STATUS: DATA_UNAVAILABLE
- No publicly available dataset provides measured social impact
  outcomes linked to specific misinformation images.
- All assessments are rule-based content analysis, NOT predictions
  of actual social outcomes.

Assessed categories:
- Misinformation severity (based on authenticity + fact-check signals)
- Potential audience scale (based on language reach + internet penetration)
- Emotional/sensational characteristics (based on text analysis)
- Claim sensitivity (health, political, communal, financial)
- Urgency indicators

This engine does NOT:
- Create political persuasion scores
- Rank political actors
- Infer intent from content
- Predict specific social outcomes
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.app.core.logging import get_logger

log = get_logger("social_impact")

# Sensitivity categories with keyword indicators
_SENSITIVITY_CATEGORIES = {
    "communal_harmony": {
        "keywords": {
            "riot", "communal", "lynch", "mob", "temple", "mosque",
            "hindu", "muslim", "christian", "sikh", "attack",
            "burn", "destroy", "hate", "kill",
        },
        "description": "Content touching communal/religious themes",
        "weight": 0.25,
    },
    "public_health": {
        "keywords": {
            "vaccine", "covid", "cure", "treatment", "medicine",
            "hospital", "death", "disease", "poison", "toxic",
            "cancer", "diabetes", "remedy", "herbal", "miracle",
        },
        "description": "Content with health/medical claims",
        "weight": 0.25,
    },
    "political_stability": {
        "keywords": {
            "election", "vote", "fraud", "rigged", "corrupt",
            "government", "conspiracy", "coup", "protest", "emergency",
            "martial", "curfew", "banned",
        },
        "description": "Content with political implications",
        "weight": 0.20,
    },
    "financial_safety": {
        "keywords": {
            "bank", "scam", "fraud", "money", "investment", "scheme",
            "ponzi", "lottery", "prize", "winner", "deposit", "withdraw",
            "account", "upi", "payment",
        },
        "description": "Content with financial fraud indicators",
        "weight": 0.15,
    },
    "child_safety": {
        "keywords": {
            "child", "children", "kidnap", "abduct", "trafficking",
            "missing", "school", "minor", "abuse",
        },
        "description": "Content potentially affecting child safety",
        "weight": 0.15,
    },
}


def is_demo_mode() -> bool:
    """Check if DEMO_MODE is enabled."""
    return os.environ.get("DEMO_MODE", "").lower() in ("true", "1", "yes")


@dataclass
class ImpactCategory:
    """Assessment for a single impact category."""
    category: str
    risk_score: float = 0.0  # 0.0 to 1.0
    indicators_found: int = 0
    indicators: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "risk_score": round(self.risk_score, 4),
            "indicators_found": self.indicators_found,
            "indicators": self.indicators,
            "description": self.description,
        }


@dataclass
class SocialImpactResult:
    """Complete social impact assessment result."""
    data_status: str = "DATA_UNAVAILABLE"  # DATA_UNAVAILABLE | DEMO
    overall_risk: float = 0.0
    risk_level: str = "unknown"  # low | moderate | high | unknown
    categories: List[ImpactCategory] = field(default_factory=list)
    misinformation_severity: str = "unknown"  # low | moderate | high | unknown
    estimated_audience_scale: str = "unknown"  # limited | moderate | broad | unknown
    urgency_level: str = "low"  # low | moderate | high
    primary_concerns: List[str] = field(default_factory=list)
    methodology: str = ""
    runtime_ms: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "data_status": self.data_status,
            "overall_risk": round(self.overall_risk, 4),
            "risk_level": self.risk_level,
            "categories": [c.to_dict() for c in self.categories],
            "misinformation_severity": self.misinformation_severity,
            "estimated_audience_scale": self.estimated_audience_scale,
            "urgency_level": self.urgency_level,
            "primary_concerns": self.primary_concerns,
            "methodology": self.methodology,
            "runtime_ms": self.runtime_ms,
            "warnings": self.warnings,
        }

        if self.data_status == "DATA_UNAVAILABLE":
            result["disclaimer"] = (
                "Social impact assessment is based on RULE-BASED content analysis. "
                "No empirical model exists linking specific content characteristics "
                "to measurable social outcomes. These indicators identify content "
                "categories associated with potential harm in academic literature, "
                "but are NOT predictions of actual social impact."
            )
        elif self.data_status == "DEMO":
            result["disclaimer"] = (
                "DEMO DATA — NOT A REAL-WORLD PREDICTION. "
                "This output is generated for UI demonstration purposes only."
            )

        return result


def _assess_misinformation_severity(
    authenticity_label: Optional[str] = None,
    factcheck_status: Optional[str] = None,
    factcheck_rating: Optional[str] = None,
) -> str:
    """Assess misinformation severity based on available signals."""
    severity = "unknown"

    if authenticity_label in ("likely_ai_generated", "likely_manipulated"):
        severity = "moderate"

        if factcheck_status == "success" and factcheck_rating:
            rating_lower = factcheck_rating.lower()
            if any(w in rating_lower for w in ["false", "fake", "pants on fire"]):
                severity = "high"
            elif any(w in rating_lower for w in ["misleading", "partly false"]):
                severity = "moderate"
    elif authenticity_label == "likely_authentic":
        severity = "low"
    elif authenticity_label == "inconclusive":
        severity = "moderate"

    return severity


def _estimate_audience_scale(
    regional_top_count: int = 0,
    language: str = "",
) -> str:
    """Estimate potential audience scale based on language and regional reach."""
    # Hindi and English have the broadest reach in India
    broad_languages = {"hi", "en"}
    moderate_languages = {"bn", "ta", "te", "mr", "gu", "kn", "ml", "pa"}

    if language in broad_languages:
        return "broad"
    elif language in moderate_languages:
        return "moderate"
    elif regional_top_count > 0:
        return "limited"
    else:
        return "unknown"


def assess_social_impact(
    ocr_text: Optional[str] = None,
    authenticity_label: Optional[str] = None,
    factcheck_status: Optional[str] = None,
    factcheck_rating: Optional[str] = None,
    content_language: str = "",
    regional_top_count: int = 0,
    virality_score: float = 0.0,
) -> SocialImpactResult:
    """
    Assess potential social impact based on content characteristics.

    This is a RULE-BASED assessment, NOT a trained prediction model.
    All results carry DATA_UNAVAILABLE or DEMO status.
    """
    t0 = time.time()

    demo_mode = is_demo_mode()
    data_status = "DEMO" if demo_mode else "DATA_UNAVAILABLE"

    text_lower = (ocr_text or "").lower()
    categories: List[ImpactCategory] = []
    primary_concerns: List[str] = []

    # Assess each sensitivity category
    for cat_name, cat_config in _SENSITIVITY_CATEGORIES.items():
        found_keywords = [w for w in cat_config["keywords"] if w in text_lower]
        indicator_count = len(found_keywords)

        if indicator_count >= 3:
            risk_score = min(indicator_count / 5.0, 1.0)
        elif indicator_count >= 1:
            risk_score = indicator_count * 0.25
        else:
            risk_score = 0.0

        category = ImpactCategory(
            category=cat_name,
            risk_score=risk_score,
            indicators_found=indicator_count,
            indicators=found_keywords[:5],
            description=cat_config["description"],
        )
        categories.append(category)

        if risk_score >= 0.5:
            primary_concerns.append(
                f"{cat_name.replace('_', ' ').title()}: {indicator_count} indicators detected"
            )

    # Compute overall risk (weighted by category weights)
    overall_risk = 0.0
    for cat, config in zip(categories, _SENSITIVITY_CATEGORIES.values()):
        overall_risk += cat.risk_score * config["weight"]

    overall_risk = max(0.0, min(1.0, overall_risk))

    # Misinformation severity
    misinfo_severity = _assess_misinformation_severity(
        authenticity_label, factcheck_status, factcheck_rating,
    )

    # Audience scale
    audience_scale = _estimate_audience_scale(regional_top_count, content_language)

    # Urgency level
    urgency_words = {"breaking", "urgent", "alert", "now", "immediately", "emergency"}
    urgency_count = sum(1 for w in urgency_words if w in text_lower)
    if urgency_count >= 2:
        urgency_level = "high"
    elif urgency_count >= 1:
        urgency_level = "moderate"
    else:
        urgency_level = "low"

    # Risk level
    if overall_risk >= 0.6:
        risk_level = "high"
    elif overall_risk >= 0.3:
        risk_level = "moderate"
    else:
        risk_level = "low"

    warnings = []
    methodology = (
        "Rule-based content analysis using keyword indicators for "
        "sensitivity categories (communal harmony, public health, "
        "political stability, financial safety, child safety). "
        "No empirical model exists linking content characteristics to "
        "measurable social outcomes. Assessment is based on academic "
        "literature on misinformation harm categories."
    )

    if demo_mode:
        warnings.append("DEMO DATA — NOT A REAL-WORLD PREDICTION")
        methodology = "DEMO MODE: " + methodology

    runtime_ms = int((time.time() - t0) * 1000)

    result = SocialImpactResult(
        data_status=data_status,
        overall_risk=overall_risk,
        risk_level=risk_level,
        categories=categories,
        misinformation_severity=misinfo_severity,
        estimated_audience_scale=audience_scale,
        urgency_level=urgency_level,
        primary_concerns=primary_concerns,
        methodology=methodology,
        runtime_ms=runtime_ms,
        warnings=warnings,
    )

    log.info(
        "Social impact: risk=%.3f level=%s severity=%s status=%s [%d ms]",
        overall_risk, risk_level, misinfo_severity, data_status, runtime_ms,
    )

    return result
