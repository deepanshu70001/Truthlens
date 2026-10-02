"""
TruthLens — India regional spread risk estimation.

Estimates the potential regional spread risk across Indian states
and union territories based on language compatibility and
internet penetration data.

DATA STATUS: DATA_UNAVAILABLE
- No dataset maps content characteristics to actual regional
  spread patterns in India.
- Language compatibility is estimated using Census 2011 data.
- Internet penetration is from TRAI quarterly reports.
- All outputs are heuristic estimates, NOT trained predictions.

Sources:
- Census of India 2011: Language speakers by state
- TRAI 2024: Internet subscribers by telecom circle
- No causal model exists linking these features to content spread.

IMPORTANT:
- Does NOT infer regional vulnerability from stereotypes
- Does NOT fabricate state-level engagement numbers
- Uses wording "Estimated regional spread risk"
- Never says "This fake news will spread here"
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.app.core.logging import get_logger

log = get_logger("regional")

# Census 2011 — Primary language speakers by state (approximate %)
# Source: Office of the Registrar General & Census Commissioner, India
# These are the DOMINANT language(s) by share of speakers in each state.
_STATE_LANGUAGES = {
    "Andhra Pradesh": {"telugu": 0.82, "urdu": 0.09, "hindi": 0.03},
    "Arunachal Pradesh": {"hindi": 0.27, "bengali": 0.10, "english": 0.08},
    "Assam": {"assamese": 0.48, "bengali": 0.28, "hindi": 0.06},
    "Bihar": {"hindi": 0.73, "maithili": 0.13, "urdu": 0.08},
    "Chhattisgarh": {"hindi": 0.78, "chhattisgarhi": 0.11},
    "Goa": {"konkani": 0.66, "marathi": 0.18, "hindi": 0.07},
    "Gujarat": {"gujarati": 0.86, "hindi": 0.06, "urdu": 0.03},
    "Haryana": {"hindi": 0.88, "punjabi": 0.05, "urdu": 0.03},
    "Himachal Pradesh": {"hindi": 0.86, "punjabi": 0.05},
    "Jharkhand": {"hindi": 0.61, "bengali": 0.09, "urdu": 0.06},
    "Karnataka": {"kannada": 0.66, "urdu": 0.10, "telugu": 0.07},
    "Kerala": {"malayalam": 0.97, "tamil": 0.01},
    "Madhya Pradesh": {"hindi": 0.86, "urdu": 0.03},
    "Maharashtra": {"marathi": 0.68, "hindi": 0.11, "urdu": 0.07},
    "Manipur": {"manipuri": 0.55, "hindi": 0.05},
    "Meghalaya": {"khasi": 0.47, "garo": 0.31, "bengali": 0.08},
    "Mizoram": {"mizo": 0.73, "hindi": 0.03},
    "Nagaland": {"english": 0.15, "hindi": 0.08},
    "Odisha": {"odia": 0.82, "hindi": 0.03},
    "Punjab": {"punjabi": 0.88, "hindi": 0.07},
    "Rajasthan": {"hindi": 0.90, "urdu": 0.03},
    "Sikkim": {"nepali": 0.63, "hindi": 0.09},
    "Tamil Nadu": {"tamil": 0.89, "telugu": 0.06, "urdu": 0.02},
    "Telangana": {"telugu": 0.77, "urdu": 0.12, "hindi": 0.05},
    "Tripura": {"bengali": 0.67, "kokborok": 0.24, "hindi": 0.03},
    "Uttar Pradesh": {"hindi": 0.81, "urdu": 0.12},
    "Uttarakhand": {"hindi": 0.88, "urdu": 0.03},
    "West Bengal": {"bengali": 0.86, "hindi": 0.07, "urdu": 0.02},
    # Union Territories
    "Delhi": {"hindi": 0.81, "punjabi": 0.05, "urdu": 0.06},
    "Chandigarh": {"hindi": 0.62, "punjabi": 0.28},
    "Jammu and Kashmir": {"kashmiri": 0.55, "dogri": 0.24, "hindi": 0.07, "urdu": 0.07},
    "Ladakh": {"ladakhi": 0.45, "hindi": 0.12},
    "Puducherry": {"tamil": 0.88, "telugu": 0.06},
    "Andaman and Nicobar Islands": {"hindi": 0.36, "bengali": 0.28, "tamil": 0.15},
    "Lakshadweep": {"malayalam": 0.84},
    "Dadra and Nagar Haveli and Daman and Diu": {"gujarati": 0.42, "hindi": 0.27},
}

# TRAI-derived internet penetration estimates (per 100 population)
# Approximate from TRAI Q4 2023-24 report, rounded
_STATE_INTERNET_PENETRATION = {
    "Andhra Pradesh": 62, "Arunachal Pradesh": 48, "Assam": 42,
    "Bihar": 38, "Chhattisgarh": 44, "Goa": 85, "Gujarat": 71,
    "Haryana": 72, "Himachal Pradesh": 82, "Jharkhand": 40,
    "Karnataka": 72, "Kerala": 75, "Madhya Pradesh": 45,
    "Maharashtra": 68, "Manipur": 55, "Meghalaya": 50,
    "Mizoram": 62, "Nagaland": 47, "Odisha": 42, "Punjab": 76,
    "Rajasthan": 56, "Sikkim": 78, "Tamil Nadu": 65,
    "Telangana": 70, "Tripura": 52, "Uttar Pradesh": 46,
    "Uttarakhand": 68, "West Bengal": 50,
    "Delhi": 137,  # Exceeds 100 due to multi-SIM and floating population
    "Chandigarh": 142, "Jammu and Kashmir": 55, "Ladakh": 40,
    "Puducherry": 80, "Andaman and Nicobar Islands": 65,
    "Lakshadweep": 70, "Dadra and Nagar Haveli and Daman and Diu": 72,
}

# Map of language codes (as used by OCR/language detection) to language names
_LANG_CODE_MAP = {
    "hi": "hindi", "en": "english", "bn": "bengali", "ta": "tamil",
    "te": "telugu", "mr": "marathi", "gu": "gujarati", "kn": "kannada",
    "ml": "malayalam", "pa": "punjabi", "or": "odia", "as": "assamese",
    "ur": "urdu", "ne": "nepali", "ks": "kashmiri",
}


def is_demo_mode() -> bool:
    """Check if DEMO_MODE is enabled."""
    return os.environ.get("DEMO_MODE", "").lower() in ("true", "1", "yes")


@dataclass
class RegionalRisk:
    """Risk estimation for a single region."""
    region: str
    risk_score: float = 0.0
    confidence: float = 0.0
    language_compatibility: float = 0.0
    internet_penetration: float = 0.0
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region": self.region,
            "risk_score": round(self.risk_score, 4),
            "confidence": round(self.confidence, 4),
            "language_compatibility": round(self.language_compatibility, 4),
            "internet_penetration_per_100": self.internet_penetration,
            "evidence": self.evidence,
        }


@dataclass
class RegionalSpreadResult:
    """Complete regional spread estimation result."""
    data_status: str = "DATA_UNAVAILABLE"  # DATA_UNAVAILABLE | DEMO
    regions: List[RegionalRisk] = field(default_factory=list)
    content_language: str = ""
    top_risk_regions: List[str] = field(default_factory=list)
    methodology: str = ""
    runtime_ms: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "data_status": self.data_status,
            "regions": [r.to_dict() for r in self.regions],
            "content_language": self.content_language,
            "top_risk_regions": self.top_risk_regions,
            "methodology": self.methodology,
            "runtime_ms": self.runtime_ms,
            "warnings": self.warnings,
        }

        if self.data_status == "DATA_UNAVAILABLE":
            result["disclaimer"] = (
                "Regional spread risk is an ESTIMATE based on language compatibility "
                "(Census 2011) and internet penetration (TRAI 2024). No dataset exists "
                "that maps content characteristics to actual regional spread patterns. "
                "These scores should NOT be interpreted as predictions of where "
                "misinformation will actually spread."
            )
        elif self.data_status == "DEMO":
            result["disclaimer"] = (
                "DEMO DATA — NOT A REAL-WORLD PREDICTION. "
                "This output is generated for UI demonstration purposes only."
            )

        return result


def _compute_language_compatibility(
    content_lang: str, state_languages: Dict[str, float]
) -> float:
    """
    Compute how compatible content language is with a state's speakers.

    Returns 0.0 to 1.0 representing the fraction of speakers who
    can potentially understand the content language.
    """
    lang_name = _LANG_CODE_MAP.get(content_lang, content_lang).lower()

    # Direct match
    if lang_name in state_languages:
        return state_languages[lang_name]

    # English is widely understood as a second language
    if lang_name == "english":
        # Rough estimate: English comprehension varies
        # States with higher literacy tend to have more English readers
        return 0.15  # Conservative base estimate

    # Hindi has broad secondary reach
    if lang_name == "hindi" and "hindi" not in state_languages:
        return 0.10  # Many non-Hindi states have some Hindi comprehension

    return 0.0


def estimate_regional_spread(
    content_language: str = "en",
    ocr_text: Optional[str] = None,
) -> RegionalSpreadResult:
    """
    Estimate regional spread risk across Indian states.

    Based on:
    1. Language compatibility (Census 2011 data)
    2. Internet penetration (TRAI 2024 data)

    NOT based on: stereotypes, political assumptions, or fabricated data.

    Args:
        content_language: ISO 639-1 language code of the content
        ocr_text: Optional OCR text for additional feature extraction
    """
    t0 = time.time()

    demo_mode = is_demo_mode()
    data_status = "DEMO" if demo_mode else "DATA_UNAVAILABLE"

    regions: List[RegionalRisk] = []

    for state, state_langs in _STATE_LANGUAGES.items():
        # Language compatibility
        lang_compat = _compute_language_compatibility(content_language, state_langs)

        # Internet penetration (normalized to 0-1)
        internet_raw = _STATE_INTERNET_PENETRATION.get(state, 50)
        internet_norm = min(internet_raw / 100.0, 1.0)

        # Combined risk score (simple weighted average)
        # 60% language, 40% internet — language is the stronger predictor
        risk_score = 0.6 * lang_compat + 0.4 * internet_norm

        # Confidence is low because this is heuristic
        confidence = 0.15  # Very low — we have no validation data

        evidence = []
        if lang_compat > 0:
            lang_name = _LANG_CODE_MAP.get(content_language, content_language)
            evidence.append(
                f"{lang_name.title()} speakers: ~{lang_compat:.0%} of state population"
            )
        evidence.append(
            f"Internet penetration: ~{internet_raw} per 100 population"
        )

        regions.append(RegionalRisk(
            region=state,
            risk_score=risk_score,
            confidence=confidence,
            language_compatibility=lang_compat,
            internet_penetration=internet_raw,
            evidence=evidence,
        ))

    # Sort by risk score descending
    regions.sort(key=lambda r: r.risk_score, reverse=True)

    # Top risk regions
    top_regions = [r.region for r in regions[:5] if r.risk_score > 0.3]

    warnings = []
    methodology = (
        "Estimated using language compatibility from Census of India 2011 "
        "(60% weight) and internet penetration from TRAI Q4 2023-24 reports "
        "(40% weight). This is a heuristic estimate — no validated model "
        "exists for predicting actual content spread by region in India."
    )

    if demo_mode:
        warnings.append("DEMO DATA — NOT A REAL-WORLD PREDICTION")
        methodology = "DEMO MODE: " + methodology

    runtime_ms = int((time.time() - t0) * 1000)

    result = RegionalSpreadResult(
        data_status=data_status,
        regions=regions,
        content_language=content_language,
        top_risk_regions=top_regions,
        methodology=methodology,
        runtime_ms=runtime_ms,
        warnings=warnings,
    )

    log.info(
        "Regional spread: %d regions scored, top=%s, status=%s [%d ms]",
        len(regions), top_regions[:3], data_status, runtime_ms,
    )

    return result
