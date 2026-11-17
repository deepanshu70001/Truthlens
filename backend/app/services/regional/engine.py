"""Regional dissemination context. No regional predictions without validated data."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_REGIONS = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
    "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands", "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
]


@dataclass
class RegionalRisk:
    region: str
    risk_score: Optional[float] = None
    confidence: Optional[float] = None
    language_compatibility: Optional[float] = None
    internet_penetration: Optional[float] = None
    evidence: List[str] = field(default_factory=list)
    data_status: str = "DATA_UNAVAILABLE"

    def to_dict(self) -> Dict[str, Any]:
        return {"region": self.region, "risk_score": self.risk_score, "confidence": self.confidence,
                "language_compatibility": self.language_compatibility,
                "internet_penetration_per_100": self.internet_penetration, "evidence": self.evidence,
                "data_status": self.data_status}


@dataclass
class RegionalSpreadResult:
    data_status: str = "DATA_UNAVAILABLE"
    regions: List[RegionalRisk] = field(default_factory=list)
    content_language: Optional[str] = None
    top_risk_regions: List[str] = field(default_factory=list)
    methodology: str = ""
    runtime_ms: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"data_status": self.data_status, "regions": [r.to_dict() for r in self.regions],
                "content_language": self.content_language, "top_risk_regions": self.top_risk_regions,
                "methodology": self.methodology, "runtime_ms": self.runtime_ms, "warnings": self.warnings,
                "disclaimer": "No local, traceable state/UT dissemination or outcome dataset is configured. Regional scores and rankings are unavailable; no state-level prediction is made.",
                "sources": [{"name": "Census of India C-16, 2011", "status": "REFERENCE_ONLY_NOT_INSTALLED"},
                            {"name": "TRAI Performance Indicator Reports", "status": "REFERENCE_ONLY_NOT_INSTALLED"}]}


def estimate_regional_spread(content_language: Optional[str] = None, ocr_text: Optional[str] = None) -> RegionalSpreadResult:
    """Return canonical administrative regions with explicit unavailable scores."""
    started = time.time()
    regions = [RegionalRisk(region=name, evidence=["No validated regional dissemination data configured."]) for name in _REGIONS]
    return RegionalSpreadResult(regions=regions, content_language=content_language,
        methodology="Regional dissemination prediction is disabled until traceable, licensed, geographically compatible outcome data is installed and evaluated.",
        runtime_ms=int((time.time() - started) * 1000))
