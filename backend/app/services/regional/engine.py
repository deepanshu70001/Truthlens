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
    simulated_exposure_count: Optional[int] = None
    simulated_share_count: Optional[int] = None
    simulated_repost_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "region": self.region,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "language_compatibility": self.language_compatibility,
            "internet_penetration_per_100": self.internet_penetration,
            "evidence": self.evidence,
            "data_status": self.data_status,
        }
        if self.simulated_exposure_count is not None:
            d["simulated_exposure_count"] = self.simulated_exposure_count
        if self.simulated_share_count is not None:
            d["simulated_share_count"] = self.simulated_share_count
        if self.simulated_repost_count is not None:
            d["simulated_repost_count"] = self.simulated_repost_count
        return d


@dataclass
class RegionalSpreadResult:
    data_status: str = "DATA_UNAVAILABLE"
    regions: List[RegionalRisk] = field(default_factory=list)
    content_language: Optional[str] = None
    top_risk_regions: List[str] = field(default_factory=list)
    methodology: str = ""
    runtime_ms: int = 0
    warnings: List[str] = field(default_factory=list)
    disclaimer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        default_disclaimer = (
            "No local, traceable state/UT dissemination or outcome dataset is configured. "
            "Regional scores and rankings are unavailable; no state-level prediction is made."
        )
        return {
            "data_status": self.data_status,
            "regions": [r.to_dict() for r in self.regions],
            "content_language": self.content_language,
            "top_risk_regions": self.top_risk_regions,
            "methodology": self.methodology,
            "runtime_ms": self.runtime_ms,
            "warnings": self.warnings,
            "disclaimer": self.disclaimer or default_disclaimer,
            "sources": [
                {"name": "Census of India C-16, 2011", "status": "REFERENCE_ONLY_NOT_INSTALLED"},
                {"name": "TRAI Performance Indicator Reports", "status": "REFERENCE_ONLY_NOT_INSTALLED"},
            ],
        }


def estimate_regional_spread(
    content_language: Optional[str] = None,
    ocr_text: Optional[str] = None,
    analysis_id: Optional[str] = None,
) -> RegionalSpreadResult:
    """Return canonical administrative regions with explicit unavailable scores or neutral demo counts."""
    started = time.time()
    from backend.app.config import settings

    if analysis_id and getattr(settings, "TRUTHLENS_SYNTHETIC_REPORT_DATA_ENABLED", False):
        from backend.app.services.synthetic.data_access import get_synthetic_data_access

        synthetic_reg = get_synthetic_data_access().get_regional_spread_records(analysis_id)
        if synthetic_reg:
            regions: List[RegionalRisk] = []
            for r in synthetic_reg.get("regions", []):
                regions.append(
                    RegionalRisk(
                        region=r.get("region", ""),
                        risk_score=None,  # Strictly neutral: no state risk score
                        confidence=None,
                        language_compatibility=None,
                        internet_penetration=None,
                        evidence=r.get("evidence", []),
                        data_status="DEMO",
                        simulated_exposure_count=r.get("simulated_exposure_count"),
                        simulated_share_count=r.get("simulated_share_count"),
                        simulated_repost_count=r.get("simulated_repost_count"),
                    )
                )
            return RegionalSpreadResult(
                data_status="DEMO",
                regions=regions,
                content_language=content_language,
                top_risk_regions=[],  # Neutral: no state risk ranking
                methodology=synthetic_reg.get("methodology", ""),
                runtime_ms=int((time.time() - started) * 1000),
                warnings=[
                    "Synthetic demonstration data: Simulated state distribution counts only. "
                    "Regional differences are not validated or representative of actual Indian states or populations. "
                    "All generator regional effects are zero; no regional risk or ranking is imputed."
                ],
                disclaimer=synthetic_reg.get("disclaimer", ""),
            )

    regions = [
        RegionalRisk(region=name, evidence=["No validated regional dissemination data configured."])
        for name in _REGIONS
    ]
    return RegionalSpreadResult(
        regions=regions,
        content_language=content_language,
        methodology="Regional dissemination prediction is disabled until traceable, licensed, geographically compatible outcome data is installed and evaluated.",
        runtime_ms=int((time.time() - started) * 1000),
    )
