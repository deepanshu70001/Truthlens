"""Evidence-linked content harm category indicators; not impact prediction."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.app.core.logging import get_logger

log = get_logger("social_impact")

_CATEGORIES = {
    "communal_harmony": {"riot", "communal", "lynch", "mob", "temple", "mosque", "hindu", "muslim", "christian", "sikh", "attack", "burn", "destroy", "hate", "kill"},
    "public_health": {"vaccine", "covid", "cure", "treatment", "medicine", "hospital", "death", "disease", "poison", "toxic", "cancer", "diabetes", "remedy", "herbal", "miracle"},
    "political_stability": {"election", "vote", "fraud", "rigged", "corrupt", "government", "conspiracy", "coup", "protest", "emergency", "martial", "curfew", "banned"},
    "financial_safety": {"bank", "scam", "fraud", "money", "investment", "scheme", "ponzi", "lottery", "prize", "winner", "deposit", "withdraw", "account", "upi", "payment"},
    "child_safety": {"child", "children", "kidnap", "abduct", "trafficking", "missing", "school", "minor", "abuse"},
}
_URGENCY = {"breaking", "urgent", "alert", "warning", "now", "immediately", "emergency", "forward", "share"}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"(?u)\b[\w']+\b", text.lower()))


@dataclass
class ImpactCategory:
    category: str
    risk_score: Optional[float] = None
    indicators_found: int = 0
    indicators: List[str] = field(default_factory=list)
    description: str = ""
    supporting_claim_ids: List[str] = field(default_factory=list)
    verification_statuses: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"category": self.category, "risk_score": None, "indicators_found": self.indicators_found,
                "indicators": self.indicators, "description": self.description,
                "supporting_claim_ids": self.supporting_claim_ids,
                "verification_statuses": self.verification_statuses, "evidence_ids": self.evidence_ids,
                "interpretation": "Potential content category only; no harm outcome or severity inferred."}


@dataclass
class SocialImpactResult:
    data_status: str = "DATA_UNAVAILABLE"
    overall_risk: Optional[float] = None
    risk_level: str = "unknown"
    categories: List[ImpactCategory] = field(default_factory=list)
    misinformation_severity: str = "unassessed"
    estimated_audience_scale: str = "unknown"
    urgency_level: str = "unassessed"
    primary_concerns: List[str] = field(default_factory=list)
    methodology: str = ""
    runtime_ms: int = 0
    warnings: List[str] = field(default_factory=list)
    urgency_indicators: List[str] = field(default_factory=list)
    observed_public_impact: str = "UNAVAILABLE"

    def to_dict(self) -> Dict[str, Any]:
        return {"data_status": self.data_status, "assessment_type": "HEURISTIC_CONTENT_CATEGORIES",
                "overall_risk": None, "risk_level": "unknown", "categories": [c.to_dict() for c in self.categories],
                "misinformation_severity": "unassessed", "estimated_audience_scale": "unknown",
                "urgency_level": self.urgency_level, "urgency_indicators": self.urgency_indicators,
                "observed_public_impact": "UNAVAILABLE", "primary_concerns": self.primary_concerns,
                "methodology": self.methodology, "runtime_ms": self.runtime_ms, "warnings": self.warnings,
                "disclaimer": "Matched terms identify possible content categories only. They do not establish falsity, intent, severity, audience reach, or observed harm. Claim verification is reported separately."}


def assess_social_impact(ocr_text: Optional[str] = None, authenticity_label: Optional[str] = None,
                         factcheck_status: Optional[str] = None, factcheck_rating: Optional[str] = None,
                         content_language: Optional[str] = None, regional_top_count: int = 0,
                         virality_score: Optional[float] = None, claims: Optional[List[Dict[str, Any]]] = None,
                         claim_verifications: Optional[Dict[str, Any]] = None) -> SocialImpactResult:
    """Match exact tokens and attach available Phase 3 claim evidence references."""
    started = time.time()
    tokens = _tokens(ocr_text or "")
    claims = claims or []
    raw_verifications = claim_verifications or {}
    if isinstance(raw_verifications, list):
        verifications = {str(item.get("claim_id")): item for item in raw_verifications if isinstance(item, dict) and item.get("claim_id")}
    elif isinstance(raw_verifications, dict):
        verifications = raw_verifications
    else:
        verifications = {}
    categories: List[ImpactCategory] = []
    concerns: List[str] = []
    for name, keywords in _CATEGORIES.items():
        found = sorted(tokens & keywords)
        claim_ids, statuses, evidence_ids = [], [], []
        for claim in claims:
            claim_text = str(claim.get("text") or claim.get("claim") or "")
            if _tokens(claim_text) & set(found):
                cid = str(claim.get("claim_id") or claim.get("id") or "")
                if cid:
                    claim_ids.append(cid)
                    v = verifications.get(cid, {})
                    if isinstance(v, dict):
                        statuses.append(str(v.get("verification_status") or v.get("status") or v.get("verdict") or "UNASSESSED"))
                        evidence_ids.extend(str(eid) for field in ("supporting_evidence_ids", "contradicting_evidence_ids", "conflicting_evidence_ids") for eid in (v.get(field) or []))
                        evidence_ids.extend(str(e.get("evidence_id") or e.get("id")) for e in (v.get("evidence") or []) if isinstance(e, dict) and (e.get("evidence_id") or e.get("id")))
        categories.append(ImpactCategory(name, indicators_found=len(found), indicators=found[:8],
            description="Potential content topic; keyword match is not evidence of harm or falsity.",
            supporting_claim_ids=sorted(set(claim_ids)), verification_statuses=statuses,
            evidence_ids=sorted(set(evidence_ids))))
        if found:
            concerns.append(name)
    urgency = sorted(tokens & _URGENCY)
    return SocialImpactResult(categories=categories, primary_concerns=concerns,
        urgency_level="indicators_present" if urgency else ("no_indicators_detected" if ocr_text else "unassessed"),
        urgency_indicators=urgency, methodology="Exact token matching for transparent category cues; no calibrated severity or outcome model is available.",
        runtime_ms=int((time.time() - started) * 1000),
        warnings=["OCR text is partial evidence; absent or missed text is not evidence of absence."] if not ocr_text else [])
