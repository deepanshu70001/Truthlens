"""
TruthLens — Evidence fusion engine.

Combines evidence from all modules into a final assessment.

Rules:
1. Evidence is weighted by strength (low=0.1, medium=0.3, high=0.6)
2. Conflicting strong evidence → "inconclusive"
3. No strong evidence → "inconclusive"
4. Uncalibrated confidence is NEVER presented as validated probability
5. The fusion engine does NOT force a result when signals conflict

Output labels:
- likely_authentic
- likely_ai_generated
- likely_manipulated
- inconclusive
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.app.core.logging import get_logger
from backend.app.services.evidence.schema import EvidenceCollection, EvidenceItem

log = get_logger("fusion")

# Weight multipliers for evidence strength
STRENGTH_WEIGHTS = {
    "low": 0.1,
    "medium": 0.3,
    "high": 0.6,
}

# Direction mappings for scoring
DIRECTION_SCORES = {
    "authentic": 1.0,
    "synthetic": -1.0,
    "manipulated": -0.8,
    "neutral": 0.0,
    "informational": 0.0,
}

# Minimum weighted evidence score to make a determination
DETERMINATION_THRESHOLD = 0.15

# If opposing weighted scores are both above this, it's inconclusive
CONFLICT_THRESHOLD = 0.10


@dataclass
class FusionAssessment:
    """Final assessment from the fusion engine."""
    label: str = "inconclusive"  # likely_authentic | likely_ai_generated | likely_manipulated | inconclusive
    overall_score: Optional[float] = None  # 0.0 = synthetic direction, 1.0 = authentic direction
    confidence: Optional[float] = None
    calibrated: bool = False
    reasoning: List[str] = field(default_factory=list)
    evidence_summary: Dict[str, int] = field(default_factory=dict)
    conflicting: bool = False
    limitations: List[str] = field(default_factory=list)
    runtime_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "label": self.label,
            "overall_score": round(self.overall_score, 4) if self.overall_score is not None else None,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "calibrated": self.calibrated,
            "reasoning": self.reasoning,
            "evidence_summary": self.evidence_summary,
            "conflicting": self.conflicting,
            "limitations": self.limitations,
            "runtime_ms": self.runtime_ms,
        }
        if not self.calibrated:
            result["confidence_note"] = (
                "This confidence score is uncalibrated. It reflects the "
                "weighted combination of available evidence signals, not a "
                "scientifically validated probability. Independent calibration "
                "has not been performed."
            )
        result["caveat"] = (
            "This is an AI-assisted assessment based on available evidence. "
            "Model predictions can be wrong. This result should not be cited "
            "as proof that the image is authentic or fabricated."
        )
        return result


def fuse_evidence(evidence: EvidenceCollection) -> FusionAssessment:
    """
    Combine all evidence into a final assessment.

    The fusion engine uses weighted voting:
    - Each evidence item votes in a direction with a weight
    - If votes strongly conflict, result is "inconclusive"
    - If no strong evidence, result is "inconclusive"
    - Otherwise, the dominant direction wins

    Confidence is the margin of the dominant direction.
    Confidence is ALWAYS marked as uncalibrated.
    """
    t0 = time.time()
    reasoning: List[str] = []
    limitations: List[str] = []

    if evidence.count == 0:
        return FusionAssessment(
            label="inconclusive",
            overall_score=None,
            confidence=None,
            reasoning=["No evidence available for assessment."],
            limitations=["No analysis modules produced evidence."],
            runtime_ms=int((time.time() - t0) * 1000),
        )

    # Compute weighted scores in each direction
    authentic_score = 0.0
    synthetic_score = 0.0
    manipulated_score = 0.0
    total_weight = 0.0

    evidence_summary = {
        "total": evidence.count,
        "authentic_signals": 0,
        "synthetic_signals": 0,
        "manipulated_signals": 0,
        "neutral_signals": 0,
        "informational_signals": 0,
    }

    seen = set()
    for item in evidence.items:
        duplicate_key = (item.type, item.source, item.direction, item.description)
        if duplicate_key in seen:
            continue
        seen.add(duplicate_key)
        # A failed/absent stage has no observed evidence and must not vote.
        # Preserve its status in the evidence record for traceability only.
        if item.status not in {"completed", "partial"}:
            reasoning.append(f"Ignored {item.status} evidence from {item.source}.")
            continue
        if item.strength not in STRENGTH_WEIGHTS or item.direction not in DIRECTION_SCORES:
            reasoning.append(f"Ignored invalid evidence from {item.source}.")
            continue
        if item.confidence is not None and (
            not math.isfinite(item.confidence) or not 0.0 <= item.confidence <= 1.0
        ):
            reasoning.append(f"Ignored evidence with invalid confidence from {item.source}.")
            continue

        # Only the classifier currently contributes a directional authenticity
        # vote. Classical forensic indicators are heuristic and informational.
        directional = item.type == "model_evidence"
        if directional and item.direction in {"authentic", "synthetic", "manipulated"} and not item.calibrated:
            evidence_summary["neutral_signals"] += 1
            reasoning.append(
                f"Ignored directional vote from {item.source}: its model output is uncalibrated."
            )
            continue
        direction = item.direction if directional else "informational"
        weight = STRENGTH_WEIGHTS[item.strength]

        if direction == "authentic":
            authentic_score += weight
            evidence_summary["authentic_signals"] += 1
        elif direction == "synthetic":
            synthetic_score += weight
            evidence_summary["synthetic_signals"] += 1
        elif direction == "manipulated":
            manipulated_score += weight
            evidence_summary["manipulated_signals"] += 1
        elif item.direction == "neutral":
            evidence_summary["neutral_signals"] += 1
        else:
            evidence_summary["informational_signals"] += 1

        total_weight += weight

    # Check for conflicts
    negative_score = synthetic_score + manipulated_score
    conflicting = (
        authentic_score >= CONFLICT_THRESHOLD
        and negative_score >= CONFLICT_THRESHOLD
    )

    if conflicting:
        reasoning.append(
            f"Evidence conflicts: authentic signals ({authentic_score:.2f}) "
            f"vs. synthetic/manipulated signals ({negative_score:.2f})."
        )

    # Determine label
    if conflicting and abs(authentic_score - negative_score) < DETERMINATION_THRESHOLD:
        label = "inconclusive"
        overall_score = None
        confidence = None
        reasoning.append(
            "Strong conflicting evidence prevents a reliable determination."
        )
    elif total_weight < 0.1:
        label = "inconclusive"
        overall_score = None
        confidence = None
        reasoning.append("Insufficient evidence for a determination.")
    elif authentic_score + negative_score == 0:
        label = "inconclusive"
        overall_score = None
        confidence = None
        reasoning.append(
            "No calibrated directional evidence is available; informational and raw classifier outputs cannot establish authenticity."
        )
    elif authentic_score > negative_score + DETERMINATION_THRESHOLD:
        label = "likely_authentic"
        overall_score = authentic_score / (authentic_score + negative_score)
        confidence = min(authentic_score - negative_score, 1.0)
        reasoning.append(
            f"Evidence leans toward authentic "
            f"(authentic={authentic_score:.2f}, negative={negative_score:.2f})."
        )
    elif synthetic_score > manipulated_score:
        label = "likely_ai_generated"
        overall_score = 1.0 - (synthetic_score / (authentic_score + negative_score))
        confidence = min(negative_score - authentic_score, 1.0)
        reasoning.append(
            f"Evidence suggests AI-generated content "
            f"(synthetic={synthetic_score:.2f}, authentic={authentic_score:.2f})."
        )
    elif manipulated_score > 0:
        label = "likely_manipulated"
        overall_score = 1.0 - (manipulated_score / (authentic_score + negative_score))
        confidence = min(negative_score - authentic_score, 1.0)
        reasoning.append(
            f"Evidence suggests manipulation "
            f"(manipulated={manipulated_score:.2f}, authentic={authentic_score:.2f})."
        )
    else:
        label = "inconclusive"
        overall_score = None
        confidence = None
        reasoning.append("No clear direction from available evidence.")

    # Clamp values
    if overall_score is not None:
        overall_score = max(0.0, min(1.0, overall_score))
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))

    # Add limitations
    limitations.append(
        "This assessment is based on automated analysis and has not been "
        "independently validated."
    )
    if not any(i.calibrated for i in evidence.items if i.confidence is not None):
        limitations.append(
            "All model confidence scores are uncalibrated — they should not "
            "be interpreted as scientifically validated probabilities."
        )
    if evidence_summary["informational_signals"] > evidence_summary["authentic_signals"] + evidence_summary["synthetic_signals"]:
        limitations.append(
            "Most evidence is informational rather than directional. "
            "The assessment relies on limited directional signals."
        )

    runtime_ms = int((time.time() - t0) * 1000)

    result = FusionAssessment(
        label=label,
        overall_score=overall_score,
        confidence=confidence,
        calibrated=False,
        reasoning=reasoning,
        evidence_summary=evidence_summary,
        conflicting=conflicting,
        limitations=limitations,
        runtime_ms=runtime_ms,
    )

    log.info(
        "Fusion: label=%s score=%.3f confidence=%.3f conflicting=%s [%d ms]",
        label, overall_score, confidence, conflicting, runtime_ms,
    )

    return result
