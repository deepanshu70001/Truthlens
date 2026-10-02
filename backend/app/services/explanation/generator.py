"""
TruthLens — Explanation generator.

Builds structured natural-language explanations from available evidence.

Rules:
- Only describes what ACTUALLY happened (results from real modules)
- NEVER invents articles, URLs, metadata, model results, statistics
- NEVER fabricates fact-check outcomes or geographic data
- Uses hedging language ("suggests", "indicates", "appears")
- Cites specific evidence items
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.app.core.logging import get_logger
from backend.app.services.evidence.schema import EvidenceCollection, EvidenceItem

log = get_logger("explanation")


@dataclass
class Explanation:
    """A structured explanation of the analysis findings."""
    summary: str = ""
    sections: List[Dict[str, str]] = field(default_factory=list)
    key_findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    runtime_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "sections": self.sections,
            "key_findings": self.key_findings,
            "limitations": self.limitations,
            "runtime_ms": self.runtime_ms,
        }


def _describe_model_evidence(items: List[EvidenceItem]) -> Optional[Dict[str, str]]:
    """Generate description for model evidence."""
    if not items:
        return None

    parts = []
    for item in items:
        parts.append(item.description)

    return {
        "title": "AI Model Assessment",
        "content": " ".join(parts),
    }


def _describe_forensic_evidence(items: List[EvidenceItem]) -> Optional[Dict[str, str]]:
    """Generate description for forensic evidence."""
    if not items:
        return None

    notable = [i for i in items if i.strength in ("medium", "high")]
    if not notable:
        return {
            "title": "Forensic Analysis",
            "content": (
                "Classical forensic analysis did not detect notable anomalies. "
                "Note: forensic signals are informational indicators and "
                "many legitimate image operations produce similar patterns."
            ),
        }

    parts = []
    for item in notable:
        parts.append(item.description)

    return {
        "title": "Forensic Analysis",
        "content": " ".join(parts) + (
            " Note: forensic signals are informational indicators derived "
            "from classical image processing and are not proof of manipulation."
        ),
    }


def _describe_ocr_evidence(items: List[EvidenceItem]) -> Optional[Dict[str, str]]:
    """Generate description for OCR evidence."""
    if not items:
        return None

    parts = []
    for item in items:
        parts.append(item.description)

    return {
        "title": "Text Content",
        "content": " ".join(parts),
    }


def _describe_factcheck_evidence(items: List[EvidenceItem]) -> Optional[Dict[str, str]]:
    """Generate description for fact-check evidence."""
    if not items:
        return None

    parts = []
    for item in items:
        parts.append(item.description)

    return {
        "title": "Fact-Check Results",
        "content": " ".join(parts),
    }


def _describe_provenance_evidence(items: List[EvidenceItem]) -> Optional[Dict[str, str]]:
    """Generate description for provenance evidence."""
    if not items:
        return None

    parts = []
    for item in items:
        parts.append(item.description)

    return {
        "title": "Web Provenance",
        "content": " ".join(parts),
    }


def _describe_xai_evidence(items: List[EvidenceItem]) -> Optional[Dict[str, str]]:
    """Generate description for XAI evidence."""
    if not items:
        return None

    parts = []
    for item in items:
        parts.append(item.description)

    return {
        "title": "Explainable AI",
        "content": " ".join(parts),
    }


def _describe_metadata_evidence(items: List[EvidenceItem]) -> Optional[Dict[str, str]]:
    """Generate description for metadata evidence."""
    if not items:
        return None

    parts = []
    for item in items:
        parts.append(item.description)

    return {
        "title": "Metadata Analysis",
        "content": " ".join(parts),
    }


def _describe_context_evidence(items: List[EvidenceItem]) -> Optional[Dict[str, str]]:
    """Generate description for contextual evidence."""
    if not items:
        return None

    parts = []
    for item in items:
        parts.append(item.description)

    return {
        "title": "Contextual Analysis",
        "content": " ".join(parts),
    }


def generate_explanation(
    evidence: EvidenceCollection,
    assessment_label: str,
) -> Explanation:
    """
    Generate a structured explanation based ONLY on available evidence.

    The explanation:
    - Describes what each module found
    - Never invents data
    - Uses hedging language
    - Cites specific evidence items
    """
    t0 = time.time()

    sections = []
    key_findings = []
    limitations = []

    # Build sections from evidence by type
    section_builders = [
        ("model_evidence", _describe_model_evidence),
        ("forensic_evidence", _describe_forensic_evidence),
        ("ocr_evidence", _describe_ocr_evidence),
        ("metadata_evidence", _describe_metadata_evidence),
        ("contextual_evidence", _describe_context_evidence),
        ("xai_evidence", _describe_xai_evidence),
        ("factcheck_evidence", _describe_factcheck_evidence),
        ("provenance_evidence", _describe_provenance_evidence),
    ]

    for evidence_type, builder in section_builders:
        items = evidence.get_by_type(evidence_type)
        section = builder(items)
        if section:
            sections.append(section)

    # Extract key findings (high-strength evidence)
    for item in evidence.get_strong_evidence():
        key_findings.append(item.description)

    # Build summary
    summary_parts = []

    # Assessment summary
    label_descriptions = {
        "likely_authentic": "The available evidence suggests this image is likely authentic.",
        "likely_ai_generated": "The available evidence suggests this image may be AI-generated.",
        "likely_manipulated": "The available evidence suggests this image may have been manipulated.",
        "inconclusive": "The available evidence is insufficient or conflicting for a reliable determination.",
    }
    summary_parts.append(label_descriptions.get(assessment_label, label_descriptions["inconclusive"]))

    # Evidence overview
    type_counts = evidence.to_dict().get("by_type", {})
    if type_counts:
        summary_parts.append(
            f"This assessment is based on {evidence.count} evidence signals "
            f"across {len(type_counts)} analysis categories."
        )

    if evidence.has_conflicting_evidence:
        summary_parts.append(
            "Note: Some evidence signals conflict with each other, "
            "which reduces overall confidence."
        )

    summary = " ".join(summary_parts)

    # Standard limitations
    limitations.extend([
        "This is an AI-assisted analysis. Model predictions can be wrong.",
        "No single signal is sufficient to determine image authenticity.",
        "All confidence scores are uncalibrated unless explicitly stated otherwise.",
    ])

    if not evidence.get_by_type("factcheck_evidence") or all(
        i.data.get("status") in ("unavailable", "failed")
        for i in evidence.get_by_type("factcheck_evidence")
    ):
        limitations.append(
            "Fact-check verification was not available for this analysis."
        )

    if not evidence.get_by_type("provenance_evidence") or all(
        i.data.get("status") in ("unavailable", "failed")
        for i in evidence.get_by_type("provenance_evidence")
    ):
        limitations.append(
            "Web provenance search was not available for this analysis."
        )

    runtime_ms = int((time.time() - t0) * 1000)

    result = Explanation(
        summary=summary,
        sections=sections,
        key_findings=key_findings,
        limitations=limitations,
        runtime_ms=runtime_ms,
    )

    log.info(
        "Explanation: %d sections, %d findings, %d limitations [%d ms]",
        len(sections), len(key_findings), len(limitations), runtime_ms,
    )

    return result
