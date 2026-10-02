"""
TruthLens — Report builder.

Assembles the final analysis report from all pipeline results.
The report is a stable JSON structure returned via the API.

Report structure:
{
  "analysis_id": "...",
  "status": "...",
  "image": {...},
  "authenticity": {...},
  "metadata": {...},
  "forensics": [...],
  "objects": [...],
  "ocr": {...},
  "claims": [...],
  "fact_checks": {...},
  "provenance": {...},
  "xai": {...},
  "evidence": [...],
  "assessment": {...},
  "explanation": {...},
  "virality": {...},
  "regional_spread": {...},
  "social_impact": {...},
  "limitations": [...],
  "disclaimer": "..."
}
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.app.core.logging import get_logger

log = get_logger("report")

DISCLAIMER = (
    "DISCLAIMER: TruthLens is an AI-assisted verification tool. Its outputs "
    "represent probabilistic assessments based on computational analysis, not "
    "definitive truth. Model predictions can be wrong. AI-generated content "
    "detection is an evolving field subject to concept drift. No result from "
    "this system should be cited as proof that an image is authentic or "
    "fabricated. Always cross-reference findings with additional sources "
    "and expert judgment."
)


def build_report(
    analysis_id: str,
    status: str,
    image_info: Optional[Dict[str, Any]] = None,
    authenticity: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    forensics: Optional[Dict[str, Any]] = None,
    objects: Optional[Dict[str, Any]] = None,
    ocr: Optional[Dict[str, Any]] = None,
    text_normalized: Optional[Dict[str, Any]] = None,
    claims: Optional[Dict[str, Any]] = None,
    fact_checks: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    xai: Optional[Dict[str, Any]] = None,
    hashes: Optional[Dict[str, Any]] = None,
    evidence: Optional[Dict[str, Any]] = None,
    assessment: Optional[Dict[str, Any]] = None,
    explanation: Optional[Dict[str, Any]] = None,
    virality: Optional[Dict[str, Any]] = None,
    regional_spread: Optional[Dict[str, Any]] = None,
    social_impact: Optional[Dict[str, Any]] = None,
    processing_time_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build the final analysis report.

    All parameters are optional — the report includes whatever
    data is available. Missing modules show null.
    """
    # Compile limitations from all sources
    limitations: List[str] = []

    if assessment and "limitations" in assessment:
        limitations.extend(assessment["limitations"])
    if explanation and "limitations" in explanation:
        for lim in explanation["limitations"]:
            if lim not in limitations:
                limitations.append(lim)

    # Add standard limitations
    standard_limitations = [
        "This system has NOT been independently validated or peer-reviewed.",
        "AI-generated content detection is subject to concept drift as "
        "new generation techniques emerge.",
        "Forensic signals can be produced by legitimate image processing "
        "operations (resizing, compression, filters).",
    ]
    for lim in standard_limitations:
        if lim not in limitations:
            limitations.append(lim)

    # Data status labeling for heuristic modules
    _data_status_note = None
    heuristic_modules = []
    for name, module_data in [("virality", virality), ("regional_spread", regional_spread), ("social_impact", social_impact)]:
        if module_data:
            ds = module_data.get("data_status", "DATA_UNAVAILABLE")
            if ds in ("DATA_UNAVAILABLE", "DEMO"):
                heuristic_modules.append(f"{name} ({ds})")
    if heuristic_modules:
        _data_status_note = (
            f"The following modules lack validated training data and produce "
            f"heuristic estimates only: {', '.join(heuristic_modules)}. "
            f"Their outputs should NOT be cited as scientific predictions."
        )
        if _data_status_note not in limitations:
            limitations.append(_data_status_note)

    report = {
        "analysis_id": analysis_id,
        "status": status,
        "image": image_info,
        "authenticity": authenticity,
        "metadata": metadata,
        "forensics": forensics,
        "objects": objects,
        "ocr": ocr,
        "text_normalized": text_normalized,
        "claims": claims,
        "fact_checks": fact_checks,
        "provenance": provenance,
        "xai": xai,
        "hashes": hashes,
        "evidence": evidence,
        "assessment": assessment,
        "explanation": explanation,
        "virality": virality,
        "regional_spread": regional_spread,
        "social_impact": social_impact,
        "processing_time_ms": processing_time_ms,
        "limitations": limitations,
        "disclaimer": DISCLAIMER,
    }

    log.info("Report built for analysis %s", analysis_id)

    return report
