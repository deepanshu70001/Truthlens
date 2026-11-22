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
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.app.core.logging import get_logger
from backend.app.config import settings

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


def _safe_report_value(value: Any) -> Any:
    """Remove local storage paths and secret-like fields before API serialization."""
    private_keys = {
        "storage_path", "thumbnail_path", "heatmap_path", "overlay_path",
        "resized_paths", "_gps_data", "all_tags",
    }
    if isinstance(value, dict):
        safe = {}
        for key, item in value.items():
            lower_key = key.lower()
            if lower_key in private_keys or any(
                secret in lower_key for secret in ("api_key", "token", "secret", "password")
            ):
                continue
            if lower_key in {"error", "error_message"} and item:
                safe[key] = "The stage or service failed; technical details are in server logs."
            else:
                safe[key] = _safe_report_value(item)
        return safe
    if isinstance(value, (list, tuple)):
        return [_safe_report_value(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)(?:[a-z]:\\|/)(?:[^\s:]+[\\/])+[^\s:]*", "[redacted path]", value)
        value = re.sub(
            r"(?i)(api[_ -]?key|key|token|secret|password)\s*[:=]\s*\S+",
            r"\1=[redacted]", value,
        )
    return value


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
    quality: Optional[Dict[str, Any]] = None,
    preprocessing: Optional[Dict[str, Any]] = None,
    processing_time_ms: Optional[int] = None,
    stages: Optional[Dict[str, Any]] = None,
    submitted_at: Optional[str] = None,
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
            f"heuristic indicators or unavailable outcomes: {', '.join(heuristic_modules)}. "
            f"These are not calibrated predictions and must not be cited as scientific forecasts."
        )
        if _data_status_note not in limitations:
            limitations.append(_data_status_note)

    stages = stages or {}
    stage_warnings = []
    for name, stage in stages.items():
        state = stage.get("status") if isinstance(stage, dict) else None
        detail = stage.get("error") if isinstance(stage, dict) else None
        if state in {"failed", "unavailable", "skipped", "partial"}:
            message = f"Stage '{name}' {state}."
            if detail:
                message += f" {detail}"
            stage_warnings.append(message)
    for name, module in (("fact_checks", fact_checks), ("provenance", provenance)):
        if isinstance(module, dict) and module.get("status") in {"failed", "unavailable"}:
            stage_warnings.append(f"{name} service status: {module['status']}.")

    if status == "completed" and any(
        isinstance(stage, dict) and stage.get("status") in {"failed", "unavailable", "partial"}
        for stage in stages.values()
    ):
        status = "partial"

    auth_stage = stages.get("authenticity", {})
    authenticity_status = auth_stage.get("status", "unavailable") if isinstance(auth_stage, dict) else "unavailable"
    if authenticity is None:
        authenticity = {
            "status": authenticity_status,
            "model_name": settings.AUTHENTICITY_MODEL,
            "label": None,
            "model_prediction": None,
            "ai_generated_score": None,
            "authentic_score": None,
            "calibrated_ai_generated_probability": None,
            "calibration_status": "not_calibrated",
            "calibration_method": None,
            "domain_validation_status": "not_evaluated",
            "empirical_accuracy": None,
            "score_semantics": "Raw softmax scores are not calibrated probabilities or estimates of real-world accuracy.",
            "warnings": [auth_stage.get("error")] if isinstance(auth_stage, dict) and auth_stage.get("error") else [],
        }
    else:
        authenticity = dict(authenticity)
        authenticity["status"] = authenticity_status
        calibrated_score = authenticity.get("calibrated_ai_generated_probability")
        authenticity["decision_band"] = (
            "AI_GENERATED" if calibrated_score is not None and authenticity.get("label") == "likely_ai_generated"
            else "AUTHENTIC" if calibrated_score is not None and authenticity.get("label") == "likely_authentic"
            else "UNCERTAIN" if calibrated_score is not None
            else None
        )

    fusion_status = stages.get("fusion", {}).get("status", "unavailable") if isinstance(stages.get("fusion"), dict) else "unavailable"
    fusion = dict(assessment) if assessment is not None else {
        "label": None,
        "overall_score": None,
        "confidence": None,
        "reasoning": ["Fusion did not produce an assessment."],
        "limitations": ["No usable evidence was available for fusion."],
    }
    fusion["status"] = fusion_status

    claim_status = (fact_checks or {}).get("status", stages.get("fact_check", {}).get("status", "unavailable"))
    fact_verdicts = []
    if isinstance(fact_checks, dict):
        fact_verdicts = [str(item.get("verdict") or item.get("verification_status") or "unassessed")
                         for item in (fact_checks.get("claim_verifications") or []) if isinstance(item, dict)]
    virality_mode = (virality or {}).get("assessment_mode", "UNAVAILABLE")
    regional_status = (regional_spread or {}).get("data_status", "DATA_UNAVAILABLE")
    social_categories = [item.get("category") for item in ((social_impact or {}).get("categories") or [])
                         if isinstance(item, dict) and item.get("indicators_found", 0)]
    summary = {
        "authenticity": {"status": authenticity_status, "classification": authenticity.get("label"),
                         "evidence_fusion_label": fusion.get("label"),
                         "note": "Detector classification and evidence-fusion assessment are separate outputs; neither is proof."},
        "factual_claims": {"status": claim_status, "verdicts": fact_verdicts,
                           "note": "Fact-check results are per-claim and source-dependent."},
        "provenance": {"status": (provenance or {}).get("status", "unavailable"),
                       "note": "Online matches provide context and do not establish authenticity."},
        "spread": {"assessment_mode": virality_mode, "regional_data_status": regional_status,
                   "note": "Content indicators are not estimates of actual reach; regional rankings are unavailable."},
        "potential_social_impact": {"status": (social_impact or {}).get("data_status", "unavailable"),
                                    "categories": social_categories,
                                    "note": "Topic cues are not observed harm or severity findings."},
    }

    report = {
        "analysis_id": analysis_id,
        "status": status,
        "submitted_at": submitted_at,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "validation_status": "completed" if image_info else "unavailable",
            "filename": image_info.get("original_filename") if image_info else None,
            "sha256": image_info.get("sha256_hash") if image_info else None,
        },
        "image": image_info,
        "authenticity": authenticity,
        "metadata": metadata,
        "quality": quality,
        "preprocessing": preprocessing,
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
        "fusion": fusion,
        "summary": summary,
        "explanation": explanation,
        "virality": virality,
        "regional_spread": regional_spread,
        "social_impact": social_impact,
        "processing_time_ms": processing_time_ms,
        "stages": stages or {},
        "warnings": stage_warnings,
        "module_statuses": {
            name: stage.get("status") for name, stage in stages.items() if isinstance(stage, dict)
        } | {
            "authenticity": authenticity_status,
            "fact_checks": (fact_checks or {}).get("status", stages.get("fact_check", {}).get("status", "unavailable")),
            "provenance": (provenance or {}).get("status", stages.get("provenance", {}).get("status", "unavailable")),
            "ocr": stages.get("ocr", {}).get("status", "unavailable"),
            "visual_context": stages.get("vision", {}).get("status", "unavailable"),
        },
        "limitations": limitations,
        "disclaimer": DISCLAIMER,
    }

    log.info("Report built for analysis %s", analysis_id)

    return _safe_report_value(report)
