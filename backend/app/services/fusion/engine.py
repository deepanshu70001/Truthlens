"""Conservative, versioned evidence fusion for production reports.

The current production policy permits a directional vote only from the
calibrated and input-domain-validated primary authenticity detector. Other
modules remain informational. The existing score weights and determination
thresholds are retained for backward compatibility; their empirical validity
has not been established by this module.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

from backend.app.core.logging import get_logger
from backend.app.services.evidence.schema import EVIDENCE_TYPES, EvidenceCollection, EvidenceItem

log = get_logger("fusion")

POLICY_VERSION = "e11.conservative.v1"
EXECUTION_STATUSES = {"pending", "running", "completed", "partial", "failed", "unavailable", "skipped"}
USABLE_STATUSES = {"completed", "partial"}
DIRECTION_SCORES = {
    "authentic": 1.0,
    "synthetic": -1.0,
    "manipulated": -0.8,
    "neutral": 0.0,
    "informational": 0.0,
}
DIRECTION_LABELS = {
    "likely_authentic": "authentic",
    "likely_ai_generated": "synthetic",
    "likely_manipulated": "manipulated",
}

# Retained production values. These were pre-existing policy constants, not
# estimated or tuned as part of E11.
STRENGTH_WEIGHTS = {"low": 0.1, "medium": 0.3, "high": 0.6}
DETERMINATION_THRESHOLD = 0.15
CONFLICT_THRESHOLD = 0.10


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _safe_identity(value: Any) -> Optional[Dict[str, str]]:
    """Keep public model/provider identifiers while dropping local paths."""
    identity = _field(value, "module_identity")
    if not isinstance(identity, Mapping):
        return None
    allowed = {
        "provider", "engine", "model_name", "model_id", "model_version",
        "model_revision", "model_sha256", "implementation",
        "implementation_version", "method", "method_version",
    }
    result: Dict[str, str] = {}
    for key in allowed:
        item = identity.get(key)
        if not isinstance(item, str) or not item.strip():
            continue
        # Windows drive paths and Unix absolute paths must not enter reports.
        if re.match(r"^[A-Za-z]:[\\/]", item) or item.startswith("/"):
            continue
        result[key] = item[:256]
    return result or None


def _stage_summary(stage_results: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(stage_results, Mapping):
        return {}
    summary: Dict[str, Dict[str, Any]] = {}
    for name, stage in stage_results.items():
        if not isinstance(name, str):
            continue
        status = _field(stage, "status")
        if not isinstance(status, str) or status not in EXECUTION_STATUSES:
            status = "unavailable"
        entry: Dict[str, Any] = {
            "status": status,
            "outcome": _field(stage, "outcome") if isinstance(_field(stage, "outcome"), str) else None,
            "stage_id": _field(stage, "stage_id") if isinstance(_field(stage, "stage_id"), str) else None,
            "runtime_ms": _field(stage, "runtime_ms") if isinstance(_field(stage, "runtime_ms"), int) else None,
            "error_category": _field(stage, "error_category") if isinstance(_field(stage, "error_category"), str) else None,
        }
        identity = _safe_identity(stage)
        if identity:
            entry["module_identity"] = identity
        summary[name] = entry
    return summary


def _evidence_ref(item: Any) -> Dict[str, Any]:
    if not isinstance(item, EvidenceItem):
        return {"evidence_id": None, "source": None, "reason": "invalid_item_schema"}
    return {
        "evidence_id": item.evidence_id if isinstance(item.evidence_id, str) else None,
        "type": item.type if isinstance(item.type, str) else None,
        "source": item.source if isinstance(item.source, str) else None,
        "stage_name": item.stage_name if isinstance(item.stage_name, str) else None,
        "status": item.status if isinstance(item.status, str) else None,
        "outcome": item.outcome if isinstance(item.outcome, str) else None,
        "direction": item.direction if isinstance(item.direction, str) else None,
        "description": item.description[:500] if isinstance(item.description, str) else "",
    }


def _schema_error(item: Any) -> Optional[str]:
    if not isinstance(item, EvidenceItem):
        return "invalid_item_schema"
    if not isinstance(item.type, str) or item.type not in EVIDENCE_TYPES:
        return "invalid_evidence_type"
    if not isinstance(item.source, str) or not item.source.strip():
        return "missing_source_identity"
    if not isinstance(item.stage_name, str) or not item.stage_name.strip():
        return "missing_stage_identity"
    if not isinstance(item.evidence_id, str) or not item.evidence_id.strip():
        return "missing_evidence_identity"
    if not isinstance(item.description, str) or not item.description.strip():
        return "missing_description"
    if not isinstance(item.strength, str) or item.strength not in STRENGTH_WEIGHTS:
        return "invalid_strength"
    if not isinstance(item.direction, str) or item.direction not in DIRECTION_SCORES:
        return "invalid_direction"
    if not isinstance(item.data, dict):
        return "invalid_data_schema"
    if not isinstance(item.status, str) or item.status not in EXECUTION_STATUSES:
        return "invalid_execution_status"
    if item.outcome is not None and not isinstance(item.outcome, str):
        return "invalid_outcome"
    if not isinstance(item.provenance, str) or not item.provenance.strip():
        return "missing_provenance"
    if not isinstance(item.calibrated, bool):
        return "invalid_calibration_flag"
    if item.confidence is not None and (
        isinstance(item.confidence, bool)
        or not isinstance(item.confidence, (int, float))
        or not math.isfinite(float(item.confidence))
        or not 0.0 <= float(item.confidence) <= 1.0
    ):
        return "invalid_confidence"
    return None


def _fingerprint(item: EvidenceItem) -> str:
    try:
        data = json.dumps(item.data, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        data = "<unserializable>"
    canonical = [item.type, item.source, item.stage_name, item.status, item.outcome,
                 item.direction, item.strength, item.description, data]
    return json.dumps(canonical, ensure_ascii=True, separators=(",", ":"))


def _stage_for_item(stage_results: Optional[Mapping[str, Any]], item: EvidenceItem) -> Tuple[Optional[str], Any]:
    if stage_results is None:
        return None, None
    stage = stage_results.get(item.stage_name)
    if stage is None:
        return "missing_stage_result", None
    status = _field(stage, "status")
    if not isinstance(status, str) or status not in USABLE_STATUSES:
        safe_status = status if isinstance(status, str) and status in EXECUTION_STATUSES else "status_invalid"
        return f"stage_{safe_status}", stage
    if _field(stage, "outcome") == "not_applicable":
        return "stage_not_applicable", stage
    return None, stage


def _calibrated_authenticity_vote(item: EvidenceItem, stage: Any, stage_results_provided: bool) -> Tuple[bool, str]:
    if item.type != "model_evidence":
        return False, "informational_module"
    if item.source != "authenticity_detector" or item.stage_name != "authenticity":
        return False, "unauthorized_model_source"
    if item.direction not in {"authentic", "synthetic", "manipulated"}:
        return False, "model_output_has_no_direction"
    stage_outcome = _field(stage, "outcome")
    if item.outcome in {"no_results", "no_text"} or (
        isinstance(stage_outcome, str) and stage_outcome in {"no_results", "no_text"}
    ):
        return False, "successful_empty_result"
    if not item.calibrated:
        return False, "uncalibrated_model_output"

    data = item.data
    if data.get("calibration_status") != "validated_temperature_scaling":
        return False, "calibration_not_validated"
    if data.get("domain_validation_status") != "validated_for_input_domain":
        return False, "input_domain_not_validated"
    label = data.get("label")
    if not isinstance(label, str) or label not in DIRECTION_LABELS or DIRECTION_LABELS[label] != item.direction:
        return False, "model_label_direction_mismatch"
    if data.get("calibrated_candidate_label") != label:
        return False, "calibrated_candidate_label_mismatch"
    model_identity = data.get("model")
    if not isinstance(model_identity, str) or not model_identity.strip():
        return False, "model_identity_missing"

    probability = data.get("calibrated_ai_generated_probability")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        return False, "calibrated_probability_missing"
    if not math.isfinite(float(probability)) or not 0.0 <= float(probability) <= 1.0:
        return False, "calibrated_probability_invalid"
    if item.confidence is None or not math.isclose(float(item.confidence), float(probability), rel_tol=0.0, abs_tol=1e-6):
        return False, "calibrated_confidence_mismatch"

    if stage_results_provided:
        identity = _field(stage, "module_identity")
        if not isinstance(identity, Mapping):
            return False, "stage_model_identity_missing"
        stage_model_name = identity.get("model_name") or identity.get("model_id")
        if not isinstance(stage_model_name, str) or stage_model_name != model_identity:
            return False, "stage_model_identity_mismatch"
    return True, "eligible_calibrated_authenticity_vote"


@dataclass
class FusionAssessment:
    """Versioned assessment with legacy fields retained for saved reports."""

    label: str = "inconclusive"
    overall_score: Optional[float] = None
    confidence: Optional[float] = None
    calibrated: bool = False
    reasoning: List[str] = field(default_factory=list)
    evidence_summary: Dict[str, int] = field(default_factory=dict)
    conflicting: bool = False
    limitations: List[str] = field(default_factory=list)
    runtime_ms: int = 0
    policy_version: str = POLICY_VERSION
    generated_at: Optional[str] = None
    correlation_id: Optional[str] = None
    evidence_considered: List[Dict[str, Any]] = field(default_factory=list)
    evidence_excluded: List[Dict[str, Any]] = field(default_factory=list)
    supporting_findings: List[Dict[str, Any]] = field(default_factory=list)
    contradictory_findings: List[Dict[str, Any]] = field(default_factory=list)
    conflicting_findings: List[Dict[str, Any]] = field(default_factory=list)
    informational_findings: List[Dict[str, Any]] = field(default_factory=list)
    missing_evidence: List[Dict[str, Any]] = field(default_factory=list)
    stage_execution_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "truthlens.fusion.v1",
            "policy_version": self.policy_version,
            "category": self.label,
            "label": self.label,
            "assessment_confidence": None,
            "uncertainty": {
                "status": "unquantified",
                "reason": "The fused assessment has not been empirically calibrated as a probability.",
            },
            # Retained for historical consumers. These are legacy, uncalibrated
            # margins and must not be presented as probability percentages.
            "overall_score": round(self.overall_score, 4) if self.overall_score is not None else None,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "score_semantics": "Legacy uncalibrated fusion margin; not a probability or validated confidence.",
            "calibrated": False,
            "reasoning": self.reasoning,
            "evidence_summary": self.evidence_summary,
            "evidence_considered": self.evidence_considered,
            "evidence_excluded": self.evidence_excluded,
            "supporting_findings": self.supporting_findings,
            "contradictory_findings": self.contradictory_findings,
            "conflicting_findings": self.conflicting_findings,
            "informational_findings": self.informational_findings,
            "missing_evidence": self.missing_evidence,
            "stage_execution_summary": self.stage_execution_summary,
            "conflicting": self.conflicting,
            "limitations": self.limitations,
            "warnings": self.warnings,
            "generated_at": self.generated_at,
            "correlation_id": self.correlation_id,
            "runtime_ms": self.runtime_ms,
            "confidence_note": "Fusion confidence is uncalibrated and must not be interpreted as a probability.",
            "caveat": (
                "This is an AI-assisted assessment based on available evidence. "
                "Model predictions can be wrong. This result should not be cited "
                "as proof that the image is authentic or fabricated."
            ),
        }


def fuse_evidence(
    evidence: EvidenceCollection,
    *,
    stage_results: Optional[Mapping[str, Any]] = None,
    analysis_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> FusionAssessment:
    """Fuse only schema-valid, completed and explicitly eligible evidence.

    The stage execution contract is supplied by the orchestrator in production.
    Direct unit callers may omit it, in which case each evidence item's own
    canonical status is still enforced.
    """
    t0 = time.perf_counter()
    reasoning: List[str] = []
    warnings: List[str] = []
    limitations: List[str] = [
        "The production fusion weights and thresholds predate E11 and have not been independently validated.",
        "The fused assessment is not calibrated as a probability.",
    ]
    evidence_considered: List[Dict[str, Any]] = []
    evidence_excluded: List[Dict[str, Any]] = []
    informational_findings: List[Dict[str, Any]] = []
    votes: List[Tuple[EvidenceItem, Dict[str, Any]]] = []
    seen_fingerprints: set[str] = set()
    seen_ids: set[str] = set()
    seen_stage_directions: set[Tuple[str, str, str]] = set()
    if stage_results is not None and not isinstance(stage_results, Mapping):
        stage_results = None
    stage_summary = _stage_summary(stage_results)
    items = list(getattr(evidence, "items", []) or []) if evidence is not None else []

    evidence_summary = {
        "total": len(items),
        "authentic_signals": 0,
        "synthetic_signals": 0,
        "manipulated_signals": 0,
        "neutral_signals": 0,
        "informational_signals": 0,
    }

    def exclude(item: Any, reason: str) -> None:
        ref = _evidence_ref(item)
        ref["reason"] = reason
        evidence_excluded.append(ref)
        status = ref.get("status")
        if isinstance(status, str) and status not in USABLE_STATUSES:
            reasoning.append(f"Ignored {status} evidence from {ref.get('source') or 'unknown source'}; excluded as {reason}.")
        else:
            reasoning.append(f"Excluded evidence: {reason}.")

    for item in items:
        schema_error = _schema_error(item)
        if schema_error:
            exclude(item, schema_error)
            continue

        assert isinstance(item, EvidenceItem)
        if analysis_id and item.analysis_id != analysis_id:
            exclude(item, "analysis_id_missing" if not item.analysis_id else "analysis_id_mismatch")
            continue
        if item.outcome == "not_applicable":
            exclude(item, "not_applicable")
            continue
        if item.status not in USABLE_STATUSES:
            exclude(item, f"stage_{item.status}")
            continue
        stage_error, stage = _stage_for_item(stage_results, item)
        if stage_error:
            exclude(item, stage_error)
            continue
        stage_correlation = _field(stage, "correlation_id") if stage is not None else None
        if analysis_id and stage is not None:
            if not stage_correlation:
                exclude(item, "stage_correlation_id_missing")
                continue
            if stage_correlation != analysis_id:
                exclude(item, "stage_correlation_id_mismatch")
                continue

        if item.evidence_id in seen_ids or _fingerprint(item) in seen_fingerprints:
            exclude(item, "duplicate_evidence")
            continue
        seen_ids.add(item.evidence_id)
        seen_fingerprints.add(_fingerprint(item))

        ref = _evidence_ref(item)
        is_vote, vote_reason = _calibrated_authenticity_vote(
            item, stage, stage_results is not None,
        )
        if item.type == "model_evidence" and vote_reason == "unauthorized_model_source":
            exclude(item, vote_reason)
            continue
        if item.type == "model_evidence" and item.direction in {"authentic", "synthetic", "manipulated"} and not is_vote:
            # Keep raw or not-domain-validated classifier output visible, but
            # explicitly neutralize it for production assessment purposes.
            ref["direction"] = "informational"
            ref["reason"] = vote_reason
            informational_findings.append(ref)
            evidence_considered.append({**ref, "disposition": "informational"})
            evidence_summary["neutral_signals"] += 1
            reasoning.append(f"Ignored directional classifier output: {vote_reason}.")
            continue

        if not is_vote:
            ref["direction"] = "informational"
            ref["reason"] = vote_reason
            informational_findings.append(ref)
            evidence_considered.append({**ref, "disposition": "informational"})
            evidence_summary["informational_signals"] += 1
            continue

        stage_direction = (item.stage_name, item.source, item.direction)
        if stage_direction in seen_stage_directions:
            exclude(item, "duplicate_module_direction")
            continue
        seen_stage_directions.add(stage_direction)
        votes.append((item, ref))
        evidence_considered.append({**ref, "disposition": "directional_vote", "eligibility": vote_reason})
        if item.direction == "authentic":
            evidence_summary["authentic_signals"] += 1
        elif item.direction == "synthetic":
            evidence_summary["synthetic_signals"] += 1
        elif item.direction == "manipulated":
            evidence_summary["manipulated_signals"] += 1

    # Existing weighted policy, applied only to eligible directional votes.
    authentic_score = sum(STRENGTH_WEIGHTS[item.strength] for item, _ in votes if item.direction == "authentic")
    synthetic_score = sum(STRENGTH_WEIGHTS[item.strength] for item, _ in votes if item.direction == "synthetic")
    manipulated_score = sum(STRENGTH_WEIGHTS[item.strength] for item, _ in votes if item.direction == "manipulated")
    total_weight = sum(STRENGTH_WEIGHTS[item.strength] for item, _ in votes)
    negative_score = synthetic_score + manipulated_score
    conflicting = authentic_score >= CONFLICT_THRESHOLD and negative_score >= CONFLICT_THRESHOLD

    if conflicting:
        conflicting_findings = [ref for item, ref in votes if item.direction in {"authentic", "synthetic", "manipulated"}]
        reasoning.append("Eligible calibrated evidence contains opposing authenticity directions.")
    else:
        conflicting_findings = []

    if conflicting and abs(authentic_score - negative_score) < DETERMINATION_THRESHOLD:
        label, overall_score, confidence = "inconclusive", None, None
        reasoning.append("Conflicting eligible evidence prevents a reliable determination.")
    elif total_weight < 0.1:
        label, overall_score, confidence = "inconclusive", None, None
        reasoning.append("Insufficient eligible directional evidence for a determination.")
    elif authentic_score + negative_score == 0:
        label, overall_score, confidence = "inconclusive", None, None
        reasoning.append("No calibrated, eligible directional evidence is available.")
    elif authentic_score > negative_score + DETERMINATION_THRESHOLD:
        label = "likely_authentic"
        overall_score = authentic_score / (authentic_score + negative_score)
        confidence = min(authentic_score - negative_score, 1.0)
        reasoning.append("Eligible evidence leans toward authentic.")
    elif synthetic_score > manipulated_score:
        label = "likely_ai_generated"
        overall_score = 1.0 - (synthetic_score / (authentic_score + negative_score))
        confidence = min(negative_score - authentic_score, 1.0)
        reasoning.append("Eligible evidence leans toward AI-generated.")
    elif manipulated_score > 0:
        label = "likely_manipulated"
        overall_score = 1.0 - (manipulated_score / (authentic_score + negative_score))
        confidence = min(negative_score - authentic_score, 1.0)
        reasoning.append("Eligible evidence leans toward manipulation.")
    else:
        label, overall_score, confidence = "inconclusive", None, None
        reasoning.append("No clear direction from eligible evidence.")

    if overall_score is not None:
        overall_score = max(0.0, min(1.0, overall_score))
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))

    supporting_findings: List[Dict[str, Any]] = []
    contradictory_findings: List[Dict[str, Any]] = []
    if label != "inconclusive":
        winning_direction = DIRECTION_LABELS[label]
        for item, ref in votes:
            if item.direction == winning_direction or (
                winning_direction in {"synthetic", "manipulated"}
                and item.direction in {"synthetic", "manipulated"}
            ):
                supporting_findings.append(ref)
            else:
                contradictory_findings.append(ref)

    missing_evidence: List[Dict[str, Any]] = []
    for name, stage in stage_summary.items():
        status = stage["status"]
        outcome = stage["outcome"]
        if status in {"failed", "unavailable", "skipped", "partial"} or outcome in {"no_results", "no_text", "not_applicable"}:
            missing_evidence.append({
                "stage_name": name,
                "status": status,
                "outcome": outcome,
                "evidence_available": False if outcome in {"no_results", "no_text", "not_applicable"} or status in {"failed", "unavailable", "skipped"} else None,
                "reason": "Stage did not provide usable evidence; this absence is not negative evidence.",
            })

    if missing_evidence:
        warnings.append("One or more stages were incomplete, unavailable, skipped, or returned no evidence.")
    if not votes:
        warnings.append("No validated directional evidence was available; the result remains inconclusive.")
    limitations.append("OCR, language, metadata, forensics, object detection, fact-check, provenance, and XAI findings are informational in E11.")
    limitations.append("Third-party ratings and provenance metadata are not TruthLens truth labels or proof of authenticity.")

    if evidence_summary["informational_signals"] > sum(evidence_summary[k] for k in ("authentic_signals", "synthetic_signals", "manipulated_signals")):
        limitations.append("Most available findings are informational rather than directional.")

    runtime_ms = int((time.perf_counter() - t0) * 1000)
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
        generated_at=datetime.now(timezone.utc).isoformat(),
        correlation_id=correlation_id or analysis_id,
        evidence_considered=evidence_considered,
        evidence_excluded=evidence_excluded,
        supporting_findings=supporting_findings,
        contradictory_findings=contradictory_findings,
        conflicting_findings=conflicting_findings,
        informational_findings=informational_findings,
        missing_evidence=missing_evidence,
        stage_execution_summary=stage_summary,
        warnings=warnings,
    )
    log.info("Fusion: label=%s eligible_votes=%d informational=%d excluded=%d conflicting=%s [%d ms]",
             label, len(votes), len(informational_findings), len(evidence_excluded), conflicting, runtime_ms)
    return result
