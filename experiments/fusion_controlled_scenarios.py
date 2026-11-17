"""Deterministic rule-path checks for fusion, not a predictive ablation study."""
from __future__ import annotations
import json
import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.services.evidence.schema import EvidenceCollection, EvidenceItem
from backend.app.services.fusion.engine import fuse_evidence


def item(kind, source, direction, status="completed", description=None):
    return EvidenceItem(type=kind, source=source, strength="high", direction=direction,
                        description=description or source, status=status)


def main():
    configs = {
        "A_model_only": [item("model_evidence", "model", "synthetic")],
        "B_model_plus_forensics": [item("model_evidence", "model", "synthetic"), item("forensic_evidence", "ela", "informational")],
        "C_model_forensics_ocr": [item("model_evidence", "model", "synthetic"), item("forensic_evidence", "ela", "informational"), item("ocr_evidence", "ocr", "informational")],
        "D_plus_provenance": [item("model_evidence", "model", "synthetic"), item("forensic_evidence", "ela", "informational"), item("provenance_evidence", "web", "informational")],
        "E_full_with_failed_detector": [item("model_evidence", "model", "synthetic"), item("forensic_evidence", "ela", "informational"), item("ocr_evidence", "ocr", "informational"), item("model_evidence", "failed_model", "authentic", "failed")],
        "F_no_directional_signals": [item("forensic_evidence", "ela", "synthetic")],
    }
    results = {}
    for name, items in configs.items():
        result = fuse_evidence(EvidenceCollection(items=items)).to_dict()
        results[name] = {"input_item_count": len(items), "assessment": result["label"], "score": result["overall_score"], "confidence": result["confidence"], "conflicting": result["conflicting"], "evidence_summary": result["evidence_summary"], "reasoning": result["reasoning"]}
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
        entries = []
        for line in status.splitlines():
            if line[3:].replace("\\", "/").startswith("experiments/results/"):
                continue
            path = ROOT / line[3:]
            content = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing-or-nonfile"
            entries.append(f"{line}\0{content}")
        worktree = {"dirty": bool(status.strip()), "status_sha256": hashlib.sha256(status.encode()).hexdigest(), "worktree_content_sha256": hashlib.sha256("\n".join(entries).encode()).hexdigest()}
    except Exception: rev, worktree = None, {"dirty": None, "status_sha256": None}
    output = {"created_utc": datetime.now(timezone.utc).isoformat(), "revision": rev, "worktree": worktree, "python": platform.python_version(), "status": "CONTROLLED_LOGIC_CHECK_ONLY", "not_a_dataset": True, "results": results, "interpretation": "Forensics/OCR/provenance evidence is informational in authenticity fusion and does not change the model-only direction; failed detector evidence is excluded. This tests code rules only, not whether the model or signals are correct."}
    path = ROOT / "experiments/results/fusion_controlled_scenarios.json"; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8"); print(json.dumps(output, indent=2))


if __name__ == "__main__": main()
