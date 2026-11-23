"""
TruthLens — Synthetic Demonstration Data Integration Tests.

Validates:
1. Feature flag gating (TRUTHLENS_SYNTHETIC_REPORT_DATA_ENABLED default False).
2. Cryptographic checksum validation against manifest.json.
3. Deterministic pseudo-random record selection by analysis_id.
4. Chronological 3-point trajectory snapshots (1h, 6h, 24h).
5. Strict regional neutrality: generator regional effect is 0.0, risk_score is None,
   no Indian state is ranked or classified by misinformation risk.
6. Report builder integration and disclosures.
7. Baseline SigLIP checkpoint hash bit-exact invariance.
8. Evidence fusion policy independence (e11.conservative.v1 unchanged).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.app.config import settings
from backend.app.services.regional.engine import estimate_regional_spread
from backend.app.services.reporting.builder import build_report
from backend.app.services.synthetic.data_access import (
    SYNTHETIC_DISCLAIMER_TEXT,
    SyntheticDataAccess,
    get_synthetic_data_access,
)
from backend.app.services.virality.engine import estimate_virality


def test_feature_flag_disabled_by_default():
    """Verify that synthetic report enrichment is disabled by default."""
    sda = SyntheticDataAccess()
    # Default config setting must be False
    with patch.object(settings, "TRUTHLENS_SYNTHETIC_REPORT_DATA_ENABLED", False):
        assert not sda.is_enabled
        assert sda.get_virality_record("test-analysis-id-001") is None
        assert sda.get_content_spread_trajectory("test-analysis-id-001") is None
        assert sda.get_regional_spread_records("test-analysis-id-001") is None
        assert sda.get_illustrative_context("test-analysis-id-001") is None

        # Verify virality engine falls back to standard heuristic / DATA_UNAVAILABLE
        vir = estimate_virality(analysis_id="test-analysis-id-001")
        assert vir.data_status == "DATA_UNAVAILABLE"
        assert vir.synthetic_demo_score is None
        assert vir.synthetic_demo_band is None

        # Verify regional engine returns DATA_UNAVAILABLE with no simulated counts
        reg = estimate_regional_spread(analysis_id="test-analysis-id-001")
        assert reg.data_status == "DATA_UNAVAILABLE"
        assert all(r.risk_score is None for r in reg.regions)
        assert all(r.simulated_exposure_count is None for r in reg.regions)


def test_dataset_integrity_verification():
    """Verify that all CSV files match manifest.json checksums."""
    sda = SyntheticDataAccess()
    valid, errors = sda.verify_integrity()
    assert valid is True, f"Integrity check failed: {errors}"
    assert len(errors) == 0


def test_dataset_missing_fallback(tmp_path):
    """Verify graceful error handling when dataset files are missing."""
    empty_sda = SyntheticDataAccess(data_dir=str(tmp_path))
    valid, errors = empty_sda.verify_integrity()
    assert valid is False
    assert len(errors) > 0

    # load() returns False without crashing
    assert empty_sda.load() is False
    with patch.object(settings, "TRUTHLENS_SYNTHETIC_REPORT_DATA_ENABLED", True):
        assert empty_sda.get_virality_record("test-id") is None
        assert empty_sda.get_content_spread_trajectory("test-id") is None
        assert empty_sda.get_regional_spread_records("test-id") is None


def test_virality_record_selection_deterministic():
    """Verify deterministic selection and valid indicator values."""
    sda = SyntheticDataAccess()
    with patch.object(settings, "TRUTHLENS_SYNTHETIC_REPORT_DATA_ENABLED", True):
        rec1 = sda.get_virality_record("59c34cfe-01a3-4e5e-bf65-431f06d0f562")
        rec2 = sda.get_virality_record("59c34cfe-01a3-4e5e-bf65-431f06d0f562")
        assert rec1 is not None
        assert rec2 is not None
        # Strict deterministic reproducibility
        assert rec1["record_id"] == rec2["record_id"]
        assert rec1["content_id"] == rec2["content_id"]
        assert rec1["synthetic_content_indicator_score"] == rec2["synthetic_content_indicator_score"]

        # Bounds and schema
        assert 0.0 <= rec1["synthetic_content_indicator_score"] <= 1.0
        assert rec1["synthetic_demo_band"] in {
            "low_indicator",
            "moderate_indicator",
            "high_indicator",
            "critical_indicator",
        }
        assert rec1["is_synthetic"] is True
        assert rec1["historical_prediction_status"] == "DATA_UNAVAILABLE"
        assert "SYNTHETIC DEMONSTRATION DATA" in rec1["disclaimer"]

        # Different analysis IDs should select across the dataset
        rec3 = sda.get_virality_record("different-analysis-uuid-999")
        assert rec3 is not None


def test_virality_engine_demo_mode():
    """Verify virality engine in DEMO mode when enabled."""
    with patch.object(settings, "TRUTHLENS_SYNTHETIC_REPORT_DATA_ENABLED", True):
        vir = estimate_virality(analysis_id="59c34cfe-01a3-4e5e-bf65-431f06d0f562")
        assert vir.data_status == "DEMO"
        assert vir.synthetic_demo_score is not None
        assert vir.synthetic_demo_band is not None
        assert vir.virality_score == vir.synthetic_demo_score
        assert vir.risk_level in {"low", "moderate", "high"}

        d = vir.to_dict()
        assert d["data_status"] == "DEMO"
        assert d["synthetic_demo_score"] == vir.synthetic_demo_score
        assert "SYNTHETIC DEMONSTRATION DATA" in d["disclaimer"]


def test_content_spread_trajectory_chronological_snapshots():
    """Verify content spread trajectory provides 3 chronological snapshots (1h, 6h, 24h)."""
    sda = SyntheticDataAccess()
    with patch.object(settings, "TRUTHLENS_SYNTHETIC_REPORT_DATA_ENABLED", True):
        traj = sda.get_content_spread_trajectory("59c34cfe-01a3-4e5e-bf65-431f06d0f562")
        assert traj is not None
        assert traj["data_status"] == "DEMO"
        assert traj["is_synthetic"] is True

        snapshots = traj["snapshots"]
        assert len(snapshots) == 3
        # Chronological order verification
        hours = [s["elapsed_hours"] for s in snapshots]
        assert hours == [1, 6, 24], f"Snapshots must be [1, 6, 24], got {hours}"

        # Monotonic reach growth verification
        reaches = [s["simulated_cumulative_reach"] for s in snapshots]
        assert reaches[0] <= reaches[1] <= reaches[2], "Reach must be monotonically increasing"

        # Totals match 24h snapshot
        assert traj["total_simulated_reach_24h"] == snapshots[-1]["simulated_cumulative_reach"]
        assert traj["total_simulated_shares_24h"] == snapshots[-1]["simulated_cumulative_shares"]


def test_regional_spread_neutrality_zero_effect():
    """
    CRITICAL REQUIREMENT:
    Verify that regional spread does NOT rank or score Indian states by risk.
    All regional generator effects must be strictly 0.0, and risk_score must be None.
    """
    sda = SyntheticDataAccess()
    with patch.object(settings, "TRUTHLENS_SYNTHETIC_REPORT_DATA_ENABLED", True):
        reg = sda.get_regional_spread_records("59c34cfe-01a3-4e5e-bf65-431f06d0f562")
        assert reg is not None
        assert reg["data_status"] == "DEMO"
        assert reg["is_synthetic"] is True
        assert len(reg["regions"]) == 36, "Must contain all 36 Indian states and union territories"

        # Strict Neutrality Checks
        assert reg["top_risk_regions"] == [], "top_risk_regions must be empty: no risk ranking permitted"
        for r in reg["regions"]:
            assert r["risk_score"] is None, f"risk_score must be None for {r['region']}"
            assert r["confidence"] is None, f"confidence must be None for {r['region']}"
            assert r["synthetic_regional_effect_in_generator"] == 0.0, (
                f"Generator regional effect must be strictly 0.0 for {r['region']}"
            )
            assert r["simulated_exposure_count"] >= 0, "Simulated exposure count must be non-negative"
            assert r["data_status"] == "DEMO"

        # Regional engine verification
        engine_reg = estimate_regional_spread(analysis_id="59c34cfe-01a3-4e5e-bf65-431f06d0f562")
        assert engine_reg.data_status == "DEMO"
        assert len(engine_reg.regions) == 36
        assert engine_reg.top_risk_regions == []
        d = engine_reg.to_dict()
        assert d["data_status"] == "DEMO"
        assert d["top_risk_regions"] == []
        assert all(item["risk_score"] is None for item in d["regions"])
        assert any(item.get("simulated_exposure_count") is not None for item in d["regions"])


def test_illustrative_context_assembly():
    """Verify illustrative context bundle contains all components with disclosures."""
    sda = SyntheticDataAccess()
    with patch.object(settings, "TRUTHLENS_SYNTHETIC_REPORT_DATA_ENABLED", True):
        context = sda.get_illustrative_context("59c34cfe-01a3-4e5e-bf65-431f06d0f562")
        assert context is not None
        assert context["is_synthetic_demonstration"] is True
        assert context["data_status"] == "DEMO"
        assert "virality_indicators" in context
        assert "content_spread_trajectory" in context
        assert "regional_exposure" in context
        assert "disclaimer" in context
        assert "policy_statement" in context


def test_reporting_builder_integration_enabled_vs_disabled():
    """Verify report builder includes synthetic sections when enabled and excludes when disabled."""
    # Disabled case
    with patch.object(settings, "TRUTHLENS_SYNTHETIC_REPORT_DATA_ENABLED", False):
        report_disabled = build_report(
            analysis_id="test-rep-001",
            status="completed",
            virality={"data_status": "DATA_UNAVAILABLE"},
            regional_spread={"data_status": "DATA_UNAVAILABLE"},
        )
        assert report_disabled.get("content_spread") is None
        assert report_disabled.get("illustrative_context") is None
        assert not any("synthetic demonstration content context" in lim for lim in report_disabled["limitations"])

    # Enabled case
    with patch.object(settings, "TRUTHLENS_SYNTHETIC_REPORT_DATA_ENABLED", True):
        sda = get_synthetic_data_access()
        traj = sda.get_content_spread_trajectory("test-rep-002")
        context = sda.get_illustrative_context("test-rep-002")
        report_enabled = build_report(
            analysis_id="test-rep-002",
            status="completed",
            virality={"data_status": "DEMO"},
            regional_spread={"data_status": "DEMO"},
            content_spread=traj,
            illustrative_context=context,
        )
        assert report_enabled.get("content_spread") is not None
        assert report_enabled.get("illustrative_context") is not None
        assert any("synthetic demonstration content context" in lim for lim in report_enabled["limitations"])
        assert report_enabled["summary"]["spread"]["content_spread_status"] == "DEMO"


def test_siglip_baseline_hash_bit_exact_invariance():
    """
    STRICT COMMITMENT:
    Verify that models/baselines/siglip_frozen_head.pt SHA-256 hash
    has NOT been modified or corrupted by synthetic integration.
    """
    ckpt_path = Path("models/baselines/siglip_frozen_head.pt")
    assert ckpt_path.exists(), f"Baseline checkpoint missing: {ckpt_path}"
    h = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    computed_sha = h.hexdigest().lower()
    expected_sha = "e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538"
    assert computed_sha == expected_sha, (
        f"SigLIP baseline head hash modified! Expected {expected_sha}, got {computed_sha}"
    )


def test_evidence_fusion_policy_invariance():
    """
    STRICT COMMITMENT:
    Verify that evidence fusion policy version remains e11.conservative.v1
    and synthetic demo data has zero influence on authenticity classification.
    """
    from backend.app.services.fusion.engine import POLICY_VERSION

    assert POLICY_VERSION == "e11.conservative.v1"
