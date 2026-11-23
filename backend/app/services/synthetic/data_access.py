"""
TruthLens — Synthetic Demonstration Data Access Layer.

Provides lightweight, verified, and encapsulated access to the synthetic demo
datasets for college presentation and report enrichment:
- virality_indicators_synthetic.csv
- content_spread_trajectories_synthetic.csv
- regional_spread_synthetic.csv

DISCLAIMER & INTEGRITY CONSTRAINTS:
- STRICTLY FOR ILLUSTRATIVE DEMONSTRATION IN ANALYSIS REPORTS.
- All records are fictional and synthetic.
- Not observed social media metrics.
- Not evaluation or calibration data for detection models.
- NEVER modifies detector classification, confidence, or evidence fusion verdicts.
- Neutral regional presentation: regional generator effect is strictly zero.
  No Indian state is ranked or classified by misinformation risk.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.app.config import settings
from backend.app.core.logging import get_logger

log = get_logger("synthetic.data_access")

SYNTHETIC_DISCLAIMER_TEXT = (
    "⚠️ SYNTHETIC DEMONSTRATION DATA: The content indicators, spread trajectories, "
    "and regional exposures shown are generated synthetic demo data intended solely for "
    "illustrative and academic demonstration. They are NOT observed social media metrics, "
    "do NOT evaluate detector accuracy, and do NOT influence the authenticity classification."
)


class SyntheticDataAccess:
    """Encapsulated data-access service for synthetic demonstration datasets."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir or settings.SYNTHETIC_DATA_DIR)
        self._manifest: Optional[Dict[str, Any]] = None
        self._virality_records: List[Dict[str, Any]] = []
        self._trajectories_by_content_id: Dict[str, List[Dict[str, Any]]] = {}
        self._trajectory_content_ids: List[str] = []
        self._regional_by_content_id: Dict[str, List[Dict[str, Any]]] = {}
        self._regional_content_ids: List[str] = []
        self._loaded: bool = False
        self._load_error: Optional[str] = None

    @property
    def is_enabled(self) -> bool:
        """Return True if synthetic report data enrichment is enabled."""
        return bool(settings.TRUTHLENS_SYNTHETIC_REPORT_DATA_ENABLED)

    def verify_integrity(self) -> Tuple[bool, List[str]]:
        """
        Verify the existence and SHA-256 hashes of the dataset files
        against the packaged manifest.json.
        """
        errors: List[str] = []
        manifest_path = self.data_dir / "manifest.json"

        if not self.data_dir.exists():
            return False, [f"Synthetic dataset directory not found: {self.data_dir}"]

        if not manifest_path.exists():
            return False, [f"Dataset manifest not found: {manifest_path}"]

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as exc:
            return False, [f"Failed to read dataset manifest: {exc}"]

        files_meta = manifest.get("files", {})
        if not files_meta:
            return False, ["Manifest does not contain 'files' metadata."]

        for filename, expected_meta in files_meta.items():
            file_path = self.data_dir / filename
            if not file_path.exists():
                errors.append(f"Missing dataset file: {filename}")
                continue

            expected_sha = expected_meta.get("sha256")
            if expected_sha:
                h = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
                computed_sha = h.hexdigest().lower()
                if computed_sha != expected_sha.lower():
                    errors.append(
                        f"Checksum mismatch for {filename}: expected {expected_sha}, got {computed_sha}"
                    )

        return len(errors) == 0, errors

    def load(self, force: bool = False) -> bool:
        """
        Lazy-load and parse the CSV datasets into memory.
        Returns True on success, False if loading fails.
        """
        if self._loaded and not force:
            return True

        valid, errors = self.verify_integrity()
        if not valid:
            self._load_error = "; ".join(errors)
            log.warning("Synthetic dataset integrity check failed: %s", self._load_error)
            return False

        try:
            # 1. Load virality indicators
            virality_path = self.data_dir / "virality_indicators_synthetic.csv"
            virality_rows: List[Dict[str, Any]] = []
            with open(virality_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    virality_rows.append({
                        "record_id": row.get("record_id"),
                        "content_id": row.get("content_id"),
                        "is_synthetic": True,
                        "generator_version": row.get("generator_version"),
                        "language_code": row.get("language_code"),
                        "topic_category": row.get("topic_category"),
                        "object_count": int(row.get("object_count") or 0),
                        "has_faces": int(row.get("has_faces") or 0) == 1,
                        "has_text_overlay": int(row.get("has_text_overlay") or 0) == 1,
                        "emotional_language_score": float(row.get("emotional_language_score") or 0.0),
                        "has_claims": int(row.get("has_claims") or 0) == 1,
                        "has_numbers": int(row.get("has_numbers") or 0) == 1,
                        "multiple_objects_score": float(row.get("multiple_objects_score") or 0.0),
                        "health_topic": int(row.get("health_topic") or 0) == 1,
                        "political_topic": int(row.get("political_topic") or 0) == 1,
                        "religious_topic": int(row.get("religious_topic") or 0) == 1,
                        "urgency_language_score": float(row.get("urgency_language_score") or 0.0),
                        "synthetic_content_indicator_score": round(float(row.get("synthetic_content_indicator_score") or 0.0), 4),
                        "synthetic_demo_band": row.get("synthetic_demo_band") or "moderate_indicator",
                        "historical_prediction_status": row.get("historical_prediction_status") or "DATA_UNAVAILABLE",
                    })
            self._virality_records = virality_rows

            # 2. Load content spread trajectories
            traj_path = self.data_dir / "content_spread_trajectories_synthetic.csv"
            trajectories_by_id: Dict[str, List[Dict[str, Any]]] = {}
            with open(traj_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row.get("content_id")
                    if not cid:
                        continue
                    if cid not in trajectories_by_id:
                        trajectories_by_id[cid] = []
                    trajectories_by_id[cid].append({
                        "record_id": row.get("record_id"),
                        "content_id": cid,
                        "is_synthetic": True,
                        "generator_version": row.get("generator_version"),
                        "language_code": row.get("language_code"),
                        "topic_category": row.get("topic_category"),
                        "platform_label": row.get("platform_label"),
                        "content_format": row.get("content_format"),
                        "elapsed_hours": int(row.get("elapsed_hours") or 0),
                        "simulated_cumulative_reach": int(row.get("simulated_cumulative_reach") or 0),
                        "simulated_cumulative_shares": int(row.get("simulated_cumulative_shares") or 0),
                        "simulated_cumulative_reposts": int(row.get("simulated_cumulative_reposts") or 0),
                        "simulated_cumulative_replies": int(row.get("simulated_cumulative_replies") or 0),
                        "simulated_engagement_rate": round(float(row.get("simulated_engagement_rate") or 0.0), 4),
                        "synthetic_demo_curve": row.get("synthetic_demo_curve"),
                        "observed_outcome_status": row.get("observed_outcome_status") or "SYNTHETIC_ONLY_NOT_OBSERVED",
                    })

            # Sort each trajectory by elapsed_hours ascending ([1, 6, 24])
            for cid in trajectories_by_id:
                trajectories_by_id[cid].sort(key=lambda item: item["elapsed_hours"])

            self._trajectories_by_content_id = trajectories_by_id
            self._trajectory_content_ids = sorted(trajectories_by_id.keys())

            # 3. Load regional spread
            reg_path = self.data_dir / "regional_spread_synthetic.csv"
            regional_by_id: Dict[str, List[Dict[str, Any]]] = {}
            with open(reg_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row.get("content_id")
                    if not cid:
                        continue
                    if cid not in regional_by_id:
                        regional_by_id[cid] = []
                    regional_by_id[cid].append({
                        "record_id": row.get("record_id"),
                        "content_id": cid,
                        "region_id": row.get("region_id"),
                        "region_name": row.get("region_name"),
                        "is_synthetic": True,
                        "generator_version": row.get("generator_version"),
                        "language_code": row.get("language_code"),
                        "topic_category": row.get("topic_category"),
                        "platform_label": row.get("platform_label"),
                        "elapsed_hours": int(row.get("elapsed_hours") or 0),
                        "simulated_exposure_count": int(row.get("simulated_exposure_count") or 0),
                        "simulated_share_count": int(row.get("simulated_share_count") or 0),
                        "simulated_repost_count": int(row.get("simulated_repost_count") or 0),
                        "synthetic_regional_effect_in_generator": float(row.get("synthetic_regional_effect_in_generator") or 0.0),
                        "regional_prediction_status": row.get("regional_prediction_status") or "DATA_UNAVAILABLE",
                        "observed_outcome_status": row.get("observed_outcome_status") or "SYNTHETIC_ONLY_NOT_OBSERVED",
                    })

            # Filter to content_ids with full 36 regions
            full_regional_ids = [cid for cid, rows in regional_by_id.items() if len(rows) == 36]
            self._regional_by_content_id = regional_by_id
            self._regional_content_ids = sorted(full_regional_ids) or sorted(regional_by_id.keys())

            self._loaded = True
            self._load_error = None
            log.info(
                "Synthetic demonstration datasets loaded: %d virality records, %d trajectories, %d regional sets",
                len(self._virality_records),
                len(self._trajectory_content_ids),
                len(self._regional_content_ids),
            )
            return True

        except Exception as exc:
            self._load_error = str(exc)
            log.exception("Error loading synthetic demonstration datasets: %s", exc)
            return False

    def _get_seed(self, analysis_id: str) -> int:
        """Derive a stable, deterministic integer seed from an analysis UUID/string."""
        raw_hash = hashlib.sha256(analysis_id.encode("utf-8")).hexdigest()
        return int(raw_hash[:8], 16)

    def get_virality_record(
        self, analysis_id: str, language_code: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Deterministically select a synthetic virality indicator record for this analysis.
        Returns None if synthetic data is disabled or unavailable.
        """
        if not self.is_enabled:
            return None
        if not self.load():
            return None
        if not self._virality_records:
            return None

        seed = self._get_seed(analysis_id)
        candidates = self._virality_records

        # If a matching language is requested, filter candidates if matches exist
        if language_code:
            norm_lang = language_code.strip().lower()
            lang_matches = [r for r in candidates if r.get("language_code") == norm_lang]
            if lang_matches:
                candidates = lang_matches

        idx = seed % len(candidates)
        record = dict(candidates[idx])
        record["disclaimer"] = SYNTHETIC_DISCLAIMER_TEXT
        return record

    def get_content_spread_trajectory(
        self, analysis_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Deterministically select a 3-point content spread trajectory (1h, 6h, 24h)
        for this analysis. Returns None if disabled or unavailable.
        """
        if not self.is_enabled:
            return None
        if not self.load():
            return None
        if not self._trajectory_content_ids:
            return None

        seed = self._get_seed(analysis_id)
        cid_idx = (seed // 13) % len(self._trajectory_content_ids)
        chosen_cid = self._trajectory_content_ids[cid_idx]
        snapshots = [dict(s) for s in self._trajectories_by_content_id.get(chosen_cid, [])]

        if not snapshots:
            return None

        # Build trajectory payload
        latest_snapshot = snapshots[-1]
        return {
            "content_id": chosen_cid,
            "data_status": "DEMO",
            "is_synthetic": True,
            "trajectory_curve": latest_snapshot.get("synthetic_demo_curve"),
            "platform_context": latest_snapshot.get("platform_label"),
            "snapshots": snapshots,
            "total_simulated_reach_24h": latest_snapshot.get("simulated_cumulative_reach"),
            "total_simulated_shares_24h": latest_snapshot.get("simulated_cumulative_shares"),
            "total_simulated_reposts_24h": latest_snapshot.get("simulated_cumulative_reposts"),
            "total_simulated_replies_24h": latest_snapshot.get("simulated_cumulative_replies"),
            "simulated_engagement_rate_24h": latest_snapshot.get("simulated_engagement_rate"),
            "methodology": "Synthetic demonstration curve generated for timeline visualization. Fictional metrics only.",
            "disclaimer": SYNTHETIC_DISCLAIMER_TEXT,
        }

    def get_regional_spread_records(
        self, analysis_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Deterministically select 36 regional exposure records for this analysis.
        STRICT REQUIREMENT: All regional effects are 0 in the generator.
        No state risk ranking or risk scoring is performed.
        Returns None if disabled or unavailable.
        """
        if not self.is_enabled:
            return None
        if not self.load():
            return None
        if not self._regional_content_ids:
            return None

        seed = self._get_seed(analysis_id)
        cid_idx = (seed // 37) % len(self._regional_content_ids)
        chosen_cid = self._regional_content_ids[cid_idx]
        raw_regions = self._regional_by_content_id.get(chosen_cid, [])

        if not raw_regions:
            return None

        regions = []
        for r in raw_regions:
            regions.append({
                "region": r.get("region_name"),
                "region_id": r.get("region_id"),
                "simulated_exposure_count": r.get("simulated_exposure_count"),
                "simulated_share_count": r.get("simulated_share_count"),
                "simulated_repost_count": r.get("simulated_repost_count"),
                "synthetic_regional_effect_in_generator": 0.0,
                # Explicitly null risk scores to prevent risk ranking:
                "risk_score": None,
                "confidence": None,
                "language_compatibility": None,
                "internet_penetration": None,
                "data_status": "DEMO",
                "evidence": [
                    "Synthetic demonstration exposure count for college presentation. "
                    "Generator regional effect is 0; no regional risk ranking is imputed."
                ],
            })

        # Sort alphabetically by region name for neutral presentation
        regions.sort(key=lambda item: item["region"])

        return {
            "content_id": chosen_cid,
            "data_status": "DEMO",
            "is_synthetic": True,
            "regions": regions,
            "top_risk_regions": [],  # Strictly empty: zero risk ranking permitted
            "methodology": (
                "Simulated regional distribution counts from synthetic demonstration data for academic presentation. "
                "Regional differences are not validated or representative of actual Indian states or populations. "
                "The dataset generator regional effect is strictly 0.0; no regional risk score, confidence score, "
                "susceptibility, prevalence, or threat ranking is calculated or imputed."
            ),
            "disclaimer": (
                "⚠️ SYNTHETIC DEMONSTRATION DATA: Simulated state-level counts for college demonstration only. "
                "All regional generator effects are zero. Regional differences are not validated or representative "
                "of actual Indian states or populations. This data does NOT represent actual misinformation spread, "
                "and no regional risk, susceptibility, prevalence, or threat ranking is calculated or displayed."
            ),
        }

    def get_illustrative_context(
        self, analysis_id: str, language_code: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Assemble the complete illustrative content context bundle for report inclusion.
        Returns None if synthetic reporting is disabled.
        """
        if not self.is_enabled:
            return None

        virality_rec = self.get_virality_record(analysis_id, language_code)
        trajectory = self.get_content_spread_trajectory(analysis_id)
        regional = self.get_regional_spread_records(analysis_id)

        if not virality_rec and not trajectory and not regional:
            return None

        return {
            "is_synthetic_demonstration": True,
            "data_status": "DEMO",
            "analysis_id": analysis_id,
            "virality_indicators": virality_rec,
            "content_spread_trajectory": trajectory,
            "regional_exposure": regional,
            "banner": "⚠️ SYNTHETIC DEMONSTRATION DATA — FOR COLLEGE PRESENTATION ONLY",
            "disclaimer": SYNTHETIC_DISCLAIMER_TEXT,
            "policy_statement": (
                "Synthetic data is strictly isolated from core verification. "
                "It does not evaluate detector accuracy, does not calibrate models, "
                "and does not alter the authenticity classification or evidence fusion verdict."
            ),
        }


# Global singleton instance
_synthetic_data_access: Optional[SyntheticDataAccess] = None


def get_synthetic_data_access() -> SyntheticDataAccess:
    """Return the global SyntheticDataAccess singleton instance."""
    global _synthetic_data_access
    if _synthetic_data_access is None:
        _synthetic_data_access = SyntheticDataAccess()
    return _synthetic_data_access
