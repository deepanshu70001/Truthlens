"""Comprehensive integrity and schema validation for synthetic demonstration datasets."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

DATA_DIR = Path("datasets/synthetic_demo")
MANIFEST_PATH = DATA_DIR / "manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    print("=" * 70)
    print("TruthLens Synthetic Demonstration Dataset Validation")
    print("=" * 70)

    if not MANIFEST_PATH.exists():
        print(f"FAIL: Manifest not found at {MANIFEST_PATH}")
        return 1

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    files_meta = manifest.get("files", {})
    all_passed = True

    for filename, meta in files_meta.items():
        file_path = DATA_DIR / filename
        print(f"\n--- Validating: {filename} ---")
        if not file_path.exists():
            print(f"  FAIL: File {filename} does not exist!")
            all_passed = False
            continue

        # 1. SHA-256 verification
        actual_sha = sha256_file(file_path)
        expected_sha = meta.get("sha256")
        if actual_sha == expected_sha:
            print(f"  [PASS] SHA-256 hash match: {actual_sha}")
        else:
            print(f"  [FAIL] SHA-256 mismatch!\n    Expected: {expected_sha}\n    Actual:   {actual_sha}")
            all_passed = False

        # 2. Row count and columns check
        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            expected_cols = meta.get("columns", [])
            if fieldnames == expected_cols:
                print(f"  [PASS] Columns match ({len(fieldnames)} columns): {', '.join(fieldnames[:5])}...")
            else:
                print(f"  [FAIL] Column mismatch!\n    Expected: {expected_cols}\n    Actual:   {fieldnames}")
                all_passed = False

            rows = list(reader)

        expected_records = meta.get("records", 3000)
        if len(rows) == expected_records:
            print(f"  [PASS] Record count match: {len(rows)} rows (expected {expected_records})")
        else:
            print(f"  [FAIL] Record count mismatch: got {len(rows)}, expected {expected_records}")
            all_passed = False

        # 3. Null values and primary key uniqueness
        record_ids = set()
        null_count = 0
        for i, row in enumerate(rows):
            rec_id = row.get("record_id")
            if rec_id in record_ids:
                print(f"  [FAIL] Duplicate record_id: {rec_id} at row {i}")
                all_passed = False
            record_ids.add(rec_id)

            for col, val in row.items():
                if val is None or val == "":
                    null_count += 1

        if null_count == 0:
            print("  [PASS] Zero null or empty values across all cells")
        else:
            print(f"  [FAIL] Found {null_count} null or empty values!")
            all_passed = False

        # 4. File-specific semantic validation
        if filename == "regional_spread_synthetic.csv":
            non_zero_effects = 0
            invalid_prediction_statuses = 0
            invalid_outcome_statuses = 0
            regions_found = set()
            for row in rows:
                eff = float(row.get("synthetic_regional_effect_in_generator", -1.0))
                if eff != 0.0:
                    non_zero_effects += 1
                if row.get("regional_prediction_status") != "DATA_UNAVAILABLE":
                    invalid_prediction_statuses += 1
                if row.get("observed_outcome_status") != "SYNTHETIC_ONLY_NOT_OBSERVED":
                    invalid_outcome_statuses += 1
                regions_found.add(row.get("region_name"))

            if non_zero_effects == 0:
                print("  [PASS] Zero generator regional effect verified across 100% of rows (3000/3000 == 0.0)")
            else:
                print(f"  [FAIL] Found {non_zero_effects} rows with non-zero generator regional effect!")
                all_passed = False

            if invalid_prediction_statuses == 0:
                print("  [PASS] All 3,000 rows have regional_prediction_status == 'DATA_UNAVAILABLE'")
            else:
                print(f"  [FAIL] {invalid_prediction_statuses} rows have invalid regional_prediction_status")
                all_passed = False

            if invalid_outcome_statuses == 0:
                print("  [PASS] All 3,000 rows have observed_outcome_status == 'SYNTHETIC_ONLY_NOT_OBSERVED'")
            else:
                print(f"  [FAIL] {invalid_outcome_statuses} rows have invalid observed_outcome_status")
                all_passed = False

            print(f"  [PASS] Distinct administrative regions covered: {len(regions_found)}")

        elif filename == "content_spread_trajectories_synthetic.csv":
            invalid_hours = 0
            invalid_outcome_statuses = 0
            trajectories: dict[str, list[dict]] = {}
            for row in rows:
                h = int(row.get("elapsed_hours", -1))
                if h not in (1, 6, 24):
                    invalid_hours += 1
                if row.get("observed_outcome_status") != "SYNTHETIC_ONLY_NOT_OBSERVED":
                    invalid_outcome_statuses += 1
                cid = row.get("content_id")
                trajectories.setdefault(cid, []).append({
                    "elapsed_hours": h,
                    "reach": int(row.get("simulated_cumulative_reach", 0)),
                })

            if invalid_hours == 0:
                print("  [PASS] All snapshot elapsed hours are valid ({1, 6, 24})")
            else:
                print(f"  [FAIL] Found {invalid_hours} invalid elapsed hours!")
                all_passed = False

            # Verify chronological ordering & cumulative reach monotonicity
            monotonic_failures = 0
            for cid, snaps in trajectories.items():
                sorted_snaps = sorted(snaps, key=lambda s: s["elapsed_hours"])
                for idx in range(len(sorted_snaps) - 1):
                    if sorted_snaps[idx]["reach"] > sorted_snaps[idx + 1]["reach"]:
                        monotonic_failures += 1

            if monotonic_failures == 0:
                print(f"  [PASS] All {len(trajectories)} content trajectories are non-decreasing over time")
            else:
                print(f"  [FAIL] {monotonic_failures} non-monotonic reach points found!")
                all_passed = False

        elif filename == "virality_indicators_synthetic.csv":
            invalid_bands = 0
            invalid_pred_status = 0
            score_out_of_range = 0
            VALID_BANDS = {"low_indicator", "moderate_indicator", "high_indicator"}
            for row in rows:
                score = float(row.get("synthetic_content_indicator_score", -1))
                if not (0 <= score <= 100):
                    score_out_of_range += 1
                if row.get("synthetic_demo_band") not in VALID_BANDS:
                    invalid_bands += 1
                if row.get("historical_prediction_status") != "DATA_UNAVAILABLE":
                    invalid_pred_status += 1

            if score_out_of_range == 0 and invalid_bands == 0:
                print("  [PASS] All content indicator scores in [0, 100] and valid demo bands")
            else:
                print(f"  [FAIL] Invalid indicator scores: {score_out_of_range}, invalid bands: {invalid_bands}")
                all_passed = False

            if invalid_pred_status == 0:
                print("  [PASS] All 3,000 rows have historical_prediction_status == 'DATA_UNAVAILABLE'")
            else:
                print(f"  [FAIL] {invalid_pred_status} invalid historical_prediction_status rows")
                all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("OVERALL RESULT: ALL SYNTHETIC DATASETS VALIDATED SUCCESSFULLY (100% PASS)")
        print("=" * 70)
        return 0
    else:
        print("OVERALL RESULT: VALIDATION FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
