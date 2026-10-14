
"""Validate TruthLens synthetic datasets without modifying them."""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "synthetic"
MANIFEST_PATH = DATA_DIR / "manifest.json"

EXPECTED_FILES = {
    "virality_scenarios.csv": {
        "id_prefix": "VIR-",
        "required_columns": {
            "record_id", "dataset_type", "data_source", "is_synthetic",
            "is_real_world_observation", "region", "content_language",
            "illustrative_scenario_score", "score_interpretation",
        },
    },
    "regional_spread_scenarios.csv": {
        "id_prefix": "REG-",
        "required_columns": {
            "record_id", "dataset_type", "data_source", "is_synthetic",
            "is_real_world_observation", "region", "content_language",
            "synthetic_language_compatibility",
            "synthetic_connectivity_index",
            "illustrative_scenario_score",
        },
    },
    "social_impact_scenarios.csv": {
        "id_prefix": "IMP-",
        "required_columns": {
            "record_id", "dataset_type", "data_source", "is_synthetic",
            "is_real_world_observation", "region", "content_language",
            "sensitivity_level", "illustrative_scenario_score",
            "actual_harm_observed",
        },
    },
}

SCORE_COLUMNS = {
    "illustrative_scenario_score",
    "synthetic_language_compatibility",
    "synthetic_connectivity_index",
    "hypothetical_share_rate",
}

def validate_csv(filename, specification):
    path = DATA_DIR / filename
    errors = []

    if not path.is_file():
        return [f"{filename}: file is missing"], 0

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            return [f"{filename}: CSV has no header"], 0

        missing_columns = (
            specification["required_columns"] - set(reader.fieldnames)
        )
        if missing_columns:
            errors.append(
                f"{filename}: missing columns {sorted(missing_columns)}"
            )
            return errors, 0

        seen_ids = set()
        count = 0

        for line_number, row in enumerate(reader, start=2):
            count += 1
            record_id = row.get("record_id", "").strip()

            if not record_id:
                errors.append(f"{filename}:{line_number}: empty record_id")
            elif record_id in seen_ids:
                errors.append(
                    f"{filename}:{line_number}: duplicate ID {record_id}"
                )
            else:
                seen_ids.add(record_id)

            if not record_id.startswith(specification["id_prefix"]):
                errors.append(
                    f"{filename}:{line_number}: unexpected ID prefix"
                )

            if row.get("is_synthetic", "").strip().lower() != "true":
                errors.append(
                    f"{filename}:{line_number}: record is not marked synthetic"
                )

            if row.get("data_source") != "synthetic_simulation":
                errors.append(
                    f"{filename}:{line_number}: unexpected data_source"
                )

            if row.get("is_real_world_observation", "").lower() != "false":
                errors.append(
                    f"{filename}:{line_number}: real-world flag must be false"
                )

            if not row.get("region", "").strip():
                errors.append(f"{filename}:{line_number}: empty region")

            if not row.get("content_language", "").strip():
                errors.append(
                    f"{filename}:{line_number}: empty content_language"
                )

            for column in SCORE_COLUMNS.intersection(row.keys()):
                try:
                    value = float(row[column])
                    if not 0.0 <= value <= 1.0:
                        errors.append(
                            f"{filename}:{line_number}: {column} "
                            "must be between 0 and 1"
                        )
                except (ValueError, TypeError):
                    errors.append(
                        f"{filename}:{line_number}: invalid number in {column}"
                    )

            if "sensitivity_level" in row:
                try:
                    level = int(row["sensitivity_level"])
                    if level not in (0, 1, 2, 3):
                        errors.append(
                            f"{filename}:{line_number}: invalid sensitivity_level"
                        )
                except (ValueError, TypeError):
                    errors.append(
                        f"{filename}:{line_number}: invalid sensitivity_level"
                    )

            if "actual_harm_observed" in row:
                if row["actual_harm_observed"].lower() != "false":
                    errors.append(
                        f"{filename}:{line_number}: synthetic data "
                        "must not claim observed harm"
                    )

            if row.get("score_interpretation") != (
                "simulated_only_not_a_prediction"
            ):
                errors.append(
                    f"{filename}:{line_number}: missing simulation disclaimer"
                )

    if count != 500:
        errors.append(
            f"{filename}: expected 500 records, found {count}"
        )

    return errors, count

def main():
    all_errors = []

    if not DATA_DIR.is_dir():
        print(f"ERROR: Dataset directory not found: {DATA_DIR}")
        return 1

    if not MANIFEST_PATH.is_file():
        print(f"ERROR: Manifest not found: {MANIFEST_PATH}")
        return 1

    try:
        with MANIFEST_PATH.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"ERROR: Cannot read manifest: {error}")
        return 1

    if manifest.get("dataset_status") != "SYNTHETIC_DEMO_ONLY":
        all_errors.append("Manifest must declare SYNTHETIC_DEMO_ONLY")

    if manifest.get("generation_seed") != 20261014:
        all_errors.append("Unexpected generation seed in manifest")

    if manifest.get("methodology_version") != "synthetic-v1.0":
        all_errors.append("Unexpected methodology version in manifest")

    manifest_datasets = manifest.get("datasets", {})

    for filename, specification in EXPECTED_FILES.items():
        errors, count = validate_csv(filename, specification)
        all_errors.extend(errors)

        declared = manifest_datasets.get(filename, {})
        if declared.get("record_count") != count:
            all_errors.append(
                f"{filename}: manifest record count does not match CSV"
            )
        if declared.get("is_synthetic") is not True:
            all_errors.append(
                f"{filename}: manifest must mark the dataset synthetic"
            )

        if not errors:
            print(f"PASS: {filename} — {count} records")

    if all_errors:
        print("\nVALIDATION FAILED:")
        for error in all_errors:
            print(f"- {error}")
        return 1

    print("\nPASS: All synthetic datasets passed validation.")
    print("Reminder: validation does not establish predictive accuracy.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
