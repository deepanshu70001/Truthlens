
"""
TruthLens synthetic dataset generator.

IMPORTANT:
- Every generated record is simulated.
- Scores are illustrative scenario scores, not trained predictions.
- No synthetic values represent observed Indian social-media behaviour.
- This script does not modify existing TruthLens engines.
"""

import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

SEED = 20261014
RECORDS_PER_DATASET = 500
METHODOLOGY_VERSION = "synthetic-v1.0"

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "synthetic"

REGIONS = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
    "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh",
    "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal", "Delhi", "Chandigarh", "Jammu and Kashmir",
    "Ladakh", "Puducherry", "Andaman and Nicobar Islands",
    "Lakshadweep",
    "Dadra and Nagar Haveli and Daman and Diu",
]

LANGUAGES = [
    "hi", "en", "bn", "ta", "te", "mr", "gu",
    "kn", "ml", "pa", "or", "ur",
]

CATEGORIES = [
    "public_safety",
    "health",
    "politics",
    "celebrity",
    "natural_disaster",
    "general_news",
]

def write_csv(path, rows):
    """Write records to CSV using the fields in the first record."""
    if not rows:
        raise ValueError(f"No rows generated for {path.name}")

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def common_fields(record_id, dataset_type):
    return {
        "record_id": record_id,
        "dataset_type": dataset_type,
        "data_source": "synthetic_simulation",
        "is_synthetic": True,
        "generation_seed": SEED,
        "methodology_version": METHODOLOGY_VERSION,
        "is_real_world_observation": False,
    }

def generate_virality(rng):
    rows = []

    for i in range(1, RECORDS_PER_DATASET + 1):
        urgency = rng.randint(0, 1)
        emotional = rng.randint(0, 1)
        has_claims = rng.randint(0, 1)
        has_faces = rng.randint(0, 1)
        multiple_objects = rng.randint(0, 1)

        # Hypothetical scenario inputs, not observed engagement data.
        audience_exposure = rng.randint(100, 100000)
        hypothetical_share_rate = round(rng.uniform(0.001, 0.20), 4)

        # A transparent illustrative formula, NOT a trained model.
        exposure_component = min(
            audience_exposure / 100000.0, 1.0
        )

        score = (
            0.20 * urgency
            + 0.20 * emotional
            + 0.15 * has_claims
            + 0.10 * has_faces
            + 0.10 * multiple_objects
            + 0.15 * exposure_component
            + 0.10 * (hypothetical_share_rate / 0.20)
        )
        score = round(max(0.0, min(1.0, score)), 4)

        rows.append({
            **common_fields(f"VIR-{i:06d}", "virality_scenario"),
            "region": rng.choice(REGIONS),
            "content_language": rng.choice(LANGUAGES),
            "content_category": rng.choice(CATEGORIES),
            "urgency_language": urgency,
            "emotional_language": emotional,
            "has_claims": has_claims,
            "has_faces": has_faces,
            "multiple_objects": multiple_objects,
            "hypothetical_audience_exposure": audience_exposure,
            "hypothetical_share_rate": hypothetical_share_rate,
            "illustrative_scenario_score": score,
            "score_interpretation": "simulated_only_not_a_prediction",
        })

    return rows

def generate_regional(rng):
    rows = []

    for i in range(1, RECORDS_PER_DATASET + 1):
        compatibility = round(rng.uniform(0.0, 1.0), 4)
        connectivity = round(rng.uniform(0.0, 1.0), 4)

        # Both inputs are randomly simulated, not state-level facts.
        scenario_score = round(
            0.60 * compatibility + 0.40 * connectivity, 4
        )

        rows.append({
            **common_fields(f"REG-{i:06d}", "regional_spread_scenario"),
            "region": rng.choice(REGIONS),
            "content_language": rng.choice(LANGUAGES),
            "synthetic_language_compatibility": compatibility,
            "synthetic_connectivity_index": connectivity,
            "illustrative_scenario_score": scenario_score,
            "score_interpretation": "simulated_only_not_a_prediction",
        })

    return rows

def generate_social_impact(rng):
    rows = []

    for i in range(1, RECORDS_PER_DATASET + 1):
        sensitivity = rng.randint(0, 3)
        hypothetical_exposure = rng.randint(100, 100000)
        urgency = rng.randint(0, 1)
        emotional = rng.randint(0, 1)

        exposure_component = min(
            hypothetical_exposure / 100000.0, 1.0
        )

        # Scenario indicator only; no actual harm is measured.
        score = round(
            0.35 * (sensitivity / 3.0)
            + 0.25 * exposure_component
            + 0.20 * urgency
            + 0.20 * emotional,
            4,
        )

        rows.append({
            **common_fields(f"IMP-{i:06d}", "social_impact_scenario"),
            "region": rng.choice(REGIONS),
            "content_language": rng.choice(LANGUAGES),
            "content_category": rng.choice(CATEGORIES),
            "sensitivity_level": sensitivity,
            "hypothetical_audience_exposure": hypothetical_exposure,
            "urgency_language": urgency,
            "emotional_language": emotional,
            "illustrative_scenario_score": score,
            "actual_harm_observed": False,
            "score_interpretation": "simulated_only_not_a_prediction",
        })

    return rows

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    datasets = {
        "virality_scenarios.csv": generate_virality(rng),
        "regional_spread_scenarios.csv": generate_regional(rng),
        "social_impact_scenarios.csv": generate_social_impact(rng),
    }

    manifest = {
        "project": "TruthLens",
        "dataset_status": "SYNTHETIC_DEMO_ONLY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_seed": SEED,
        "methodology_version": METHODOLOGY_VERSION,
        "records_per_dataset": RECORDS_PER_DATASET,
        "datasets": {},
        "limitations": [
            "These records are simulated, not real-world observations.",
            "Scores are illustrative formulas, not trained predictions.",
            "Synthetic scores must not be reported as real virality or harm.",
            "Random regional inputs do not represent actual state-level data.",
            "Do not use this data alone to claim model accuracy.",
        ],
    }

    for filename, rows in datasets.items():
        destination = OUTPUT_DIR / filename
        write_csv(destination, rows)
        manifest["datasets"][filename] = {
            "record_count": len(rows),
            "is_synthetic": True,
            "columns": list(rows[0].keys()),
        }
        print(f"Created: {destination} ({len(rows)} records)")

    manifest_path = OUTPUT_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    print(f"Created: {manifest_path}")
    print("Finished. All generated data is marked synthetic.")

if __name__ == "__main__":
    main()
