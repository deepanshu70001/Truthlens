"""Demo-only API for browsing synthetic TruthLens scenario datasets."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_SYNTHETIC_DIR = _PROJECT_ROOT / "data" / "synthetic"

_DATASETS = {
    "virality": "virality_scenarios.csv",
    "regional_spread": "regional_spread_scenarios.csv",
    "social_impact": "social_impact_scenarios.csv",
}

_LIMITATIONS = [
    "These records are synthetic simulations, not real-world observations.",
    "Illustrative scores are not trained predictions.",
    "Do not use these records to claim model accuracy or actual harm.",
]


def _require_demo_mode() -> None:
    """Prevent access to synthetic examples unless demo mode is enabled."""
    enabled = os.environ.get("DEMO_MODE", "").strip().lower()
    if enabled not in {"true", "1", "yes"}:
        raise HTTPException(
            status_code=404,
            detail="Demo datasets are unavailable when DEMO_MODE is disabled.",
        )


def _read_manifest() -> dict[str, Any]:
    manifest_path = _SYNTHETIC_DIR / "manifest.json"
    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            import json
            manifest = json.load(file)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Synthetic dataset manifest is unavailable.",
        ) from exc
    return manifest


@router.get(
    "/datasets",
    summary="List synthetic demo datasets",
    description="Lists synthetic scenario datasets. Available only in demo mode.",
)
async def list_demo_datasets() -> dict[str, Any]:
    _require_demo_mode()
    manifest = _read_manifest()

    return {
        "data_status": "SYNTHETIC_DEMO_ONLY",
        "is_synthetic": True,
        "is_real_world_observation": False,
        "datasets": [
            {
                "name": name,
                "file": filename,
                "available": (_SYNTHETIC_DIR / filename).is_file(),
            }
            for name, filename in _DATASETS.items()
        ],
        "manifest": manifest,
        "limitations": _LIMITATIONS,
    }


@router.get(
    "/datasets/{dataset_name}",
    summary="Read synthetic demo scenarios",
    description=(
        "Returns a page of synthetic scenario records. "
        "Available only in demo mode; these are not real predictions."
    ),
)
async def get_demo_dataset(
    dataset_name: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    _require_demo_mode()

    filename = _DATASETS.get(dataset_name)
    if filename is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown dataset. Use /api/v1/datasets to list available datasets.",
        )

    path = _SYNTHETIC_DIR / filename
    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Dataset file is unavailable: {filename}",
        )

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            columns = reader.fieldnames or []
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not read the requested dataset.",
        ) from exc

    return {
        "dataset": dataset_name,
        "data_status": "SYNTHETIC_DEMO_ONLY",
        "is_synthetic": True,
        "is_real_world_observation": False,
        "score_interpretation": "simulated_only_not_a_prediction",
        "total_records": len(rows),
        "limit": limit,
        "offset": offset,
        "columns": columns,
        "records": rows[offset : offset + limit],
        "limitations": _LIMITATIONS,
    }
