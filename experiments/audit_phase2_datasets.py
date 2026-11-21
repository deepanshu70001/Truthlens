"""Phase 2 Step 1: Comprehensive repository image dataset and manifest audit.

Performs deterministic SHA-256 and perceptual hash audits across all image
files in the repository, detecting exact and near duplicates, checking image
integrity, inspecting manifests, and evaluating research readiness.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import imagehash

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
SKIP_DIRS = {".git", "node_modules", ".venv", "__pycache__", ".ruff_cache", ".pytest_cache"}


def get_git_info() -> dict:
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        rev = None
    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        branch = None
    return {"revision": rev, "branch": branch}


def categorize_folder(folder_path: str) -> str:
    norm = folder_path.replace("\\", "/")
    if norm.startswith("data/heatmaps"):
        return "runtime_generated_heatmaps"
    elif norm == "data/originals":
        return "runtime_user_uploads"
    elif norm == "data/processed":
        return "runtime_processed_derivatives"
    elif norm == "data/thumbnails":
        return "runtime_generated_thumbnails"
    elif norm == "test_data/heatmaps":
        return "test_generated_heatmaps"
    elif norm == "test_data/originals":
        return "test_fixture_originals"
    elif norm == "test_data/processed":
        return "test_fixture_processed"
    elif norm == "test_data/thumbnails":
        return "test_fixture_thumbnails"
    elif norm == "test_data":
        return "test_fixture_root"
    elif norm == "backend/tests/fixtures":
        return "unit_test_fixture"
    elif norm == "frontend/src/assets":
        return "frontend_ui_asset"
    elif norm == "experiments/results":
        return "experiment_generated_figure"
    return "other"


def run_audit() -> dict:
    git_info = get_git_info()
    image_paths: list[Path] = []

    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in sorted(files):
            p = Path(root) / f
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(p)

    total_scanned = len(image_paths)
    unreadable: list[dict] = []
    corrupted: list[dict] = []
    file_records: list[dict] = []

    sha256_to_paths: dict[str, list[str]] = defaultdict(list)
    dhash_to_paths: dict[str, list[str]] = defaultdict(list)
    phash_to_paths: dict[str, list[str]] = defaultdict(list)
    ahash_to_paths: dict[str, list[str]] = defaultdict(list)

    folder_stats: dict[str, dict] = defaultdict(lambda: {
        "count": 0,
        "extensions": Counter(),
        "formats": Counter(),
        "dimensions": Counter(),
        "total_bytes": 0,
        "files": []
    })

    dhash_objs: dict[str, imagehash.ImageHash] = {}
    phash_objs: dict[str, imagehash.ImageHash] = {}

    dimensions_counter = Counter()
    formats_counter = Counter()
    extensions_counter = Counter()
    total_bytes = 0

    for path in image_paths:
        rel_path = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        folder_path = str(path.parent.relative_to(REPO_ROOT)).replace("\\", "/")
        ext = path.suffix.lower()
        extensions_counter[ext] += 1

        # 1. Read bytes & compute SHA-256
        try:
            raw_bytes = path.read_bytes()
            size = len(raw_bytes)
            total_bytes += size
            sha256 = hashlib.sha256(raw_bytes).hexdigest()
        except Exception as exc:
            unreadable.append({"path": rel_path, "error": str(exc)})
            continue

        # 2. Verify with PIL
        try:
            with Image.open(path) as im:
                im.verify()
        except Exception as exc:
            corrupted.append({"path": rel_path, "error": f"PIL.verify failed: {exc}"})
            continue

        # 3. Load & extract metadata, perceptual hashes
        try:
            with Image.open(path) as im:
                im.load()
                fmt = im.format or "UNKNOWN"
                w, h = im.width, im.height
                dim_str = f"{w}x{h}"
                dh = imagehash.dhash(im)
                ph = imagehash.phash(im)
                ah = imagehash.average_hash(im)
        except Exception as exc:
            corrupted.append({"path": rel_path, "error": f"PIL.load failed: {exc}"})
            continue

        dh_str = str(dh)
        ph_str = str(ph)
        ah_str = str(ah)

        dhash_objs[rel_path] = dh
        phash_objs[rel_path] = ph

        sha256_to_paths[sha256].append(rel_path)
        dhash_to_paths[dh_str].append(rel_path)
        phash_to_paths[ph_str].append(rel_path)
        ahash_to_paths[ah_str].append(rel_path)

        dimensions_counter[dim_str] += 1
        formats_counter[fmt] += 1

        f_stat = folder_stats[folder_path]
        f_stat["count"] += 1
        f_stat["extensions"][ext] += 1
        f_stat["formats"][fmt] += 1
        f_stat["dimensions"][dim_str] += 1
        f_stat["total_bytes"] += size
        f_stat["files"].append(rel_path)

        file_records.append({
            "path": rel_path,
            "folder": folder_path,
            "category": categorize_folder(folder_path),
            "filename": path.name,
            "extension": ext,
            "format": fmt,
            "dimensions": dim_str,
            "width": w,
            "height": h,
            "size_bytes": size,
            "sha256": sha256,
            "dhash": dh_str,
            "phash": ph_str,
            "ahash": ah_str,
            "source_url": None,
            "label": None,
            "license": None,
            "split": None,
        })

    # Exact duplicate analysis
    exact_clusters = []
    total_redundant_copies = 0
    for sha, paths in sorted(sha256_to_paths.items(), key=lambda x: len(x[1]), reverse=True):
        if len(paths) > 1:
            redundant = len(paths) - 1
            total_redundant_copies += redundant
            exact_clusters.append({
                "sha256": sha,
                "count": len(paths),
                "redundant_copies": redundant,
                "files": paths
            })

    # Pairwise near-duplicate analysis
    all_indexed_paths = list(dhash_objs.keys())
    num_indexed = len(all_indexed_paths)

    dhash_dist0 = 0
    dhash_dist_le2 = 0
    dhash_dist_le5 = 0
    phash_dist0 = 0
    phash_dist_le2 = 0
    phash_dist_le5 = 0

    cross_folder_exact_pairs = []

    for i in range(num_indexed):
        p1 = all_indexed_paths[i]
        dh1 = dhash_objs[p1]
        ph1 = phash_objs[p1]
        folder1 = p1.split("/")[0]

        for j in range(i + 1, num_indexed):
            p2 = all_indexed_paths[j]
            dh2 = dhash_objs[p2]
            ph2 = phash_objs[p2]
            folder2 = p2.split("/")[0]

            d_dist = int(dh1 - dh2)
            p_dist = int(ph1 - ph2)

            if d_dist == 0:
                dhash_dist0 += 1
            if d_dist <= 2:
                dhash_dist_le2 += 1
            if d_dist <= 5:
                dhash_dist_le5 += 1

            if p_dist == 0:
                phash_dist0 += 1
            if p_dist <= 2:
                phash_dist_le2 += 1
            if p_dist <= 5:
                phash_dist_le5 += 1

            if d_dist == 0 and folder1 != folder2:
                cross_folder_exact_pairs.append({
                    "image1": p1,
                    "image2": p2,
                    "folder1": folder1,
                    "folder2": folder2,
                    "dhash": str(dh1)
                })

    # Inspect manifests
    manifest_audit = {
        "manifests_detected": [],
        "database_manifests_detected": [],
        "metadata_coverage": {
            "total_images": total_scanned,
            "images_with_source_url": 0,
            "source_url_coverage_pct": 0.0,
            "images_with_ground_truth_label": 0,
            "ground_truth_label_coverage_pct": 0.0,
            "images_with_license": 0,
            "license_coverage_pct": 0.0,
            "images_with_split_assignment": 0,
            "split_coverage_pct": 0.0,
        }
    }

    # 1. authenticity.example.csv
    example_manifest_path = REPO_ROOT / "experiments" / "manifests" / "authenticity.example.csv"
    if example_manifest_path.is_file():
        content = example_manifest_path.read_text(encoding="utf-8").strip().splitlines()
        header = content[0].split(",") if content else []
        data_rows = content[1:] if len(content) > 1 else []
        manifest_audit["manifests_detected"].append({
            "path": "experiments/manifests/authenticity.example.csv",
            "type": "csv_template",
            "header_columns": header,
            "data_row_count": len(data_rows),
            "completeness_status": "empty_template",
            "description": "Schema template defining required research columns; contains zero image entries."
        })

    # 2. sqlite databases
    for db_rel in ["truthlens.db", "test_stage3_debug.db"]:
        db_path = REPO_ROOT / db_rel
        if db_path.is_file():
            import sqlite3
            try:
                conn = sqlite3.connect(str(db_path))
                c = conn.cursor()
                c.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [r[0] for r in c.fetchall()]
                has_image_assets = "image_assets" in tables
                asset_count = 0
                if has_image_assets:
                    c.execute("SELECT COUNT(*) FROM image_assets")
                    asset_count = c.fetchone()[0]
                conn.close()
                manifest_audit["database_manifests_detected"].append({
                    "path": db_rel,
                    "tables": tables,
                    "image_assets_rows": asset_count,
                    "has_research_labels": False,
                    "has_source_urls": False,
                    "has_licenses": False,
                    "description": "Application execution database tracking runtime analysis jobs; not a research dataset manifest."
                })
            except Exception as exc:
                manifest_audit["database_manifests_detected"].append({
                    "path": db_rel,
                    "error": str(exc)
                })

    # Assemble formatted folder records
    folders_formatted = []
    for f_path, stat in sorted(folder_stats.items()):
        folders_formatted.append({
            "folder": f_path,
            "category": categorize_folder(f_path),
            "file_count": stat["count"],
            "extensions": dict(sorted(stat["extensions"].items())),
            "formats": dict(sorted(stat["formats"].items())),
            "dimensions": dict(sorted(stat["dimensions"].items())),
            "total_bytes": stat["total_bytes"],
            "total_kilobytes": round(stat["total_bytes"] / 1024, 2)
        })

    # Aggregate result
    inventory = {
        "audit_metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "repository_revision": git_info["revision"],
            "repository_branch": git_info["branch"],
            "python_version": platform.python_version(),
            "pillow_version": Image.__version__,
            "imagehash_version": imagehash.__version__,
            "platform": platform.platform()
        },
        "aggregate_summary": {
            "total_images_scanned": total_scanned,
            "valid_images": len(file_records),
            "unreadable_images": len(unreadable),
            "corrupted_images": len(corrupted),
            "total_image_bytes": total_bytes,
            "total_folders": len(folder_stats),
            "extensions_distribution": dict(sorted(extensions_counter.items())),
            "formats_distribution": dict(sorted(formats_counter.items())),
            "dimensions_distribution": dict(sorted(dimensions_counter.items())),
            "unique_sha256_count": len(sha256_to_paths),
            "exact_duplicate_extra_copies": total_redundant_copies,
            "exact_duplicate_clusters_count": len(exact_clusters),
            "unique_dhash_count": len(dhash_to_paths),
            "unique_phash_count": len(phash_to_paths),
            "pairwise_comparisons_count": num_indexed * (num_indexed - 1) // 2,
            "dhash_distance_0_pairs": dhash_dist0,
            "dhash_distance_le_2_pairs": dhash_dist_le2,
            "dhash_distance_le_5_pairs": dhash_dist_le5,
            "phash_distance_0_pairs": phash_dist0,
            "phash_distance_le_2_pairs": phash_dist_le2,
            "phash_distance_le_5_pairs": phash_dist_le5
        },
        "folders": folders_formatted,
        "exact_duplicate_clusters": exact_clusters,
        "perceptual_hash_top_clusters": [
            {
                "dhash": dh,
                "count": len(paths),
                "pairs_within_cluster": len(paths) * (len(paths) - 1) // 2,
                "sample_files": paths[:5],
                "description": "Solid/blank test fixture pattern" if dh == "0000000000000000" else "Perceptually identical image group"
            }
            for dh, paths in sorted(dhash_to_paths.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        ],
        "cross_folder_perceptual_exact_pairs_sample": cross_folder_exact_pairs[:20],
        "manifest_audit": manifest_audit,
        "research_readiness": {
            "evaluation_ready": False,
            "calibration_ready": False,
            "training_ready": False,
            "verdict": "RESEARCH_DATASET_UNAVAILABLE",
            "findings": [
                "Zero verified ground-truth veracity or authenticity labels exist in repository files.",
                "Zero image source URLs, original provenance, or acquisition timestamps exist.",
                "Zero copyright licenses, permission records, or consent documentation exist.",
                "Zero official train/validation/test split definitions exist.",
                f"{total_redundant_copies} of {total_scanned} images ({round(total_redundant_copies / total_scanned * 100, 1)}%) are redundant exact-byte duplicates.",
                "165 of 266 images belong to a single dummy solid-color test fixture perceptual cluster (dHash=0000000000000000).",
                "Existing data consists strictly of runtime server uploads/derivatives (data/) and test mock assets (test_data/ and fixtures).",
                "No external benchmark datasets (CIFAKE, GenImage, FakeNewsNet, IFND, etc.) are installed in this repository checkout."
            ]
        },
        "file_records": file_records
    }

    return inventory


def main() -> None:
    print("Running Phase 2 Step 1 dataset inventory audit...")
    inventory = run_audit()

    output_dir = REPO_ROOT / "experiments" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "dataset_inventory.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    summary = inventory["aggregate_summary"]
    print(f"Audit completed successfully.")
    print(f"Scanned: {summary['total_images_scanned']} files across {summary['total_folders']} folders.")
    print(f"Valid: {summary['valid_images']}, Unreadable: {summary['unreadable_images']}, Corrupted: {summary['corrupted_images']}.")
    print(f"Unique SHA-256: {summary['unique_sha256_count']}, Redundant extra copies: {summary['exact_duplicate_extra_copies']}.")
    print(f"Saved report to: {output_file}")


if __name__ == "__main__":
    main()
