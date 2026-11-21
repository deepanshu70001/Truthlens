"""Unit and regression tests for Phase 2 Step 3 group-aware dataset splitting."""
from __future__ import annotations

import csv
import json
import pytest
from pathlib import Path

from experiments.create_evaluation_splits import (
    load_yaml_config,
    partition_tiny_genimage,
    partition_frontier,
    partition_wildaigc,
    run_leakage_audit,
    LeakageError,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def base_config():
    cfg_path = REPO_ROOT / "experiments" / "configs" / "phase2_split_protocol.yaml"
    return load_yaml_config(cfg_path)


def test_config_loading_and_structure(base_config):
    """Verify split protocol YAML loads cleanly and contains required structure."""
    assert base_config["protocol_version"] == "2.0.0"
    assert "tiny_genimage" in base_config["datasets"]
    assert "frontier_benchmark_2026_10" in base_config["datasets"]
    assert "wildaigc_bench" in base_config["datasets"]
    assert base_config["split_seed"] == 42


def test_deterministic_split_generation_across_runs(base_config):
    """Verify that repeated partitioning with the same seed produces identical results."""
    rows1, _ = partition_frontier(base_config, seed=42)
    rows2, _ = partition_frontier(base_config, seed=42)

    assert len(rows1) == len(rows2)
    for r1, r2 in zip(rows1, rows2):
        assert r1["image_path"] == r2["image_path"]
        assert r1["evaluation_role"] == r2["evaluation_role"]
        assert r1["group_id"] == r2["group_id"]


def test_seed_variation_changes_allocations(base_config):
    """Verify that different random seeds produce different randomized partitionings."""
    rows_seed42, _ = partition_frontier(base_config, seed=42)
    rows_seed99, _ = partition_frontier(base_config, seed=99)

    # Some groups should have different roles between seed 42 and seed 99
    differences = sum(
        1 for r1, r2 in zip(rows_seed42, rows_seed99)
        if r1["evaluation_role"] != r2["evaluation_role"]
    )
    assert differences > 0


def test_frontier_social_variants_strictly_isolated_in_same_partition(base_config):
    """Verify that all social variants (original, whatsapp, instagram, reshared) share the same role."""
    rows, _ = partition_frontier(base_config, seed=42)
    group_to_roles = {}

    for r in rows:
        gid = r["group_id"]
        role = r["evaluation_role"]
        if gid not in group_to_roles:
            group_to_roles[gid] = set()
        group_to_roles[gid].add(role)

    # Every single group must map to exactly 1 role
    for gid, roles in group_to_roles.items():
        assert len(roles) == 1, f"Group {gid} leaked across roles: {roles}"


def test_wildaigc_relations_connected_components_isolated(base_config):
    """Verify that related AI/real image pairs in WildAIGC never cross partitions."""
    rows, _ = partition_wildaigc(base_config, seed=42)
    group_to_roles = {}

    for r in rows:
        gid = r["group_id"]
        role = r["evaluation_role"]
        if gid not in group_to_roles:
            group_to_roles[gid] = set()
        group_to_roles[gid].add(role)

    for gid, roles in group_to_roles.items():
        assert len(roles) == 1, f"WildAIGC connected component group {gid} leaked across roles: {roles}"


def test_tiny_genimage_official_val_retained_as_test(base_config):
    """Verify that official validation images are retained as test and never in calibration/validation."""
    rows, _ = partition_tiny_genimage(base_config, seed=42)

    val_roles = {r["evaluation_role"] for r in rows if r["split_original"] == "val"}
    train_roles = {r["evaluation_role"] for r in rows if r["split_original"] == "train"}

    assert val_roles == {"test"}, f"Official val split leaked into non-test roles: {val_roles}"
    assert "test" not in train_roles, f"Official train split leaked into test role: {train_roles}"
    assert train_roles == {"calibration", "validation", "excluded"}


def test_tiny_genimage_exact_duplicate_held_in_same_role(base_config):
    """Verify that the duplicate image in Tiny-GenImage (3eba56bc...) is in the same role."""
    rows, _ = partition_tiny_genimage(base_config, seed=42)
    target_sha = "3eba56bc8ccd68cfe9b9f780829d081446e944156a114af3b08c874609f3631b"

    roles = {r["evaluation_role"] for r in rows if r["sha256"] == target_sha}
    assert len(roles) == 1, f"Duplicate SHA256 {target_sha} leaked across roles: {roles}"


def test_leakage_audit_detects_deliberate_leakage(base_config):
    """Verify that run_leakage_audit detects and raises on cross-role leakage."""
    synthetic_rows = {
        "synthetic_test": [
            {"sha256": "hash_a", "group_id": "grp_1", "evaluation_role": "calibration", "image_path": "img1.png"},
            {"sha256": "hash_a", "group_id": "grp_1", "evaluation_role": "test", "image_path": "img2.png"},
        ]
    }
    audit = run_leakage_audit(synthetic_rows, base_config)
    assert audit["any_leakage_detected"] is True
    assert audit["datasets"]["synthetic_test"]["cross_role_identical_sha256_count"] == 1
    assert audit["datasets"]["synthetic_test"]["cross_role_group_id_leak_count"] == 1


def test_original_ingestion_manifests_remain_unmodified():
    """Verify that original Step 2 ingestion manifests exist and were not overwritten."""
    manifest_files = [
        "experiments/manifests/tiny_genimage.csv",
        "experiments/manifests/frontier_benchmark_2026_10.csv",
        "experiments/manifests/wildaigc_bench.csv",
    ]
    for mf in manifest_files:
        p = REPO_ROOT / mf
        assert p.is_file(), f"Original manifest missing: {mf}"
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Original manifests must NOT have split_original (they have split)
            assert "split" in reader.fieldnames
            assert "split_original" not in reader.fieldnames


def test_derived_split_manifests_have_required_columns():
    """Verify that derived evaluation split manifests contain all required audit fields."""
    derived_files = [
        "experiments/manifests/tiny_genimage_evaluation_splits.csv",
        "experiments/manifests/frontier_benchmark_2026_10_evaluation_splits.csv",
        "experiments/manifests/wildaigc_bench_evaluation_splits.csv",
    ]
    required_fields = [
        "image_path", "label", "split_original", "evaluation_role", "protocol_id",
        "group_id", "group_id_basis", "group_id_status", "split_seed", "split_reason",
        "generator", "source_url", "license_ref", "sha256"
    ]
    for df in derived_files:
        p = REPO_ROOT / df
        assert p.is_file(), f"Derived manifest missing: {df}"
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for rf in required_fields:
                assert rf in reader.fieldnames, f"Missing required column {rf} in {df}"
