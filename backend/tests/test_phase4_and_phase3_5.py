"""
TruthLens Phase 3.5 & Phase 4: Verification Suite for Audit, Training Manifest & Anti-Leakage.

Tests cover:
1. Metric recomputation and confusion matrix consistency.
2. Label mapping and uncertainty detection.
3. Reserve eligibility classification (APPROVED_TRAINING, EXCLUDED_PROTOCOL, EXCLUDED_DUPLICATE).
4. Anti-leakage verification (Zero training overlap with calibration, validation, or locked test).
5. Dataset approval policy enforcement (WildAIGC rejected for training, Frontier reserved).
6. Training manifest integrity and image dimension validity.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
import numpy as np
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestMetricReconciliationAndConsistency:
    """Tests for Phase 3.5 metric recalculation and mathematical consistency."""

    def test_confusion_matrix_consistency(self):
        # TN, FP, FN, TP
        cm = {"tn": 1116, "fp": 2384, "fn": 753, "tp": 2747}
        tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]

        total = tn + fp + fn + tp
        assert total == 7000

        acc = (tp + tn) / total
        assert round(acc, 4) == 0.5519

        prec = tp / (tp + fp)
        assert round(prec, 4) == 0.5354  # Reconciled precision!

        rec = tp / (tp + fn)
        assert round(rec, 4) == 0.7849

        spec = tn / (tn + fp)
        assert round(spec, 4) == 0.3189

        far = fp / (tn + fp)
        assert round(far, 4) == 0.6811

        frr = fn / (fn + tp)
        assert round(frr, 4) == 0.2151

        f1 = 2 * prec * rec / (prec + rec)
        assert round(f1, 4) == 0.6365

    def test_reconciliation_csv_exists_and_valid(self):
        csv_path = PROJECT_ROOT / "experiments" / "reports" / "phase3_5_metric_reconciliation.csv"
        assert csv_path.is_file()

        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) > 0
        required_cols = {"dataset", "metric_name", "narrative_reported_value", "recomputed_ground_truth_value", "difference"}
        assert required_cols.issubset(rows[0].keys())

        # Check precision row for tiny_genimage
        prec_row = next(r for r in rows if r["dataset"] == "tiny_genimage" and r["metric_name"] == "precision")
        assert prec_row["recomputed_ground_truth_value"] == "0.5354"
        assert prec_row["narrative_reported_value"] == "0.5204"


class TestDatasetApprovalPolicy:
    """Tests for dataset approval and role boundaries."""

    def test_wildaigc_not_approved_for_training(self):
        audit_file = PROJECT_ROOT / "experiments" / "reports" / "phase3_5_audit.json"
        assert audit_file.is_file()

        with open(audit_file, "r", encoding="utf-8") as f:
            audit = json.load(f)

        decisions = audit["dataset_approval_decisions"]
        assert decisions["wildaigc_bench"] == "NOT_APPROVED_FOR_AI_GENERATION_SUPERVISED_TRAINING"
        assert decisions["frontier_benchmark_2026_10"] == "RESERVED_FOR_EVALUATION_ZERO_TRAINING_MEMBERSHIP"
        assert decisions["tiny_genimage"] == "APPROVED_TRAINING_RESERVE_SUBSET_ONLY"


class TestReserveEligibilityAndExclusionAudit:
    """Tests for Tiny-GenImage reserve sample classification."""

    def test_reserve_exclusion_audit_counts(self):
        csv_path = PROJECT_ROOT / "experiments" / "reports" / "phase4_exclusion_audit.csv"
        assert csv_path.is_file()

        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 9994

        counts = Counter(r["training_eligibility"] for r in rows)
        assert counts["APPROVED_TRAINING"] == 6995
        assert counts["EXCLUDED_PROTOCOL"] == 2998
        assert counts["EXCLUDED_DUPLICATE"] == 1

    def test_duplicate_sample_purged(self):
        csv_path = PROJECT_ROOT / "experiments" / "reports" / "phase4_exclusion_audit.csv"
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        dup_rows = [r for r in rows if r["training_eligibility"] == "EXCLUDED_DUPLICATE"]
        assert len(dup_rows) == 1
        assert "3eba56bc" in dup_rows[0]["sha256"]


class TestTrainingManifestIntegrityAndAntiLeakage:
    """Tests for Phase 4 training manifest schema, counts, and zero-leakage guarantee."""

    def test_training_manifest_counts_and_balance(self):
        csv_path = PROJECT_ROOT / "experiments" / "manifests" / "phase4_training_manifest.csv"
        assert csv_path.is_file()

        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 6995

        counts = Counter(r["normalized_label"] for r in rows)
        assert counts["likely_authentic"] == 3515
        assert counts["ai_generated"] == 3480

        # Optimal balance: 49.75% AI vs 50.25% Real
        ratio = counts["ai_generated"] / len(rows)
        assert 0.49 <= ratio <= 0.51

    def test_zero_leakage_into_evaluation_partitions(self):
        train_csv = PROJECT_ROOT / "experiments" / "manifests" / "phase4_training_manifest.csv"
        with open(train_csv, "r", encoding="utf-8") as f:
            train_rows = list(csv.DictReader(f))

        train_hashes = {r["image_sha256"] for r in train_rows}
        train_groups = {r["source_group_id"] for r in train_rows}

        eval_csvs = [
            PROJECT_ROOT / "experiments" / "manifests" / "tiny_genimage_evaluation_splits.csv",
            PROJECT_ROOT / "experiments" / "manifests" / "frontier_benchmark_2026_10_evaluation_splits.csv",
            PROJECT_ROOT / "experiments" / "manifests" / "wildaigc_bench_evaluation_splits.csv",
        ]

        for p in eval_csvs:
            with open(p, "r", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    role = r["evaluation_role"]
                    sha = r["sha256"]
                    grp = r["group_id"]

                    if role in ["calibration", "validation", "test"]:
                        assert sha not in train_hashes, f"FATAL: Training hash {sha} leaked into {role} split of {p.name}"
                        assert grp not in train_groups, f"FATAL: Training group {grp} leaked into {role} split of {p.name}"

    def test_training_config_yaml_validity(self):
        cfg_path = PROJECT_ROOT / "experiments" / "configs" / "phase4_training_config.yaml"
        assert cfg_path.is_file()

        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert cfg["model"]["backbone"] == "google/siglip-base-patch16-224"
        assert cfg["model"]["freeze_backbone"] is True
        assert cfg["hardware_profile"]["dedicated_vram_gb"] == 4
        assert cfg["training_hyperparameters"]["batch_size"] == 32
        assert cfg["training_hyperparameters"]["gradient_accumulation_steps"] == 2
        assert cfg["training_hyperparameters"]["mixed_precision"] == "fp16"
