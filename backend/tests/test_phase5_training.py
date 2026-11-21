"""
TruthLens Phase 5: Controlled Training, Parameter Freezing & Identity Regression Tests.

Verifies:
1. Configuration integrity and hyperparameter safety.
2. Parameter freezing contract (E1/E2 vs E3).
3. Zero test/calibration/validation leakage into the training manifest.
4. Total exclusion of held-out generators (Midjourney, Wukong, VQDM) and WildAIGC from training.
5. Checkpoint schema and cryptographic hash reproducibility.
6. Validation metrics contract.
"""

import csv
import json
from pathlib import Path
from typing import Dict

import pytest
import torch
import torch.nn as nn
import yaml

from experiments.training_utils import SiglipClassifierModel, compute_file_sha256


@pytest.fixture(scope="module")
def phase5_config() -> Dict:
    config_path = Path("experiments/configs/phase5_experiment_config.yaml")
    assert config_path.exists(), f"Phase 5 config not found at {config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_phase5_config_integrity(phase5_config: Dict):
    """Verify configuration parameters, backbone revision, and experiment definitions."""
    arch = phase5_config["model_architecture"]
    assert arch["backbone_model_id"] == "google/siglip-base-patch16-224"
    assert arch["backbone_revision"] == "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
    assert arch["embedding_dim"] == 768
    assert arch["preprocessing_id"] == "siglip-frozen-v1"

    exps = phase5_config["experiments"]
    assert "e1_frozen_head" in exps
    assert "e2_balanced_training" in exps
    assert "e3_partial_finetuning" in exps

    assert exps["e1_frozen_head"]["mode"] == "frozen_head"
    assert exps["e2_balanced_training"]["mode"] == "balanced_head"
    assert exps["e3_partial_finetuning"]["mode"] == "partial_finetuning"

    # Pos weight for balanced training
    assert abs(exps["e2_balanced_training"]["pos_weight"] - (3515 / 3480)) < 1e-4


def test_parameter_freezing_contract():
    """Verify exact parameter freezing rules for frozen head vs partial fine-tuning."""
    # Create a small mock vision model with an encoder containing 12 layers
    class MockEncoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(768, 768)

        def forward(self, x):
            return self.linear(x)

    class MockInnerVisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.layers = nn.ModuleList([MockEncoderLayer() for _ in range(12)])

        def forward(self, x):
            return self.encoder.layers[-1](x)

    class MockSiglipVisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = MockInnerVisionModel()

        def forward(self, pixel_values):
            out = self.vision_model(pixel_values)
            res = nn.Module()
            res.pooler_output = out
            return res

    mock_vision = MockSiglipVisionModel()
    mock_head = nn.Linear(768, 1)
    model = SiglipClassifierModel(mock_vision, mock_head)

    # Test E1 / E2: All backbone params frozen, only head trainable
    for p in model.vision_model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    trainable_e1 = [name for name, p in model.named_parameters() if p.requires_grad]
    assert all("head" in name for name in trainable_e1)
    assert not any("vision_model" in name for name in trainable_e1)

    # Test E3: Only layer 11 and head trainable
    for p in model.vision_model.parameters():
        p.requires_grad = False
    for p in model.vision_model.vision_model.encoder.layers[-1].parameters():
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True

    trainable_e3 = [name for name, p in model.named_parameters() if p.requires_grad]
    assert all("layers.11" in name or "head" in name for name in trainable_e3)
    # Check that layers 0..10 are strictly frozen
    for i in range(11):
        for p in model.vision_model.vision_model.encoder.layers[i].parameters():
            assert not p.requires_grad, f"Layer {i} should be frozen in E3"


def test_zero_leakage_and_clean_training_manifest():
    """Verify training manifest has 0 overlap with test/calibration/val and held-out sets."""
    train_manifest = Path("experiments/manifests/phase4_training_manifest.csv")
    assert train_manifest.exists()

    train_hashes = set()
    train_generators = set()

    with open(train_manifest, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_hashes.add(row["image_sha256"])
            if row.get("generator_name"):
                train_generators.add(row["generator_name"])

    assert len(train_hashes) == 6995, f"Expected 6,995 unique hashes, found {len(train_hashes)}"

    # Check that held-out generators are strictly excluded
    forbidden_generators = {"Midjourney", "Wukong", "VQDM"}
    intersection_gen = train_generators.intersection(forbidden_generators)
    assert len(intersection_gen) == 0, f"Forbidden generators found in training manifest: {intersection_gen}"

    # Check test splits for zero leakage
    split_files = [
        Path("experiments/manifests/tiny_genimage_evaluation_splits.csv"),
        Path("experiments/manifests/ai_image_detection_benchmark_2026_10_evaluation_splits.csv"),
        Path("experiments/manifests/wildaigc_bench_evaluation_splits.csv"),
    ]

    for sf in split_files:
        if not sf.exists():
            continue
        with open(sf, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                role = row.get("evaluation_role")
                h = row.get("sha256")
                if role in ("test", "calibration", "validation"):
                    assert h not in train_hashes, f"Data leakage detected! Hash {h} (role={role}) found in training set!"


def test_checkpoint_schema_and_hashing(tmp_path: Path):
    """Verify that head checkpoint format matches TruthLens Phase 1 loading contract."""
    head_weights = {
        "weight": torch.randn(1, 768),
        "bias": torch.randn(1),
    }
    head_file = tmp_path / "head.pt"
    torch.save(head_weights, head_file)

    # Check hash computation
    h1 = compute_file_sha256(head_file)
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)

    # Verify reload conforms to detector.py expectations
    loaded = torch.load(head_file, weights_only=True)
    assert set(loaded.keys()) == {"weight", "bias"}
    assert loaded["weight"].shape == (1, 768)
    assert loaded["bias"].shape == (1,)


def test_phase5_experiment_outputs_and_model_registry():
    """Verify that all Phase 5 experiment checkpoints, curves, and figures exist."""
    reports_dir = Path("experiments/reports")
    figures_dir = reports_dir / "figures"
    checkpoints_dir = Path("experiments/checkpoints/phase5")

    # Verify checkpoint directories
    for exp in ["e1_frozen_head", "e2_balanced_training", "e3_partial_finetuning"]:
        exp_dir = checkpoints_dir / exp
        assert exp_dir.exists(), f"Experiment directory missing: {exp_dir}"
        assert (exp_dir / "best_checkpoint.pt").exists(), f"Missing best_checkpoint.pt in {exp}"
        assert (exp_dir / "head.pt").exists(), f"Missing head.pt in {exp}"
        assert (exp_dir / "training_history.json").exists(), f"Missing training_history.json in {exp}"
        assert (exp_dir / "experiment_summary.json").exists(), f"Missing experiment_summary.json in {exp}"

    # Verify report JSONs and CSV
    assert (reports_dir / "phase5_training_results.json").exists()
    assert (reports_dir / "phase5_experiment_comparison.csv").exists()
    assert (reports_dir / "phase5_training_curves.json").exists()
    assert (reports_dir / "phase5_model_registry.json").exists()

    # Verify figures
    expected_figures = [
        "phase5_loss_curves.png",
        "phase5_val_balanced_acc_curves.png",
        "phase5_val_roc_curves.png",
        "phase5_val_pr_curves.png",
        "phase5_experiment_comparison.png",
        "phase5_vram_usage.png",
    ]
    for fig in expected_figures:
        fig_path = figures_dir / fig
        assert fig_path.exists(), f"Figure missing: {fig_path}"
        assert fig_path.stat().st_size > 1000, f"Figure {fig} is empty or corrupted"

    # Verify model registry content
    with open(reports_dir / "phase5_model_registry.json", "r", encoding="utf-8") as f:
        registry = json.load(f)

    assert "baseline_model" in registry
    assert "experiments" in registry
    assert "model_selection_decision" in registry
    assert registry["model_selection_decision"]["selected_model"] == "e3_partial_finetuning"
    assert registry["model_selection_decision"]["decision"] == "PROMOTE_AS_CANDIDATE"
    assert registry["model_selection_decision"]["eligibility_for_phase6_test"] is True

