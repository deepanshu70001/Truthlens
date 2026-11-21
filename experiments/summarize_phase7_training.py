"""
Summarize Phase 7 training results across all 5 configurations.
"""

import json
from pathlib import Path

exps = [
    "s1_siglip_linear_probe",
    "s2_siglip_partial_ft",
    "r1_resnet50_full_ft",
    "r2_resnet50_last_stage",
    "a1_siglip_no_aug_ablation",
]
base_dir = Path("experiments/checkpoints/phase7")

print("=" * 125)
print(f"{'Experiment':<26} | {'Model ID':<32} | {'Epoch':<5} | {'Score':<6} | {'Tiny Bal':<8} | {'Front Bal':<9} | {'Front AUC':<9} | {'Duration':<8} | {'Peak VRAM':<9}")
print("=" * 125)

results = {}
for exp in exps:
    p = base_dir / exp / "experiment_summary.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        results[exp] = data
        mid = data.get("model_id", "")
        ep = data.get("best_epoch", -1)
        score = data.get("best_composite_score", 0.0)
        tiny_bal = data.get("best_tiny_val_metrics", {}).get("balanced_accuracy", 0.0)
        front_bal = data.get("best_frontier_val_metrics", {}).get("balanced_accuracy", 0.0)
        front_auc = data.get("best_frontier_val_metrics", {}).get("roc_auc", 0.0)
        dur = data.get("total_training_duration_sec", 0.0)
        vram = data.get("peak_vram_mb", 0.0)
        print(f"{exp:<26} | {mid:<32} | {ep:<5} | {score:<6.4f} | {tiny_bal:<8.4f} | {front_bal:<9.4f} | {front_auc:<9.4f} | {dur:<8.1f} | {vram:<9.1f}")
print("=" * 125)

# Also write combined summary to experiments/reports/phase7_training_summary.json
out_dir = Path("experiments/reports")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "phase7_training_summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Saved combined summary to experiments/reports/phase7_training_summary.json")
