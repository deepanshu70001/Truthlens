"""Render a small audit figure from the recorded aggregate dataset scan."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("experiments/results/test_originals_inventory.json"))
    ap.add_argument("--output", type=Path, default=Path("experiments/results/test_data_readiness.png"))
    args = ap.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    labels = ["Readable image files", "Exact duplicate\nextra copies", "dHash exact\ncandidates", "Labeled examples", "License-reviewed"]
    values = [data["valid_images"], data["exact_duplicate_extra_copies"], data["dhash_exact_candidate_extra_copies"], 0 if not data["labels_available"] else data["valid_images"], 0 if not data["license_reviewed"] else data["valid_images"]]
    colors = ["#2563eb", "#dc2626", "#f59e0b", "#64748b", "#64748b"]
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("Count")
    ax.set_title("Repository test originals: data readiness audit")
    ax.set_ylim(0, max(values + [1]) * 1.18)
    near_pairs = data.get("dhash_hamming_le_2_candidate_pairs", 0)
    ax.text(0.01, -0.24, f"Aggregate scan of test_data/originals. dHash Hamming≤2 candidate pairs: {near_pairs} (manual review only). No labels, license review, or research split were available.", transform=ax.transAxes, fontsize=8, wrap=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__": main()
