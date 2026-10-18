"""Build results/ml/comparison.csv and comparison.md from the experiment JSON files."""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["cnn_scratch", "resnet18_frozen"]  # baseline first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results" / "ml")
    a = ap.parse_args()

    rows = []
    for name in ORDER:
        f = a.results_dir / f"{name}.json"
        if not f.exists():
            print(f"missing {f.name}, skipping")
            continue
        r = json.loads(f.read_text())
        t = r["test"]
        rows.append({"experiment": r["name"], "model": r["model"], "mode": r["mode"],
                     "img_size": r["img_size"], "trainable_params": r["trainable_params"],
                     "total_params": r["total_params"], "accuracy": t["accuracy"],
                     "precision": t["precision"], "recall": t["recall"], "f1": t["f1"],
                     "roc_auc": t["roc_auc"], "train_seconds": r["train_seconds"]})
    if not rows:
        raise SystemExit("no result files found - run train_baselines.py first")

    df = pd.DataFrame(rows)
    df.to_csv(a.results_dir / "comparison.csv", index=False)

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    (a.results_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
