"""Make figures from experiment results: results/ml/figures/*.png"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["cnn_scratch", "siglip_frozen", "resnet18_frozen"]
LABELS = {"cnn_scratch": "CNN (scratch, baseline)", "siglip_frozen": "SigLIP (frozen, linear head)", "resnet18_frozen": "ResNet-18 (ImageNet, frozen)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results" / "ml")
    a = ap.parse_args()
    out = a.results_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    res = {n: json.loads((a.results_dir / f"{n}.json").read_text())
           for n in ORDER if (a.results_dir / f"{n}.json").exists()}
    if not res:
        raise SystemExit("no result files found")

    # 1. training curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for n, r in res.items():
        ep = [h["epoch"] for h in r["history"]]
        ax[0].plot(ep, [h["train_loss"] for h in r["history"]], label=LABELS[n])
        ax[1].plot(ep, [h["val_acc"] for h in r["history"]], label=LABELS[n])
    ax[0].set(title="Training loss", xlabel="epoch", ylabel="BCE loss")
    ax[1].set(title="Validation accuracy", xlabel="epoch", ylabel="accuracy")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)

    # 2. confusion matrices
    fig, ax = plt.subplots(1, len(res), figsize=(4.5 * len(res), 4), squeeze=False)
    for a_, (n, r) in zip(ax[0], res.items()):
        c = r["test"]["confusion_matrix"]
        m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        a_.imshow(m, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a_.text(j, i, m[i, j], ha="center", va="center",
                        color="white" if m[i, j] > m.max() / 2 else "black", fontsize=14)
        a_.set(title=LABELS[n], xticks=[0, 1], yticks=[0, 1],
               xticklabels=["real", "fake"], yticklabels=["real", "fake"],
               xlabel="predicted", ylabel="actual")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    # 3. ROC curves
    fig, ax = plt.subplots(figsize=(5, 5))
    for n, r in res.items():
        p = a.results_dir / f"{n}_preds.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        fpr, tpr, _ = roc_curve(d["y_true"], d["prob_fake"])
        ax.plot(fpr, tpr, label=f"{LABELS[n]} (AUC {r['test']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(title="ROC curve (test set)", xlabel="false positive rate", ylabel="true positive rate")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "roc_curves.png", dpi=150)
    plt.close(fig)

    # 4. metric bars
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.8 / len(res)
    for k, (n, r) in enumerate(res.items()):
        ax.bar(np.arange(len(metrics)) + k * w, [r["test"][m] for m in metrics], w, label=LABELS[n])
    ax.set(title="Test metrics", ylim=(0, 1.05), xticks=np.arange(len(metrics)) + w * (len(res) - 1) / 2,
           xticklabels=metrics)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "metrics_comparison.png", dpi=150)
    plt.close(fig)
    print("saved figures to", out)


if __name__ == "__main__":
    main()
