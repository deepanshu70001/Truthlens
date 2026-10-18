"""CIFAKE real-vs-AI baselines for TruthLens.

Two experiments only:
  cnn      -> basic CNN trained from scratch (the BASELINE)
  resnet18 -> ImageNet-pretrained ResNet-18, backbone frozen, new final layer trained

Usage (from repo root):
  python scripts/ml/train_baselines.py --model cnn
  python scripts/ml/train_baselines.py --model resnet18

Outputs:
  results/ml/<name>.json        metrics + training history
  results/ml/<name>_preds.csv   per-image test predictions (used for ROC plots)
  models/baselines/<name>.pt    best weights by validation accuracy (git-ignored)
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261014
POSITIVE = "ai_generated"  # positive class = fake
DEFAULTS = {  # per-model defaults: (mode, img_size, lr, epochs)
    "cnn": ("scratch", 32, 1e-3, 20),
    "resnet18": ("frozen", 224, 1e-3, 10),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleCNN(nn.Module):
    """Basic CNN: 3 conv blocks + small classifier. Input 3x32x32, output 1 logit."""

    def __init__(self):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name):
    if name == "cnn":
        return SimpleCNN()
    m = models.resnet18(weights="IMAGENET1K_V1")
    for p in m.parameters():  # freeze the whole ImageNet backbone
        p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, 1)  # new head, trainable
    return m


def get_loaders(data_dir, size, pretrained, batch_size, workers):
    mean, std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if pretrained \
        else ((0.5,) * 3, (0.5,) * 3)
    norm = transforms.Normalize(mean, std)
    train_tf = transforms.Compose([transforms.Resize(size), transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), norm])

    train_full = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_full = datasets.ImageFolder(data_dir / "train", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)
    pos_idx = train_full.class_to_idx[POSITIVE]
    print("classes:", train_full.class_to_idx, "-> positive (fake) index:", pos_idx)

    # stratified 80/20 train/val split with a fixed seed
    rng = np.random.RandomState(SEED)
    targets = np.array(train_full.targets)
    tr_idx, va_idx = [], []
    for c in np.unique(targets):
        idx = rng.permutation(np.where(targets == c)[0])
        cut = int(0.8 * len(idx))
        tr_idx += idx[:cut].tolist()
        va_idx += idx[cut:].tolist()

    kw = dict(batch_size=batch_size, num_workers=workers)
    return (DataLoader(Subset(train_full, tr_idx), shuffle=True, **kw),
            DataLoader(Subset(val_full, va_idx), **kw),
            DataLoader(test_ds, **kw), pos_idx, test_ds,
            {"train": len(tr_idx), "val": len(va_idx), "test": len(test_ds)})


@torch.no_grad()
def predict(model, loader, pos_idx, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(x.to(device)).squeeze(1)).cpu().numpy())
        ys.append((y.numpy() == pos_idx).astype(int))
    return np.concatenate(probs), np.concatenate(ys)


def compute_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(DEFAULTS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "external" / "cifake")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results" / "ml")
    ap.add_argument("--weights-dir", type=Path, default=ROOT / "models" / "baselines")
    args = ap.parse_args()

    mode, img_size, lr, epochs = DEFAULTS[args.model]
    lr, epochs = args.lr or lr, args.epochs or epochs
    pretrained = args.model != "cnn"
    name = f"{args.model}_{mode}"

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"experiment={name} device={device} img_size={img_size} lr={lr} epochs={epochs}")

    train_dl, val_dl, test_dl, pos_idx, test_ds, sizes = get_loaders(
        args.data_dir, img_size, pretrained, args.batch_size, args.workers)
    model = build_model(args.model).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.weights_dir / f"{name}.pt"
    best_val, history, t0 = -1.0, [], time.time()

    for ep in range(1, epochs + 1):
        # frozen backbone: keep it in eval mode so BatchNorm statistics are not changed
        model.eval() if mode == "frozen" else model.train()
        total = 0.0
        for x, y in train_dl:
            x = x.to(device)
            y = torch.as_tensor((y.numpy() == pos_idx).astype("float32")).to(device)
            opt.zero_grad()
            loss = loss_fn(model(x).squeeze(1), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        vp, vy = predict(model, val_dl, pos_idx, device)
        val_acc = float(accuracy_score(vy, vp >= 0.5))
        history.append({"epoch": ep, "train_loss": total / sizes["train"], "val_acc": val_acc})
        print(f"epoch {ep:02d} loss {total / sizes['train']:.4f} val_acc {val_acc:.3f}")
        if val_acc > best_val:  # choose best epoch on VALIDATION only, never on test
            best_val = val_acc
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    tp_, ty_ = predict(model, test_dl, pos_idx, device)
    result = {"name": name, "model": args.model, "mode": mode, "img_size": img_size,
              "lr": lr, "epochs": epochs, "seed": SEED, "sizes": sizes,
              "trainable_params": n_trainable, "total_params": n_total,
              "best_val_acc": best_val, "train_seconds": round(time.time() - t0, 1),
              "test": compute_metrics(ty_, tp_), "history": history}
    (args.results_dir / f"{name}.json").write_text(json.dumps(result, indent=2))

    with open(args.results_dir / f"{name}_preds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "y_true", "prob_fake"])
        for (path, _), y, p in zip(test_ds.samples, ty_, tp_):
            w.writerow([Path(path).name, int(y), round(float(p), 6)])

    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()
