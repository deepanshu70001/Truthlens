"""Run an auditable binary detector evaluation from a reviewed CSV manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import hashlib
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def worktree_fingerprint() -> dict:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
        entries = []
        for line in status.splitlines():
            if line[3:].replace("\\", "/").startswith("experiments/results/"):
                continue
            path = ROOT / line[3:]
            content = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing-or-nonfile"
            entries.append(f"{line}\0{content}")
        return {"dirty": bool(status.strip()), "status_sha256": hashlib.sha256(status.encode()).hexdigest(), "worktree_content_sha256": hashlib.sha256("\n".join(entries).encode()).hexdigest()}
    except Exception:
        return {"dirty": None, "status_sha256": None}


def binary_metrics(y: list[int], pred: list[int], score: list[float]) -> dict:
    tp = sum(a == b == 1 for a, b in zip(y, pred)); tn = sum(a == b == 0 for a, b in zip(y, pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, pred)); fn = sum(a == 1 and b == 0 for a, b in zip(y, pred))
    div = lambda n, d: n / d if d else None
    # ROC-AUC with average ranks for ties; unavailable if one class is absent.
    auc = None
    average_precision = None
    if len(set(y)) == 2:
        order = sorted(range(len(score)), key=lambda i: score[i])
        ranks = [0.0] * len(score); i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and score[order[j]] == score[order[i]]: j += 1
            rank = (i + 1 + j) / 2
            for k in order[i:j]: ranks[k] = rank
            i = j
        n1 = sum(y); n0 = len(y) - n1
        auc = (sum(r for r, label in zip(ranks, y) if label == 1) - n1 * (n1 + 1) / 2) / (n1 * n0)
        descending = sorted(range(len(score)), key=lambda i: score[i], reverse=True)
        seen_pos = 0
        precision_at_positive = []
        for rank, idx in enumerate(descending, 1):
            if y[idx] == 1:
                seen_pos += 1
                precision_at_positive.append(seen_pos / rank)
        average_precision = sum(precision_at_positive) / n1 if n1 else None
    p, r = div(tp, tp + fp), div(tp, tp + fn)
    return {"n": len(y), "class_counts": {"real": y.count(0), "ai_generated": y.count(1)}, "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}, "accuracy": div(tp + tn, len(y)), "precision_ai_generated": p, "recall_ai_generated": r, "specificity_real": div(tn, tn + fp), "f1_ai_generated": div(2*p*r, p+r) if p is not None and r is not None else None, "roc_auc": auc, "average_precision_pr_auc": average_precision}


def calibration_diagnostics(y: list[int], score: list[float], bins: int = 10) -> dict:
    """Descriptive diagnostics only; scores are not assumed calibrated probabilities."""
    if not y:
        return {"brier_score": None, "ece": None, "bin_count": bins}
    brier = sum((p - target) ** 2 for p, target in zip(score, y)) / len(y)
    ece, rows = 0.0, []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        indices = [i for i, p in enumerate(score) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if not indices: continue
        confidence = sum(score[i] for i in indices) / len(indices)
        observed = sum(y[i] for i in indices) / len(indices)
        ece += len(indices) / len(y) * abs(confidence - observed)
        rows.append({"lower": lo, "upper": hi, "n": len(indices), "mean_score": confidence, "observed_positive_rate": observed})
    return {"brier_score": brier, "ece": ece, "bin_count": bins, "bins": rows, "warning": "Diagnostics are reported for raw uncalibrated scores; they do not establish calibrated probabilities. ECE depends on binning."}


def transforms(image: Image.Image) -> dict[str, Image.Image]:
    image = image.convert("RGB")
    w, h = image.size
    crop = image.crop((int(w*.05), int(h*.05), max(int(w*.95), 1), max(int(h*.95), 1))).resize((w, h))
    buf = __import__("io").BytesIO()
    image.save(buf, format="JPEG", quality=75)
    buf.seek(0)
    jpeg = Image.open(buf).convert("RGB").copy()
    noise = np.asarray(image).astype(np.int16)
    rng = np.random.default_rng(1729)
    noise = np.clip(noise + rng.normal(0, 5, noise.shape), 0, 255).astype(np.uint8)
    return {"resize_roundtrip_75pct": image.resize((max(1,w//2),max(1,h//2))).resize((w,h)), "jpeg_q75": jpeg, "center_crop_90pct": crop, "gaussian_blur_radius_1": image.filter(ImageFilter.GaussianBlur(1)), "brightness_0.9": ImageEnhance.Brightness(image).enhance(.9), "contrast_1.1": ImageEnhance.Contrast(image).enhance(1.1), "gaussian_noise_sigma5_seed1729": Image.fromarray(noise)}


def read_manifest(path: Path, split: str) -> tuple[list[dict], list[str]]:
    rows, errors = [], []
    required = {"image_path", "label", "split", "group_id", "source_url", "license_ref"}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Manifest needs columns: {sorted(required)}")
        for line, row in enumerate(reader, 2):
            if row["split"].strip() != split: continue
            if row["label"].strip() not in {"real", "ai_generated"}: errors.append(f"line {line}: unsupported/missing label"); continue
            if not row["group_id"].strip(): errors.append(f"line {line}: group_id required"); continue
            if not row["source_url"].strip() or not row["license_ref"].strip(): errors.append(f"line {line}: provenance/license reference required"); continue
            image = (path.parent / row["image_path"]).resolve()
            if not image.is_file(): errors.append(f"line {line}: image missing"); continue
            row["_path"] = str(image); rows.append(row)
    return rows, errors


def inspect_split_leakage(path: Path) -> dict:
    by_sha: dict[str, set[str]] = defaultdict(set)
    by_dhash: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            split = row.get("split", "").strip()
            if split not in {"train", "validation", "test"} or not row.get("image_path"):
                continue
            image = (path.parent / row["image_path"]).resolve()
            if not image.is_file():
                continue
            try:
                by_sha[hashlib.sha256(image.read_bytes()).hexdigest()].add(split)
                with Image.open(image) as im:
                    pixels = list(im.convert("L").resize((9, 8)).tobytes())
                code = 0
                for y0 in range(8):
                    for x0 in range(8): code = (code << 1) | int(pixels[y0*9+x0] > pixels[y0*9+x0+1])
                by_dhash[f"{code:016x}"].add(split)
            except Exception:
                continue
    exact = sum(1 for splits in by_sha.values() if len(splits) > 1)
    candidates = sum(1 for splits in by_dhash.values() if len(splits) > 1)
    return {"exact_hashes_crossing_splits": exact, "dhash_candidates_crossing_splits": candidates, "dhash_caveat": "candidate-only signal; inspect by group before exclusion"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--split", choices=["validation", "test"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--allow-unreviewed", action="store_true", help="diagnostic only; never present resulting metrics as research results")
    ap.add_argument("--robustness", action="store_true", help="also score fixed, documented image transformations")
    args = ap.parse_args()
    np.random.seed(1729)
    torch.manual_seed(1729)
    rows, manifest_errors = read_manifest(args.manifest, args.split)
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    with args.manifest.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("group_id"): groups_by_split[row.get("split", "")].add(row["group_id"])
    split_overlap = sorted(groups_by_split.get("validation", set()) & groups_by_split.get("test", set()))
    content_leakage = inspect_split_leakage(args.manifest)
    result = {"experiment_id": f"auth-{args.split}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}", "created_utc": datetime.now(timezone.utc).isoformat(), "revision": revision(), "worktree": worktree_fingerprint(), "split": args.split, "random_seed": 1729, "manifest": str(args.manifest), "model": None, "thresholds": None, "software": {"python": platform.python_version(), "torch": torch.__version__, "pillow": Image.__version__, "device": "cuda" if torch.cuda.is_available() else "cpu"}, "n_manifest_rows": len(rows), "manifest_errors": manifest_errors, "validation_test_group_overlap": split_overlap, "content_split_leakage": content_leakage, "status": "NOT_EVALUABLE", "reason": None}
    if not rows:
        result["reason"] = "No eligible labeled rows in requested split; no model inference or accuracy metric was run."
    elif (manifest_errors or split_overlap or content_leakage["exact_hashes_crossing_splits"]) and not args.allow_unreviewed:
        result["reason"] = "Manifest validation failed. Correct labels/provenance or split leakage before evaluation."
    else:
        from backend.app.config import settings
        from backend.app.services.authenticity.detector import AuthenticityDetector
        detector = AuthenticityDetector(); start = time.perf_counter(); detector.load(settings.DEVICE)
        load_ms = (time.perf_counter() - start) * 1000
        y, scores, labels, times = [], [], [], []
        per_class = defaultdict(list)
        for row in rows:
            with Image.open(row["_path"]) as im: image = im.convert("RGB")
            t0 = time.perf_counter(); pred = detector.detect(image); times.append((time.perf_counter() - t0) * 1000)
            y.append(int(row["label"] == "ai_generated")); scores.append(float(pred.ai_generated_score)); labels.append(pred.label)
            per_class[row["label"]].append({"label": pred.label, "score": float(pred.ai_generated_score)})
        pred = [int(s >= settings.AI_GENERATED_THRESHOLD) if s >= settings.AI_GENERATED_THRESHOLD or s <= settings.AUTHENTIC_THRESHOLD else -1 for s in scores]
        decided = [i for i, p in enumerate(pred) if p >= 0]
        metrics = binary_metrics([y[i] for i in decided], [pred[i] for i in decided], [scores[i] for i in decided]) if decided else None
        thresholds = {}
        for lo, hi in ((0.35, 0.65), (0.40, 0.60), (0.45, 0.55)):
            indices = [i for i, s in enumerate(scores) if s <= lo or s >= hi]
            p = [int(scores[i] >= hi) for i in indices]
            thresholds[f"band_{lo:.2f}_{hi:.2f}"] = {"coverage": len(indices)/len(scores), "uncertain_count": len(scores)-len(indices), "covered_metrics": binary_metrics([y[i] for i in indices], p, [scores[i] for i in indices]) if indices else None}
        p50 = [int(s >= 0.50) for s in scores]
        thresholds["single_threshold_0.50_no_abstention"] = {"coverage": 1.0, "uncertain_count": 0, "metrics": binary_metrics(y, p50, scores)}
        majority = int(sum(y) > len(y) / 2)
        majority_pred = [majority] * len(y)
        result.update({"status": "COMPLETED_DIAGNOSTIC" if args.allow_unreviewed and (manifest_errors or split_overlap or content_leakage["exact_hashes_crossing_splits"]) else "COMPLETED", "model": settings.AUTHENTICITY_MODEL, "thresholds": {"operational_real_max": settings.AUTHENTIC_THRESHOLD, "operational_ai_min": settings.AI_GENERATED_THRESHOLD, "semantics": "uncalibrated detector score; abstention lies between thresholds"}, "load_ms": load_ms, "inference_ms": {"median": statistics.median(times), "mean": statistics.mean(times), "min": min(times), "max": max(times)}, "coverage": len(decided)/len(rows), "uncertain_count": len(rows)-len(decided), "operational_metrics_on_covered_only": metrics, "fixed_threshold_sweep_no_tuning": thresholds, "raw_score_calibration_diagnostics": calibration_diagnostics(y, scores), "majority_class_baseline": {"predicted_class": "ai_generated" if majority else "real", "metrics": binary_metrics(y, majority_pred, [float(majority)]*len(y))}, "per_example_predictions": [{"group_id": r["group_id"], "label": r["label"], "score": s, "prediction": lab} for r,s,lab in zip(rows,scores,labels)], "warning": "Metrics are valid only if labels, source permissions and split independence have been independently reviewed. Score is not a calibrated probability."})
        if args.robustness:
            robustness = {}
            for row, base_score, base_label in zip(rows, scores, labels):
                with Image.open(row["_path"]) as im: source = im.convert("RGB")
                for name, transformed in transforms(source).items():
                    result_pred = detector.detect(transformed)
                    item = robustness.setdefault(name, {"score_abs_deltas": [], "label_flips": 0, "n": 0})
                    item["score_abs_deltas"].append(abs(float(result_pred.ai_generated_score) - base_score))
                    item["label_flips"] += int(result_pred.label != base_label); item["n"] += 1
            for item in robustness.values():
                item["mean_absolute_score_delta"] = statistics.mean(item.pop("score_abs_deltas"))
                item["label_flip_rate"] = item["label_flips"] / item["n"]
            result["robustness_sensitivity_only"] = {"transform_parameters": "half-size resize roundtrip; JPEG quality 75; central 90% crop then resize; Gaussian blur radius 1; brightness 0.9; contrast 1.1; additive Gaussian noise sigma 5/255 with seed 1729", "results": robustness, "warning": "Score/label sensitivity on supplied examples does not establish robustness without an appropriate independent dataset."}
        detector.unload()
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
