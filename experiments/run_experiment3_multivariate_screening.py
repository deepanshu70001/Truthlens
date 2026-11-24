"""Exploratory Experiment 3: multivariate forensic complementarity screening.

Uses only the authorized Phase 4 training manifest and verified Experiment 2
derived artifacts. Source images and protected validation/calibration/test/
reserve partitions are never opened.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import platform
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
E2 = ROOT / "research" / "experiment2_forensic_screening"
OUT = ROOT / "research" / "experiment3_multivariate_screening"
MANIFEST = ROOT / "experiments" / "manifests" / "phase4_training_manifest.csv"
CHECKPOINT = ROOT / "models" / "baselines" / "siglip_frozen_head.pt"

EXPECTED_MANIFEST_SHA = "22ea2825c7ab75ef110c7a6e2a7c67e69459a3afd3ed6159adbcbed7abc168ed"
EXPECTED_CHECKPOINT_SHA = "e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538"
N_EXPECTED = 6995
LABEL_COUNTS = {"likely_authentic": 3515, "ai_generated": 3480}
SEED = 314159
OUTER_FOLDS = (1, 2, 3, 4, 5)
INNER_FOLDS = 4
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
CORR_THRESHOLD = 0.85
BOOTSTRAPS = 5000
SIGNFLIPS = 20000
FDR_ALPHA = 0.05
EPS = 1e-15

FEATURES = {
    "noise_consistency": "noise_consistency_score",
    "edge_consistency": "edge_consistency_score",
    "texture_consistency": "texture_consistency_score",
    "resampling_clues": "resampling_clues_score",
}
MODELS = {
    "M0_siglip_baseline": ([], False),
    "M1_noise_edge": (["noise_consistency", "edge_consistency"], False),
    "M2_texture_resampling": (["texture_consistency", "resampling_clues"], False),
    "M3_all_four": (list(FEATURES), False),
    "M4_correlation_adjusted": (list(FEATURES), True),
}
COMPARISONS = [name for name in MODELS if name != "M0_siglip_baseline"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("truthlens.experiment3")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def crlf_reconstruction(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")


def verify_hash(path: Path, expected: str, expected_size: int | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required input missing: {path.relative_to(ROOT)}")
    raw = path.read_bytes()
    raw_hash = sha256(raw)
    reconstructed = crlf_reconstruction(raw)
    reconstructed_hash = sha256(reconstructed)
    exact = raw_hash == expected
    normalized = reconstructed_hash == expected
    if not exact and not normalized:
        raise RuntimeError(
            f"SHA-256 mismatch for {path.relative_to(ROOT)}: "
            f"raw={raw_hash}, CRLF-reconstructed={reconstructed_hash}, expected={expected}"
        )
    matched_size = len(raw) if exact else len(reconstructed)
    if expected_size is not None and matched_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {path.relative_to(ROOT)}: "
            f"matched={matched_size}, expected={expected_size}"
        )
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "expected_sha256": expected,
        "actual_sha256": raw_hash,
        "crlf_reconstructed_sha256": reconstructed_hash,
        "actual_byte_size": len(raw),
        "expected_original_byte_size": expected_size,
        "status": "exact" if exact else "verified_after_crlf_reconstruction",
    }


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def finite_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def verify_inputs() -> dict:
    repro_path = E2 / "reproducibility_manifest.json"
    summary_path = E2 / "manifest_summary.json"
    identity_path = E2 / "model_identity.json"
    config_path = E2 / "exp_config.json"
    for path in (repro_path, summary_path, identity_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing Experiment 2 provenance: {path.relative_to(ROOT)}")

    repro = json.loads(repro_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    exp2_config = json.loads(config_path.read_text(encoding="utf-8"))

    if summary.get("total_rows") != N_EXPECTED or summary.get("sha256") != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("Experiment 2 manifest summary does not match the authorized manifest.")
    if summary.get("label_counts") != LABEL_COUNTS:
        raise RuntimeError(f"Unexpected training label distribution: {summary.get('label_counts')}")
    if identity.get("model_id") != "truthlens/siglip-cifake-frozen-head":
        raise RuntimeError("Unexpected primary baseline model identity.")
    if identity.get("checkpoint_sha256_initial") != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Experiment 2 baseline initial checkpoint hash differs from authorization.")
    if identity.get("checkpoint_sha256_final") != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Experiment 2 reports the baseline checkpoint was modified.")

    checked = {
        "training_manifest": verify_hash(MANIFEST, EXPECTED_MANIFEST_SHA),
        "baseline_checkpoint": verify_hash(CHECKPOINT, EXPECTED_CHECKPOINT_SHA),
    }
    for filename in ("forensic_features.csv", "fold_assignments.csv", "siglip_logits.csv"):
        artifact = repro.get("artifacts", {}).get(filename)
        if not isinstance(artifact, dict):
            raise RuntimeError(f"Missing artifact integrity record: {filename}")
        checked[filename] = verify_hash(
            E2 / filename, artifact["sha256"], int(artifact["byte_size"])
        )
    return {
        "experiment2_protocol_version": exp2_config.get("protocol_version"),
        "experiment2_reproducibility_manifest": repro_path.relative_to(ROOT).as_posix(),
        "authorized_manifest_sha256": EXPECTED_MANIFEST_SHA,
        "authorized_row_count": N_EXPECTED,
        "baseline_model_identity": identity,
        "verified_inputs": checked,
        "hash_policy": "Accept exact bytes or CRLF reconstruction only; otherwise fail closed.",
    }


def load_and_validate_rows():
    manifest = read_csv(MANIFEST)
    features = read_csv(E2 / "forensic_features.csv")
    folds = read_csv(E2 / "fold_assignments.csv")
    logits = read_csv(E2 / "siglip_logits.csv")
    if any(len(rows) != N_EXPECTED for rows in (manifest, features, folds, logits)):
        raise RuntimeError(
            f"Expected {N_EXPECTED} rows in every input; found "
            f"manifest={len(manifest)}, features={len(features)}, folds={len(folds)}, logits={len(logits)}"
        )

    maps = [
        {r["sample_id"]: r for r in rows}
        for rows in (manifest, features, folds, logits)
    ]
    if any(len(m) != N_EXPECTED for m in maps):
        raise RuntimeError("Duplicate sample identifiers in one or more input CSVs.")
    manifest_by_id, feature_by_id, fold_by_id, logit_by_id = maps
    ids = set(manifest_by_id)
    if any(set(m) != ids for m in maps[1:]):
        raise RuntimeError("Sample ID sets disagree across manifest and Experiment 2 artifacts.")

    label_counts = Counter(r["normalized_label"] for r in manifest)
    if dict(label_counts) != LABEL_COUNTS:
        raise RuntimeError(f"Authorized class distribution mismatch: {dict(label_counts)}")

    groups_to_folds = defaultdict(set)
    fold_counts = defaultdict(Counter)
    records = []
    for sid, source in manifest_by_id.items():
        f = feature_by_id[sid]
        fold = fold_by_id[sid]
        logit = logit_by_id[sid]
        label_name = source["normalized_label"]
        label = 1 if label_name == "ai_generated" else 0 if label_name == "likely_authentic" else None
        if label is None or f["normalized_label"] != label_name or int(f["label_int"]) != label:
            raise RuntimeError(f"Label mismatch for sample {sid}.")
        if source.get("source_group_id") != f.get("group_id") or fold.get("group_id") != f.get("group_id"):
            raise RuntimeError(f"Group mismatch for sample {sid}.")
        if int(fold["label_int"]) != label or int(fold["outer_fold"]) != int(f["outer_fold"]):
            raise RuntimeError(f"Fold assignment mismatch for sample {sid}.")
        z1, z2 = finite_float(f.get("siglip_ai_logit")), finite_float(logit.get("siglip_ai_logit"))
        if z1 is None or z2 is None or not math.isclose(z1, z2, rel_tol=1e-10, abs_tol=1e-10):
            raise RuntimeError(f"SigLIP logit cross-check failed for sample {sid}.")
        fold_id = int(f["outer_fold"])
        if fold_id not in OUTER_FOLDS:
            raise RuntimeError(f"Invalid outer fold {fold_id}.")
        groups_to_folds[f["group_id"]].add(fold_id)
        fold_counts[fold_id][label] += 1
        row = {
            "sample_id": sid, "group_id": f["group_id"], "label_int": label,
            "normalized_label": label_name, "outer_fold": fold_id,
            "siglip_ai_logit": z1, "file_format": f.get("file_format"),
        }
        for name, column in FEATURES.items():
            row[column] = finite_float(f.get(column))
            row[f"{name}_outcome"] = f.get(f"{name}_outcome")
        records.append(row)

    crossed = [group for group, assigned in groups_to_folds.items() if len(assigned) != 1]
    if crossed:
        raise RuntimeError(f"Found {len(crossed)} source groups crossing outer folds.")
    fold_summary = []
    for fold_id in OUTER_FOLDS:
        counts = fold_counts[fold_id]
        if counts[0] == 0 or counts[1] == 0:
            raise RuntimeError(f"Outer fold {fold_id} does not contain both classes.")
        fold_summary.append({
            "outer_fold": fold_id, "n": int(counts[0] + counts[1]),
            "real_n": int(counts[0]), "ai_n": int(counts[1]),
            "groups": sum(1 for assigned in groups_to_folds.values() if fold_id in assigned),
        })

    format_by_label = {
        label: dict(Counter(r.get("file_format") for r in manifest if r["normalized_label"] == label))
        for label in LABEL_COUNTS
    }
    return records, {
        "manifest_rows": len(manifest), "unique_groups": len(groups_to_folds),
        "groups_crossing_outer_folds": len(crossed),
        "fold_summary": fold_summary, "label_counts": dict(label_counts),
        "format_by_label": format_by_label,
    }


def common_complete_cases(records):
    cohort, exclusions = [], Counter()
    for row in records:
        reasons = []
        for name, column in FEATURES.items():
            if row[f"{name}_outcome"] != "success":
                reasons.append(f"{name}_outcome_{row[f'{name}_outcome'] or 'missing'}")
            if row[column] is None:
                reasons.append(f"{name}_score_missing_or_nonfinite")
        if reasons:
            exclusions[";".join(reasons)] += 1
        else:
            cohort.append(dict(row))
    classes = Counter(r["normalized_label"] for r in cohort)
    if not cohort or len(classes) != 2:
        raise RuntimeError("Complete-case cohort does not contain both classes.")
    return cohort, {
        "definition": "Common intersection where all four continuous probes are successful and finite.",
        "n": len(cohort),
        "coverage_pct": round(100.0 * len(cohort) / len(records), 4),
        "class_counts": dict(classes), "unique_groups": len({r["group_id"] for r in cohort}),
        "excluded_n": len(records) - len(cohort),
        "exclusion_reason_combinations": dict(exclusions),
        "imputation": "none", "ela_included": False, "jpeg_artifacts_included": False,
    }


def correlation_clusters(x_train, feature_names, threshold=CORR_THRESHOLD):
    if x_train.ndim != 2 or x_train.shape[1] != len(feature_names):
        raise ValueError("Feature matrix and feature names do not align.")
    if x_train.shape[1] <= 1:
        return [[i] for i in range(x_train.shape[1])]
    ranks = np.column_stack([rankdata(x_train[:, j], method="average") for j in range(x_train.shape[1])])
    corr = np.corrcoef(ranks, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, -1.0, 1.0)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    tree = linkage(squareform(dist, checks=False), method="complete")
    labels = fcluster(tree, t=1.0 - threshold, criterion="distance")
    groups = defaultdict(list)
    for index, cluster_id in enumerate(labels):
        groups[int(cluster_id)].append(index)
    return sorted(groups.values(), key=lambda g: min(g))


def align_cluster_signs(x_train, feature_names, clusters):
    """Align negatively correlated cluster members to a deterministic train-only anchor."""
    if x_train.shape[1] == 0:
        return [], []
    ranks = np.column_stack([rankdata(x_train[:, j], method="average") for j in range(x_train.shape[1])])
    corr = np.corrcoef(ranks, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    signs = [1 for _ in feature_names]
    metadata = []
    for cluster in clusters:
        anchor = min(cluster)
        cluster_signs = []
        for member in cluster:
            sign = 1 if member == anchor or corr[anchor, member] >= 0 else -1
            signs[member] = sign
            cluster_signs.append({"feature": feature_names[member], "sign_aligned_to_anchor": sign})
        metadata.append({"anchor": feature_names[anchor], "members": cluster_signs})
    return signs, metadata


def design_matrices(z_train, z_eval, f_train, f_eval, names, correlation_adjusted):
    z_scaler = StandardScaler()
    ztr = z_scaler.fit_transform(z_train.reshape(-1, 1))
    zev = z_scaler.transform(z_eval.reshape(-1, 1))
    if not names:
        return ztr, zev, {"clusters": [], "dimensions_after_adjustment": 0}

    scaler = StandardScaler()
    ftr = scaler.fit_transform(f_train)
    fev = scaler.transform(f_eval)
    if not correlation_adjusted:
        return (
            np.column_stack([ztr, ftr]), np.column_stack([zev, fev]),
            {"clusters": [[n] for n in names], "dimensions_after_adjustment": len(names)},
        )

    clusters = correlation_clusters(f_train, names)
    signs, alignment = align_cluster_signs(f_train, names, clusters)
    aligned_train = ftr * np.asarray(signs)
    aligned_eval = fev * np.asarray(signs)
    aggregate_train = np.column_stack([aligned_train[:, g].mean(axis=1) for g in clusters])
    aggregate_eval = np.column_stack([aligned_eval[:, g].mean(axis=1) for g in clusters])
    aggregate_scaler = StandardScaler()
    aggregate_train = aggregate_scaler.fit_transform(aggregate_train)
    aggregate_eval = aggregate_scaler.transform(aggregate_eval)
    return (
        np.column_stack([ztr, aggregate_train]), np.column_stack([zev, aggregate_eval]),
        {
            "clusters": [[names[i] for i in g] for g in clusters],
            "sign_alignment": alignment,
            "dimensions_after_adjustment": len(clusters),
            "threshold_abs_spearman": CORR_THRESHOLD,
            "clustering": "complete_linkage",
        },
    )


def make_inner_splits(rows, outer_fold):
    y = np.asarray([r["label_int"] for r in rows], dtype=int)
    groups = np.asarray([r["group_id"] for r in rows], dtype=str)
    splitter = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + outer_fold)
    splits = list(splitter.split(np.zeros((len(y), 1)), y, groups))
    if len(splits) != INNER_FOLDS:
        raise RuntimeError("Incorrect number of inner folds.")
    for i, (tr, va) in enumerate(splits, 1):
        if len(np.unique(y[tr])) != 2 or len(np.unique(y[va])) != 2:
            raise RuntimeError(f"Inner fold {i} in outer fold {outer_fold} lacks a class; fail closed.")
        if set(groups[tr]).intersection(set(groups[va])):
            raise RuntimeError(f"Group leakage in inner fold {i} of outer fold {outer_fold}.")
    return splits


def logloss(y, p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return float(log_loss(y, p, labels=[0, 1]))


def fit_outer(train, test, splits, feature_names, correlation_adjusted):
    y = np.asarray([r["label_int"] for r in train], dtype=int)
    z = np.asarray([r["siglip_ai_logit"] for r in train], dtype=float)
    z_test = np.asarray([r["siglip_ai_logit"] for r in test], dtype=float)
    f = np.asarray([[r[FEATURES[n]] for n in feature_names] for r in train], dtype=float) if feature_names else np.empty((len(train), 0))
    f_test = np.asarray([[r[FEATURES[n]] for n in feature_names] for r in test], dtype=float) if feature_names else np.empty((len(test), 0))

    losses_by_c = {}
    for c in C_GRID:
        fold_losses = []
        for tr, va in splits:
            xtr, xva, _ = design_matrices(z[tr], z[va], f[tr], f[va], feature_names, correlation_adjusted)
            model = LogisticRegression(C=float(c), penalty="l2", solver="lbfgs", max_iter=3000, random_state=SEED)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                model.fit(xtr, y[tr])
            fold_losses.append(logloss(y[va], model.predict_proba(xva)[:, 1]))
        if len(fold_losses) != INNER_FOLDS:
            raise RuntimeError("Inner-CV fold omission.")
        losses_by_c[str(c)] = float(np.mean(fold_losses))

    best_c = min(C_GRID, key=lambda c: losses_by_c[str(c)])
    xtr, xte, transform = design_matrices(z, z_test, f, f_test, feature_names, correlation_adjusted)
    final = LogisticRegression(C=float(best_c), penalty="l2", solver="lbfgs", max_iter=3000, random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        final.fit(xtr, y)
    return final.predict_proba(xte)[:, 1], float(best_c), {
        "inner_cv_mean_log_loss_by_C": losses_by_c, "selected_C": float(best_c), "transform": transform,
    }


def metric_block(y, p):
    pred = p >= 0.5
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "n": int(len(y)), "log_loss": logloss(y, p),
        "brier_score": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "confusion_matrix_real_ai": cm.tolist(),
    }


def aggregate_metrics(rows, predictions):
    y = np.asarray([r["label_int"] for r in rows], dtype=int)
    p = np.asarray([predictions[r["sample_id"]] for r in rows], dtype=float)
    folds = {}
    for fold in OUTER_FOLDS:
        selected = [r for r in rows if r["outer_fold"] == fold]
        yy = np.asarray([r["label_int"] for r in selected], dtype=int)
        pp = np.asarray([predictions[r["sample_id"]] for r in selected], dtype=float)
        folds[str(fold)] = metric_block(yy, pp)
    return {
        "pooled_log_loss": logloss(y, p),
        "pooled_brier_score": float(brier_score_loss(y, p)),
        "macro_outer_fold_roc_auc": float(np.mean([m["roc_auc"] for m in folds.values()])),
        "macro_outer_fold_pr_auc": float(np.mean([m["pr_auc"] for m in folds.values()])),
        "macro_outer_fold_balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in folds.values()])),
        "descriptive_only_pooled_roc_auc": float(roc_auc_score(y, p)),
        "pooled_confusion_matrix_real_ai_at_0_5": confusion_matrix(y, p >= 0.5, labels=[0, 1]).tolist(),
        "outer_fold_metrics": folds,
        "pooled_auc_inference": "descriptive only; cross-fold intercept shifts can distort pooled ranking",
    }


def paired_group_deltas(rows, loss0, lossm):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        sid = row["sample_id"]
        grouped[int(row["outer_fold"])][row["group_id"]].append(loss0[sid] - lossm[sid])
    return {
        fold: {group: (float(np.sum(values)), len(values)) for group, values in groups.items()}
        for fold, groups in grouped.items()
    }


def bootstrap_ci(groups_by_fold, n_boot=BOOTSTRAPS, seed=SEED + 700):
    rng = np.random.default_rng(seed)
    folds = {fold: list(groups.values()) for fold, groups in groups_by_fold.items()}
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        numerator, denominator = 0.0, 0
        for groups in folds.values():
            chosen = rng.integers(0, len(groups), size=len(groups))
            numerator += sum(groups[i][0] for i in chosen)
            denominator += sum(groups[i][1] for i in chosen)
        samples[b] = numerator / denominator
    return {
        "method": "paired group-cluster bootstrap, sampling groups with replacement within fixed outer folds",
        "replicates": n_boot, "seed": seed,
        "mean": float(np.mean(samples)), "standard_error": float(np.std(samples, ddof=1)),
        "ci_95_percentile": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))],
    }


def signflip_test(groups_by_fold, n_perm=SIGNFLIPS, seed=SEED + 900):
    rng = np.random.default_rng(seed)
    group_values = {
        fold: np.asarray([value[0] for value in groups.values()], dtype=float)
        for fold, groups in groups_by_fold.items()
    }
    n_total = sum(value[1] for groups in groups_by_fold.values() for value in groups.values())
    observed = sum(float(v.sum()) for v in group_values.values()) / n_total
    all_sums = np.concatenate(list(group_values.values()))
    extreme, completed = 0, 0
    while completed < n_perm:
        batch = min(1000, n_perm - completed)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(batch, len(all_sums)))
        permuted = (signs @ all_sums) / n_total
        extreme += int(np.sum(np.abs(permuted) >= abs(observed)))
        completed += batch
    return {
        "method": "two-sided paired group-cluster sign-flip test, group signs independently flipped within fixed outer folds",
        "replicates": n_perm, "seed": seed, "observed_delta": float(observed),
        "extreme_permutations": extreme, "p_value_two_sided": float((extreme + 1) / (n_perm + 1)),
    }


def benjamini_hochberg(p_values):
    ordered = sorted(p_values, key=lambda name: p_values[name])
    adjusted, running, m = {}, 1.0, len(ordered)
    for index in range(m - 1, -1, -1):
        name = ordered[index]
        running = min(running, p_values[name] * m / (index + 1))
        adjusted[name] = float(min(1.0, running))
    return adjusted


def run():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = verify_inputs()
    records, integrity = load_and_validate_rows()
    cohort, coverage = common_complete_cases(records)
    log.info("Complete-case cohort: %d/%d (%.2f%%)", len(cohort), len(records), coverage["coverage_pct"])

    predictions = {name: {} for name in MODELS}
    fit_details, inner_details, corr_details = {name: {} for name in MODELS}, {}, {}
    for outer in OUTER_FOLDS:
        train = [r for r in cohort if r["outer_fold"] != outer]
        test = [r for r in cohort if r["outer_fold"] == outer]
        if {r["group_id"] for r in train}.intersection({r["group_id"] for r in test}):
            raise RuntimeError(f"Outer fold {outer} group leakage.")
        inner = make_inner_splits(train, outer)
        inner_details[str(outer)] = {
            "train_n": len(train), "test_n": len(test),
            "train_groups": len({r["group_id"] for r in train}),
            "test_groups": len({r["group_id"] for r in test}),
            "inner_fold_sizes": [{"train_n": len(tr), "validation_n": len(va)} for tr, va in inner],
        }
        all_names = list(FEATURES)
        x_train = np.asarray([[r[FEATURES[n]] for n in all_names] for r in train], dtype=float)
        clusters = correlation_clusters(x_train, all_names)
        signs, alignment = align_cluster_signs(x_train, all_names, clusters)
        corr_details[str(outer)] = {
            "fit_scope": "outer training fold only", "threshold_abs_spearman": CORR_THRESHOLD,
            "clusters": [[all_names[i] for i in g] for g in clusters], "sign_alignment": alignment,
            "reduced_dimensions": len(clusters),
        }
        for model_name, (names, adjusted) in MODELS.items():
            probs, c_value, detail = fit_outer(train, test, inner, names, adjusted)
            for row, p in zip(test, probs):
                predictions[model_name][row["sample_id"]] = float(p)
            fit_details[model_name][str(outer)] = {
                "selected_C": c_value, "n_train": len(train), "n_test": len(test), "fit": detail,
            }
            log.info("%s fold %d complete", model_name, outer)

    expected = {r["sample_id"] for r in cohort}
    for name, pred in predictions.items():
        if set(pred) != expected:
            raise RuntimeError(f"Incomplete OOF predictions for {name}: {len(pred)}/{len(expected)}")

    metrics = {name: aggregate_metrics(cohort, pred) for name, pred in predictions.items()}
    labels = {r["sample_id"]: r["label_int"] for r in cohort}
    losses = {
        name: {
            sid: -math.log(max(EPS, min(1.0 - EPS, p if labels[sid] == 1 else 1.0 - p)))
            for sid, p in pred.items()
        }
        for name, pred in predictions.items()
    }

    paired, bootstraps, permutations, p_values = {}, {}, {}, {}
    baseline = "M0_siglip_baseline"
    for name in COMPARISONS:
        groups = paired_group_deltas(cohort, losses[baseline], losses[name])
        boot = bootstrap_ci(groups)
        perm = signflip_test(groups)
        bootstraps[name], permutations[name] = boot, perm
        p_values[name] = perm["p_value_two_sided"]
        paired[name] = {
            "comparison": f"{baseline} minus {name}",
            "n_common_samples": len(cohort),
            "delta_log_loss_improvement": metrics[baseline]["pooled_log_loss"] - metrics[name]["pooled_log_loss"],
            "bootstrap_ci_95": boot["ci_95_percentile"],
            "signflip_p_value": perm["p_value_two_sided"],
            "delta_brier_improvement": metrics[baseline]["pooled_brier_score"] - metrics[name]["pooled_brier_score"],
            "delta_macro_fold_roc_auc": metrics[name]["macro_outer_fold_roc_auc"] - metrics[baseline]["macro_outer_fold_roc_auc"],
            "delta_macro_fold_pr_auc": metrics[name]["macro_outer_fold_pr_auc"] - metrics[baseline]["macro_outer_fold_pr_auc"],
            "delta_macro_fold_balanced_accuracy": metrics[name]["macro_outer_fold_balanced_accuracy"] - metrics[baseline]["macro_outer_fold_balanced_accuracy"],
        }
    q_values = benjamini_hochberg(p_values)
    for name in COMPARISONS:
        lo, hi = bootstraps[name]["ci_95_percentile"]
        q = q_values[name]
        if q < FDR_ALPHA and lo > 0:
            verdict = "SUPPORTED_INCREMENTAL_IMPROVEMENT"
        elif q < FDR_ALPHA and hi < 0:
            verdict = "SUPPORTED_PERFORMANCE_DETRIMENT"
        else:
            verdict = "EXPLORATORY_INCONCLUSIVE"
        paired[name]["adjusted_q_value_bh"] = q
        paired[name]["verdict"] = verdict

    oof = []
    for row in sorted(cohort, key=lambda r: (r["outer_fold"], r["sample_id"])):
        item = {k: row[k] for k in ("sample_id", "group_id", "outer_fold", "label_int", "normalized_label")}
        for name in MODELS:
            item[f"{name}_prob_ai"] = predictions[name][row["sample_id"]]
        oof.append(item)
    with (OUT / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(oof[0]))
        writer.writeheader()
        writer.writerows(oof)

    config = {
        "experiment": "Experiment 3: Multivariate Forensic Complementarity Screening",
        "protocol_version": "1.0.0-exploratory",
        "study_status": "EXPLORATORY_POST_HOC_AFTER_EXPERIMENT2_GATE_FAILURE",
        "authorization_date": "2026-11-24",
        "seed": SEED, "authorized_training_rows": N_EXPECTED,
        "outer_folds": list(OUTER_FOLDS), "outer_fold_source": "Experiment 2 immutable group-aware assignments",
        "inner_folds": INNER_FOLDS, "inner_splitter": "StratifiedGroupKFold",
        "classifier": "L2 LogisticRegression(lbfgs, max_iter=3000)",
        "regularization_grid_C": list(C_GRID),
        "feature_scaling": "training-fold StandardScaler only",
        "correlation_adjustment": {
            "method": "absolute Spearman, complete linkage, signed alignment to deterministic cluster anchor",
            "threshold": CORR_THRESHOLD, "fit_scope": "inner and outer training folds only",
        },
        "no_imputation": True,
        "primary_endpoint": "paired OOF delta log-loss improvement = log_loss(M0) - log_loss(Mj)",
        "bootstrap_replicates": BOOTSTRAPS, "signflip_replicates": SIGNFLIPS,
        "multiple_testing": "Benjamini-Hochberg over M1-M4", "fdr_alpha": FDR_ALPHA,
        "models": MODELS,
        "not_accessed": [
            "diagnostic validation partition", "calibration partition", "held-out benchmark",
            "protected reserve", "source images", "Phase 9 validation features",
        ],
    }
    fold_summary = {"outer_folds": integrity["fold_summary"], "inner_folds": inner_details}
    (OUT / "experiment3_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (OUT / "input_integrity.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (OUT / "dataset_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    (OUT / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    (OUT / "fold_summary.json").write_text(json.dumps(fold_summary, indent=2), encoding="utf-8")
    (OUT / "correlation_clusters_by_outer_fold.json").write_text(json.dumps(corr_details, indent=2), encoding="utf-8")
    (OUT / "model_fit_details.json").write_text(json.dumps(fit_details, indent=2), encoding="utf-8")
    (OUT / "paired_metrics.json").write_text(json.dumps({"models": metrics, "comparisons": paired}, indent=2), encoding="utf-8")
    (OUT / "bootstrap_confidence_intervals.json").write_text(json.dumps(bootstraps, indent=2), encoding="utf-8")
    (OUT / "permutation_results.json").write_text(json.dumps(permutations, indent=2), encoding="utf-8")
    (OUT / "multiple_testing_results.json").write_text(json.dumps({
        "method": "Benjamini-Hochberg", "family": COMPARISONS, "alpha": FDR_ALPHA,
        "raw_p_values": p_values, "adjusted_q_values": q_values,
        "verdicts": {n: paired[n]["verdict"] for n in COMPARISONS},
    }, indent=2), encoding="utf-8")

    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "cpu_count": os.cpu_count(), "runtime_seconds": round(time.time() - started, 3),
    }
    (OUT / "environment_versions.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    rows = []
    for name in COMPARISONS:
        item = paired[name]
        ci = item["bootstrap_ci_95"]
        rows.append(
            f"| {name} | {item['delta_log_loss_improvement']:+.6f} | "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}] | {item['signflip_p_value']:.6f} | "
            f"{item['adjusted_q_value_bh']:.6f} | {item['verdict']} |"
        )
    report = f"""# TruthLens — Experiment 3: Multivariate Forensic Complementarity Screening

## Exploratory Research Protocol 1.0.0

**Execution date:** {time.strftime('%Y-%m-%d', time.gmtime())} UTC  
**Status:** COMPLETED — EXPLORATORY  
**Authorized partition:** experiments/manifests/phase4_training_manifest.csv (N={N_EXPECTED})  
**Common complete-case cohort:** {coverage['n']} ({coverage['coverage_pct']:.2f}% coverage)  
**Baseline:** truthlens/siglip-cifake-frozen-head  
**Output:** research/experiment3_multivariate_screening/

> Experiment 2's uploaded closeout audit explicitly recorded that Experiment 3 was not recommended because no univariate probe passed the pre-defined gate. The user subsequently authorized this follow-on. It is therefore exploratory and post hoc, not confirmatory. No result authorizes production directional forensic voting.

## 1. Research question

Do combinations of the four continuous Experiment 2 probes provide incremental predictive information beyond the SigLIP AI logit on a common complete-case group-aware nested-CV cohort?

ELA and JPEG block artifacts are excluded because Experiment 2 found zero AI-class overlap in their applicable subgroups.

## 2. Data integrity and isolation

- Authorized manifest rows: {integrity['manifest_rows']}; source groups: {integrity['unique_groups']}.
- Group clusters crossing the immutable Experiment 2 outer folds: {integrity['groups_crossing_outer_folds']}.
- All input CSVs and checkpoint hash were verified against the current checked-in Experiment 2 manifest and approved identities.
- Source images were not opened; only the authorized manifest metadata and Experiment 2 derived artifacts were used.
- Diagnostic validation, calibration, held-out benchmarks, and protected reserve partitions were not accessed.
- Format by class: {integrity['format_by_label']}. The training set is fully format-confounded; any gain may reflect file format or dataset source, not a universal forensic authenticity cue.

## 3. Cohort

- Complete-case N: {coverage['n']}
- Real: {coverage['class_counts'].get('likely_authentic', 0)}
- AI-generated: {coverage['class_counts'].get('ai_generated', 0)}
- Unique source groups: {coverage['unique_groups']}
- Excluded rows: {coverage['excluded_n']}
- Imputation: none.

## 4. Candidate models

| Model | Added inputs | Adjustment |
|---|---|---|
| M0 | None | SigLIP logit baseline |
| M1 | Noise + edge | Standardized linear features |
| M2 | Texture + resampling | Standardized linear features |
| M3 | All four | Standardized linear features |
| M4 | All four | Correlation clustering and signed alignment |

All models use L2 logistic regression. C is selected by 4-fold StratifiedGroupKFold within each outer training fold. Five outer folds are inherited unchanged from Experiment 2. Scaling and correlation transformation are fit within training folds only.

M4 uses absolute Spearman correlation, complete linkage, threshold {CORR_THRESHOLD:.2f}, and sign-aligns cluster members to a deterministic training-fold anchor before averaging.

## 5. Primary results

Primary endpoint: Δ log-loss = log-loss(M0) − log-loss(Mj); positive values favor the augmented model.

| Comparison | Δ Log-Loss | 95% Group Bootstrap CI | Raw p | BH q | Verdict |
|---|---:|---:|---:|---:|---|
{chr(10).join(rows)}

Uncertainty: {BOOTSTRAPS:,} paired source-group bootstrap replicates sampled within fixed outer folds.  
Hypothesis test: {SIGNFLIPS:,} two-sided cluster sign-flip permutations within fixed outer-fold strata.  
Multiplicity: Benjamini-Hochberg FDR across M1–M4, alpha={FDR_ALPHA:.2f}.

Supported improvement requires q < {FDR_ALPHA:.2f} and a 95% group-bootstrap interval entirely above zero. Supported detriment uses the symmetric condition. All other outcomes are exploratory-inconclusive.

## 6. Secondary results

paired_metrics.json includes pooled log-loss and Brier score, macro outer-fold ROC-AUC/PR-AUC/balanced accuracy, fold-level metrics, and confusion matrices. Pooled cross-fold ROC-AUC is descriptive only due possible fold-specific intercept/base-rate shifts.

## 7. Limitations

1. Post hoc exploratory follow-on, not an independent confirmatory replication.
2. All approved real samples are JPEG and all AI samples are PNG. Class and native format are perfectly confounded.
3. Source-group size imbalance limits effective independent generator information.
4. Any observed gain applies only to this partition and does not establish a universal forensic cue or external generalization.
5. Classical forensic indicators remain non-directional in production fusion regardless of these results.
6. The uploaded Experiment 2 closeout audit records commit b988e2f; the live GitHub branch has since advanced. This run pins inputs to the live branch's current Experiment 2 reproducibility manifest instead of assuming the older inventory is interchangeable.

## 8. Reproducibility

See experiment3_config.json, input_integrity.json, dataset_integrity.json, coverage.json, fold_summary.json, model_fit_details.json, oof_predictions.csv, paired_metrics.json, bootstrap_confidence_intervals.json, permutation_results.json, multiple_testing_results.json, environment_versions.json, and reproducibility_manifest.json.

Runtime: {environment['runtime_seconds']:.3f} seconds.
"""
    (OUT / "experiment3_report.md").write_text(report, encoding="utf-8")

    (OUT / "execution.log").write_text(
        f"Experiment 3 completed; exploratory post hoc.\n"
        f"Authorized rows: {N_EXPECTED}\nComplete cases: {coverage['n']}\n"
        f"Manifest SHA-256: {EXPECTED_MANIFEST_SHA}\nCheckpoint SHA-256: {EXPECTED_CHECKPOINT_SHA}\n"
        f"Runtime seconds: {environment['runtime_seconds']:.3f}\n",
        encoding="utf-8",
    )
    artifacts = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "reproducibility_manifest.json")
    repro = {
        "experiment": "TruthLens Experiment 3",
        "protocol_status": "exploratory_post_hoc",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": "f4baf15e67d2e837a17df7da0383cd0ede1a8b0a",
        "artifacts": {
            p.name: {"sha256": sha256_file(p), "byte_size": p.stat().st_size}
            for p in artifacts
        },
    }
    (OUT / "reproducibility_manifest.json").write_text(json.dumps(repro, indent=2), encoding="utf-8")
    log.info("Experiment 3 completed: n=%d, runtime=%.1fs", coverage["n"], environment["runtime_seconds"])


if __name__ == "__main__":
    run()
