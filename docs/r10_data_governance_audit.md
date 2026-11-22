# R10 Data Governance Audit

**Date:** 2026-11-22  
**Status:** Pre-fit archive boundary verified for the exact allowlisted file.  
**Read scope:** Named project documents and exact implementation paths only.

## Prior blocker and fresh boundary

`docs/phase10_blocker_report.md` records that an earlier broad `experiments/` text search did not preserve enough output to prove that protected test artifacts were not surfaced. It also records that no Phase 10 fitting, tuning, threshold selection, or validation evaluation ran. This R10 run establishes a fresh explicit allowlist and does not repeat a broad search.

`docs/phase4_training_dataset_report.md` identifies `experiments/manifests/phase4_training_manifest.csv` as the approved training manifest: 6,995 samples, 3,515 authentic and 3,480 AI-generated. It records zero overlap with calibration, validation, and locked test roles. The manifest itself will not be opened in R10 because the Phase 9 feature archive already contains the training vectors, labels, and group metadata.

`docs/phase2_split_protocol.md` defines validation as a development partition for operating-policy exploration and decision trade-offs. The Phase 9 feature-space and frequency reports identify the existing validation populations as Tiny-GenImage (5,958), Frontier 2026-10 (297), and WildAIGC-Bench (181), and state that the 13,431-row derived feature archive contains only approved training plus those validation rows. It states that calibration and locked test images were not opened for the Phase 9 archive.

## Exact R10 data allowlist

| Path | Allowed content/use | Prohibited operation |
|---|---|---|
| `experiments/reports/phase9_feature_vectors.npz` | Read only its split/count metadata and then the 6,995 training plus documented validation feature rows; fit only on training; evaluate only on validation | Do not load or use any unlisted partition. Stop if any unexpected role, count, or class distribution appears. Do not export sample paths or labels. |
| `models/baselines/siglip_frozen_head.pt` | Read the frozen linear head to reproduce the production baseline logit; the SHA-256 is checked before fitting | No modification, replacement, or production inference routing change. |
| `experiments/forensics/phase9_frequency_features.py` | Read the implementation and feature names | No new image reads or feature extraction in this R10 run. |

The allowed roles and counts are: approved training `n=6,995` (3,515 authentic / 3,480 AI), Tiny validation `n=5,958` (3,410 / 2,548), Frontier validation `n=297` (193 / 104), and WildAIGC validation `n=181` (63 / 118). WildAIGC is contextual-label diagnostic only and is excluded from the primary question and any candidate selection.

The Phase 2 combined split files are explicitly excluded from reads because they mix protected roles with validation:

- `experiments/manifests/tiny_genimage_evaluation_splits.csv`
- `experiments/manifests/frontier_benchmark_2026_10_evaluation_splits.csv`
- `experiments/manifests/wildaigc_bench_evaluation_splits.csv`

No calibration, held-out test, reserve, exclusion audit, protected predictions, or raw image paths are accessed. No broad recursive search is run over `experiments/`, `datasets/`, `storage/`, or model-cache trees.

## Operations permitted by this audit

1. Hash the exact Phase 9 feature archive and verify its metadata roles/counts before loading labels, embeddings, or frequency vectors.
2. If and only if the archive contains exactly the four documented roles/counts above, fit the preregistered 35-parameter frequency residual using approved training rows only.
3. Evaluate the frozen baseline and candidate on the named validation partitions after the candidate is fixed, using the metrics and bootstrap procedure in `docs/r10_research_protocol.md`.
4. Save only aggregate results, model coefficients, config, and checksums in the new `research/r10_e11/` directory. Do not write row-level predictions, identifiers, paths, images, or labels.

If the pre-fit checks fail, stop before loading feature matrices or fitting. Do not inspect a combined split manifest to repair the boundary. The project owner must provide a separately reviewed validation manifest or feature archive before a future attempt.

## Exact archive preflight completed

The exact allowlisted archive exists and its SHA-256 is `a1b9d3530f9555ba4b67cc4a5d2b3793ebac7804bb150eb0a2856da4efb00ed9` (56,274,384 bytes). Its ZIP member names/array headers and aggregate split/dataset/class counts were read at preflight. The five members are `metadata_json`, `siglip` `(13,431 × 768)`, `resnet50` `(13,431 × 2,048)`, `frequency` `(13,431 × 34)`, and `frequency_valid` `(13,431,)`.

The archive contains exactly `training=6,995` and `validation=6,436` rows, with no other split role. Dataset counts match the allowlist exactly: `tiny_genimage_training=6,995`, `tiny_genimage=5,958`, `frontier_benchmark_2026_10=297`, and `wildaigc_bench=181`. Aggregate class counts match the documented authentic/AI counts for each partition. The metadata was parsed to aggregate counts; individual identifiers were not selected, printed, or exported. No feature matrices, raw images, combined split manifests, calibration/test/reserve rows, or prediction files were opened.

This preflight confirms the previously documented train-plus-validation archive boundary. Feature values and the baseline checkpoint are still subject to the preregistered finite-value/dimension/hash checks in the runner before fitting.

## Existing historical evidence

Phase 9 reports contain historical training/validation summaries for SigLIP and forensic probes. Those results are background only; they are not R10 candidate results and do not substitute for the preregistered comparison. The prior Phase 10 draft is explicitly non-authorizing and is not treated as this R10 protocol. This document plus `docs/r10_research_protocol.md` establishes the run-specific boundary before R10 data access.
