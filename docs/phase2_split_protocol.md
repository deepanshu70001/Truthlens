# Phase 2 — Step 3: Group-Aware Dataset Splitting & Leakage Prevention Protocol

**Document Version:** `2.0.0`
**Protocol ID:** `truthlens-phase2-step3-group-aware-v1`
**Date:** 2026-11-21
**Repository Branch:** `fix/truthlens-remediation`
**Execution Scope:** Phase 2, Step 3 Only (Data Preparation & Evaluation Readiness)
**Configuration File:** [experiments/configs/phase2_split_protocol.yaml](..\experiments\configs\phase2_split_protocol.yaml)
**Audit Report:** [experiments/reports/phase2_split_audit.json](..\experiments\reports\phase2_split_audit.json) | [experiments/reports/phase2_split_audit.md](..\experiments\reports\phase2_split_audit.md)
**Execution Utility:** [experiments/create_evaluation_splits.py](..\experiments\create_evaluation_splits.py)

---

## 1. Primary Objective

The primary objective of this protocol is to establish a rigorous, mathematically defensible, and reproducible dataset partitioning strategy across the three externally ingested image benchmarks:
1. **Tiny-GenImage (v1.0-tiny)**
2. **WasItAIGenerated Frontier Image & Video Benchmark (October 2026)**
3. **WildAIGC-Bench (v3)**

This protocol prevents information leakage between:
* **Calibration Data:** Reserved exclusively for probability calibration (fitting temperature scaling parameters).
* **Validation Data:** Reserved for operating-policy exploration, threshold selection, and decision trade-off analysis.
* **Held-Out Test Data:** Strictly locked and reserved for final empirical benchmark reporting.

> [!IMPORTANT]
> A simple random image-level split is scientifically invalid because it silently leaks identical images, compressed social media variants, and shared-event reference photos across evaluation partitions. Group-level partitioning is strictly enforced.

---

## 2. Evaluation Role Definitions & Preservation of Source Splits

To prevent confusion between source dataset partitions and new TruthLens evaluation roles, source split semantics are preserved in `split_original`, while the assigned evaluation role is stored in `evaluation_role`.

| Evaluation Role | Definition & Purpose | Permitted Use in Phase 3 |
| :--- | :--- | :--- |
| **`calibration`** | Partition reserved exclusively for probability calibration (e.g. temperature scaling, Platt scaling, conformal prediction). | Fit calibration parameters $T$ and evaluate calibration fit diagnostics. Never for final accuracy reporting. |
| **`validation`** | Partition reserved for threshold selection, decision policy tuning, and sensitivity trade-offs. | Tune decision thresholds (e.g. high-precision vs. balanced). Never for final held-out test evaluation. |
| **`test`** | Completely held-out evaluation population. Never seen by the model encoder, calibration fitter, or threshold selector. | Final controlled empirical benchmark evaluation. |
| **`excluded`** | Administrative state for records with unresolved provenance, duplicate ambiguity, or reserve partitions. | Excluded from quantitative performance metrics. |

### Field Tracking Schema

Every record in the derived manifests records:

| Field | Description | Example Values |
| :--- | :--- | :--- |
| `split_original` | Original partition defined by the upstream dataset creators. | `train`, `val`, `test` |
| `evaluation_role` | Assigned TruthLens evaluation role. | `calibration`, `validation`, `test`, `excluded` |
| `protocol_id` | Versioned identifier of the splitting protocol. | `truthlens-phase2-step3-group-aware-v1` |
| `group_id` | Unique identifier of the leakage-control group. | `frontier_ie48760d71a8b`, `wildaigc_cc_0042` |
| `group_id_basis` | Explicit evidence supporting the grouping. | `metadata.csv:id_variant_family`, `connected_components` |
| `group_id_status` | Verification status of the group relationship. | `verified`, `provisional`, `review_required` |
| `split_seed` | Random seed used for deterministic reproducibility. | `42` |
| `split_reason` | Explanatory reason for assignment or exclusion. | Detailed documentation string |

---

## 3. Group Identification & Leakage Prevention Strategy

### A. Tiny-GenImage (1.0-tiny)
* **Dataset Scope:** 35,000 images (28,000 train, 7,000 val across 7 generators and ImageNet real controls).
* **Source Split Preservation:**
  * **Official `val` (7,000 images):** Retained 100% as the held-out `test` partition (`split_original = "val"`, `evaluation_role = "test"`). This preserves the exact canonical validation benchmark defined by Zhu et al. (NeurIPS 2023).
  * **Official `train` (28,000 images):** Because TruthLens models are frozen and will NOT be trained on this data, the official `train` partition is used to derive:
    * `calibration`: 12,048 images (balanced across all 7 generators and real ImageNet synsets).
    * `validation`: 5,958 images (balanced across all 7 generators and real ImageNet synsets).
    * `excluded` (reserve): 9,994 images (held in reserve to avoid calibration over-representation).
* **Duplicate Control:** The single duplicate image in Tiny-GenImage (`3eba56bc8ccd68cfe9b9f780829d081446e944156a114af3b08c874609f3631b`, shared between ADM and Midjourney nature folders) is mapped to a single unified group ID (`tiny_genimage_train_nature_shared_3eba56bc`) ensuring both copies reside in the exact same partition.

### B. WasItAIGenerated Frontier Image & Video Benchmark (October 2026)
* **Dataset Scope:** 2,976 images (747 original uncompressed + 2,229 social-platform-compressed variants).
* **Grouping Evidence:** Upstream metadata assigns an explicit `id` to each image capture. Exactly 743 IDs have 4 variants (`original`, `whatsapp`, `instagram`, `reshared`), and 4 IDs have 1 variant.
* **Leakage-Control Rule:** All variants sharing an `id` are bound into a single group (`frontier_<id>`).
* **Stratified Allocation:**
  * Groups are stratified by `(label, generator)` with seed 42:
    * **Calibration:** 588 images (147 groups, ~20%)
    * **Validation:** 297 images (75 groups, ~10%)
    * **Held-Out Test:** 2,091 images (525 groups, ~70%)
* **Zero-Leakage Guarantee:** An original image and its WhatsApp, Instagram, or reshared variants can **never** appear in different evaluation partitions.

### C. WildAIGC-Bench (v3)
* **Dataset Scope:** 1,839 images (1,238 raw web media files + 601 cropped/processed benchmark images).
* **Grouping Evidence:**
  1. `sample_id`: Both `raw/<sample_id>.*` and `processed/<sample_id>.*` represent the same event sample.
  2. `relations.jsonl`: Contains 44 confirmed reference relationships where an AI-edited image (`aigc_sample_id`) is paired with its verified real original photograph (`real_sample_id`).
* **Leakage-Control Rule:** A graph of all sample IDs was constructed with undirected edges for all confirmed reference relationships. Connected components analysis revealed **1,194 independent case groups**, binding 78 multi-case related samples into 34 unified groups.
* **Stratified Allocation:**
  * Groups are stratified by `label` with seed 42:
    * **Calibration:** 364 images (239 groups, ~20%)
    * **Validation:** 181 images (119 groups, ~10%)
    * **Held-Out Test:** 1,294 images (836 groups, ~70%)
* **Zero-Leakage Guarantee:** Related original and AI-edited versions of the same event can **never** cross between calibration, validation, and test.

---

## 4. Leakage Audit Verification

| Leakage Verification Item | Expected | Measured Result | Verdict |
| :--- | :---: | :---: | :---: |
| **Cross-Role Identical SHA-256 (Tiny-GenImage)** | 0 | **0** | **PASS** |
| **Cross-Role Group ID Collisions (Tiny-GenImage)** | 0 | **0** | **PASS** |
| **Cross-Role Identical SHA-256 (Frontier Benchmark)** | 0 | **0** | **PASS** |
| **Cross-Role Group ID Collisions (Frontier Benchmark)** | 0 | **0** | **PASS** |
| **Cross-Role Social Variant Separation (Frontier Benchmark)** | 0 | **0** | **PASS** |
| **Cross-Role Identical SHA-256 (WildAIGC-Bench)** | 0 | **0** | **PASS** |
| **Cross-Role Group ID Collisions (WildAIGC-Bench)** | 0 | **0** | **PASS** |
| **Cross-Role Related Case Separation (WildAIGC-Bench)** | 0 | **0** | **PASS** |
| **Cross-Dataset Exact SHA-256 Collisions** | 0 | **0** | **PASS** |
| **Overlap with TruthLens Internal Test Fixtures** | 0 | **0** | **PASS** |
| **Overlap with Detector CIFAKE Training Data** | 0 | **0** | **PASS** |

---

## 5. Partition Allocation & Class Balance Summary

```
========================================================================================
DATASET                          CALIBRATION    VALIDATION    HELD-OUT TEST    RESERVE
========================================================================================
Tiny-GenImage (1.0-tiny)            12,048         5,958          7,000          9,994
Frontier Benchmark (2026-10)           588           297          2,091              0
WildAIGC-Bench (v3)                    364           181          1,294              0
========================================================================================
TOTAL IMAGES                        13,000         6,436         10,385          9,994
========================================================================================
```

### Label Distribution Across Roles

* **Tiny-GenImage:**
  * Held-Out Test: 3,500 Real / 3,500 AI (1:1 balance)
  * Calibration: 6,024 Real / 6,024 AI (1:1 balance)
  * Validation: 2,979 Real / 2,979 AI (1:1 balance)
* **Frontier Benchmark 2026-10:**
  * Held-Out Test: 1,357 Real / 734 AI (64.9% Real, 35.1% AI)
  * Calibration: 382 Real / 206 AI (65.0% Real, 35.0% AI)
  * Validation: 193 Real / 104 AI (65.0% Real, 35.0% AI)
* **WildAIGC-Bench (v3):**
  * Held-Out Test: 447 Real / 847 AI (34.5% Real, 65.5% AI)
  * Calibration: 129 Real / 235 AI (35.4% Real, 64.6% AI)
  * Validation: 62 Real / 119 AI (34.3% Real, 65.7% AI)

---

## 6. Secondary Protocol: Generator-Held-Out Evaluation

To test whether detectors generalize to completely unseen generator families rather than familiar architectures, a secondary configuration is defined in [phase2_split_protocol.yaml](..\experiments\configs\phase2_split_protocol.yaml):

* **Tiny-GenImage Generator Isolation:**
  * Calibration / Validation Generators: `Stable Diffusion v1.5`, `BigGAN`, `ADM`, `GLIDE`.
  * Held-Out Test Generators: `Midjourney`, `Wukong`, `VQDM`.
* **Frontier Benchmark Generator Isolation:**
  * Calibration / Validation Generators: `Seedream v5 Pro`, `Flux.2 Pro`, `Ideogram v4.5`, `Krea 2 Turbo`, `Nano Banana Pro`, `Recraft v4.1`.
  * Held-Out Test Generators: `OpenAI GPT-Image-2`, `Black Forest Labs Flux.3`, `Grok Imagine 2`, `Google Nano Banana 2.1`, `Meta Muse Image`.

---

## 7. Derived Manifest Deliverables

The derived evaluation manifests are generated without altering the Step 2 ingestion manifests:

1. **Tiny-GenImage:** [`experiments/manifests/tiny_genimage_evaluation_splits.csv`](..\experiments\manifests\tiny_genimage_evaluation_splits.csv) (35,000 rows)
2. **Frontier Benchmark:** [`experiments/manifests/frontier_benchmark_2026_10_evaluation_splits.csv`](..\experiments\manifests\frontier_benchmark_2026_10_evaluation_splits.csv) (2,976 rows)
3. **WildAIGC-Bench:** [`experiments/manifests/wildaigc_bench_evaluation_splits.csv`](..\experiments\manifests\wildaigc_bench_evaluation_splits.csv) (1,839 rows)

---

## 8. Reproducibility & Commands

To re-run the deterministic splitting pipeline:

```powershell
# Dry-run validation (checks all constraints without modifying files)
python experiments/create_evaluation_splits.py --dry-run

# Apply partitioning and generate derived manifests
python experiments/create_evaluation_splits.py --apply --seed 42

# Execute unit and regression test suite
python -m pytest backend/tests/test_phase2_dataset_splits.py -q
```

---

## 9. Research Readiness Decision

**Status:** `READY_FOR_PHASE_3`

* All three benchmark datasets have validated, leakage-safe partitions.
* Calibration, validation, and held-out test splits have 0 cross-role hash, group, or variant leaks.
* No model was trained, fine-tuned, or evaluated.
* Phase 3 (Controlled Benchmark Evaluation & Calibration) may now proceed safely.
