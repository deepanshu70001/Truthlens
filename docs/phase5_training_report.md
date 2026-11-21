# TruthLens Phase 5: Controlled Model Training, Fine-Tuning & Validation-Only Model Selection Report

**Repository:** `https://github.com/deepanshu70001/Truthlens`  
**Branch:** `fix/truthlens-remediation`  
**Milestone:** Phase 5 — Controlled Model Training & Fine-Tuning  
**Date:** November 2026  
**Hardware Environment:** NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MB VRAM)  
**Evaluated Baseline:** `truthlens/siglip-cifake-frozen-head`  
**Promoted Candidate:** `truthlens/siglip-base-tiny-genimage-e3_partial_finetuning`  

---

## 1. Executive Summary

Phase 5 executed controlled training experiments to determine whether fine-tuning the `google/siglip-base-patch16-224` vision encoder and training new binary classification heads on the approved Tiny-GenImage training pool improves generalization and discrimination over the existing frozen CIFAKE baseline (`truthlens/siglip-cifake-frozen-head`).

Three controlled experiments were conducted under identical seed (`42`), optimizer (`AdamW`), scheduler (cosine with linear warmup), batch size (`16` with gradient accumulation `2`, effective batch size `32`), and mixed precision (`FP16`):
1. **Experiment E1 (Frozen Backbone):** New linear classification head (`768 -> 1`), vision backbone completely frozen (`769` trainable parameters, `0.0008%`), standard BCE loss.
2. **Experiment E2 (Class-Balanced Training):** Same frozen architecture, trained with class-weighted BCE loss ($w_{\text{pos}} = 1.010057$) compensating for authentic/synthetic sample distribution.
3. **Experiment E3 (Partial Fine-Tuning):** SigLIP vision backbone with only the 12th transformer encoder layer (`vision_model.encoder.layers[-1]`) unfrozen alongside the linear head (`7,088,641` trainable parameters, `7.6316%`), trained with differential learning rates ($1 \times 10^{-5}$ backbone, $1 \times 10^{-4}$ head).

### Key Results Summary
- **Baseline CIFAKE Head:** On the Tiny-GenImage validation split ($N=5,958$), the baseline scored **0.5542 Balanced Accuracy** and **0.6020 ROC-AUC**, heavily overpredicting AI on natural photos.
- **E1 (Frozen Head):** Reached **0.7644 Balanced Accuracy** (+21.02 percentage points) and **0.8656 ROC-AUC** (+26.36 percentage points), demonstrating that training the linear head directly on domain-specific feature embeddings provides immediate benefits.
- **E2 (Class-Balanced):** Reached **0.7655 Balanced Accuracy** (+21.13 percentage points) and **0.8656 ROC-AUC**, showing modest alignment gains over E1.
- **E3 (Partial Fine-Tuning):** Reached **0.8667 Balanced Accuracy** (+31.25 percentage points over baseline, +10.12 percentage points over E2) and **0.9577 ROC-AUC** (+35.57 percentage points over baseline), cutting validation loss from $0.4824$ down to $0.3990$.
- **Strict Protocol Compliance:** **The locked test set was NEVER evaluated during Phase 5.** Model selection was conducted exclusively on the Tiny-GenImage validation split ($N=5,958$).
- **Formal Decision:** `e3_partial_finetuning` is formally selected and designated as `PROMOTE_AS_CANDIDATE` for held-out evaluation in Phase 6.

---

## 2. Hardware & Resource Usage

All experiments were executed locally on dedicated laptop hardware with mixed-precision autocast (`torch.amp.autocast("cuda", dtype=torch.float16)`). Peak VRAM usage was monitored throughout execution:

| Experiment | Dedicated VRAM Limit | Peak VRAM Utilized | VRAM Headroom | Training Time | Epochs Trained |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline Evaluation** | 4,096 MB | 445.1 MB | 3,650.9 MB (89.1%) | N/A (Inference) | N/A |
| **E1 (Frozen Head)** | 4,096 MB | 538.0 MB | 3,558.0 MB (86.9%) | 1,151.0 s (~19.2 min) | 10 (Full) |
| **E2 (Class-Balanced)** | 4,096 MB | 538.0 MB | 3,558.0 MB (86.9%) | 1,290.7 s (~21.5 min) | 10 (Full) |
| **E3 (Partial Fine-Tuning)** | 4,096 MB | 622.7 MB | 3,473.3 MB (84.8%) | 975.8 s (~16.3 min) | 8 (Early Stopped at Ep 5) |

Hardware stability was 100%: zero CUDA out-of-memory errors occurred, and GPU memory consumption remained well below 700 MB across all training runs.

---

## 3. Dataset Governance, Splitting & Leakage Verification

### 3.1 Training Dataset Composition
Training strictly adhered to the approved Phase 4 manifest `experiments/manifests/phase4_training_manifest.csv`:
- **Total Approved Samples:** 6,995
- **Likely Authentic (Real):** 3,515 nature photographs from ImageNet-Camera (`ILSVRC2012`).
- **AI-Generated (Synthetic):** 3,480 images across 4 approved generators:
  - `ADM` (Pixel diffusion): 502
  - `BigGAN` (Adversarial GAN): 482
  - `SD 1.5` (Latent diffusion): 496
  - `GLIDE` (Cascaded diffusion): 2,000

### 3.2 Total Exclusion of Held-Out Protocol Sets
- **Held-Out Generators:** All 2,998 reserve samples from `Midjourney` (1,000), `Wukong` (999), and `VQDM` (999) were completely excluded from training under `EXCLUDED_PROTOCOL`.
- **WildAIGC-Bench:** 0 samples utilized in training (classified `NOT APPROVED FOR SUPERVISED TRAINING` in Phase 3.5).
- **Test Partitions:** 0 samples from test splits were included in training.
- **Cross-Split Hash Overlap:** Automated verification confirmed **0 cross-partition hash collisions** between the training manifest and any calibration, validation, or test sets.

---

## 4. Domain Confounding Analysis

A critical scientific finding from the Phase 4 audit and Phase 5 training data inspection is the presence of domain confounding in the Tiny-GenImage dataset:

1. **File Format & Compression Asymmetry:**
   - Real images: **100% JPEG** (ImageNet nature photographs, variable rectangular aspect ratios).
   - AI images: **100% PNG** (ADM, BigGAN, SD 1.5, GLIDE, fixed square dimensions $256 \times 256$, $512 \times 512$, $128 \times 128$).
2. **Semantic / WordNet Overlap:**
   - Class prefix overlap between Real and AI subsets is exactly **0** (`ADM_879`, `BigGAN_239`, etc. have no class-paired authentic counterparts).
3. **Scientific Implication & Risk:**
   - Standard vision models trained on such datasets risk learning low-level file format artifacts (e.g. JPEG blocking artifacts vs PNG lossless sharpness) rather than genuine generative synthesis patterns.
   - In E1 and E2, the linear head operates on frozen features which mitigates direct feature corruption, but the head can still latch onto representations correlated with compression.
   - In E3, fine-tuning the 12th transformer encoder layer allows the network to refine representations, achieving superior in-distribution validation performance ($0.8667$ balanced accuracy). However, verifying whether this improvement reflects true synthesis detection or format-bias memorization requires cross-dataset generalization testing on the locked test sets (Phase 6).

---

## 5. Controlled Experiment Performance Comparison

Model selection was conducted strictly against the Tiny-GenImage validation split ($N=5,958$: 3,410 Authentic, 2,548 AI-Generated).

### Side-by-Side Validation Performance Table

| Model / Experiment | Mode | Trainable Params | Best Epoch | Val Loss | Accuracy | Balanced Accuracy | Precision | Recall | Specificity | F1 Score | ROC-AUC | PR-AUC | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (CIFAKE Head)** | Frozen Baseline | 769 (0.0008%) | N/A | N/A | 0.5213 | 0.5542 | 0.4646 | 0.7818 | 0.3267 | 0.5828 | 0.6020 | 0.5539 | 0.1168 |
| **E1 (Frozen Head)** | Frozen Backbone | 769 (0.0008%) | 10 | 0.4826 | 0.7837 | 0.7644 | 0.8217 | 0.6311 | 0.8977 | 0.7139 | 0.8656 | 0.8333 | 0.5570 |
| **E2 (Class-Balanced)** | Weighted BCE | 769 (0.0008%) | 9 | 0.4824 | 0.7845 | 0.7655 | 0.8208 | 0.6346 | 0.8965 | 0.7158 | 0.8656 | 0.8333 | 0.5585 |
| **E3 (Partial Fine-Tuning)** | Unfreeze Layer 11 | 7,088,641 (7.63%) | 5 | **0.3990** | **0.8823** | **0.8667** | **0.9574** | **0.7586** | **0.9748** | **0.8465** | **0.9577** | **0.9551** | **0.7666** |

### Confusion Matrix Breakdown (Validation Split, N = 5,958)

- **Baseline (CIFAKE Head):**
  - True Negatives (Authentic Correct): 1,114 / 3,410 (32.7%)
  - False Positives (Authentic False Alarm): 2,296 / 3,410 (67.3%)
  - False Negatives (AI Missed): 556 / 2,548 (21.8%)
  - True Positives (AI Detected): 1,992 / 2,548 (78.2%)
- **E1 (Frozen Head, Best Epoch 10):**
  - True Negatives: 3,061 / 3,410 (89.8%)
  - False Positives: 349 / 3,410 (10.2%)
  - False Negatives: 940 / 2,548 (36.9%)
  - True Positives: 1,608 / 2,548 (63.1%)
- **E2 (Class-Balanced, Best Epoch 9):**
  - True Negatives: 3,057 / 3,410 (89.6%)
  - False Positives: 353 / 3,410 (10.4%)
  - False Negatives: 931 / 2,548 (36.5%)
  - True Positives: 1,617 / 2,548 (63.5%)
- **E3 (Partial Fine-Tuning, Best Epoch 5):**
  - True Negatives: **3,324 / 3,410 (97.5%)**
  - False Positives: **86 / 3,410 (2.5%)**
  - False Negatives: 615 / 2,548 (24.1%)
  - True Positives: **1,933 / 2,548 (75.9%)**

---

## 6. Validation-Only Model Selection Decision

Under TruthLens evaluation governance, model selection must be based strictly on validation performance prior to any test-set evaluation:

1. **Comparison against Baseline:**
   - `e3_partial_finetuning` achieved **0.8667 Balanced Accuracy**, outperforming the baseline ($0.5542$) by **+31.25 percentage points**.
   - `e3_partial_finetuning` achieved **0.9577 ROC-AUC**, outperforming the baseline ($0.6020$) by **+35.57 percentage points**.
   - `e3_partial_finetuning` drastically suppressed false alarms on real nature images, increasing specificity from $32.67\%$ (baseline) and $89.77\%$ (E1) up to **$97.48\%$**.
2. **Comparison among Phase 5 Experiments:**
   - E1 and E2 performed similarly ($0.7644$ vs $0.7655$ balanced accuracy), showing that class reweighting offers only slight adjustments when the class ratio is nearly balanced ($3,515 : 3,480$).
   - E3 substantially outperformed E1 and E2 across every single evaluation metric (+10.12 percentage points in balanced accuracy, +9.21 percentage points in ROC-AUC, +13.07 percentage points in F1).
3. **Formal Governance Decision:**
   - **Status:** `PROMOTE_AS_CANDIDATE`
   - **Selected Model ID:** `truthlens/siglip-base-tiny-genimage-e3_partial_finetuning`
   - **Eligibility for Phase 6 Test Evaluation:** `ELIGIBLE_FOR_PHASE6_HELD_OUT_EVALUATION`
   - **Production Readiness:** *Not yet eligible for production deployment.* Deployment requires Phase 6 verification against the locked test split (including held-out generators Midjourney, Wukong, VQDM and Frontier benchmark).

---

## 7. Artifact & Cryptographic Registry Ledger

All generated checkpoints, standalone head weights, and experiment reports are registered with their exact cryptographic SHA-256 digests:

| Model ID | Artifact File | File Path | SHA-256 Digest |
| :--- | :--- | :--- | :--- |
| `truthlens/siglip-cifake-frozen-head` | Baseline Head | `models/baselines/siglip_frozen_head.pt` | `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538` |
| `truthlens/siglip-base-tiny-genimage-e1_frozen_head` | Full Checkpoint | `experiments/checkpoints/phase5/e1_frozen_head/best_checkpoint.pt` | `b1a415575d53e9167bbbf0c49924c4e9893d7b293d519088e2afd5858e333a74` |
| `truthlens/siglip-base-tiny-genimage-e1_frozen_head` | Head Weights | `experiments/checkpoints/phase5/e1_frozen_head/head.pt` | `c318a3b9efbdbe0da08fe4d660f24e76149c2d2afa199f0b08c58326ab774aff` |
| `truthlens/siglip-base-tiny-genimage-e2_balanced_training` | Full Checkpoint | `experiments/checkpoints/phase5/e2_balanced_training/best_checkpoint.pt` | `4436cfab95e9afe79e3832a15547ab3d66fe4b7e5fb9f892d29815ce9b858344` |
| `truthlens/siglip-base-tiny-genimage-e2_balanced_training` | Head Weights | `experiments/checkpoints/phase5/e2_balanced_training/head.pt` | `29ca24e7cb9d68a9ce790cb77d7084207e0e60474dd239a76beb3a2f2ff878d9` |
| `truthlens/siglip-base-tiny-genimage-e3_partial_finetuning` | Full Checkpoint | `experiments/checkpoints/phase5/e3_partial_finetuning/best_checkpoint.pt` | `541956a1f766d5093aaec4733f616afaf6deaf5aad1e0bbd6e59f95687a4b96d` |
| `truthlens/siglip-base-tiny-genimage-e3_partial_finetuning` | Head Weights | `experiments/checkpoints/phase5/e3_partial_finetuning/head.pt` | `38b12fb1931158a771cb6b64bd5745ef7f7796b27df6b9eb292c281566689b68` |

### Serialization Manifests & Reports
- `experiments/reports/phase5_experiment_comparison.csv`
- `experiments/reports/phase5_training_results.json`
- `experiments/reports/phase5_training_curves.json`
- `experiments/reports/phase5_model_registry.json`
- `experiments/configs/phase5_experiment_config.yaml`

### Visualization Figures
- `experiments/reports/figures/phase5_loss_curves.png`: Training vs Validation BCE loss across epochs.
- `experiments/reports/figures/phase5_val_balanced_acc_curves.png`: Validation balanced accuracy progression vs baseline.
- `experiments/reports/figures/phase5_val_roc_curves.png`: Validation ROC-AUC progression vs baseline.
- `experiments/reports/figures/phase5_val_pr_curves.png`: Validation PR-AUC curves across epochs.
- `experiments/reports/figures/phase5_experiment_comparison.png`: Comprehensive bar chart comparing Baseline, E1, E2, E3 across 4 core metrics.
- `experiments/reports/figures/phase5_vram_usage.png`: Peak VRAM utilization across experiments on RTX 3050 Laptop GPU.

---

## 8. Explicit Answers to the 13 Stop Condition Questions

1. **Were all three experiments (E1, E2, E3) successfully executed to completion?**  
   **Yes.** E1 trained for 10 epochs (1,151.0 s), E2 trained for 10 epochs (1,290.7 s), and E3 trained for 8 epochs before triggering early stopping at its peak validation checkpoint at Epoch 5 (975.8 s).
2. **What was the exact hardware configuration and peak VRAM utilized per experiment?**  
   NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MB dedicated VRAM, 8,192 MB shared). Peak VRAM: E1 = 538.0 MB, E2 = 538.0 MB, E3 = 622.7 MB. All runs operated well within the 4 GB VRAM ceiling.
3. **Was the locked test set accessed or evaluated during Phase 5?**  
   **No. Zero test-set evaluation was performed.** The test split remains completely locked, unread, and untouched throughout Phase 5.
4. **Were held-out generator models (Midjourney, Wukong, VQDM) strictly excluded from training?**  
   **Yes.** All 2,998 reserve samples from Midjourney, Wukong, and VQDM were audited and excluded under `EXCLUDED_PROTOCOL`. Zero samples entered the training pipeline.
5. **Was WildAIGC-Bench included in training?**  
   **No.** WildAIGC-Bench was completely excluded, in compliance with the Phase 3.5 audit finding classifying it as `NOT APPROVED FOR AI-GENERATION SUPERVISED TRAINING`.
6. **What were the best validation metrics for each experiment?**  
   - E1 (Frozen Head): Bal Acc = 0.7644, Acc = 0.7837, AUC = 0.8656, F1 = 0.7139, Loss = 0.4826 (Epoch 10).
   - E2 (Class-Balanced): Bal Acc = 0.7655, Acc = 0.7845, AUC = 0.8656, F1 = 0.7158, Loss = 0.4824 (Epoch 9).
   - E3 (Partial Fine-Tuning): Bal Acc = 0.8667, Acc = 0.8823, AUC = 0.9577, F1 = 0.8465, Loss = 0.3990 (Epoch 5).
7. **Did any experiment outperform the existing frozen baseline?**  
   **Yes, all three experiments substantially outperformed the frozen baseline.** E1 exceeded baseline balanced accuracy by +21.02 percentage points, E2 by +21.13 percentage points, and E3 by +31.25 percentage points.
8. **Which model was selected by the validation-only model selection rule?**  
   **`e3_partial_finetuning`** (SigLIP base with layer 11 unfrozen + linear head) was selected, achieving the highest balanced accuracy (0.8667) and highest ROC-AUC (0.9577).
9. **What is the formal model selection decision and status?**  
   `PROMOTE_AS_CANDIDATE` for held-out evaluation in Phase 6. It is not approved for production deployment until held-out generalization and calibration are proven in Phase 6.
10. **What are the SHA-256 hashes of the best checkpoints and head weights?**  
    - E1 Best Checkpoint: `b1a415575d53e9167bbbf0c49924c4e9893d7b293d519088e2afd5858e333a74` (Head: `c318a3b9efbdbe0da08fe4d660f24e76149c2d2afa199f0b08c58326ab774aff`)  
    - E2 Best Checkpoint: `4436cfab95e9afe79e3832a15547ab3d66fe4b7e5fb9f892d29815ce9b858344` (Head: `29ca24e7cb9d68a9ce790cb77d7084207e0e60474dd239a76beb3a2f2ff878d9`)  
    - E3 Best Checkpoint: `541956a1f766d5093aaec4733f616afaf6deaf5aad1e0bbd6e59f95687a4b96d` (Head: `38b12fb1931158a771cb6b64bd5745ef7f7796b27df6b9eb292c281566689b68`)  
11. **How does domain confounding impact the interpretation of these results?**  
    100% of real images are JPEG natural photographs from ImageNet, while 100% of AI images are PNG synthetic images with 0 class prefix overlap. While E3 shows impressive in-distribution validation metrics, cross-dataset evaluation on heterogeneous real/synthetic test sets in Phase 6 is required to confirm whether the learned representations generalize beyond compression artifacts.
12. **Were production files (`backend/app/services/authenticity/detector.py`, `calibration.py`) preserved untouched?**  
    **Yes.** Zero modifications were made to production detector or calibration files.
13. **Did all unit tests and regression tests pass?**  
    **Yes.** 71/71 tests passed across the entire TruthLens test suite (`test_phase5_training.py`, `test_phase4_and_phase3_5.py`, `test_phase3_evaluation.py`, `test_phase2_dataset_splits.py`, `test_phase1.py`, `test_authenticity_calibration.py`).
