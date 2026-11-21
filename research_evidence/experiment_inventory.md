# TruthLens Experiment Inventory

> **Document Status:** Verified Against Execution Logs & Scripts  
> **Inspection Date:** 2026-10-20  
> **Primary References:** `results/ml/`, `scripts/ml/train_baselines.py`, `scripts/ml/make_comparison.py`, `docs/EVALUATION.md`

---

## 1. Overview of Experimental Program

To support an empirical research paper, TruthLens defines a structured experimental agenda. This tracks:
1. **Completed Experiments:** Rigorous benchmark trials with recorded JSON outputs, predictions, and confusion matrices in `results/ml/`.
2. **Partially Implemented / Validated Trials:** Modules verified via automated unit and stage tests in `backend/tests/`.
3. **Proposed / Planned Paper Experiments:** Structured experiments designed for final manuscript inclusion once computational resources or external API quotas permit.

---

## 2. Master Experiment Table

| Experiment ID | Title | Objective | Status | Dataset / Split | Model / Component | Primary Metrics | Evidence / Result File |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | CNN from Scratch Baseline | Establish foundational non-pretrained performance on synthetic image detection | **COMPLETED** | CIFAKE Subset (800 train, 200 val, 200 test; Seed: 20261014) | 4-layer Custom ConvNet (356,097 params) | Acc: 89.50%, F1: 89.12%, ROC-AUC: 0.9585, Time: 500.0s | `results/ml/cnn_scratch.json`, `cnn_scratch_preds.csv` |
| **EXP-02** | ResNet-18 Transfer Learning Probe | Benchmark standard vision transfer learning with frozen backbone and linear probe | **COMPLETED** | CIFAKE Subset (800 train, 200 val, 200 test; Seed: 20261014) | ResNet-18 ImageNet-1k (513 trainable, 11.18M total) | Acc: 81.50%, F1: 81.77%, ROC-AUC: 0.9255, Time: 382.1s | `results/ml/resnet18_frozen.json`, `resnet18_frozen_preds.csv` |
| **EXP-03** | SigLIP Foundation Linear Probe | Evaluate vision-language foundation model zero/few-shot feature representations | **COMPLETED** | CIFAKE Subset (800 train, 200 val, 200 test; Seed: 20261014) | `google/siglip-base-patch16-224` (769 trainable, 92.88M total) | Acc: 87.00%, F1: 86.46%, ROC-AUC: 0.9613, Time: 73.0s | `results/ml/siglip_frozen.json`, `siglip_frozen_preds.csv` |
| **EXP-04** | Cross-Baseline Efficiency Comparison | Compare parameter efficiency, latency, and discriminatory capability across baselines | **COMPLETED** | CIFAKE Subset (Test split: 200 samples) | CNN vs ResNet-18 vs SigLIP | Parameter efficiency, Train duration, ROC-AUC | `results/ml/comparison.csv`, `results/ml/comparison.md` |
| **EXP-05** | Grad-CAM Interpretability Localization | Validate spatial localization of synthetic generator artifacts against human annotations | Proposed / Implemented | Curated deepfake test set (faces, objects, text) | SigLIP2 + Layer-targeted Grad-CAM | Pointing Game Accuracy, IoU against manipulation masks | `backend/app/services/xai/gradcam.py` |
| **EXP-06** | Forensic Compression Robustness | Measure performance degradation of ELA and noise residuals under aggressive JPEG re-compression | Proposed | Synthetic images resaved at Q=85, 75, 50, 30 | Classical Forensics Engine (`analyzer.py`) | ELA contrast decay, false positive spike rate | `backend/app/services/forensics/` |
| **EXP-07** | Multimodal Fusion Ablation | Quantify accuracy improvement of multi-criteria fusion over individual model predictions | Proposed | Multi-signal test suite with cross-domain contradictions | Evidence Fusion Engine (`engine.py`) | Overall F1 score, False Positive Reduction on recycled real imagery | `backend/app/services/fusion/` |
| **EXP-08** | Regional Spread Demographic Consistency | Stress-test demographic mapping and linguistic routing across 36 Indian States/UTs | Validated via Synthetic Suite | `data/synthetic/regional_spread_scenarios.csv` (500 cases) | Regional Spread Engine (`engine.py`) | State ranking stability, linguistic coverage percentage | `backend/tests/test_stage4.py` |

---

## 3. Detailed Review of Completed Experiments

### 3.1 EXP-01: CNN from Scratch
- **Protocol:** Trained with Adam optimizer ($\text{lr} = 0.001$, batch size 32) on $32 \times 32$ input resolution for 20 epochs.
- **Outcome:** Achieved a top validation accuracy of 90.50% at epoch 16, stabilizing at 89.50% test accuracy. Confusion matrix on 200 test samples:
  - True Negatives (Real detected as Real): 93
  - False Positives (Real misclassified as Fake): 7
  - False Negatives (Fake misclassified as Real): 14
  - True Positives (Fake detected as Fake): 86
- **Significance:** Demonstrates that high-frequency convolutional kernels easily detect local pixel co-occurrence artifacts inherent in Stable Diffusion v1.4, even without pretraining.

### 3.2 EXP-02: ResNet-18 Frozen Linear Probe
- **Protocol:** Input images upsampled to $224 \times 224 \times 3$. Backbone frozen; linear projection head trained for 10 epochs.
- **Outcome:** Best validation accuracy 85.50%; test accuracy 81.50%; ROC-AUC 0.9255.
- **Error Analysis:** High false positive count (20 real images misclassified as fake) and 17 false negatives. Standard ImageNet-1k representations emphasize semantic object categories rather than subtle pixel generation artifacts.

### 3.3 EXP-03: SigLIP Foundation Probe
- **Protocol:** `google/siglip-base-patch16-224` vision backbone with cached 768-dimensional embeddings. Linear head trained for 50 epochs.
- **Outcome:** Test accuracy 87.00%; Precision 90.22%; ROC-AUC **0.9613** (highest among all baselines). Training completed in only 73.0 seconds due to precomputed feature embeddings.
- **Significance:** Achieved superior discrimination (ROC-AUC 0.9613) with only 769 trainable parameters, confirming that vision-language foundation backbones retain fine-grained distributional cues necessary for synthetic media detection.
