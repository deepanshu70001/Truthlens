# TruthLens Experimental Results & Comparative Evaluation

> **Document Status:** Comprehensive Empirical Results across Phases 0–7
> **Hardware:** Intel Core i5-12500H (12 cores, 16 threads), 16 GB RAM, NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)
> **Source Artifacts:** `experiments/reports/phase7_locked_test_results.json`, `experiments/reports/phase7_candidate_decision.json`, `experiments/reports/phase7_temperature_calibration.json`, `experiments/reports/phase7_training_results.json`

---

## 1. Master Benchmark Evaluation on Locked Test Partitions ($N = 10,385$)

All evaluations executed on held-out, leak-free test partitions isolated via group-aware stratification:

| Metric / Evaluation Angle | Production Baseline (`siglip-cifake-frozen-head`) | Candidate E3 (Unaugmented FT, Phase 5) | Candidate S2 (`siglip-base-v2-s2`, Phase 7) | $\Delta$ (S2 vs. Baseline) | 95% Group-Aware Bootstrap CI (S2 vs. Baseline) |
|---|---|---|---|---|---|
| **Tiny-GenImage Test ($N=7,000$)** | | | | | |
| • Balanced Accuracy | 0.5521 | 0.8840 | **0.8460** | +0.2939 | $[+0.2798, +0.3082]$ |
| • ROC-AUC | 0.6123 | 0.9610 | **0.9423** | +0.3300 | $[+0.3184, +0.3412]$ |
| • Real Accuracy ($N=3,500$) | 0.1609 | 0.8710 | **0.7849** | +0.6240 | — |
| • AI Accuracy ($N=3,500$) | 0.9434 | 0.8970 | **0.9071** | -0.0363 | — |
| **Tiny-GenImage Matched Aug ($N=7,000$)** | | | | | |
| • Balanced Accuracy | 0.5330 | 0.5120 | **0.8071** | +0.2741 | $[+0.2584, +0.2897]$ |
| • ROC-AUC | 0.5595 | 0.5210 | **0.9044** | +0.3449 | $[+0.3312, +0.3582]$ |
| **Frontier Benchmark 2026-10 ($N=2,091$)** | | | | | |
| • Balanced Accuracy | **0.6228** | 0.5012 | 0.4849 | -0.1379 | $[-0.1712, -0.1044]$ |
| • ROC-AUC | **0.6501** | 0.4410 | 0.4234 | -0.2267 | $[-0.2781, -0.1720]$ |
| • Real Accuracy ($N=730$) | 0.4000 | 0.1540 | **0.6603** | +0.2603 | — |
| • AI Accuracy ($N=1,361$) | 0.8457 | 0.8480 | **0.3093** | -0.5364 | — |
| **WildAIGC-Bench Test ($N=1,294$)** | | | | | |
| • Balanced Accuracy | 0.4294 | 0.4890 | **0.5589** | +0.1295 | $[+0.0984, +0.1610]$ |
| • ROC-AUC | 0.4226 | 0.4950 | **0.6049** | +0.1823 | $[+0.1472, +0.2178]$ |
| **In-Memory JPEG-90 Transcoding Probe** | | | | | |
| • Prediction Flip Rate | 22.4% | 45.6% (Catastrophic shortcut) | **11.0%** (Stable) | -11.4% | (vs. E3: -34.6%) |
| • Mean Logit Absolute Shift | 1.45 | 6.12 (Extreme shift) | **0.793** (Robust) | -0.657 | (vs. E3: -5.327) |

---

## 2. Phase 7 Architecture Comparison

Five models were trained on the approved 6,995-sample Tiny-GenImage training manifest under a 4 GB VRAM envelope:

| Model ID | Architecture / Training Strategy | Trainable Params | Peak VRAM | Epochs (Best) | Tiny Val Acc | Frontier Val AUC | Validation Score | Status |
|---|---|---|---|---|---|---|---|---|
| **S1** (`truthlens/siglip-base-v2-s1`) | SigLIP Linear Probe ($K=5$ views, 34,975 samples) | 769 (0.0009%) | 700.5 MB | 15 (12) | 0.8124 | **0.5991** | 0.6678 | Candidate |
| **S2** (`truthlens/siglip-base-v2-s2`) | SigLIP Layers 10, 11 + Head Partial FT | 14,178,817 (16.4%) | 839.1 MB | 7 (4) | **0.8675** | 0.4769 | **0.6722** | Primary Candidate |
| **R1** (`truthlens/resnet50-v2-r1`) | ResNet-50 Full Fine-Tuning | 23,510,081 (100%) | 1,121.5 MB | 7 (3) | 0.8245 | 0.4803 | 0.6524 | Candidate |
| **R2** (`truthlens/resnet50-v2-r2`) | ResNet-50 Stage 4 Fine-Tuning | 14,976,001 (63.7%) | 606.9 MB | 8 (5) | 0.8012 | 0.4744 | 0.6378 | Candidate |
| **A1** (`truthlens/siglip-base-v2-a1`) | SigLIP Ablation (No Augmentation) | 14,178,817 (16.4%) | 839.1 MB | 6 (2) | 0.9412 | 0.5676 | 0.7544 | Shortcut Ablation |

---

## 3. Pre-Training Container Format Audit

To demonstrate container-format confounding, Random Forest models were trained exclusively on image metadata features (file extension, MIME type, file size, byte entropy, aspect ratio, width, height):

| Experiment | Dataset Partition | Metadata Features Only (ROC-AUC) | Interpretation |
|---|---|---|---|
| **Raw Dataset (Unaugmented)** | Tiny-GenImage Train ($N=6,995$) | **1.0000** | Perfect separation: 100% real images are JPEG; 100% AI images are PNG. |
| **Raw Dataset (Unaugmented)** | Tiny-GenImage Val ($N=3,464$) | **1.0000** | Confirms container format artifact generalises across all raw splits. |
| **Matched In-Memory Augmentation** | Tiny-GenImage Augmented Train ($N=6,995$) | **0.5028** | Metadata separability collapsed to random chance ($p \approx 0.50$). |
| **Matched In-Memory Augmentation** | Tiny-GenImage Augmented Val ($N=3,464$) | **0.4994** | Proves matched transcoding completely eliminates container shortcut. |

---

## 4. Post-Hoc Temperature Calibration Results

Temperature scaling ($T$) fitted on the 13,000-sample calibration pool using Negative Log-Likelihood (NLL) optimization:

| Calibration Split | $N$ | Optimized Temperature ($T$) | Uncalibrated NLL | Calibrated NLL | Uncalibrated ECE | Calibrated ECE | Optimization Status |
|---|---|---|---|---|---|---|---|
| **Tiny-GenImage Calib** | 10,000 | **1.2174** | 0.412 | **0.398** | 0.084 | **0.031** | Converged |
| **Frontier Benchmark Calib** | 2,000 | **10.3621** | 1.482 | **0.781** | 0.312 | **0.065** | Converged |
| **WildAIGC-Bench Calib** | 1,000 | **5.6099** | 0.941 | **0.652** | 0.198 | **0.048** | Converged |
| **Pooled Calibration** | 13,000 | **1.3187** | 0.521 | **0.485** | 0.112 | **0.038** | Converged |

---

## 5. Candidate Governance & Production Retention Decision

- **Pre-Registered Criteria:** A candidate model may replace the production baseline only if it achieves non-inferiority or statistically significant improvement across all held-out benchmarks without negative transfer ($\Delta \text{AUC} \ge 0$).
- **Empirical Outcome:** Candidate S2 achieved impressive in-distribution gains (+0.3300 AUC on Tiny-GenImage) and robust compression stability, but suffered severe negative transfer on modern 2026 generators (-0.2267 AUC on Frontier Benchmark).
- **Formal Decision:**
  ```json
  {
    "candidate_model_id": "truthlens/siglip-base-v2-s2",
    "production_model_id": "truthlens/siglip-cifake-frozen-head",
    "decision": "REJECT_CANDIDATE_FOR_PRODUCTION",
    "retention": "RETAIN_FROZEN_BASELINE_IN_PRODUCTION",
    "governance_rationale": "Candidate S2 failed cross-generator generalization on modern 2026 generators (Frontier AUC 0.4234 vs Baseline 0.6501, Delta AUC -0.2267). Frozen baseline retained in production."
  }
  ```
