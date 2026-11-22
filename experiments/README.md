# TruthLens Experiments & Evaluation Infrastructure

This directory contains the reproducible experimental pipelines, dataset split manifests, training scripts, post-hoc calibration utilities, and evaluation reports across TruthLens Phases 0 through 9.

---

## 1. Directory Structure

```
experiments/
├── checkpoints/              # Checkpoint weights for research candidates (Phases 5 & 7)
│   ├── phase5/               # E1 (frozen head), E2 (balanced), E3 (layer 11 FT)
│   └── phase7/               # S1, S2, R1, R2, A1
├── configs/                  # Experiment configurations (phase7_config.yaml)
├── manifests/                # Approved training, validation, calibration, and test manifests
├── reports/                  # Machine-readable evaluation reports, calibration parameters, and audit summaries
├── results/                  # Per-sample prediction CSVs (Phases 3, 6, 7)
├── train_phase7.py           # Matched in-memory augmentation trainer across 5 architectures
├── evaluate_phase7_test.py   # Locked test evaluation across all three external benchmarks
├── calibrate_phase7_candidate.py # Post-hoc temperature calibration parameter optimizer
├── audit_phase7_shortcuts.py # Container format shortcut verification script
└── README.md                 # This guide
```

---

## 2. Ingested External Benchmarks & Partitions ($N = 29,640$)

The experimental suite operates across three cryptographically audited external image benchmarks:
1. **Tiny-GenImage ($N = 25,006$):** 8 generators (BigGAN, Cascade Diffusion, DeepFloyd-IF, GLIDE, LDM, Midjourney v4, SD 1.4, SD 1.5).
2. **Frontier Benchmark 2026-10 ($N = 2,976$):** 8 modern generators (DALL-E 3, Flux.1-dev, Flux.1-schnell, Midjourney v6, Recraft v3, SD 3 Medium, SD 3.5 Large, SD 3.5 Turbo) across 4 compression tiers (Original, Instagram, Reshared, WhatsApp).
3. **WildAIGC-Bench ($N = 1,658$):** Real-world in-the-wild digital media captures.

### Split Governance Partitions
- **Approved Training:** 6,995 samples (`experiments/manifests/phase4_training_manifest.csv`)
- **Validation:** 6,255 samples
- **Calibration Pool:** 13,000 samples
- **Locked Test Set:** 10,385 samples (Tiny: 7,000; Frontier: 2,091; WildAIGC: 1,294)

---

## 3. Key Experimental Commands

From the repository root with the project environment active:

### 3.1 Pre-Training Container Format Audit
```powershell
python experiments/audit_phase7_shortcuts.py
# Outputs: experiments/reports/phase7_format_audit.json
# Verifies: raw metadata RF AUC = 1.0000; augmented metadata RF AUC = 0.5028
```

### 3.2 Matched In-Memory Augmentation Training
```powershell
python experiments/train_phase7.py --config experiments/configs/phase7_config.yaml
# Produces candidate checkpoints under experiments/checkpoints/phase7/
```

### 3.3 Post-Hoc Temperature Scaling Calibration
```powershell
python experiments/calibrate_phase7_candidate.py
# Outputs: experiments/reports/phase7_calibration.json
```

### 3.4 Locked Test Benchmark Evaluation
```powershell
python experiments/evaluate_phase7_test.py
# Outputs: experiments/reports/phase7_test_evaluation.json, phase7_statistical_comparison.json
# Computes: group-aware bootstrap 95% confidence intervals (B = 1,000)
```

---

## 4. Evaluation Governance & Model Retention Policy

TruthLens enforces strict pre-registered governance:
- Candidate models may replace the production baseline only if they achieve statistically significant improvement or non-inferiority across all held-out benchmarks without negative transfer.
- In Phase 7 evaluations, while Candidate S2 (`truthlens/siglip-base-v2-s2`) achieved a 0.9423 ROC-AUC on Tiny-GenImage (+0.3300 gain), it suffered negative transfer on late-2026 generators on the Frontier Benchmark (ROC-AUC 0.4234 vs. baseline 0.6501, $\Delta = -0.2267$, 95% CI: $[-0.2781, -0.1720]$).
- In accordance with pre-registered safety rules, Candidate S2 was rejected for production, and the production detector retains the frozen baseline (`truthlens/siglip-cifake-frozen-head`, SHA-256: `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`).

## 5. Phase 9: Cross-Generator and Forensic Feature Diagnostics

Run from the repository root after installing the backend research dependencies and ensuring the approved local datasets and pinned pretrained weights are already available:

```powershell
python experiments/analyze_phase9.py
python experiments/analyze_phase9_candidate_validation.py
python experiments/analyze_phase9_baseline_validation.py
```

The first runner verifies baseline and Phase 7 checkpoint hashes, then uses only the approved 6,995-sample training role and validation roles (Tiny-GenImage 5,958; Frontier 297; WildAIGC 181). The next two scripts evaluate existing Phase 7 candidate checkpoints and the production frozen head on those same validation roles. They do no training or model selection. None of the Phase 9 scripts opens calibration or locked test images. Feature vectors are cached in `experiments/reports/phase9_feature_vectors.npz`; the archive is reused only when sample identities and dimensions match.

Outputs include the machine-readable reports `phase9_model_baseline_audit.json`, `phase9_generator_error_analysis.json`, `phase9_candidate_validation_errors.json`, `phase9_feature_space_analysis.json`, `phase9_frequency_feature_analysis.json`, `phase9_feature_complementarity.json`, group/source failure cases, the univariate ablation CSV, and PCA figures. The forensic extractor has focused edge-case tests at `experiments/forensics/test_phase9_frequency_features.py`. The Phase 10 document is a draft protocol; these scripts do not train a detector or alter production.
