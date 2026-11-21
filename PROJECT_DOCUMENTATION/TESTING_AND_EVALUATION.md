# TruthLens Testing and Evaluation

## 1. Scope of this audit

This document records the comprehensive automated test suite and empirical evaluation evidence across the TruthLens repository. It covers unit, integration, and regression testing in `backend/tests/`, as well as the complete multi-phase benchmark evaluations conducted across three externally ingested datasets ($N = 29,640$ total images) culminating in Phase 7 locked test evaluations and candidate governance decisions.

## 2. Test infrastructure

- **Backend Test Suite:** Automated test files in `backend/tests/` cover API endpoints, security bounds, lifecycle state machines, service logic, forensic analyzers, OCR services, rule-based claims, evidence fusion, confidence calibration, and Phase 7 governance contracts (`test_phase7.py`).
- **Evaluation Scripts:** `experiments/evaluate_phase7_test.py`, `experiments/calibrate_phase7_candidate.py`, `experiments/statistical_comparison_phase6.py`, and `experiments/audit_phase7_shortcuts.py` provide reproducible evaluation pipelines with cryptographic seed controls (`seed = 42`).
- **Benchmark Partitions:** Controlled by split manifests under `experiments/manifests/`, with 10,385 samples in the locked test set, 13,000 in the calibration pool, 6,255 in validation, and 6,995 in approved training.

## 3. Documented automated test results

| Test Suite / Scope | Command | Passed | Skipped | Failed | Execution Time | Coverage / Notes |
|---|---|---|---|---|---|---|
| **Full Backend Suite** | `python -m pytest backend/tests/ -q` | **267** | **10** | **0** | **28.60s** | Comprehensive backend coverage: API routes, upload security, async orchestration, stage execution, fusion edge cases, calibration schemas, and Phase 7 candidate governance. |
| **Phase 7 Governance & Contracts** | `pytest backend/tests/test_phase7.py -q` | **7** | **0** | **0** | **0.85s** | Verifies candidate rejection, frozen baseline preservation in production, calibration temperature bounds, and in-memory augmentation determinism. |
| **Fusion Engine Regressions** | `pytest backend/tests/test_phase6_fusion.py -q` | **8** | **0** | **0** | **0.32s** | Verifies that unavailable/non-completed evidence cannot vote and that forensic cues remain informational. |
| **Model Smoke Tests** | (Part of backend suite) | — | 10 | 0 | — | Opt-in real weight checks requiring downloaded local model weights and active CUDA runtime. |

## 4. Benchmark dataset inventory and split governance

Three external image benchmarks were ingested and cryptographically audited:

| Benchmark | Total Samples | Authentic (Real) | Synthetic (AI) | Generator Coverage / Modality | Partition Allocations |
|---|---|---|---|---|---|
| **Tiny-GenImage** | 25,006 | 12,506 | 12,500 | 8 generators (BigGAN, Cascade Diffusion, DeepFloyd-IF, GLIDE, LDM, Midjourney v4, SD 1.4, SD 1.5) | Train: 6,995<br>Val: 3,464<br>Calib: 10,000<br>Locked Test: 7,000 |
| **Frontier Benchmark 2026-10** | 2,976 | 984 | 1,992 | 8 modern generators (DALL-E 3, Flux.1-dev, Flux.1-schnell, Midjourney v6, Recraft v3, SD 3 Medium, SD 3.5 Large, SD 3.5 Turbo) across 4 compression tiers | Val: 1,791<br>Calib: 2,000<br>Locked Test: 2,091 |
| **WildAIGC-Bench** | 1,658 | 829 | 829 | Real-world in-the-wild digital media | Val: 1,000<br>Calib: 1,000<br>Locked Test: 1,294 |
| **Total** | **29,640** | **14,319** | **15,321** | Diverse multi-generator ecosystem | **Locked Test: 10,385** |

Splits were generated using group-aware stratification (`experiments/create_evaluation_splits.py`) to isolate prompt identities and generator classes, preventing evaluation data leakage.

## 5. Model benchmark evaluation results

### 5.1 Baseline evaluation: `truthlens/siglip-cifake-frozen-head`

The production baseline (frozen SigLIP-base image encoder with local linear head, SHA-256: `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`) was evaluated across all locked test partitions ($N = 10,385$):

| Test Partition | $N$ | Balanced Accuracy | ROC-AUC | Real Accuracy | AI Accuracy |
|---|---|---|---|---|---|
| **Tiny-GenImage Test** | 7,000 | 0.5521 | 0.6123 | 0.1609 | 0.9434 |
| **Tiny-GenImage (Matched Aug)** | 7,000 | 0.5330 | 0.5595 | 0.1191 | 0.9469 |
| **Frontier Benchmark 2026-10** | 2,091 | 0.6228 | 0.6501 | 0.4000 | 0.8457 |
| **WildAIGC-Bench Test** | 1,294 | 0.4294 | 0.4226 | 0.0402 | 0.8176 |

### 5.2 Phase 6 shortcut discovery & candidate E3 failure

In Phase 5, Candidate E3 (`truthlens/siglip-base-tiny-genimage-e3_partial_finetuning`) was fine-tuned on the approved Tiny-GenImage training data without augmentation. Evaluation in Phase 6 revealed that E3 learned an unintended dataset shortcut:
- In the Tiny-GenImage training manifest, 100% of authentic images were JPEG and 100% of synthetic images were PNG.
- A Random Forest classifier trained solely on file metadata achieved an ROC-AUC of 1.0000.
- When subjected to an in-memory JPEG-90 re-encoding probe, Candidate E3 flipped 45.6% of its predictions with a mean logit shift of 6.12.
- Candidate E3 was decisively rejected for production deployment.

### 5.3 Phase 7 shortcut-resistant candidates & locked test comparison

In Phase 7, TruthLens implemented matched in-memory augmentation (randomized JPEG/PNG/WebP transcoding, scale jitter, and compression quality variation applied identically to both classes). Candidate S2 (`truthlens/siglip-base-v2-s2`) was trained and evaluated on the locked test sets against the frozen baseline:

| Evaluation Partition / Angle | Baseline | Candidate S2 (`siglip-base-v2-s2`) | Difference ($\Delta$) | 95% Group-Aware Bootstrap CI |
|---|---|---|---|---|
| **Tiny-GenImage Test (Standard)** | BalAcc: 0.5521<br>AUC: 0.6123 | **BalAcc: 0.8460**<br>**AUC: 0.9423** | +0.2939 BalAcc<br>+0.3300 AUC | $[+0.3184, +0.3412]$ |
| **Tiny-GenImage Test (Matched Aug)** | BalAcc: 0.5330<br>AUC: 0.5595 | **BalAcc: 0.8071**<br>**AUC: 0.9044** | +0.2741 BalAcc<br>+0.3449 AUC | $[+0.3312, +0.3582]$ |
| **Frontier Benchmark 2026-10** | **BalAcc: 0.6228**<br>**AUC: 0.6501** | BalAcc: 0.4849<br>AUC: 0.4234 | -0.1379 BalAcc<br>-0.2267 AUC | $[-0.2781, -0.1720]$ |
| **WildAIGC-Bench Test** | BalAcc: 0.4294<br>AUC: 0.4226 | **BalAcc: 0.5589**<br>**AUC: 0.6049** | +0.1295 BalAcc<br>+0.1823 AUC | $[+0.1472, +0.2178]$ |
| **JPEG-90 In-Memory Probe** | 22.4% flip rate | **11.0% flip rate** | -11.4% flips | Shift: 0.793 (vs. E3: 6.12) |

### 5.4 Candidate governance decision

**Decision:** `REJECT_CANDIDATE_FOR_PRODUCTION`, `RETAIN_FROZEN_BASELINE_IN_PRODUCTION`.
- **Reasoning:** Candidate S2 successfully resolved the container format shortcut (+0.3300 AUC on Tiny-GenImage; 11.0% probe flips), but fine-tuning on 2023 diffusion data degraded performance on late-2026 frontier architectures (-0.2267 AUC on Frontier Benchmark).
- Production detector remains strictly `truthlens/siglip-cifake-frozen-head`.

## 6. Post-hoc calibration results

Temperature scaling was fitted on the 13,000-sample disjoint calibration pool:
- Tiny-GenImage ($N = 10,000$): $T = 1.2174$
- Frontier Benchmark ($N = 2,000$): $T = 10.3621$
- WildAIGC-Bench ($N = 1,000$): $T = 5.6099$
- Pooled Calibration: $T = 1.3187$

All optimizations converged without hitting parameter boundaries ($0.1 \le T \le 100.0$), successfully mitigating overconfidence on out-of-domain distributions.

## 7. Recommended evaluation protocol for future iterations

1. **Multi-Generator Training Pools:** Expand training data beyond early diffusion models (2023) to include modern flow-matching and diffusion transformer architectures (FLUX, SD 3.5).
2. **Strict Disjoint Stratification:** Preserve group-aware splitting to prevent subject, prompt, or source leakage across train, val, calib, and test sets.
3. **Mandatory Transcoding Probes:** Test all candidate models against in-memory format and compression probes to ensure representations do not rely on container artifacts.
4. **Independent Benchmark Gates:** Require non-inferiority or improvement across held-out out-of-domain benchmarks before promoting any candidate to production.
