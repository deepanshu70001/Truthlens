# TruthLens Phase 7: Locked Test Evaluation, Generalization & Candidate Decision Report

**Repository:** `https://github.com/deepanshu70001/Truthlens`  
**Working Branch:** `fix/truthlens-remediation`  
**Phase:** Phase 7 — Step 4: Model Selection, Calibration & Locked Test Evaluation  
**Evaluation Date:** November 22, 2026  
**Primary Evaluated Candidate:** `truthlens/siglip-base-v2-s2` (S2: SigLIP-base layers 10, 11 + head fine-tuned with matched in-memory augmentation)  
**Secondary Evaluated Candidate:** `truthlens/siglip-base-v2-s1` (S1: SigLIP-base frozen linear probe on $K=5$ precomputed augmented views)  
**Production Baseline:** `truthlens/siglip-cifake-frozen-head` (`models/baselines/siglip_frozen_head.pt`)  

---

## 1. Executive Summary & Verdict

Phase 7 evaluated whether training with matched in-memory augmentation (rescaling, cropping, randomized JPEG/WebP/PNG transcoding, and blur applied identically to authentic and synthetic images) resolves the file-format container shortcut observed in Phase 6, and whether the resulting models generalize to modern digital media.

### Key Empirical Findings:
1. **Shortcut Neutralization Confirmed on In-Distribution Data:**  
   Unlike Candidate E3 from Phase 6 (which suffered 45.6% prediction flips under JPEG transcoding), Candidate S2 achieved **0.8460 Balanced Accuracy** and **0.9423 ROC-AUC** on the raw Tiny-GenImage locked test partition ($N=7,000$). When evaluated under **matched in-memory augmentation**, S2 maintained **0.8071 Balanced Accuracy** and **0.9044 ROC-AUC** (compared to 0.5330 and 0.5595 for the frozen baseline). This confirms that S2 learned generative image features rather than container metadata artifacts.
2. **WildAIGC-Bench Improvement:**  
   On WildAIGC-Bench ($N=1,294$), S2 scored **0.5589 Balanced Accuracy** and **0.6049 ROC-AUC** (a $+18.23\%$ improvement over the baseline's $0.4226$ ROC-AUC). Secondary candidate S1 achieved **0.6214 Balanced Accuracy** and **0.6628 ROC-AUC**. (Reported with the established caveat that WildAIGC labels are contextual).
3. **Cross-Generator Distribution Gap on Frontier Benchmark:**  
   On the Frontier Benchmark October 2026 test partition ($N=2,091$, representing cutting-edge 2026 generators including Flux-2/3, Ideogram 4.5, GPT-Image-2, Grok-Imagine-v2), S2 achieved **0.4849 Balanced Accuracy** and **0.4234 ROC-AUC**, underperforming the frozen baseline ($0.6228$ Balanced Accuracy, $0.6501$ ROC-AUC). Paired group-aware bootstrap analysis (1,000 resamples) yields $\Delta \text{ROC-AUC} = -0.2267$ with a 95% confidence interval strictly below zero ($[-0.2781, -0.1720]$).
4. **Pre-Registered Decision:**  
   Candidate S2 **FAILED** three of the four pre-registered promotion conditions. Consequently, Candidate S2 is formally designated **`REJECT_CANDIDATE_FOR_PRODUCTION`**. The TruthLens production system strictly retains the frozen baseline detector (`truthlens/siglip-cifake-frozen-head`).

---

## 2. Audited Model Identities & Cryptographic Integrity

All models were evaluated under strict cryptographic provenance:

| Model Role | Model ID | Weights File | File SHA-256 Digest | Trainable Parameters | Preprocessing Pipeline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Production Baseline** | `truthlens/siglip-cifake-frozen-head` | `models/baselines/siglip_frozen_head.pt` | `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538` | Head only (769 params) | `siglip-frozen-v1` (canonical 224x224) |
| **Selected Primary (S2)** | `truthlens/siglip-base-v2-s2` | `experiments/checkpoints/phase7/s2_siglip_partial_ft/best_checkpoint.pt` | `620e42f74bb7ae24fb1deeaedf24dbf9147162d3612c8f5171b64e53834f9805` | Layers 10, 11 + Head (14,176,513 params) | Matched in-memory augmentation |
| **Audited Secondary (S1)** | `truthlens/siglip-base-v2-s1` | `experiments/checkpoints/phase7/s1_siglip_linear_probe/best_checkpoint.pt` | `2554519107c8ad8de07805e55225e7d5a5ac8f9959aecb47815ac86f6fd92111` | Head only (769 params) | $K=5$ precomputed augmented views |

*Pre-evaluation identity freezing was committed to Git in commit `b823ba9` prior to accessing the locked test partitions.*

---

## 3. Comprehensive Benchmark Test Performance

### 3.1 Primary Classification Metrics Across Partitions

| Benchmark Partition | Sample Count | Evaluated Model | Accuracy | Balanced Accuracy | Precision | Recall | Specificity | F1 Score | ROC-AUC | PR-AUC |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tiny-GenImage Test (Raw)** | 7,000 | Baseline | 0.5521 | 0.5521 | 0.5356 | 0.7854 | 0.3189 | 0.6369 | 0.6123 | 0.6181 |
| | | **S2 Candidate** | **0.8460** | **0.8460** | **0.9420** | 0.7374 | **0.9546** | **0.8272** | **0.9423** | **0.9489** |
| | | S1 Candidate | 0.7849 | 0.7849 | 0.8339 | 0.7114 | 0.8583 | 0.7678 | 0.8741 | 0.8773 |
| **Tiny-GenImage (Augmented)** | 7,000 | Baseline | 0.5330 | 0.5330 | 0.5204 | 0.8417 | 0.2243 | 0.6432 | 0.5595 | 0.5604 |
| | | **S2 Candidate** | **0.8071** | **0.8071** | **0.8768** | 0.7131 | **0.9011** | **0.7866** | **0.9044** | **0.9056** |
| | | S1 Candidate | 0.7267 | 0.7267 | 0.7410 | 0.6974 | 0.7560 | 0.7186 | 0.8066 | 0.8016 |
| **Frontier Benchmark Test** | 2,091 | **Baseline** | **0.5801** | **0.6228** | **0.4459** | **0.7689** | 0.4767 | **0.5645** | **0.6501** | **0.4533** |
| | | S2 Candidate | 0.5361 | 0.4849 | 0.3328 | 0.3095 | **0.6603** | 0.3207 | 0.4234 | 0.3369 |
| | | S1 Candidate | 0.5347 | 0.5568 | 0.4003 | 0.6324 | 0.4811 | 0.4903 | 0.5464 | 0.4000 |
| **WildAIGC-Bench Test** | 1,294 | Baseline | 0.4938 | 0.4294 | 0.6586 | 0.4682 | 0.3906 | 0.5475 | 0.4226 | 0.6121 |
| | | S2 Candidate | 0.6360 | 0.5589 | 0.7099 | 0.7748 | 0.3430 | 0.7410 | 0.6049 | 0.7441 |
| | | **S1 Candidate** | **0.6870** | **0.6214** | **0.7473** | **0.8042** | **0.4395** | **0.7747** | **0.6628** | **0.7812** |

---

## 4. Paired Group-Aware Bootstrap Analysis

Group-aware bootstrap resampling (1,000 iterations, clustering synthetic images by `group_id`) was executed on paired sample predictions:

| Benchmark Partition | Candidate Comparison | Metric Difference | Mean Point Estimate | 95% Bootstrap Confidence Interval | Statistically Significant |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Tiny-GenImage Test** | **S2 vs Baseline** | $\Delta \text{BalancedAccuracy}$ | $+0.2938$ | $[+0.2703, +0.3189]$ | **Yes** ($\Delta > 0$) |
| | | $\Delta \text{ROC-AUC}$ | $+0.3287$ | $[+0.3034, +0.3554]$ | **Yes** ($\Delta > 0$) |
| | S1 vs Baseline | $\Delta \text{BalancedAccuracy}$ | $+0.2322$ | $[+0.2102, +0.2545]$ | **Yes** ($\Delta > 0$) |
| | | $\Delta \text{ROC-AUC}$ | $+0.2605$ | $[+0.2392, +0.2800]$ | **Yes** ($\Delta > 0$) |
| **Frontier Benchmark Test** | **S2 vs Baseline** | $\Delta \text{BalancedAccuracy}$ | $-0.1380$ | $[-0.1827, -0.0941]$ | **Yes** ($\Delta < 0$) |
| | | $\Delta \text{ROC-AUC}$ | $-0.2258$ | $[-0.2781, -0.1720]$ | **Yes** ($\Delta < 0$) |
| | S1 vs Baseline | $\Delta \text{BalancedAccuracy}$ | $-0.0658$ | $[-0.1032, -0.0264]$ | **Yes** ($\Delta < 0$) |
| | | $\Delta \text{ROC-AUC}$ | $-0.1037$ | $[-0.1450, -0.0620]$ | **Yes** ($\Delta < 0$) |
| **WildAIGC-Bench Test** | **S2 vs Baseline** | $\Delta \text{BalancedAccuracy}$ | $+0.1293$ | $[+0.0901, +0.1684]$ | **Yes** ($\Delta > 0$) |
| | | $\Delta \text{ROC-AUC}$ | $+0.1823$ | $[+0.1450, +0.2205]$ | **Yes** ($\Delta > 0$) |
| | S1 vs Baseline | $\Delta \text{BalancedAccuracy}$ | $+0.1918$ | $[+0.1550, +0.2290]$ | **Yes** ($\Delta > 0$) |
| | | $\Delta \text{ROC-AUC}$ | $+0.2402$ | $[+0.2030, +0.2780]$ | **Yes** ($\Delta > 0$) |

---

## 5. Generator Generalization Analysis (Seen vs Unseen)

Tracking detection rates across individual generator subsets on the Tiny-GenImage locked test partition:

| Generator Name | Training Status | Sample Count | Baseline Recall | S2 Recall | S1 Recall | Difference (S2 - Base) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ImageNet-Camera** | Authentic Reference | 3,500 | 0.3189 | **0.9546** | 0.8583 | $+0.6357$ |
| **ADM** | Seen in Training | 500 | 0.6760 | **0.8640** | 0.7720 | $+0.1880$ |
| **BigGAN** | Seen in Training | 500 | 0.6060 | **1.0000** | 0.9480 | $+0.3940$ |
| **Stable Diffusion v1.5** | Seen in Training | 500 | **0.9280** | 0.7040 | 0.6820 | $-0.2240$ |
| **GLIDE** | Seen in Training | 500 | 0.8940 | **0.9880** | 0.9320 | $+0.0940$ |
| **Midjourney** | Held-Out / Unseen | 500 | **0.8900** | 0.5780 | 0.5360 | $-0.3120$ |
| **Wukong** | Held-Out / Unseen | 500 | **0.8740** | 0.5380 | 0.5180 | $-0.3360$ |
| **VQDM** | Held-Out / Unseen | 500 | **0.6300** | 0.4900 | 0.4920 | $-0.1400$ |

*Findings:* S2 demonstrates high accuracy on seen generators (86.4% on ADM, 100% on BigGAN, 98.8% on GLIDE) and high specificity on real images (95.5% vs 31.9% for baseline). However, on held-out generators, detection recall decreases: Midjourney is 57.8%, Wukong is 53.8%, and VQDM is 49.0% (falling below the 50.0% criterion).

---

## 6. Social Media Compression Robustness

Frontier Benchmark locked test performance broken down across compression channels:

| Channel / Variant | Sample Count | Baseline BalAcc | Baseline ROC-AUC | S2 BalAcc | S2 ROC-AUC | Difference ($\Delta \text{AUC}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original** | 525 | 0.6164 | 0.6395 | 0.4825 | 0.4270 | $-0.2125$ |
| **Instagram** | 522 | 0.6375 | 0.6646 | 0.4817 | 0.4189 | $-0.2457$ |
| **Reshared** | 522 | 0.6200 | 0.6518 | 0.4884 | 0.4270 | $-0.2248$ |
| **WhatsApp** | 522 | 0.6175 | 0.6442 | 0.4869 | 0.4215 | $-0.2227$ |

*Findings:* Performance across all four variants is highly consistent (Balanced Accuracy 0.4817 - 0.4884, ROC-AUC 0.4189 - 0.4270). This confirms that social-media compression artifacts do not induce localized instability, but rather reflect an underlying domain shift relative to the 2026 generators.

---

## 7. Pre-Registered Criteria Audit & Candidate Decision

Evaluation of Candidate S2 against the four pre-registered conditions in `docs/phase7_protocol.md`:

| Condition | Protocol Requirement | Empirical Measurement | Threshold / Standard | Pass / Fail Status |
| :--- | :--- | :--- | :--- | :---: |
| **Condition 1** | Frontier Test Generalization $\text{CI}_{95\%} > 0$ | $\Delta \text{BalAcc} = -0.1380$ ($[-0.1827, -0.0941]$)<br>$\Delta \text{AUC} = -0.2258$ ($[-0.2781, -0.1720]$) | $\text{CI}_{95\%} > 0$ | **FAIL** |
| **Condition 2** | Real-Image False Positive Rate $\le 5.0\%$ | $\text{FAR} = 33.97\%$ (at threshold 0.5)<br>Operating threshold $t = 0.9293$ required | $\text{FAR} \le 5.0\%$ | **FAIL** |
| **Condition 3** | JPEG-90 Transcoding Flip Rate $< 25.0\%$ | Flip rate = **11.0%** (mean $|\Delta z| = 0.793$) | Flip $< 25.0\%$ | **PASS** |
| **Condition 4** | Unseen Generator Recall $\ge 50.0\%$ | Midjourney: 57.8%, Wukong: 53.8%, VQDM: **49.0%** | Recall $\ge 50.0\%$ | **FAIL** |

### Formal Decision
- **Verdict:** **`REJECT_CANDIDATE_FOR_PRODUCTION`**
- **Action:** **`RETAIN_FROZEN_BASELINE_IN_PRODUCTION`**
- **Deployment Status:** Model `truthlens/siglip-base-v2-s2` is archived as a research milestone only. The production detector remains strictly locked to `truthlens/siglip-cifake-frozen-head`.

---

## 8. Scientific Insights & Research Synthesis

1. **Format Shortcut Neutralization is Successful:**  
   The primary technical objective of Phase 7—eliminating the container format shortcut that crippled Candidate E3—was successfully achieved. Unlike E3 (which flipped 45.6% under JPEG-90 and collapsed to random chance when transcoded), Candidate S2 maintained an **80.71% Balanced Accuracy and 0.9044 ROC-AUC on matched augmented data**.
2. **The Generation Gap (2023 Training Data vs 2026 Test Frontier):**  
   Tiny-GenImage consists of early generative architectures (BigGAN, GLIDE, SD 1.5, ADM). The Frontier Benchmark consists of late-2026 diffusion transformers, flow-matching architectures, and unified multimodal models (Flux-2, Flux-3, Ideogram 4.5, GPT-Image-2, Grok-Imagine-v2). S2 learned features specific to 2023 diffusion/GAN synthesis that do not reliably transfer to modern 2026 architectures.
3. **The Importance of Pre-Registered Governance:**  
   Without pre-registered multi-benchmark evaluation, a model scoring 0.8460 accuracy on Tiny-GenImage might have been prematurely deployed. The group-aware evaluation protocol successfully prevented a regression in production authenticity verification.
