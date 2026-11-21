# TruthLens Research Foundation

## 1. Domain and research context

TruthLens operates at the intersection of AI-generated image detection, digital image forensics, OCR and claim verification, provenance tracking, explainability, and online misinformation research:

- **Image authenticity:** Discriminating synthetic or AI-generated media from camera-origin photographs.
- **Manipulation detection:** Localizing digital splicing, retouching, or inpainting (classical forensics provide heuristic cues; no separate deep segmentation model is implemented).
- **Claim accuracy:** Verifying whether textual propositions correspond to factual reality (supported via OCR, rule-based extraction, and optional third-party fact-check integration).
- **Provenance:** Tracking where and when visual media appeared online (supported via perceptual hashing and optional Google Vision reverse lookup).
- **Spread & impact:** Estimating potential content propagation and societal sensitivity (heuristic rules and domain token dictionaries; empirical spread prediction is explicitly marked `DATA_UNAVAILABLE`).

## 2. Theoretical motivation

A single machine learning classifier is fundamentally insufficient for comprehensive digital media verification. An authentic photograph may be paired with fabricated claims (miscontextualization), while an AI-generated artwork may illustrate an entirely factual historical event. TruthLens therefore structures evidence into distinct, status-aware channels, ensuring that technical uncertainty, component unavailability, and conflicting cues are explicitly presented to human analysts.

## 3. Related literature and theoretical lineage

| Reference | Domain | Relevance to TruthLens | Methodological Insight |
|---|---|---|---|
| **Wang et al. (CVPR 2020)** | CNN-generated image detection | Demonstrates that CNN generators leave systematic high-frequency spectral artifacts | Motivates rigorous cross-generator holdout testing to assess artifact overfitting |
| **Ojha et al. (CVPR 2023)** | Universal fake image detection | Leverages pretrained vision-language representations (CLIP) for linear probing | Demonstrates that frozen multimodal representations linearize synthetic/authentic separation |
| **Zhu et al. (NeurIPS 2023)** | GenImage benchmark | 1M+ multi-generator benchmark covering GANs and early diffusion models | Source of the curated Tiny-GenImage benchmark ($N = 25,006$) used in TruthLens |
| **Tschannen et al. (2025)** | SigLIP / SigLIP 2 | Multilingual vision-language encoders with sigmoid loss | Architectural backbone for TruthLens production detector (`google/siglip-base-patch16-224`) |
| **Selvaraju et al. (ICCV 2017)** | Grad-CAM | Visual explanations via gradient-based localization | Provides visual attribution maps indicating model focus areas |
| **Vosoughi et al. (Science 2018)** | Diffusion of online falsehoods | Large-scale empirical analysis of information propagation | Underscores that spread prediction requires longitudinal outcome data rather than static visual heuristics |

## 4. Implemented methodology and empirical progression

### 4.1 Production authenticity detector

The TruthLens production detector uses a frozen `google/siglip-base-patch16-224` vision encoder paired with a local linear classification head (`models/baselines/siglip_frozen_head.pt`, SHA-256: `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`). Preprocessing enforces normalized bicubic resizing to $224 \times 224$ (`siglip-frozen-v1`). Inference emits calibrated softmax probabilities with explicit uncertainty bands ($[0.35, 0.65]$).

### 4.2 Dataset ingestion and shortcut discovery

In Phase 2, TruthLens ingested three external benchmarks ($N = 29,640$ total images): Tiny-GenImage, Frontier Benchmark October 2026, and WildAIGC-Bench. Group-aware splitting isolated disjoint generator families, prompt groups, and semantic domains.

Pre-training container format audits revealed that 100% of authentic images in Tiny-GenImage were JPEG and 100% of synthetic images were PNG. Early unaugmented fine-tuning experiments (Candidate E3, Phase 5/6) exploited this container artifact, causing severe prediction flips (45.6%) under in-memory JPEG-90 probes.

### 4.3 Matched in-memory augmentation (Phase 7)

To neutralize container-format confounding without modifying raw dataset files on disk, TruthLens engineered a matched in-memory augmentation pipeline (`experiments/train_phase7.py`):
- Random in-memory transcoding into JPEG, PNG, or WebP formats.
- Quality jittering ($Q \in [30, 95]$).
- Random aspect-ratio and multi-scale cropping.
- Identical application to both authentic and synthetic classes.

This intervention collapsed container metadata classification ROC-AUC from 1.0000 to chance level (0.5028).

## 5. Research questions: empirical findings

| Research Question | Focus | Empirical Status & Evidence |
|---|---|---|
| **RQ1: Cross-Generator Generalization** | How do frozen and fine-tuned SigLIP detectors generalize across disparate generator architectures? | **Evaluated on $N = 10,385$ locked test samples.** Frozen baseline achieved 0.6123 AUC on Tiny-GenImage (2023 diffusion) and 0.6501 AUC on Frontier Benchmark (late-2026 generators). Fine-tuned candidate S2 achieved 0.9423 AUC on Tiny-GenImage (+0.3300 gain), but degraded to 0.4234 AUC on Frontier Benchmark (-0.2267 drop), demonstrating catastrophic negative transfer across generator eras. |
| **RQ2: Container Shortcut Susceptibility** | Can image detectors exploit encoding containers rather than generative semantics? | **Confirmed and Resolved.** Without augmentation, detectors exploit JPEG/PNG class confounding (E3: 45.6% flip rate). Matched in-memory augmentation successfully neutralized this shortcut (S2: 11.0% flip rate, 0.793 mean shift). |
| **RQ3: Forensic & Contextual Contribution** | Do classical forensics improve multi-modal verification accuracy? | **Audited and Constrained.** Classical forensics (ELA, noise, texture) were proven encoding-sensitive. Phase 6 fusion logic restricts forensics to informational evidence; non-completed evidence cannot vote. |
| **RQ4: Confidence Calibration** | Can post-hoc temperature scaling mitigate out-of-domain overconfidence? | **Implemented & Fitted.** Temperature scaling fitted on 13,000 calibration samples ($T = 1.2174$ Tiny, $T = 10.3621$ Frontier, $T = 5.6099$ WildAIGC, $T = 1.3187$ Pooled), smoothing extreme logit overconfidence. |
| **RQ5: Social & Regional Predictability** | Can static visual and textual cues predict real-world viral spread or regional reach? | **Empirically Bound.** Static cues lack correlation with dissemination outcomes. Regional spread is designated `DATA_UNAVAILABLE` to maintain scientific integrity. |

## 6. Academic contributions

1. **Empirical Shortcut Discovery & Mitigation:** Rigorous demonstration of container format confounding in popular benchmark subsets and formulation of an in-memory matched augmentation defense that collapses metadata separability to chance.
2. **Evaluation Governance Framework:** Demonstration of a pre-registered evaluation protocol that prevented deployment of a shortcut-prone candidate despite impressive in-distribution metrics (+0.3300 AUC gain).
3. **Reproducible Multi-Benchmark Protocol:** Group-aware, leakage-free split definitions and cryptographic tracking across 29,640 images spanning 16 generator architectures.
4. **Transparent Multi-Modal Architecture:** A modular verification prototype integrating visual detection, forensic heuristics, OCR, claim extraction, and status-aware evidence fusion.
