# TruthLens Research Foundation

## 1. Domain and research context

TruthLens sits at the intersection of AI-generated image detection, digital image forensics, OCR and claim verification, provenance, explainability, and misinformation research. These are related but distinct tasks:

- **Image authenticity:** whether an image resembles samples from the classifier's AI-generated or real classes.
- **Manipulation:** whether/how an image was edited; the current classifier has no separate manipulation class.
- **Claim accuracy:** whether a text proposition is supported by evidence; image authenticity alone cannot answer this.
- **Provenance:** where an image may have appeared; a web match does not establish the first source.
- **Spread/impact:** observed dissemination and outcomes over time; content cues alone are not measured spread.

The project integrates signals to support review. Its documented implementation is an engineering prototype, not a validated scientific detector.

## 2. Motivation and interpretation

The project documentation motivates a multi-signal workflow because a single classifier can be insufficient for contextual review. This is a defensible design rationale: authenticity, caption truth, source and impact are different dimensions. It is not evidence that the integrated application improves reviewer accuracy or reduces misinformation; no controlled user experiment has established those outcomes.

## 3. Related research already cited in repository

The following works are cited in `docs/phase6_final_research_report.md`. They provide context, not direct performance comparisons with TruthLens. Verify final bibliographic metadata from the publisher/source before dissertation submission.

| Work | Research topic | Relevance to TruthLens | Boundary |
|---|---|---|---|
| S. Wang et al. (CVPR 2020), “CNN-Generated Images Are Surprisingly Easy to Spot... for Now” | CNN-generated image detection and generalization concerns | Motivates testing detector shifts across generators | Does not evaluate the TruthLens checkpoint |
| U. Ojha, Y. Li, Y. J. Lee (CVPR 2023), “Towards Universal Fake Image Detectors That Generalize Across Generative Models” | Cross-generator fake-image detector generalization | Relevant to a robust detector evaluation protocol | No compatible head-to-head evaluation was run in this checkout |
| M. Zhu et al. (NeurIPS Datasets and Benchmarks 2023), “GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image” | Large-scale AI-generated image benchmark | Potential evaluation/data reference subject to licensing and protocol review | Not installed or used by TruthLens here |
| M. Tschannen et al. (2025), “SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features” | Vision-language encoder | Backbone family context | Encoder results do not measure TruthLens binary head effectiveness |
| R. R. Selvaraju et al. (ICCV 2017), “Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization” | Gradient-based activation maps | Methodological background for attribution view | A visualization does not prove faithful explanation/manipulation localization |
| S. Vosoughi, D. Roy, S. Aral (Science 2018), “The spread of true and false news online” | Observed online news diffusion | Context for why outcome data is required for spread claims | TruthLens has no matching spread outcome dataset or trained predictor |

These citations are supplied as repository content; the project team should check exact publication metadata and reference format before final submission. Do not infer that TruthLens reproduces any study's results.

## 4. Implemented method

### Authenticity stage

The default configuration uses a local `siglip_frozen_head.pt` checkpoint paired with `google/siglip-base-patch16-224`. The repository's authenticity report says the encoder was frozen, pooled image features were used to train a one-logit binary head with BCE, and AI-generated is the positive class. Runtime inference reports class scores/tag through the configured detector. A prior hosted SigLIP2 detector is an alternate backend. Scores are uncalibrated. There is no independently reviewed calibration artifact matched to the evaluated checkpoint.

The `ml/models` branch reports a 200-image CIFAKE test subset (100 images/class): accuracy 0.87, AI recall 0.83 and ROC-AUC 0.9613. These are branch-reported values and were not reproduced from a reviewed dataset manifest in this checkout. The local approved-manifest evaluation is `NOT_EVALUABLE`, n=0. Neither the reported subset nor the encoder literature establishes generalization to current generators or real-world images.

### Forensic indicators

The code calculates classical image statistics/cues such as ELA, noise, JPEG, edge, texture and resampling. Their interpretation is sensitive to encoding and editing history; they are not probabilities. Phase 6 restricts their contribution to informational evidence under the current fusion rules.

### OCR, text, claims and fact-check retrieval

OCR is conditionally provided by EasyOCR/Tesseract. Text normalization and candidate claim extraction are rule-based. The optional fact-check adapter retrieves provider reviews. No local annotated OCR benchmark, extraction gold set or reviewed retrieval relevance dataset is present; therefore component accuracy is unmeasured. A provider rating is not a TruthLens-generated truth decision.

### Explanations and context analytics

Grad-CAM is optional attribution, requiring a compatible model path. Evidence fusion is rule-based. Virality and social impact use content cues; regional predictions are unavailable. These are not learned estimates of real-world dissemination.

## 5. Research gap and contribution claims

The repository supports the following careful formulation of a project-level gap: TruthLens explores an integrated interface and report that preserves stage-level status and separates image-classifier output from OCR, forensic indicators and optional external context. The code demonstrates that integration, not that the signals improve accuracy.

Potential contribution:

1. A modular local web prototype for submitting and inspecting image-analysis outputs.
2. An explicit pipeline/report schema that can show completed, partial, skipped and unavailable stages.
3. A reproducibility and research-readiness package identifying data, evaluation and responsible-use requirements.

Claims not supported: a novel detector, state-of-the-art performance, calibrated confidence, reliable manipulation detection, enhanced truth-detection accuracy from fusion, measured virality prediction, regional risk prediction, reduced user error or production readiness.

## 6. Research questions for future empirical work

| RQ | Proposed question | Needed data/measurements | Current status |
|---|---|---|---|
| RQ1 | How does the exact TruthLens checkpoint perform across independent real/AI sources and generators? | Rights-cleared labeled samples, generator/source groups, fixed splits; per-class precision/recall/F1, ROC-AUC, PR-AUC, coverage and confidence intervals | Not evaluable, approved manifest n=0 |
| RQ2 | Are Grad-CAM maps class-consistent and useful to reviewers? | Region masks, parameter/transform sanity checks, reviewer protocol | No masks/reviewer study |
| RQ3 | Do forensic cues add useful information beyond the authenticity model? | Paired independent set, source/transform labels, controlled ablations | No measured gain; current cues informational |
| RQ4 | How accurately are text claims extracted and matched to provider reviews? | OCR/claim annotations, independently rated retrieval relevance, provider/outage tracking | No gold set or human relevance labels |
| RQ5 | What are OCR/claim performance differences across supported scripts and image quality? | Licensed multilingual benchmark, CER/WER, language strata | Not measured; configured OCR en/hi |
| RQ6 | Can social outputs predict actual spread or harm outcomes? | Prospective, permitted timestamped outcome data with region/source definitions | No outcome dataset; current indicators heuristic/unavailable |
| RQ7 | Does the integrated report improve reviewers' accuracy or calibration? | Consent-based controlled user study, task design, independent ground truth | No user study conducted |

Hypotheses should be preregistered and tied to a valid sample, metrics and statistical plan before data collection. No hypothesis result is asserted here.

## 7. Dataset and evaluation readiness

Repository audit results in Phase 6:

- 35 readable files under `test_data/originals` (31 JPEG, 4 PNG).
- No source/license/provenance record, labels, annotation guide, event/source/generator identifiers or train/validation/test split was found.
- The audit identified 31 exact duplicate extra copies and 33 exact dHash candidate repeats; candidate matches are not all confirmed near duplicates and should not be silently removed.
- `test_data/` contains 190 total files including derived sizes and heatmaps; this is not 190 independent labeled examples.
- Runtime `data/` is excluded as user/application artifacts.
- The reviewed manifest template is header-only. Running evaluator returned `NOT_EVALUABLE` with zero eligible rows and did not perform inference.
- No model metrics, calibration measures, robustness results, OCR CER/WER, retrieval relevance, participant study or predictive spread result can be claimed from this evidence.

The fine-tuned model branch's CIFAKE subset metrics need exact source/version/split/checkpoint and preprocessing provenance reviewed and reproduced. Even a reproduced benchmark result would need independent, leakage-controlled, generator/source-aware tests to support generalization claims.

## 8. Proposed scientific evaluation plan

1. Define task labels separately: AI-generated vs camera-origin; manipulation/editing; claim relevance/truth; provenance match; observed spread outcome.
2. Acquire datasets under documented licenses/permissions and record source, capture/generator, event, time and label rationale.
3. Use at least two independent annotators for ambiguous labels and adjudicate disagreements; preserve exclusions with reasons.
4. Group duplicates and related event/source/generator material before split to reduce leakage. Hash exact duplicates and review perceptual-hash candidates.
5. Freeze model/checkpoint, preprocessing, threshold policy and evaluation plan before test-set use. Select thresholds only on validation data.
6. Report denominators, per-class counts, confusion matrix, precision, recall, specificity, F1, ROC-AUC and PR-AUC where appropriate; report coverage/abstention separately. Include group-level bootstrap intervals when enough independent groups exist.
7. Evaluate OCR with language/script-stratified CER/WER; claim extraction with span/type precision-recall-F1; retrieval with independently judged P@K/MRR and verdict-attribution agreement.
8. Evaluate Grad-CAM with localization/sanity checks and controlled human review, not aesthetic examples alone.
9. Compare fusion ablations only on identical labeled samples and protocol; include majority baseline and compatible detector baseline if licenses/code permit.
10. For virality/regional outcomes, collect permitted time-stamped propagation data with explicit outcomes, geographic definitions and privacy/ethics review. Do not train/publish predictive claims from synthetic cues.
11. Run a consent-based usability study with independent tasks and disclose conflicts, uncertainty and false-positive harms.
12. Archive manifest version, code revision, checkpoint digest, dependency versions, hardware, seeds, command, JSON output and exclusions without storing secrets or personal images unnecessarily.

## 9. Responsible research and publication readiness

Risks documented in the repository include false accusation, false reassurance, overreliance, stale/mismatched fact-check reviews, OCR language disparity, political false alarms, privacy/retention risks and over-interpretation of regional/social signals. The app should be presented as human-review support, not used for high-impact determinations. No fairness/bias-free claim or human-subject usability claim is supported.

The current paper draft is transparent prototype documentation, not a completed efficacy study. Before publication, confirm author/institution metadata; verify all literature references; review backbone/head licenses and training-data provenance; obtain valid data permissions; complete independent experiments and statistical analysis; report negative/indeterminate results; and obtain any necessary ethics review for annotators or user studies.

## 10. References available in project documentation

Use the canonical full citations/URLs already provided in `docs/phase6_final_research_report.md` and verify them before submission. The core references there are Wang et al. (CVPR 2020), Ojha et al. (CVPR 2023), Zhu et al. (GenImage, NeurIPS Datasets and Benchmarks 2023), Tschannen et al. (SigLIP 2, 2025), Selvaraju et al. (Grad-CAM, ICCV 2017), and Vosoughi et al. (Science 2018). Additional datasets in repository inventories (FakeNewsNet, PHEME, MuMiN, GenImage and RRDataset) are candidate sources only and were not used in TruthLens metrics.
