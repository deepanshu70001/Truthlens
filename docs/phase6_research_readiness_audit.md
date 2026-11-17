# Phase 6 research-readiness audit

Audit date: 2026-11-17. Scope: current checkout, Phase 1–5 reports, code, tests, model/configuration files, data/test assets, and recorded runtime checks. Unit tests establish software behavior only; they do not establish detector accuracy.

| Research component | Existing implementation | Status | Available data | Existing evaluation | Missing work |
|---|---|---|---|---|---|
| Image authenticity | Configured SigLIP2 two-class checkpoint; operational AI/real thresholds 0.65/0.35; scores explicitly uncalibrated | IMPLEMENTED_BUT_NOT_EVALUATED | Checkpoint cache; model card declares Apache-2.0 and names the SigLIP2 base checkpoint; upstream training-data provenance still needs review. No labeled independent evaluation set | CPU smoke inference and software tests only | Provenance, license, real/AI labels, generator/event groups, held-out test, calibration and robustness study |
| Manipulation classification | Detector explicitly returns `manipulation_score: null` | NOT_IMPLEMENTED | None | None | Separate task/data/model if retained as a research objective |
| Classical forensics | ELA, JPEG/noise/edge/texture/resampling cues; scores are indicators, not probabilities | IMPLEMENTED_BUT_NOT_EVALUATED | No annotated manipulation masks or class labels | Unit tests and earlier synthetic timing microbenchmark | Labeled authentic/edited comparisons; JPEG-aware protocols; validated signal interpretation |
| Explainability | Conditional Grad-CAM for configured SigLIP checkpoint; artifacts and target class recorded | IMPLEMENTED_BUT_NOT_EVALUATED | No region masks or reviewer labels | Unit tests; no human study; actual checkpoint UI rendering not verified | Correctness/sanity checks, stability, localization annotations, user study |
| OCR | EasyOCR primary with opportunistic Tesseract fallback; configured `en,hi`; reports status/engine | PARTIALLY_IMPLEMENTED | No text-image reference corpus | Component tests; runtime lacked OCR packages in Phase 5 | Install/licensed benchmark, language-specific transcripts, CER/WER, actual engine and version capture |
| Translation | No translation component | NOT_IMPLEMENTED | None | None | Do not claim translation support; future provider/model and reference set if required |
| Language identification | Unicode-script estimate, not full language identification | DEMONSTRATION_ONLY | No independently labeled multilingual corpus | Unit behavior only | Language-specific annotated set; distinguish script from language |
| YOLO objects | Optional Ultralytics object detector with configurable checkpoint | PARTIALLY_IMPLEMENTED | Checkpoint/runtime readiness not confirmed in Phase 5 | Mocks/unit behavior; no licensed detector benchmark | Confirm exact weights/version/license; COCO-style labeled evaluation only if in scope |
| Claim extraction | Rule-based English cues and limited Hindi cues; eligibility separated from truth verdict | IMPLEMENTED_BUT_NOT_EVALUATED | No manually labeled claim spans | Phase 3 unit tests | Expert-annotated span/type set, adjudication, precision/recall |
| Fact-check retrieval/verdict | Optional Google Fact Check API, matching and traceable provider ratings | PARTIALLY_IMPLEMENTED | No reviewed claim/evidence benchmark; credentials absent in test run | Mock/service tests | Stable human-reviewed queries, retrieval metrics, separate provider unavailability and verdict agreement |
| Image provenance | Optional Google Vision Web Detection | PARTIALLY_IMPLEMENTED | No reference-source corpus; outbound credentials absent | Mock/service tests | Ground-truth provenance references and provider-specific evaluation; privacy review |
| C2PA credentials | Explicitly unavailable | NOT_IMPLEMENTED | None | None | No content-credential verification claim |
| Evidence fusion | Rule-based strength/direction weighted aggregation; duplicate suppression and conflict handling | IMPLEMENTED_BUT_NOT_EVALUATED | Controlled synthetic evidence scenarios only | Phase 3/5 tests; Phase 6 controlled logic checks | Labeled paired examples for predictive ablation; review correlated signals and weight rationale |
| Virality | Content cue heuristic; no observed reach prediction | HEURISTIC_ONLY | No historical engagement/cascade data | Contract tests and prior runtime microbenchmark | If claimed predictive, acquire permitted longitudinal outcome data and define horizon |
| Regional context | 36 administrative units; scores/ranks null when data unavailable | DATA_UNAVAILABLE | No regional spread outcome table | Contract tests | Validated outcome data and geographic alignment; avoid susceptibility inference |
| Social impact | Exact-token potential-harm cues; no measured harm outcome | HEURISTIC_ONLY | No independent harm labels/outcomes | Contract tests | Human-reviewed neutral/controversial controls and inter-rater protocol |
| Integrated API/report | FastAPI upload → background pipeline → structured report/artifact routes | IMPLEMENTED_AND_VERIFIED (API path) | API-generated fixtures only | Full backend suite: 184 passed, 10 skipped; includes upload/report/image retrieval | More scenario fault injection and retained-artifact browser verification |
| Frontend | React/TypeScript report and restore flow | PARTIALLY_IMPLEMENTED | No usability participants | Build/lint passed; browser check used uncontrolled preselected image and was not a usability evaluation | Controlled demonstration script, cross-browser verification, consented user study |
| Reliability/performance | Stage logs/timing and partial result status | IMPLEMENTED_BUT_NOT_EVALUATED | Synthetic microbenchmark; one uncontrolled local UI execution (not benchmark data) | Prior five-run component microbenchmark; one local pipeline completed 14/20 in 40.4 s on unknown preselected JPEG | Repeated controlled end-to-end runs, resource telemetry, failure rate, load profile |

## Data audit result

`experiments/dataset_audit.py` scanned only `test_data/originals` (not user upload storage): **35 readable image files**, JPEG 31 / PNG 4; exact duplicate extra copies 31; exact dHash candidate extra copies 33; dHash pairs within Hamming distance 2: **496 candidate pairs**. This is a deliberately loose candidate threshold for tiny/low-resolution fixtures, not an identity or exclusion count. There are no labels or license records. These files are not suitable for model metrics or presentation until provenance and rights are established. The broader `test_data/` tree has 190 image files, including processed/resized/thumbnail/heatmap artifacts; it is not a 190-example independent corpus. Aggregate machine output is in `experiments/results/test_originals_inventory.json`.

The repository also contains runtime storage under `data/` and ignored temporary artifacts. These are treated as user/application data and excluded from the research corpus and all experiments. The single image observed during the Phase 5 browser smoke attempt was submitted locally without an intentional evaluation protocol; it is excluded.

## Experiments genuinely possible now

- Audit file validity, dimensions, exact duplicate hashes and duplicate candidates in explicitly selected directories; this says nothing about source rights or labels.
- Re-run existing software tests and controlled fusion behavior checks using explicitly constructed synthetic evidence objects.
- Measure implementation latency on a documented synthetic fixture for code-path characterization only.
- Prepare evaluator scripts that enforce a labeled manifest and report `NOT_EVALUABLE` when none exists.
- Describe model/API architecture and current availability from code/configuration.

Not currently possible: independent authenticity accuracy/calibration/robustness, fair detector benchmark, OCR CER/WER, claim/evidence quality, quantitative Grad-CAM localization, predictive virality/regional/harm metrics, participant usability results, or a statistical end-to-end failure rate.

## Acceptance boundary

Phase 6 prepares a reproducible protocol and submission draft. It cannot establish publishable empirical claims without licensed data, documented labels, independent splits, actual experiments, and where needed independent reviewers/participants. The accompanying paper is a transparent prototype report, not a completed efficacy study.
