# TruthLens Testing and Evaluation

## 1. Scope of this audit

This document records test and evaluation evidence already present in the checkout's reports and result artifacts. Tests were not rerun for this documentation task. Historical results are labeled by report/date; they should not be represented as fresh execution in the current session.

## 2. Existing test infrastructure

- Backend: pytest files under `backend/tests/` cover API, phases 1–6, stage-level behavior, service logic, calibration and integration paths. Test fixture includes `backend/tests/fixtures/test_photo.jpg`; some tests generate synthetic images in code.
- Frontend: Vite build and lint scripts are defined under `frontend/package.json`; historical build/lint outcomes are recorded in `docs/phase5_e2e_test_report.md`.
- Experiments: `experiments/dataset_audit.py`, `evaluate_authenticity.py`, calibration utilities, fusion controlled scenarios and metric utilities provide reproducibility/evaluation entry points.
- Dataset example: `experiments/manifests/authenticity.example.csv` is a header-only template, not a labeled dataset.

## 3. Documented automated test/build results

| Source/date | Command or check | Reported result | Interpretation and caveat |
|---|---|---|---|
| `docs/phase6_final_research_report.md` / Phase 6 | Full backend suite | 188 passed, 10 skipped | Software regression checks; model-dependent skips are not detector metrics. Historical result. |
| `docs/phase6_implementation_report.md` | Phase 6 targeted checks | Reports 8 passed after fusion correction; metric utility checks 3 passed | Rule-path and utility verification; not research efficacy. See source report for commands. |
| `docs/phase5_e2e_test_report.md` (2026-11-17) | Focused synchronous backend suite | 165 passed, 10 skipped, 2 deselected | Async tests were deselected due sandbox loop issue for that run. Historical environment-specific report. |
| Same Phase 5 report | Full backend test command | 184 passed, 10 skipped | API and service code paths, not accuracy. Historical outcome. |
| Same Phase 5 report | Frontend build and lint | Passed; CSS import-order and >500 kB bundle warnings noted for build | Build correctness, not visual/accessibility acceptance. |
| Same Phase 5 report | Local browser upload | 14/20 stages completed; report built in 40.4 seconds | Uncontrolled image of unknown origin; no ground truth and no visual report verification. Not performance benchmark or usability study. |

The Phase 6 report supersedes the Phase 5 count for the later backend full-suite run, but these numbers are not interchangeable measures of model quality.

## 4. Test-case inventory from source/report evidence

Actual test names and assertions live in `backend/tests/`; this inventory summarizes verified coverage areas without claiming every suggested scenario executed in this audit.

| Test group | Feature/path | Documented coverage | Actual-vs-proposed status |
|---|---|---|---|
| Phase 1/API | Upload type/signature/size/decode; analysis record and status/report reads | Covered by backend tests and documented API integration | Executed historically, not rerun now |
| Phase 2 | Image processing, metadata, hashes, quality and OCR service behavior | Unit/service behavior tested; OCR real runtime depends on package/model availability | Software tests do not establish OCR quality |
| Phase 3 | Claims, fact-check/provenance adapters, evidence/report structures | Service and mocked/no-credential paths are tested | External provider correctness/relevance not established |
| Phase 4 | Virality/regional/social-heuristic deterministic behavior | Phase 4 tests and controlled examples | No observed spread/harm dataset; no predictive evaluation |
| Phase 5 | Orchestration, partial failure, safe report, UI/API workflow | Upload-to-partial-report and image retrieval integration documented | Does not test every external provider or browser visual state |
| Phase 6 fusion | Completed/informational/unavailable evidence scenarios | Controlled scenarios and regressions documented; non-completed evidence excluded | Deterministic rule behavior only; no accuracy benefit shown |
| Authenticity model | Optional real-model smoke tests | 10 model tests skipped in reports unless opt-in/configured | No approved labeled local test set; no performance validation |

### Suggested tests not represented as completed evidence

The following should remain proposed unless a future execution log is supplied: valid/invalid upload matrix against a release build; browser end-to-end tests with controlled images; API authorization tests after auth exists; durable worker restart/retry tests; security penetration tests; load/resource tests; cross-browser/accessibility tests; external provider live-contract tests; real labeled model/OCR/claim/user studies.

## 5. Model evaluation result

The local, manifest-driven authenticity evaluation ran against the header-only reviewed-manifest template and returned:

| Item | Result |
|---|---|
| Evaluation state | `NOT_EVALUABLE` |
| Eligible examples | 0 |
| Model inference in evaluation run | None |
| Accuracy/precision/recall/F1/AUC/calibration/robustness | Not produced |
| Machine result | `experiments/results/authenticity_test.json` |

The separate test-data audit found 35 readable originals under `test_data/originals`, without label/license/provenance metadata; duplicates/derived files prevent treating the directory's file count as independent labeled samples.

### Branch-reported metric: keep separate

`docs/phase6_authenticity_evaluation.md` records an upstream `ml/models` branch report for a 200-image CIFAKE subset (100/class): accuracy 0.87, AI recall 0.83, ROC-AUC 0.9613. It was not reproduced from a reviewed local manifest in this checkout, is not an independent benchmark, and does not establish out-of-distribution or real-world generalization. If referenced, attribute it explicitly to that branch report and state these caveats.

## 6. Dataset/readiness findings

- No reviewed labeled data manifest or independent train/validation/test split exists locally.
- No source/license/permission record, annotation guide, source/event/generator group ID or adjudication data exists for test originals.
- `test_data/` contains originals and derived sizes/heatmaps, not 190 unique independent items.
- Runtime `data/` contains user/application content and is intentionally excluded.
- External research datasets listed in docs are candidates only; they are not installed/used in reported metrics.
- Synthetic images in unit tests exercise code paths only and have no ground-truth semantics for detector performance.

## 7. Performance and usability evidence

Phase 5's performance report records narrow synthetic image microbenchmarks for preprocessing/quality/forensics and older-model timing. Those were not measured on the current fine-tuned model and are not representative end-to-end service SLAs. One browser run completed 14/20 stages in 40.4 seconds on an uncontrolled sample. There is no repeated load/performance study, resource envelope, confidence interval or target-hardware benchmark.

No user study, SUS score, completion/error rate or trust/overreliance measurement is documented. A usability protocol exists, but protocol is not results.

## 8. Defects and risk observations from records

Phase 6 controlled fusion tests exposed and corrected two logic inconsistencies: unavailable/non-completed evidence could vote, and forensic evidence was treated directionally despite its informational limitation. Regression tests were documented after correction. Potential remaining error categories in `phase6_error_analysis.md` are risks, not measured rates: model domain shift, image transforms, OCR failure, limited multilingual cues, provider mismatch/outage, source ambiguity, missing metadata, and over-interpretation of heuristics.

## 9. Recommended evaluation protocol

1. Acquire permitted, diverse, labeled samples and document label instructions and source rights.
2. Group duplicates, sources, events and generators before splitting to avoid leakage.
3. Freeze exact checkpoint digest, backbone revision, preprocessing, thresholds and code commit.
4. Use validation for any threshold/calibration choices; keep test data held out.
5. Report per-class counts/confusion matrix/precision/recall/specificity/F1/ROC-AUC/PR-AUC, coverage and uncertainty separately, and group-bootstrap intervals when sufficient groups exist.
6. Evaluate OCR with CER/WER by language/script and image quality; extraction with span metrics; retrieval with human-rated P@K/MRR; explanation with localization/sanity protocols.
7. Compare fusion only on paired identical records and keep forensic/context channels distinct from authenticity direction unless evidence supports directional use.
8. For spread prediction, collect permitted observed outcomes and define time windows, geography and ground-truth reach; validate prospective splits. Synthetic cues are not a substitute.
9. Conduct consented user testing for whether the report improves decisions or causes overreliance.
10. Preserve run manifests, source revision, dependencies, hardware, seeds, exclusions and JSON outputs; do not store secrets in artifacts.

## 10. Reproduction references

See `docs/phase6_experiment_setup.md` for recorded commands and protocol controls; `experiments/README.md` for script usage; `docs/phase6_experiment_dataset_inventory.md` for audit details; and `docs/phase6_authenticity_evaluation.md` for interpretation of the current detector result.
