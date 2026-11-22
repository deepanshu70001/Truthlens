# E1 failure-handling implementation report

## Delivered

- Added a shared normalizer for service outcomes and the canonical stage execution statuses.
- Expanded stage reports with outcome, safe error message, timestamps/runtime, stable IDs, correlation ID, warnings/limitations, report output reference, and allowlisted module identity.
- Persisted complete stage-status progress while preserving old JSON-list reads and keeping `completed_steps` success-only.
- Normalized service `success`, `no_results`, `no_text`, and `not_applicable` without turning them into stage failures.
- Preserved `partial`, `failed`, `unavailable`, and `skipped` service statuses. Missing imports map to unavailable; provider timeouts and other raised exceptions map to failed.
- Made terminal analysis failure recovery roll back failed transactions, persist a safe failure report, and retry once with a minimal report after a report/commit failure.
- Kept database status enum and API lifecycle compatible; report-derived API status remains `partial` when optional stages degrade.
- Marked fact-check/provenance failure and unavailable evidence records correctly so fusion ignores them.
- Added focused stage-contract failure injection tests and taught the existing progress tracker to display standardized statuses without redesigning it.

## Research and production safeguards

No detector, checkpoint, calibration artifact, identity contract, inference route, threshold, or fusion weight was edited. No research data or locked-test artifact was opened. No training, calibration, detector evaluation, or E2 OCR feature work was performed. Existing uncommitted OCR Phase 1 files were preserved.

## Validation

Validation completed:

| Command | Result |
|---|---|
| `python -m pytest backend/tests/test_stage_contract.py -q` | **16 passed** |
| Focused contract + OCR runtime + Phase 1/API + OCR/fact-check/fusion regressions | **73 passed, 2 skipped** (opt-in real OCR cases) |
| Safe backend suite with research-artifact tests excluded | **249 passed, 12 skipped, 9 warnings** |
| `python -m compileall -q backend/app` | Passed; no output |
| `git diff --check` | Passed; no output |
| E1 documentation whitespace/local-link check | Passed |
| `npm run build` | Failed on pre-existing ignored local `frontend/src/lib` files: missing `vitest`, missing `shared/india_regions.json`, one implicit `any`, and missing exported `AnalysisReport`. Those files were left unchanged. |

The safe backend suite excluded `test_phase2_dataset_splits.py`, `test_phase3_evaluation.py`, `test_phase4_and_phase3_5.py`, `test_phase5_training.py`, `test_phase6_evaluation.py`, and `test_phase7.py`: their test code reads split/training manifests, calibration records, held-out evaluation reports, or prediction CSVs. The full backend suite was therefore not run because it would violate the explicit prohibition on accessing locked-test/calibration research artifacts. The nine warnings were Pillow deprecation warnings from EasyOCR during the existing full-pipeline API integration test; pytest also emitted its `pytest-asyncio` fixture-loop-scope deprecation notice.

## Residuals

- If the database remains unavailable through both terminal persistence attempts, the failure is logged but cannot be made durable without an external durable queue/recovery mechanism.
- The database enum intentionally does not gain a `partial` member; public `partial` status is derived from the report to avoid an unplanned migration.
- E10 still needs browser-level regression tests for polling and partial/terminal progress states.
