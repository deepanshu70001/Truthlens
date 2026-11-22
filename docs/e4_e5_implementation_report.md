# E4/E5 Implementation Report

## Delivered

- Added abbreviation-aware normalized sentence segmentation with source offsets.
- Added claim provenance fields, extraction method/version, exact-safe normalized deduplication, and explicit E3 language-detection status.
- Connected claim extraction to normalized text while preserving E2 OCR and E3 optional-language behavior.
- Removed the implicit English default from Fact Check Tools requests; provider language filtering is optional and validated.
- Added request bounds and per-call deduplication, provider error categories, bounded retries/timeouts, and preservation of successful queries when other queries fail.
- Implemented `POST /api/v1/fact-check` with bounded request/response schemas.
- Kept legacy publisher rating summaries compatible and labeled them as third-party ratings. Fact-check evidence remains informational and outside fusion voting.
- Added focused E4/E5 regression tests and this contract/report.

## Initial implementation validation (historical)

Validation results:

- Focused E4/E5, Phase 3, API, OCR runtime, and E3 language tests: **85 passed, 2 skipped**; the final E4/E5 plus Phase 3 rerun after provider-identity serialization: **19 passed**.
- Safe backend suite (`backend/tests`, excluding the six designated protected evaluation/research files): **292 passed, 12 skipped, 9 warnings**.
- `python -m compileall -q backend/app`: passed.
- `git diff --check`: passed.

Protected evaluation/research tests and artifacts were not accessed. No production detector checkpoint, calibration, identity contract, fusion policy, or authenticity inference routing was changed.

## Known limits at initial implementation validation (historical)

See [the stage contract](e4_e5_claim_factcheck_contract.md). At that earlier validation point, live Google credentials were not used and the local E3 checkpoint was unavailable. E3 was subsequently provisioned and verified as documented in `docs/e3_model_verification.md`.

## Targeted remediation verification (2026-11-22)

### Changes

- Claim payloads now retain optional E3 provenance: language code/name, source, model ID/revision, confidence, and detection outcome. The code/name/model/revision/confidence fields are copied only from an actual completed successful E3 result with a supported language code. Other outcomes remain `unknown`; missing revisions and confidence remain null. The existing `language` and `language_detection_status` compatibility fields remain.
- Claim span and OCR region inputs are validated before use. Malformed records are ignored with a warning. Offsets continue to refer to normalized OCR text. Sentence splitting now permits selected abbreviations such as `etc.` to close a sentence before a capitalized next sentence; this is a bounded heuristic, not a linguistic parser.
- The standalone fact-check endpoint reuses the E4 extractor/classifier, sends only eligible normalized claims, and returns explicit `SKIPPED`/`INELIGIBLE` records and reasons for other inputs. If no claim is eligible, it reports `skipped` and does not call the provider. Request strings and language codes are bounded and language codes must be among E3's supported codes.
- The provider adapter streams response bytes and enforces configurable `FACTCHECK_MAX_RESPONSE_BYTES` (default 1,000,000 bytes; effective service bounds 1 KiB–10 MB) before JSON parsing. Nested records and text fields are validated; malformed items are skipped when valid items remain and yield a `partial` result, while wholly unusable/invalid payloads fail safely.
- Review URLs still require HTTPS. Credential-bearing and common tracking query parameters are removed; legitimate navigation parameters are retained and fragments are dropped. URL values are not logged.
- No production authenticity, SigLIP, calibration, routing, or fusion policy was changed.

### Fresh validation

Focused command (E4/E5, Phase 3, Stage 4, API, E3, OCR runtime, E1 stage contract, E11): **191 passed, 0 failed, 2 skipped**, exit 0. Safe backend command with the six protected-file exclusions: **379 passed, 0 failed, 12 skipped, 6 warnings**, exit 0. Warnings are Python 3.12 `aiosqlite` datetime-adapter deprecations.

Focused command:

```text
python -m pytest backend/tests/test_e4_e5_contract.py backend/tests/test_phase3.py backend/tests/test_stage4.py backend/tests/test_api.py backend/tests/test_language_detector.py backend/tests/test_ocr_runtime.py backend/tests/test_stage_contract.py backend/tests/test_e11_fusion.py -q
```

Safe backend command:

```text
python -m pytest backend/tests -q --ignore=backend/tests/test_phase2_dataset_splits.py --ignore=backend/tests/test_phase3_evaluation.py --ignore=backend/tests/test_phase4_and_phase3_5.py --ignore=backend/tests/test_phase5_training.py --ignore=backend/tests/test_phase6_evaluation.py --ignore=backend/tests/test_phase7.py
```

Compilation command:

```text
python -m compileall -q backend/app/services/claims/extractor.py backend/app/services/text/normalizer.py backend/app/services/factcheck/service.py backend/app/api/v1/endpoints/stubs.py backend/app/services/orchestrator.py backend/app/config.py backend/tests/test_e4_e5_contract.py backend/tests/test_phase3.py backend/tests/test_api.py
```

`git diff --check` exited 0.

Coverage includes mocked provider responses only. No live provider request or credential-backed verification was performed. E4 remains a heuristic extraction system and these tests are contract/regression evidence, not a linguistic or accuracy evaluation.

### Current statuses

- **E4: PARTIALLY VERIFIED.** Claim-level language provenance, normalized offsets, malformed-span handling, eligibility, and optional-stage behavior are covered. Sentence boundary and eligibility rules remain heuristic and are not validated as general language understanding.
- **E5: PARTIALLY VERIFIED.** Standalone eligibility enforcement, bounded streaming responses, nested validation, URL sanitization, retries, status semantics, and fusion isolation are covered with unit/API mocks. Live Google provider operation remains unverified.
