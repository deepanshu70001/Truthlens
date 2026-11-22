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

## Validation

Validation results:

- Focused E4/E5, Phase 3, API, OCR runtime, and E3 language tests: **85 passed, 2 skipped**; the final E4/E5 plus Phase 3 rerun after provider-identity serialization: **19 passed**.
- Safe backend suite (`backend/tests`, excluding the six designated protected evaluation/research files): **292 passed, 12 skipped, 9 warnings**.
- `python -m compileall -q backend/app`: passed.
- `git diff --check`: passed.

Protected evaluation/research tests and artifacts were not accessed. No production detector checkpoint, calibration, identity contract, fusion policy, or authenticity inference routing was changed.

## Known limits

See [the stage contract](e4_e5_claim_factcheck_contract.md). In particular, live Google credentials were not used and the local E3 model checkpoint remains unavailable. Existing frontend build failures are tracked in the prior E3 report; this milestone made only the small compatibility label/order update.
