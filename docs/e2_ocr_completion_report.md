# E2 OCR Completion and Reliability Report

## Scope and baseline

E2 completes the existing integrated OCR path and its failure handling. Work was performed on branch `fix/truthlens-remediation`, starting at `6e188e0`. The worktree was clean at the start of E2. The ignored local `models/ocr/` directory and its weights were preserved. No commit or push was made.

Changes are limited to OCR service/configuration, the OCR orchestration timeout, OCR and API regression tests, and the E0 module status register. No production SigLIP checkpoint, calibration, identity contract, inference routing, fusion policy, or research artifact was modified. E3 language identification and E5 standalone OCR endpoint work were not started.

## Implementation

- EasyOCR remains the cache-only primary engine (`download_enabled=False`). A Tesseract fallback is attempted when EasyOCR raises or completes without usable text; fallback output replaces the primary result rather than being merged with it.
- Empty readable output is a successful `completed` result with `has_text=false` and a user-safe `no_text_reason`. OCR inference failure is `failed`; absence of all usable engines is `unavailable`. Malformed region data can yield `partial` while preserving usable text.
- Image decoding, format, dimensions, OCR-local RGB conversion, region geometry, and confidence values are checked. Boxes are clipped to the input coordinate frame; invalid confidence does not discard otherwise usable text. Engine confidence remains uncalibrated.
- The orchestrator bounds the OCR wait using `OCR_TIMEOUT_SECONDS` (default 60 seconds), reports timeout as a failed E1 stage with `STAGE_TIMEOUT`, and keeps exception details out of public results. A timed-out worker thread cannot be forcibly stopped by Python and may continue until native inference returns.
- OCR text continues through the existing normalization pipeline. The existing OCR-derived script estimate remains a heuristic; no generic language-identification capability was added.

## Runtime evidence

The configured environment has EasyOCR 1.7.2, PyTesseract 0.3.13, and Torch 2.7.1+cu118 importable. The app's EasyOCR model-storage directory is `D:\Projects\Truthlens\models\ocr`, selected by `OCR_MODEL_DIR`. At the E3 inspection, it contained the detector file `craft_mlt_25k.pth` and the recognition file `devanagari.pth`; `english_g2.pth` and `latin_g2.pth` were absent there. EasyOCR 1.7.2's reader selection logic selects recognition file `devanagari.pth` for both `en,hi` and `hi`, while `craft_mlt_25k.pth` is the separate CRAFT text-detection model. The E2 smoke record did not log the resolved recognition pathname, so that exact historical file consumption could not be independently confirmed; the filename above is confirmed from the recorded language configuration, current configured storage, and installed EasyOCR 1.7.2 selection logic. The default user cache `C:\Users\DELL\.EasyOCR\model` also exists and contains `english_g2.pth`, but the app passes the explicit `OCR_MODEL_DIR` and does not select from that directory. The system Tesseract executable was absent from `PATH`, so Tesseract fallback could not be exercised against a real executable and remains mock-verified.

Cache-only EasyOCR inference was run on `backend/tests/fixtures/test_photo.jpg` (640 x 480):

| Configured languages | Result | Detected regions | Mean engine confidence |
|---|---|---:|---:|
| `en,hi` | completed, readable text | 7 | ~0.033327 |
| `hi` | completed, readable text | 8 | ~0.002894 |

Both real runs completed without warnings. The fixture has no ground-truth transcript, so these runs establish runtime readiness only, not transcription accuracy. Confidence values are not calibrated probabilities.

## Validation

- Focused E2/OCR/E1 regression command: `python -m pytest backend/tests/test_ocr_runtime.py backend/tests/test_phase2.py::test_ocr_preserves_text_and_structured_confidence backend/tests/test_phase2.py::test_ocr_empty_invalid_unavailable_and_fallback backend/tests/test_api.py::test_ocr_text_is_passed_to_text_normalization backend/tests/test_api.py::test_upload_to_partial_report_and_image_retrieval backend/tests/test_stage_contract.py backend/tests/test_phase1.py -q` - **58 passed, 2 skipped**.
- Safe backend suite: `python -m pytest backend/tests -q --ignore=backend/tests/test_phase2_dataset_splits.py --ignore=backend/tests/test_phase3_evaluation.py --ignore=backend/tests/test_phase4_and_phase3_5.py --ignore=backend/tests/test_phase5_training.py --ignore=backend/tests/test_phase6_evaluation.py --ignore=backend/tests/test_phase7.py` - **262 passed, 12 skipped, 9 warnings**. The skips include opt-in OCR/runtime cases and existing optional tests. Warnings were the pytest-asyncio fixture-loop-scope deprecation and Pillow `mode` deprecations emitted through EasyOCR.
- Protected evaluation/research test files were intentionally excluded: the E2 repository constraints prohibit access to locked-test and calibration research artifacts. Therefore the entire backend suite was not run.
- `python -m compileall -q backend/app/services/ocr/service.py backend/app/services/orchestrator.py backend/app/config.py` — passed.
- `git diff --check` — passed (no whitespace errors).

## Remaining work

Verify Tesseract fallback on a host with the executable and configured language packs, and measure OCR accuracy on permitted labeled samples. A process-isolated worker would be needed for hard cancellation of native OCR work. E5 owns a standalone `/ocr` endpoint; E3 owns dedicated language identification. The module register records these limits and retains OCR as `PARTIALLY IMPLEMENTED`.
