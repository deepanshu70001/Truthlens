# OCR and Fact-Checking Implementation Audit

**Audit date:** 2026-11-22

**Branch:** `fix/truthlens-remediation`

**HEAD:** `8e7d5511e8a84737260e4d3ff935338feea1eb8b`

**Scope:** Read-only audit of backend/frontend implementation and targeted offline tests. No source, configuration, model, or data files were changed. No external provider calls or OCR model inference were performed.

## Repository state and method

The working tree was clean at the start and remained clean after testing, before adding this new report. Recent commits at audit start were `8e7d551` (Phase 10 blocker report), `ecc19c6` (Phase 9 findings), `f0c2877` (Phase 9 diagnostics), `fde9800` (Phase 9 forensic features), and `c180a43` (Phase 8.1 audit). This report is the only audit-task file added; nothing was committed.

Inspection was restricted to relevant `backend/app`, `backend/tests`, `frontend/src`, and documentation files. No training, calibration, or locked-test image data, manifests, predictions, or model evaluation outputs were opened. Secret values and `.env` contents were not printed or inspected. No external request was made.

## Executive assessment

OCR, deterministic claim extraction, Google Fact Check Tools retrieval, and report display are implemented in the **main asynchronous image-analysis pipeline**. This is not the same as all components being operationally verified: current checks confirm the EasyOCR Python package and `pytesseract` wrapper are installed, but do not verify EasyOCR inference or model availability. The Tesseract executable was not found, and fact-check credentials are not configured in the current runtime.

Language identification is not integrated as an ML inference stage. The OCR service assigns a dominant-script estimate, and that value flows to claims and the provider language parameter. The standalone OCR and fact-check API routes remain 501 placeholders. No real OCR inference or live provider query was performed in this audit.

## Component status

| Component | Code exists | Integrated into orchestrator | Tests | Actual runtime verified | Status |
|---|---|---|---|---|---|
| EasyOCR | Yes; lazy reader initialization and `readtext` | Yes; main OCR stage | Unit tests use a fake reader; opt-in real-model tests exist | Package present; inference/model cache not tested | IMPLEMENTED BUT NOT VERIFIED |
| Tesseract | Yes; `pytesseract.image_to_data` adapter | Yes; load fallback and inference-exception fallback | Fallback is mocked | Python wrapper present; system executable absent | BLOCKED |
| Preprocessing | Yes; upload preprocessing plus EXIF transpose/RGB conversion before OCR | Yes | General API tests and mocked pipeline coverage; no engine preprocessing test run | Upload code exists; OCR engine was not run | PARTIALLY IMPLEMENTED |
| Bounding boxes | Yes; EasyOCR quadrilateral and Tesseract rectangle conversion | Yes through OCR result serialization | Mocked box assertions | No real engine boxes checked | IMPLEMENTED BUT NOT VERIFIED |
| Language identification | Dominant Unicode-script heuristic in OCR; XLM-R classifier class exists | Script estimate is integrated; XLM-R detector is not called | No language detector tests found in inspected test files | No language model inference | PARTIALLY IMPLEMENTED |
| Claim extraction | Yes; deterministic rules, regex-based fields | Yes, after OCR text normalization | Unit tests, including rule behavior | Orchestrator path exists; not exercised with real OCR | IMPLEMENTED BUT NOT VERIFIED |
| Fact-check retrieval | Yes; Google Fact Check Tools API HTTP adapter | Yes, for eligible extracted claims | Offline tests mock settings and/or HTTP | No live request; configured key flags are false | IMPLEMENTED BUT NOT VERIFIED |
| Claim matching | Yes; lexical overlap plus number/name anchors | Used before attributing review ratings | Unit tests for exact/irrelevant examples | No live retrieval corpus checked | PARTIALLY IMPLEMENTED |
| Verdict attribution | Yes; maps publisher ratings to supported/contradicted/mixed/insufficient | Returned in report | Mocked rating tests, including conflicting ratings | No live provider response checked | IMPLEMENTED BUT NOT VERIFIED |
| OCR API | Route exists at `POST /api/v1/ocr` | Not applicable | No standalone OCR API test found | Route returns 501 by source | STUB / PLACEHOLDER |
| Fact-check API | Route exists at `POST /api/v1/fact-check` | Not applicable | No standalone fact-check API test found | Route returns 501 by source | STUB / PLACEHOLDER |
| Frontend integration | Report sections and upload/report polling exist | Consumes orchestrator report | No frontend test script or test files found in inspected frontend | UI was not launched | IMPLEMENTED BUT NOT VERIFIED |

## OCR implementation

### Code and execution behavior

- `backend/app/services/ocr/service.py`: `OCRService.load()` tries EasyOCR first with configured `OCR_LANGUAGES` and `OCR_MODEL_DIR`; if initialization raises, it attempts Tesseract. EasyOCR uses `Reader.readtext()` on an RGB NumPy array. Tesseract uses `image_to_data()` and configured language packs.
- EasyOCR and Tesseract are **not normally run in parallel or sequentially for every image**. EasyOCR is primary. Tesseract becomes the engine if EasyOCR cannot initialize, or if EasyOCR inference raises an exception. An empty successful EasyOCR result does **not** trigger Tesseract.
- Before OCR, `run_analysis_pipeline()` opens the saved upload, applies `ImageOps.exif_transpose()`, and converts to RGB. The OCR service itself does not resize, crop, deskew, threshold, or otherwise preprocess pixels. Upload-time thumbnail/resized products are not passed to `extract()`.
- Text normalization uses Unicode NFC plus whitespace collapse, preserving punctuation. OCR raw text is retained separately. OCR confidence is the engine-reported block confidence (EasyOCR `[0,1]`; Tesseract `[0,100]` scaled to `[0,1]`), then averaged arithmetically across recognized blocks with confidence values. It is not calibrated and is not weighted by word count or box area.
- EasyOCR returns quadrilateral points converted to integer coordinates; malformed, non-finite, out-of-bounds points, and confidence values outside `[0,1]` are discarded. Coordinates are in the decoded, EXIF-corrected image frame because no OCR resize is applied. Tesseract boxes are converted to four corners, but the code checks positive width/height without validating that the resulting coordinates lie inside the image.
- A successful empty engine output is `status="completed"`, `has_text=false`, with a no-readable-text explanation. Missing engines return `unavailable`; engine exceptions return `failed` unless the Tesseract retry succeeds. Thus empty output and unavailable dependencies are distinct in the service result.

### Current local readiness and verification boundary

- `easyocr` package: present according to `importlib.util.find_spec`. At this audit's time, weights, language initialization, and inference were not checked. OCR Phase 1 later disabled automatic weight downloads and performed the cache-only checks recorded in `ocr_runtime_verification.md`.
- `pytesseract` package: present.
- `tesseract` system executable: not found on PATH. Since there is no executable, language-pack availability could not be checked. Tesseract is not operationally verified in this environment.
- `OCR_LANGUAGES` is configured. The code default is English and Hindi (`en,hi`). No OCR reader was initialized, so no model download or image inference occurred.
- `backend/tests/test_stage3.py` contains real EasyOCR smoke tests guarded by `TRUTHLENS_REAL_MODEL_TESTS=1`; those tests load the model and a test fixture and were not run. This audit therefore makes no real OCR accuracy or runtime claim.

### API and frontend

`POST /api/v1/analyze` validates and stores the uploaded image, then creates a background task that calls `run_analysis_pipeline()`. The pipeline calls OCR, sends OCR text through normalization and claim extraction, sends eligible claims to fact-check retrieval, and stores the combined report. The frontend uploads to `/api/v1/analyze`, polls `/api/v1/analysis/{id}`, and retrieves `/api/v1/analysis/{id}/report`.

The report UI displays OCR status, selected engine, configured languages, script estimate, word count, mean confidence, text, and basic failure/no-text states. It does not display the returned OCR boxes. The standalone `POST /api/v1/ocr` handler in `backend/app/api/v1/endpoints/stubs.py` has no upload/request schema and returns HTTP 501. The integrated report path is the usable OCR interface.

## Language identification

`backend/app/services/ocr/service.py::_detect_primary_script()` counts Unicode character names and returns the most frequent recognized script among a fixed set (including Latin as `en` and Devanagari as `hi`). This is a script heuristic, not language identification. Latin-script Hindi transliteration is labeled English; mixed Hindi-English becomes whichever recognized script has the higher count; unknown scripts become `unknown`.

`backend/app/services/nlp/language_detector.py` defines an XLM-RoBERTa text classifier (`papluca/xlm-roberta-base-language-detection`) with a `load()` method that loads tokenizer and classifier through Transformers, potentially from the Hugging Face cache/network. `detect()` returns top language predictions, rejects fewer than three non-whitespace characters as too short, truncates to 512 characters, and emits one top language rather than a mixed-language result. However, no call to `LanguageDetector`, `get_language_detector()`, or `.detect()` exists in the inspected application pipeline. No schema/report field receives its output.

The orchestrator passes the OCR script estimate to the claim extractor and as the fact-check API `languageCode` when it is a two-character string; otherwise it falls back to English. There is no translation implementation. Claim records expose `translated_text`, but extraction leaves it `None`; UI uses the script-estimate label when present.

## Claim extraction

`backend/app/services/claims/extractor.py::extract_claims()` is deterministic rule/regex code, not an LLM or general-purpose NLP model. It operates on normalized OCR sentences, deduplicates normalized text, caps candidate count, and preserves non-eligible candidates. It has rules for factual-looking text, quotes, questions, opinions, slogans, and ambiguous fragments. It extracts digit-based numbers, a short fixed list of English date formats and place names, and capitalized Latin-script spans as heuristic entities. `confidence` is a hand-built ranking score mixed with OCR confidence; the serialized field explicitly says it is not a truth probability.

The orchestrator passes `text_result.sentences`, OCR script estimate, and aggregate OCR confidence. Only claims marked `ELIGIBLE` are sent to the fact-check service. Other extracted items are retained as `UNVERIFIABLE` in the result without retrieval.

Concrete limitations implied by the implementation (not measured error rates):

- Sentence splitting recognizes `.`, `!`, and `?`, but not Devanagari danda (`।`); adjacent Hindi statements can remain one candidate or be segmented incorrectly.
- Entity extraction only recognizes capitalized Latin-script spans; it is not NER and misses many names, transliterations, and non-Latin entities. Place extraction is a finite list.
- A numeric year or number can help trigger `FACTUAL` classification even when the proposition lacks enough context to search accurately; conversely, factual statements without the listed cue words/numbers can be marked ambiguous.
- The returned claim text is an OCR sentence candidate rather than a compact, context-resolved proposition. It does not resolve pronouns, event identity, source attribution, or duplicated claims across the image.
- No extraction confidence calibration or span-level OCR-to-claim alignment is implemented.

## Fact-checking

### Provider and result handling

`backend/app/services/factcheck/service.py::search_fact_checks()` makes a real HTTP request to Google Fact Check Tools `claims:search` through `httpx` when `GOOGLE_API_KEY` or `FACTCHECK_API_KEY` is configured. The key is passed as the API `key` query parameter. The current settings check reports both key flags as absent; this audit did not make a provider request. With no key, the service returns `status="unavailable"` and per-claim `RETRIEVAL_UNAVAILABLE`. A successful empty response becomes `no_results` / `INSUFFICIENT_EVIDENCE`, distinct from provider unavailable.

The request has a bounded timeout, total-time budget, configurable retry count (default 0), claim cap, and result cap. HTTP 403 returns a failed result with an invalid-credentials-or-quota message; HTTP 429 stops further requests and records a rate-limit warning; other HTTP/request/parsing errors are reduced to warnings and summarized as failed or partial. Error details are not copied into the public error string. There is no dedicated structured quota status, and with default retries a transient error is not retried.

The adapter preserves publisher name/site, URL, title, reviewed claim text, publisher textual rating, review date, claim date, language code, and retrieval timestamp. It rejects non-HTTPS or malformed review URLs. The UI renders publisher rating as a **third-party review**, shows publisher and review date, and includes a note that results reflect third-party reviews rather than TruthLens assessments. It does not display claim date or identify stale reviews. The service does not fetch or independently validate the linked article's full contents or explanation.

### Matching and verdicts

`_claim_match()` uses exact normalized equality or lexical token overlap (at least 0.70 of query tokens) plus a number or capitalized-name anchor and matching extracted anchors. It is not semantic entailment or event-level verification. Reviews that miss the heuristic are marked related/unconfirmed and are not used by `_assess_ratings()` to assign a verdict. This reduces some unrelated matches but can reject paraphrases or accept a high-overlap wrong-context review.

`_assess_ratings()` maps a fixed set of textual ratings to `SUPPORTED`, `CONTRADICTED`, `MIXED`, or `INSUFFICIENT_EVIDENCE`. Conflicting matched supported and contradicted ratings yield `MIXED`; unknown ratings abstain. These are summaries of publisher ratings, not independently established truth. The serialized limitations and UI note say so. There is no freshness policy for old reviews; conflicting ratings are surfaced as mixed only when both pass the lexical match gate.

The pipeline calls fact-check retrieval only when eligible claims exist. The standalone `POST /api/v1/fact-check` route has no request schema and returns HTTP 501; the usable path is through an uploaded-image analysis report. The report is returned as untyped JSON by `/api/v1/analysis/{id}/report`; `AnalysisOut` does not define nested OCR/claims/fact-check schemas.

## Tests and verification

Command run:

```text
python -m pytest backend/tests/test_phase2.py backend/tests/test_phase3.py -q
```

Result: **22 passed in 2.54s**. Pytest emitted a `pytest-asyncio` deprecation warning about the unset fixture loop scope.

These are offline unit tests, not live verification:

- Phase 2 OCR tests inject a fake EasyOCR reader and monkeypatch the Tesseract fallback; they cover serialization, empty/invalid results, unavailable state, and fallback control flow without invoking either OCR engine.
- Phase 3 claim/fact-check tests exercise regex classification and matching. Provider behavior is tested with patched settings, mocked `_search_single_claim`, or a mocked `httpx.Client`; no external provider was called.
- `backend/tests/conftest.py` also replaces OCR in API/background-pipeline fixtures with a mock service. Existing API integration tests therefore do not demonstrate real OCR inference.
- `backend/tests/test_stage3.py` has opt-in cache-backed real model smoke tests, but they were not run. No current recorded test evidence establishes EasyOCR/Tesseract OCR quality or latency.
- No language-detector tests were found in the inspected test files. The frontend `package.json` has build/lint scripts but no test script; frontend tests were not run.

## Actual execution path

```text
POST /api/v1/analyze (multipart image)
  -> validate/store upload and create analysis
  -> background run_analysis_pipeline
  -> EXIF transpose + RGB decode
  -> OCRService (EasyOCR primary; Tesseract only on load failure or EasyOCR exception)
  -> normalize_text(OCR text)
  -> extract_claims(normalized sentences)
  -> eligible claims only -> Google Fact Check Tools query, if configured
  -> report builder stores OCR, claims, third-party reviews/verifications
  -> GET /api/v1/analysis/{id}/report
  -> frontend renders OCR, claims and fact-check sections
```

The XLM-R language detector is absent from this path. Standalone `/api/v1/ocr` and `/api/v1/fact-check` do not enter it; both currently return 501.

## Ten highest-priority outstanding issues

1. **Tesseract fallback is unavailable in the audited environment.** The Python wrapper exists but the system executable does not; neither English nor Hindi Tesseract packs can be verified.
2. **EasyOCR runtime readiness is unknown.** The package is installed, but model files, initialization, actual extraction, and image-quality behavior were not verified. Deployment may need first-run model downloads.
3. **Standalone OCR and fact-check endpoints are placeholders.** Both return 501 and lack request/response schemas; API consumers must use the full analysis report.
4. **The XLM-R language detector is dead/unintegrated code.** The displayed language and query language code use script frequency, not language inference; mixed language is not represented.
5. **OCR fallback misses empty-result failures.** A successful empty EasyOCR result is treated as completed no-text and does not attempt Tesseract.
6. **Fact-check matching is lexical, not semantic.** The overlap/anchor gate can miss paraphrases and does not prove same event, time, location, or actor.
7. **Claim extraction is language- and syntax-limited.** Regex entity/date/place rules, limited factual cues, and missing Devanagari sentence punctuation limit Hindi and multilingual coverage.
8. **No freshness or outdated-review handling exists.** Review dates are preserved and shown, but results are not flagged stale; claim dates are not displayed.
9. **OCR confidence and geometry need stronger validation.** Confidence is an uncalibrated simple average; Tesseract boxes are not bounds-checked, and neither engine's real boxes were verified here.
10. **Runtime test coverage is incomplete.** No executed real OCR test, no language-detector tests, no live-provider test evidence, no full orchestrator test with actual OCR, and no frontend test suite were established by this audit.

## Recommended implementation sequence

1. Make readiness reporting check the Tesseract executable and configured language packs, EasyOCR language/model readiness, and configured provider state without exposing secrets. Document the deployment prerequisites.
2. Add a safe OCR runtime smoke-test path using a repository-owned synthetic image with known text and no implicit model download; test empty output, missing dependencies, inference errors, fallback behavior, coordinates, and confidence semantics.
3. Decide whether the product supports standalone OCR/fact-check calls. If yes, replace the 501 handlers with validated request/response schemas and add offline API integration tests; otherwise document the report-only contract clearly.
4. Integrate the language model behind explicit readiness/error states, test short and mixed-script text, and represent script estimates separately from inferred language. Do not claim translation unless a translation stage is added.
5. Improve claim segmentation and multilingual extraction with span provenance, locale-aware sentence splitting, broader entity/date/place extraction, and calibrated extraction confidence; retain the rule baseline for comparison.
6. Strengthen retrieval matching with context-aware claim/entity/date checks and expose `match_unconfirmed` distinctly from any attributed rating.
7. Add review freshness/conflict metadata and preserve the publisher attribution in every API/UI response. Keep unavailable, failed, no-results, and inconclusive states distinct.
8. Add frontend tests for OCR errors/no-text, claims, provider unavailable/failed/no-results, third-party attribution, and review dates.
9. After local deterministic tests pass, consider a separately authorized provider smoke test using non-sensitive synthetic text. Do not use user-uploaded images or claims for that test.

## Historical statements versus current verification

Repository documentation (`docs/system_architecture.md`, `docs/api_reference.md`, and `docs/deployment_guide.md`) describes OCR and fact-check adapters in the integrated report pipeline and explicitly documents the standalone 501 routes and optional dependencies. Source inspection confirms those code paths exist. The passing tests in this audit verify mocked/unit behavior only. They do **not** establish real OCR inference, successful Tesseract fallback, XLM-R language detection, external-provider availability, or end-to-end runtime behavior. No historical evaluation claim is treated as current runtime verification.
