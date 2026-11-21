# TruthLens Phase 3 Implementation Report

## Repository audit and baseline

| Component | Existing implementation | Actual gap | Change made |
|---|---|---|---|
| OCR and text normalization | EasyOCR with optional Tesseract fallback; whitespace/NFC normalization; language value is a script estimate. | Claim candidates did not retain structured language/OCR references; translation is not implemented. | Pass original normalized OCR sentences, language estimate and mean OCR confidence into extraction. Keep `translated_text: null`; no fabricated translation. |
| Claim extraction | `backend/app/services/claims/extractor.py`, rule-based and dependency-free. | Weak confidence heuristics treated many text fragments as facts; no stable IDs, claim categories, eligibility, ambiguity or duplicate handling. | Reworked existing extractor to preserve original text, stable IDs, factual/opinion/question/slogan/ambiguous classification, eligibility, entity/date/location/number hints, OCR confidence, source reference, Hindi cues, deduplication and limits. |
| Fact-check retrieval | `backend/app/services/factcheck/service.py` uses Google Fact Check Tools API. | Any OCR text could be queried as fallback; results lacked claim/evidence IDs, safe URL validation, per-claim outcomes and distinction between failed retrieval and successful zero results. | Only eligible claims are sent. Added normalized review metadata, HTTPS URL validation, bounded retrieval and conservative claim-level verdicts tied to returned review IDs. |
| Evidence assessment | Fact-check reviews already entered the evidence collection as informational items. | They were not associated with claim-verification outputs, and search result relevance was not assessed. | Match reviews conservatively to the originating claim using exact normalized text or strong lexical overlap plus entity/number anchors. Unmatched results cannot decide the verdict. Preserve each review's evidence ID; keep direction informational for image-authenticity fusion. |
| Image provenance | Google Cloud Vision Web Detection, with matching pages/images and labels. | URLs and malformed provider shapes were weakly validated; publisher and retrieval timestamps were missing; no C2PA parser exists. | Validate provider response shape and HTTPS result URLs; include provider, retrieval time and page hostname. Expose Content Credentials as unavailable because no parser is configured. |
| Hashing | SHA-256 and perceptual hashes were already computed. | The report UI did not explain exact-byte versus visual-similarity meaning. | Preserve algorithms and add a UI panel with limits. No hash is treated as proof of authenticity or originality. |
| Orchestrator and report | Existing pipeline already had claim, fact-check, provenance, evidence and report stages/fields. | Raw OCR fallback bypassed eligibility; claim verification was not returned. | Reused the same orchestrator. No eligible claims means fact-check lookup is skipped and ineligible text is marked unverifiable. Results reach the existing report schema. |
| Frontend | Existing fact-check and provenance panels displayed provider results. | No extracted-claim or claim-verdict panel; links lacked HTTPS checks; no hash display. | Added OCR Claims, claim-level verification summaries, safe source links, result match qualification, provenance result links and hash explanations. |
| Configuration and tests | Existing API key and timeout settings; Phase 2 tests covered connectors. | `.env.example` used a key name the code did not read and lacked retrieval bounds. | Corrected the example key and added claim/result/time/retry limits. Added offline Phase 3 tests. |

Baseline, before Phase 3 code changes: `python -m pytest --noconftest backend/tests/test_phase2.py backend/tests/test_stage4.py -q` — **56 passed in 15.63s**.

## Implemented data shapes

### Claim extraction

`claims` now contains `status`, `claims`, `total_claims`, `runtime_ms`, `method`, and `warnings`. Each claim contains:

- Stable `claim_id`; original OCR wording in `claim` and `original_text`; normalized text; original `language`; `translated_text` (null because translation is not implemented).
- `classification`: `FACTUAL`, `OPINION`, `QUESTION`, `SLOGAN`, `QUOTE`, or `AMBIGUOUS`; attributed quotes may be eligible, while unattributed quotes are preserved but not searched. Legacy `claim_type` is retained for existing consumers.
- `verification_eligibility`: `ELIGIBLE` or `INELIGIBLE`, with `eligibility_reason`.
- Heuristic entity, date, location and number strings; source sentence reference; OCR confidence; warnings.
- `confidence` is explicitly a candidate-extraction heuristic, not truth probability.

The extractor is rule-based and does not generate text. It supports English cues and a limited set of Hindi cues; OCR's language field is a script estimate, not robust language identification.

### Retrieval and claim verification

`fact_checks` retains `status`, provider `claims`, counts, runtime, warnings and note, and now includes `claim_verifications`. Each returned provider review carries claim ID, stable evidence ID, original query, reviewed statement, publisher/title/URL, review date/rating/language, retrieval timestamp, source type, and a `claim_match` label.

Each `claim_verifications` item has `claim_id`, `original_claim`, `verdict`, `reasoning_summary`, supporting/contradicting/conflicting evidence IDs, `verification_status`, null confidence and limitations. The verdict is an aggregation of matching third-party rating labels, not an independent TruthLens investigation. Conflicting ratings yield `MIXED`; unknown labels or no matching review yield `INSUFFICIENT_EVIDENCE`; unavailable/failed retrieval yields `RETRIEVAL_UNAVAILABLE`. Non-factual/ineligible candidates are `UNVERIFIABLE` and never queried.

### Provenance

`provenance` retains the existing Google Vision Web Detection fields and adds `provider`, `retrieved_at`, publisher hostname and per-result retrieval timestamps. `content_credentials.status` is `unavailable` with an explanation because the project has no C2PA/Content Credentials parser. The provider does not return a proven original source or reliable first-seen date; the report makes no such claim.

## Integration and source-of-truth boundaries

The orchestrator passes OCR sentences, script estimate and confidence to the existing extractor. It sends only eligible candidates to Google Fact Check Tools, tags each lookup with the stable claim ID, then attaches returned review IDs to the claim verdict and normalized evidence. Google Vision Web Detection and existing SHA-256/pHash run in their existing stages.

Fact-check and provenance evidence remain informational in the existing image-authenticity fusion engine. Claim accuracy and image-pixel authenticity therefore remain distinct. Missing or failed fact-check retrieval does not add an authenticity score or imply a false caption.

## Files changed for Phase 3

- `backend/app/services/claims/extractor.py`
- `backend/app/services/factcheck/service.py`
- `backend/app/services/provenance/service.py`
- `backend/app/services/evidence/schema.py`
- `backend/app/services/orchestrator.py`
- `backend/app/config.py`
- `.env.example`
- `frontend/src/App.tsx`
- `backend/tests/test_phase3.py`
- `docs/phase3_implementation_report.md`

## Provider status and limitations

- Google Fact Check Tools is implemented and requires `FACTCHECK_API_KEY` or `GOOGLE_API_KEY`; requests use a timeout and default to zero retries. `FACTCHECK_RETRY_ATTEMPTS` may enable at most two retries for transient errors. No live provider credentials were configured during this run, so live fact-check retrieval was **not** verified.
- Google Cloud Vision Web Detection is implemented and requires `GOOGLE_API_KEY`; no live provenance call was made because the key was not configured. Returned matches are online appearances/context, not proof of original source or manipulation.
- No general web search connector, translation provider, C2PA parser, or independent LLM verifier exists in the repository, and none was invented here.
- Claim extraction and claim/review relevance are deterministic heuristics. They can miss nuanced claims and should be reviewed, particularly for Hindi OCR, mixed-language text, entity aliasing, and old or time-sensitive claims.
- Fact Check Tools returns prior third-party reviews. The system does not independently verify a publisher's work or fetch and analyze full source pages. Unknown rating labels abstain.
- Requests are sequential with bounded claims, results, per-request timeout, total time, and optional bounded retries. No arbitrary OCR URLs are fetched. External result links are restricted to HTTPS in the UI.
- Existing hash semantics: SHA-256 fingerprints the exact uploaded bytes; pHash is a visual near-duplicate hint. Neither establishes authenticity.

## Verification and actual output

- Baseline before edits: **56 passed**.
- After changes: `python -m pytest --noconftest backend/tests/test_phase3.py backend/tests/test_phase2.py backend/tests/test_stage4.py -q` — **68 passed** (12 Phase 3 tests plus 56 adjacent regression tests).
- Python `compileall` for changed backend modules — passed.
- Frontend TypeScript check (`tsc --noEmit -p tsconfig.app.json`) — passed.
- Frontend production Vite build — passed. Existing CSS `@import` ordering and large JavaScript chunk warnings remain.
- `git diff --check` — passed (only Git line-ending conversion warnings).
- Actual local extractor example: the sentence `Government announced 120 new clinics in Delhi in 2025.` produced candidate ID `claim_98bbfaa8aa7b8b85`, classification `FACTUAL`, eligibility `ELIGIBLE`, number hints `120` and `2025`, and location hint `Delhi`. An opinion example produced `OPINION` and `INELIGIBLE`.
- Actual local provider configuration check found Fact Check Tools and Vision Web Detection **not configured**. Calling the fact-check service therefore returned `unavailable`, `RETRIEVAL_UNAVAILABLE`, null confidence and no evidence IDs. This is a real local output, not a mocked success or source verdict.
- Mocked tests cover supported/contradicted/mixed/insufficient/unavailable outcomes, ID linkage, no-results versus unavailable, irrelevant results, evidence fusion isolation, HTTPS provenance links, multilingual OCR text preservation and report assembly. They do not demonstrate real-world fact-check accuracy.
- A broader legacy test command that included Phase 1/Stage 3/Stage 5 did not finish or produce a summary after about one minute and was stopped; the deterministic adjacent suites above were run separately.
