# E4/E5 Claim Extraction and Fact-Check Contract

## Claim extraction

The pipeline consumes normalized OCR text and preserves sentence offsets into that normalized source. Sentence segmentation avoids common title abbreviations, and each extracted record retains the original sentence, normalized query, deterministic claim ID, source reference, optional source offsets/region references, extraction method/version, language code and language-detection outcome. Deduplication collapses only whitespace/case variants; punctuation and numeric content remain intact. Candidate classification is a conservative heuristic. Its score is explicitly a ranking hint, never a probability that the statement is true.

Language code/name, source (`e3_language_identification`), model ID/revision, and confidence are copied only from a completed successful E3 result whose code is in the supported model label set. The model ID and revision come from that result, not a hardcoded revision. For unavailable, failed, insufficient, undetermined, missing, or unsupported results, claims remain `unknown`; provenance and confidence fields are null, with `language_detection_outcome` retaining the result outcome where available. The legacy `language` and `language_detection_status` fields remain for compatibility. OCR script estimates are not promoted to language identification. Extraction remains independent of the optional language model.

Offsets are Python string indices into the normalized OCR text after Unicode normalization, control filtering, and whitespace collapse. They are not raw OCR-buffer, image-byte, or pixel offsets. Malformed sentence spans and OCR region ranges/identifiers are skipped and produce a warning. Segmentation handles common title abbreviations and selected sentence-final abbreviations heuristically; it is not a complete linguistic parser.

## Fact-check lookup

Only claims marked `ELIGIBLE` are sent to the Google Fact Check Tools API. The pipeline filters E4 output; the standalone endpoint now runs every supplied claim through that same E4 extraction/eligibility policy and skips ineligible input with a per-claim reason. Valid standalone claims are whitespace-normalized, bounded to 2,000 characters, limited by request and configured query counts, and deduplicated case-insensitively. All-ineligible requests return `skipped` without a provider call. A standalone language filter must be a supported E3 code; the pipeline includes it only when E3 returned a successful supported prediction. Credentials remain server-side, and responses/logs contain safe provider error categories rather than credential-bearing exception text.

Provider responses are streamed and capped before parsing by `FACTCHECK_MAX_RESPONSE_BYTES` (default 1,000,000 bytes, clamped to 1 KiB–10 MB). Top-level and nested claim/review/publisher structures and text bounds are validated. Malformed individual entries are skipped when valid entries remain and cause a `partial` result; an invalid or wholly unusable response fails safely. HTTPS review URLs have credential-bearing parameters (`key`, `api_key`, `access_token`, `token`, `authorization`, `signature`, `sig`, and related fields) and common tracking parameters removed; valid navigation parameters are retained, fragments are dropped, and malformed/non-HTTPS URLs are rejected. Successful lookups are retained when another query fails. The service distinguishes successful empty results (`no_results`), unavailable credentials (`unavailable`), provider failures (`failed`) and mixed success/failure (`partial`). Missing eligible claims skip the pipeline stage with `not_applicable`.

Legacy `verdict` fields remain for report compatibility, but are explicitly marked and rendered as summaries of third-party publisher ratings. Claim/review evidence is informational and never contributes a directional fusion vote. Retrieval matching and publisher-rating interpretation remain heuristic and do not constitute TruthLens fact adjudication.

## Endpoint

`POST /api/v1/fact-check` accepts 1–20 claim strings (5–2,000 characters each) and an optional language code. It returns normalized reviews, per-claim retrieval status, safe warnings, runtime, and the existing compatibility fields. Missing provider configuration returns a normal `unavailable` result; the endpoint does not fabricate reviews.

The endpoint returns one indexed verification entry for each submitted claim. Ineligible items have `verification_status=SKIPPED`, `eligibility_status=INELIGIBLE`, and an explicit reason. If no input claim is eligible, the top-level result is `skipped` and the provider is not called. Supported E3 language-code validation is applied to standalone requests.

## Remaining limits

- Provider behavior is mock-tested only; no live Google credentials, quota, or production network verification was performed.
- Response size defaults to 1 MB and can be configured with `FACTCHECK_MAX_RESPONSE_BYTES`; the service clamps the effective value to 1 KiB–10 MB.
- Standalone request eligibility uses the same heuristic E4 classifier as the pipeline. This is not a semantic or truth determination.
- Sentence offsets refer to normalized OCR text, not byte offsets into the original image or raw OCR buffer. OCR region references are populated only when upstream regions include aligned source ranges.
- Existing verdict aggregation is retained as a publisher-rating summary for compatibility; it is not an independent semantic fact-check.
- The standalone OCR endpoint remains outside this combined E4/E5 fact-check API change.
