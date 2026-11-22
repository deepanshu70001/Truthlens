# E4/E5 Claim Extraction and Fact-Check Contract

## Claim extraction

The pipeline consumes normalized OCR text and preserves sentence offsets into that normalized source. Sentence segmentation avoids common title abbreviations, and each extracted record retains the original sentence, normalized query, deterministic claim ID, source reference, optional source offsets/region references, extraction method/version, language code and language-detection outcome. Deduplication collapses only whitespace/case variants; punctuation and numeric content remain intact. Candidate classification is a conservative heuristic. Its score is explicitly a ranking hint, never a probability that the statement is true.

Language metadata is copied only from a successful, confident E3 classifier result. Otherwise claims remain `unknown` with unavailable/undetermined status; OCR script estimates are not promoted to language identification. Extraction remains independent of the optional language model.

## Fact-check lookup

Only claims marked `ELIGIBLE` are sent to the Google Fact Check Tools API. Queries are whitespace-normalized, bounded to 2,000 characters in the standalone API, limited by configured claim count, and deduplicated case-insensitively. A language filter is included only when a validated language code is supplied. The API key is server-side and responses contain only safe provider error categories/messages.

Provider reviews are normalized from returned data and unsafe non-HTTPS review URLs are rejected. Successful lookups are retained when another query fails. The service distinguishes successful empty results (`no_results`), unavailable credentials (`unavailable`), provider failures (`failed`) and mixed success/failure (`partial`). Missing eligible claims skip the pipeline stage with `not_applicable`.

Legacy `verdict` fields remain for report compatibility, but are explicitly marked and rendered as summaries of third-party publisher ratings. Claim/review evidence is informational and never contributes a directional fusion vote. Retrieval matching and publisher-rating interpretation remain heuristic and do not constitute TruthLens fact adjudication.

## Endpoint

`POST /api/v1/fact-check` accepts 1–20 claim strings (5–2,000 characters each) and an optional language code. It returns normalized reviews, per-claim retrieval status, safe warnings, runtime, and the existing compatibility fields. Missing provider configuration returns a normal `unavailable` result; the endpoint does not fabricate reviews.

## Remaining limits

- The Google provider has not been live-tested with credentials in this milestone.
- The E3 checkpoint is absent locally, so language identification remains optional/unavailable until provisioned and verified.
- Sentence offsets refer to normalized OCR text, not byte offsets into the original image or raw OCR buffer. OCR region references are populated only when upstream regions include aligned source ranges.
- Existing verdict aggregation is retained as a publisher-rating summary for compatibility; it is not an independent semantic fact-check.
- The standalone OCR endpoint remains outside this combined E4/E5 fact-check API change.
