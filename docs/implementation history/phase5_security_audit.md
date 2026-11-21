# Phase 5 security and privacy audit

Audit date: 2026-11-17. This is a code-level audit and targeted test review; it is not a penetration test.

## Controls verified or added

- Upload reads at most configured size plus one byte, detects actual type from file bytes, checks allow-listed MIME, verifies filename extension matches bytes, checks magic signature and decodes with Pillow.
- Added configurable `MAX_IMAGE_PIXELS` (40,000,000 default) before image processing. Server filenames are random UUIDs; display filenames are sanitized. Storage uses the configured originals directory.
- Image-return route validates analysis UUID and confines resolved image paths beneath configured originals storage. Heatmap and ELA routes similarly constrain resolved artifact paths.
- Report sanitizer strips local paths and secret-like keys. API responses omit storage paths. User-facing stage errors are generic; logs avoid OCR text and image bytes.
- CORS uses configured explicit origins and is restricted to GET/POST and Content-Type headers.
- React renders evidence text as text, not raw HTML. External evidence links are accepted only for HTTPS URLs and open with `noopener noreferrer`. Evidence pages are not fetched by this application, so embedded instructions are not executed or treated as prompts.
- Fact-check/provenance requests use configured HTTP timeouts. Fact-check retrieval does not crawl returned page URLs; provenance sends image bytes to Google Cloud Vision only when configured.

## Data handling

Original image bytes, thumbnails, resized inputs, heatmaps, and forensic artifacts are stored under `STORAGE_PATH`; the database stores storage references, hashes, image metadata, model runs, events, and a JSON report. Originals and artifacts currently have no retention TTL or deletion API. They persist until an operator removes the storage and corresponding database records. Do not use sensitive/private images in a shared deployment without an approved retention and consent policy.

When configured, Google Cloud Vision Web Detection receives the uploaded image. Google Fact Check Tools receives OCR-derived claim query text. No external AI call is made by virality/regional/social modules. Logs include analysis identifiers, stage names, timing and error categories; uploaded image bytes and OCR text are not deliberately logged.

## Unresolved risks / deployment boundary

- No authentication or ownership isolation is implemented. UUIDs are identifiers, not access control. Keep the service on localhost/trusted network unless protected by an authenticated reverse proxy.
- `RATE_LIMIT_PER_MINUTE` is configuration only; no rate limiter is installed. Apply proxy request-rate and body-size limits for exposed environments.
- Multipart parsing occurs before endpoint handling; set an upstream request body cap too.
- No deletion/retention controls or automated stale-file cleanup exist.
- Database tables are created via SQLAlchemy `create_all`; there are no migration scripts.
- Secrets must be provided through environment/secret manager; `.env` must not be committed.
- Verify provider privacy terms and obtain user consent before enabling outbound image/text integrations.
- Dependency audit and active penetration testing were not performed in this implementation session.

## Test evidence

Image signature, extension, decoding, and size handling passed unit/API tests. The full backend suite passed (184 passed, 10 skipped); the upload-to-report and original-image retrieval API path is included. Browser upload flow and penetration testing remain unverified.
