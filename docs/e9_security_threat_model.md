# E9 — Security threat model and controls

## Assets and trust boundaries

| Asset | Boundary / exposure | Main risks |
|---|---|---|
| Uploaded originals, thumbnails and derived reports | Authenticated browser/API to API process and `STORAGE_PATH` | Unauthorized retrieval, oversized/malformed files, path traversal, storage disclosure, retention beyond user expectations |
| Analysis/job IDs, stage progress and reports | API to shared database and worker | IDOR, leaked error data, inconsistent job/report states |
| Bearer signing key and provisioned password hashes | Deployment secret source to API | Forged tokens, credential exposure, weak/default production secret |
| Provider keys and outbound provider requests | Worker/API to configured external providers | Secret leakage, unbounded provider use, unsafe URL/response handling |
| Database and rate/lease state | API and worker processes to SQLite locally or PostgreSQL in production | Shared-state outage, cross-process races, data loss, bypassed rate controls |
| Browser session | Browser memory/session storage to API | Expired/stolen token use, accidental credential forwarding to external artifact URLs |

The browser, uploaded content, MIME declarations, filenames, report content and provider response fields are untrusted. API and worker processes are trusted code but run with access to uploaded files, configured secrets and the database. The database is the shared coordination boundary. Production proxy headers are untrusted unless the operator configures trusted proxy peers in the ASGI server.

## Authentication and ownership

The project has no self-service account lifecycle. E9 implements the smallest maintainable boundary for its current use: operators provision stable user IDs and scrypt password hashes in `AUTH_USERS_JSON`; `/api/v1/auth/token` issues short-lived HS256 bearer tokens with issuer, audience, issue/expiry timestamps and token ID. The development bypass is an explicit setting accepted only for `development`/`test`. Production startup requires a non-placeholder 32-byte secret, valid provisioned scrypt users, debug off, bypass off, explicit trusted hosts/CORS and PostgreSQL.

Every analysis/report/image/heatmap/ELA read and analysis deletion checks `Analysis.owner_id`. Other users and ownerless legacy rows return the same 404. Analysis submission persists owner identity on both analysis and job. Fact-check, provenance, OCR and model-readiness operations require a principal. `/auth/mode` and basic `/health` are intentionally public and reveal only mode/health data; model readiness is protected and does not return local checkpoint paths.

There is no refresh token, token revocation list or user-admin interface. Operators rotate the signing secret to invalidate issued tokens and manage user provisioning through their approved secret process. Production deployment must document access and rotation procedures.

## Rate limits, capacity and upload controls

Rate counters are fixed-window rows keyed by a secret-keyed hash of the authenticated user or direct peer IP. Only the ASGI client peer is trusted; arbitrary `X-Forwarded-For` values are ignored. Counters are atomic SQLite/PostgreSQL upserts shared across API processes. Expensive endpoints use configurable analysis, fact-check, provenance and auth limits; retrieval uses a higher quota to allow normal polling. Exceeding a limit returns 429 and `Retry-After`. If the shared DB/backend is unavailable, the request fails closed with 503; there is no local-memory fallback.

Analysis queue admission is bounded separately. The ASGI request middleware limits request bodies; upload validation checks declared MIME consistency, detected MIME, magic bytes, supported image types, Pillow decode, dimensions/pixel count and configured byte limits. Filenames are sanitized and storage IDs are generated. Retrieval constructs paths only under approved roots and requires ownership. No endpoint accepts an arbitrary filesystem path. Provider response URLs are treated as external links; the backend does not fetch those returned URLs. The frontend only attaches bearer credentials to same-origin analysis artifact paths.

## Configuration, logs and output handling

Production defaults are non-debug and loopback-bound; production requires explicit hosts, CORS, secret, users and PostgreSQL. Interactive API docs are disabled in production. PostgreSQL TLS uses the platform certificate store. Response headers include `nosniff`, frame denial, referrer policy, restrictive permissions, no-store and a restrictive CSP. Cookies are not used; bearer tokens are kept in browser `sessionStorage`.

API errors do not return traceback or raw validation input. Public report/status/model errors redact paths, credentials and stack traces; model-readiness excludes local paths. Logs avoid raw image/EXIF values and log exception class/category rather than full exception tracebacks. Provider keys are not emitted by the service result serializer. Uploaded content is not automatically retained/deleted by age; a documented operator retention decision is required.

## Tests and remaining risks

Focused tests cover missing/malformed/expired tokens, login, cross-user analysis/report/image/delete access, production secret/config rejection, database-backed rate threshold and failure, request-size rejection, MIME/signature/decode/pixel checks, filename traversal, CORS/trusted host/security headers, secret/path/traceback redaction, and safe provider URL handling. Tests use test credentials and temporary/database fixtures; they do not establish production deployment or penetration resistance.

Remaining risks: no external security review or penetration test; PostgreSQL rate-limit concurrency and trusted reverse-proxy behavior remain unverified; credential provisioning/rotation/revocation is operator-managed; rate-limit records contain keyed pseudonymous client identifiers; storage retention/backups and incident response need deployment-specific approval; no production server TLS/proxy stack is supplied by the repository.
