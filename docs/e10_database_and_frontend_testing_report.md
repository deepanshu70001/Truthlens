# E10 — Database lifecycle and frontend testing

## Database schema audit

The pre-E10 SQLite schema contained `analyses`, `image_assets`, `model_runs` and `system_events`, with analysis/report JSON persisted in the analysis row and upload paths persisted in image metadata. There was no Alembic history. Existing relationships had no deletion cascades, and deleting an analysis was not exposed. `main.py` called `Base.metadata.create_all` at application startup. Existing local `truthlens.db` was inspected read-only before changes; it had the four expected legacy tables and no `alembic_version`. Its rows and upload data were not read or modified.

## Migration architecture and evidence

Alembic is configured at repository root with async SQLite/PostgreSQL migration execution and SQLite batch support.

1. `20261122_01` is a baseline. For a fresh DB it creates the legacy four-table schema. For an existing DB, it validates required table columns and does not recreate or drop user tables. Its downgrade is intentionally a no-op.
2. `20261122_02` adds nullable analysis ownership, required indexes, `analysis_jobs`, `queue_capacity`, `worker_leases` and `rate_limit_buckets`. It seeds shared capacity/lease rows. Existing queued/processing analyses are adopted as queued jobs while retaining report and evidence JSON; interrupted stage states are reset to pending. Downgrade refuses to discard job history, ownership or outstanding work.

Fresh temporary SQLite upgrade reached `20261122_02`; `alembic check` found no metadata drift. Existing-schema test upgraded a baseline DB containing a queued analysis and sentinel partial report; the sentinel report survived, legacy ownership remained null, the job was backfilled, and capacity was updated. A downgrade attempt refused to remove the durable job and left the report intact. The repository's ignored `truthlens.db` was not migrated or written.

Normal application startup no longer creates schema implicitly. Production API startup validates the database is at the current Alembic head. Apply `python -m alembic upgrade head` explicitly before API/worker startup. See `docs/migration_runbook.md`.

## Integrity, deletion and retention

SQLite foreign keys are enabled on connections. Analysis deletion refuses a running job, removes job/model/event/image/analysis rows in one DB transaction, then deletes only generated files under known per-analysis storage roots. If file deletion fails, the DB deletion remains authoritative and the orphan scanner can report the file. The scanner is dry-run/report-only. Startup cleanup removes only expired app-generated staging files matching the constrained `.uploading` name under the temp root. There is no automatic analysis, image or report retention policy; operators must approve and document one before shared production use. No cleanup path targets research/evaluation directories.

## Frontend test infrastructure

The frontend uses Vite 8, TypeScript, React 19 and npm. E10 added Vitest, React Testing Library, JSDOM and Playwright scripts. `npm run typecheck` checks application code; `npm test` runs only maintained `src/__tests__` paths; `npm run build` produces the production bundle; `npm run test:e2e` runs a mocked Chromium browser flow.

The previously ignored `frontend/src/lib/pipelineProgress.test.ts` still imports a missing `shared/india_regions.json`, has an implicit-any callback because that data type is missing, and previously had no Vitest module. E10 does not fabricate a regional data file or overwrite the ignored test. Test discovery and app typecheck exclude that legacy ignored test until its data owner supplies the real catalog. The `AnalysisReport` type is exported from the API client so existing ignored helper code can consume the report shape.

Component tests exercise login gating, invalid-file feedback, upload/idempotency key, queued job handling, terminal partial report rendering, restored analysis after reload, auth-expiry behavior and protected artifact URL restrictions. The browser E2E test mocks authentication, job submission/status/report and image retrieval, then performs `Authenticate → Upload → Submit → Poll → View partial report`. It is not live backend or model inference.

## Compatibility and limits

Older report JSON remains served through the report sanitizer and tolerant frontend fields. Progress/status keeps E1 stage semantics; job-delivery state is a separate optional `job` response object. The UI layout was not redesigned. PostgreSQL migrations, backup/restore, production API/browser integration and data-retention operations were not tested. GitHub Actions was updated but not executed from this workspace.
