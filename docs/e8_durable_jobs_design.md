# E8 — Durable analysis jobs

## Starting state

At E8 baseline, `POST /api/v1/analyze` committed an `Analysis` row and uploaded asset, then launched `asyncio.create_task`. The live task registry existed only in the API process. A process exit lost active work; an in-memory task was not shared between server workers. There was no queue record, lease, bounded retry, idempotency key, queue capacity, or recoverable worker state. Reports were stored in `analyses.report_json`; uploaded and derived artifacts were stored as files. SQLite was configured locally and Alembic was installed but no migration history existed.

## Architecture

E8 adds a database-backed job table and a separate worker process. `Analysis.id` is reused as the stable `AnalysisJob.analysis_id` and public `job_id`, so existing clients continue polling the analysis ID. Job delivery state (`queued`, `running`, `retry_wait`, `completed`, `failed`) is internal; E1 stage states and analysis statuses remain authoritative for stage/report semantics. The persisted analysis enum remains `queued | processing | completed | failed`; partial completion continues to be derived from `report_json.status`.

`POST /api/v1/analyze` validates and stages the image, then commits `Analysis`, `ImageAsset`, `AnalysisJob` and audit events together. It returns after durable registration. A same-owner `Idempotency-Key` plus matching SHA-256 returns the original job; reuse with different bytes returns 409. Without a key, submissions are independent. Queue admission is an atomic update to one `QueueCapacity` row; when the configured outstanding limit is full, submission returns 429 with `Retry-After`.

Run the worker independently:

```powershell
python -m backend.app.workers.analysis_worker
```

Each worker competes for one durable `WorkerLease` row. One global lease serializes the pipeline and GPU/model inference, so deploying extra worker processes does not increase model concurrency. Heartbeats renew the job and global lease. Multiple API instances share jobs, leases, queue capacity, ownership, and rate-limit records through the database.

## Retry and recovery

- The default bound is three attempts. `TimeoutError`, connection and storage `OSError`, SQLAlchemy `DBAPIError`, and graceful worker cancellation are retryable. Other exceptions become safe terminal failures.
- Retry delay is exponential from `ANALYSIS_RETRY_BASE_SECONDS`, capped by `ANALYSIS_RETRY_MAX_SECONDS`.
- A worker crash leaves a running lease. Once the lease expires, the next worker can atomically acquire the global slot, reset interrupted `running` stage progress to pending, and queue the attempt again.
- If the report is already terminal when a stale running job is inspected, the job is acknowledged without re-running the pipeline. Reports are stored once in the analysis row; generated artifact names are based on the analysis ID.
- Delivery is **at-least-once**. A crash before report persistence may repeat stages and external provider requests. Exactly-once external execution is not claimed.
- If retry attempts are exhausted, analysis and job become terminal failed with a safe error category/message. API/report retrieval sanitizes old and new report output.

Progress is persisted by the E1 orchestrator in `Analysis.completed_steps` and is returned with the durable job state by the existing status endpoint. Status/report/artifact reads continue to enforce analysis ownership. Existing clients that only use `analysis_id` remain compatible; the submit response adds `job_id`.

## Persistence and limits

Database commit and filesystem storage cannot form one atomic transaction. Uploads use an app-generated staging name, atomic rename and best-effort compensation when DB persistence fails. A crash between filesystem rename and DB commit can leave an orphan. `find_orphan_storage_files` reports orphan candidates and does not delete them. User-triggered analysis deletion removes dependent rows in a transaction and only generated paths under known per-analysis roots afterward. There is no automatic analysis/report retention policy.

The worker has one global active slot and the queue has a finite configured outstanding capacity. Rate limits and leases are database-backed and fail closed if storage is unavailable. Production config requires PostgreSQL with `asyncpg`; SQLite is supported for local development/tests.

## Verification boundary

The focused tests use deterministic mocked pipeline functions and isolated SQLite. They verify durable registration, idempotency, success/partial/failed terminal results, transient/permanent failures, capacity, progress, serialized claims, stale lease recovery and terminal-report acknowledgement. No real model inference, production PostgreSQL, multi-process worker kill/restart, provider billing behavior, deployment supervisor or production load test was run. Those remain release checks.
