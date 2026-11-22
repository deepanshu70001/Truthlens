# E1 unified stage execution contract

## Canonical execution status

Every pipeline stage uses exactly one execution status:

`pending` · `running` · `completed` · `partial` · `failed` · `unavailable` · `skipped`

Execution status is separate from outcome. `success`, `no_results`, `no_text`, and `not_applicable` are outcomes, not errors. For example, a provider query returning no reviews is `completed` with outcome `no_results`; missing provider configuration is `unavailable`; a timeout is `failed`; OCR returning no readable text is `completed` with outcome `no_text`; an intentionally unrun prerequisite-dependent stage is `skipped` with outcome `not_applicable`.

Unknown status strings, invalid status types, and `None` module results are malformed module results and become `failed` with error category `INVALID_MODULE_RESULT`. Exceptions become `unavailable` for missing imports and `failed` for other execution errors. A partial service result remains `partial`.

## Stage record

Each stage report contains:

- `name` (stage name) and canonical `status`;
- nullable `outcome`, timestamps, runtime, error category/message, output reference, and module identity;
- stable `stage_id` derived from analysis ID and stage name, plus the analysis `correlation_id`;
- warning and limitation lists;
- a small result descriptor and `output_reference` to the report payload.

The full service payload remains in its report section. The stage record describes execution independently from the evidence and service output. `error` is retained as a compatibility alias for `error_message`; both contain only user-safe text. Exception details remain in server logs. Public stage messages are sanitized for path and secret-like values.

`StageResultOut` documents the public typed stage shape. Existing report responses remain JSON-compatible and are not changed to a new envelope.

## Analysis progress and terminal status

The expected pipeline contains 20 named stages. The existing database enum is unchanged (`queued`, `processing`, `completed`, `failed`) to avoid an unplanned database migration. At completion, the persisted row is `completed`; the report is `partial` when any stage is `partial`, `failed`, or `unavailable`. The existing status API derives `partial` from that report. A pipeline-level unrecoverable exception persists `failed`, a safe terminal report, and `completed_at` when the database is available.

The database `completed_steps` text field now stores a backward-compatible JSON object with `completed_steps` (only successfully completed stages) and `stage_statuses` (all 20 stages). Readers accept older JSON-list values. Progress percentage counts every terminal stage status, including partial, failed, unavailable, and skipped; the successful-step list remains success-only. The status response exposes `progress.stage_statuses`, which the existing frontend uses for small status-indicator updates.

## Evidence and fusion boundary

Stage status and evidence item status remain independent. Fact-check and provenance failure/unavailability evidence now carries the corresponding status. Fusion ignores failed, unavailable, skipped, or otherwise non-completed evidence; deduplication remains in place. Only the existing calibrated authenticity model pathway can contribute a directional model vote. Forensic, OCR, fact-check, provenance, object, and attribution signals remain informational under the existing policy. No fusion weights, calibration, detector identity, thresholds, inference routing, or inconclusive gating were changed in E1.

Missing data is not converted into negative evidence. A no-result lookup is recorded as a successful execution with an informational result.

## Compatibility and remaining work

- Service payloads retain their own established statuses where the report UI depends on them; the enclosing stage record is canonical.
- Database enum and upload lifecycle statuses are unchanged.
- Frontend visual design is unchanged; detailed terminal stage icons/status mapping is consumed from `progress.stage_statuses`.
- Persistent database outages cannot be made durable by application-level retries alone. Terminal failure persistence makes one bounded fallback attempt after rollback; a continued outage is logged and requires worker/recovery semantics in E8.
- Durable jobs, schema migrations, retries across process restarts, and frontend E2E coverage remain later roadmap work.
