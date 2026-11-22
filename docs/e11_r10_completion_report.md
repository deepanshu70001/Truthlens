# E11 + R10 completion report

Date: 2026-11-22. Branch: `fix/truthlens-remediation`. Base: `93b551c`.

## Status

- **E11: COMPLETED AND VERIFIED.** Conservative fusion contract and report integration implemented with legacy compatibility. It admits only validated, calibrated primary authenticity evidence to directional fusion; failed/unavailable/skipped stages and informational evidence do not vote. No production fusion weights or decision policy changed.
- **R10: PARTIALLY IMPLEMENTED.** The preregistered research candidate ran and met the fixed AUC acceptance checks. It remains development-only; no production integration, calibration, or locked evaluation.

## E11 implementation

Evidence items now retain stage execution outcome separately from evidence content. Fusion applies source, stage, identity, calibration, direction, correlation, and duplicate checks and records exclusions and missing evidence. Explanation/report fallback handling avoids treating absent or failed stages as negative evidence. Frontend distinguishes the final evidence assessment from the primary model output and reads older reports. See `docs/e11_evidence_fusion_contract.md`.

Changed E11 files: `backend/app/services/evidence/schema.py`, `backend/app/services/explanation/generator.py`, `backend/app/services/fusion/engine.py`, `backend/app/services/orchestrator.py`, `backend/app/services/reporting/builder.py`, `backend/tests/test_e11_fusion.py`, `backend/tests/test_phase6.py`, `backend/tests/test_stage4.py`, `frontend/src/App.tsx`, `frontend/src/__tests__/App.test.tsx`.

## R10 governance and results

Protocol `truthlens-r10-siglip-frequency-residual-v1` was fixed before fitting. Inputs were restricted to `experiments/reports/phase9_feature_vectors.npz` (SHA-256 `a1b9d3530f9555ba4b67cc4a5d2b3793ebac7804bb150eb0a2856da4efb00ed9`) and `models/baselines/siglip_frozen_head.pt` (SHA-256 `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`). Training used 6,995 rows; validation comprised Tiny 5,958, Frontier 297 (75 groups), and WildAIGC 181 (diagnostic only). No raw images, mixed split manifests, protected calibration/test partitions, or row-level prediction outputs were accessed or saved.

Exact command: `python research/r10_e11/run_experiment.py` (exit 0; CPU optimizer runtime 7.037 s; Python 3.12.10, NumPy 2.2.6). Frontier baseline AUC 0.7497, candidate 0.8128, delta +0.063123. Paired source-group bootstrap (2,000 replicates, seed 2026): mean +0.062571, 95% CI [+0.017259, +0.118160]. Tiny balanced-accuracy delta +0.063272; Frontier FPR delta −0.310881. All preregistered acceptance checks passed. However, Frontier balanced accuracy declined 0.080137 and FNR increased 0.471154 at the fixed zero-logit threshold. This is a substantial operating-point tradeoff. Hybrid logits are uncalibrated; this does not justify production deployment or a decision-policy change.

Artifacts: `research/r10_e11/config.json`, `result.json`, `candidate_coefficients.json`, and `run_experiment.py`. Coefficients SHA-256: `d6c88fd365a4ca1c609ea135db69f7534c254512f1730a19be2a81b35d06c4ed`. The exact reports are `docs/r10_research_protocol.md`, `docs/r10_data_governance_audit.md`, `docs/r10_hybrid_architecture.md`, `docs/r10_experiment_report.md`, `docs/r10_reproducibility_manifest.md`, and `docs/r10_blockers_and_limitations.md`.

## Validation evidence

- `python -m pytest backend/tests/test_e11_fusion.py backend/tests/test_stage_contract.py backend/tests/test_e4_e5_contract.py backend/tests/test_e6_e7_contract.py backend/tests/test_stage4.py backend/tests/test_phase1.py backend/tests/test_phase2.py backend/tests/test_phase3.py backend/tests/test_phase6.py -q -rs`: **153 passed**, 0 failed, 0 skipped.
- Safe backend selection: `python -m pytest backend/tests -q -rs --ignore-glob="backend/tests/test_phase*.py" --ignore=backend/tests/test_authenticity_calibration.py --ignore=backend/tests/integration`: **270 passed, 12 skipped, 0 failed**. Skips are real-runtime/model tests; six existing aiosqlite datetime deprecation warnings. This is not the full backend suite; protected and integration suites were excluded. Focused allowed Phase tests are separately included above.
- Frontend isolated install: `npm test -- --reporter=dot`: **6 passed**; `npm run typecheck`: exit 0; `npm run build`: exit 0; `npm run lint`: exit 0 with two existing set-state-in-effect warnings. Build warns about CSS import order and an 868 kB bundle chunk.
- `python -m compileall -q backend/app backend/tests research/r10_e11`: exit 0.
- `git diff --check`: exit 0; Git emitted existing working-copy LF-to-CRLF notices.
- R10 exact command above: exit 0; one deterministic run, no test skip/failure concept applies.

Tests were mocked/unit-level except the research run on the permitted precomputed feature vectors. No production service, multi-process worker, live provider, or deployment verification is claimed. E11 regression tests do not certify scientific calibration or generalization. Frontend dependencies were installed and verified in an isolated temporary copy because the repository's ignored `node_modules` lacked Vitest; workspace dependency artifacts were not changed.

## Preservation and limitations

Initial worktree was clean; all work described here is the current uncommitted change set. Production checkpoint and identity were hash-checked and unchanged; calibration, inference routing, protected evaluation artifacts, and production fusion policy were not modified. No commit or push was made. R11–R13 and Stage 5 completion remain out of scope.
