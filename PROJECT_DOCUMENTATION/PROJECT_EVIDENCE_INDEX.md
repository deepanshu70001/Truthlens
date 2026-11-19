# TruthLens Project Evidence Index

Line numbers refer to the checked-out repository at the time this index was created. They can change in later commits. For a formal dissertation, cite repository revision and re-check line references before submission. No secret values are included.

| Finding / claim | Source path and anchor | Evidence description | Verification status |
|---|---|---|---|
| Project overview and scope | `README.md`, sections “Current capabilities”, “Architecture and stack”, “Dataset, evaluation and research limits” | Describes intended prototype capabilities and key research/deployment caveats | Verified against source/docs |
| Current system architecture | `docs/system_architecture.md`, “Components and data flow” and “Runtime and persistence” | Mermaid pipeline and storage/task notes | Cross-checked with code |
| API router composition | `backend/app/api/v1/router.py` lines 1–20 | Registers analyze, analysis, health and stubs routers under `/api/v1` | Source verified |
| Upload route | `backend/app/api/v1/endpoints/analyze.py`, `create_analysis` (around lines 57–190) | Bounded read, validation, hashes, storage, records, in-process task creation | Source verified |
| Upload size/pixel/type defaults | `backend/app/config.py` lines 27–70 and `backend/app/core/security.py` `validate_image_file` | 20 MB, 40M pixels, configured image MIME types and settings | Source verified |
| File signature and name security | `backend/app/core/security.py`, `validate_magic_bytes`, `sanitize_filename`, `generate_storage_filename`, `validate_image_file` | Header/signature checks, generated storage names and image validation | Source verified |
| Analysis lifecycle persistence | `backend/app/models/analysis.py` | Analysis status, progress, report JSON, scores and timestamps | Source verified |
| Asset metadata | `backend/app/models/image_asset.py` | Analysis link, hashes, path, file format and dimensions | Source verified |
| Model runs and system events | `backend/app/models/model_run.py`; `backend/app/models/system_event.py` | Inference and operational event records | Source verified |
| Database default and session | `backend/app/config.py`; `backend/app/db/session.py`; `docs/deployment_guide.md` | SQLite default; async SQLAlchemy; PostgreSQL configurable | Source/docs verified |
| Frontend routes/components and report sections | `frontend/src/App.tsx` functions `AuthenticitySection`, `EvidenceSection`, `PipelineDetails`, `ForensicsSection`, `ImageViewer`, `OcrSection`, `ClaimsSection`, `FactCheckSection`, `ProvenanceSection`, `ViralitySection`, `RegionalSection`, `SocialImpactSection`, `LimitationsSection` | UI sections for report domains | Source verified; visual behavior not inspected in this audit |
| Frontend request calls | `frontend/src/services/api.ts` functions `uploadImage`, `getAnalysisStatus`, `getAnalysisReport`, health/model helpers | Axios calls to upload/status/report/health and placeholder model listing | Source verified |
| In-process task behavior | `backend/app/api/v1/endpoints/analyze.py`; `backend/app/services/orchestrator.py`; `docs/system_architecture.md` | Asyncio task tracked in API process; no durable queue described | Source/docs verified |
| Analysis pipeline and optional failure handling | `backend/app/services/orchestrator.py`, `run_analysis_pipeline`, `_run_stage` | Ordered stage execution and per-stage state/error handling | Source verified |
| Structured report generation | `backend/app/services/reporting/builder.py`, `build_report` | Report assembly and safe-value filtering | Source verified |
| Authenticity checkpoint/config | `backend/app/config.py`; `backend/app/services/authenticity/detector.py`; `models/baselines/siglip_frozen_head.pt` | Default backend/backbone/checkpoint paths and inference implementation | Source verified; no fresh inference run in this documentation task |
| Authenticity local evaluation | `docs/phase6_authenticity_evaluation.md`; `experiments/results/authenticity_test.json` | `NOT_EVALUABLE`, zero eligible manifest rows; no metrics produced | Existing artifact/report inspected |
| Upstream CIFAKE subset result caveat | `docs/phase6_authenticity_evaluation.md` | Records 200-image branch-reported result and lack of local reproduction/generalization | Reported claim, not independently verified |
| Calibration artifact requirements | `backend/app/services/authenticity/calibration.py`; `docs/phase6_authenticity_evaluation.md` | Calibration helper exists; no matched validated artifact/metrics | Source/docs verified |
| Classical forensics signals | `backend/app/services/forensics/analyzer.py`, `analyze_forensics` | ELA/noise/JPEG/edge/texture/resampling signal implementations and limitations | Source verified |
| OCR dependencies/status | `backend/app/services/ocr/service.py`; `backend/app/config.py`; `docs/phase2_implementation_report.md` | EasyOCR, optional Tesseract, language/cache configuration and reported fix | Source/docs verified; current external OCR provider availability not rerun |
| Claim extraction | `backend/app/services/claims/extractor.py`, `extract_claims` | Heuristic candidate claims and surface features | Source verified |
| Fact-check integration | `backend/app/services/factcheck/service.py`, `search_fact_checks`; `backend/app/config.py` | Config-gated Google Fact Check Tools adapter and bounded settings | Source verified; no live provider query run |
| No verified ClaimBuster adapter | Search of `backend/app/services` and implementation docs | ClaimBuster key setting is present, but no implemented service adapter was identified | Code search verified; re-check before publication |
| Provenance integration | `backend/app/services/provenance/service.py`, `search_provenance` | Config-gated external web detection adapter | Source verified; no live provider query run |
| YOLO optional implementation | `backend/app/services/vision/detector.py` | Ultralytics detector configured by checkpoint path | Source verified; availability not exercised here |
| Grad-CAM implementation | `backend/app/services/xai/gradcam.py` | Conditional activation/gradient hooks and heatmap output | Source verified; no faithfulness study |
| Evidence schema and conversion | `backend/app/services/evidence/schema.py` | Converts model, forensic, OCR, metadata, fact-check, provenance, XAI and vision outputs | Source verified |
| Fusion rule and Phase 6 correction | `backend/app/services/fusion/engine.py`; `docs/phase6_ablation_study.md`; `docs/phase6_final_research_report.md` | Controlled scenarios and exclusion of unavailable/noncompleted evidence documented | Source/docs verified; no efficacy gain claim |
| Explanation generation | `backend/app/services/explanation/generator.py` | Evidence-specific descriptions and limitations | Source verified |
| Virality uses heuristics | `backend/app/services/virality/engine.py`; `docs/virality_methodology.md`; `docs/VIRALITY_DATA.md` | Hand-weighted/text feature indicators; no trained measured spread outcomes | Source/docs verified |
| Regional score unavailable | `backend/app/services/regional/engine.py`; `docs/regional_spread_methodology.md`; `docs/phase6_social_analytics_evaluation.md` | Result structure and unavailable/no outcome data status | Source/docs verified |
| Social impact uses token cues | `backend/app/services/social_impact/engine.py`; `docs/social_impact_methodology.md` | Exact token/category cue implementation | Source/docs verified |
| Standalone endpoint stubs | `backend/app/api/v1/endpoints/stubs.py` lines 1–60 | OCR/fact-check/provenance return 501; models returns empty placeholder | Source verified |
| Health checks | `backend/app/api/v1/endpoints/health.py` | Database, uptime, GPU and model presence/status checks | Source verified |
| Image/artifact path checks | `backend/app/api/v1/endpoints/analysis.py`, image/heatmap/ELA handlers | UUID and resolved path containment; missing artifacts return not found | Source verified |
| No authentication/user model | ORM `backend/app/models/`; API route dependencies; `docs/phase5_security_audit.md` | No user/account/role entity or auth guard found; audit says no authentication | Source/docs verified |
| No retention/deletion and production queue | `docs/deployment_guide.md`; `docs/phase5_system_audit.md`; orchestrator/task code | Local persistence, no deletion workflow/durable queue; local demo recommendation | Source/docs verified |
| Backend test result (Phase 6) | `docs/phase6_final_research_report.md`; `docs/phase6_implementation_report.md` | 188 passed, 10 skipped, historical report | Existing report inspected; not rerun |
| Backend/UI historical verification (Phase 5) | `docs/phase5_e2e_test_report.md` | Earlier test counts, build/lint, uncontrolled browser run | Existing report inspected; not rerun |
| No current approved benchmark corpus | `docs/phase6_experiment_dataset_inventory.md`; `experiments/results/test_originals_inventory.json` | Test originals lack labels/rights/group/split metadata | Existing audit artifact inspected |
| Research-question plan | `docs/phase6_research_questions.md`; `docs/phase6_experiment_setup.md` | Proposed metrics/protocols and caveats | Existing documentation inspected; proposed, not completed |
| Literature citations present | `docs/phase6_final_research_report.md`; `docs/TruthLens_IEEE_research_paper.md` | Bibliography and related research discussion | References present; publication metadata should be independently checked |
| Environment variable names only | `.env.example`; `backend/app/config.py` | Configuration names/defaults | Values from any private local `.env` intentionally not copied |
| Git history at inspection | `git log -8 --oneline`; `git status --short --branch` | Main branch ahead seven commits; runtime data untracked | Snapshot verified; history not treated as proof of runtime feature behavior |

## Verification boundaries

- This index is a static code/repository inspection; it does not imply a live deployment, fresh test run, real-provider call or model inference during this documentation task.
- Local `data/`, databases, `.venv`, `.pytest_cache`, `temp/`, private environment values and untracked uploaded images were not used as evidence for research efficacy.
- UI sections were inspected in source, not visually validated with a running browser in this task.
- Documentation claims can become stale; source code at a pinned commit should be checked for final academic citation.
