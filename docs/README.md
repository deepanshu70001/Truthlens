# TruthLens Documentation Directory (`docs/`)

Welcome to the reference documentation for **TruthLens** — an academic prototype for AI-assisted image verification, digital media analysis, and contextual indicator estimation.

For high-level project summaries and formal evaluation dossiers, see also the [PROJECT_DOCUMENTATION/](../PROJECT_DOCUMENTATION/PROJECT_MASTER_DOSSIER.md) directory.

---

## 1. System Architecture & Operation

These documents describe the operational design, API contracts, deployment workflows, and model contracts:

| Document | Description |
|----------|-------------|
| [system_architecture.md](system_architecture.md) | Component data flow, pipeline orchestrator, storage layout, and implemented vs. unavailable capabilities. |
| [deployment_guide.md](deployment_guide.md) | Local environment setup, dependencies (Python 3.12, Node 22), configuration (`.env.example`), and demo flow. |
| [api_reference.md](api_reference.md) | Endpoint specifications (`/api/v1/*`), request schemas, lifecycle statuses, and structured report JSON schema. |

---

## 2. Methodology & Feature Specifications

Specifications detailing how heuristic content indicators, forensics, and review workflows are computed:

| Document | Description |
|----------|-------------|
| [virality_methodology.md](virality_methodology.md) | Content spread indicator calculation; explains why indicators are heuristic and not predictive likelihoods. |
| [VIRALITY_DATA.md](VIRALITY_DATA.md) | Data readiness, dictionary rules, and heuristic scoring weights for content spread cues. |
| [regional_spread_methodology.md](regional_spread_methodology.md) | Canonical India 28 states & 8 UTs mapping; explains `DATA_UNAVAILABLE` status for predictive regional spread. |
| [social_impact_methodology.md](social_impact_methodology.md) | Token-based review cues across 5 sensitive topic categories (communal, health, politics, finance, child safety). |
| [ai_generation_misclassification_review.md](ai_generation_misclassification_review.md) | Review of detector behavior, calibration requirements, and argmax vs. application label separation. |

---

## 3. Implementation Reports & Audits (Phases 1–6)

Chronological and technical reports detailing the implementation and hardening of system phases:

### Phase 1–3: Core Analysis Pipeline
- [phase1_implementation_report.md](phase1_implementation_report.md) — Upload validation, SigLIP detector integration, orchestrator, and report builder.
- [phase2_implementation_report.md](phase2_implementation_report.md) — Image preprocessing, metadata extraction, classical forensics, and OCR service.
- [phase3_implementation_report.md](phase3_implementation_report.md) — Heuristic claim extraction, fact-check adapter, and provenance search.

### Phase 4: Social & Contextual Indicators
- [phase4_implementation_report.md](phase4_implementation_report.md) — Virality, regional spread, and social impact heuristic engine implementations.
- [phase4_dataset_inventory.md](phase4_dataset_inventory.md) — Input data audit and justification for `DATA_UNAVAILABLE` designations.
- [phase4_evaluation.md](phase4_evaluation.md) — Evaluation boundary for indicator modules.

### Phase 5: Hardening & System Audits
- [phase5_implementation_report.md](phase5_implementation_report.md) — Error handling, partial-stage recovery, and graceful degradation.
- [phase5_e2e_test_report.md](phase5_e2e_test_report.md) — Historical end-to-end integration and smoke test results.
- [phase5_performance_report.md](phase5_performance_report.md) — Preprocessing and inference timing microbenchmarks.
- [phase5_security_audit.md](phase5_security_audit.md) — Security boundaries, input validation, and upload sanitization review.
- [phase5_system_audit.md](phase5_system_audit.md) — Architectural resilience, SQLite vs. PostgreSQL, and task lifecycle audit.

### Phase 6: Research Readiness & Evaluation
- [phase6_implementation_report.md](phase6_implementation_report.md) — Controlled scenario evaluation, evidence fusion hardening, and metric checks.
- [phase6_final_research_report.md](phase6_final_research_report.md) — Comprehensive research findings, system capabilities, and empirical boundaries.
- [phase6_authenticity_evaluation.md](phase6_authenticity_evaluation.md) — Authenticity evaluation status (`NOT_EVALUABLE` on local approved data).
- [phase6_ablation_study.md](phase6_ablation_study.md) — Evidence fusion ablation analysis and exclusion of non-completed evidence.
- [phase6_error_analysis.md](phase6_error_analysis.md) — Failure mode taxonomy (domain shift, OCR omissions, external provider outages).
- [phase6_experiment_setup.md](phase6_experiment_setup.md) — Reproducibility protocol, hardware environment, and metric calculation rules.
- [phase6_experiment_dataset_inventory.md](phase6_experiment_dataset_inventory.md) — Audit of repository image assets and license limitations.
- [phase6_explainability_evaluation.md](phase6_explainability_evaluation.md) — Grad-CAM attribution limitations and evaluation boundaries.
- [phase6_multilingual_evaluation.md](phase6_multilingual_evaluation.md) — OCR script estimation and multilingual coverage boundaries.
- [phase6_social_analytics_evaluation.md](phase6_social_analytics_evaluation.md) — Evaluation status of social indicators.
- [phase6_research_questions.md](phase6_research_questions.md) — Core academic research questions and investigation status.
- [phase6_research_readiness_audit.md](phase6_research_readiness_audit.md) — Readiness review for academic submission.
- [phase6_responsible_ai_assessment.md](phase6_responsible_ai_assessment.md) — Ethical considerations, bias risks, and misrepresentation safeguards.
- [phase6_usability_protocol.md](phase6_usability_protocol.md) — Proposed (unconducted) user study protocol.
- [phase6_viva_preparation.md](phase6_viva_preparation.md) — Comprehensive panel Q&A guide and technical defense preparation.

---

## 4. Academic Papers & Presentations

- [TruthLens_IEEE_research_paper.md](TruthLens_IEEE_research_paper.md) — Formatted IEEE prototype research paper draft documenting the TruthLens architecture, evaluation boundaries, and findings.

---

## 5. Master Dossier Cross-Reference

For the consolidated institutional documentation suite, consult the `PROJECT_DOCUMENTATION/` directory:
- [PROJECT_MASTER_DOSSIER.md](../PROJECT_DOCUMENTATION/PROJECT_MASTER_DOSSIER.md) — Master project overview and executive summary.
- [TECHNICAL_ARCHITECTURE.md](../PROJECT_DOCUMENTATION/TECHNICAL_ARCHITECTURE.md) — Deep technical architecture specification.
- [FUNCTIONALITY_AND_WORKFLOW.md](../PROJECT_DOCUMENTATION/FUNCTIONALITY_AND_WORKFLOW.md) — Comprehensive user and pipeline workflows.
- [TESTING_AND_EVALUATION.md](../PROJECT_DOCUMENTATION/TESTING_AND_EVALUATION.md) — Testing evidence, benchmarks, and protocol guidelines.
- [RESEARCH_FOUNDATION.md](../PROJECT_DOCUMENTATION/RESEARCH_FOUNDATION.md) — Literature review, methodology, and scientific rationale.
- [LIMITATIONS_AND_FUTURE_SCOPE.md](../PROJECT_DOCUMENTATION/LIMITATIONS_AND_FUTURE_SCOPE.md) — Known constraints, ethical boundaries, and priority roadmap.
- [PROJECT_EVIDENCE_INDEX.md](../PROJECT_DOCUMENTATION/PROJECT_EVIDENCE_INDEX.md) — Codebase line citations and verification status.

---

## 6. Superseded Documents

Early speculative drafts from October 2026 (`ARCHITECTURE.md`, `IMPLEMENTATION_PLAN.md`, `MODELS.md`, and `EVALUATION.md`) were drafted before code was written and described hypothetical architectures (e.g. Celery, Redis, external VLM pipelines). They have been removed to avoid confusion. All current system architecture and evaluation records are authoritatively captured in [system_architecture.md](system_architecture.md) and the `PROJECT_DOCUMENTATION/` suite.
