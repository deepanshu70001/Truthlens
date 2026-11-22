# TruthLens documentation

Use these documents as the current reference. Older phase reports are retained as historical evidence; where they conflict with current implementation, prefer the current status register, roadmap, and operational docs.

## Start here

- [Module status register](MODULE_STATUS_REGISTER.md) — implementation status, verification boundaries, and known limitations.
- [Project roadmap](PROJECT_ROADMAP.md) — milestone sequence and current progress.
- [System architecture](system_architecture.md) — current components and data flow.
- [API reference](api_reference.md) — endpoints and report contracts.
- [Deployment guide](deployment_guide.md) — local setup and configuration.
- [Migration runbook](migration_runbook.md) — database upgrade procedure.

## Current implementation and operations

- [E1 stage contract](e1_stage_contract.md) and [failure handling report](e1_failure_handling_report.md).
- [E2 OCR report](e2_ocr_completion_report.md); [E3 language report](e3_language_identification_report.md).
- [E4/E5 claim and fact-check contract](e4_e5_claim_factcheck_contract.md) and [implementation report](e4_e5_implementation_report.md).
- [E6/E7 forensics and provenance report](e6_e7_forensics_provenance_report.md).
- [E8 durable jobs design](e8_durable_jobs_design.md), [E9 security threat model](e9_security_threat_model.md), [E10 database/frontend report](e10_database_and_frontend_testing_report.md), and [combined E8–E10 report](e8_e10_production_reliability_report.md).
- [E11 evidence fusion contract](e11_evidence_fusion_contract.md) and [E11/R10 completion report](e11_r10_completion_report.md).
- [Priority 1/2 environment preflight](priority1_priority2_preflight_report.md) — local readiness snapshot; recheck environment before relying on it.
- [Synthetic demo data card](SYNTHETIC_DEMO_DATA_CARD.md).

## Research evidence

- [Phase 2 split protocol](phase2_split_protocol.md), [Phase 3 evaluation](phase3_evaluation_report.md), [Phase 4 training data](phase4_training_dataset_report.md), [Phase 5 training](phase5_training_report.md), [Phase 6 evaluation](phase6_evaluation_report.md), [Phase 7 protocol](phase7_protocol.md), [Phase 7 training](phase7_training_report.md), and [Phase 7 evaluation](phase7_evaluation_report.md) preserve the model/data governance history.
- Phase 8/8.1 audit records and Phase 9 diagnostic reports remain available in this directory as research traceability records. They are not a substitute for a current evaluation.
- R10 protocol, data governance, architecture, results, reproducibility, and limitations: [protocol](r10_research_protocol.md), [data audit](r10_data_governance_audit.md), [architecture](r10_hybrid_architecture.md), [experiment report](r10_experiment_report.md), [reproducibility manifest](r10_reproducibility_manifest.md), [limitations](r10_blockers_and_limitations.md).
- Older implementation records are grouped under [`implementation history/`](implementation%20history/). Draft papers and project dossiers are not authoritative implementation or evaluation status.

## Methods and limitations

Methodology documents for regional spread, virality, social impact, and detector misclassification are retained because they explain user-facing indicators and their evidence limits. See [regional spread](regional_spread_methodology.md), [virality](virality_methodology.md), [social impact](social_impact_methodology.md), [virality data](VIRALITY_DATA.md), and [misclassification review](ai_generation_misclassification_review.md).
