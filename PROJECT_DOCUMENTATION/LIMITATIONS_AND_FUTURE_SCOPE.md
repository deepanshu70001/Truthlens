# TruthLens Limitations and Future Scope

## 1. Purpose

This document articulates the verified technical, empirical, and operational boundaries of the TruthLens system. Distinguishing demonstrated capabilities from future research objectives prevents misleading claims regarding detection reliability, spread prediction, or production readiness.

## 2. Empirical and model limitations

- **Cross-Generator Architectural Gap:** Machine learning models fine-tuned on early diffusion and GAN architectures (2023 era: Stable Diffusion 1.4/1.5, Midjourney v4, GLIDE) fail to generalize to modern late-2026 generative architectures (FLUX.1-dev, Stable Diffusion 3.5 Large, Recraft v3). In Phase 7 evaluations, fine-tuning on Tiny-GenImage degraded Frontier Benchmark ROC-AUC from 0.6501 down to 0.4234.
- **Open-Domain Real-Image Specificity:** On diverse in-the-wild and frontier benchmarks, authentic photographs frequently trigger false alarms at standard 0.5 decision thresholds (FAR = 33.97% on Frontier Benchmark test). Mitigating false accusations requires strict operational thresholding ($t \approx 0.93$) or routing borderline samples to human review.
- **Absence of Ground-Truth Spread Data:** The application does not possess longitudinal propagation data (timestamps, share graphs, geographic reach). Virality indicators are heuristic rule approximations, and regional spread is marked `DATA_UNAVAILABLE`.
- **Informational Forensic Indicators:** Classical forensic cues (ELA, noise variance, resampling) are sensitive to standard JPEG compression, social platform transcoding, and resizing. They cannot be used as standalone proof of manipulation.
- **OCR and Text Extraction Boundaries:** EasyOCR and rule-based claim extraction are subject to character recognition errors and hallucinated entities when processing low-contrast, heavily stylized, or non-English text.

## 3. Technical and architectural limitations

- **In-Process Task Orchestration:** Analysis pipelines execute asynchronously within the FastAPI server process via `asyncio.create_task`. Server restarts or unhandled worker crashes will interrupt active analyses without automatic recovery.
- **Storage and Data Retention:** Uploaded images and generated heatmaps are stored locally on the host filesystem (`data/` or `storage/`). There is no automated TTL expiration, quota management, or user deletion workflow.
- **Lack of Authentication & Multi-Tenancy:** The current implementation is an unauthenticated research prototype. It does not provide user accounts, role-based access control, or tenant data isolation.
- **Database Scalability:** SQLite is the default configuration, sufficient for single-user academic demonstrations. High-concurrency operations require PostgreSQL and an external connection pool.

## 4. Ethical, legal and responsible AI boundaries

- **Risk of False Accusations:** Misclassifying an authentic photograph as AI-generated can discredit legitimate journalists and whistleblowers. The system must always be presented as an analyst aid, never as an autonomous truth adjudicator.
- **Context Misalignment:** A genuine image may accompany a fabricated text headline. Image authenticity analysis cannot determine whether a caption or historical claim is true.
- **Privacy and Data Transmission:** Optional external integrations (Google Vision, Google Fact Check Tools) transmit image bytes or OCR-derived query text across third-party networks. These integrations must be disabled or disclosed when analyzing sensitive or private imagery.

## 5. Completed research milestones (Phases 1–7)

| Phase | Milestone | Empirical Achievement & Evidence |
|---|---|---|
| **Phase 1** | Model Identity & Safety | Registered production detector contract (`truthlens/siglip-cifake-frozen-head`) and calibration schemas. |
| **Phase 2** | Dataset Ingestion & Splits | Ingested 3 external benchmarks ($N = 29,640$); established leakage-free group-aware splits ($N = 10,385$ locked test set). |
| **Phase 3** | Baseline Evaluation | Benchmarked frozen SigLIP baseline across all 3 test partitions (Tiny AUC: 0.6123, Frontier AUC: 0.6501, WildAIGC AUC: 0.4226). |
| **Phase 3.5 & 4** | Audit & Training Data Prep | Cryptographically audited split integrity; prepared approved 6,995-sample training dataset. |
| **Phase 5 & 6** | Shortcut Discovery | Uncovered container format shortcut in Tiny-GenImage (100% JPEG real vs 100% PNG AI); rejected shortcut candidate E3. |
| **Phase 7** | Matched Augmentation & Audit | Implemented matched in-memory augmentation (metadata AUC dropped from 1.0000 to 0.5028); trained 5 models; retained frozen baseline in production due to modern generator gap. |

## 6. Priority future roadmap

| Priority | Objective | Technical Scope & Deliverables |
|---|---|---|
| **1. Modern Generator Training** | Cross-Generator Transfer | Train multi-generator representations incorporating 2026 flow-matching architectures (FLUX, SD 3.5) to bridge the out-of-domain frontier gap. |
| **2. Multi-Modal Claim Verification** | Semantic Alignment | Integrate vision-language models (e.g., Qwen2.5-VL) to assess image-text consistency rather than relying solely on surface OCR token rules. |
| **3. Real-World Propagation Studies** | Empirical Spread Data | Partner with academic data archives to collect longitudinal media propagation traces, replacing heuristic virality with validated statistical models. |
| **4. Enterprise Hardening** | Scalability & Security | Implement user authentication (OAuth2 / JWT), database migrations (Alembic), durable task queues (Celery/Redis), and automated data retention lifecycles. |
| **5. Human-in-the-Loop Usability** | Analyst Evaluation | Conduct controlled user studies with professional fact-checkers to measure decision time, calibration accuracy, and cognitive reliance on automated reports. |

## 7. Verified claims safe to make today

- TruthLens provides a comprehensive, multi-modal verification prototype with an end-to-end asynchronous processing pipeline.
- The repository contains an audited, leakage-free benchmark evaluation suite of 29,640 images across 16 generator families.
- Pre-training container format confounding in Tiny-GenImage was formally demonstrated and neutralized via matched in-memory augmentation.
- Automated regression test suites confirm 267 passing tests with zero failures across the backend platform.
- Production deployment strictly retains the frozen baseline detector (`truthlens/siglip-cifake-frozen-head`) in adherence to pre-registered evaluation governance.
