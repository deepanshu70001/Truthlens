# TruthLens Technical, Scientific, and Ethical Limitations

> **Document Status:** Comprehensive Limitation & Risk Registry
> **Traceability:** Aligned with `docs/phase7_evaluation_report.md` and `PROJECT_DOCUMENTATION/LIMITATIONS_AND_FUTURE_SCOPE.md`

---

## 1. Scientific & Empirical Limitations

- **Cross-Generator Architectural Shift:** Machine learning detectors fine-tuned on early diffusion engines (2023 era) suffer catastrophic negative transfer when evaluated against modern 2026 flow-matching and autoregressive architectures (Frontier Benchmark ROC-AUC collapsed from 0.6501 to 0.4234 for Candidate S2).
- **Elevated False Alarm Rates on Authentic Media:** At default 0.5 decision thresholds, real-world web photographs produce substantial false positive rates (up to 33.97% on Frontier Benchmark test and 95.98% on WildAIGC test for baseline). High-confidence operations require elevated operational thresholds ($t \approx 0.93$).
- **Absence of Ground-Truth Dissemination Data:** TruthLens does not possess longitudinal propagation or reach datasets. Virality indicators are heuristic rule approximations, and regional spread is marked `DATA_UNAVAILABLE`.
- **Informational Forensic Limitations:** Classical forensic algorithms (ELA, noise variance, resampling) are sensitive to standard JPEG compression, social platform transcoding, and image resizing; they cannot establish proof of manipulation.
- **Surface-Level OCR & Claim Rules:** EasyOCR and regex-based claim extractors detect surface tokens but lack semantic truth verification capabilities.

---

## 2. Technical & Operational Limitations

- **In-Process Task Orchestration:** Analyses execute asynchronously within the API process via `asyncio.create_task`. A server restart terminates running analyses without automatic recovery.
- **Local Media Persistence:** Original uploads and diagnostic heatmaps reside on local disk without automated TTL cleanup, quotas, or user-triggered deletion.
- **Absence of Authentication & Multi-Tenancy:** The prototype does not implement user accounts, role-based access control (RBAC), or tenant isolation.
- **Database Scalability:** SQLite is configured by default; concurrent multi-user workloads require PostgreSQL and an external connection pool.

---

## 3. Ethical, Legal & Responsible AI Considerations

- **Risk of False Accusations:** Mislabeling genuine photographs as synthetic can cause reputational harm to photojournalists and media organizations. TruthLens is designed strictly as human decision support, never as an automated censor.
- **Context Detachment:** Image authenticity is independent of claim veracity; an authentic photograph can accompany deceptive misinformation.
- **Data Privacy & Third-Party APIs:** Optional Google Vision and Fact Check Tools integrations transmit image data or OCR text to third-party endpoints. Analysts must consider confidentiality before analyzing sensitive media.
