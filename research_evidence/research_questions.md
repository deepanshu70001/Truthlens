# TruthLens Research Questions & Author Decision Matrix

> **Document Status:** Actionable Review for Authors & Advisors  
> **Inspection Date:** 2026-10-20  
> **Primary References:** `research_evidence/README.md`, `research_paper_draft.md`

---

## 1. Formal Academic Research Questions (RQs)

To ensure the research paper delivers a sharp scientific contribution rather than a mere software overview, the following four Research Questions are formulated directly around the system's architecture and empirical findings:

### **RQ1: Multi-Signal Evidence Fusion vs. Single Oracle**
> *Can an auditable multi-criteria evidence fusion framework reconcile conflicting multimodal signals (visual forensics, deepfake detection, reverse image lookup, and fact-checking) more reliably than single-model deep learning classifiers, particularly on "cheapfakes" and recycled media?*
- **Hypothesis:** Single models fail on unmanipulated historical images paired with deceptive text. Fusing reverse image provenance with claim verification detects out-of-context recycling where vision classifiers return false negatives.

### **RQ2: High-Frequency Artifact Sensitivity vs. Foundation Representations**
> *How do compact, un-pretrained convolutional neural networks compare with large, frozen vision-language foundation backbones (e.g., SigLIP) in terms of parameter efficiency, convergence rate, and discriminatory capability on synthetic diffusion artifacts?*
- **Hypothesis:** While CNNs trained from scratch exploit high-frequency spatial grid artifacts at native resolution, frozen vision transformers capture semantic-distributional discrepancies with several orders of magnitude fewer trainable parameters (0.00083% of weights).

### **RQ3: Sociodemographic Exposure Modeling Under Platform Data Scarcity**
> *In the structural absence of closed-messaging forward cascades (WhatsApp/Telegram), how effectively can regional linguistic distributions and telecom penetration statistics approximate geographic vulnerability to image-borne disinformation in multilingual environments like India?*
- **Hypothesis:** Cross-referencing detected Indic OCR scripts with Census 2011 linguistic speakers and TRAI broadband density produces an actionable, auditable exposure ranking without fabricating pseudo-empirical virality metrics.

### **RQ4: Pipeline Resilience & Fail-Soft Fault Tolerance**
> *How does an asynchronous, staged orchestration architecture maintain diagnostic utility when confronted with missing metadata, third-party API rate-limiting, and network timeouts?*
- **Hypothesis:** Decoupled stage runners with individual exception boundaries preserve forensic integrity, allowing internal visual and heuristic probes to succeed even when external web lookups fail.

---

## 2. Unresolved Author Decisions & Input Requirements

The following strategic and technical decisions require alignment by the project author:

| # | Question / Decision Item | Options / Alternatives | Author Recommendation / Impact |
| :--- | :--- | :--- | :--- |
| **1** | **Target Publication Venue** | • IEEE / ACM Conference (e.g. IEEE Access, ACM Web Conf, ICASSP)<br>• Capstone University Evaluation / Departmental Review<br>• arXiv Preprint | Determines manuscript formatting (IEEE two-column vs. Springer LNCS) and strict page limits (6, 8, or 12 pages). |
| **2** | **Full 19-Stage End-to-End Live Run** | • Execute pipeline on a curated sample image on local GPU and record timing logs.<br>• Retain static unit test audit in `test_results.txt`. | Recording real end-to-end execution wall-clock time (seconds per stage) adds concrete operational credibility to Section IV. |
| **3** | **External API Integration Policy** | • Provide live Google Cloud Vision & Fact Check API credentials for real test evaluation.<br>• Present offline mock/cached evaluations as the primary paper benchmark. | Recommends mock/cached benchmarks for absolute replication without exposing private API keys or incurring external query costs. |
| **4** | **Out-of-Distribution (OOD) Generative Evaluation** | • Restrict empirical section strictly to CIFAKE (Stable Diffusion v1.4 vs CIFAR-10).<br>• Download a tiny 50-image test set of modern generators (FLUX.1, Midjourney v6) to illustrate concept drift. | Adding a modern OOD test set provides a compelling narrative for Section V on generative concept drift. |
| **5** | **VLM (Qwen2.5-VL-3B) Role in Paper** | • Present VLM as a completed offline ablation study.<br>• Frame VLM as an architected future extension deferred due to 4 GB VRAM hardware boundary. | **Recommended:** Present as an architected modular extension, citing the 4 GB VRAM boundary as a practical edge-computing constraint. |
