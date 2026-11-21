# TruthLens Proposed Research Paper Figure & Table Index

> **Document Status:** Complete Academic Visuals Specification  
> **Inspection Date:** 2026-10-20  
> **Target Paper Style:** IEEE / ACM Two-Column Format

---

## 1. Master List of Figures

### Figure 1: TruthLens System Architecture & Modular Infrastructure
- **Type:** Architectural schematic / block diagram.
- **Caption:** *Figure 1: End-to-end system architecture of TruthLens, depicting the React SPA client layer, asynchronous FastAPI routing layer, 19-stage analysis orchestrator, multi-model execution engines, and dual PostgreSQL/SQLite persistence layer.*
- **Sub-panels:**
  - (a) Client-Server REST interface and polling bus.
  - (b) Core orchestrator with fail-soft execution boundaries.
  - (c) Persistence and artifact filesystem.
- **Source / Generator:** `research_evidence/architecture.mmd`, `docs/ARCHITECTURE.md`.
- **Recommended Placement:** Section III (System Architecture).

---

### Figure 2: 19-Stage Asynchronous Execution Pipeline DAG
- **Type:** Directed Acyclic Graph (DAG) flowchart.
- **Caption:** *Figure 2: Execution workflow and inter-stage data dependencies across the 19 automated analysis stages, highlighting sequential extraction pathways, independent forensic probes, and final evidence fusion.*
- **Sub-panels:** Highlighting the dependency chain from Stage 5 (OCR) through Stage 10 (Fact-Check), and Stage 3 (Authenticity) to Stage 7 (Grad-CAM).
- **Source / Generator:** `backend/app/services/orchestrator.py`.
- **Recommended Placement:** Section IV (Methodology & Pipeline Implementation).

---

### Figure 3: Receiver Operating Characteristic (ROC) Baseline Comparison
- **Type:** Multi-line ROC curve plot with AUC annotations.
- **Caption:** *Figure 3: Test split ROC curves for the three benchmarked authenticity classifiers evaluated on the CIFAKE dataset (N=200): custom CNN from scratch (AUC = 0.9585), ResNet-18 frozen probe (AUC = 0.9255), and SigLIP foundation frozen probe (AUC = 0.9613).*
- **Data Source:** `results/ml/cnn_scratch_preds.csv`, `resnet18_frozen_preds.csv`, `siglip_frozen_preds.csv`.
- **Generator Script:** `scripts/ml/make_plots.py`.
- **Recommended Placement:** Section V (Experimental Evaluation).

---

### Figure 4: Training Dynamics & Validation Accuracy Convergence
- **Type:** Dual-axis or faceted line plots (Training Loss vs. Validation Accuracy over Epochs).
- **Caption:** *Figure 4: Convergence behavior of benchmark models. (a) Cross-entropy training loss progression showing rapid feature stabilization in SigLIP. (b) Validation accuracy trajectories demonstrating peak accuracy of 90.50% for CNN at epoch 16 and 89.50% for SigLIP.*
- **Data Source:** `history` arrays in `results/ml/*.json`.
- **Generator Script:** `scripts/ml/make_plots.py`.
- **Recommended Placement:** Section V (Experimental Evaluation).

---

### Figure 5: Multi-Model Confusion Matrices
- **Type:** $2 \times 2$ heatmap grids (Real vs. Synthetic).
- **Caption:** *Figure 5: Normalized test split confusion matrices for (a) CNN Scratch, (b) ResNet-18 Frozen, and (c) SigLIP Frozen, illustrating the distinct error trade-offs between high-frequency convolutional kernels and pretrained vision transformers.*
- **Data Source:** `confusion_matrix` fields in `results/ml/*.json`.
- **Recommended Placement:** Section V (Experimental Evaluation).

---

### Figure 6: Explainable AI: Layer-Targeted Grad-CAM Saliency Maps
- **Type:** Three-column comparative photographic figure.
- **Caption:** *Figure 6: Visual interpretability via layer-targeted Grad-CAM. Left: Input image (synthetic face / scene). Middle: Raw gradient activation map from SigLIP2 transformer encoder block. Right: Alpha-blended heatmap overlay localizing unnatural boundary textures and generator artifacts.*
- **Source / Generator:** `backend/app/services/xai/gradcam.py`.
- **Recommended Placement:** Section IV or V (Explainability & Interpretability).

---

### Figure 7: Classical Forensics: Error Level Analysis (ELA) Differential Maps
- **Type:** Paired visual analysis figure.
- **Caption:** *Figure 7: Forensic artifact localization using Error Level Analysis (Q=95). (a) Authentic image displaying uniform compression degradation across all frequency bands. (b) Digitally spliced composite exhibiting high-contrast error intensity along edited contour boundaries.*
- **Source / Generator:** `backend/app/services/forensics/analyzer.py`.
- **Recommended Placement:** Section IV (Visual Forensics).

---

### Figure 8: India Regional Spread Risk Geographic Choropleth
- **Type:** 36-region choropleth map of India.
- **Caption:** *Figure 8: Sociodemographic regional exposure mapping for Hindi-language disinformation text, illustrating weighted exposure across Uttar Pradesh, Bihar, and Madhya Pradesh based on Census 2011 linguistic distributions and TRAI telecom broadband density.*
- **Source / Generator:** `backend/app/services/regional/engine.py`, `frontend/src/components/RegionalMap.tsx`.
- **Recommended Placement:** Section VI (Sociodemographic Risk Assessment).

---

### Figure 9: Multi-Hazard Social Impact Severity Radar Chart
- **Type:** 4-axis radar / spider plot.
- **Caption:** *Figure 9: Quantitative risk profile generated by the Social Impact Risk Engine across four critical societal dimensions: Public Order, Electoral Integrity, Public Health, and Financial Scams.*
- **Source / Generator:** `backend/app/services/social_impact/engine.py`.
- **Recommended Placement:** Section VI (Sociodemographic Risk Assessment).

---

## 2. Master List of Tables

### Table 1: Comprehensive Baseline Performance Benchmark
- **Caption:** *Table 1: Quantitative evaluation of authenticity detection baselines on the CIFAKE benchmark subset (800 train, 200 val, 200 test; Seed: 20261014).*
- **Columns:** Model, Architecture, Total Parameters, Trainable Parameters, Epochs, Training Time (s), Accuracy, Precision, Recall, F1 Score, ROC-AUC.
- **Source:** `research_evidence/experimental_results.md`, `results/ml/comparison.csv`.
- **Placement:** Section V.

### Table 2: Module Taxonomy and Data Availability Framework
- **Caption:** *Table 2: TruthLens module categorization distinguishing empirically validated ML models, heuristic risk engines with declared DATA_UNAVAILABLE status, and external cloud verification APIs.*
- **Columns:** Module Name, Method Type, Underlying Backbone / Rules, Ground-Truth Status, Primary Metric.
- **Source:** `docs/EVALUATION.md`, `docs/VIRALITY_DATA.md`.
- **Placement:** Section III or IV.

### Table 3: Summary of Identified Failure Modes and Countermeasures
- **Caption:** *Table 3: Systematic breakdown of vulnerability vectors (concept drift, platform transcoding, cheapfakes) and corresponding system-level architectural defenses.*
- **Source:** `research_evidence/failure_analysis.md`.
- **Placement:** Section VII (Discussion & Limitations).
