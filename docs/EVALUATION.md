# TruthLens — Evaluation Framework

> **Document Status**: Living document — updated as modules are validated.
> **Last Updated**: 2026-10-02

---

## 1. Philosophy

TruthLens is an AI-assisted verification system. Evaluation must be honest:

- **Report only measured metrics** — never fabricate accuracy numbers
- **Distinguish between validated and heuristic modules**
- **Use appropriate metrics per task type**
- **Document limitations alongside results**

---

## 2. Module Classification

### Trained ML Models (Validated on Test Sets)

| Module | Task Type | Model | Status |
|--------|-----------|-------|--------|
| Authenticity Detection | Multi-class classification | SigLIP2 (Deepfake/Real) | ✅ Model loaded; evaluation on TruthLens-specific data pending |
| OCR | Text extraction | EasyOCR | ✅ Working; standard benchmark comparison pending |
| Language Detection | Classification | XLM-RoBERTa | ✅ Working; accuracy on Indian languages pending |
| Object Detection | Detection | YOLOv8n | ✅ Working; uses COCO-pretrained weights |

### Heuristic / Rule-Based Modules (No Training Data)

| Module | Status | Data Status | Validation |
|--------|--------|-------------|------------|
| Virality Estimation | ✅ Implemented | `DATA_UNAVAILABLE` | No validation possible — no ground truth |
| Regional Spread | ✅ Implemented | `DATA_UNAVAILABLE` | No validation possible — no ground truth |
| Social Impact | ✅ Implemented | `DATA_UNAVAILABLE` | No validation possible — no ground truth |
| Evidence Fusion | ✅ Implemented | N/A (aggregation) | Evaluation requires labeled multi-signal corpus |
| Forensic Analysis | ✅ Implemented | N/A (classical methods) | Signal-level validation pending |

### External API Modules

| Module | Status | Validation |
|--------|--------|------------|
| Google Fact Check API | ✅ Integrated (optional) | Depends on API availability and query quality |
| Google Vision Web Detection | ✅ Integrated (optional) | Depends on API availability and image indexing |

---

## 3. Metrics by Task Type

### 3.1 Classification Tasks

For authenticity detection (binary/multi-class):

| Metric | Description | When to Use |
|--------|-------------|-------------|
| **Precision** | TP / (TP + FP) | When false positives are costly |
| **Recall** | TP / (TP + FN) | When false negatives are costly |
| **F1 Score** | Harmonic mean of P and R | General balanced evaluation |
| **ROC-AUC** | Area under ROC curve | Threshold-independent evaluation |
| **PR-AUC** | Area under Precision-Recall curve | Imbalanced datasets |
| **Confusion Matrix** | Full TP/FP/TN/FN breakdown | Detailed error analysis |

### 3.2 Regression Tasks

For virality score estimation (if trained on real data in future):

| Metric | Description | When to Use |
|--------|-------------|-------------|
| **MAE** | Mean Absolute Error | Interpretable average error |
| **RMSE** | Root Mean Squared Error | Penalizes large errors |
| **R²** | Coefficient of determination | Explained variance |
| **Spearman ρ** | Rank correlation | Ordinal ranking quality |

### 3.3 Ranking Tasks

For regional spread risk ranking:

| Metric | Description | When to Use |
|--------|-------------|-------------|
| **Spearman ρ** | Rank correlation coefficient | Comparing predicted vs. actual rank order |
| **NDCG** | Normalized Discounted Cumulative Gain | Top-k ranking quality |
| **Kendall τ** | Concordance measure | Pairwise agreement |

---

## 4. Current Evaluation Status

### 4.1 Authenticity Detection

**Model**: `prithivMLmods/Deepfake-Real-Class-Siglip2`

| Metric | Value | Dataset | Note |
|--------|-------|---------|------|
| Reported accuracy (model card) | ~93% | Training set | **Not independently validated by TruthLens** |
| TruthLens-specific evaluation | **Pending** | — | Requires curated test set of Indian context images |

**Limitations**:
- Model was trained on predominantly Western image datasets
- Performance on AI-generated images from newer models (2025+) is unknown
- No calibration study has been performed for this model
- Subject to concept drift as generation techniques evolve

### 4.2 Forensic Analysis (ELA, Noise, JPEG)

| Signal | Validation Status | Note |
|--------|-------------------|------|
| ELA (Error Level Analysis) | Literature-validated method | Known false positive rate with re-compressed images |
| Noise Analysis | Literature-validated method | Affected by image quality and compression |
| JPEG Artifact Analysis | Literature-validated method | Only meaningful for JPEG format |

### 4.3 Virality Estimation

**Status**: `DATA_UNAVAILABLE` — **No evaluation possible**

| Aspect | Assessment |
|--------|-----------|
| Training data | None available for Indian image-level virality |
| Validation data | None available |
| Feature weights | Literature-derived heuristics, not learned |
| Confidence calibration | Not applicable (not a trained model) |

**Why we cannot evaluate**:
- No publicly available dataset provides image-level engagement metrics in Indian context
- Platform-specific data (WhatsApp, ShareChat) is not accessible
- Cross-platform virality is fundamentally unpredictable from image features alone

### 4.4 Regional Spread Estimation

**Status**: `DATA_UNAVAILABLE` — **No evaluation possible**

| Aspect | Assessment |
|--------|-----------|
| Language data source | Census of India 2011 (official, but 15 years old) |
| Internet data source | TRAI Q4 2023-24 (official, quarterly) |
| Spread pattern data | None available |
| Validation | Not applicable |

### 4.5 Social Impact Assessment

**Status**: `DATA_UNAVAILABLE` — **No evaluation possible**

| Aspect | Assessment |
|--------|-----------|
| Methodology | Rule-based keyword detection |
| Ground truth | No measured social outcomes dataset |
| Validation | Not applicable |

---

## 5. Future Evaluation Plan

### Phase 1: Curated Test Set (Required)

Create a TruthLens-specific test set:

- [ ] 500+ images: mix of authentic, AI-generated, manipulated
- [ ] Indian context: multiple languages, topics, platforms
- [ ] Ground truth: fact-checked by reputable organizations
- [ ] Diverse: political, health, social, religious, financial
- [ ] Labeled: authenticity, manipulation type, engagement (if available)

### Phase 2: Model Evaluation (After Test Set)

- [ ] Authenticity detection: precision, recall, F1, ROC-AUC on test set
- [ ] Per-class metrics: authentic, AI-generated, manipulated
- [ ] Calibration plot: predicted probability vs. actual frequency
- [ ] Confusion matrix analysis
- [ ] Cross-model comparison (if multiple detectors)

### Phase 3: Heuristic Validation (If Data Becomes Available)

- [ ] Virality: correlation study with actual engagement metrics
- [ ] Regional: comparison with actual platform analytics
- [ ] Social impact: expert panel assessment of risk categorization

---

## 6. Reporting Metrics Honestly

### DO
- Report metrics measured on specific datasets with clear provenance
- State confidence intervals when sample size permits
- Document the test set composition and potential biases
- Report per-class metrics, not just overall accuracy
- Acknowledge when a metric is meaningless (e.g., accuracy on imbalanced data)

### DO NOT
- ❌ Report accuracy claimed by model card authors as TruthLens performance
- ❌ Claim validated accuracy for heuristic modules
- ❌ Report metrics without specifying the evaluation dataset
- ❌ Use accuracy alone on imbalanced datasets
- ❌ Imply that high accuracy on one dataset generalizes to all inputs

---

## 7. Known Limitations (Honest Assessment)

1. **No TruthLens-specific evaluation has been performed yet** — all ML model metrics are from model authors' reported results on their training/test sets, not independently validated.

2. **Concept drift** — AI-generated content detection models degrade as new generation techniques emerge. A model validated in 2024 may perform differently on 2026 content.

3. **Population bias** — Most training data for deepfake/AI detection comes from English-language, Western-context datasets. Performance on Indian vernacular content is unvalidated.

4. **Platform bias** — Available engagement/virality datasets are Twitter/Reddit-centric. Indian social media behavior (WhatsApp, ShareChat, Instagram) may follow different patterns.

5. **No causal claims** — Correlation between image features and virality does not imply causation. Content spread depends on platform algorithms, network structure, and timing — not just content.
