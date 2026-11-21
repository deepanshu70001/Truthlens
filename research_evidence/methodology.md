# TruthLens Methodology

> **Document Status:** Complete & Traced to Implementation  
> **Inspection Date:** 2026-10-20  
> **Primary References:** `backend/app/services/`, `scripts/ml/train_baselines.py`, `docs/ARCHITECTURE.md`

---

## 1. Methodological Framework Overview

TruthLens addresses the challenge of multimodal misinformation through a disciplined, six-stage pipeline of evidence collection, feature extraction, forensic probes, multi-criteria fusion, and sociotechnical impact estimation. 

Rather than relying on a single deep neural network whose internal representations are opaque and subject to concept drift, TruthLens implements an **auditable evidential hierarchy**. Each module functions as an independent evidentiary probe that returns a calibrated signal, confidence level, and execution metadata.

```
┌────────────────────────────────────────────────────────┐
│  Stage I: Multimodal Signal Extraction & Preprocessing │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│  Stage II: Deep Synthetic Detection & Grad-CAM XAI    │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│  Stage III: Visual Forensics & Provenance Tracking    │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│  Stage IV: Multilingual NLP & Claim Verification      │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│  Stage V: Multi-Criteria Evidence Fusion Engine        │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│  Stage VI: Sociodemographic Impact & Virality Analysis │
└────────────────────────────────────────────────────────┘
```

---

## 2. Stage I: Multimodal Signal Extraction & Preprocessing

When an image $\mathbf{I} \in \mathbb{R}^{H \times W \times C}$ is ingested:
1. **Cryptographic & Integrity Verification:**
   - Evaluates SHA-256 digest: $h_{\text{SHA}} = \text{SHA256}(\mathbf{I})$.
   - Validates MIME type and magic-byte headers against allowed container specifications (`image/jpeg`, `image/png`, `image/webp`).
2. **Metadata & Provenance Excavation:**
   - Parses EXIF metadata tags: camera make/model, lens parameters, original timestamp, editing software tags (e.g., "Photoshop", "GIMP", "Stable Diffusion"), and GPS coordinates.
3. **Image Quality Metrics:**
   - **Laplacian Variance (Blur):** $Q_{\text{blur}} = \text{Var}(\nabla^2 \mathbf{I}_{\text{gray}})$, where values below $100.0$ flag motion or compression blur.
   - **Noise Variance:** Evaluates high-frequency noise variance via median filter residual: $\mathbf{R}_{\text{noise}} = \mathbf{I}_{\text{gray}} - \text{MedianFilter}(\mathbf{I}_{\text{gray}}, k=3)$.
   - **JPEG Block Artifact Grid:** Measures 8×8 DCT grid boundary discontinuities.

---

## 3. Stage II: Deep Synthetic Detection & Explainability (XAI)

### 3.1 Primary Classifier: SigLIP2 Vision Transformer
The primary authenticity detector employs `prithivMLmods/Deepfake-Real-Class-Siglip2`, built upon `google/siglip2-base-patch16-224`.
- The input image $\mathbf{I}$ is transformed to tensor $\mathbf{X} \in \mathbb{R}^{3 \times 224 \times 224}$ using bicubic interpolation and standard SigLIP normalization ($\mu=[0.5, 0.5, 0.5], \sigma=[0.5, 0.5, 0.5]$).
- The patch embedding layer projects non-overlapping $16 \times 16$ patches into hidden dimension $D=768$.
- Classification logits $\mathbf{z} = [z_{\text{real}}, z_{\text{fake}}] \in \mathbb{R}^2$ are transformed via softmax:
  $$p_{\text{fake}} = \frac{e^{z_{\text{fake}}}}{e^{z_{\text{real}}} + e^{z_{\text{fake}}}}, \quad p_{\text{real}} = 1 - p_{\text{fake}}$$
- **Confidence Metric:** $c_{\text{auth}} = \max(p_{\text{fake}}, p_{\text{real}})$.
- **Label Assignment:**
  $$\text{Label} = \begin{cases} \text{AI\_GENERATED}, & \text{if } p_{\text{fake}} \ge 0.65 \\ \text{AUTHENTIC}, & \text{if } p_{\text{fake}} \le 0.35 \\ \text{UNCERTAIN}, & \text{otherwise} \end{cases}$$

### 3.2 Explainable AI via Grad-CAM
To satisfy auditability requirements, the system computes Gradient-weighted Class Activation Mapping (Grad-CAM) targeting the final transformer encoder block:
1. Let $A^k$ denote the feature activation map of channel $k$ in the target layer, and $y^c$ the score for class $c$ ($c = \text{fake}$).
2. Neuron importance weights $\alpha_k^c$ are computed via global average pooling of gradients:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
3. The spatial saliency map $L_{\text{Grad-CAM}}^c$ is synthesized:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
4. The map is upsampled via bilinear interpolation to original dimensions $(H, W)$, normalized to $[0, 1]$, and pseudo-colored using the OpenCV `COLORMAP_JET` overlay.

---

## 4. Stage III: Visual Forensics & Provenance Tracking

1. **Error Level Analysis (ELA):**
   - The input $\mathbf{I}$ is resaved as JPEG at a fixed quality factor $Q_0 = 95$ yielding $\mathbf{I}_{95}$.
   - The absolute difference image is computed:
     $$\mathbf{D}_{\text{ELA}} = |\mathbf{I} - \mathbf{I}_{95}| \times \gamma, \quad \gamma = 10$$
   - Uniform error distributions characterize unedited authentic images, whereas localized spikes in high-frequency regions indicate digital splicing or localized inpainting.
2. **Multi-Scale Perceptual Hashing:**
   - Computes 64-bit perceptual hashes: **pHash** (DCT low-frequency coefficients), **dHash** (horizontal gradient differences), and **aHash** (mean intensity thresholding).
   - Enables fast duplicate detection against known debunked imagery via Hamming distance threshold $d_H \le 5$.
3. **Reverse Image Provenance:**
   - Integrates Google Cloud Vision Web Detection to query historical web index appearances, identifying first known crawl date, source URLs, and visually matching pages.

---

## 5. Stage IV: Multilingual Text & Claim Verification

1. **Optical Character Recognition (OCR):**
   - Implemented via EasyOCR (CRAFT detector + CRNN recognizer) with fallback to PyTesseract.
   - Configured for multi-script extraction: Devanagari (Hindi, Marathi), Bengali, Tamil, Telugu, and Latin (English).
2. **Linguistic Normalization & Script Identification:**
   - Cleans unicode formatting, removes meme punctuation noise, and tags language codes.
3. **Claim Extraction:**
   - Identifies declarative factual assertions using syntactic dependency parsing and entity recognition.
4. **Fact-Checking Knowledge Base Query:**
   - Queries the Google Fact Check Tools API with normalized claim strings.
   - Extracts existing reviews from verified signatories of the International Fact-Checking Network (IFCN) (e.g., BOOM Live, Alt News, Quint WebQoof, Vishwas News).

---

## 6. Stage V: Multi-Criteria Evidence Fusion Engine

The fusion engine (`backend/app/services/fusion/engine.py`) synthesizes heterogeneous signals into a unified credibility assessment.

### Mathematical Formulation
Let $\mathcal{E} = \{e_1, e_2, \dots, e_n\}$ be the set of collected evidence items. Each item $e_i$ is characterized by:
- Polarity $p_i \in \{+1 \text{ (supports authentic)}, -1 \text{ (supports fake)}, 0 \text{ (neutral)}\}$
- Internal confidence $c_i \in [0.0, 1.0]$
- Modality reliability weight $w_i \in [0.0, 1.0]$

Modality weights $w_i$ are calibrated based on empirical error bounds:
- Fact-Check match (IFCN verified debunk): $w = 0.95$
- Primary Authenticity Model (SigLIP2): $w = 0.80$
- Classical Forensics (ELA / Noise inconsistency): $w = 0.50$
- Metadata anomalies (Editing software string): $w = 0.40$
- Reverse Image Match (Recycled context): $w = 0.70$

The normalized global authenticity score $S \in [0, 1]$ is computed as:
$$S = \frac{\sum_{i=1}^n w_i \cdot \left(\frac{1 + p_i \cdot c_i}{2}\right)}{\sum_{i=1}^n w_i}$$

### Conflict Resolution & Categorization
- If high-confidence signals diverge (e.g. model predicts $p_{\text{fake}} = 0.92$, but reverse search finds authentic 2018 news photograph), the system flags **CONTEXTUAL_MISUSE** rather than synthetic generation.
- Categorical outputs: `AUTHENTIC`, `AI_GENERATED`, `DIGITALLY_MANIPULATED`, `OUT_OF_CONTEXT`, or `INCONCLUSIVE`.

---

## 7. Stage VI: Sociodemographic Impact & Virality Analysis

### 7.1 Virality Estimation Heuristics
Because no ground-truth forward traces exist for closed messaging apps, the engine computes a heuristic Virality Risk Index ($V \in [0, 100]$):
$$V = 0.30 \cdot S_{\text{sentiment}} + 0.25 \cdot S_{\text{entity}} + 0.25 \cdot S_{\text{sensationalism}} + 0.20 \cdot S_{\text{visual}}$$
- **Crucial Rule:** The API explicitly returns `data_status: "DATA_UNAVAILABLE"` to signify that this score represents a structural risk assessment, not an empirical forecasting model.

### 7.2 Indian Regional Spread Engine
Maps content vulnerability across 36 Indian States and Union Territories:
- Correlates detected language script $\mathcal{L}$ with regional primary speaker distributions (Census of India 2011).
- Weights demographic risk using telecom broadband penetration metrics (TRAI quarterly statistics).
- Identifies Top-5 high-exposure states (e.g. Hindi text with communal keywords flags Uttar Pradesh, Bihar, Madhya Pradesh, Rajasthan, and Haryana).

### 7.3 Multi-Hazard Social Impact Severity Matrix
Combines the fused authenticity verdict, fact-check ratings, and regional exposure into four societal risk dimensions:
1. **Public Order & Communal Harmony**
2. **Electoral & Democratic Integrity**
3. **Public Health & Safety**
4. **Financial & Institutional Trust**

Each dimension is scored into qualitative severity tiers: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`, accompanied by actionable civil advisory recommendations.
