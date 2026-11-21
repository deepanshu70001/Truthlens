# TruthLens Dataset Inventory

> **Document Status:** Verified Against Filesystem & Data Directories  
> **Inspection Date:** 2026-10-20  
> **Primary References:** `data/`, `docs/VIRALITY_DATA.md`, `scripts/download_cifake_subset.py`, `scripts/generate_synthetic_data.py`

---

## 1. Overview of Data Assets in Repository

TruthLens interfaces with three categories of datasets:
1. **Empirical Benchmark Datasets (Local Subsets):** Used for training and benchmarking authenticity classifiers.
2. **Synthetic Simulation Scenarios (Repository Generated):** Structured scenarios generated to stress-test the virality, regional spread, and social impact scoring engines.
3. **Investigated External Datasets (Literature & Feasibility Survey):** Catalogued during the initial data availability study to evaluate the feasibility of supervised training.

---

## 2. Dataset Master Inventory Table

| Dataset Identifier | Domain / Task | Origin / Source | Sample Size (Repo Status) | Variables / Modalities | License | Repository Path |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CIFAKE (Subset)** | Synthetic vs. Real Image Classification | Bird & Lotfi (2023) / Kaggle | 1,200 images referenced in manifest (800 train, 200 val, 200 test) | 32×32 RGB images (CIFAR-10 real vs Stable Diffusion v1.4 synthetic), binary labels | CC0 / Research | `data/external/cifake/manifest.json` |
| **Synthetic Virality Scenarios** | Virality Risk Simulation & Pipeline Testing | Self-generated script (`scripts/generate_synthetic_data.py`) | 500 scenario rows (87.3 KB) | Content length, emotional triggers, sensationalism index, claim count, forward velocity index | Open / Academic | `data/synthetic/virality_scenarios.csv` |
| **Synthetic Regional Spread** | Indian Regional Vulnerability Simulation | Self-generated script (`scripts/generate_synthetic_data.py`) | 500 scenario rows (80.4 KB) | Detected language, state/UT code (36 entities), internet penetration, literacy, risk band | Open / Academic | `data/synthetic/regional_spread_scenarios.csv` |
| **Synthetic Social Impact** | Societal Hazard Stress-Testing | Self-generated script (`scripts/generate_synthetic_data.py`) | 500 scenario rows (87.2 KB) | Hazard category (Communal, Health, Financial, Electoral), severity score, demographic vulnerability | Open / Academic | `data/synthetic/social_impact_scenarios.csv` |
| **BharatFakeNewsKosh** | Indian Multilingual Misinformation | Kaggle / Academic Survey | ~26,000 articles (Investigated) | Headline, body, 9 Indian languages, fact-check verdict | CC BY-NC 4.0 | Documented in `docs/VIRALITY_DATA.md` |
| **FactDrill** | Indian Fact-Checked Multimodal Posts | Precog IIIT-Delhi | ~22,000 items (Investigated) | Multilingual text (13 Indian languages), claims, fact-check ratings, domains | Academic Use | Documented in `docs/VIRALITY_DATA.md` |
| **IFND (Indian Fake News Dataset)** | Multimodal Indian Misinformation | Zenodo / ResearchGate | ~10,000 articles (Investigated) | Image + text pairs, publication date, claim labels | Research Use | Documented in `docs/VIRALITY_DATA.md` |
| **FakeNewsNet** | Social Media Virality & Misinformation | ASU DMML / PolitiFact / GossipCop | ~23,000 articles (Investigated) | News content, social network graphs, retweet timestamps | MIT | Documented in `docs/VIRALITY_DATA.md` |
| **MuMiN** | Multilingual Multimodal Misinformation | MUMIN Consortium | 21M tweets, 12k claims (Investigated) | Multilingual claim-tweet knowledge graph, 41 languages | CC BY-NC-SA 4.0 | Documented in `docs/VIRALITY_DATA.md` |
| **CREDBANK** | Event Credibility & Temporal Streams | MIT / Social Computing | ~60M tweets across 1,049 events (Investigated) | Tweet text, temporal volume, crowd credibility ratings | Academic Use | Documented in `docs/VIRALITY_DATA.md` |

---

## 3. Detailed Data Analysis

### 3.1 CIFAKE Benchmark Subset
- **Composition:** CIFAKE consists of 60,000 synthetic images generated via Stable Diffusion v1.4 and 60,000 authentic images from CIFAR-10.
- **Local Implementation:** A stratified, balanced subset of 1,200 images was constructed with a deterministic seed (`20261014`):
  - **Train split:** 800 images (400 real, 400 fake)
  - **Validation split:** 200 images (100 real, 100 fake)
  - **Test split:** 200 images (100 real, 100 fake)
- **Manifest:** Recorded in `data/external/cifake/manifest.json` (402.5 KB) including image indices, labels, and file paths.

### 3.2 Synthetic Misinformation Scenarios
Because real-world virality data is proprietary to platforms like WhatsApp and Telegram, TruthLens created reproducible synthetic datasets:
1. `virality_scenarios.csv`: Models correlations between visual quality, emotional claim density, OCR word counts, and potential forwarding velocity.
2. `regional_spread_scenarios.csv`: Integrates demographic baselines from the Census of India 2011 and TRAI Telecom Reports across all 36 States and Union Territories.
3. `social_impact_scenarios.csv`: Evaluates simulated fallout across 4 distinct hazard categories:
   - **Communal Harmony / Public Order:** Sensitivity to inter-faith tensions.
   - **Public Health:** False medical remedies and vaccine rumors.
   - **Electoral Integrity:** Deepfake impersonation of political figures during polling cycles.
   - **Financial Scams:** Fraudulent investment schemes with fabricated institutional letterheads.

---

## 4. Ground-Truth Data Availability Limitations

The comprehensive audit in `docs/VIRALITY_DATA.md` established critical empirical findings for academic reporting:
1. **Absence of Image-Level Virality Traces:** Existing academic datasets (FakeNewsNet, MuMiN, CREDBANK) capture Twitter/X text engagement (retweets, likes). None capture end-to-end forward cascades for standalone image memes within closed messaging networks (WhatsApp/Telegram), which represent >80% of Indian misinformation propagation.
2. **Temporal Decay:** Twitter datasets from 2015–2021 are subject to massive link-rot and deleted posts, severely hindering reproducibility.
3. **Research Integrity Decision:** Based on these findings, the paper authors elected **not** to fit deep neural networks on synthetic virality metrics, avoiding false claims of predictive capability.
