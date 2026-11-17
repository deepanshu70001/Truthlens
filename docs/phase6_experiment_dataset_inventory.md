# Phase 6 experiment dataset inventory

Inventory date: 2026-11-17. Files were counted using the aggregate script in `experiments/dataset_audit.py`; rights and labels require documentary review and cannot be inferred from file bytes. The 35-image scan yielded 496 dHash pairs at Hamming distance ≤2; these are loose candidates, not confirmed duplicates or an exclusion count.

| Dataset/source | Type / source | Count/coverage | Labels/languages/geography | License / annotation / limitations | Experimental role |
|---|---|---|---|---|---|
| `test_data/originals/` | Internal test assets; origin not documented | 35 readable images: 31 JPEG, 4 PNG; 31 exact duplicate extra copies; 33 exact dHash candidate extra copies | No labels, language or geography metadata | No source or license manifest; likely includes repeated fixture-scale images; source events/splits unknown | Software code-path fixtures only; excluded from science results |
| `test_data/` derived artifacts | Internal resized images, thumbnails, heatmaps, and originals | 190 image files total across tree; 172 JPEG, 18 PNG; many derived/duplicate copies | No ground truth | Not independent records; generated artifacts tied to prior analysis IDs | Excluded from model evaluation |
| `backend/tests/fixtures/test_photo.jpg` | Generated test fixture if absent; created by test helper | One nominal fixture | Synthetic visual pattern; no real/AI class semantics | Software test only; not a natural-image sample | API/unit integration only |
| `data/` and `temp/` runtime storage | Application/user input and derived outputs | Present; not inventoried as research data | No reviewed truth labels | Privacy-sensitive user/application records; intentionally not scanned for research | Excluded from all experiments |
| SigLIP2 model cache | `prithivMLmods/Deepfake-Real-Class-Siglip2` configured checkpoint | Cached checkpoint previously observed (~743 MB aggregate cache) | Model labels fake/real; checkpoint training provenance is external | Hugging Face model card currently declares Apache-2.0 and identifies the SigLIP2 base; pin an immutable Hub revision and review upstream training-data provenance before submission. No TruthLens-specific benchmark | Inference smoke test only; not evaluation data |
| FakeNewsNet, PHEME, MuMiN, IFND, GenImage, RRDataset | External research candidate datasets | Not installed in this checkout | Different labels/task/scope; several may require gated access or careful media rights | Must review each official dataset release and media reuse rights before retrieval or use; incompatible sources cannot be pooled without protocol | Candidate reading list only; none used for metrics |

## Data-quality and split findings

No research manifest, source-event IDs, generator IDs, train/validation/test split, human labels, or label adjudication record was found. Thus duplicate leakage cannot be evaluated across splits. Aggregate `test_data/originals` results show extensive exact byte repetition (31 extra copies) and exact dHash repeats (33 extra copies). Hash candidate counts are not proof of near-duplicate identity and do not justify silent removal. `test_data/` includes derived copies and must not be counted as 190 unique samples.

No cleaning exclusions were applied. Invalid image count was zero in the 35-file originals scan. The audit reports dHash candidate pairs within Hamming distance 2 as candidates only; they are not silently removed. The experiment manifest requires source/permission references, binary labels, group IDs and split assignment; see `experiments/manifests/authenticity.example.csv`. Difficult cases must remain in the manifest with documented labels or exclusion reasons.

## External references (not used as data)

- FakeNewsNet: https://github.com/KaiDMML/FakeNewsNet
- PHEME: https://www.pheme.eu/software-downloads/
- MuMiN: https://mumin-dataset.github.io/
- GenImage: https://arxiv.org/abs/2306.08571
- RRDataset (ICCV 2025 paper): https://openaccess.thecvf.com/content/ICCV2025/html/Li_Bridging_the_Gap_Between_Ideal_and_Real-world_Evaluation_Benchmarking_AI-Generated_ICCV_2025_paper.html

These entries are pointers for rights and scope review, not claims that TruthLens used or has access to these datasets.
