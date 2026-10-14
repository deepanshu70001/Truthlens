# TruthLens Dataset Guide

## 1. Purpose

This document describes the datasets prepared for TruthLens and explains how to validate, regenerate, and access them.

The repository contains two categories of data:

1. **Synthetic scenario datasets** for demonstrating virality, regional spread, and social impact workflows.
2. **CIFAKE image subset** for experimenting with real-versus-AI-generated image classification.

These datasets serve different purposes and must not be treated as interchangeable.

## 2. Synthetic Scenario Datasets

Location: `data/synthetic/`

| File | Records | Purpose |
|---|---:|---|
| `virality_scenarios.csv` | 500 | Illustrative content-spread scenarios |
| `regional_spread_scenarios.csv` | 500 | Illustrative regional-spread scenarios |
| `social_impact_scenarios.csv` | 500 | Illustrative social-impact scenarios |
| `manifest.json` | — | Dataset metadata and limitations |

### Important limitations

- All three CSV files contain synthetically generated scenarios, not real-world observations.
- Their scores are illustrative scenario values, not predictions from trained models.
- The regional data does not establish actual state-level spread or audience behavior.
- These datasets must not be used alone to claim model accuracy or real-world effectiveness.
- Do not mix synthetic scenario records with the CIFAKE image dataset.

### Regenerate the synthetic datasets

From the repository root, run:

```powershell
python scripts/generate_synthetic_data.py
```

**Note:** Regeneration may overwrite the existing CSV files. Review the script before changing its seed or generation logic.

### Validate the synthetic datasets

```powershell
python scripts/validate_datasets.py
```

A successful validation reports that all three CSV datasets passed the implemented checks. Validation checks data integrity and expected structure; it does not establish predictive accuracy.

## 3. CIFAKE Image Subset

Source dataset: `dragonintelligence/CIFAKE-image-dataset` on Hugging Face.

Download page: https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset

Local location: `data/external/cifake/`

The exported subset contains 1,200 images:

| Split | Real | AI-generated | Total |
|---|---:|---:|---:|
| Train | 500 | 500 | 1,000 |
| Test | 100 | 100 | 200 |
| **Total** | **600** | **600** | **1,200** |

The source dataset uses `REAL` and `FAKE` labels. In the exported subset, `FAKE` is mapped to `ai_generated`. Always consult the source dataset card for its precise label definitions.

### Download or regenerate the subset

Install the project's required Python dependencies first. Then run:

```powershell
python scripts/download_cifake_subset.py
```

The script downloads the source dataset, selects a reproducible subset using the configured seed, exports images, and creates `data/external/cifake/manifest.json`.

To view the script's command-line help:

```powershell
python scripts/download_cifake_subset.py --help
```

The image directories are excluded from Git to avoid committing large numbers of binary files. Each teammate who needs the actual images should run the download script locally.

### CIFAKE limitations

- The subset is intended for pipeline testing and baseline image-classification experiments.
- Images are commonly low-resolution; check the source dataset card for details.
- Performance on CIFAKE does not establish the ability to detect all AI-generated images, deepfakes, image manipulation, or misinformation.
- A dataset label is not universal proof of an image's authenticity.
- Review the source dataset's current license and terms before redistribution or commercial use.

## 4. Dataset API

The backend exposes read-only endpoints for the synthetic scenario datasets when demo mode is enabled.

Start the backend from the repository root:

```powershell
$env:DEMO_MODE = "true"
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In another terminal, list the available datasets:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/datasets
```

Retrieve a page of records, for example:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/datasets/virality_scenarios?limit=10&offset=0"
```

Available dataset names:

- `virality_scenarios`
- `regional_spread_scenarios`
- `social_impact_scenarios`

Pagination uses `limit` and `offset`. Check the API response for dataset labels and disclaimers. These endpoints expose synthetic scenario records; they do not turn their illustrative scores into trained model predictions.

## 5. Recommended Workflow for the ML/DL

1. Read this guide and inspect `data/synthetic/manifest.json`.
2. Run `python scripts/validate_datasets.py`.
3. Download CIFAKE locally if performing image-classification experiments.
4. Inspect the source dataset card and licensing terms before use.
5. Keep image-classification labels separate from synthetic scenario labels.
6. Establish a baseline with an appropriate train/test split and documented evaluation metrics.
7. Report limitations and avoid claiming general deepfake-detection accuracy from CIFAKE alone.
8. Keep downloaded image files and model weights out of Git unless the team explicitly approves a suitable, licensed distribution method.

## 6. Run the Backend Tests

From the repository root:

```powershell
python -m pytest backend/tests -q
```

The tests verify implemented application behavior. A passing test suite does not by itself establish the real-world accuracy of any detection or risk-estimation system.

## 7. Reproducibility and Responsible Use

- Synthetic dataset generation uses the configured seed `20261014`.
- The CIFAKE subset script also uses seed `20261014`.
- Preserve the manifests when sharing dataset metadata.
- Record changes to the source dataset, seed, schema, and preprocessing when updating a dataset.
- Never present simulated values as observed facts or use a model score as conclusive proof that content is true, false, real, or fake.
