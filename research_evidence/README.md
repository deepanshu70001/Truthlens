# TruthLens — Research Evidence Package

> **Inspection date:** 2026-10-20
> **Repository:** `c:\Users\DELL\Desktop\final year project`
> **Git HEAD:** `270afbb` (6 commits total)
> **Project version:** 0.1.0 (as declared in config)

## Scope

This package collects verifiable evidence from the current TruthLens repository to support writing a draft research paper. Every claim is traced to specific source files. Unverified information is marked as such.

## Package Contents

| # | File | Description |
|---|------|-------------|
| 1 | `README.md` | This file — scope, date, unresolved questions |
| 2 | `project_inventory.md` | Repository structure and component responsibilities |
| 3 | `implementation_status.csv` | Feature-by-feature status with evidence |
| 4 | `plan_vs_current_changes.md` | Original plan vs. current implementation |
| 5 | `architecture.mmd` | Mermaid diagram of actual architecture |
| 6 | `architecture_explanation.md` | Component and data-flow explanation |
| 7 | `model_inventory.md` | ML models, versions, roles, limitations |
| 8 | `dataset_inventory.md` | Datasets, sources, sizes, splits, limitations |
| 9 | `methodology.md` | Current methodology as implemented |
| 10 | `experiment_inventory.md` | Completed and proposed experiments |
| 11 | `experimental_results.md` | Real measured results with context |
| 12 | `test_results.txt` | Available tests (not executed — see notes) |
| 13 | `failure_analysis.md` | Known failures, edge cases, gaps |
| 14 | `reproducibility.md` | Environment, setup, run commands |
| 15 | `api_inventory.md` | Actual API endpoints and schemas |
| 16 | `limitations.md` | Technical, scientific, ethical limitations |
| 17 | `references.md` | Verified and unverified references |
| 18 | `figure_index.md` | Proposed research-paper figures |
| 19 | `research_questions.md` | Questions requiring your input |
| 20 | `implementation_summary.md` | Concise summary for paper writing |
| 21 | `metrics.json` | Machine-readable metrics |
| 22 | `paper_alignment.md` | Suggested title, scope, objectives |

## Unresolved Questions

See `research_questions.md` for the full list. Key items:

1. Has the full 19-stage pipeline been executed end-to-end on real images? (No execution logs found in the repository.)
2. Are the CIFAKE images currently downloaded locally? (Only `manifest.json` found in `data/external/cifake/`.)
3. Which external APIs (Google Vision, Fact-Check) have been tested with real credentials?
4. Has the VLM (Qwen2.5-VL) been loaded and tested within the pipeline?
5. What is the intended submission venue — conference, journal, or internal report?

## Important Caveats

- **Tests were NOT executed** during this inspection to avoid installing missing dependencies or consuming API credits.
- **No source code was modified.**
- The three ML baseline experiments (CNN, ResNet-18, SigLIP) have real, recorded results in `results/ml/`. These are the only measured experimental results found.
- The heatmaps directory (`data/heatmaps/`) is empty — no actual Grad-CAM outputs were found.
