#!/usr/bin/env bash
# Run both experiments, then build the comparison table and figures.
# Run from the repo root:  bash scripts/ml/run_all.sh
set -euo pipefail
python scripts/ml/train_baselines.py --model cnn
python scripts/ml/train_baselines.py --model siglip
python scripts/ml/make_comparison.py
python scripts/ml/make_plots.py
