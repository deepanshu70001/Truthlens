# TruthLens Reproducibility & Replication Guide

> **Document Status:** Complete Reproduction Protocol  
> **Inspection Date:** 2026-10-20  
> **Target OS:** Windows / Linux / macOS  
> **Host Environment:** Windows 11, Intel i5-12500H, 16 GB RAM, NVIDIA RTX 3050 Laptop (4 GB VRAM)

---

## 1. System Requirements & Hardware Specification

| Component | Minimum Specification | Recommended / Author Testbed |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11 or Ubuntu 22.04 LTS | Windows 11 Home 64-bit |
| **Processor** | 6-core x86_64 CPU | Intel Core i5-12500H (12 Cores, 16 Threads) |
| **System Memory** | 16 GB DDR4/DDR5 | 16 GB DDR4 |
| **GPU / Acceleration** | CUDA-capable GPU with $\ge 4$ GB VRAM | NVIDIA GeForce RTX 3050 Laptop (4,096 MB VRAM) |
| **CUDA Driver** | CUDA 11.8 or higher | CUDA 11.8 runtime via PyTorch 2.7.1 |
| **Python** | 3.10 – 3.12 | Python 3.12.10 |
| **Node.js / npm** | Node v18+ | Node v22.23.2, npm 10.9.8 |

---

## 2. Environment Setup Protocol

### 2.1 Backend Python Environment
From the repository root (`c:\Users\DELL\Desktop\final year project`):

```powershell
# 1. Create a dedicated Python virtual environment
python -m venv venv

# 2. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# 3. Upgrade core packaging tools
python -m pip install --upgrade pip setuptools wheel

# 4. Install CUDA-enabled PyTorch (CUDA 11.8 build matching testbed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 5. Install backend dependencies
pip install -r backend/requirements.txt
```

### 2.2 Environment Configuration File (`.env`)
Create a `.env` file in the repository root:

```ini
APP_NAME=TruthLens
APP_ENV=development
APP_VERSION=0.1.0
DEBUG=True
HOST=0.0.0.0
PORT=8000

# Storage & Path Settings
STORAGE_PATH=data/artifacts
HEATMAP_PATH=data/heatmaps
DEMO_MODE=True

# Database Connection (Default: zero-config SQLite)
DATABASE_URL=sqlite+aiosqlite:///./truthlens.db

# Hardware Configuration
DEVICE=cuda
MAX_VRAM_GB=4.0

# Optional External API Credentials (leave empty for mock fallbacks)
GOOGLE_VISION_API_KEY=
GOOGLE_FACT_CHECK_API_KEY=
```

---

## 3. Data Preparation & Baseline Replication

### 3.1 Generate Synthetic Scenarios & Fetch Benchmark Subsets
```powershell
# Generate the 500-sample synthetic scenarios (virality, regional spread, social impact)
python scripts/generate_synthetic_data.py

# Download / verify the CIFAKE stratified 1,200-image benchmark subset
python scripts/download_cifake_subset.py
```

### 3.2 Train & Evaluate ML Baselines (EXP-01, EXP-02, EXP-03)
To replicate the exact baseline results recorded in `results/ml/`:

```powershell
# Train CNN Scratch (20 epochs), ResNet-18 Frozen (10 epochs), and SigLIP Frozen (50 epochs)
python scripts/ml/train_baselines.py --data-dir data/external/cifake --results-dir results/ml --seed 20261014

# Compile comparative tables and markdown summaries
python scripts/ml/make_comparison.py --results-dir results/ml
```

Expected artifacts produced:
- `results/ml/cnn_scratch.json`, `cnn_scratch_preds.csv`
- `results/ml/resnet18_frozen.json`, `resnet18_frozen_preds.csv`
- `results/ml/siglip_frozen.json`, `siglip_frozen_preds.csv`
- `results/ml/comparison.csv`, `results/ml/comparison.md`

---

## 4. Launching the Full System

### 4.1 Launch FastAPI Backend Server
```powershell
# Start Uvicorn ASGI server with hot reloading
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
- Swagger API Documentation: `http://127.0.0.1:8000/docs`
- ReDoc Interactive Documentation: `http://127.0.0.1:8000/redoc`

### 4.2 Launch React Frontend Application
In a separate terminal window:
```powershell
cd frontend
npm install
npm run dev
```
- Web Application Dashboard: `http://localhost:5173/`

---

## 5. Verification & Test Suite Execution

Run automated unit and integration tests across the 19 pipeline stages:
```powershell
# Run API endpoint contracts
pytest backend/tests/test_api.py -v

# Run dataset and scenario inspection tests
pytest backend/tests/test_datasets_api.py -v

# Run classical forensic and metadata tests
pytest backend/tests/test_services.py -v

# Run authenticity and Grad-CAM tests
pytest backend/tests/test_stage3.py -v

# Run multilingual NLP and OCR extraction tests
pytest backend/tests/test_stage4.py -v

# Run fusion, virality, regional spread, and social impact tests
pytest backend/tests/test_stage5.py -v
```
