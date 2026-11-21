# TruthLens Reproducibility & Replication Guide

> **Document Status:** Complete Reproduction Protocol
> **Repository:** `https://github.com/deepanshu70001/Truthlens`
> **Working Branch:** `fix/truthlens-remediation`
> **Host Environment:** Windows 11, Intel Core i5-12500H, 16 GB RAM, NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)

---

## 1. System Requirements & Hardware Specification

| Component | Minimum Specification | Author Testbed Specification |
|---|---|---|
| **Operating System** | Windows 10/11 or Ubuntu 22.04 LTS | Windows 11 Home 64-bit |
| **Processor** | 6-core x86_64 CPU | Intel Core i5-12500H (12 Cores, 16 Threads) |
| **System Memory** | 16 GB RAM | 16 GB DDR4 |
| **GPU / Acceleration** | NVIDIA GPU with $\ge 4$ GB dedicated VRAM | NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MB VRAM) |
| **Python** | 3.11 – 3.12 | Python 3.12.10 |
| **PyTorch** | 2.5+ with CUDA support | PyTorch 2.7.1 |
| **Node.js** | Node v18+ | Node v22.12+, npm 10.9+ |

---

## 2. Environment Setup Protocol

### 2.1 Backend Python Environment
From repository root (`d:\Projects\Truthlens`):

```powershell
# 1. Activate or create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r backend/requirements.txt
```

### 2.2 Frontend Node Environment
From `frontend/` directory:

```bash
cd frontend
npm install
cd ..
```

---

## 3. Automated Verification & Testing Commands

### 3.1 Full Test Suite Execution
```powershell
python -m pytest backend/tests/ -q
# Expected output: 267 passed, 10 skipped in ~28 seconds
```

### 3.2 Phase 7 Governance & Regression Tests
```powershell
python -m pytest backend/tests/test_phase7.py -v
# Expected output: 7 passed in <1 second
```

### 3.3 Documentation Link Validation
```powershell
python scripts/check_doc_links.py
# Expected output: All doc links verified: 0 broken links found across repository
```

---

## 4. Empirical Benchmark Replication Commands

### 4.1 In-Memory Pre-Training Shortcut Audit
To verify container format confounding and matched augmentation neutralization:
```powershell
python experiments/audit_phase7_shortcuts.py
# Outputs: experiments/reports/phase7_shortcut_audit.json
# Verifies: raw metadata RF AUC = 1.0000; augmented metadata RF AUC = 0.5028
```

### 4.2 Matched In-Memory Augmentation Training
To train candidate models across architectures (S1, S2, R1, R2, A1) within a 4 GB VRAM envelope:
```powershell
python experiments/train_phase7.py --config experiments/configs/phase7_config.yaml
# Produces model checkpoints under experiments/checkpoints/phase7/
```

### 4.3 Post-Hoc Temperature Calibration
To fit temperature scaling parameters on the 13,000-sample calibration pool:
```powershell
python experiments/calibrate_phase7_candidate.py
# Outputs: experiments/reports/phase7_temperature_calibration.json
```

### 4.4 Locked Test Benchmark Evaluation
To evaluate candidate S2 against the frozen baseline across 10,385 locked test samples:
```powershell
python experiments/evaluate_phase7_test.py
# Outputs: experiments/reports/phase7_locked_test_results.json
# Computes: group-aware bootstrap confidence intervals (B = 1,000)
```

---

## 5. Local Application Execution

### 5.1 Run FastAPI Backend
```powershell
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be available at `http://localhost:8000/docs`.

### 5.2 Run React Frontend
```powershell
cd frontend
npm run dev
```
Interactive dashboard will be available at `http://localhost:5173`.
