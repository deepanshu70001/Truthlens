# TruthLens Phase 2 Step 3: Dataset Split & Leakage Audit Summary

**Audit Date:** 2026-11-21T12:13:17.145035+00:00  
**Protocol ID:** `truthlens-phase2-step3-group-aware-v1`  
**Split Seed:** `42`  
**Readiness Decision:** `READY_FOR_PHASE_3`  

---

## 1. Partition Allocation Overview

| Dataset | Total Images | Independent Groups | Calibration | Validation | Held-Out Test | Reserve / Excluded |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **tiny_genimage** | 35,000 | 7,362 | 12,048 | 5,958 | 7,000 | 9,994 |
| **frontier_benchmark_2026_10** | 2,976 | 747 | 588 | 297 | 2,091 | 0 |
| **wildaigc_bench** | 1,839 | 1,194 | 364 | 181 | 1,294 | 0 |

---

## 2. Mandatory Leakage Checks

| Leakage Check Item | Expected | Measured Result | Verdict |
| :--- | :---: | :---: | :---: |
| `tiny_genimage`: Cross-Role Identical SHA-256 | 0 | 0 | PASS |
| `tiny_genimage`: Cross-Role Verified Group ID Overlap | 0 | 0 | PASS |
| `frontier_benchmark_2026_10`: Cross-Role Identical SHA-256 | 0 | 0 | PASS |
| `frontier_benchmark_2026_10`: Cross-Role Verified Group ID Overlap | 0 | 0 | PASS |
| `wildaigc_bench`: Cross-Role Identical SHA-256 | 0 | 0 | PASS |
| `wildaigc_bench`: Cross-Role Verified Group ID Overlap | 0 | 0 | PASS |
| Cross-Dataset Exact SHA-256 Collisions | 0 | 0 | PASS |
| Overlap with Internal TruthLens Test Fixtures | 0 | 0 | PASS |
| Overlap with CIFAKE Training Data | 0 | 0 | PASS |

All leakage prevention checks passed with zero cross-role collisions.
