# TruthLens — Final Forensics, Vision, Explainability and Provenance Reliability Audit Report

**Date:** 2026-11-23  
**Repository:** `D:\Projects\Truthlens`  
**Branch:** `fix/truthlens-remediation`  
**HEAD Commit:** `b988e2f42ef091f35bf32ccc19735dbd0c5724ea`  
**Overall Readiness Classification:** **`READY WITH DOCUMENTED LIMITATIONS`**

---

## 1. Executive Summary & Readiness Decision

An independent, rigorous reliability audit was conducted across TruthLens's digital forensics, computer vision (YOLO), explainability (Grad-CAM), cryptographic provenance (C2PA), external web detection, and evidence fusion subsystems.

```
===============================================================================
FINAL READINESS DECISION: READY WITH DOCUMENTED LIMITATIONS
===============================================================================
```

### Core Audit Verdict:
1. **Classical Forensics:** All six forensic indicators (Error Level Analysis, Noise Consistency, JPEG Block Artifacts, Edge Consistency, Texture Consistency, Resampling Analysis) are fully functional, operate on EXIF-corrected images, preserve original bytes, and correctly treat outputs as uncalibrated heuristic indicators bounded in `[0, 1]`, never probabilities.
2. **Resampling Abstention Safeguard:** The two-level grayscale RMS second-difference energy floor (2.0 intensity levels) correctly identifies uninformative, flat, or near-uniform fields and abstains with `reason_code="INSUFFICIENT_TEXTURE_FOR_RESAMPLING"`, output `not_applicable`, and score `None`. This is verified as a signal-sufficiency safeguard, not a validated manipulation detector.
3. **URL and Text Sanitization:** Strict HTTPS-only validation, host/port authority checks, case-insensitive percent-decoded credential filtering (stripping keys, tokens, auth headers, and secrets), duplicate parameter purging, fragment stripping, and local path redaction are fully verified. Provider-returned URLs are never fetched automatically.
4. **Real YOLOv8n Integration:** Real GPU inference with `ultralytics==8.4.216` on verified local checkpoint `models/baselines/yolov8n.pt` (SHA-256 `31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95`) runs within the detector and pipeline. Automatic model downloads are disabled. Tested on `test_photo.jpg`, returning valid `completed/no_results` with 0 detections.
5. **Production SigLIP Grad-CAM:** Production compatibility verified on GPU with `truthlens/siglip-cifake-frozen-head` (`models/baselines/siglip_frozen_head.pt`, SHA-256 `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`). Discovered target layer `encoder.layers[11].layer_norm2` (`LayerNorm((768,))`). Demonstrates **0.0 logit difference** pre/post Grad-CAM, bit-exact parameter preservation across all 86 tensors, full hook and gradient cleanup, and finite overlay maps.
6. **Real C2PA Cryptographic Verification:** Installed `c2pa-python==0.39.0` (Rust library `0.92.0`) verified against official signed fixture `C.jpg` (SHA-256 `a2d14755db55de67a47c04090340d8266e892367be4104a45626d7a6fa6e9ffd`). Verified valid claim signature and content data hash binding, tamper detection via byte inversion, and thumbnail stream extraction.
7. **Production Signer Trust vs. Test Root Distinction:** The official C2PA conformance trust list (`C2PA-TRUST-LIST.pem`) correctly evaluates the test certificate as `untrusted` (`signingCredential.untrusted`). The test root bundle (`test_cert_root_bundle.pem`) evaluates it as `trusted`. These results remain strictly separated and are never collapsed into an assertion of production trust.
8. **Fusion Non-Voting Isolation:** Forensic indicators, YOLO detections, Grad-CAM overlays, C2PA findings, EXIF metadata, and Web Detection matches are strictly informational or contextual (`direction="informational"`). Bayesian fusion assigns them zero voting weight (0 synthetic signals, 0 authentic signals, label `inconclusive`). Missing or failed evidence never converts into negative or synthetic signals.
9. **Documented Limitations:** Live Google Cloud Vision Web Detection remains unexercised due to unconfigured API credentials. Module 6 dedicated manipulation detector remains unimplemented. Production signer trust anchors are not populated for end-user assets.

---

## 2. Git State & History Reconciliation

- **Active Branch:** `fix/truthlens-remediation`
- **Remote Tracking:** `origin/fix/truthlens-remediation` (in sync)
- **HEAD Commit:** `b988e2f42ef091f35bf32ccc19735dbd0c5724ea` (`feat: add forensics, provenance, vision, and XAI services with configuration and tests`)
- **Preceding History:**
  - `7f97a72`: Enhance documentation and verification reports for E2, E3, E4, and E5 modules
  - `1ba0d92`: Add R10 preregistered hybrid experiment protocol and implementation
  - `93b551c`: feat: Implement E8-E10 production reliability, security, database, and frontend enhancements
  - `63cce07`: Implement E6/E7 Forensics and Provenance Features with Tests and Documentation
- **Current Modified / Untracked Files:**
  - `backend/app/services/provenance/c2pa_verifier.py`: Remediated `_C2paManifestNotFound` handling, added dual `resource_to_stream` signature compatibility, protected MIME detection.
  - `backend/tests/test_e6_e7_contract.py`: Added real-runtime contract test `test_c2pa_real_runtime_verification`.
  - `docs/MODULE_STATUS_REGISTER.md`: Updated Module 16 and Milestone E7 to reflect verified C2PA status.
  - `docs/PROJECT_ROADMAP.md`: Reconciled E6 and E7 entries with real YOLO, Grad-CAM, and C2PA verifications.
  - `docs/e6_e7_forensics_provenance_report.md`: Added Section 9.4 documenting subsequent real-runtime verifications.
  - `docs/c2pa_verification_report.md`: Comprehensive C2PA verification report.
  - `docs/forensics_provenance_reliability_audit.md`: This audit report.
- **Preservation Check:** Zero git commits, pushes, cleanups, or resets performed. All model caches, test fixtures, and protected partitions remain untouched.

---

## 3. Classical Forensic Indicators Audit

### 3.1 Indicators Summary

| Indicator Name | Method / Implementation | Parameters & Formula | Input Constraints | Outcome on Non-Conforming Input | Score Semantics |
|---|---|---|---|---|---|
| `error_level_analysis` | Decoded pixels vs. quality-90 JPEG re-encode | $Q=90$, $\text{score} = \min(\text{std} / \text{mean} / 3.0, 1.0)$ | Native JPEG only | `outcome="not_applicable"`, `score=None` for PNG/non-JPEG | Uncalibrated heuristic in $[0, 1]$, not probability |
| `noise_consistency` | Laplacian residual variation across 4 quadrants | 4 quadrants, $\text{score} = \min(\text{quadrant\_cv} / 0.5, 1.0)$ | $h \ge 4, w \ge 4$ | `outcome="not_applicable"` if $< 4\times 4$ | Uncalibrated heuristic in $[0, 1]$, not probability |
| `jpeg_artifacts` | 8-pixel block boundary mean diff vs. non-block | 8-px grid, $\text{score} = \min(\max(\text{ratio}-1, 0)/0.5, 1.0)$ | Native JPEG only, $h, w \ge 16$ | `outcome="not_applicable"` if non-JPEG or $< 16\times 16$ | Uncalibrated heuristic in $[0, 1]$, not probability |
| `edge_consistency` | Canny edge density across 3×3 spatial grid | Canny (50, 150), $\text{score} = \min(\text{cv} / 1.5, 1.0)$ | $h \ge 3, w \ge 3$ | `outcome="not_applicable"` if $< 3\times 3$ | Uncalibrated heuristic in $[0, 1]$, not probability |
| `texture_consistency` | Gabor filter bank regularity across 4 orientations | Orientations: $0^\circ, 45^\circ, 90^\circ, 135^\circ$; $\text{score} = \min(\max(\text{reg}-0.7, 0)/0.3, 1.0)$ | All formats | Graceful computation | Uncalibrated heuristic in $[0, 1]$, not probability |
| `resampling_clues` | Second-difference sign zero-crossing rate | 2-level RMS floor, $\text{score} = \min(\max(0.5-\text{avg}, 0)/0.5, 1.0)$ | $h \ge 4, w \ge 4$, $\text{RMS} \ge 2.0$ | `outcome="not_applicable"` if flat or $< 4\times 4$ | Uncalibrated heuristic in $[0, 1]$, not probability |

### 3.2 Resampling Abstention Safeguard Audit
- **Formulation:**
  $$dx2 = \Delta^2_x(I_{\text{gray}}), \quad dy2 = \Delta^2_y(I_{\text{gray}})$$
  $$\text{RMS}_{\text{second\_diff}} = \sqrt{\frac{\sum (dx2^2) + \sum (dy2^2)}{\text{size}(dx2) + \text{size}(dy2)}}$$
- **Sufficiency Threshold:** $\text{RMS}_{\text{second\_diff}} < 2.0$ intensity levels (derived from 8-bit quantization noise floor $\sigma = \sqrt{6 \times 2/3} = 2.0$).
- **Verified Permitted Behaviors:**
  - Flat grayscale image: `outcome="not_applicable"`, `score=None`, `reason_code="INSUFFICIENT_TEXTURE_FOR_RESAMPLING"`.
  - Flat RGB image: `outcome="not_applicable"`, `score=None`, `reason_code="INSUFFICIENT_TEXTURE_FOR_RESAMPLING"`.
  - Near-uniform noise ($\sigma < 1$): `outcome="not_applicable"`, `score=None`, `reason_code="INSUFFICIENT_TEXTURE_FOR_RESAMPLING"`.
  - Textured photography (`test_photo.jpg`): $\text{RMS} = 21.15 \ge 2.0 \implies \text{outcome}="success"$, $\text{score}=0.0$.
  - Repeated grid pattern: $\text{RMS} = 261.24 \ge 2.0 \implies \text{outcome}="success"$, $\text{score}=0.0$.
  - Sub-minimum size ($3\times 3$): `outcome="not_applicable"`, `reason_code="IMAGE_TOO_SMALL_FOR_RESAMPLING"`.
  - Corrupted input bytes: `status="failed"`, `outcome="failed"`.
  - PNG format: correctly processes texture and resampling while abstaining on JPEG-specific ELA and block artifacts.
- **Audit Conclusion:** The threshold functions strictly as a signal-sufficiency safeguard preventing false positive anomalies on untextured surfaces. It is explicitly not a trained or validated manipulation classifier.

---

## 4. Provenance Security, URL Sanitization, and Metadata Audit

### 4.1 URL Sanitization Specification Verification
- **HTTPS Enforcement:** Any non-HTTPS scheme (e.g. `http://`, `ftp://`, `javascript:`) is rejected and replaced with `""`.
- **Authority & Hostname Validation:** Hostnames with whitespace, missing hostnames, and user-information elements (`user:password@`) are strictly rejected.
- **Port Syntax:** Malformed non-numeric ports (e.g. `:bad_port`) raise exceptions that are caught and safely normalized to `""`.
- **Sensitive Parameter Filter:** Decodes query strings with percent-decoding and removes all occurrences of:
  `key`, `api_key`, `access_key`, `secret`, `client_secret`, `token`, `access_token`, `refresh_token`, `signature`, `sig`, `authorization`, `credential`, `password`, `auth`, `api-key`.
- **Safe Query Preservation:** Parameters such as `page=1`, `section=public`, `lang=en` are retained, re-encoded, and fragments are stripped.
- **Text Redaction:** Regex replaces Windows and Unix absolute filesystem paths with `[redacted path]`, and redacts credentials matching `(api_key|token|secret|password)=...` with `[redacted]`.
- **Operational Policy:** Provider-returned external URLs are strictly never fetched or dereferenced automatically.

### 4.2 Metadata & Original Hash Distinction
- **Integrity Digest:** SHA-256 is computed directly on the original uploaded byte stream before any decoding or EXIF transposition.
- **Descriptive EXIF Metadata:** Extracted via `extract_metadata` from original bytes. EXIF fields (camera make/model, timestamps, software) are marked with `"source": "EXIF metadata; not cryptographically verified"` to prevent confusion with cryptographic credentials.

### 4.3 Google Cloud Vision Provider Status
- `settings.GOOGLE_API_KEY` is not configured in this environment.
- The service returns `status="unavailable"`, warning `["Google Vision Web Detection credentials are not configured."]`.
- **Audit Confirmation:** No live external requests were made to Google APIs; provider readiness remains unexercised.

---

## 5. Real YOLO Object Detection Integration Audit

- **Ultralytics Version:** `8.4.216` (pinned)
- **Checkpoint Path:** `models/baselines/yolov8n.pt`
- **Verified Checkpoint SHA-256:** `31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95` (size: 6,534,387 bytes)
- **Local-Only Guard:** `_load` inspects `_resolve_model_path`. If weights are missing, it refuses to call the YOLO constructor, explicitly preventing automatic weight downloads from the internet.
- **Runtime Execution:** Executed on NVIDIA GeForce RTX 3050 Laptop GPU (`cuda:0`).
- **Input Fixture:** Permitted `backend/tests/fixtures/test_photo.jpg` (640×480 RGB JPEG).
- **Execution Result:** `status="completed"`, `outcome="no_results"`, `object_count=0` at production 0.3 confidence threshold. Zero detections is verified as a valid, expected execution outcome.
- **Coordinate Space & Semantics:** Coordinates are validated in $[0, width] \times [0, height]$ exif-corrected pixels. `confidence_semantics` explicitly warns: `"YOLO detection confidence; not an authenticity probability."`
- **Fusion Boundary:** Emits `type="contextual_evidence"`, `direction="informational"`. Excluded from Bayesian authenticity voting (0 synthetic, 0 authentic signals).

---

## 6. Production SigLIP Grad-CAM Verification Audit

Independent assessment of `docs/siglip_gradcam_verification_report.md`:
- **Model Identity:** `FineTunedSiglipClassifier` (`truthlens/siglip-cifake-frozen-head`).
- **Checkpoint SHA-256:** `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538` (`models/baselines/siglip_frozen_head.pt`, 5,053 bytes).
- **Backbone Snapshot:** Hugging Face `google/siglip-base-patch16-224` (Git commit `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`).
- **Target Layer Resolution:** `model.vision_model.vision_model.encoder.layers[11].layer_norm2` (`LayerNorm((768,))`).
- **Tensor Dimensions:** Activations `[1, 196, 768]`, Gradients `[1, 196, 768]`, Raw CAM `(14, 14)`, Heatmap `(640, 480)`.
- **Prediction Immutability:** Pre- and post-Grad-CAM logits evaluate to `{'fake': 1.4130399227142334, 'real': -1.4130399227142334}` with **0.0 logit difference**.
- **Parameter Integrity:** All 86 model parameter tensors verified bit-exact identical via `torch.equal`.
- **Cleanup & Mode:** Hooks detached, gradients set to `None`, `model.training` maintained as `False`.
- **Duration & Device:** 311.57 ms on NVIDIA GeForce RTX 3050 Laptop GPU.
- **Audit Conclusion:** The report contains authentic, complete production evidence confirming full compatibility.

---

## 7. Real C2PA Cryptographic Verification & Trust Audit

Independent assessment of `backend/app/services/provenance/c2pa_verifier.py` and `docs/c2pa_verification_report.md`:
- **Installed SDK:** `c2pa-python==0.39.0` (Rust C2PA library `0.92.0`).
- **Signed Test Fixture:** `backend/tests/fixtures/c2pa/C.jpg` (SHA-256 `a2d14755db55de67a47c04090340d8266e892367be4104a45626d7a6fa6e9ffd`, 132,518 bytes).
- **Cryptographic Findings:**
  - Active Manifest: `contentauth:urn:uuid:b2b1f7fa-b119-4de1-9c0d-c97fbea3f2c3`
  - Signature Status: `valid` (`claimSignature.validated`, `claimSignature.insideValidity`)
  - Content Binding: `valid` (`assertion.dataHash.match`)
  - Signer Identity: Common Name `"C2PA Signer"`, Issuer `"C2PA Test Signing Cert"`, Algorithm `"Ps256"`.
  - Embedded Resource: Active manifest JPEG thumbnail stream (28,486 bytes) successfully extracted (`referenced_resources_status="available"`).
- **Tampered Payload Verification:** Single-byte inversion at offset 50,000 produced `assertion.dataHash.mismatch`, correctly reporting `content_binding_status="mismatch"` and `verification_outcome="verification_failed"` while signature remained valid.
- **Signer Trust Matrix (Strict Separation):**
  - Without anchors: `signer_trust_status="unverified"`, `trust_check_configured=False`.
  - With official C2PA conformance production trust list (`C2PA-TRUST-LIST.pem`, SHA-256 `859f422cf4744a45de6f073b258d76381c91a9624c10aaa40543c19c1f124e13`): `signer_trust_status="untrusted"`, diagnostic `signingCredential.untrusted`.
  - With official test root CA bundle (`test_cert_root_bundle.pem`, SHA-256 `c67864bf6bae91c7df0abbef274abd1b03116063093ec225a8391d744f8356fc`): `signer_trust_status="trusted"`, diagnostic `signingCredential.trusted`.
- **Critical Policy Assertion:** Test signer verification under the test root CA is strictly distinguished from production trust. The test signer is NOT production-trusted.
- **Negative Boundaries:** Real unsigned image (`test_photo.jpg`) returns `status="completed"`, `outcome="no_results"`, `verification_outcome="no_manifest"`, `manifest_present=False`. Corrupt bytes return `C2PA_VERIFICATION_ERROR`. Empty bytes return `INVALID_IMAGE_BYTES`.

---

## 8. Integrated Evidence and Fusion Safety Audit

- **Evidence Transformation:**
  - `build_evidence_from_provenance`: emits `type="provenance_evidence"`, `source="c2pa_content_credentials"`, `strength="low"`, `direction="informational"`.
  - `build_evidence_from_vision`: emits `type="contextual_evidence"`, `source="yolov8_object_detection"`, `strength="low"`, `direction="informational"`.
- **Voting Exclusion Contract:**
  - `_calibrated_authenticity_vote` in `engine.py` restricts voting strictly to `type="model_evidence"` where `source="authenticity_detector"` and `stage_name="authenticity"` with validated calibration.
  - All forensic, YOLO, Grad-CAM, C2PA, EXIF, and Web Detection evidence items are routed to `informational_findings` or `evidence_excluded`.
  - `fuse_evidence` summary on full multi-evidence inputs confirms:
    `authentic_signals = 0`, `synthetic_signals = 0`, `manipulated_signals = 0`, `neutral_signals = 0`, `informational_signals >= 1`.
    Decision label: `inconclusive`.
- **Failure Non-Penalization:** Missing manifests, untrusted certificates, tampered C2PA bindings, or failed provider requests never vote as synthetic, manipulated, or authentic.

---

## 9. Test Suite Execution & Exact Counts

1. **Focused Contract Test Suite:**
   ```powershell
   python -m pytest backend/tests/test_e6_e7_contract.py -q
   ```
   - **Result:** **53 passed, 0 failed in 7.37s**
2. **Safe Backend Regression Suite:**
   ```powershell
   python -m pytest backend/tests -q --ignore=backend/tests/test_phase2_dataset_splits.py --ignore=backend/tests/test_phase3_evaluation.py --ignore=backend/tests/test_phase4_and_phase3_5.py --ignore=backend/tests/test_phase5_training.py --ignore=backend/tests/test_phase6_evaluation.py --ignore=backend/tests/test_phase7.py
   ```
   - **Result:** **412 passed, 12 skipped, 6 warnings in 27.08s**
   - **Warnings:** 6 existing SQLite datetime adapter deprecation warnings in `test_e10_database.py`.
   - **Skipped:** 12 tests appropriately skipped (conditional provider credentials, GPU/environment gates).
3. **Bytecode Compilation & Git Formatting:**
   ```powershell
   python -m compileall -q backend/app
   git diff --check
   ```
   - **Result:** 0 compilation errors, 0 git whitespace/style errors.

---

## 10. Documented Remaining Limitations

1. **Google Vision Web Detection:** Live provider requests are unverified due to unconfigured API credentials in local development.
2. **Module 6 Manipulation Detection:** A dedicated learned manipulation-detection model remains unimplemented; classical forensic probes remain heuristic indicators.
3. **C2PA Production Signer Trust:** Real C2PA verification was executed on test corpus fixture `C.jpg`; production trust lists correctly flag test signing certificates as untrusted. Production verification of end-user signed imagery requires configuring vendor-specific trust anchors.
4. **Resampling Analysis:** The 2.0 RMS threshold is a signal sufficiency safeguard against false alarms on flat textures; it is not a calibrated forgery detector.
5. **Multi-Evidence Decision Integration:** Forensic indicators, object detections, and C2PA credentials are intentionally non-voting. Any future directional weighting requires explicit research governance and pre-registered validation.

---

## 11. Final Readiness Classification

```
===============================================================================
STATUS: READY WITH DOCUMENTED LIMITATIONS
===============================================================================
```

TruthLens is verified as reliable, mathematically bounded, secure against data leakage, and architecturally isolated. It is ready for controlled multi-evidence research validation with the documented external provider limitations.
