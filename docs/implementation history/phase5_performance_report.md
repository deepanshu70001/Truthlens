# Phase 5 performance report

Measured 2026-11-17 on Windows 10.0.26200, Python 3.11.9, 8 logical CPU cores, PyTorch 2.14.1+cpu, CUDA unavailable. Each CPU timing below used a generated-in-memory 640x480 checkerboard JPEG (8,059 bytes), five repeated runs. Values are mean and median wall-clock milliseconds; this microbenchmark excludes API/database and disk/network overhead except preprocessing writes to a temporary directory.

| Stage | Mean ms | Median ms | Range ms | Runs |
|---|---:|---:|---:|---:|
| Preprocessing | 31.50 | 28.86 | 27.06–44.80 | 5 |
| Quality | 11.91 | 11.07 | 10.91–14.66 | 5 |
| Classical forensics (artifact writing disabled) | 70.99 | 70.95 | 69.27–74.10 | 5 |
| Virality indicators | 0.03 | 0.03 | 0.02–0.07 | 5 |
| Regional unavailable-result construction | 0.02 | 0.01 | 0.01–0.03 | 5 |
| Social category cues | 0.03 | 0.02 | 0.01–0.11 | 5 |
| Report assembly | 0.10 | 0.09 | 0.09–0.14 | 5 |

Authenticity model timing was separately measured from the local cache with network disabled, using the same synthetic 640x480 input. Model load: **40,474.82 ms**. Warm inference (3 runs): **641.05, 262.56, 250.45 ms**, median **262.56 ms**. Device CPU. The model emitted configuration warnings for `bos_token_id`/`eos_token_id`; inference completed. Its classification of the synthetic image is not ground truth and is not an accuracy measurement. A separate authenticity smoke inference confirmed the checkpoint can load and execute from cache.

Not measurable in this environment: OCR (EasyOCR and Tesseract not installed), object detection (Ultralytics package not installed), XAI as dependent on a loaded compatible model, and external fact-check/provenance requests (credentials absent). End-to-end upload-to-report latency was not measured; the API test client cannot start because asyncio loop creation hangs in Windows socketpair in this sandbox. No percentile latency is reported because five local microbenchmark samples are insufficient for a representative service latency estimate. No concurrency/load benchmark was run.

Reproduce the local subset with `backend/app/services/` on a generated 640x480 JPEG and five runs; model load/inference figures depend on hardware, cache state and exact checkpoint. They should not be used as deployment SLAs.
