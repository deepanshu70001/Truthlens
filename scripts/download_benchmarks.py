"""Download benchmark datasets from Hugging Face Hub to local datasets directory.

Datasets:
- yuzj1997yzj/WildAIGC-Bench (in-the-wild AI vs real news images benchmark)
- wasitaigeneratedcom/frontier-image-video-bench-2026-10 (frontier multi-generator benchmark)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    {
        "name": "WildAIGC-Bench",
        "repo_id": "yuzj1997yzj/WildAIGC-Bench",
        "dir_name": "WildAIGC-Bench",
    },
    {
        "name": "WasItAIGenerated Frontier Image & Video Benchmark (2026-10)",
        "repo_id": "wasitaigeneratedcom/frontier-image-video-bench-2026-10",
        "dir_name": "frontier-image-video-bench-2026-10",
    },
]


def download_all():
    print(f"Target datasets directory: {DATASETS_DIR}")
    print("=" * 70)

    for target in TARGETS:
        repo_id = target["repo_id"]
        name = target["name"]
        dest = DATASETS_DIR / target["dir_name"]
        dest.mkdir(parents=True, exist_ok=True)

        print(f"\n[Starting Download] {name}")
        print(f"  Repository ID: {repo_id}")
        print(f"  Destination:   {dest}")
        start_time = time.time()

        try:
            snapshot_path = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(dest),
            )
            elapsed = time.time() - start_time
            file_count = sum(1 for _ in dest.rglob("*") if _.is_file())
            total_bytes = sum(_.stat().st_size for _ in dest.rglob("*") if _.is_file())
            print(f"[Completed] {name}")
            print(f"  Files: {file_count}, Total Size: {total_bytes / (1024**3):.2f} GB in {elapsed:.1f}s")
        except Exception as exc:
            print(f"[Error] Failed to download {repo_id}: {exc}", file=sys.stderr)
            raise

    print("\n" + "=" * 70)
    print("All requested benchmark datasets downloaded successfully.")


if __name__ == "__main__":
    download_all()
