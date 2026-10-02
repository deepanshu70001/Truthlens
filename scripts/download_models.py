"""
TruthLens — Model preparation/download script.

Downloads all required ML models to the local cache BEFORE app startup.
Run this once after installation:

    python -m scripts.download_models

Models are NOT downloaded at application startup to avoid delays.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def download_models():
    """Download all required models to local cache."""
    from backend.app.config import settings

    cache_dir = settings.MODEL_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Model cache directory: {cache_dir}")
    print("=" * 60)

    models_to_download = [
        {
            "name": "Authenticity Detector (SigLIP2)",
            "id": "prithivMLmods/Deepfake-Real-Class-Siglip2",
            "type": "image-classification",
        },
        {
            "name": "Language Detector (XLM-RoBERTa)",
            "id": "papluca/xlm-roberta-base-language-detection",
            "type": "text-classification",
        },
    ]

    for model_info in models_to_download:
        print(f"\n[Downloading] {model_info['name']}")
        print(f"  Model ID: {model_info['id']}")

        try:
            if model_info["type"] == "image-classification":
                from transformers import AutoImageProcessor, AutoModelForImageClassification

                print("  Downloading processor...")
                AutoImageProcessor.from_pretrained(
                    model_info["id"],
                    cache_dir=cache_dir,
                    token=settings.HF_TOKEN or None,
                )

                print("  Downloading model weights...")
                AutoModelForImageClassification.from_pretrained(
                    model_info["id"],
                    cache_dir=cache_dir,
                    token=settings.HF_TOKEN or None,
                )

            elif model_info["type"] == "text-classification":
                from transformers import AutoTokenizer, AutoModelForSequenceClassification

                print("  Downloading tokenizer...")
                AutoTokenizer.from_pretrained(
                    model_info["id"],
                    cache_dir=cache_dir,
                    token=settings.HF_TOKEN or None,
                )

                print("  Downloading model weights...")
                AutoModelForSequenceClassification.from_pretrained(
                    model_info["id"],
                    cache_dir=cache_dir,
                    token=settings.HF_TOKEN or None,
                )

            print(f"  ✅ {model_info['name']} — downloaded successfully")

        except Exception as e:
            print(f"  ❌ {model_info['name']} — FAILED: {e}")

    # Download EasyOCR models
    print(f"\n[Downloading] EasyOCR (English + Hindi)")
    try:
        import easyocr
        reader = easyocr.Reader(["en", "hi"], gpu=False, verbose=False)
        del reader
        print("  ✅ EasyOCR models — downloaded successfully")
    except Exception as e:
        print(f"  ❌ EasyOCR — FAILED: {e}")

    # Download YOLO model
    print(f"\n[Downloading] YOLOv8n (Object Detection)")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        del model
        print("  ✅ YOLOv8n — downloaded successfully")
    except ImportError:
        print("  ⚠️  ultralytics not installed — skipping YOLO download")
    except Exception as e:
        print(f"  ❌ YOLOv8n — FAILED: {e}")

    print("\n" + "=" * 60)
    print("Model download complete.")
    print(f"Cache location: {cache_dir}")


if __name__ == "__main__":
    download_models()
