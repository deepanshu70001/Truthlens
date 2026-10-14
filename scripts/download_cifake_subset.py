import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "external" / "cifake"
TRAIN_DIR = OUTPUT_DIR / "train"
TEST_DIR = OUTPUT_DIR / "test"

# Small starter subset: 500 images per class for each split.
PER_CLASS_TRAIN = 500
PER_CLASS_TEST = 100
SEED = 20261014

DATASET_ID = "dragonintelligence/CIFAKE-image-dataset"


def main():
    parser = argparse.ArgumentParser(
        description="Download a small, reproducible CIFAKE subset for TruthLens."
    )
    parser.parse_args()

    for folder in (TRAIN_DIR, TEST_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    for folder in (TRAIN_DIR, TEST_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID)

    print("Available splits:", list(dataset.keys()))

    # Stop rather than silently guessing if the repository has a different schema.
    if "train" not in dataset or "test" not in dataset:
        raise RuntimeError(
            "Expected train and test splits. "
            f"Found: {list(dataset.keys())}. Inspect the dataset card first."
        )

    train = dataset["train"]
    test = dataset["test"]

    print("Train columns:", train.column_names)
    print("Test columns:", test.column_names)

    def find_column(columns, candidates):
        for candidate in candidates:
            if candidate in columns:
                return candidate
        return None

    image_col = find_column(
        train.column_names, ["image", "img", "pixel_values"]
    )
    label_col = find_column(
        train.column_names, ["label", "labels", "class"]
    )

    if image_col is None or label_col is None:
        raise RuntimeError(
            "Could not identify image and label columns. "
            "No images were exported. Inspect the printed columns."
        )

    if image_col not in test.column_names or label_col not in test.column_names:
        raise RuntimeError("Train and test schemas do not match.")

    features = train.features[label_col]
    label_names = getattr(features, "names", None)

    if not label_names:
        raise RuntimeError(
            "Label names were not provided by the dataset schema. "
            "Inspect the dataset card before mapping numeric labels."
        )

    print("Label names:", label_names)

    # CIFAKE normally uses REAL and FAKE labels. Verify before saving.
    normalized_names = {name.upper(): i for i, name in enumerate(label_names)}
    if not {"REAL", "FAKE"}.issubset(normalized_names):
        raise RuntimeError(
            f"Unexpected labels: {label_names}. "
            "Update the mapping only after verifying the dataset card."
        )

    label_map = {
        normalized_names["REAL"]: "real",
        normalized_names["FAKE"]: "ai_generated",
    }

    def export_split(split_data, split_name, per_class):
        counts = {name: 0 for name in label_map.values()}
        rows = []

        # Shuffle deterministically, but retain the original split.
        shuffled = split_data.shuffle(seed=SEED)

        for item in shuffled:
            raw_label = int(item[label_col])
            if raw_label not in label_map:
                continue

            class_name = label_map[raw_label]
            if counts[class_name] >= per_class:
                continue

            image = item[image_col]
            if not isinstance(image, Image.Image):
                raise RuntimeError(
                    f"Expected PIL image, got {type(image).__name__}"
                )

            image = image.convert("RGB")
            filename = f"{class_name}_{counts[class_name]:04d}.png"
            output_path = OUTPUT_DIR / split_name / class_name / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, format="PNG")

            rows.append({
                "file": str(output_path.relative_to(ROOT)).replace("\\", "/"),
                "split": split_name,
                "label": class_name,
                "source_dataset": DATASET_ID,
                "is_synthetic_record": False,
                "image_origin": (
                    "CIFAKE dataset label: AI-generated"
                    if class_name == "ai_generated"
                    else "CIFAKE dataset label: real"
                ),
                "original_split": split_name,
            })

            counts[class_name] += 1

            if all(counts[c] >= per_class for c in counts):
                break

        if any(counts[c] != per_class for c in counts):
            raise RuntimeError(
                f"Insufficient examples for {split_name}: {counts}; "
                f"expected {per_class} per class."
            )

        return counts, rows

    train_counts, train_rows = export_split(
        train, "train", PER_CLASS_TRAIN
    )
    test_counts, test_rows = export_split(
        test, "test", PER_CLASS_TEST
    )

    manifest = {
        "dataset_name": "CIFAKE",
        "dataset_id": DATASET_ID,
        "purpose": "Starter benchmark for real-vs-AI-generated image classification",
        "dataset_status": "EXTERNAL_DATASET_SUBSET",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "image_format": "PNG",
        "original_image_resolution": "See dataset card; commonly 32x32",
        "class_mapping": {
            "real": "Images labelled REAL in the source dataset",
            "ai_generated": "Images labelled FAKE in the source dataset",
        },
        "counts": {
            "train": train_counts,
            "test": test_counts,
        },
        "total_images": len(train_rows) + len(test_rows),
        "limitations": [
            "This subset is for pipeline and baseline testing.",
            "Source labels are not a universal measure of image authenticity.",
            "Performance on CIFAKE does not establish deepfake or tampering detection.",
            "Do not mix this dataset with synthetic scenario CSVs.",
            "Review the source dataset card and license before redistribution or commercial use.",
        ],
        "records": train_rows + test_rows,
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nDownload complete.")
    print("Train counts:", train_counts)
    print("Test counts:", test_counts)
    print("Total exported:", manifest["total_images"])
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
