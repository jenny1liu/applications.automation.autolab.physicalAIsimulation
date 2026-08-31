"""Build a reproducible YOLO-OBB dataset from the C-Cover ground-truth assets."""

from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_IMAGES = PROJECT_ROOT / "hotspot_detector" / "pseudo_ground_truth" / "test" / "images"
SOURCE_LABELS = PROJECT_ROOT / "hotspot_detector" / "pseudo_ground_truth" / "yolo_obb_labels"
DATASET_ROOT = PROJECT_ROOT / "datasets" / "c_cover_obb"
VALIDATION_STEMS = {
    "ir_thermal004",
    "ir_thermal009",
    "ir_thermal014",
    "ir_thermal019",
}


def validate_label(label_path: Path) -> None:
    """Validate one single-class YOLO OBB annotation before copying it."""
    values = label_path.read_text(encoding="utf-8").strip().split()
    if len(values) != 9:
        raise ValueError(f"{label_path.name} must contain class ID plus eight coordinates")
    if values[0] != "0":
        raise ValueError(f"{label_path.name} must use class ID 0 for c_cover")
    coordinates = [float(value) for value in values[1:]]
    if any(value < 0.0 or value > 1.0 for value in coordinates):
        raise ValueError(f"{label_path.name} has coordinates outside the normalized range")


def copy_split(split_name: str, image_paths: list[Path]) -> None:
    """Copy a validated split into the standard YOLO images/labels layout."""
    image_target = DATASET_ROOT / "images" / split_name
    label_target = DATASET_ROOT / "labels" / split_name
    image_target.mkdir(parents=True, exist_ok=True)
    label_target.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        label_path = SOURCE_LABELS / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing OBB label for {image_path.name}")
        validate_label(label_path)
        shutil.copy2(image_path, image_target / image_path.name)
        shutil.copy2(label_path, label_target / label_path.name)


def write_data_yaml() -> None:
    """Write the Ultralytics dataset manifest using portable relative paths."""
    data_yaml = f"""path: {DATASET_ROOT.as_posix()}
train: images/train
val: images/val
names:
  0: c_cover
"""
    (DATASET_ROOT / "data.yaml").write_text(data_yaml, encoding="utf-8")


def main() -> None:
    """Create a deterministic 80/20 dataset split."""
    image_paths = sorted(SOURCE_IMAGES.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No source images found in {SOURCE_IMAGES}")

    validation_images = [path for path in image_paths if path.stem in VALIDATION_STEMS]
    training_images = [path for path in image_paths if path.stem not in VALIDATION_STEMS]
    if len(validation_images) != 4 or len(training_images) != 16:
        raise RuntimeError("Expected exactly 16 training images and 4 validation images")

    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)
    copy_split("train", training_images)
    copy_split("val", validation_images)
    write_data_yaml()
    print(f"Created {DATASET_ROOT}")
    print(f"Training images: {len(training_images)}")
    print(f"Validation images: {len(validation_images)}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Dataset preparation failed: {error}") from error