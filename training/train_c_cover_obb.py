"""Train a C-Cover-specific YOLO-OBB model."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "datasets" / "c_cover_obb" / "data.yaml"


def main() -> None:
    """Train from the YOLOv8 nano OBB checkpoint with CPU-safe settings."""
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {DATA_YAML}")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Ultralytics is required: pip install ultralytics") from error

    model = YOLO("yolov8n-obb.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=2,
        device="cpu",
        workers=0,
        patience=25,
        project=str(PROJECT_ROOT / "runs"),
        name="c_cover_obb",
        exist_ok=True,
        pretrained=True,
        degrees=8.0,
        translate=0.05,
        scale=0.15,
        fliplr=0.5,
        mosaic=0.0,
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, OSError) as error:
        raise SystemExit(f"C-Cover OBB training failed: {error}") from error