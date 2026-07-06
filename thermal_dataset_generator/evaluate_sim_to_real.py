"""Evaluate hotspot models on real thermal set with annotated ground truth points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np

from thermal.detectors.opencv_detector import OpenCVHotspotDetector
from thermal.detectors.openvino_detector import OpenVINOYOLODetector
from thermal.detectors.yolo_detector import YOLOv8PyTorchDetector
from thermal.metrics import MetricsCalculator


def _load_temperature_image(path: Path) -> np.ndarray:
    """Load real thermal frame as float matrix (prefer npy, fallback grayscale PNG/JPG)."""
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D npy at {path}")
        return arr

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return image.astype(np.float32)


def _percentiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float32)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95)), float(np.percentile(arr, 99))


def _build_detector(kind: str, model_path: str, device: str):
    if kind == "opencv":
        return OpenCVHotspotDetector()
    if kind == "pytorch":
        return YOLOv8PyTorchDetector(model_name=model_path, device=device)
    if kind == "openvino":
        return OpenVINOYOLODetector(model_path=model_path, device=device)
    raise ValueError(f"Unsupported detector kind: {kind}")


def _print_group_stats(title: str, values: list[float]) -> None:
    if not values:
        print(f"{title}: n=0")
        return
    p50, p95, p99 = _percentiles(values)
    print(
        f"{title}: n={len(values)}, mean={mean(values):.2f}px, "
        f"median={median(values):.2f}px, p95={p95:.2f}px, p99={p99:.2f}px"
    )


def run_eval(
    manifest_path: Path,
    detector_kind: str,
    model_path: str,
    device: str,
) -> None:
    """Run sim-to-real evaluation using manifest annotations."""
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, list):
        raise ValueError("Manifest must be a JSON list of samples")

    detector = _build_detector(detector_kind, model_path, device)

    all_errors: list[float] = []
    reversal_errors: list[float] = []
    normal_errors: list[float] = []

    for item in manifest:
        if not isinstance(item, dict):
            continue
        image_path = Path(str(item.get("image", "")))
        gt = item.get("target_point")
        if not image_path.exists() or not isinstance(gt, (list, tuple)) or len(gt) != 2:
            continue

        image = _load_temperature_image(image_path)
        result = detector.detect(image)

        gt_x, gt_y = float(gt[0]), float(gt[1])
        err = MetricsCalculator.localization_error(result.center_x, result.center_y, gt_x, gt_y)
        all_errors.append(float(err))

        case_type = str(item.get("case_type", "normal")).strip().lower()
        if case_type in {"reversal", "vent_reversal", "trap_reversal"}:
            reversal_errors.append(float(err))
        else:
            normal_errors.append(float(err))

    print("=== Sim-to-Real Evaluation ===")
    print(f"detector={detector_kind}, device={device}")
    _print_group_stats("all", all_errors)
    _print_group_stats("reversal", reversal_errors)
    _print_group_stats("normal", normal_errors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate synthetic-trained detector on real thermal set")
    parser.add_argument("--manifest", type=Path, required=True, help="JSON list with image path and target_point")
    parser.add_argument("--detector", choices=["opencv", "pytorch", "openvino"], default="opencv")
    parser.add_argument("--model", type=str, default="thermal/yolov8n.pt", help="Model path for pytorch/openvino")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_eval(
        manifest_path=args.manifest,
        detector_kind=args.detector,
        model_path=args.model,
        device=args.device,
    )


if __name__ == "__main__":
    main()
