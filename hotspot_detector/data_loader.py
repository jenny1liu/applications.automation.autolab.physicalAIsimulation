"""Loads real pseudo ground-truth assets (thermal images, YOLO-OBB C-Cover labels,
hotspot point annotations) used by the hotspot detection benchmark desktop UI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

_BASE_DIR = Path(__file__).resolve().parent / "pseudo_ground_truth"
IMAGE_DIR = _BASE_DIR / "test" / "images"
TEMPERATURE_DIRS = (
    IMAGE_DIR,
    _BASE_DIR / "temperature",
    Path(__file__).resolve().parent.parent / "thermal_dataset_generator" / "output" / "temperature",
)
YOLO_OBB_LABEL_DIR = _BASE_DIR / "yolo_obb_labels"
HOTSPOT_JSON_PATH = _BASE_DIR / "hotspot_points.json"
TEMPERATURE_CALIBRATION: dict[int, tuple[float, float]] = {
    1: (25.0, 47.0), 2: (24.0, 53.0), 3: (24.0, 41.0), 4: (26.0, 53.0),
    5: (25.0, 42.0), 6: (24.0, 43.0), 7: (22.0, 44.0), 8: (18.0, 51.0),
    9: (25.0, 51.0), 10: (19.0, 38.0), 11: (18.0, 42.0), 12: (18.0, 42.0),
    13: (24.0, 40.0), 14: (24.0, 57.0), 15: (26.0, 41.0), 16: (25.0, 50.0),
    17: (22.0, 51.0), 18: (28.0, 58.0), 19: (24.0, 51.0), 20: (18.0, 39.0),
}


def _decode_colormap_intensity(image_array: np.ndarray) -> np.ndarray:
    """Decode RGB pixels into normalized intensity using the INFERNO color map."""
    if image_array.ndim == 2:
        return image_array.astype(np.float32) / 255.0
    if image_array.ndim != 3 or image_array.shape[2] < 3:
        raise ValueError(f"unsupported thermal image shape: {image_array.shape}")

    rgb_pixels = image_array[..., :3].astype(np.int32)
    lookup_bgr = cv2.applyColorMap(
        np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_INFERNO
    )[:, 0, :]
    lookup_rgb = lookup_bgr[:, ::-1].astype(np.int32)
    distances = np.sum((rgb_pixels[..., None, :] - lookup_rgb) ** 2, axis=-1)
    return np.argmin(distances, axis=2).astype(np.float32) / 255.0


def list_images() -> list[str]:
    """Return the sorted list of thermal image file names available for review.

    Returns an empty list (instead of raising) if the dataset folder is missing,
    so the UI can show an error state without crashing.
    """
    try:
        if not IMAGE_DIR.is_dir():
            raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")
        return sorted(p.name for p in IMAGE_DIR.glob("*.png"))
    except Exception as error:
        print(f"[data_loader] Failed to list images: {error}")
        return []


def load_thermal_image(image_file_name: str) -> Optional[np.ndarray]:
    """Load a thermal image as an RGB numpy array, or None on failure."""
    try:
        image_path = IMAGE_DIR / image_file_name
        with Image.open(image_path) as img:
            return np.array(img.convert("RGB"))
    except Exception as error:
        print(f"[data_loader] Failed to load image '{image_file_name}': {error}")
        return None


def load_raw_temperature(image_file_name: str) -> Optional[np.ndarray]:
    """Load the Celsius matrix associated with a rendered thermal image."""
    image_stem = Path(image_file_name).stem
    candidates = [f"{image_stem}.npy"]
    suffix = image_stem.removeprefix("ir_thermal")
    if suffix.isdigit():
        candidates.append(f"{int(suffix):06d}.npy")

    try:
        for directory in TEMPERATURE_DIRS:
            for candidate in candidates:
                temperature_path = directory / candidate
                if not temperature_path.is_file():
                    continue
                temperature = np.asarray(np.load(temperature_path, allow_pickle=False), dtype=np.float32)
                if temperature.ndim != 2:
                    raise ValueError(f"temperature matrix must be 2D, got shape {temperature.shape}")
                if not np.all(np.isfinite(temperature)):
                    raise ValueError("temperature matrix contains non-finite values")
                return temperature
        return None
    except Exception as error:
        print(f"[data_loader] Failed to load raw temperature for '{image_file_name}': {error}")
        return None


def load_temperature_matrix(image_file_name: str, image_array: np.ndarray) -> Optional[np.ndarray]:
    """Load raw Celsius data or calibrate rendered pixels using the supplied range table."""
    raw_temperature = load_raw_temperature(image_file_name)
    if raw_temperature is not None:
        return raw_temperature

    try:
        image_number_text = Path(image_file_name).stem.removeprefix("ir_thermal")
        image_number = int(image_number_text)
        tmin, tmax = TEMPERATURE_CALIBRATION[image_number]
        normalized_intensity = _decode_colormap_intensity(image_array)
        if not np.all(np.isfinite(normalized_intensity)):
            raise ValueError("thermal image contains non-finite pixel values")
        return (tmin + normalized_intensity * (tmax - tmin)).astype(np.float32)
    except Exception as error:
        print(f"[data_loader] Failed to calibrate temperature for '{image_file_name}': {error}")
        return None


def load_yolo_cover_box(image_file_name: str, image_width: int, image_height: int) -> Optional[dict]:
    """Parse a normalized YOLO-OBB label into its four pixel-space corners."""
    label_path = YOLO_OBB_LABEL_DIR / (Path(image_file_name).stem + ".txt")
    try:
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError("label file is empty")
        parts = [float(value) for value in text.split()]
        if len(parts) != 9:
            raise ValueError(f"expected 9 YOLO-OBB values, got {len(parts)}")
        (
            _,
            top_left_x_ratio,
            top_left_y_ratio,
            bottom_left_x_ratio,
            bottom_left_y_ratio,
            bottom_right_x_ratio,
            bottom_right_y_ratio,
            top_right_x_ratio,
            top_right_y_ratio,
        ) = parts
        return {
            "top_left": (top_left_x_ratio * image_width, top_left_y_ratio * image_height),
            "top_right": (top_right_x_ratio * image_width, top_right_y_ratio * image_height),
            "bottom_left": (bottom_left_x_ratio * image_width, bottom_left_y_ratio * image_height),
            "bottom_right": (bottom_right_x_ratio * image_width, bottom_right_y_ratio * image_height),
        }
    except Exception as error:
        print(f"[data_loader] Failed to load YOLO-OBB cover box for '{image_file_name}': {error}")
        return None


def load_hotspot_ground_truth_map() -> dict[str, list[list[float]]]:
    """Load the full hotspot ground-truth JSON once into a {file_name: points} lookup."""
    try:
        records = json.loads(HOTSPOT_JSON_PATH.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("hotspot ground-truth payload is not a list")
        lookup: dict[str, list[list[float]]] = {}
        for record in records:
            if isinstance(record, dict) and "file" in record and "points" in record:
                lookup[record["file"]] = record["points"]
        return lookup
    except Exception as error:
        print(f"[data_loader] Failed to load hotspot ground truth: {error}")
        return {}
