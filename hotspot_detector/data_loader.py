"""Loads real pseudo ground-truth assets (thermal images, YOLO-OBB C-Cover labels,
hotspot point annotations) used by the hotspot detection benchmark desktop UI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

_BASE_DIR = Path(__file__).resolve().parent / "pseudo_ground_truth"
IMAGE_DIR = _BASE_DIR / "test" / "images"
YOLO_OBB_LABEL_DIR = _BASE_DIR / "yolo_obb_labels"
HOTSPOT_JSON_PATH = _BASE_DIR / "hotspot_points.json"


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
