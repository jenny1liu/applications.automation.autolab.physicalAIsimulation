"""Utility helpers for IO, random seeding, and color-map handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


COLORMAP_LOOKUP: dict[str, int] = {
    "JET": cv2.COLORMAP_JET,
    "HOT": cv2.COLORMAP_HOT,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "MAGMA": cv2.COLORMAP_MAGMA,
    "TURBO": cv2.COLORMAP_TURBO,
}


def ensure_output_dirs(base_dir: Path) -> dict[str, Path]:
    """Create required output folders and return named paths."""
    paths = {
        "images": base_dir / "images",
        "labels": base_dir / "labels",
        "labels_keypoint": base_dir / "labels_keypoint",
        "labels_seg": base_dir / "labels_seg",
        "metadata": base_dir / "metadata",
        "masks": base_dir / "masks",
        "temperature": base_dir / "temperature",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def index_name(index: int) -> str:
    """Convert sample index to zero-padded six-digit name."""
    return f"{index:06d}"


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with UTF-8 and deterministic indentation."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def render_colormap(
    temp_matrix: np.ndarray,
    colormap_name: str,
    temp_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Convert temperature matrix into an RGB thermal image using given color map."""
    if temp_range is None:
        low = float(np.min(temp_matrix))
        high = float(np.max(temp_matrix))
    else:
        low = float(temp_range[0])
        high = float(temp_range[1])
    if high <= low:
        high = low + 1e-6
    norm = np.clip((temp_matrix - low) / (high - low), 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    cmap_id = COLORMAP_LOOKUP[colormap_name]
    bgr = cv2.applyColorMap(u8, cmap_id)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
