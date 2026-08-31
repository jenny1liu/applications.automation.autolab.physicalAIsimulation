"""Ultralytics YOLO-OBB detector for trained C-Cover models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CoverOBBDetectionResult:
    polygon: list[tuple[float, float]]
    confidence: float
    inference_time_ms: float
    runtime_device: str


class CCoverOBBDetector:
    """Run a C-Cover-specific Ultralytics OBB model on a thermal RGB image."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        conf_threshold: float = 0.5,
        image_size: int = 640,
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.image_size = image_size
        if not self.model_path.exists():
            raise FileNotFoundError(f"C-Cover OBB model not found: {self.model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Ultralytics is required: pip install ultralytics") from error
        self.model = YOLO(str(self.model_path), task="obb")

    def warmup(self) -> None:
        """Trigger backend compilation before the first user-requested inference."""
        warmup_image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        try:
            self.model(
                warmup_image,
                conf=self.conf_threshold,
                imgsz=self.image_size,
                verbose=False,
                device=self.device,
            )
        except Exception as error:
            raise RuntimeError(f"C-Cover OBB warm-up failed: {error}") from error

    def detect(self, image_array: np.ndarray) -> CoverOBBDetectionResult:
        """Return the highest-confidence rotated C-Cover polygon."""
        if image_array.ndim != 3 or image_array.shape[2] < 3:
            raise ValueError("image_array must be an RGB image")
        start_time = time.perf_counter()
        try:
            results = self.model(
                image_array[..., :3],
                conf=self.conf_threshold,
                imgsz=self.image_size,
                verbose=False,
                device=self.device,
            )
        except Exception as error:
            raise RuntimeError(f"C-Cover OBB inference failed: {error}") from error
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        if not results or results[0].obb is None or len(results[0].obb) == 0:
            raise RuntimeError("No C-Cover OBB detection above confidence threshold")

        obb = results[0].obb
        confidences = obb.conf.cpu().numpy()
        best_index = int(np.argmax(confidences))
        points = obb.xyxyxyxy.cpu().numpy()[best_index]
        return CoverOBBDetectionResult(
            polygon=[(float(point[0]), float(point[1])) for point in points],
            confidence=float(confidences[best_index]),
            inference_time_ms=inference_time_ms,
            runtime_device=str(self.device).upper(),
        )