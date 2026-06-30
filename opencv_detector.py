from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class DetectionResult:
    center_x: float
    center_y: float
    bbox: Tuple[int, int, int, int]
    confidence: float
    inference_time_ms: float
    max_temperature: float


class OpenCVHotspotDetector:
    """Hotspot detection using classical OpenCV image processing."""

    def __init__(self, blur_kernel: int = 7, threshold_percentile: float = 0.85):
        self.blur_kernel = blur_kernel
        self.threshold_percentile = threshold_percentile

    def detect(self, thermal_image: np.ndarray) -> DetectionResult:
        """Detect hotspot using OpenCV classical methods."""
        t0 = time.perf_counter()

        if thermal_image.ndim != 2:
            raise ValueError("thermal_image must be 2D")

        image = thermal_image.astype(np.float32)
        max_temp = float(np.max(image))

        image_norm = ((image - image.min()) / (image.max() - image.min() + 1e-6) * 255).astype(np.uint8)

        blurred = cv2.GaussianBlur(image_norm, (self.blur_kernel, self.blur_kernel), 0)

        threshold_val = int(np.percentile(blurred, self.threshold_percentile * 100))
        _, binary = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            y, x = np.unravel_index(np.argmax(image), image.shape)
            center_x, center_y = float(x), float(y)
            bbox = (int(x) - 5, int(y) - 5, 10, 10)
            confidence = 0.1
        else:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                center_x = M["m10"] / M["m00"]
                center_y = M["m01"] / M["m00"]
            else:
                center_x, center_y = image.shape[1] / 2, image.shape[0] / 2

            x, y, w, h = cv2.boundingRect(largest_contour)
            bbox = (x, y, w, h)

            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [largest_contour], 0, 255, -1)
            region_temps = image[mask > 0]
            bg_temps = image[mask == 0]

            region_mean = np.mean(region_temps) if region_temps.size > 0 else image.mean()
            bg_mean = np.mean(bg_temps) if bg_temps.size > 0 else image.mean()
            bg_std = np.std(bg_temps) if bg_temps.size > 0 else 1.0

            contrast = (region_mean - bg_mean) / (bg_std + 1e-6)
            confidence = float(np.clip(contrast / 15.0, 0.0, 1.0))

        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        return DetectionResult(
            center_x=center_x,
            center_y=center_y,
            bbox=bbox,
            confidence=confidence,
            inference_time_ms=inference_time_ms,
            max_temperature=max_temp,
        )
