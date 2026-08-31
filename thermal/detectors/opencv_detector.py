from __future__ import annotations

import time
from dataclasses import dataclass, field
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
    detections: list[dict] = field(default_factory=list)


@dataclass
class CoverDetectionResult:
    polygon: list[tuple[float, float]]
    confidence: float
    inference_time_ms: float


class OpenCVCoverDetector:
    """Detect a C-Cover contour independently from hotspot detection."""

    def __init__(
        self,
        min_area_ratio: float = 0.02,
        top_exclusion_ratio: float = 0.18,
        side_exclusion_ratio: float = 0.03,
    ):
        self.min_area_ratio = min_area_ratio
        self.top_exclusion_ratio = top_exclusion_ratio
        self.side_exclusion_ratio = side_exclusion_ratio

    def detect(self, thermal_image: np.ndarray) -> CoverDetectionResult:
        """Find the strongest large thermal contour and return its rotated rectangle."""
        if thermal_image.ndim != 2 or thermal_image.size == 0:
            raise ValueError("thermal_image must be a non-empty 2D matrix")

        start_time = time.perf_counter()
        image = thermal_image.astype(np.float32)
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        top_exclusion = int(round(binary.shape[0] * self.top_exclusion_ratio))
        side_exclusion = int(round(binary.shape[1] * self.side_exclusion_ratio))
        binary[:top_exclusion, :] = 0
        binary[:, :side_exclusion] = 0
        binary[:, binary.shape[1] - side_exclusion:] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_area = float(image.shape[0] * image.shape[1])
        candidates = [
            contour for contour in contours
            if cv2.contourArea(contour) >= image_area * self.min_area_ratio
        ]
        if not candidates:
            raise RuntimeError("No C-Cover contour found")

        contour = max(candidates, key=cv2.contourArea)
        contour_mask = np.zeros(binary.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        content_rows = np.count_nonzero(contour_mask, axis=0)
        min_column_coverage = max(1, int(round(binary.shape[0] * 0.08)))
        content_columns = np.flatnonzero(content_rows >= min_column_coverage)
        if content_columns.size:
            left = int(content_columns[0])
            right = int(content_columns[-1])
            top = int(np.min(np.where(contour_mask > 0)[0]))
            bottom = int(np.max(np.where(contour_mask > 0)[0]))
            rectangle = ((
                (left + right) / 2.0,
                (top + bottom) / 2.0,
            ), (
                float(max(1, right - left)),
                float(max(1, bottom - top)),
            ), 0.0)
        else:
            rectangle = cv2.minAreaRect(contour)
        polygon = cv2.boxPoints(rectangle).astype(np.float32)
        contour_area = float(cv2.contourArea(contour))
        rectangle_area = max(float(rectangle[1][0] * rectangle[1][1]), 1.0)
        confidence = float(np.clip((contour_area / rectangle_area) * (contour_area / image_area), 0.0, 1.0))
        return CoverDetectionResult(
            polygon=[(float(point[0]), float(point[1])) for point in polygon],
            confidence=confidence,
            inference_time_ms=(time.perf_counter() - start_time) * 1000.0,
        )


class OpenCVHotspotDetector:
    """Hotspot detection using classical OpenCV image processing."""

    def __init__(self, blur_kernel: int = 7, threshold_percentile: float = 0.95, opening_kernel: int = 3):
        self.blur_kernel = blur_kernel
        self.threshold_percentile = threshold_percentile
        self.opening_kernel = opening_kernel

    def detect(
        self,
        thermal_image: np.ndarray,
        temperature_image: np.ndarray | None = None,
        roi_polygon: list[tuple[float, float]] | None = None,
    ) -> DetectionResult:
        """Detect hotspot using OpenCV classical methods."""
        t0 = time.perf_counter()

        if thermal_image.ndim != 2:
            raise ValueError("thermal_image must be 2D")
        if temperature_image is not None and temperature_image.shape != thermal_image.shape:
            raise ValueError("temperature_image must have the same shape as thermal_image")

        image = thermal_image.astype(np.float32)
        temperature_matrix = temperature_image.astype(np.float32) if temperature_image is not None else image
        max_temp = float(np.max(temperature_matrix))

        roi_mask = np.ones(image.shape, dtype=np.uint8) * 255
        if roi_polygon is not None:
            if len(roi_polygon) < 3:
                raise ValueError("roi_polygon must contain at least three points")
            polygon = np.asarray(roi_polygon, dtype=np.int32).reshape((-1, 1, 2))
            roi_mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.fillPoly(roi_mask, [polygon], 255)
            if not np.any(roi_mask):
                raise ValueError("roi_polygon does not intersect the thermal image")

        blurred_temperature = cv2.GaussianBlur(temperature_matrix, (self.blur_kernel, self.blur_kernel), 0)
        roi_temperature = temperature_matrix[roi_mask > 0]
        threshold_val = float(np.percentile(roi_temperature, self.threshold_percentile * 100))
        binary = np.where(blurred_temperature > threshold_val, 255, 0).astype(np.uint8)
        binary[roi_mask == 0] = 0

        opening_size = max(1, int(self.opening_kernel))
        if opening_size % 2 == 0:
            opening_size += 1
        opening_kernel = np.ones((opening_size, opening_size), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, opening_kernel)

        component_count, component_labels, component_stats, component_centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        components = [
            index for index in range(1, component_count)
            if component_stats[index, cv2.CC_STAT_AREA] > 0
        ]

        detections: list[dict] = []
        if not components:
            masked_image = np.where(roi_mask > 0, image, -np.inf)
            y, x = np.unravel_index(np.argmax(masked_image), image.shape)
            center_x, center_y = float(x), float(y)
            bbox = (int(x) - 5, int(y) - 5, 10, 10)
            confidence = 0.1
            detections.append({
                "center_x": center_x, "center_y": center_y, "bbox": bbox,
                "confidence": confidence, "max_temperature": float(temperature_matrix[y, x]),
            })
        else:
            ranked_components = sorted(
                components,
                key=lambda index: int(component_stats[index, cv2.CC_STAT_AREA]),
                reverse=True,
            )[:2]
            background_temps = temperature_matrix[roi_mask == 0]
            if background_temps.size == 0:
                background_temps = temperature_matrix[roi_mask > 0]
            background_mean = float(np.mean(background_temps))
            background_std = float(np.std(background_temps))

            for component_index in ranked_components:
                component_mask = component_labels == component_index
                region_temperatures = temperature_matrix[component_mask]
                hottest_flat_index = int(np.argmax(region_temperatures))
                hottest_coordinates = np.argwhere(component_mask)
                hottest_y, hottest_x = hottest_coordinates[hottest_flat_index]
                center_x = float(component_centroids[component_index][0])
                center_y = float(component_centroids[component_index][1])
                x = int(component_stats[component_index, cv2.CC_STAT_LEFT])
                y = int(component_stats[component_index, cv2.CC_STAT_TOP])
                w = int(component_stats[component_index, cv2.CC_STAT_WIDTH])
                h = int(component_stats[component_index, cv2.CC_STAT_HEIGHT])
                region_mean = float(np.mean(region_temperatures))
                contrast = (region_mean - background_mean) / (background_std + 1e-6)
                confidence = float(np.clip(contrast / 15.0, 0.0, 1.0))
                detections.append({
                    "center_x": center_x,
                    "center_y": center_y,
                    "bbox": (x, y, w, h),
                    "confidence": confidence,
                    "max_temperature": float(temperature_matrix[hottest_y, hottest_x]),
                    "hottest_x": int(hottest_x),
                    "hottest_y": int(hottest_y),
                })

            primary_detection = detections[0]
            center_x = float(primary_detection["center_x"])
            center_y = float(primary_detection["center_y"])
            bbox = primary_detection["bbox"]
            confidence = float(primary_detection["confidence"])

        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        return DetectionResult(
            center_x=center_x,
            center_y=center_y,
            bbox=bbox,
            confidence=confidence,
            inference_time_ms=inference_time_ms,
            max_temperature=max_temp,
            detections=detections,
        )
