from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class YOLODetectionResult:
    center_x: float
    center_y: float
    bbox: tuple[int, int, int, int]
    confidence: float
    inference_time_ms: float
    max_temperature: float


class YOLOv8PyTorchDetector:
    """YOLOv8 detection using PyTorch runtime."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: str = "cpu",
        conf_threshold: float = 0.12,
        iou_threshold: float = 0.50,
        imgsz: int = 640,
    ):
        self.model_name = model_name
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("YOLOv8 requires: pip install ultralytics") from exc

        self.model = YOLO(self.model_name)
        self.model.to(self.device)

    def detect(self, thermal_image: np.ndarray) -> YOLODetectionResult:
        """Run YOLOv8 detection on thermal image."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        t0 = time.perf_counter()

        image_3channel = self._prepare_thermal_input(thermal_image)

        results = self.model(
            image_3channel,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )

        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        max_temp = float(np.max(thermal_image))

        if not results or len(results[0].boxes) == 0:
            center_y, center_x = np.unravel_index(np.argmax(thermal_image), thermal_image.shape)
            center_x = float(center_x)
            center_y = float(center_y)
            box_w = max(12, int(thermal_image.shape[1] * 0.08))
            box_h = max(12, int(thermal_image.shape[0] * 0.08))
            x = int(np.clip(center_x - box_w / 2, 0, max(0, thermal_image.shape[1] - box_w)))
            y = int(np.clip(center_y - box_h / 2, 0, max(0, thermal_image.shape[0] - box_h)))
            bbox = (x, y, box_w, box_h)
            confidence = 0.05
        else:
            boxes = results[0].boxes
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()

            scores = []
            for idx, (x1, y1, x2, y2) in enumerate(xyxy):
                x1i = int(np.clip(np.floor(x1), 0, thermal_image.shape[1] - 1))
                y1i = int(np.clip(np.floor(y1), 0, thermal_image.shape[0] - 1))
                x2i = int(np.clip(np.ceil(x2), x1i + 1, thermal_image.shape[1]))
                y2i = int(np.clip(np.ceil(y2), y1i + 1, thermal_image.shape[0]))

                roi = thermal_image[y1i:y2i, x1i:x2i]
                if roi.size == 0:
                    thermal_score = 0.0
                else:
                    p20 = float(np.percentile(thermal_image, 20.0))
                    p95 = float(np.percentile(thermal_image, 95.0))
                    thermal_score = float(np.clip((np.mean(roi) - p20) / (p95 - p20 + 1e-6), 0.0, 1.0))

                score = float(confs[idx]) * (0.65 + 0.35 * thermal_score)
                scores.append(score)

            best_idx = int(np.argmax(np.asarray(scores, dtype=np.float32)))
            x1, y1, x2, y2 = xyxy[best_idx]
            x1i = int(np.clip(np.floor(x1), 0, thermal_image.shape[1] - 1))
            y1i = int(np.clip(np.floor(y1), 0, thermal_image.shape[0] - 1))
            x2i = int(np.clip(np.ceil(x2), x1i + 1, thermal_image.shape[1]))
            y2i = int(np.clip(np.ceil(y2), y1i + 1, thermal_image.shape[0]))

            center_x, center_y = self._thermal_weighted_center(thermal_image, x1i, y1i, x2i, y2i)
            bbox = (x1i, y1i, max(1, x2i - x1i), max(1, y2i - y1i))
            confidence = float(confs[best_idx])

        return YOLODetectionResult(
            center_x=center_x,
            center_y=center_y,
            bbox=bbox,
            confidence=confidence,
            inference_time_ms=inference_time_ms,
            max_temperature=max_temp,
        )

    def _prepare_thermal_input(self, thermal_image: np.ndarray) -> np.ndarray:
        """Convert raw thermal matrix into YOLO-friendly RGB input."""
        img = thermal_image.astype(np.float32)
        p2 = float(np.percentile(img, 2.0))
        p98 = float(np.percentile(img, 98.0))
        if p98 <= p2:
            p98 = p2 + 1e-3

        norm = np.clip((img - p2) / (p98 - p2), 0.0, 1.0)
        u8 = (norm * 255.0).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        enhanced = clahe.apply(u8)

        pseudo = cv2.applyColorMap(enhanced, cv2.COLORMAP_INFERNO)
        return cv2.cvtColor(pseudo, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _thermal_weighted_center(
        thermal_image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> tuple[float, float]:
        """Use temperature-weighted centroid inside bbox for more stable localization."""
        roi = thermal_image[y1:y2, x1:x2].astype(np.float32)
        if roi.size == 0:
            return float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)

        base = float(np.percentile(roi, 65.0))
        weights = np.clip(roi - base, 0.0, None)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-6:
            return float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)

        yy, xx = np.indices(roi.shape, dtype=np.float32)
        cx = float(np.sum((xx + x1) * weights) / weight_sum)
        cy = float(np.sum((yy + y1) * weights) / weight_sum)
        return cx, cy
