"""Label generation utilities for YOLO detection and segmentation masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class BBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int


class LabelBuilder:
    """Creates detection labels and masks for CPU supervision."""

    @staticmethod
    def bbox_from_mask(mask: np.ndarray) -> BBox:
        """Compute tight bounding box around non-zero mask pixels."""
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return BBox(0, 0, 1, 1)
        return BBox(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    @staticmethod
    def yolo_detection_line(bbox: BBox, width: int, height: int, class_id: int = 0) -> str:
        """Convert pixel bbox to YOLO normalized detection label line."""
        cx = (bbox.x1 + bbox.x2) / 2.0 / width
        cy = (bbox.y1 + bbox.y2) / 2.0 / height
        bw = max(1.0, (bbox.x2 - bbox.x1 + 1.0)) / width
        bh = max(1.0, (bbox.y2 - bbox.y1 + 1.0)) / height
        return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    @staticmethod
    def yolo_keypoint_line(
        bbox: BBox,
        width: int,
        height: int,
        keypoint_x: float,
        keypoint_y: float,
        class_id: int = 0,
    ) -> str:
        """Create one-keypoint YOLO label line: cls cx cy w h x y v."""
        det = LabelBuilder.yolo_detection_line(bbox, width, height, class_id=class_id)
        x = float(np.clip(keypoint_x / max(1.0, width), 0.0, 1.0))
        y = float(np.clip(keypoint_y / max(1.0, height), 0.0, 1.0))
        v = 2
        return f"{det} {x:.6f} {y:.6f} {v}"

    @staticmethod
    def yolo_segmentation_line_from_mask(mask: np.ndarray, width: int, height: int, class_id: int = 0) -> str:
        """Create simple YOLO segmentation line from largest contour polygon."""
        import cv2

        mask_u8 = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return f"{class_id}"

        contour = max(contours, key=cv2.contourArea)
        if contour.shape[0] < 3:
            return f"{class_id}"

        epsilon = max(1.0, 0.004 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if approx.shape[0] < 3:
            approx = contour

        coords: list[str] = []
        for pt in approx[:, 0, :]:
            x = float(np.clip(float(pt[0]) / max(1.0, width), 0.0, 1.0))
            y = float(np.clip(float(pt[1]) / max(1.0, height), 0.0, 1.0))
            coords.append(f"{x:.6f} {y:.6f}")

        return f"{class_id} " + " ".join(coords)
