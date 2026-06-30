from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class LocMetrics:
    """Localization accuracy metrics."""

    localization_error: float
    inference_time_ms: float
    confidence: float
    max_temperature: float
    fps: float = field(init=False)

    def __post_init__(self) -> None:
        self.fps = 1000.0 / self.inference_time_ms if self.inference_time_ms > 0 else 0.0


class MetricsCalculator:
    """Calculate localization and performance metrics."""

    @staticmethod
    def localization_error(
        detected_x: float,
        detected_y: float,
        ground_truth_x: float,
        ground_truth_y: float,
    ) -> float:
        """Compute Euclidean distance between detected and ground truth centers."""
        return float(np.sqrt((detected_x - ground_truth_x) ** 2 + (detected_y - ground_truth_y) ** 2))

    @staticmethod
    def compute_metrics(
        detected_x: float,
        detected_y: float,
        ground_truth_x: float,
        ground_truth_y: float,
        inference_time_ms: float,
        confidence: float,
        max_temperature: float,
    ) -> LocMetrics:
        """Compute comprehensive localization metrics."""
        localization_error = MetricsCalculator.localization_error(
            detected_x, detected_y, ground_truth_x, ground_truth_y
        )
        return LocMetrics(
            localization_error=localization_error,
            inference_time_ms=inference_time_ms,
            confidence=confidence,
            max_temperature=max_temperature,
        )
