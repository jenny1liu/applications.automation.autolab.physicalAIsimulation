"""Hybrid calibration utilities that align synthetic thermal maps with reference IR statistics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class ReferenceStats:
    """Aggregated normalized statistics derived from real IR reference images."""

    row_profile: np.ndarray
    target_values: np.ndarray
    target_cdf: np.ndarray
    image_count: int


class ReferenceCalibrator:
    """Applies lightweight reference-based style calibration to synthetic thermal matrices."""

    def __init__(self, reference_dir: Path):
        self.reference_dir = reference_dir
        self.stats: ReferenceStats | None = self._load_stats(reference_dir)

    @property
    def available(self) -> bool:
        """Return True when valid reference statistics were loaded."""
        return self.stats is not None and self.stats.image_count > 0

    def calibrate(
        self,
        matrix: np.ndarray,
        blend_weight: float,
        detail_preserve_strength: float = 0.8,
        lowfreq_sigma: float = 2.2,
    ) -> np.ndarray:
        """Calibrate toward reference while preserving local keyboard texture details."""
        if not self.available:
            return matrix.astype(np.float32)

        stats = self.stats
        assert stats is not None

        weight = float(np.clip(blend_weight, 0.0, 1.0))
        if weight <= 0.0:
            return matrix.astype(np.float32)

        src = matrix.astype(np.float32)
        src_min = float(np.min(src))
        src_max = float(np.max(src))
        if src_max <= src_min + 1e-6:
            return src

        src_norm = np.clip((src - src_min) / (src_max - src_min), 0.0, 1.0)
        sigma = max(0.1, float(lowfreq_sigma))
        src_low = cv2.GaussianBlur(src_norm, (0, 0), sigmaX=sigma, sigmaY=sigma)
        src_high = src_norm - src_low

        matched_low = self._histogram_match(src_low, stats.target_values, stats.target_cdf)
        matched_low = self._align_row_profile(matched_low, stats.row_profile)

        blended_low = (1.0 - weight) * src_low + weight * matched_low

        detail_keep = float(np.clip(detail_preserve_strength, 0.0, 1.0))
        blended_norm = blended_low + detail_keep * src_high
        blended_norm = np.clip(blended_norm, 0.0, 1.0)
        calibrated = src_min + blended_norm * (src_max - src_min)
        return calibrated.astype(np.float32)

    def distance_to_reference(self, matrix: np.ndarray) -> float:
        """Return distance score between matrix and reference style (lower is better)."""
        if not self.available:
            return float("inf")

        stats = self.stats
        assert stats is not None

        src = matrix.astype(np.float32)
        src_min = float(np.min(src))
        src_max = float(np.max(src))
        if src_max <= src_min + 1e-6:
            return float("inf")

        src_norm = np.clip((src - src_min) / (src_max - src_min), 0.0, 1.0)

        row = src_norm.mean(axis=1)
        row_resized = cv2.resize(row[:, None], (1, 512), interpolation=cv2.INTER_LINEAR).reshape(-1)
        row_l1 = float(np.mean(np.abs(row_resized - stats.row_profile)))

        cdf_values = stats.target_cdf
        flattened = np.sort(src_norm.reshape(-1))
        sample_idx = np.clip((cdf_values * (len(flattened) - 1)).astype(np.int32), 0, len(flattened) - 1)
        src_quantiles = flattened[sample_idx]
        cdf_l1 = float(np.mean(np.abs(src_quantiles - stats.target_values)))

        return 0.55 * cdf_l1 + 0.45 * row_l1

    def _load_stats(self, reference_dir: Path) -> ReferenceStats | None:
        """Load and aggregate normalized histogram and row-profile stats from references."""
        if not reference_dir.exists() or not reference_dir.is_dir():
            return None

        image_paths = sorted(
            [
                p
                for p in reference_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            ]
        )
        if not image_paths:
            return None

        row_profiles: list[np.ndarray] = []
        cdf_values = np.linspace(0.0, 1.0, 512, dtype=np.float32)
        cdf_stack: list[np.ndarray] = []

        for path in image_paths:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            norm = gray / 255.0

            row = norm.mean(axis=1)
            row_resized = cv2.resize(row[:, None], (1, 512), interpolation=cv2.INTER_LINEAR).reshape(-1)
            row_profiles.append(row_resized.astype(np.float32))

            flattened = np.sort(norm.reshape(-1))
            sample_idx = np.clip((cdf_values * (len(flattened) - 1)).astype(np.int32), 0, len(flattened) - 1)
            quantiles = flattened[sample_idx]
            cdf_stack.append(quantiles.astype(np.float32))

        if not row_profiles or not cdf_stack:
            return None

        mean_row_profile = np.mean(np.stack(row_profiles, axis=0), axis=0).astype(np.float32)
        mean_cdf = np.mean(np.stack(cdf_stack, axis=0), axis=0).astype(np.float32)

        return ReferenceStats(
            row_profile=mean_row_profile,
            target_values=mean_cdf,
            target_cdf=cdf_values,
            image_count=len(row_profiles),
        )

    @staticmethod
    def _histogram_match(source: np.ndarray, target_values: np.ndarray, target_cdf: np.ndarray) -> np.ndarray:
        """Map source normalized values to reference CDF via interpolation."""
        src_flat = source.reshape(-1)
        src_sorted = np.sort(src_flat)
        src_cdf = np.linspace(0.0, 1.0, len(src_sorted), dtype=np.float32)

        mapped_sorted = np.interp(src_cdf, target_cdf, target_values)
        ranks = np.searchsorted(src_sorted, src_flat, side="left")
        ranks = np.clip(ranks, 0, len(mapped_sorted) - 1)
        mapped = mapped_sorted[ranks].reshape(source.shape)
        return np.clip(mapped.astype(np.float32), 0.0, 1.0)

    @staticmethod
    def _align_row_profile(matrix_norm: np.ndarray, target_row_profile: np.ndarray) -> np.ndarray:
        """Align row-wise mean trend toward target profile while preserving local structure."""
        h, _ = matrix_norm.shape
        target_row = cv2.resize(target_row_profile[:, None], (1, h), interpolation=cv2.INTER_LINEAR).reshape(-1)

        src_row_mean = matrix_norm.mean(axis=1)
        src_centered = matrix_norm - src_row_mean[:, None]

        target_row = np.clip(target_row, 0.0, 1.0)
        adjusted = src_centered + target_row[:, None]

        out = np.clip(adjusted, 0.0, 1.0).astype(np.float32)
        return out
