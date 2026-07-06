"""Ground-truth target point selection for genuine source hotspot localization."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class TargetPointResult:
    """Container for section-8 target point outputs and QA fields."""

    target_point: tuple[int, int]
    target_temperature: float
    target_fallback_tier: int
    excluded_hottest_point: tuple[int, int]
    excluded_hottest_region: str | None
    trap_zone_was_global_hottest: bool
    runner_up_point: tuple[int, int]
    runner_up_temperature: float
    target_source_attribution_fraction: float
    source_attribution_fraction_map: np.ndarray


def _argmax_with_mask(temp: np.ndarray, valid_mask: np.ndarray) -> tuple[int, int]:
    masked = np.where(valid_mask, temp, -np.inf)
    idx = int(np.argmax(masked))
    y, x = np.unravel_index(idx, temp.shape)
    return int(y), int(x)


def _runner_up(temp: np.ndarray, valid_mask: np.ndarray, selected_y: int, selected_x: int) -> tuple[int, int]:
    masked = np.where(valid_mask, temp, -np.inf)
    masked[selected_y, selected_x] = -np.inf
    if not np.isfinite(float(np.max(masked))):
        return int(selected_y), int(selected_x)
    idx = int(np.argmax(masked))
    y, x = np.unravel_index(idx, temp.shape)
    return int(y), int(x)


def _dilate_bool(mask: np.ndarray | None, radius_px: int = 3) -> np.ndarray:
    if mask is None:
        return np.zeros((1, 1), dtype=bool)
    m = np.asarray(mask)
    if m.size == 0:
        return np.zeros_like(m, dtype=bool)
    m_u8 = (m > 0).astype(np.uint8)
    if radius_px <= 0:
        return m_u8 > 0
    k = max(1, int(radius_px) * 2 + 1)
    kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.dilate(m_u8, kernel, iterations=1) > 0


def select_target_point(
    matrix: np.ndarray,
    ambient_temperature: float,
    keyboard_mask: np.ndarray,
    vent_mask: np.ndarray | None,
    hinge_mask: np.ndarray | None,
    cpu_mask: np.ndarray | None,
    source_numerator_map: np.ndarray,
    trap_rise_map: np.ndarray | None,
    source_attribution_threshold: float,
    blur_sigma: float,
    exclusion_dilation_px: int = 3,
) -> TargetPointResult:
    """Select target point using section-8 rule with fallback tiers and QA fields."""
    temp = np.asarray(matrix, dtype=np.float32)
    h, w = temp.shape

    kb = np.asarray(keyboard_mask) > 0
    if kb.shape != (h, w):
        kb = cv2.resize(kb.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

    vent = None if vent_mask is None else np.asarray(vent_mask)
    if vent is not None and vent.shape != (h, w):
        vent = cv2.resize((vent > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    hinge = None if hinge_mask is None else np.asarray(hinge_mask)
    if hinge is not None and hinge.shape != (h, w):
        hinge = cv2.resize((hinge > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    cpu = None if cpu_mask is None else np.asarray(cpu_mask)
    if cpu is not None and cpu.shape != (h, w):
        cpu = cv2.resize((cpu > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    source_num = np.asarray(source_numerator_map, dtype=np.float32)
    if source_num.shape != (h, w):
        source_num = cv2.resize(source_num, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    thermal_rise = np.maximum(temp - float(ambient_temperature), 1e-4)
    source_attr = np.clip(source_num / thermal_rise, 0.0, 1.0).astype(np.float32)

    vent_d = _dilate_bool(vent, exclusion_dilation_px) if vent is not None else np.zeros((h, w), dtype=bool)
    hinge_d = _dilate_bool(hinge, exclusion_dilation_px) if hinge is not None else np.zeros((h, w), dtype=bool)

    base_valid = kb & (~vent_d) & (~hinge_d)
    tier0 = base_valid & (source_attr > float(source_attribution_threshold))
    tier1 = base_valid
    tier2 = (cpu > 0) if cpu is not None else np.zeros((h, w), dtype=bool)

    if np.any(tier0):
        valid = tier0
        fallback_tier = 0
    elif np.any(tier1):
        valid = tier1
        fallback_tier = 1
    elif np.any(tier2):
        valid = tier2
        fallback_tier = 2
    else:
        valid = np.ones((h, w), dtype=bool)
        fallback_tier = 2

    sigma = max(0.05, float(blur_sigma))
    selection_map = cv2.GaussianBlur(temp, (0, 0), sigmaX=sigma, sigmaY=sigma)

    ty, tx = _argmax_with_mask(selection_map, valid)
    ry, rx = _runner_up(selection_map, valid, ty, tx)

    gy, gx = np.unravel_index(int(np.argmax(temp)), temp.shape)

    trap_mask = np.zeros((h, w), dtype=bool)
    if trap_rise_map is not None:
        tr = np.asarray(trap_rise_map, dtype=np.float32)
        if tr.shape != (h, w):
            tr = cv2.resize(tr, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        trap_mask = tr > max(0.30, float(np.max(tr)) * 0.45)

    excluded_region: str | None = None
    if bool(vent_d[gy, gx]):
        excluded_region = "vent"
    elif bool(hinge_d[gy, gx]):
        excluded_region = "hinge"
    elif bool(trap_mask[gy, gx]):
        excluded_region = "trap"

    return TargetPointResult(
        target_point=(int(tx), int(ty)),
        target_temperature=float(temp[ty, tx]),
        target_fallback_tier=int(fallback_tier),
        excluded_hottest_point=(int(gx), int(gy)),
        excluded_hottest_region=excluded_region,
        trap_zone_was_global_hottest=bool(trap_mask[gy, gx]),
        runner_up_point=(int(rx), int(ry)),
        runner_up_temperature=float(temp[ry, rx]),
        target_source_attribution_fraction=float(source_attr[ty, tx]),
        source_attribution_fraction_map=source_attr,
    )


def cpu_min_distance_to_hinge_px(cpu_mask: np.ndarray, hinge_mask: np.ndarray | None) -> float:
    """Compute minimum CPU->hinge distance in pixels for metadata QA."""
    cpu = np.asarray(cpu_mask) > 0
    if not np.any(cpu):
        return -1.0
    if hinge_mask is None:
        return -1.0
    hinge = np.asarray(hinge_mask) > 0
    if hinge.shape != cpu.shape:
        hinge = cv2.resize(hinge.astype(np.uint8), (cpu.shape[1], cpu.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    if not np.any(hinge):
        return -1.0

    inv_hinge = (~hinge).astype(np.uint8)
    dist = cv2.distanceTransform(inv_hinge, cv2.DIST_L2, 3)
    return float(np.min(dist[cpu])) if np.any(cpu) else -1.0
