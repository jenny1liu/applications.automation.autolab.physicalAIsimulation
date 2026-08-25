"""Deterministic mock generator for the shared YOLO C-Cover detection box (no real C-Cover
model exists yet - every model panel reuses this same simulated detection).
"""

from __future__ import annotations

import random


def _seeded_random(seed_text: str) -> random.Random:
    """Build a deterministic RNG from a text seed (stable across app runs)."""
    return random.Random(seed_text)


def _jitter_corner(rng: random.Random, point: tuple[float, float], max_jitter_px: float) -> tuple[float, float]:
    x, y = point
    return (
        round(x + rng.uniform(-max_jitter_px, max_jitter_px)),
        round(y + rng.uniform(-max_jitter_px, max_jitter_px)),
    )


def generate_yolo_cover_result(gt_cover_box: dict, seed_text: str) -> dict:
    """Generate the shared YOLO C-Cover detection result (same detection reused by all models)."""
    if not gt_cover_box:
        raise ValueError("gt_cover_box is required to build a YOLO cover result")

    rng = _seeded_random(f"yolo:{seed_text}")
    max_jitter_px = 4
    detected_cover_box = {
        "top_left": _jitter_corner(rng, gt_cover_box["top_left"], max_jitter_px),
        "top_right": _jitter_corner(rng, gt_cover_box["top_right"], max_jitter_px),
        "bottom_left": _jitter_corner(rng, gt_cover_box["bottom_left"], max_jitter_px),
        "bottom_right": _jitter_corner(rng, gt_cover_box["bottom_right"], max_jitter_px),
    }
    confidence = round(rng.uniform(88, 98), 1)
    iou = round(rng.uniform(0.8, 0.95), 2)
    return {
        "detected_cover_box": detected_cover_box,
        "confidence": confidence,
        "iou": iou,
        "status": "PASS" if iou >= 0.7 else "FAIL",
    }
