"""Static title/description text shown on each model panel in the benchmark UI."""

from __future__ import annotations

MODEL_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "opencv": {
        "title": "OpenCV Baseline",
        "description": "Traditional image processing approach.",
    },
    "pytorch": {
        "title": "PyTorch Model",
        "description": "Original FP32 model.",
    },
    "openvino": {
        "title": "OpenVINO Accelerated",
        "description": "Optimized OpenVINO inference result.",
    },
}
