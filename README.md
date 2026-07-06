# PhysicalAI Thermal Hotspot Project

This repository contains two connected workflows:

1. Thermal hotspot detection and benchmarking UI
2. Synthetic thermal dataset generation for training and evaluation

The main runtime compares three localization approaches:

1. OpenCV baseline
2. YOLOv8 PyTorch
3. YOLOv8 OpenVINO

## Repository Map

- thermal/: detector runtime, UI, metrics, mapping, model utilities
- thermal_dataset_generator/: synthetic data pipeline and sim-to-real tools
- robot/: PyBullet action layer and robot UI
- shared/: cross-module interfaces
- tests/: unit and smoke tests
- run_thermal_detection.bat: Windows launcher for thermal UI
- run_robot.bat: Windows launcher for robot UI

For generator internals and options, see thermal_dataset_generator/README.md.

## Quick Start

### 1. Environment setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate OpenVINO model once

thermal/yolov8n.pt is in-repo. Export OpenVINO IR locally:

```bash
python -c "from ultralytics import YOLO; YOLO('thermal/yolov8n.pt').export(format='openvino')"
```

Expected output folder:

```text
thermal/yolov8n_openvino_model/
```

### 3. Run thermal UI

```bash
python -m thermal.ui
```

Windows shortcut:

```bash
run_thermal_detection.bat
```

## Core Workflows

### A. Detector benchmark UI

1. Generate a frame in UI
2. Run OpenCV, PyTorch, OpenVINO detectors
3. Compare localization and latency
4. Run multi-sample benchmark from UI panel

### B. Synthetic dataset generation

```bash
python -m thermal_dataset_generator.main --size 1000 --image-size 640 --workers 8
```

Outputs are written under:

```text
thermal_dataset_generator/output/
```

Main exports:

- images
- detection labels
- keypoint labels
- segmentation labels
- masks
- temperature matrices
- metadata json

## Useful Commands

### Physics-gated generation

```bash
python -m thermal_dataset_generator.main --size 1000 --physics-threshold 0.80 --physics-max-attempts 12
```

### Hybrid calibration against reference IR

```bash
python -m thermal_dataset_generator.main --size 1000 --enable-hybrid --hybrid-reference-dir "thermal_dataset_generator/reference_ir" --hybrid-weight 0.35
```

### Sim-to-real evaluation

```bash
python -m thermal_dataset_generator.evaluate_sim_to_real --manifest thermal_dataset_generator/reference_ir/real_manifest.json --detector opencv
```

## Testing

Run from repository root.

### Shared interface tests

```bash
python -m unittest discover -s tests -p "test_interfaces.py" -v
```

### Mock end-to-end smoke

```bash
python -m tests.test_pipeline_mock
```

### Robot action layer tests

```bash
python -m unittest discover -s tests -p "test_robot_action_layer.py" -v
```

## Robot Notes

- Robot runtime uses PyBullet and Franka Panda model from pybullet_data.
- Action layer implementation is in robot/action_layer.py.
- Robot UI demo is in robot/ui.py.

## Requirements

- Python 3.9+
- Optional GPU for faster PyTorch inference
- OpenVINO runtime for optimized CPU inference

## Notes

- OpenVINO IR files are generated locally and are not committed.
- Thermal frames are synthetic; use real IR validation before deployment.
- For generator design details, target-point logic, no-cheat policy, and split manifests, see thermal_dataset_generator/README.md.
