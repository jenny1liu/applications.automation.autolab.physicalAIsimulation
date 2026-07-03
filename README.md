# Thermal Hotspot Localization - OpenVINO Performance Evaluation

Complete Python desktop application for evaluating three hotspot localization approaches:

1. **OpenCV-based** (classical image processing baseline)
2. **YOLOv8 PyTorch** (deep learning with PyTorch runtime)
3. **YOLOv8 OpenVINO** (deep learning with optimized OpenVINO runtime)

## Features

### Thermal Image Generation
- Synthetic thermal frame generator with configurable parameters
- Support for single/multiple hotspots
- Circular and irregular hotspot shapes
- Configurable noise levels
- Ground truth annotations (centers, mask, temperatures)

### Detection Methods
- **OpenCV**: Gaussian blur, thresholding, contour analysis
- **PyTorch**: YOLOv8 model inference with PyTorch runtime
- **OpenVINO**: YOLOv8 model converted to OpenVINO IR format

### Metrics Calculation
- Localization error (Euclidean distance to ground truth)
- Inference latency (ms/frame)
- Throughput (FPS)
- Confidence scores
- Temperature analysis

### Robot Integration
- Pixel-to-robot coordinate transformation
- Camera FOV and distance calibration
- Target coordinate generation for robot arm control

### GUI Features
- Real-time visualization of all three methods
- Side-by-side detection comparison
- Metrics dashboard
- Model loading and configuration
- Frame generation controls

## Project Structure

```
PhysicalAI/
├── thermal/
│   ├── ui.py                       # Thermal analysis Tkinter UI
│   ├── create_openvino_model.py    # Utility to create demo OpenVINO model
│   ├── generator.py                # Realistic laptop thermal image generator
│   ├── metrics.py                  # Metrics calculation
│   ├── mapper.py                   # Robot coordinate transformation
│   ├── yolov8n.pt                  # YOLOv8n PyTorch weights
│   ├── yolov8n_openvino_model/     # YOLOv8n OpenVINO export (generate if missing)
│   │   ├── yolov8n.xml
│   │   └── yolov8n.bin
│   ├── models/                     # Additional IR/demo models
│   │   ├── demo_heatmap.xml
│   │   └── demo_heatmap.bin
│   └── detectors/
│       ├── opencv_detector.py      # OpenCV-based hotspot detection
│       ├── yolo_detector.py        # YOLOv8 PyTorch detector
│       └── openvino_detector.py    # YOLOv8 OpenVINO detector
├── robot/
│   ├── action_layer.py             # PyBullet action execution layer
│   └── ui.py                       # Robot pick-place Tkinter UI
├── shared/
│   └── interfaces.py               # Shared contracts for cross-module workflow
├── tests/
├── requirements.txt                # Python dependencies
├── run_thermal_detection.bat       # Thermal UI launch script
├── run_robot.bat                   # Robot UI launch script
└── README.md
```

## Installation

### 1. Create Virtual Environment (Optional)

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate OpenVINO Model Files (Required for OpenVINO detector)

The `thermal/yolov8n_openvino_model/` directory containing `yolov8n.xml` and `yolov8n.bin`
is **not included in the repository** due to binary file size.
Generate it once before running the app:

```bash
# Make sure you are in the project root with .venv activated
python -c "from ultralytics import YOLO; YOLO('thermal/yolov8n.pt').export(format='openvino')"
```

This will create `thermal/yolov8n_openvino_model/`.
The GUI will auto-load it on startup.

> **Note**: `thermal/yolov8n.pt` is included in the repo and will be downloaded automatically
> by Ultralytics if missing.

### Optional: Generate Demo OpenVINO Model via Script

You can also generate the demo heatmap model using:

```bash
python -m thermal.create_openvino_model
```

Default output path:

```text
thermal/models/demo_heatmap.xml
```

Custom output example:

```bash
python -m thermal.create_openvino_model --output models/demo_heatmap.xml
```

### 2. Run GUI Application

```bash
python -m thermal.ui
# or on Windows:
run_thermal_detection.bat
```

### 3. Workflow

1. **Generate Frame** → Creates a realistic laptop thermal scene
2. **Run Detection** → OpenCV / PyTorch / OpenVINO all run automatically
3. **View Metrics** → Single-frame results shown in right panel
4. **Run Benchmark** → Set sample count (default 100) and click **Run Benchmark**
   for mean / median / P95 report across all three methods

## Command-Line Standalone Scripts

### Generate Single Frame

```python
from thermal.generator import ThermalImageGenerator

gen = ThermalImageGenerator(width=320, height=240)
frame = gen.generate(hotspot_count=1)
```

### Run Individual Detectors

```python
from thermal.detectors.opencv_detector import OpenCVHotspotDetector
detector = OpenCVHotspotDetector()
result = detector.detect(thermal_image)
print(f"Center: ({result.center_x}, {result.center_y})")
print(f"Latency: {result.inference_time_ms:.2f} ms")
```

## Testing

Run tests from the project root.

### 1. Interface Unit Tests (unittest)

Validates interface contracts in `shared/interfaces.py`:
- field constraints and `validate()` logic
- enum values
- auto-generated `task_id`

```bash
python -m unittest discover -s tests -p "test_interfaces.py" -v
```

Expected result:
- all tests show `ok`
- summary ends with `OK`

### 2. End-to-End Mock Smoke Test

Runs the fake pipeline once and prints four JSON records
(`HotspotDetection`, `RobotTarget`, `ActionCommand.target`, `ActionResult`).

```bash
python -m tests.test_pipeline_mock
```

Expected result:
- no exception
- output includes `"status": "success"` in `ActionResult`

### 3. Robot Pick-Place UI Demo (Franka Panda)

If the hotspot pixel-to-robot transform is not ready yet, you can still run a
robot-only demo with manual target input.

The demo does this:
- start PyBullet with Franka Panda
- pick a demo payload object on the table
- place it at user-input robot-base coordinates (x, y, z)

```bash
python -m robot.ui
```

In the UI:
- click Start Simulation
- input target x/y/z in meters
- click Pick Then Place

## PyBullet Integration Start Guide

This repo now includes a minimal action execution layer aligned with
`shared/interfaces.py`:

- `ActionCommand` input
- `ActionResult` output
- `RobotTarget.coordinate_frame == robot_base` validation

Implementation file:
- `robot/action_layer.py`

### 1. Set up PyBullet environment

```bash
pip install -r requirements.txt
```

If `pybullet` install fails with proxy/network errors (for example
`No matching distribution found`, or fails with C++ compiler errors such as
`Microsoft Visual C++ 14.0 or greater is required`, install:

- Microsoft C++ Build Tools
- Desktop development with C++
- MSVC v143 (or newer)
- Windows 10/11 SDK

Download page:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

### 2. Configure robot model

`PyBulletActionExecutor.configure_robot_model()` loads
`franka_panda/panda.urdf` from `pybullet_data` by default.

### 3. Develop action execution layer

Current minimal flow in `PyBulletActionExecutor.execute_action()`:
- validate interface contract
- dispatch by action type (`MOVE`, `PICK`, `PLACE`)
- solve IK for end effector targets
- control Panda gripper open/close for pick/place
- step simulation
- return achieved pose and error in mm as `ActionResult`

### 4. Validate robot interface

Run unit tests:

```bash
python -m unittest discover -s tests -p "test_robot_action_layer.py" -v
```

Run a smoke demo (loads robot and executes one PICK command):

```bash
python -m robot.action_layer
```

## Performance Metrics Explained

- **Localization Error**: Euclidean distance (pixels) between detected and true hotspot center
- **Inference Latency**: Time for model to produce output
- **FPS**: Frames processed per second (1000 / latency_ms)
- **Confidence**: Model confidence in the detection (0.0 - 1.0)

## Robot Coordinate System

The application includes a simple camera-to-robot mapping:

- **Camera FOV**: 60° (X) × 45° (Y)
- **Distance to Surface**: 0.5 meters
- **Camera Height**: 0.1 meters

Map pixel (x, y) → Robot (X, Y, Z) for arm positioning.

## System Requirements

- Python 3.9+
- Sufficient disk space for model files (~100 MB for YOLOv8n)
- GPU optional (but recommended for PyTorch)

## Notes

- `thermal/yolov8n_openvino_model/` must be generated locally (see step 1 above); it is excluded from the repo.
- PyTorch detection is slower on CPU (~60–200 ms); OpenVINO runs ~6–12 ms on the same CPU.
- Thermal images are synthetic (laptop-style scene); use real FLIR frames for production validation.
- Benchmark uses a fixed random seed so results are reproducible across runs.
