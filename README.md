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
├── gui.py                          # Tkinter GUI application
├── thermal_generator.py            # Realistic laptop thermal image generator
├── opencv_detector.py              # OpenCV-based hotspot detection
├── yolo_detector.py                # YOLOv8 PyTorch detector
├── openvino_detector.py            # YOLOv8 OpenVINO detector
├── metrics.py                      # Metrics calculation
├── robot_mapper.py                 # Robot coordinate transformation
├── create_openvino_model.py        # Utility to create demo OpenVINO model
├── requirements.txt                # Python dependencies
├── run.bat / run.ps1               # Launch scripts
├── yolov8n.pt                      # YOLOv8n PyTorch weights
├── yolov8n_openvino_model/         # YOLOv8n OpenVINO export (generate if missing)
│   ├── yolov8n.xml
│   └── yolov8n.bin
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

The `yolov8n_openvino_model/` directory containing `yolov8n.xml` and `yolov8n.bin`
is **not included in the repository** due to binary file size.
Generate it once before running the app:

```bash
# Make sure you are in the project root with .venv activated
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='openvino')"
```

This will create `yolov8n_openvino_model/` in the project root.
The GUI will auto-load it on startup.

> **Note**: `yolov8n.pt` is included in the repo and will be downloaded automatically
> by Ultralytics if missing.

### 2. Run GUI Application

```bash
python gui.py
# or on Windows:
run.bat
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
from thermal_generator import ThermalImageGenerator

gen = ThermalImageGenerator(width=320, height=240)
frame = gen.generate(hotspot_count=1)
```

### Run Individual Detectors

```python
from opencv_detector import OpenCVHotspotDetector
detector = OpenCVHotspotDetector()
result = detector.detect(thermal_image)
print(f"Center: ({result.center_x}, {result.center_y})")
print(f"Latency: {result.inference_time_ms:.2f} ms")
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

- `yolov8n_openvino_model/` must be generated locally (see step 1 above); it is excluded from the repo.
- PyTorch detection is slower on CPU (~60–200 ms); OpenVINO runs ~6–12 ms on the same CPU.
- Thermal images are synthetic (laptop-style scene); use real FLIR frames for production validation.
- Benchmark uses a fixed random seed so results are reproducible across runs.
