"""
Simple OpenVINO IR model generator for demonstration.
Creates a minimal heatmap inference model.
"""

import argparse
from pathlib import Path

import numpy as np


def create_openvino_model(output_path: str = "models/demo_heatmap.xml") -> None:
    """Create a minimal OpenVINO IR model for heatmap inference."""
    try:
        import openvino as ov
    except ImportError as exc:
        raise ImportError("OpenVINO required: pip install openvino") from exc

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Create a simple heatmap model
    ops = ov.opset10

    # Input: [1, 1, 240, 320] thermal image
    input_shape = [1, 1, 240, 320]
    inp = ops.parameter(input_shape, np.float32, name="input_image")

    # Simple processing: just pass through with normalization
    # In real scenario, this would be a YOLOv8 or other detection model
    zero = ops.constant(np.array(0.0, dtype=np.float32))
    added = ops.add(inp, zero)
    out = ops.relu(added)

    # Create model
    model = ov.Model([out], [inp], "thermal_heatmap_model")

    # Save as IR format
    ov.save_model(model, str(output_file))

    print(f"✓ OpenVINO model created: {output_file}")
    print(f"  Input shape:  {input_shape}")
    print(f"  Output shape: [1, 1, 240, 320]")
    print(f"\nTo use in GUI:")
    print(f"  1. In the GUI, enter path: {output_file}")
    print(f"  2. Click 'Load' button")
    print(f"  3. Run Detection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OpenVINO demo model")
    parser.add_argument(
        "--output",
        type=str,
        default="models/demo_heatmap.xml",
        help="Output model path",
    )
    args = parser.parse_args()

    create_openvino_model(args.output)
