import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from hotspot_detector.ui import HotspotBenchmarkApp, resolve_runtime_model_path


class TestRuntimeModelPaths(unittest.TestCase):
    def test_resolve_runtime_model_path_uses_packaged_bundle_when_available(self):
        project_root = Path(__file__).resolve().parent.parent
        packaged_root = project_root / "runs" / "c_cover_obb" / "weights"

        with patch.object(sys, "frozen", True, create=True), patch.object(sys, "_MEIPASS", str(project_root), create=True):
            resolved = resolve_runtime_model_path("runs", "c_cover_obb", "weights", "best.pt")

        self.assertTrue(resolved.exists())
        self.assertEqual(resolved.name, "best.pt")
        self.assertEqual(resolved, packaged_root / "best.pt")

    def test_openvino_devices_use_ultralytics_intel_device_syntax(self):
        expectedDevices = {
            "CPU": "cpu",
            "GPU": "intel:gpu",
            "NPU": "intel:npu",
            "AUTO": "intel:auto",
        }

        for uiDevice, backendDevice in expectedDevices.items():
            with self.subTest(uiDevice=uiDevice):
                self.assertEqual(
                    HotspotBenchmarkApp._get_openvino_backend_device(uiDevice),
                    backendDevice,
                )

    def test_unsupported_openvino_device_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported OpenVINO device"):
            HotspotBenchmarkApp._get_openvino_backend_device("CUDA")


if __name__ == "__main__":
    unittest.main()
