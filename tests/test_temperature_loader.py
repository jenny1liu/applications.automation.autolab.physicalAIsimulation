import tempfile
import unittest
from pathlib import Path

import numpy as np
import cv2

from hotspot_detector import data_loader
from hotspot_detector.ui import HotspotBenchmarkApp


class TestTemperatureLoader(unittest.TestCase):
    def test_hotspot_metrics_report_distance_normalized_distance_and_result(self):
        predictions = [
            {"coordinate": (11.0, 10.0), "temperature": 35.5},
            {"coordinate": (90.0, 92.0), "temperature": 41.0},
        ]

        metrics = dict(HotspotBenchmarkApp._calculate_hotspot_metrics(
            predictions, [[10.0, 10.0], [90.0, 90.0]], 100.0
        ))

        self.assertEqual(metrics["Distance (px)"], "1.0 px / 2.0 px")
        self.assertEqual(metrics["Normalized Dist (%)"], "1.00% / 2.00%")
        self.assertEqual(metrics["Match Quality"], "99.00% / 98.00%")
        self.assertEqual(metrics["Result"], "#1: Acceptable Hit\n#2: Miss")

        perfect_metrics = dict(HotspotBenchmarkApp._calculate_hotspot_metrics(
            [{"coordinate": (0.5, 0.0), "temperature": 35.0}],
            [[0.0, 0.0]],
            100.0,
        ))
        self.assertEqual(perfect_metrics["Result"], "#1: Perfect Hit")

    def test_prediction_count_does_not_exceed_gt_count(self):
        predictions = [
            {"center_x": 10.0, "center_y": 10.0},
            {"center_x": 90.0, "center_y": 90.0},
        ]

        actual = HotspotBenchmarkApp._limit_predictions_to_gt_count(predictions, [[10.0, 10.0]])

        self.assertEqual(len(actual), 1)

    def test_comparison_slots_match_gt_count_without_fake_predictions(self):
        slots = HotspotBenchmarkApp._make_comparison_slots(
            [
                {"center_x": 10.0, "center_y": 10.0, "temperature": 40.0, "confidence": 0.9},
                {"center_x": 90.0, "center_y": 90.0, "temperature": 42.0, "confidence": 0.8},
            ],
            [[10.0, 10.0], [90.0, 90.0]],
        )

        self.assertEqual(len(slots), 2)
        self.assertTrue(slots[0]["has_prediction"])
        self.assertTrue(slots[1]["has_prediction"])
        self.assertEqual(slots[1]["temperature"], 42.0)

    def test_each_prediction_uses_its_closest_gt(self):
        slots = HotspotBenchmarkApp._make_comparison_slots(
            [{"center_x": 88.0, "center_y": 88.0, "temperature": 42.0, "confidence": 0.8}],
            [[10.0, 10.0], [90.0, 90.0]],
        )

        self.assertEqual(len(slots), 1)
        self.assertEqual(slots[0]["ground_truth_id"], 2)
        self.assertEqual(slots[0]["error_px"], 3)

    def test_each_prediction_gets_its_own_nearest_gt_error(self):
        slots = HotspotBenchmarkApp._make_comparison_slots(
            [
                {"center_x": 12.0, "center_y": 10.0, "temperature": 35.0, "confidence": 0.9},
                {"center_x": 88.0, "center_y": 90.0, "temperature": 42.0, "confidence": 0.8},
            ],
            [[10.0, 10.0], [90.0, 90.0]],
        )

        self.assertEqual([slot["ground_truth_id"] for slot in slots], [1, 2])
        self.assertEqual([slot["error_px"] for slot in slots], [2, 2])

    def test_temperature_at_coordinate_reads_predicted_pixel(self):
        temperature = np.array([[25.0, 31.0], [37.0, 47.0]], dtype=np.float32)

        actual = HotspotBenchmarkApp._temperature_at_coordinate(temperature, 1.0, 0.0)

        self.assertEqual(actual, 31.0)

    def test_temperature_at_coordinate_clamps_out_of_bounds_prediction(self):
        temperature = np.array([[25.0, 31.0], [37.0, 47.0]], dtype=np.float32)

        actual = HotspotBenchmarkApp._temperature_at_coordinate(temperature, 100.0, -10.0)

        self.assertEqual(actual, 31.0)

    def test_calibrates_rendered_image_to_configured_temperature_range(self):
        intensity = np.array([[0, 128], [192, 255]], dtype=np.uint8)
        image = cv2.cvtColor(cv2.applyColorMap(intensity, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        actual = data_loader.load_temperature_matrix("ir_thermal001.png", image)

        self.assertIsNotNone(actual)
        self.assertAlmostEqual(float(actual[0, 0]), 25.0, places=4)
        self.assertAlmostEqual(float(actual[1, 1]), 47.0, places=4)

    def test_colormap_decode_keeps_blue_and_red_order(self):
        intensity = np.array([[32, 224]], dtype=np.uint8)
        image = cv2.cvtColor(cv2.applyColorMap(intensity, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        actual = data_loader.load_temperature_matrix("ir_thermal001.png", image)

        self.assertLess(float(actual[0, 0]), float(actual[0, 1]))

    def test_load_raw_temperature_matches_ir_thermal_sequence_name(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temperature_directory = Path(temporary_directory)
            expected = np.array([[31.5, 42.25], [28.0, 47.75]], dtype=np.float32)
            np.save(temperature_directory / "000001.npy", expected)

            original_directories = data_loader.TEMPERATURE_DIRS
            data_loader.TEMPERATURE_DIRS = (temperature_directory,)
            try:
                actual = data_loader.load_raw_temperature("ir_thermal001.png")
            finally:
                data_loader.TEMPERATURE_DIRS = original_directories

            self.assertIsNotNone(actual)
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()