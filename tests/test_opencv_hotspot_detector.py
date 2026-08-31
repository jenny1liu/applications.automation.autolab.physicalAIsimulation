import unittest

import numpy as np

from thermal.detectors.opencv_detector import OpenCVCoverDetector, OpenCVHotspotDetector
from hotspot_detector.ui import HotspotBenchmarkApp


class TestOpenCVHotspotDetector(unittest.TestCase):
    def test_detects_components_and_reports_region_maximum_temperature(self):
        image = np.full((80, 100), 25.0, dtype=np.float32)
        image[20:28, 15:23] = 40.0
        image[50:58, 70:78] = 45.0

        result = OpenCVHotspotDetector(threshold_percentile=0.95).detect(
            image, temperature_image=image
        )

        self.assertEqual(len(result.detections), 2)
        self.assertEqual(sorted(round(item["max_temperature"]) for item in result.detections), [40, 45])
        self.assertEqual(sorted(round(item["center_x"]) for item in result.detections), [18, 74])
        self.assertEqual(sorted(round(item["center_y"]) for item in result.detections), [24, 54])

    def test_cover_detector_returns_rotated_four_point_polygon(self):
        image = np.zeros((100, 120), dtype=np.float32)
        cv2 = __import__("cv2")
        rectangle = ((60, 50), (70, 30), 15)
        points = cv2.boxPoints(rectangle).astype(np.int32)
        cv2.fillConvexPoly(image, points, 40.0)

        result = OpenCVCoverDetector(min_area_ratio=0.01).detect(image)

        self.assertEqual(len(result.polygon), 4)
        self.assertGreater(result.confidence, 0.0)

    def test_polygon_cover_result_reports_perfect_iou_for_matching_polygon(self):
        ground_truth = {
            "top_left": (10.0, 10.0),
            "top_right": (50.0, 10.0),
            "bottom_right": (50.0, 30.0),
            "bottom_left": (10.0, 30.0),
        }

        result = HotspotBenchmarkApp._make_polygon_cover_result(
            [(10.0, 10.0), (50.0, 10.0), (50.0, 30.0), (10.0, 30.0)],
            0.9,
            ground_truth,
        )

        self.assertEqual(result["iou"], 1.0)
        self.assertEqual(result["confidence"], 90.0)
        self.assertEqual(result["status"], "PASS")


if __name__ == "__main__":
    unittest.main()