import unittest

import numpy as np

from thermal.detectors.opencv_detector import OpenCVHotspotDetector


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


if __name__ == "__main__":
    unittest.main()