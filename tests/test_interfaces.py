import unittest

from shared.interfaces import (
    ActionCommand,
    ActionStatus,
    ActionType,
    CoordinateFrame,
    HotspotDetection,
    RobotTarget,
)


class TestInterfaces(unittest.TestCase):
    def test_hotspot_validate_pixel_requires_image_size(self):
        detection = HotspotDetection(
            hotspot_id="h0",
            x=10.0,
            y=20.0,
            confidence=0.8,
            coordinate_frame=CoordinateFrame.PIXEL,
            image_width=256,
            image_height=192,
        )
        detection.validate()

        bad_detection = HotspotDetection(
            hotspot_id="h1",
            x=10.0,
            y=20.0,
            confidence=0.8,
            coordinate_frame=CoordinateFrame.PIXEL,
        )
        with self.assertRaises(AssertionError):
            bad_detection.validate()

    def test_hotspot_validate_confidence_range(self):
        bad_detection = HotspotDetection(
            hotspot_id="h2",
            x=1.0,
            y=2.0,
            confidence=1.5,
            coordinate_frame=CoordinateFrame.CAMERA,
            z=0.3,
        )
        with self.assertRaises(AssertionError):
            bad_detection.validate()

    def test_robot_target_validate_requires_robot_base(self):
        target = RobotTarget(
            target_id="t0",
            source_hotspot_id="h0",
            x=0.3,
            y=0.05,
            z=0.1,
            coordinate_frame=CoordinateFrame.ROBOT_BASE,
        )
        target.validate()

        bad_target = RobotTarget(
            target_id="t1",
            source_hotspot_id="h1",
            x=0.3,
            y=0.05,
            z=0.1,
            coordinate_frame=CoordinateFrame.PIXEL,
        )
        with self.assertRaises(AssertionError):
            bad_target.validate()

    def test_action_command_auto_task_id(self):
        target = RobotTarget(
            target_id="h0",
            source_hotspot_id="h0",
            x=0.3,
            y=0.05,
            z=0.1,
        )
        command = ActionCommand(action_type=ActionType.PLACE, target=target)
        self.assertEqual(command.task_id, "place-h0")

    def test_action_status_values(self):
        self.assertEqual(ActionStatus.SUCCESS.value, "success")
        self.assertEqual(ActionStatus.FAILED.value, "failed")


if __name__ == "__main__":
    unittest.main()
