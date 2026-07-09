import unittest

from robot.action_layer import PyBulletActionExecutor
from shared.interfaces import ActionCommand, ActionType, CoordinateFrame, RobotTarget, ActionStatus


class TestRobotActionLayer(unittest.TestCase):
    def test_executor_proxies_pybullet_resource_ids(self):
        executor = PyBulletActionExecutor()

        self.assertIsNone(executor.client_id)
        self.assertIsNone(executor.robot_id)
        self.assertIsNone(executor.plane_id)

    def test_validate_robot_interface_rejects_non_robot_base(self):
        executor = PyBulletActionExecutor()
        command = ActionCommand(
            action_type=ActionType.MOVE,
            target=RobotTarget(
                target_id="t0",
                source_hotspot_id="h0",
                x=0.1,
                y=0.2,
                z=0.3,
                coordinate_frame=CoordinateFrame.PIXEL,
            ),
        )
        message = executor.validate_robot_interface(command)
        self.assertEqual(message, "ActionCommand.target must be in robot_base frame")

    def test_execute_action_returns_failed_when_not_configured(self):
        executor = PyBulletActionExecutor()
        command = ActionCommand(
            action_type=ActionType.MOVE,
            target=RobotTarget(
                target_id="t1",
                source_hotspot_id="h1",
                x=0.1,
                y=0.2,
                z=0.3,
                coordinate_frame=CoordinateFrame.ROBOT_BASE,
            ),
        )
        result = executor.execute_action(command)

        # If pybullet is installed, this checks missing robot setup path.
        # If pybullet is not installed, this checks missing dependency path.
        self.assertEqual(result.task_id, command.task_id)
        self.assertEqual(result.status, ActionStatus.FAILED)
        self.assertTrue(result.message)


if __name__ == "__main__":
    unittest.main()
