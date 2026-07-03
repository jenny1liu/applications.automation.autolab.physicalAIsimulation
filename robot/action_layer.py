from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional

from shared.interfaces import ActionCommand, ActionResult, ActionStatus, CoordinateFrame, ActionType

try:
    import pybullet as p
    import pybullet_data
except Exception:  # pragma: no cover - optional runtime dependency
    p = None
    pybullet_data = None


@dataclass
class RobotExecutionConfig:
    gui: bool = False
    time_step: float = 1.0 / 240.0
    use_fixed_base: bool = True
    robot_urdf: str = "franka_panda/panda.urdf"
    end_effector_link_index: int = 11
    arm_joint_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    gripper_joint_indices: tuple[int, int] = (9, 10)
    gripper_open_position: float = 0.04
    gripper_close_position: float = 0.0
    ik_steps: int = 180
    gripper_steps: int = 120
    settle_steps: int = 120
    approach_offset_m: float = 0.12
    place_clearance_m: float = 0.02
    lift_offset_m: float = 0.15
    # Left-side view for clear arm and ball motion visibility.
    camera_distance: float = 1.25
    camera_yaw: float = 180.0
    camera_pitch: float = -18.0
    camera_target_x: float = 0.50
    camera_target_y: float = 0.00
    camera_target_z: float = 0.14


class PyBulletActionExecutor:
    """Execute ActionCommand using a PyBullet robot and return ActionResult."""

    def __init__(self, config: Optional[RobotExecutionConfig] = None):
        self.config = config or RobotExecutionConfig()
        self.client_id: Optional[int] = None
        self.robot_id: Optional[int] = None
        self.plane_id: Optional[int] = None

    def setup_environment(self) -> None:
        if p is None or pybullet_data is None:
            raise RuntimeError("pybullet is not installed. Run: pip install pybullet")

        if self.client_id is not None:
            return

        connection_mode = p.GUI if self.config.gui else p.DIRECT
        self.client_id = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.config.time_step)

        if self.config.gui:
            self._configure_viewer()

        self.plane_id = p.loadURDF("plane.urdf")

    def _configure_viewer(self) -> None:
        if p is None:
            return

        # Keep the simulation viewer focused on the scene instead of debug widgets.
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

        p.resetDebugVisualizerCamera(
            cameraDistance=self.config.camera_distance,
            cameraYaw=self.config.camera_yaw,
            cameraPitch=self.config.camera_pitch,
            cameraTargetPosition=[
                self.config.camera_target_x,
                self.config.camera_target_y,
                self.config.camera_target_z,
            ],
        )

    def configure_robot_model(self, urdf_path: Optional[str] = None) -> None:
        if p is None:
            raise RuntimeError("pybullet is not available")
        if self.client_id is None:
            raise RuntimeError("setup_environment() must be called before configure_robot_model()")

        self.robot_id = p.loadURDF(
            urdf_path or self.config.robot_urdf,
            useFixedBase=self.config.use_fixed_base,
        )

    def _move_end_effector(self, x: float, y: float, z: float, steps: Optional[int] = None) -> None:
        if p is None or self.robot_id is None:
            raise RuntimeError("Robot model is not configured")

        joint_targets = p.calculateInverseKinematics(
            self.robot_id,
            self.config.end_effector_link_index,
            [x, y, z],
        )

        for joint_index in self.config.arm_joint_indices:
            p.setJointMotorControl2(
                self.robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_targets[joint_index],
            )

        for _ in range(steps or self.config.ik_steps):
            p.stepSimulation()

    def _set_gripper(self, opening: float) -> None:
        if p is None or self.robot_id is None:
            raise RuntimeError("Robot model is not configured")

        for joint_index in self.config.gripper_joint_indices:
            p.setJointMotorControl2(
                self.robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=opening,
                force=100,
            )

        for _ in range(self.config.gripper_steps):
            p.stepSimulation()

    def _get_achieved_pose(self) -> tuple[float, float, float]:
        if p is None or self.robot_id is None:
            raise RuntimeError("Robot model is not configured")
        link_state = p.getLinkState(self.robot_id, self.config.end_effector_link_index)
        ax, ay, az = link_state[0]
        return float(ax), float(ay), float(az)

    def _execute_move(self, command: ActionCommand) -> tuple[float, float, float]:
        self._move_end_effector(command.target.x, command.target.y, command.target.z)
        return self._get_achieved_pose()

    def _execute_pick(self, command: ActionCommand) -> tuple[float, float, float]:
        tx, ty, tz = command.target.x, command.target.y, command.target.z
        approach_z = tz + self.config.approach_offset_m

        self._set_gripper(self.config.gripper_open_position)
        self._move_end_effector(tx, ty, approach_z)
        self._move_end_effector(tx, ty, tz)
        self._set_gripper(self.config.gripper_close_position)
        self._move_end_effector(tx, ty, tz + self.config.lift_offset_m)
        return self._get_achieved_pose()

    def _execute_place(self, command: ActionCommand) -> tuple[float, float, float]:
        tx, ty, tz = command.target.x, command.target.y, command.target.z
        approach_z = tz + self.config.approach_offset_m
        place_z = tz + self.config.place_clearance_m

        self._move_end_effector(tx, ty, approach_z)
        self._move_end_effector(tx, ty, place_z)
        self._set_gripper(self.config.gripper_open_position)
        self._move_end_effector(tx, ty, tz + self.config.lift_offset_m)
        return self._get_achieved_pose()

    def validate_robot_interface(self, command: ActionCommand) -> Optional[str]:
        if command.target.coordinate_frame != CoordinateFrame.ROBOT_BASE:
            return "ActionCommand.target must be in robot_base frame"
        if self.robot_id is None:
            return "Robot model is not configured. Call configure_robot_model() first"
        return None

    def execute_action(self, command: ActionCommand) -> ActionResult:
        if p is None:
            return ActionResult(
                task_id=command.task_id,
                status=ActionStatus.FAILED,
                message="pybullet is not installed",
            )

        error_message = self.validate_robot_interface(command)
        if error_message:
            return ActionResult(
                task_id=command.task_id,
                status=ActionStatus.FAILED,
                message=error_message,
            )

        tx, ty, tz = command.target.x, command.target.y, command.target.z

        try:
            if command.action_type == ActionType.MOVE:
                ax, ay, az = self._execute_move(command)
            elif command.action_type == ActionType.PICK:
                ax, ay, az = self._execute_pick(command)
            elif command.action_type == ActionType.PLACE:
                ax, ay, az = self._execute_place(command)
            else:
                return ActionResult(
                    task_id=command.task_id,
                    status=ActionStatus.FAILED,
                    message=f"unsupported action type: {command.action_type.value}",
                )

            for _ in range(self.config.settle_steps):
                p.stepSimulation()
        except Exception as ex:
            return ActionResult(
                task_id=command.task_id,
                status=ActionStatus.FAILED,
                message=f"pybullet execution failed: {ex}",
            )

        error_mm = 1000.0 * sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)

        return ActionResult(
            task_id=command.task_id,
            status=ActionStatus.SUCCESS,
            achieved_x=float(ax),
            achieved_y=float(ay),
            achieved_z=float(az),
            error_mm=float(error_mm),
            message=f"executed {command.action_type.value} with franka panda",
        )

    def shutdown(self) -> None:
        if p is None:
            return
        if self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None
            self.robot_id = None
            self.plane_id = None


def run_smoke_demo() -> None:
    from shared.interfaces import ActionType, RobotTarget

    executor = PyBulletActionExecutor()
    try:
        executor.setup_environment()
        executor.configure_robot_model()
        command = ActionCommand(
            action_type=ActionType.PICK,
            target=RobotTarget(
                target_id="demo-target",
                source_hotspot_id="demo-hotspot",
                x=0.35,
                y=0.0,
                z=0.12,
            ),
        )
        result = executor.execute_action(command)
        print(result)
    finally:
        executor.shutdown()


if __name__ == "__main__":
    run_smoke_demo()
