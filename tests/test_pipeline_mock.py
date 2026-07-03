"""End-to-end mock smoke test for the Phase 1 pipeline."""

from dataclasses import asdict
import json

from shared.interfaces import (
    ActionCommand,
    ActionStatus,
    ActionType,
    CoordinateFrame,
    HotspotDetection,
    RobotTarget,
    ActionResult,
)


def to_json(obj) -> str:
    return json.dumps(asdict(obj), default=str)


def mock_hotspot_detection() -> HotspotDetection:
    d = HotspotDetection(
        hotspot_id="h0",
        x=128.0,
        y=96.0,
        confidence=0.87,
        coordinate_frame=CoordinateFrame.PIXEL,
        image_width=256,
        image_height=192,
    )
    d.validate()
    return d


def mock_coordinate_transform(detection: HotspotDetection) -> RobotTarget:
    """Placeholder transform: replace with real camera/hand-eye calibration later."""
    t = RobotTarget(
        target_id=detection.hotspot_id,
        source_hotspot_id=detection.hotspot_id,
        x=0.30,
        y=0.05,
        z=0.10,
    )
    t.validate()
    return t


def mock_action_execution(command: ActionCommand) -> ActionResult:
    """Placeholder for the PyBullet execution layer."""
    return ActionResult(
        task_id=command.task_id,
        status=ActionStatus.SUCCESS,
        achieved_x=command.target.x,
        achieved_y=command.target.y,
        achieved_z=command.target.z,
        error_mm=1.2,
        message="mock execution ok",
    )


def run_mock_pipeline() -> None:
    detection = mock_hotspot_detection()
    target = mock_coordinate_transform(detection)
    command = ActionCommand(action_type=ActionType.PLACE, target=target)
    result = mock_action_execution(command)

    for step, obj in [
        ("HotspotDetection", detection),
        ("RobotTarget", target),
        ("ActionCommand.target", command.target),
        ("ActionResult", result),
    ]:
        print(f"[{step}] {to_json(obj)}")


if __name__ == "__main__":
    run_mock_pipeline()
