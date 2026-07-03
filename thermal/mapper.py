from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class RobotCoordinate:
    X: float
    Y: float
    Z: float


class RobotTargetMapper:
    """Map pixel coordinates to robot arm target coordinates."""

    def __init__(
        self,
        camera_fov_x: float = 60.0,
        camera_fov_y: float = 45.0,
        distance_to_surface: float = 0.5,
        camera_height: float = 0.1,
        pixel_width: int = 320,
        pixel_height: int = 240,
    ):
        self.camera_fov_x = camera_fov_x
        self.camera_fov_y = camera_fov_y
        self.distance_to_surface = distance_to_surface
        self.camera_height = camera_height
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height

    def pixel_to_robot(
        self,
        pixel_x: float,
        pixel_y: float,
    ) -> RobotCoordinate:
        """Convert pixel coordinate to robot coordinate."""
        norm_x = (pixel_x - self.pixel_width / 2.0) / self.pixel_width
        norm_y = (pixel_y - self.pixel_height / 2.0) / self.pixel_height

        angle_x_rad = np.radians(self.camera_fov_x / 2.0) * norm_x
        angle_y_rad = np.radians(self.camera_fov_y / 2.0) * norm_y

        X = self.distance_to_surface * np.tan(angle_x_rad)
        Y = self.distance_to_surface * np.tan(angle_y_rad)
        Z = self.camera_height + self.distance_to_surface

        return RobotCoordinate(X=float(X), Y=float(Y), Z=float(Z))

    def set_camera_params(
        self,
        fov_x: float,
        fov_y: float,
        distance: float,
        height: float,
        pixel_width: int,
        pixel_height: int,
    ) -> None:
        """Update camera parameters."""
        self.camera_fov_x = fov_x
        self.camera_fov_y = fov_y
        self.distance_to_surface = distance
        self.camera_height = height
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
