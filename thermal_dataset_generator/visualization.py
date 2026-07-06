"""Debug visualization tools for synthetic thermal samples."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from .labels import BBox


class DebugVisualizer:
    """Renders diagnostic plots for generated thermal samples."""

    @staticmethod
    def show(
        rgb_image: np.ndarray,
        temperature_matrix: np.ndarray,
        bbox: BBox,
        cpu_center: tuple[float, float],
        gpu_center: tuple[float, float] | None,
        heatpipe_points: tuple[tuple[float, float], tuple[float, float]],
        fan_center: tuple[float, float],
        fan_radius: float,
        vent_points: list[tuple[tuple[float, float], tuple[float, float]]],
        trap_center: tuple[float, float] | None = None,
        target_point: tuple[float, float] | None = None,
        hottest_point: tuple[float, float] | None = None,
        hottest_category: str | None = None,
        target_source_attribution: float | None = None,
        title: str = "Thermal Debug View",
    ) -> None:
        """Display thermal debug plots including histogram and vertical profile."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(title)

        axes[0, 0].imshow(rgb_image)
        axes[0, 0].set_title("RGB Thermal Image")
        axes[0, 0].axis("off")
        rect = Rectangle(
            (bbox.x1, bbox.y1),
            bbox.x2 - bbox.x1,
            bbox.y2 - bbox.y1,
            linewidth=2,
            edgecolor="cyan",
            facecolor="none",
        )
        axes[0, 0].add_patch(rect)
        axes[0, 0].plot(cpu_center[0], cpu_center[1], "r+", markersize=12)
        if gpu_center is not None and gpu_center[0] >= 0 and gpu_center[1] >= 0:
            axes[0, 0].plot(gpu_center[0], gpu_center[1], marker="x", color="deepskyblue", markersize=9, markeredgewidth=2)
        if trap_center is not None and trap_center[0] >= 0 and trap_center[1] >= 0:
            axes[0, 0].plot(trap_center[0], trap_center[1], marker="D", color="yellow", markersize=6)

        axes[0, 1].imshow(temperature_matrix, cmap="inferno")
        axes[0, 1].set_title("Temperature Matrix")
        if target_point is not None:
            axes[0, 1].plot(target_point[0], target_point[1], marker="x", color="cyan", markersize=10, markeredgewidth=2)
        if hottest_point is not None:
            axes[0, 1].plot(hottest_point[0], hottest_point[1], marker="+", color="magenta", markersize=12, markeredgewidth=2)
        axes[0, 1].axis("off")

        hx0, hy0 = heatpipe_points[0]
        hx1, hy1 = heatpipe_points[1]
        axes[1, 0].imshow(rgb_image)
        axes[1, 0].set_title("Heat Pipe, Fan, and Vent")
        axes[1, 0].plot([hx0, hx1], [hy0, hy1], color="yellow", linewidth=2)
        for vent_line in vent_points:
            vx0, vy0 = vent_line[0]
            vx1, vy1 = vent_line[1]
            axes[1, 0].plot([vx0, vx1], [vy0, vy1], color="lime", linewidth=2)
        fan = Circle((fan_center[0], fan_center[1]), fan_radius, fill=False, color="blue", linewidth=2)
        axes[1, 0].add_patch(fan)
        axes[1, 0].plot(cpu_center[0], cpu_center[1], "r+", markersize=10)
        if gpu_center is not None and gpu_center[0] >= 0 and gpu_center[1] >= 0:
            axes[1, 0].plot(gpu_center[0], gpu_center[1], marker="x", color="deepskyblue", markersize=8, markeredgewidth=2)
        if trap_center is not None and trap_center[0] >= 0 and trap_center[1] >= 0:
            axes[1, 0].plot(trap_center[0], trap_center[1], marker="D", color="yellow", markersize=6)
        axes[1, 0].axis("off")

        axes[1, 1].hist(temperature_matrix.flatten(), bins=40, color="orangered", alpha=0.8)
        axes[1, 1].set_title("Temperature Histogram")
        axes[1, 1].set_xlabel("Temperature (C)")
        axes[1, 1].set_ylabel("Pixel Count")

        row_mean = temperature_matrix.mean(axis=1)
        axes[0, 2].plot(row_mean, np.arange(len(row_mean)), color="white", linewidth=2)
        axes[0, 2].invert_yaxis()
        axes[0, 2].set_title("Vertical Temperature Profile")
        axes[0, 2].set_xlabel("Mean Row Temp (C)")
        axes[0, 2].set_ylabel("Row Index")
        axes[0, 2].set_facecolor("#111111")
        axes[0, 2].grid(alpha=0.25)

        axes[1, 2].axis("off")
        if hottest_category is not None:
            axes[1, 2].text(0.02, 0.90, f"hottest_category: {hottest_category}", fontsize=10)
        if target_source_attribution is not None:
            axes[1, 2].text(0.02, 0.78, f"target_source_attr: {target_source_attribution:.3f}", fontsize=10)
        axes[1, 2].text(0.02, 0.66, "legend: red +=CPU, blue x=GPU, yellow D=trap", fontsize=9)
        axes[1, 2].text(0.02, 0.56, "legend: yellow line=heatpipe, lime=vent, blue circle=fan", fontsize=9)
        axes[1, 2].text(0.02, 0.46, "legend: cyan x=target, magenta +=global hottest", fontsize=9)

        plt.tight_layout()
        plt.show()
