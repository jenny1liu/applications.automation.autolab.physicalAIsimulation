from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

from thermal_dataset_generator.config import GeneratorConfig as DatasetGeneratorConfig
from thermal_dataset_generator.generator import ThermalDatasetGenerator


class HotspotShape(Enum):
    CIRCULAR = "circular"
    IRREGULAR = "irregular"


@dataclass
class ThermalFrame:
    # Raw temperature matrix in Celsius.
    image: np.ndarray
    # Union hotspot mask for model training.
    mask: np.ndarray
    centers: list[Tuple[float, float]]
    temperatures: list[float]
    width: int
    height: int
    # FLIR-style rendered RGB image.
    thermal_image: Optional[np.ndarray] = None
    # Per-source binary masks: cpu, gpu, heatpipe, keyboard, fan, palm_rest.
    region_masks: dict[str, np.ndarray] = field(default_factory=dict)
    # Highest temperature pixel as global hotspot ground truth.
    hotspot_coordinate: Optional[Tuple[float, float]] = None
    hotspot_temperature: Optional[float] = None
    hottest_region_category: str = "genuine_source"
    excluded_hottest_region: Optional[str] = None
    target_source_attribution_fraction: Optional[float] = None
    workload: str = "medium"
    power_class: str = "unknown"
    keyboard_plateau_coverage: float = 0.0
    dgpu_enabled: bool = False


class ThermalImageGenerator:
    """Generate realistic laptop thermal scenes for synthetic data creation.

    Calibration notes (v3):
    Tuned against a real Surface Laptop thermal reference where keyboard
    center is hottest (~35C), top hinge band is warm, and left/right side
    edges are visibly cooler. Surface temperatures are diffuse, so the scene
    uses broad blobs and a global diffusion pass instead of sharp hotspots.
    """

    WORKLOAD_PROFILES = {
        "light": {
            "cpu_temp": (31.5, 34.5),
            "gpu_temp": (30.0, 33.0),
            "fan_temp": (28.5, 31.5),
            "keyboard_gain": (0.10, 0.16),
            "max_temp": (33.0, 36.0),
        },
        "medium": {
            "cpu_temp": (33.5, 37.5),
            "gpu_temp": (31.0, 35.0),
            "fan_temp": (29.0, 33.0),
            "keyboard_gain": (0.14, 0.22),
            "max_temp": (35.0, 39.0),
        },
        "heavy": {
            "cpu_temp": (36.0, 41.0),
            "gpu_temp": (33.0, 38.0),
            "fan_temp": (30.0, 35.0),
            "keyboard_gain": (0.20, 0.30),
            "max_temp": (38.0, 43.0),
        },
        "cpu_stress": {
            "cpu_temp": (38.0, 44.0),
            "gpu_temp": (31.0, 36.0),
            "fan_temp": (31.0, 36.0),
            "keyboard_gain": (0.18, 0.28),
            "max_temp": (40.0, 46.0),
        },
        "gpu_stress": {
            "cpu_temp": (32.0, 38.0),
            "gpu_temp": (37.0, 43.0),
            "fan_temp": (31.0, 36.0),
            "keyboard_gain": (0.17, 0.27),
            "max_temp": (40.0, 46.0),
        },
        "dual_stress": {
            "cpu_temp": (39.0, 46.0),
            "gpu_temp": (37.0, 44.0),
            "fan_temp": (32.0, 38.0),
            "keyboard_gain": (0.22, 0.34),
            "max_temp": (43.0, 50.0),
        },
    }

    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        background_temp: float = 28.0,
        noise_std: float = 1.0,
        seed: Optional[int] = None,
    ):
        if (width, height) not in {(320, 240), (640, 480)}:
            raise ValueError("Supported resolutions are 320x240 and 640x480")
        self.width = width
        self.height = height
        self.background_temp = background_temp
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        hotspot_count: int = 1,
        hotspot_temp_range: Tuple[float, float] = (80.0, 100.0),
        hotspot_radius_range: Tuple[int, int] = (8, 20),
        shape: HotspotShape = HotspotShape.CIRCULAR,
        workload: Optional[str] = None,
        skip_area: Optional[Tuple[int, int, int, int]] = None,
        target_area: Optional[Tuple[int, int, int, int]] = None,
    ) -> ThermalFrame:
        """Generate a realistic FLIR-like laptop thermal scene."""
        del hotspot_temp_range, hotspot_radius_range, shape

        scene_workload = workload or self._workload_from_hotspot_count(hotspot_count)
        if scene_workload not in self.WORKLOAD_PROFILES:
            raise ValueError(f"Unknown workload '{scene_workload}'")

        profile = self.WORKLOAD_PROFILES[scene_workload]
        ambient = float(np.clip(self.background_temp + self.rng.uniform(-1.8, 0.6), 24.0, 32.0))

        yy, xx = np.indices((self.height, self.width), dtype=np.float32)
        temp = np.full((self.height, self.width), ambient, dtype=np.float32)

        # Global base: keyboard deck warmer than side walls and palm corners.
        top_bias = np.clip(1.0 - (yy / max(self.height - 1, 1)), 0.0, 1.0)
        center_bias = 1.0 - np.abs((xx - self.width * 0.5) / (self.width * 0.5 + 1e-6))
        edge_x = np.abs((xx - self.width * 0.5) / (self.width * 0.5 + 1e-6))
        side_cooling = np.clip((edge_x - 0.62) / 0.38, 0.0, 1.0)
        temp += 1.0 * top_bias + 0.9 * center_bias - 1.6 * side_cooling

        # Randomize layout family to avoid a single concentrated hotspot shape.
        # Bias toward split_keyboard for broader multi-peak distributions.
        layout_mode = str(
            self.rng.choice(
                ["keyboard_center", "top_vent_band", "split_keyboard"],
                p=[0.18, 0.36, 0.46],
            )
        )
        if layout_mode == "top_vent_band":
            cpu_cx = float(self.width * self.rng.uniform(0.44, 0.56))
            cpu_cy = float(self.height * self.rng.uniform(0.18, 0.25))
            gpu_cx = float(np.clip(cpu_cx + self.width * self.rng.uniform(-0.10, 0.10), 0, self.width - 1))
            gpu_cy = float(np.clip(cpu_cy + self.height * self.rng.uniform(-0.03, 0.04), 0, self.height - 1))
            fan_cx = float(self.width * self.rng.uniform(0.60, 0.78))
            fan_cy = float(self.height * self.rng.uniform(0.14, 0.22))
            cpu_sigma_x, cpu_sigma_y = self.width * 0.25, self.height * 0.10
            gpu_sigma_x, gpu_sigma_y = self.width * 0.22, self.height * 0.10
            fan_sigma_x, fan_sigma_y = self.width * 0.36, self.height * 0.08
        elif layout_mode == "split_keyboard":
            cpu_cx = float(self.width * self.rng.uniform(0.35, 0.46))
            cpu_cy = float(self.height * self.rng.uniform(0.42, 0.54))
            gpu_cx = float(self.width * self.rng.uniform(0.56, 0.68))
            gpu_cy = float(self.height * self.rng.uniform(0.40, 0.54))
            fan_cx = float(self.width * self.rng.uniform(0.48, 0.58))
            fan_cy = float(self.height * self.rng.uniform(0.16, 0.24))
            cpu_sigma_x, cpu_sigma_y = self.width * 0.20, self.height * 0.13
            gpu_sigma_x, gpu_sigma_y = self.width * 0.20, self.height * 0.13
            fan_sigma_x, fan_sigma_y = self.width * 0.29, self.height * 0.09
        else:
            # keyboard_center
            cpu_cx = float(self.width * self.rng.uniform(0.48, 0.57))
            cpu_cy = float(self.height * self.rng.uniform(0.44, 0.54))
            gpu_cx = float(np.clip(cpu_cx - self.width * self.rng.uniform(0.12, 0.22), 0, self.width - 1))
            gpu_cy = float(np.clip(cpu_cy + self.height * self.rng.uniform(-0.04, 0.05), 0, self.height - 1))
            fan_cx = float(self.width * self.rng.uniform(0.48, 0.58))
            fan_cy = float(self.height * self.rng.uniform(0.16, 0.24))
            cpu_sigma_x, cpu_sigma_y = self.width * 0.22, self.height * 0.12
            gpu_sigma_x, gpu_sigma_y = self.width * 0.18, self.height * 0.12
            fan_sigma_x, fan_sigma_y = self.width * 0.34, self.height * 0.08

        cpu_peak = float(self.rng.uniform(*profile["cpu_temp"]))
        gpu_peak = float(min(cpu_peak - self.rng.uniform(1.5, 7.0), self.rng.uniform(*profile["gpu_temp"])))
        fan_peak = float(self.rng.uniform(*profile["fan_temp"]))

        cpu_amp = max(cpu_peak - ambient, 1.0)
        gpu_amp = max(gpu_peak - ambient, 1.0)
        fan_amp = max(fan_peak - ambient, 0.5)

        cpu_blob = self._gaussian_2d(xx, yy, cpu_cx, cpu_cy, cpu_sigma_x, cpu_sigma_y)
        gpu_blob = self._gaussian_2d(xx, yy, gpu_cx, gpu_cy, gpu_sigma_x, gpu_sigma_y)
        fan_blob = self._gaussian_2d(xx, yy, fan_cx, fan_cy, fan_sigma_x, fan_sigma_y)

        # Heatpipe effect: shallow warm strip between top band and keyboard.
        pipe_cx = float((cpu_cx + fan_cx) * 0.5)
        pipe_cy = float((cpu_cy + fan_cy) * 0.5 + self.height * self.rng.uniform(-0.01, 0.01))
        heatpipe_blob = self._gaussian_2d(
            xx,
            yy,
            pipe_cx,
            pipe_cy,
            self.width * 0.30,
            self.height * 0.065,
        )

        # Warm strip above F1-F12 region and a broader keyboard blanket.
        top_strip = self._gaussian_2d(
            xx,
            yy,
            self.width * self.rng.uniform(0.50, 0.58),
            self.height * self.rng.uniform(0.20, 0.27),
            self.width * 0.42,
            self.height * 0.085,
        )
        keyboard_blanket = self._gaussian_2d(
            xx,
            yy,
            self.width * 0.50,
            self.height * 0.50,
            self.width * 0.40,
            self.height * 0.28,
        )

        side_left = self._gaussian_2d(
            xx,
            yy,
            self.width * self.rng.uniform(0.02, 0.07),
            self.height * self.rng.uniform(0.74, 0.86),
            self.width * 0.06,
            self.height * 0.07,
        )
        side_right = self._gaussian_2d(
            xx,
            yy,
            self.width * self.rng.uniform(0.93, 0.98),
            self.height * self.rng.uniform(0.74, 0.86),
            self.width * 0.06,
            self.height * 0.07,
        )

        temp += 0.72 * cpu_amp * cpu_blob
        temp += 0.64 * gpu_amp * gpu_blob
        temp += 0.50 * fan_amp * fan_blob
        temp += (0.24 * cpu_amp + 0.20 * gpu_amp) * heatpipe_blob
        temp += (0.30 * fan_amp + 0.18 * cpu_amp) * (side_left + side_right)
        temp += (0.24 * cpu_amp + 0.14 * fan_amp) * top_strip
        temp += 0.26 * cpu_amp * keyboard_blanket

        # Add a few diffuse local sources so hotspots are distributed instead
        # of collapsing to one dominant point.
        for _ in range(int(self.rng.integers(4, 7))):
            local_blob = self._gaussian_2d(
                xx,
                yy,
                self.width * float(self.rng.uniform(0.28, 0.72)),
                self.height * float(self.rng.uniform(0.26, 0.78)),
                self.width * float(self.rng.uniform(0.08, 0.16)),
                self.height * float(self.rng.uniform(0.07, 0.14)),
            )
            local_weight = float(self.rng.uniform(0.05, 0.11))
            temp += local_weight * (0.7 * cpu_amp + 0.3 * gpu_amp) * local_blob

        # Add low-amplitude warm patches around lower keyboard/palm transition
        # to prevent the map from collapsing into only top-band peaks.
        for _ in range(int(self.rng.integers(2, 4))):
            lower_blob = self._gaussian_2d(
                xx,
                yy,
                self.width * float(self.rng.uniform(0.30, 0.70)),
                self.height * float(self.rng.uniform(0.62, 0.82)),
                self.width * float(self.rng.uniform(0.12, 0.22)),
                self.height * float(self.rng.uniform(0.09, 0.16)),
            )
            lower_weight = float(self.rng.uniform(0.04, 0.10))
            temp += lower_weight * (0.6 * cpu_amp + 0.4 * fan_amp) * lower_blob

        # Keyboard conduction with smooth diffusion and gradients.
        source_energy = np.clip(
            cpu_blob + gpu_blob + 1.0 * heatpipe_blob + 0.9 * top_strip + 0.8 * keyboard_blanket,
            0.0,
            3.2,
        ).astype(np.float32)
        conduction = cv2.GaussianBlur(source_energy, (0, 0), sigmaX=self.width * 0.13, sigmaY=self.height * 0.25)
        conduction /= np.max(conduction) + 1e-6

        keyboard_mask = np.zeros((self.height, self.width), dtype=np.float32)
        key_x0 = int(self.width * 0.12)
        key_x1 = int(self.width * 0.88)
        key_y0 = int(self.height * 0.24)
        key_y1 = int(self.height * 0.75)
        keyboard_mask[key_y0:key_y1, key_x0:key_x1] = 1.0

        # Faint key-row texture; kept subtle since real IR footage shows a
        # smooth gradient across the keyboard, not a visible row pattern.
        row_pattern = np.sin(np.linspace(0, np.pi * 8, self.height, dtype=np.float32))[:, None]
        row_pattern = 0.02 * (row_pattern + 1.0)
        keyboard_gain = float(self.rng.uniform(*profile["keyboard_gain"]))
        temp += keyboard_mask * (keyboard_gain * cpu_amp * conduction + row_pattern * (cpu_amp * 0.02))

        # Palm rest: slightly cooler region, but avoid heavy bottom-only tint.
        palm_y0 = int(self.height * 0.76)
        palm_cool = float(self.rng.uniform(0.8, 2.1))
        temp[palm_y0:, :] -= palm_cool

        # Overall thermal diffusion blur: real chassis surfaces spread heat
        # smoothly, so sharpen-free blending here avoids the hard synthetic
        # edges a raw sum of Gaussians otherwise produces.
        temp = cv2.GaussianBlur(temp, (0, 0), sigmaX=self.width * 0.028, sigmaY=self.height * 0.028)

        temp = self._add_sensor_noise(temp, ambient)

        target_max = float(self.rng.uniform(*profile["max_temp"]))
        current_max = float(np.max(temp))
        if current_max > ambient + 1e-6:
            gain = (target_max - ambient) / (current_max - ambient)
            gain = min(gain, 1.35)
            temp = ambient + (temp - ambient) * gain

        skip_mask = np.zeros((self.height, self.width), dtype=bool)
        if skip_area is not None:
            x0, y0, x1, y1 = self._normalize_rect(skip_area)
            skip_mask[y0:y1, x0:x1] = True

        targetPoint: Optional[Tuple[int, int]] = None
        if target_area is not None:
            tx0, ty0, tx1, ty1 = self._normalize_rect(target_area)
            fx = int(np.clip(round((tx0 + tx1 - 1) * 0.5), 0, self.width - 1))
            fy = int(np.clip(round((ty0 + ty1 - 1) * 0.5), 0, self.height - 1))
            targetPoint = (fx, fy)
            spanX = max(3.0, float(tx1 - tx0) * 0.38)
            spanY = max(3.0, float(ty1 - ty0) * 0.38)
            fixed_blob = self._gaussian_2d(
                xx,
                yy,
                float(fx),
                float(fy),
                spanX,
                spanY,
            )
            desired_fixed_temp = min(95.0, max(target_max, profile["max_temp"][1]))
            delta = max(0.0, desired_fixed_temp - float(temp[fy, fx]))
            if delta > 0.0:
                temp += delta * fixed_blob

            # Keep whole target area hotter than surroundings with smooth floor temperature.
            areaFloor = ambient + max(8.0, delta * 0.55)
            temp[ty0:ty1, tx0:tx1] = np.maximum(temp[ty0:ty1, tx0:tx1], areaFloor)

        if np.any(skip_mask):
            # Keep forbidden area warm enough for realism, but below hotspot threshold.
            temp[skip_mask] = np.minimum(temp[skip_mask], ambient + 3.0)

        temp = np.clip(temp, 22.0, 95.0).astype(np.float32)

        region_masks = {
            "cpu": (cpu_blob > 0.35).astype(np.uint8) * 255,
            "gpu": (gpu_blob > 0.35).astype(np.uint8) * 255,
            "heatpipe": (heatpipe_blob > 0.24).astype(np.uint8) * 255,
            "keyboard": (keyboard_mask > 0.5).astype(np.uint8) * 255,
            "fan": (fan_blob > 0.35).astype(np.uint8) * 255,
            "palm_rest": np.pad(
                np.ones((self.height - palm_y0, self.width), dtype=np.uint8) * 255,
                ((palm_y0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            ),
        }

        # Keep threshold tied to ambient so masks remain stable after profile tuning.
        train_mask = ((temp > (ambient + 3.2)) & (yy < self.height * 0.80) & (~skip_mask)).astype(np.uint8) * 255

        hottest_flat_idx = int(np.argmax(temp))
        hot_y, hot_x = np.unravel_index(hottest_flat_idx, temp.shape)
        hotspot_coordinate = (float(hot_x), float(hot_y))
        hotspot_temperature = float(temp[hot_y, hot_x])

        display_low = float(max(20.0, ambient - 2.5))
        display_high = float(max(display_low + 8.0, target_max + 1.5))
        thermal_rgb = self.render_flir(temp, temp_range=(display_low, display_high))

        forcedPeak: Optional[Tuple[float, float, float]] = None
        effectiveHotspotCount = int(hotspot_count)
        if targetPoint is not None:
            fx, fy = targetPoint
            if not skip_mask[fy, fx]:
                forcedPeak = (float(fx), float(fy), float(temp[fy, fx]))
                effectiveHotspotCount += 1

        top_hotspots = self._extract_top_hotspots(
            temp,
            effectiveHotspotCount,
            forbidden_mask=skip_mask,
            forced_peak=forcedPeak,
        )
        hotspot_centers = [(x, y) for x, y, _ in top_hotspots]
        hotspot_temps = [t for _, _, t in top_hotspots]

        return ThermalFrame(
            image=temp,
            mask=train_mask,
            centers=hotspot_centers,
            temperatures=hotspot_temps,
            width=self.width,
            height=self.height,
            thermal_image=thermal_rgb,
            region_masks=region_masks,
            hotspot_coordinate=hotspot_coordinate,
            hotspot_temperature=hotspot_temperature,
            workload=scene_workload,
        )

    def generate_batch(
        self,
        sample_count: int,
        workloads: Optional[Iterable[str]] = None,
    ) -> list[ThermalFrame]:
        """Generate many unique laptop thermal scenes, suitable for 100+ samples."""
        if sample_count <= 0:
            return []

        workload_list = list(workloads) if workloads is not None else list(self.WORKLOAD_PROFILES.keys())
        if not workload_list:
            workload_list = list(self.WORKLOAD_PROFILES.keys())

        frames: list[ThermalFrame] = []
        for _ in range(sample_count):
            selected = str(self.rng.choice(workload_list))
            frames.append(self.generate(workload=selected))
        return frames

    def render_flir(
        self,
        temperature_matrix: np.ndarray,
        temp_range: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """Render a FLIR-like RGB thermal image from raw temperature matrix."""
        if temp_range is None:
            low = float(np.min(temperature_matrix))
            high = float(np.max(temperature_matrix))
        else:
            low = float(temp_range[0])
            high = float(temp_range[1])
        if high <= low:
            high = low + 1e-3

        norm = np.clip((temperature_matrix - low) / (high - low), 0.0, 1.0)
        norm_u8 = (norm * 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_INFERNO)

        # Mild sensor blur and vignette for camera-like rendering.
        colored = cv2.GaussianBlur(colored, (3, 3), 0.8)
        yy, xx = np.indices((self.height, self.width), dtype=np.float32)
        rr = np.sqrt((xx - self.width * 0.5) ** 2 + (yy - self.height * 0.5) ** 2)
        rr /= np.max(rr) + 1e-6
        vignette = np.clip(1.0 - 0.18 * rr, 0.80, 1.0)
        colored = np.clip(colored.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)

        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    def _workload_from_hotspot_count(self, hotspot_count: int) -> str:
        """Keep backward compatibility with existing GUI hotspot slider."""
        map_by_count = {
            1: "light",
            2: "medium",
            3: "heavy",
            4: "cpu_stress",
            5: "dual_stress",
        }
        return map_by_count.get(int(np.clip(hotspot_count, 1, 5)), "medium")

    def _add_sensor_noise(self, temp: np.ndarray, ambient: float) -> np.ndarray:
        """Add realistic thermal sensor noise and dead pixels."""
        noisy = temp.copy()

        gaussian = self.rng.normal(0.0, self.noise_std, size=noisy.shape).astype(np.float32)
        low_freq = cv2.GaussianBlur(
            self.rng.normal(0.0, self.noise_std * 0.35, size=noisy.shape).astype(np.float32),
            (0, 0),
            sigmaX=self.width * 0.02,
            sigmaY=self.height * 0.02,
        )
        noisy += gaussian + low_freq

        # Small line-wise fluctuation similar to sensor readout drift.
        row_drift = self.rng.normal(0.0, 0.08, size=(self.height, 1)).astype(np.float32)
        noisy += row_drift

        dead_pixel_ratio = self.rng.uniform(0.0003, 0.0013)
        dead_count = max(1, int(self.width * self.height * dead_pixel_ratio))
        ys = self.rng.integers(0, self.height, dead_count)
        xs = self.rng.integers(0, self.width, dead_count)
        dead_delta = self.rng.choice([-1.0, 1.0], size=dead_count).astype(np.float32) * self.rng.uniform(
            2.0,
            7.0,
            size=dead_count,
        ).astype(np.float32)
        noisy[ys, xs] = np.clip(noisy[ys, xs] + dead_delta, ambient - 3.0, 98.0)

        return noisy

    def _extract_top_hotspots(
        self,
        temp: np.ndarray,
        hotspot_count: int,
        forbidden_mask: Optional[np.ndarray] = None,
        forced_peak: Optional[Tuple[float, float, float]] = None,
    ) -> list[Tuple[float, float, float]]:
        """Extract top-N distinct hotspot peaks using simple non-maximum suppression."""
        count = max(1, int(hotspot_count))
        work = temp.copy()
        peaks: list[Tuple[float, float, float]] = []

        if forbidden_mask is not None and forbidden_mask.shape == work.shape:
            work[forbidden_mask] = -1e9

        suppress_r = max(8, int(min(self.width, self.height) * 0.070))

        def suppress_neighborhood(x: int, y: int) -> None:
            y0 = max(0, y - suppress_r)
            y1 = min(self.height, y + suppress_r + 1)
            x0 = max(0, x - suppress_r)
            x1 = min(self.width, x + suppress_r + 1)
            work[y0:y1, x0:x1] = -1e9

        if forced_peak is not None and len(peaks) < count:
            fx = int(np.clip(round(float(forced_peak[0])), 0, self.width - 1))
            fy = int(np.clip(round(float(forced_peak[1])), 0, self.height - 1))
            if work[fy, fx] > -1e8:
                ft = float(temp[fy, fx])
                peaks.append((float(fx), float(fy), ft))
                suppress_neighborhood(fx, fy)

        while len(peaks) < count:
            flat_idx = int(np.argmax(work))
            yRaw, xRaw = np.unravel_index(flat_idx, work.shape)
            yIdx = int(yRaw)
            xIdx = int(xRaw)
            t = float(work[yIdx, xIdx])
            if t <= -1e8:
                break
            peaks.append((float(xIdx), float(yIdx), t))
            suppress_neighborhood(xIdx, yIdx)

        return peaks

    def _normalize_rect(self, rect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Clamp and normalize rectangle to image bounds, returning [x0, y0, x1, y1)."""
        x0, y0, x1, y1 = [int(v) for v in rect]
        left = int(np.clip(min(x0, x1), 0, self.width - 1))
        right = int(np.clip(max(x0, x1), 0, self.width - 1))
        top = int(np.clip(min(y0, y1), 0, self.height - 1))
        bottom = int(np.clip(max(y0, y1), 0, self.height - 1))
        return left, top, right + 1, bottom + 1

    @staticmethod
    def _gaussian_2d(
        xx: np.ndarray,
        yy: np.ndarray,
        cx: float,
        cy: float,
        sigma_x: float,
        sigma_y: float,
    ) -> np.ndarray:
        """Generate normalized anisotropic Gaussian field."""
        sx = max(float(sigma_x), 1.0)
        sy = max(float(sigma_y), 1.0)
        val = np.exp(-(((xx - cx) ** 2) / (2.0 * sx * sx) + ((yy - cy) ** 2) / (2.0 * sy * sy)))
        return val.astype(np.float32)

    def set_seed(self, seed: int) -> None:
        """Reset RNG seed for reproducibility."""
        self.rng = np.random.default_rng(seed)


class DatasetBackedThermalGenerator:
    """Adapter that exposes dataset generator samples as ThermalFrame."""

    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        background_temp: float = 28.0,
        noise_std: float = 1.0,
        seed: Optional[int] = None,
        physics_score_threshold: float = 0.72,
        strict_max_attempts: int = 20,
    ):
        self.width = width
        self.height = height
        self.background_temp = background_temp
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.physics_score_threshold = float(np.clip(physics_score_threshold, 0.0, 1.0))
        self.strict_max_attempts = max(1, int(strict_max_attempts))

        base_size = self._select_base_size(max(width, height))
        self._dataset_generator = ThermalDatasetGenerator(
            DatasetGeneratorConfig(
                image_size=base_size,
                dataset_size=1,
                output_dir=Path("thermal_dataset_generator") / "output",
                random_seed=int(seed if seed is not None else 42),
                workers=0,
                debug_mode=False,
                save_mask=False,
                save_temperature=False,
                save_json=False,
                noise_level=max(0.2, float(noise_std) * 0.55),
                enable_physics_gate=True,
                physics_score_threshold=self.physics_score_threshold,
                physics_max_attempts=max(8, self.strict_max_attempts),
            )
        )
        cfg = self._dataset_generator.config
        self._default_power_class_weights = tuple(cfg.power_class_weights)
        self._default_enable_gpu = bool(cfg.enable_gpu)
        self._default_gpu_probs = (
            float(cfg.thin_gpu_probability),
            float(cfg.mainstream_gpu_probability),
            float(cfg.gaming_gpu_probability),
        )
        self.power_class_override: Optional[str] = None
        self.force_gpu_mode: str = "auto"

    def set_runtime_controls(
        self,
        *,
        noise_std: Optional[float] = None,
        physics_threshold: Optional[float] = None,
        power_class: Optional[str] = None,
        force_gpu_mode: Optional[str] = None,
    ) -> None:
        """Update runtime semantic controls used by dataset-backed generation."""
        if noise_std is not None:
            self.noise_std = float(max(0.0, noise_std))
        if physics_threshold is not None:
            self.physics_score_threshold = float(np.clip(physics_threshold, 0.0, 1.0))
        if power_class is not None:
            normalized = str(power_class).strip().lower()
            self.power_class_override = normalized if normalized in {"thin", "mainstream", "gaming"} else None
        if force_gpu_mode is not None:
            normalized_gpu = str(force_gpu_mode).strip().lower()
            self.force_gpu_mode = normalized_gpu if normalized_gpu in {"auto", "on", "off"} else "auto"

    def generate(
        self,
        hotspot_count: int = 1,
        hotspot_temp_range: Tuple[float, float] = (80.0, 100.0),
        hotspot_radius_range: Tuple[int, int] = (8, 20),
        shape: HotspotShape = HotspotShape.CIRCULAR,
        workload: Optional[str] = None,
        skip_area: Optional[Tuple[int, int, int, int]] = None,
        target_area: Optional[Tuple[int, int, int, int]] = None,
    ) -> ThermalFrame:
        """Generate frame using the synthetic dataset pipeline for UI and benchmarking."""
        del hotspot_temp_range, hotspot_radius_range, shape
        del hotspot_count, workload

        sample = self._generate_threshold_passed_sample()

        temp = sample.matrix.astype(np.float32)
        cpu_mask = sample.cpu_mask.astype(np.uint8)
        thermal_rgb = sample.rgb.astype(np.uint8)
        source_h, source_w = temp.shape

        if target_area is not None:
            tx0, ty0, tx1, ty1 = self._normalize_rect(target_area)
            temp[ty0:ty1, tx0:tx1] = np.maximum(temp[ty0:ty1, tx0:tx1], float(np.max(temp)) + 2.0)
            cpu_mask[ty0:ty1, tx0:tx1] = 255

        if skip_area is not None:
            sx0, sy0, sx1, sy1 = self._normalize_rect(skip_area)
            temp[sy0:sy1, sx0:sx1] = np.minimum(temp[sy0:sy1, sx0:sx1], self.background_temp + 2.5)
            cpu_mask[sy0:sy1, sx0:sx1] = 0

        if temp.shape != (self.height, self.width):
            temp = cv2.resize(temp, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            cpu_mask = cv2.resize(cpu_mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            thermal_rgb = cv2.resize(thermal_rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        region_masks: dict[str, np.ndarray] = {"cpu": cpu_mask.astype(np.uint8)}
        for name, mask in sample.component_masks.items():
            mask_u8 = mask.astype(np.uint8)
            if mask_u8.shape != (self.height, self.width):
                mask_u8 = cv2.resize(mask_u8, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            region_masks[name] = mask_u8

        keyboard_mask = region_masks.get("keyboard_area")
        if keyboard_mask is None or not np.any(keyboard_mask > 0):
            keyboard_mask = region_masks.get("keyboard")

        if keyboard_mask is not None and np.any(keyboard_mask > 0):
            keyboard_bool = keyboard_mask > 0
            selection_mask_bool = keyboard_bool.copy()

            # Build a small keyboard-edge exclusion band (left/right + top),
            # because those borders often behave like vent-adjacent structure.
            edge_exclusion = np.zeros_like(selection_mask_bool, dtype=bool)
            ys, xs = np.where(keyboard_bool)
            if xs.size > 0 and ys.size > 0:
                kx0, kx1 = int(np.min(xs)), int(np.max(xs))
                ky0, ky1 = int(np.min(ys)), int(np.max(ys))
                k_w = max(1, kx1 - kx0 + 1)
                k_h = max(1, ky1 - ky0 + 1)
                side_band = max(2, int(round(k_w * 0.08)))
                top_band = max(2, int(round(k_h * 0.10)))

                edge_exclusion[ky0:ky1 + 1, kx0:min(kx1 + 1, kx0 + side_band)] = True
                edge_exclusion[ky0:ky1 + 1, max(kx0, kx1 - side_band + 1):kx1 + 1] = True
                edge_exclusion[ky0:min(ky1 + 1, ky0 + top_band), kx0:kx1 + 1] = True

            selection_mask_bool &= ~edge_exclusion

            # Exclude structural cooling regions from keyboard hotspot GT.
            vent_mask = region_masks.get("vent")
            if vent_mask is not None and np.any(vent_mask > 0):
                vent_bool = vent_mask > 0
                vent_bool = cv2.dilate(vent_bool.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1) > 0
                selection_mask_bool &= ~vent_bool

            hinge_mask = region_masks.get("hinge")
            if hinge_mask is not None and np.any(hinge_mask > 0):
                hinge_bool = hinge_mask > 0
                hinge_bool = cv2.dilate(hinge_bool.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1) > 0
                selection_mask_bool &= ~hinge_bool

            if not np.any(selection_mask_bool):
                selection_mask_bool = keyboard_bool.copy()

                if vent_mask is not None and np.any(vent_mask > 0):
                    vent_bool = vent_mask > 0
                    vent_bool = cv2.dilate(vent_bool.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1) > 0
                    selection_mask_bool &= ~vent_bool
                if hinge_mask is not None and np.any(hinge_mask > 0):
                    hinge_bool = hinge_mask > 0
                    hinge_bool = cv2.dilate(hinge_bool.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1) > 0
                    selection_mask_bool &= ~hinge_bool

            if not np.any(selection_mask_bool):
                selection_mask_bool = keyboard_bool

            hot_y, hot_x = self._argmax_with_mask(temp, selection_mask_bool.astype(np.uint8))
        elif target_area is not None:
            # Keep user-directed target-area behavior as fallback when keyboard mask is unavailable.
            hot_y, hot_x = self._argmax_with_mask(temp, cpu_mask)
        else:
            hot_y, hot_x = self._argmax_with_mask(temp, cpu_mask)
        hotspot_temp = float(temp[hot_y, hot_x])
        cpu_center = self._mask_center(cpu_mask)
        gpu_enabled = bool(sample.metadata.get("gpu_enabled", False))
        gpu_center = sample.metadata.get("gpu_center", [-1.0, -1.0])

        centers = [cpu_center]
        temperatures = [float(sample.metadata.get("cpu_temperature", hotspot_temp))]
        if gpu_enabled:
            centers.append((float(gpu_center[0]), float(gpu_center[1])))
            temperatures.append(float(sample.metadata.get("gpu_temperature", hotspot_temp - 4.0)))

        workload_name = "post_stress"

        return ThermalFrame(
            image=temp.astype(np.float32),
            mask=(cpu_mask > 0).astype(np.uint8) * 255,
            centers=centers,
            temperatures=temperatures,
            width=self.width,
            height=self.height,
            thermal_image=thermal_rgb.astype(np.uint8),
            region_masks=region_masks,
            hotspot_coordinate=(float(hot_x), float(hot_y)),
            hotspot_temperature=hotspot_temp,
            hottest_region_category=str(sample.metadata.get("hottest_region_category", "genuine_source")),
            excluded_hottest_region=sample.metadata.get("excluded_hottest_region"),
            target_source_attribution_fraction=float(sample.metadata.get("target_source_attribution_fraction"))
            if sample.metadata.get("target_source_attribution_fraction") is not None
            else None,
            workload=workload_name,
            power_class=str(sample.metadata.get("power_class", "unknown")),
            keyboard_plateau_coverage=float(
                sample.metadata.get(
                    "keyboard_plateau_effective_coverage",
                    sample.metadata.get("keyboard_plateau_coverage", 0.0),
                )
            ),
            dgpu_enabled=bool(gpu_enabled),
        )

    def _generate_threshold_passed_sample(self):
        """Generate samples until physics score passes the configured threshold."""
        cfg = self._dataset_generator.config
        cfg.noise_level = max(0.2, float(self.noise_std) * 0.55)
        cfg.physics_score_threshold = float(self.physics_score_threshold)
        cfg.physics_max_attempts = max(8, int(self.strict_max_attempts))

        if self.power_class_override == "thin":
            cfg.power_class_weights = (1.0, 0.0, 0.0)
        elif self.power_class_override == "mainstream":
            cfg.power_class_weights = (0.0, 1.0, 0.0)
        elif self.power_class_override == "gaming":
            cfg.power_class_weights = (0.0, 0.0, 1.0)
        else:
            cfg.power_class_weights = tuple(self._default_power_class_weights)

        if self.force_gpu_mode == "off":
            cfg.enable_gpu = False
            cfg.thin_gpu_probability = 0.0
            cfg.mainstream_gpu_probability = 0.0
            cfg.gaming_gpu_probability = 0.0
            if self.power_class_override is None:
                cfg.power_class_weights = (0.70, 0.30, 0.0)
        elif self.force_gpu_mode == "on":
            cfg.enable_gpu = True
            cfg.thin_gpu_probability = 1.0
            cfg.mainstream_gpu_probability = 1.0
            cfg.gaming_gpu_probability = 1.0
            if self.power_class_override is None:
                cfg.power_class_weights = (0.0, 0.35, 0.65)
            cfg.gpu_temp_range = (34.0, 60.0)
        else:
            cfg.enable_gpu = bool(self._default_enable_gpu)
            cfg.thin_gpu_probability = float(self._default_gpu_probs[0])
            cfg.mainstream_gpu_probability = float(self._default_gpu_probs[1])
            cfg.gaming_gpu_probability = float(self._default_gpu_probs[2])
            cfg.gpu_temp_range = (30.0, 60.0)

        best_score = -1.0

        for _ in range(self.strict_max_attempts):
            sample_idx = int(self.rng.integers(1, 1_000_000))
            candidate = self._dataset_generator.generate_sample(index=sample_idx, seed_offset=sample_idx)
            score = float(candidate.metadata.get("physics_score", 0.0))
            passed = bool(candidate.metadata.get("physics_gate_passed", score >= self.physics_score_threshold))

            if score > best_score:
                best_score = score

            if passed and score >= self.physics_score_threshold:
                return candidate

        raise RuntimeError(
            f"Unable to generate frame with physics_score >= {self.physics_score_threshold:.3f} "
            f"within {self.strict_max_attempts} attempts. Best score: {best_score:.3f}"
        )

    def generate_batch(
        self,
        sample_count: int,
        workloads: Optional[Iterable[str]] = None,
    ) -> list[ThermalFrame]:
        """Generate a batch of ThermalFrame values using dataset backend."""
        if sample_count <= 0:
            return []
        workload_list = list(workloads) if workloads else [None]
        frames: list[ThermalFrame] = []
        for _ in range(sample_count):
            selected_workload = str(self.rng.choice(workload_list)) if workloads else None
            frames.append(self.generate(workload=selected_workload))
        return frames

    def set_seed(self, seed: int) -> None:
        """Reset random seed for adapter generation."""
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _select_base_size(size: int) -> int:
        """Select closest supported square dataset size for backend generation."""
        candidates = np.array([320, 640, 1024], dtype=np.int32)
        idx = int(np.argmin(np.abs(candidates - int(size))))
        return int(candidates[idx])

    @staticmethod
    def _mask_center(mask: np.ndarray) -> Tuple[float, float]:
        """Compute centroid of non-zero mask pixels."""
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            h, w = mask.shape
            return float(w) * 0.5, float(h) * 0.5
        return float(np.mean(xs)), float(np.mean(ys))

    @staticmethod
    def _argmax_with_mask(temp: np.ndarray, mask: Optional[np.ndarray]) -> tuple[int, int]:
        """Return hottest point (y, x), preferring points inside mask when available."""
        if mask is not None:
            valid = mask > 0
            if np.any(valid):
                masked = np.where(valid, temp, -np.inf)
                idx = int(np.argmax(masked))
                y, x = np.unravel_index(idx, temp.shape)
                if np.isfinite(float(masked[y, x])):
                    return int(y), int(x)
        y, x = np.unravel_index(int(np.argmax(temp)), temp.shape)
        return int(y), int(x)

    def _normalize_rect(self, rect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Clamp and normalize rectangle to image bounds, returning [x0, y0, x1, y1)."""
        x0, y0, x1, y1 = [int(v) for v in rect]
        left = int(np.clip(min(x0, x1), 0, self.width - 1))
        right = int(np.clip(max(x0, x1), 0, self.width - 1))
        top = int(np.clip(min(y0, y1), 0, self.height - 1))
        bottom = int(np.clip(max(y0, y1), 0, self.height - 1))
        return left, top, right + 1, bottom + 1

    @staticmethod
    def _workload_from_hotspot_count(hotspot_count: int) -> str:
        """Map legacy hotspot count slider values into workload labels."""
        map_by_count = {
            1: "light",
            2: "medium",
            3: "heavy",
            4: "cpu_stress",
            5: "dual_stress",
        }
        return map_by_count.get(int(np.clip(hotspot_count, 1, 5)), "medium")
