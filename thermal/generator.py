from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np


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
    workload: str = "medium"


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

        thermal_rgb = self.render_flir(temp)

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

    def render_flir(self, temperature_matrix: np.ndarray) -> np.ndarray:
        """Render a FLIR-like RGB thermal image from raw temperature matrix."""
        low = float(np.percentile(temperature_matrix, 2.0))
        high = float(np.percentile(temperature_matrix, 99.2))
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
