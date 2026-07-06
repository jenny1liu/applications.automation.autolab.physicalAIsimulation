"""Component layout primitives and random samplers for thermal scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .config import GeneratorConfig

CpuShape = Literal["circle", "ellipse", "rotated_ellipse"]


@dataclass(slots=True)
class ComponentSpec:
    """Describes one thermal component instance in image coordinates."""

    center: tuple[float, float]
    size: tuple[float, float]
    angle: float
    temperature: float
    enabled: bool


@dataclass(slots=True)
class HeatPipeSpec:
    """Describes a heat pipe as a line source with physical attributes."""

    start: tuple[float, float]
    end: tuple[float, float]
    width: float
    conductivity: float
    angle: float
    length: float
    enabled: bool


@dataclass(slots=True)
class VentSpec:
    """Describes a narrow chassis-edge exhaust strip."""

    start: tuple[float, float]
    end: tuple[float, float]
    width: float
    length: float
    edge: str
    temperature: float
    enabled: bool


@dataclass(slots=True)
class HingeSpec:
    """Describes hinge boundary line between screen and keyboard deck."""

    start: tuple[float, float]
    end: tuple[float, float]
    width: float
    enabled: bool


@dataclass(slots=True)
class PowerClassProfile:
    """Correlated sampled operating class driving thermal envelope."""

    name: str
    keyboard_plateau_coverage: float
    target_max_surface_temp: float
    gpu_present: bool


@dataclass(slots=True)
class SceneLayout:
    """Container for all thermal component specifications."""

    cpu: ComponentSpec
    cpu_shape: CpuShape
    cpu_power: int
    power_class: PowerClassProfile
    gpu: ComponentSpec
    heatpipe: HeatPipeSpec
    fan: ComponentSpec
    vents: list[VentSpec]
    battery: ComponentSpec
    ssd: ComponentSpec
    vrm: ComponentSpec
    palm_rest: ComponentSpec
    keyboard_area: ComponentSpec
    screen_area: ComponentSpec
    hinge: HingeSpec
    screen_visible: bool
    camera_tilt_deg: float
    screen_region_fraction: float
    framing_fill_fraction: float
    keyboard_layout: str


class ComponentSampler:
    """Samples randomized laptop component parameters for one scene."""

    def __init__(self, config: GeneratorConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self._last_hinge_y: float | None = None

    def sample_layout(self) -> SceneLayout:
        """Sample a randomized scene layout with all component specs."""
        size = self.config.image_size
        power_class = self._sample_power_class()

        camera_tilt = float(self.rng.uniform(*self.config.camera_tilt_deg_range))
        screen_prob = float(self.rng.uniform(*self.config.screen_visible_probability_range))
        screen_visible = bool(self.config.enable_screen_region and (self.rng.random() < screen_prob))

        if screen_visible:
            tilt_lo, tilt_hi = self.config.camera_tilt_deg_range
            tilt_span = max(1e-6, float(tilt_hi - tilt_lo))
            tilt_norm = float(np.clip((camera_tilt - tilt_lo) / tilt_span, 0.0, 1.0))
            frac_hi = float(self.config.screen_region_fraction_range[1])
            frac_lo = float(self.config.screen_region_fraction_range[0])
            # Smaller tilt angle (more frontal) reveals more screen area.
            base_frac = frac_hi - (frac_hi - frac_lo) * tilt_norm
            jitter = float(self.rng.uniform(-0.02, 0.02))
            screen_fraction = float(np.clip(base_frac + jitter, frac_lo, frac_hi))
        else:
            screen_fraction = 0.0

        framing_fill_fraction = float(self.rng.uniform(*self.config.framing_fill_fraction_range))
        framing_fill_fraction = float(np.clip(framing_fill_fraction, 0.60, 0.98))

        hinge_width = int(self.rng.integers(self.config.hinge_width_px_range[0], self.config.hinge_width_px_range[1] + 1))
        screen_h = int(round(size * screen_fraction)) if screen_visible else 0
        screen_h = int(np.clip(screen_h, 0, int(size * 0.45)))
        hinge_y0 = int(np.clip(screen_h, 0, size - 1))
        hinge_y1 = int(np.clip(hinge_y0 + hinge_width, 0, size))

        # Keep hinge span within visible laptop body width (avoid full-frame line).
        body_margin = int(round(size * (1.0 - framing_fill_fraction) * 0.5))
        body_margin = int(np.clip(body_margin, round(size * 0.03), round(size * 0.18)))
        body_x0 = int(np.clip(body_margin, 0, size - 2))
        body_x1 = int(np.clip(size - 1 - body_margin, body_x0 + 1, size - 1))

        if screen_visible:
            keyboard_y0 = int(np.clip(hinge_y1 + int(size * 0.01), int(size * 0.16), int(size * 0.42)))
            keyboard_y1 = int(np.clip(size * 0.78, keyboard_y0 + int(size * 0.28), int(size * 0.88)))
        else:
            keyboard_y0 = int(size * 0.20)
            keyboard_y1 = int(size * 0.76)

        numpad_prob = self._numpad_probability_for_class(power_class.name)
        keyboard_layout = "numpad" if self.rng.random() < numpad_prob else "no_numpad"
        keyboard_cx = size * (0.54 if keyboard_layout == "numpad" else 0.50)
        keyboard_w = size * (0.90 if keyboard_layout == "numpad" else 0.84)

        keyboard_area = ComponentSpec(
            center=(float(keyboard_cx), float((keyboard_y0 + keyboard_y1) * 0.5)),
            size=(float(keyboard_w), float(max(8, keyboard_y1 - keyboard_y0))),
            angle=0.0,
            temperature=float(self.rng.uniform(*self.config.ambient_temp_range)),
            enabled=self.config.enable_keyboard_area,
        )

        palm_top = int(np.clip(keyboard_y1 + int(size * 0.015), int(size * 0.71), int(size * 0.92)))
        palm_bottom = size
        palm_rest = ComponentSpec(
            center=(size * 0.5, float((palm_top + palm_bottom) * 0.5)),
            size=(size * 0.9, float(max(8, palm_bottom - palm_top))),
            angle=0.0,
            temperature=float(self.rng.uniform(*self.config.ambient_temp_range)),
            enabled=self.config.enable_palm_rest,
        )

        screen_area = ComponentSpec(
            center=(float((body_x0 + body_x1) * 0.5), float(max(1, screen_h) * 0.5)),
            size=(float(max(1, body_x1 - body_x0)), float(max(1, screen_h))),
            angle=0.0,
            temperature=float(self.rng.uniform(*self.config.ambient_temp_range)),
            enabled=screen_visible,
        )

        hinge = HingeSpec(
            start=(float(body_x0), float(hinge_y0)),
            end=(float(body_x1), float(hinge_y0)),
            width=float(max(1, hinge_width)),
            enabled=bool(self.config.enable_hinge and screen_visible),
        )
        self._last_hinge_y = float(hinge_y0) if hinge.enabled else None

        cpu_center = (
            size * self.rng.uniform(0.35, 0.65),
            float(self.rng.uniform(keyboard_y0 + size * 0.02, keyboard_y0 + (keyboard_y1 - keyboard_y0) * 0.55)),
        )
        cpu_w = float(self.rng.integers(*self.config.cpu_size_range))
        cpu_h = float(self.rng.integers(*self.config.cpu_size_range))
        cpu_shape: CpuShape = self.rng.choice(["circle", "ellipse", "rotated_ellipse"]).item()
        cpu_angle = float(self.rng.uniform(-45.0, 45.0))
        cpu_power = self._sample_cpu_power(power_class.name)
        cpu_temp_range = self._cpu_temp_range_for_class(power_class.name)
        cpu = ComponentSpec(
            center=cpu_center,
            size=(cpu_w, cpu_h),
            angle=cpu_angle,
            temperature=float(self.rng.uniform(*cpu_temp_range)),
            enabled=self.config.enable_cpu,
        )

        gpu_enabled = self.config.enable_gpu and power_class.gpu_present
        gpu_temp_range = self._gpu_temp_range_for_class(power_class.name)
        gpu = ComponentSpec(
            center=(
                size * self.rng.uniform(0.18, 0.82),
                float(self.rng.uniform(keyboard_y0 + size * 0.01, keyboard_y0 + (keyboard_y1 - keyboard_y0) * 0.58)),
            ),
            size=(
                float(self.rng.integers(*self.config.gpu_size_range)),
                float(self.rng.integers(*self.config.gpu_size_range)),
            ),
            angle=float(self.rng.uniform(-35.0, 35.0)),
            temperature=float(self.rng.uniform(*gpu_temp_range)),
            enabled=gpu_enabled,
        )

        # Fan placement is correlated with CPU side to keep a plausible airflow route.
        fan_band_y0 = max(0.06 * size, keyboard_y0 - 0.10 * size)
        fan_band_y1 = min(keyboard_y0 + 0.06 * size, keyboard_y0 + 0.14 * size)
        cpu_right_side = bool(cpu.center[0] >= size * 0.5)
        if cpu_right_side:
            fan_x = size * self.rng.uniform(0.68, 0.92)
        else:
            fan_x = size * self.rng.uniform(0.08, 0.32)
        fan_center = (float(fan_x), float(self.rng.uniform(fan_band_y0, fan_band_y1)))
        fan_radius = float(self.rng.integers(*self.config.fan_radius_range))
        fan = ComponentSpec(
            center=fan_center,
            size=(fan_radius, fan_radius),
            angle=0.0,
            temperature=float(self.rng.uniform(*self.config.fan_surface_temp_range)),
            enabled=self.config.enable_fan,
        )

        vents = self._sample_vents(size, power_class.name, fan_center=fan_center)

        hp_length_target = float(self.rng.integers(*self.config.heatpipe_length_range))
        hp_width = float(self.rng.integers(*self.config.heatpipe_width_range))
        hp_start = np.array(cpu.center, dtype=np.float32)

        # Heatpipe should connect source side (CPU/GPU) toward cooling side (fan/vent).
        if vents:
            vent_mids = [
                ((float(v.start[0]) + float(v.end[0])) * 0.5, (float(v.start[1]) + float(v.end[1])) * 0.5)
                for v in vents if v.enabled
            ]
        else:
            vent_mids = []

        if vent_mids:
            nearest_vent_mid = min(
                vent_mids,
                key=lambda p: float(np.hypot(p[0] - fan_center[0], p[1] - fan_center[1])),
            )
            outlet_target = np.array(
                [
                    fan_center[0] * 0.62 + nearest_vent_mid[0] * 0.38,
                    fan_center[1] * 0.62 + nearest_vent_mid[1] * 0.38,
                ],
                dtype=np.float32,
            )
        else:
            outlet_target = np.array(fan_center, dtype=np.float32)

        direction = outlet_target - hp_start
        dir_norm = float(np.linalg.norm(direction))
        if dir_norm < 1e-6:
            fallback_angle = float(self.rng.uniform(-40.0, 40.0))
            direction = np.array([np.cos(np.deg2rad(fallback_angle)), np.sin(np.deg2rad(fallback_angle))], dtype=np.float32)
            dir_norm = 1.0
        direction = direction / max(1e-6, dir_norm)

        dist_to_target = float(np.linalg.norm(outlet_target - hp_start))
        hp_length = float(np.clip(dist_to_target * self.rng.uniform(0.90, 1.12), *self.config.heatpipe_length_range))
        hp_end = hp_start + direction * hp_length
        hp_end[0] = np.clip(hp_end[0], 0, size - 1)
        hp_end[1] = np.clip(hp_end[1], 0, size - 1)
        hp_angle = float(np.rad2deg(np.arctan2(float(hp_end[1] - hp_start[1]), float(hp_end[0] - hp_start[0]))))

        # If clipping shortened pipe too much, re-extend to preserve non-trivial conduction path.
        effective_len = float(np.linalg.norm(hp_end - hp_start))
        if effective_len < max(40.0, hp_length_target * 0.45):
            direction = direction / max(1e-6, float(np.linalg.norm(direction)))
            hp_end = hp_start + direction * float(max(40.0, hp_length_target * 0.60))
            hp_end[0] = np.clip(hp_end[0], 0, size - 1)
            hp_end[1] = np.clip(hp_end[1], 0, size - 1)
            hp_angle = float(np.rad2deg(np.arctan2(float(hp_end[1] - hp_start[1]), float(hp_end[0] - hp_start[0]))))

        heatpipe = HeatPipeSpec(
            start=(float(hp_start[0]), float(hp_start[1])),
            end=(float(hp_end[0]), float(hp_end[1])),
            width=hp_width,
            conductivity=float(self.rng.uniform(*self.config.heatpipe_conductivity_range)),
            angle=hp_angle,
            length=float(np.linalg.norm(hp_end - hp_start)),
            enabled=self.config.enable_heatpipe,
        )

        battery = ComponentSpec(
            center=(size * 0.50, float(self.rng.uniform(max(0.82 * size, palm_top), min(0.95 * size, size - 1)))),
            size=(size * 0.66, size * 0.10),
            angle=0.0,
            temperature=float(self.rng.uniform(*self.config.ambient_temp_range)),
            enabled=self.config.enable_battery,
        )
        ssd = ComponentSpec(
            center=(size * self.rng.uniform(0.10, 0.22), size * self.rng.uniform(0.52, 0.78)),
            size=(size * 0.14, size * 0.05),
            angle=float(self.rng.uniform(-20.0, 20.0)),
            temperature=float(self.rng.uniform(*self.config.ambient_temp_range)) + 3.0,
            enabled=self.config.enable_ssd,
        )
        vrm = ComponentSpec(
            center=(size * self.rng.uniform(0.22, 0.40), size * self.rng.uniform(0.18, 0.36)),
            size=(size * 0.16, size * 0.05),
            angle=float(self.rng.uniform(-30.0, 30.0)),
            temperature=float(self.rng.uniform(*self.config.ambient_temp_range)) + 4.0,
            enabled=self.config.enable_vrm,
        )
        return SceneLayout(
            cpu=cpu,
            cpu_shape=cpu_shape,
            cpu_power=cpu_power,
            power_class=power_class,
            gpu=gpu,
            heatpipe=heatpipe,
            fan=fan,
            vents=vents,
            battery=battery,
            ssd=ssd,
            vrm=vrm,
            palm_rest=palm_rest,
            keyboard_area=keyboard_area,
            screen_area=screen_area,
            hinge=hinge,
            screen_visible=screen_visible,
            camera_tilt_deg=float(camera_tilt),
            screen_region_fraction=float(screen_fraction),
            framing_fill_fraction=float(framing_fill_fraction),
            keyboard_layout=keyboard_layout,
        )

    def _numpad_probability_for_class(self, power_class: str) -> float:
        """Return numpad layout probability for selected power class."""
        if power_class == "thin":
            return float(self.config.thin_numpad_probability)
        if power_class == "mainstream":
            return float(self.config.mainstream_numpad_probability)
        return float(self.config.gaming_numpad_probability)

    def _sample_power_class(self) -> PowerClassProfile:
        """Sample correlated laptop power class controlling thermal coverage and maxima."""
        weights = np.array(self.config.power_class_weights, dtype=np.float32)
        weights = weights / np.sum(weights)
        class_name = str(self.rng.choice(self.config.power_class_names, p=weights).item())

        if class_name == "thin":
            coverage = float(self.rng.uniform(*self.config.thin_plateau_coverage_range))
            max_temp = float(self.rng.uniform(*self.config.thin_max_temp_range))
            gpu_present = bool(self.rng.random() < self.config.thin_gpu_probability)
        elif class_name == "mainstream":
            coverage = float(self.rng.uniform(*self.config.mainstream_plateau_coverage_range))
            max_temp = float(self.rng.uniform(*self.config.mainstream_max_temp_range))
            gpu_present = bool(self.rng.random() < self.config.mainstream_gpu_probability)
        else:
            coverage = float(self.rng.uniform(*self.config.gaming_plateau_coverage_range))
            max_temp = float(self.rng.uniform(*self.config.gaming_max_temp_range))
            gpu_present = bool(self.rng.random() < self.config.gaming_gpu_probability)

        return PowerClassProfile(
            name=class_name,
            keyboard_plateau_coverage=coverage,
            target_max_surface_temp=max_temp,
            gpu_present=gpu_present,
        )

    def _sample_cpu_power(self, power_class: str) -> int:
        """Sample CPU power values correlated with power class."""
        if power_class == "thin":
            return int(self.rng.choice(self.config.thin_cpu_power_values).item())
        if power_class == "mainstream":
            return int(self.rng.choice(self.config.mainstream_cpu_power_values).item())
        return int(self.rng.choice(self.config.gaming_cpu_power_values).item())

    def _sample_vents(self, size: int, power_class: str, fan_center: tuple[float, float]) -> list[VentSpec]:
        """Sample one or two vents correlated with fan side for plausible exhaust flow."""
        if not self.config.enable_vent:
            return []

        vent_temp_range = self._vent_temp_range_for_class(power_class)

        count = int(self.rng.integers(self.config.vent_count_range[0], self.config.vent_count_range[1] + 1))
        vents: list[VentSpec] = []

        fan_right_side = bool(fan_center[0] >= size * 0.5)
        vent_width_frac_range = self._vent_width_fraction_range_for_class(power_class)

        if count == 2 and self.rng.random() < 0.65:
            y = float(size * self.rng.uniform(0.02, 0.08))
            if hasattr(self, "_last_hinge_y") and self._last_hinge_y is not None:
                y = float(np.clip(self._last_hinge_y, 0, size - 1))
            target_frac = float(self.rng.uniform(*vent_width_frac_range))
            target_len = float(np.clip(size * target_frac, *self.config.vent_length_range))
            length = float(target_len)
            width = float(self.rng.integers(*self.config.vent_width_range))
            if power_class == "gaming":
                width = float(max(2.0, round(width * 0.75)))
            # Keep one vent near fan side and optionally another opposite for dual-exhaust designs.
            if fan_right_side:
                right_x1 = float(np.clip(fan_center[0] + self.rng.uniform(size * 0.05, size * 0.18), size * 0.70, size * 0.96))
                left_x0 = float(size * self.rng.uniform(0.05, 0.20))
            else:
                left_x0 = float(np.clip(fan_center[0] - self.rng.uniform(size * 0.05, size * 0.18), size * 0.04, size * 0.30))
                right_x1 = float(size * self.rng.uniform(0.80, 0.95))
            left_x1 = float(np.clip(left_x0 + length, 0, size - 1))
            right_x0 = float(np.clip(right_x1 - length, 0, size - 1))
            temp_left = float(self.rng.uniform(*vent_temp_range))
            temp_right = float(self.rng.uniform(*vent_temp_range))
            vents.append(
                VentSpec(
                    start=(left_x0, y),
                    end=(left_x1, y),
                    width=width,
                    length=length,
                    edge="top_left",
                    temperature=temp_left,
                    enabled=True,
                )
            )
            vents.append(
                VentSpec(
                    start=(right_x0, y),
                    end=(right_x1, y),
                    width=width,
                    length=length,
                    edge="top_right",
                    temperature=temp_right,
                    enabled=True,
                )
            )
            return vents

        for _ in range(count):
            base_len = float(self.rng.integers(*self.config.vent_length_range))
            width = float(self.rng.integers(*self.config.vent_width_range))
            if self.rng.random() < 0.70:
                edge = "top"
            else:
                edge = "right" if fan_right_side else "left"

            if edge == "top":
                target_frac = float(self.rng.uniform(*vent_width_frac_range))
                length = float(np.clip(size * target_frac, *self.config.vent_length_range))
                y = float(size * self.rng.uniform(0.02, 0.08))
                if hasattr(self, "_last_hinge_y") and self._last_hinge_y is not None:
                    y = float(np.clip(self._last_hinge_y, 0, size - 1))
                x0 = float(np.clip(fan_center[0] - self.rng.uniform(size * 0.05, size * 0.18), 0, size - 1))
                x1 = float(np.clip(x0 + length, 0, size - 1))
                start = (x0, y)
                end = (x1, y)
                if power_class == "gaming":
                    width = float(max(2.0, round(width * 0.75)))
            elif edge == "left":
                length = base_len
                x = float(size * self.rng.uniform(0.02, 0.08))
                y0 = float(np.clip(fan_center[1] - self.rng.uniform(size * 0.03, size * 0.14), size * 0.10, size * 0.76))
                y1 = float(np.clip(y0 + length, 0, size - 1))
                start = (x, y0)
                end = (x, y1)
            else:
                length = base_len
                x = float(size * self.rng.uniform(0.92, 0.98))
                y0 = float(np.clip(fan_center[1] - self.rng.uniform(size * 0.03, size * 0.14), size * 0.10, size * 0.76))
                y1 = float(np.clip(y0 + length, 0, size - 1))
                start = (x, y0)
                end = (x, y1)

            vents.append(
                VentSpec(
                    start=start,
                    end=end,
                    width=width,
                    length=length,
                    edge=edge,
                    temperature=float(self.rng.uniform(*vent_temp_range)),
                    enabled=True,
                )
            )

        return vents

    def _vent_width_fraction_range_for_class(self, power_class: str) -> tuple[float, float]:
        """Return top-vent width fraction range for selected power class."""
        if power_class == "thin":
            return self.config.thin_vent_width_fraction_range
        if power_class == "mainstream":
            return self.config.mainstream_vent_width_fraction_range
        return self.config.gaming_vent_width_fraction_range

    def _cpu_temp_range_for_class(self, power_class: str) -> tuple[float, float]:
        """Return stressed CPU temperature range for selected power class."""
        if power_class == "thin":
            return self.config.thin_cpu_temp_range
        if power_class == "mainstream":
            return self.config.mainstream_cpu_temp_range
        return self.config.gaming_cpu_temp_range

    def _gpu_temp_range_for_class(self, power_class: str) -> tuple[float, float]:
        """Return stressed GPU temperature range for selected power class."""
        if power_class == "thin":
            return self.config.thin_gpu_temp_range
        if power_class == "mainstream":
            return self.config.mainstream_gpu_temp_range
        return self.config.gaming_gpu_temp_range

    def _vent_temp_range_for_class(self, power_class: str) -> tuple[float, float]:
        """Return stressed vent temperature range for selected power class."""
        if power_class == "thin":
            return self.config.thin_vent_temp_range
        if power_class == "mainstream":
            return self.config.mainstream_vent_temp_range
        return self.config.gaming_vent_temp_range
