"""Configuration definitions for thermal dataset generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GeneratorConfig:
    """Configuration object controlling synthetic thermal data generation."""

    image_size: int = 640
    # Optional final output resolution (width, height) to match real thermal sensor aspect ratio.
    # When None, outputs keep square simulation size (image_size x image_size).
    sensor_output_size: tuple[int, int] | None = None
    dataset_size: int = 1000
    output_dir: Path = Path("output")
    random_seed: int = 42
    generator_version: str = "v2.0.0"
    workers: int = 0
    debug_mode: bool = False

    enable_cpu: bool = True
    enable_gpu: bool = True
    enable_heatpipe: bool = True
    enable_fan: bool = True
    enable_vent: bool = True
    enable_battery: bool = True
    enable_ssd: bool = True
    enable_vrm: bool = True
    enable_palm_rest: bool = True
    enable_touchpad: bool = True
    enable_keyboard_area: bool = True
    enable_screen_region: bool = True
    enable_hinge: bool = True

    cpu_temp_range: tuple[float, float] = (32.0, 63.0)
    cpu_power_values: tuple[int, ...] = (15, 28, 35, 45, 65)
    gpu_temp_range: tuple[float, float] = (30.0, 60.0)
    fan_surface_temp_range: tuple[float, float] = (28.0, 48.0)
    vent_surface_temp_range: tuple[float, float] = (28.0, 48.0)
    fan_speed_values: tuple[int, ...] = (0, 30, 60, 100)
    ambient_temp_range: tuple[float, float] = (22.0, 30.0)

    # Deployment mode: robot scans only after stress workload completes.
    deployment_thermal_state: str = "post_stress"

    # Per-power-class stressed operating bands used as primary sampling axis.
    thin_cpu_temp_range: tuple[float, float] = (40.0, 50.0)
    mainstream_cpu_temp_range: tuple[float, float] = (48.0, 56.0)
    gaming_cpu_temp_range: tuple[float, float] = (55.0, 63.0)

    thin_gpu_temp_range: tuple[float, float] = (38.0, 48.0)
    mainstream_gpu_temp_range: tuple[float, float] = (46.0, 55.0)
    gaming_gpu_temp_range: tuple[float, float] = (53.0, 62.0)

    thin_vent_temp_range: tuple[float, float] = (36.0, 48.0)
    mainstream_vent_temp_range: tuple[float, float] = (44.0, 54.0)
    gaming_vent_temp_range: tuple[float, float] = (50.0, 61.0)

    thin_trap_temp_range: tuple[float, float] = (34.0, 46.0)
    mainstream_trap_temp_range: tuple[float, float] = (42.0, 52.0)
    gaming_trap_temp_range: tuple[float, float] = (48.0, 59.0)

    power_class_names: tuple[str, ...] = ("thin", "mainstream", "gaming")
    power_class_weights: tuple[float, float, float] = (0.35, 0.40, 0.25)
    thin_cpu_power_values: tuple[int, ...] = (15, 28)
    mainstream_cpu_power_values: tuple[int, ...] = (35, 45)
    gaming_cpu_power_values: tuple[int, ...] = (45, 65)
    thin_plateau_coverage_range: tuple[float, float] = (0.30, 0.50)
    mainstream_plateau_coverage_range: tuple[float, float] = (0.50, 0.70)
    gaming_plateau_coverage_range: tuple[float, float] = (0.70, 1.00)
    thin_max_temp_range: tuple[float, float] = (34.0, 50.0)
    mainstream_max_temp_range: tuple[float, float] = (42.0, 55.0)
    gaming_max_temp_range: tuple[float, float] = (50.0, 63.0)
    thin_gpu_probability: float = 0.20
    mainstream_gpu_probability: float = 0.65
    gaming_gpu_probability: float = 1.00

    # Keyboard layout diversity by power class.
    thin_numpad_probability: float = 0.10
    mainstream_numpad_probability: float = 0.45
    gaming_numpad_probability: float = 0.75

    # Vent horizontal coverage fraction by power class for top-edge vents.
    thin_vent_width_fraction_range: tuple[float, float] = (0.24, 0.45)
    mainstream_vent_width_fraction_range: tuple[float, float] = (0.35, 0.65)
    gaming_vent_width_fraction_range: tuple[float, float] = (0.60, 1.00)

    # Open-lid framing fill sampled independently from camera tilt.
    framing_fill_fraction_range: tuple[float, float] = (0.84, 0.94)

    # Seen-vs-held-out split export.
    export_seen_heldout_split: bool = True
    held_out_power_classes: tuple[str, ...] = ("gaming",)

    cpu_size_range: tuple[int, int] = (26, 74)
    gpu_size_range: tuple[int, int] = (20, 62)
    heatpipe_length_range: tuple[int, int] = (120, 280)
    heatpipe_width_range: tuple[int, int] = (8, 22)
    heatpipe_conductivity_range: tuple[float, float] = (0.5, 1.0)
    fan_radius_range: tuple[int, int] = (20, 60)
    vent_length_range: tuple[int, int] = (60, 180)
    vent_width_range: tuple[int, int] = (4, 12)
    vent_count_range: tuple[int, int] = (1, 2)

    keyboard_plateau_taper: float = 0.35
    keyboard_plateau_strength_range: tuple[float, float] = (2.8, 5.0)
    touchpad_cool_strength_range: tuple[float, float] = (2.5, 5.5)
    touchpad_size_ratio: tuple[float, float] = (0.30, 0.17)
    touchpad_center_y_ratio_range: tuple[float, float] = (0.82, 0.88)
    touchpad_post_cool_range: tuple[float, float] = (0.4, 1.4)
    touchpad_uniform_blend: float = 0.50
    touchpad_border_cool: float = 0.25
    enable_key_grid_texture: bool = True
    key_grid_rows_range: tuple[int, int] = (5, 7)
    key_grid_cols_range: tuple[int, int] = (15, 19)
    key_grid_blend_range: tuple[float, float] = (0.42, 0.62)
    key_grid_gap_cool_range: tuple[float, float] = (1.2, 2.8)
    key_grid_gap_px_range: tuple[int, int] = (1, 2)
    key_grid_light_blur_sigma: float = 0.35

    screen_visible_probability_range: tuple[float, float] = (0.40, 0.60)
    camera_tilt_deg_range: tuple[float, float] = (18.0, 55.0)
    screen_region_fraction_range: tuple[float, float] = (0.10, 0.30)
    hinge_width_px_range: tuple[int, int] = (2, 5)
    hinge_heating_probability: float = 0.65
    hinge_heat_boost_range: tuple[float, float] = (0.25, 0.85)
    screen_texture_noise_std_range: tuple[float, float] = (0.08, 0.32)
    palm_rest_warm_offset_range: tuple[float, float] = (3.0, 6.0)
    touchpad_target_offset_range: tuple[float, float] = (1.2, 2.8)

    enable_trap_zone: bool = True
    trap_zone_probability: float = 0.42
    trap_zone_hotter_than_source_probability: float = 0.22
    trap_zone_sigma_x_ratio_range: tuple[float, float] = (0.17, 0.30)
    trap_zone_sigma_y_ratio_range: tuple[float, float] = (0.12, 0.24)
    trap_zone_rise_range: tuple[float, float] = (1.2, 3.6)
    trap_zone_hot_rise_range: tuple[float, float] = (4.8, 8.4)
    distractor_heat_source_probability: float = 0.20
    occlusion_probability: float = 0.0

    source_attribution_threshold: float = 0.60
    target_point_blur_sigma: float = 0.85

    use_fixed_display_temp_range: bool = True
    display_temp_range: tuple[float, float] = (22.0, 52.0)
    global_max_surface_temp: float = 65.0
    max_temp_overrun_probability: float = 0.08
    max_temp_overrun_delta: float = 1.8

    diffusion_steps: int = 28
    diffusion_alpha: float = 0.24
    source_injection: float = 0.11
    fan_cooling_strength_range: tuple[float, float] = (1.8, 5.5)
    ambient_relaxation: float = 0.018
    keyboard_conductivity_range: tuple[float, float] = (0.95, 1.20)
    palm_rest_conductivity_range: tuple[float, float] = (0.62, 0.88)
    touchpad_conductivity_range: tuple[float, float] = (0.48, 0.72)
    heatpipe_conductivity_gain_range: tuple[float, float] = (1.4, 2.4)
    fan_flow_cooling_strength_range: tuple[float, float] = (0.9, 2.0)
    source_softening_sigma_scale: float = 0.010
    sink_softening_sigma_scale: float = 0.008
    final_smoothing_sigma_scale: float = 0.016

    noise_level: float = 0.6
    blur_level: float = 1.2
    hot_pixel_ratio: float = 0.0005
    dead_pixel_ratio: float = 0.0005
    impulse_noise_probability: float = 0.45
    thermal_drift_range: tuple[float, float] = (-1.5, 1.5)

    rotation_range: tuple[float, float] = (-12.0, 12.0)
    translation_range: tuple[int, int] = (-12, 12)
    brightness_range: tuple[float, float] = (0.9, 1.1)
    contrast_range: tuple[float, float] = (0.9, 1.15)

    randomize_colormap: bool = False
    fixed_colormap: str = "INFERNO"
    color_maps: tuple[str, ...] = ("JET", "HOT", "INFERNO", "MAGMA", "TURBO")

    enable_hybrid_strategy: bool = False
    hybrid_reference_dir: Path = Path("")
    hybrid_reference_weight: float = 0.35
    hybrid_detail_preserve_strength: float = 0.80
    hybrid_lowfreq_sigma: float = 2.2

    enable_physics_gate: bool = True
    physics_score_threshold: float = 0.72
    physics_max_attempts: int = 8

    save_mask: bool = True
    save_temperature: bool = True
    save_json: bool = True

    @property
    def valid_sizes(self) -> tuple[int, ...]:
        """Return supported square output sizes."""
        return (320, 640, 1024)

    @property
    def common_sensor_output_sizes(self) -> tuple[tuple[int, int], ...]:
        """Return common real-world thermal sensor output resolutions."""
        return (
            (160, 120),
            (320, 240),
            (640, 480),
            (640, 512),
            (640, 640),
            (1024, 1024),
        )

    def validate(self) -> None:
        """Validate configuration values and raise ValueError on invalid setup."""
        if self.image_size not in self.valid_sizes:
            raise ValueError(f"image_size must be one of {self.valid_sizes}, got {self.image_size}")
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be > 0")
        if self.sensor_output_size is not None:
            w, h = int(self.sensor_output_size[0]), int(self.sensor_output_size[1])
            if w <= 0 or h <= 0:
                raise ValueError("sensor_output_size values must be > 0")
            if min(w, h) < 64:
                raise ValueError("sensor_output_size minimum edge must be >= 64")
        if self.workers < 0:
            raise ValueError("workers must be >= 0")
        if self.deployment_thermal_state not in {"post_stress"}:
            raise ValueError("deployment_thermal_state must be 'post_stress'")
        if self.diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be > 0")
        if not 0.0 < self.diffusion_alpha <= 1.0:
            raise ValueError("diffusion_alpha must be in (0, 1]")
        if not 0.0 <= self.ambient_relaxation <= 1.0:
            raise ValueError("ambient_relaxation must be in [0, 1]")
        if self.source_softening_sigma_scale < 0.0 or self.sink_softening_sigma_scale < 0.0:
            raise ValueError("source/sink softening sigma scales must be >= 0")
        if self.final_smoothing_sigma_scale < 0.0:
            raise ValueError("final_smoothing_sigma_scale must be >= 0")
        if not 0.0 <= self.impulse_noise_probability <= 1.0:
            raise ValueError("impulse_noise_probability must be in [0, 1]")
        if not 0.0 <= self.keyboard_plateau_taper <= 1.0:
            raise ValueError("keyboard_plateau_taper must be in [0, 1]")
        if self.key_grid_rows_range[0] <= 0 or self.key_grid_cols_range[0] <= 0:
            raise ValueError("key grid row/col ranges must be > 0")
        if self.touchpad_size_ratio[0] <= 0 or self.touchpad_size_ratio[1] <= 0:
            raise ValueError("touchpad_size_ratio must be > 0")
        if not 0.0 <= self.touchpad_uniform_blend <= 1.0:
            raise ValueError("touchpad_uniform_blend must be in [0, 1]")
        if self.touchpad_border_cool < 0.0:
            raise ValueError("touchpad_border_cool must be >= 0")
        if not 0.0 <= self.key_grid_blend_range[0] <= self.key_grid_blend_range[1] <= 1.0:
            raise ValueError("key_grid_blend_range must be within [0, 1]")
        if self.key_grid_gap_px_range[0] <= 0:
            raise ValueError("key_grid_gap_px_range minimum must be > 0")
        if self.key_grid_light_blur_sigma < 0.0:
            raise ValueError("key_grid_light_blur_sigma must be >= 0")
        if not 0.0 <= self.screen_visible_probability_range[0] <= self.screen_visible_probability_range[1] <= 1.0:
            raise ValueError("screen_visible_probability_range must be within [0, 1]")
        if self.camera_tilt_deg_range[1] <= self.camera_tilt_deg_range[0]:
            raise ValueError("camera_tilt_deg_range max must be > min")
        if not 0.0 < self.screen_region_fraction_range[0] <= self.screen_region_fraction_range[1] < 0.5:
            raise ValueError("screen_region_fraction_range must be in (0, 0.5)")
        if self.hinge_width_px_range[0] <= 0:
            raise ValueError("hinge_width_px_range minimum must be > 0")
        if not 0.0 <= self.hinge_heating_probability <= 1.0:
            raise ValueError("hinge_heating_probability must be in [0, 1]")
        if self.hinge_heat_boost_range[1] < self.hinge_heat_boost_range[0]:
            raise ValueError("hinge_heat_boost_range max must be >= min")
        if self.screen_texture_noise_std_range[0] < 0.0:
            raise ValueError("screen_texture_noise_std_range minimum must be >= 0")
        if self.palm_rest_warm_offset_range[0] < 0.0:
            raise ValueError("palm_rest_warm_offset_range minimum must be >= 0")
        if self.touchpad_target_offset_range[1] < self.touchpad_target_offset_range[0]:
            raise ValueError("touchpad_target_offset_range max must be >= min")
        if not 0.0 <= self.trap_zone_probability <= 1.0:
            raise ValueError("trap_zone_probability must be in [0, 1]")
        if not 0.0 <= self.distractor_heat_source_probability <= 1.0:
            raise ValueError("distractor_heat_source_probability must be in [0, 1]")
        if not 0.0 <= self.occlusion_probability <= 1.0:
            raise ValueError("occlusion_probability must be in [0, 1]")
        if not 0.0 <= self.trap_zone_hotter_than_source_probability <= 1.0:
            raise ValueError("trap_zone_hotter_than_source_probability must be in [0, 1]")
        if self.trap_zone_sigma_x_ratio_range[0] <= 0.0 or self.trap_zone_sigma_y_ratio_range[0] <= 0.0:
            raise ValueError("trap_zone sigma ratio ranges must be > 0")
        if self.trap_zone_rise_range[1] < self.trap_zone_rise_range[0]:
            raise ValueError("trap_zone_rise_range max must be >= min")
        if self.trap_zone_hot_rise_range[1] < self.trap_zone_hot_rise_range[0]:
            raise ValueError("trap_zone_hot_rise_range max must be >= min")
        if not 0.0 <= self.source_attribution_threshold <= 1.0:
            raise ValueError("source_attribution_threshold must be in [0, 1]")
        if self.target_point_blur_sigma < 0.0:
            raise ValueError("target_point_blur_sigma must be >= 0")
        if self.display_temp_range[1] <= self.display_temp_range[0]:
            raise ValueError("display_temp_range max must be > min")
        if self.fixed_colormap not in self.color_maps:
            raise ValueError(f"fixed_colormap must be one of {self.color_maps}")
        if not 0.0 <= self.hybrid_reference_weight <= 1.0:
            raise ValueError("hybrid_reference_weight must be in [0, 1]")
        if not 0.0 <= self.hybrid_detail_preserve_strength <= 1.0:
            raise ValueError("hybrid_detail_preserve_strength must be in [0, 1]")
        if self.hybrid_lowfreq_sigma < 0.1:
            raise ValueError("hybrid_lowfreq_sigma must be >= 0.1")
        if len(self.power_class_weights) != 3:
            raise ValueError("power_class_weights must have 3 values for thin/mainstream/gaming")
        if sum(self.power_class_weights) <= 0:
            raise ValueError("power_class_weights sum must be > 0")
        if not 0.0 <= self.thin_numpad_probability <= 1.0:
            raise ValueError("thin_numpad_probability must be in [0, 1]")
        if not 0.0 <= self.mainstream_numpad_probability <= 1.0:
            raise ValueError("mainstream_numpad_probability must be in [0, 1]")
        if not 0.0 <= self.gaming_numpad_probability <= 1.0:
            raise ValueError("gaming_numpad_probability must be in [0, 1]")
        if not 0.0 < self.framing_fill_fraction_range[0] <= self.framing_fill_fraction_range[1] <= 1.0:
            raise ValueError("framing_fill_fraction_range must be within (0, 1]")
        for name, rng in (
            ("thin_vent_width_fraction_range", self.thin_vent_width_fraction_range),
            ("mainstream_vent_width_fraction_range", self.mainstream_vent_width_fraction_range),
            ("gaming_vent_width_fraction_range", self.gaming_vent_width_fraction_range),
        ):
            if not 0.0 < float(rng[0]) <= float(rng[1]) <= 1.0:
                raise ValueError(f"{name} must be within (0, 1]")
        if len(self.held_out_power_classes) == 0:
            raise ValueError("held_out_power_classes must contain at least one class")
        invalid_held_out = set(self.held_out_power_classes) - set(self.power_class_names)
        if invalid_held_out:
            raise ValueError(
                f"held_out_power_classes contains invalid classes: {sorted(invalid_held_out)}"
            )
        if not 0.0 <= self.physics_score_threshold <= 1.0:
            raise ValueError("physics_score_threshold must be in [0, 1]")
        if self.physics_max_attempts <= 0:
            raise ValueError("physics_max_attempts must be > 0")

        for name, rng in (
            ("thin_cpu_temp_range", self.thin_cpu_temp_range),
            ("mainstream_cpu_temp_range", self.mainstream_cpu_temp_range),
            ("gaming_cpu_temp_range", self.gaming_cpu_temp_range),
            ("thin_gpu_temp_range", self.thin_gpu_temp_range),
            ("mainstream_gpu_temp_range", self.mainstream_gpu_temp_range),
            ("gaming_gpu_temp_range", self.gaming_gpu_temp_range),
            ("thin_vent_temp_range", self.thin_vent_temp_range),
            ("mainstream_vent_temp_range", self.mainstream_vent_temp_range),
            ("gaming_vent_temp_range", self.gaming_vent_temp_range),
            ("thin_trap_temp_range", self.thin_trap_temp_range),
            ("mainstream_trap_temp_range", self.mainstream_trap_temp_range),
            ("gaming_trap_temp_range", self.gaming_trap_temp_range),
        ):
            if float(rng[1]) < float(rng[0]):
                raise ValueError(f"{name} max must be >= min")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "GeneratorConfig":
        """Build a config from a plain dictionary."""
        normalized = dict(values)
        if "sensor_output_size" in normalized and normalized["sensor_output_size"] is not None:
            raw_size = normalized["sensor_output_size"]
            if isinstance(raw_size, (list, tuple)) and len(raw_size) == 2:
                normalized["sensor_output_size"] = (int(raw_size[0]), int(raw_size[1]))
        if "output_dir" in normalized and not isinstance(normalized["output_dir"], Path):
            normalized["output_dir"] = Path(str(normalized["output_dir"]))
        if "hybrid_reference_dir" in normalized and not isinstance(normalized["hybrid_reference_dir"], Path):
            normalized["hybrid_reference_dir"] = Path(str(normalized["hybrid_reference_dir"]))
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a JSON/YAML-serializable dictionary."""
        result = asdict(self)
        if result.get("sensor_output_size") is not None:
            result["sensor_output_size"] = [int(result["sensor_output_size"][0]), int(result["sensor_output_size"][1])]
        result["output_dir"] = str(self.output_dir)
        result["hybrid_reference_dir"] = str(self.hybrid_reference_dir)
        return result
