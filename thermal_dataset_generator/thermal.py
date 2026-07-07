"""Thermal simulation core for synthetic keyboard-deck temperature fields."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .components import ComponentSpec, HeatPipeSpec, SceneLayout, VentSpec
from .config import GeneratorConfig


@dataclass(slots=True)
class ThermalSample:
    """Container for thermal outputs before RGB rendering."""

    temperature: np.ndarray
    cpu_mask: np.ndarray
    component_masks: dict[str, np.ndarray]
    source_numerator_map: np.ndarray
    trap_rise_map: np.ndarray
    trap_center: tuple[float, float] | None


class ThermalSimulator:
    """Generates raw temperature matrix from randomized component layout."""

    def __init__(self, config: GeneratorConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.size = config.image_size

    def generate_matrix(self, layout: SceneLayout, ambient_temp: float, fan_speed: int) -> ThermalSample:
        """Build physically plausible thermal map using source-diffusion-sink dynamics."""
        yy, xx = np.indices((self.size, self.size), dtype=np.float32)
        temp = np.full((self.size, self.size), ambient_temp, dtype=np.float32)
        sink_map = np.zeros((self.size, self.size), dtype=np.float32)
        conductivity_map = np.ones((self.size, self.size), dtype=np.float32)
        cpu_rise_map = np.zeros((self.size, self.size), dtype=np.float32)
        gpu_rise_map = np.zeros((self.size, self.size), dtype=np.float32)
        heatpipe_rise_map = np.zeros((self.size, self.size), dtype=np.float32)
        keyboard_conduction_rise_map = np.zeros((self.size, self.size), dtype=np.float32)
        trap_rise_map = np.zeros((self.size, self.size), dtype=np.float32)
        auxiliary_rise_map = np.zeros((self.size, self.size), dtype=np.float32)
        component_masks: dict[str, np.ndarray] = {}
        touchpad_mask = np.zeros((self.size, self.size), dtype=np.uint8)
        trap_center: tuple[float, float] | None = None

        screen_mask = np.zeros((self.size, self.size), dtype=np.uint8)
        hinge_mask = np.zeros((self.size, self.size), dtype=np.uint8)
        if layout.screen_area.enabled:
            screen_mask = self._rect_mask(layout.screen_area)
            component_masks["screen"] = screen_mask
        if layout.hinge.enabled:
            hinge_mask = self._line_mask_from_hinge(layout.hinge.start, layout.hinge.end, layout.hinge.width)
            component_masks["hinge"] = hinge_mask

        keyboard_mask = np.zeros((self.size, self.size), dtype=np.uint8)
        keyboard_bounds: tuple[int, int, int, int] | None = None
        if layout.keyboard_area.enabled:
            keyboard_mask = self._rect_mask(layout.keyboard_area)
            keyboard_bounds = self._rect_bounds(layout.keyboard_area)
            component_masks["keyboard_area"] = keyboard_mask
            k_keyboard = float(self.rng.uniform(*self.config.keyboard_conductivity_range))
            conductivity_map[keyboard_mask > 0] = k_keyboard
            center_bias = self._gaussian(xx, yy, self.size * 0.5, self.size * 0.48, self.size * 0.45, self.size * 0.35)
            # Broad keyboard plateau that remains elevated above ambient across most key area.
            plateau_strength = float(self.rng.uniform(*self.config.keyboard_plateau_strength_range))
            kx0, ky0, kx1, ky1 = keyboard_bounds
            keyboard_top = int(ky0)
            keyboard_height = max(1, int(ky1 - ky0))
            keyboard_bottom = int(keyboard_top + keyboard_height * layout.power_class.keyboard_plateau_coverage)
            keyboard_bottom = int(np.clip(keyboard_bottom, keyboard_top + 1, int(ky1)))
            taper_vec = np.ones((self.size,), dtype=np.float32)
            key_height = max(1, keyboard_bottom - keyboard_top)
            rows = np.arange(self.size, dtype=np.float32)
            key_progress = np.clip((rows - keyboard_top) / key_height, 0.0, 1.0)
            taper_vec *= (1.0 - self.config.keyboard_plateau_taper * key_progress)
            taper_vec = np.clip(taper_vec, 0.50, 1.0)

            # Soft coverage gate: above coverage end ~1, then smoothly decays to ~0.
            transition_px = max(4.0, self.size * 0.018)
            z = (rows - (keyboard_bottom - transition_px)) / (2.0 * transition_px)
            coverage_window = 1.0 - np.clip(z, 0.0, 1.0)
            coverage_window = coverage_window * coverage_window * (3.0 - 2.0 * coverage_window)

            taper = taper_vec[:, None]
            coverage = coverage_window[:, None]
            # Couple plateau laterally to actual source side (CPU/GPU), not fixed symmetry.
            source_x = float(layout.cpu.center[0])
            source_weight_sum = 1.0
            if layout.gpu.enabled:
                source_x += float(layout.gpu.center[0]) * 0.75
                source_weight_sum += 0.75
            source_x /= max(1e-6, source_weight_sum)
            lateral_sigma = max(10.0, float(kx1 - kx0) * 0.24)
            lateral_bias = np.exp(-((xx - source_x) ** 2) / (2.0 * lateral_sigma * lateral_sigma)).astype(np.float32)
            lateral_bias = 0.58 + 0.42 * lateral_bias

            plateau_term = plateau_strength * taper * coverage * lateral_bias
            broad_term = 0.42 * center_bias * (0.20 + 0.80 * coverage) * (0.78 + 0.22 * lateral_bias)
            keyboard_conduction_rise_map += keyboard_mask * (plateau_term + broad_term)

        if layout.palm_rest.enabled:
            palm_mask = self._rect_mask(layout.palm_rest)
            component_masks["palm_rest"] = palm_mask
            # Keep palm cooling moderate to avoid unrealistically tiny warm regions.
            palm_cooling_scale = self._palm_cooling_scale_for_power_class(layout.power_class.name)
            sink_map += palm_mask * self.rng.uniform(0.10, 0.30) * palm_cooling_scale
            k_palm = float(self.rng.uniform(*self.config.palm_rest_conductivity_range))
            conductivity_map[palm_mask > 0] = np.minimum(conductivity_map[palm_mask > 0], k_palm)

            if self.config.enable_touchpad:
                palm_x0, palm_y0, palm_x1, palm_y1 = self._rect_bounds(layout.palm_rest)
                if layout.keyboard_area.enabled:
                    _, _, _, keyboard_y1 = self._rect_bounds(layout.keyboard_area)
                else:
                    keyboard_y1 = int(self.size * 0.78)

                # Keep touchpad strictly below keyboard with a visual gap.
                min_gap_px = max(3, int(round(self.size * 0.009)))
                safe_top = max(palm_y0, keyboard_y1 + min_gap_px)
                safe_bottom = palm_y1

                touchpad_w = int(round(self.size * self.config.touchpad_size_ratio[0]))
                touchpad_h = int(round(self.size * self.config.touchpad_size_ratio[1]))
                touchpad_w = max(12, min(touchpad_w, max(12, palm_x1 - palm_x0 - 2)))
                touchpad_h = max(10, min(touchpad_h, max(10, safe_bottom - safe_top - 2)))

                if safe_bottom - safe_top > 10 and palm_x1 - palm_x0 > 12:
                    cx = int(self.size * 0.50)
                    cy_nominal = int(self.size * self.rng.uniform(*self.config.touchpad_center_y_ratio_range))
                    cy_min = safe_top + touchpad_h // 2
                    cy_max = safe_bottom - touchpad_h // 2
                    if cy_max > cy_min:
                        cy = int(np.clip(cy_nominal, cy_min, cy_max))
                    else:
                        cy = int((safe_top + safe_bottom) // 2)

                    cx_min = palm_x0 + touchpad_w // 2
                    cx_max = palm_x1 - touchpad_w // 2
                    if cx_max > cx_min:
                        cx = int(np.clip(cx, cx_min, cx_max))

                    touchpad_component = ComponentSpec(
                        center=(float(cx), float(cy)),
                        size=(float(touchpad_w), float(touchpad_h)),
                        angle=0.0,
                        temperature=ambient_temp,
                        enabled=True,
                    )
                    touchpad_mask = self._rect_mask(touchpad_component)
                    touchpad_mask = np.where((touchpad_mask > 0) & (palm_mask > 0), 1, 0).astype(np.uint8)
                    component_masks["touchpad"] = touchpad_mask
                    touchpad_cool = self.rng.uniform(*self.config.touchpad_cool_strength_range)
                    sink_map += touchpad_mask * touchpad_cool
                    k_touchpad = float(self.rng.uniform(*self.config.touchpad_conductivity_range))
                    conductivity_map[touchpad_mask > 0] = np.minimum(conductivity_map[touchpad_mask > 0], k_touchpad)

        cpu_mask = np.zeros((self.size, self.size), dtype=np.uint8)
        if layout.cpu.enabled:
            cpu_mask = self._cpu_mask(layout.cpu, layout.cpu_shape)
            component_masks["cpu"] = cpu_mask
            sigma_x = max(7.0, layout.cpu.size[0] * 0.75)
            sigma_y = max(7.0, layout.cpu.size[1] * 0.75)
            cpu_source = self._gaussian(xx, yy, layout.cpu.center[0], layout.cpu.center[1], sigma_x, sigma_y)
            power_gain = (layout.cpu_power / 65.0) ** 0.9
            cpu_rise_map += cpu_source * (layout.cpu.temperature - ambient_temp) * power_gain

        if layout.gpu.enabled:
            component_masks["gpu"] = self._ellipse_mask(layout.gpu)
            gpu_source = self._gaussian(
                xx,
                yy,
                layout.gpu.center[0],
                layout.gpu.center[1],
                max(6.0, layout.gpu.size[0] * 0.8),
                max(6.0, layout.gpu.size[1] * 0.8),
            )
            gpu_rise_map += gpu_source * (layout.gpu.temperature - ambient_temp) * 0.75

        if layout.heatpipe.enabled:
            heatpipe_mask = self._line_mask(layout.heatpipe)
            component_masks["heatpipe"] = heatpipe_mask
            pipe_source, pipe_progress = self._line_heat_source_with_progress(xx, yy, layout.heatpipe)
            cpu_factor = (layout.cpu.temperature - ambient_temp) * 0.30
            cpu_t = self._project_progress_on_line(layout.heatpipe.start, layout.heatpipe.end, layout.cpu.center)
            gpu_t = cpu_t
            if layout.gpu.enabled:
                gpu_t = self._project_progress_on_line(layout.heatpipe.start, layout.heatpipe.end, layout.gpu.center)

            # Chain multiple source peaks along the pipe (CPU/GPU) to avoid a flat painted stripe.
            axial_sigma = float(self.rng.uniform(0.14, 0.20))
            cpu_peak = np.exp(-((pipe_progress - cpu_t) ** 2) / (2.0 * axial_sigma * axial_sigma))
            gpu_sigma = axial_sigma * 1.10
            gpu_peak = np.exp(-((pipe_progress - gpu_t) ** 2) / (2.0 * gpu_sigma * gpu_sigma))
            if layout.gpu.enabled:
                axial_peaks = np.clip(0.78 * cpu_peak + 0.55 * gpu_peak, 0.0, None)
            else:
                axial_peaks = cpu_peak

            peak_norm = float(np.max(axial_peaks))
            if peak_norm > 1e-6:
                axial_peaks = axial_peaks / peak_norm
            axial_floor = float(self.rng.uniform(0.16, 0.24))
            # Enforce directional decay from source-side toward outlet-side along heatpipe.
            outlet_point = np.array(layout.fan.center if layout.fan.enabled else layout.heatpipe.end, dtype=np.float32)
            if layout.vents:
                enabled_vents = [v for v in layout.vents if v.enabled]
                if enabled_vents:
                    vent_mid = np.mean(
                        np.array(
                            [
                                [(v.start[0] + v.end[0]) * 0.5, (v.start[1] + v.end[1]) * 0.5]
                                for v in enabled_vents
                            ],
                            dtype=np.float32,
                        ),
                        axis=0,
                    )
                    outlet_point = 0.60 * outlet_point + 0.40 * vent_mid

            source_t = float(cpu_t)
            outlet_t = self._project_progress_on_line(layout.heatpipe.start, layout.heatpipe.end, (float(outlet_point[0]), float(outlet_point[1])))
            direction_sign = 1.0 if outlet_t >= source_t else -1.0
            travel = np.clip((pipe_progress - source_t) * direction_sign, 0.0, None)
            decay_scale = float(self.rng.uniform(0.30, 0.44))
            directional_decay = np.exp(-travel / max(1e-4, decay_scale)).astype(np.float32)

            directional_boost = (
                (axial_floor + (1.0 - axial_floor) * axial_peaks)
                * directional_decay
                * layout.heatpipe.conductivity
            )
            heatpipe_rise_map += pipe_source * directional_boost * cpu_factor

            # Couple keyboard conduction to actual heatpipe route to preserve source-side asymmetry.
            if np.any(keyboard_mask > 0):
                pipe_keyboard = cv2.GaussianBlur(
                    (pipe_source * directional_decay).astype(np.float32),
                    (0, 0),
                    sigmaX=max(1.5, self.size * 0.010),
                    sigmaY=max(1.5, self.size * 0.010),
                )
                keyboard_conduction_rise_map += (keyboard_mask > 0).astype(np.float32) * pipe_keyboard * cpu_factor * 0.12

            k_gain = float(self.rng.uniform(*self.config.heatpipe_conductivity_gain_range))
            conductivity_map[heatpipe_mask > 0] = np.maximum(conductivity_map[heatpipe_mask > 0], k_gain)

        if self.config.enable_trap_zone and self.rng.random() < float(self.config.trap_zone_probability):
            trap_cx, trap_cy = self._select_trap_zone_center(layout)
            trap_center = (float(trap_cx), float(trap_cy))
            sigma_x = float(self.size) * float(self.rng.uniform(*self.config.trap_zone_sigma_x_ratio_range))
            sigma_y = float(self.size) * float(self.rng.uniform(*self.config.trap_zone_sigma_y_ratio_range))
            trap_blob = self._gaussian(xx, yy, trap_cx, trap_cy, sigma_x, sigma_y)
            trap_temp_min, trap_temp_max = self._trap_temp_range_for_power_class(layout.power_class.name)
            trap_temp = float(self.rng.uniform(trap_temp_min, trap_temp_max))
            trap_rise = max(0.4, trap_temp - float(ambient_temp))
            if self.rng.random() < float(self.config.trap_zone_hotter_than_source_probability):
                trap_rise *= float(self.rng.uniform(1.05, 1.18))
            trap_rise_map += trap_blob * trap_rise
            trap_mask = (trap_blob > 0.42).astype(np.uint8)
            component_masks["trap_zone"] = trap_mask

        if np.any(hinge_mask > 0):
            if self.rng.random() < float(self.config.hinge_heating_probability):
                hinge_boost = float(self.rng.uniform(*self.config.hinge_heat_boost_range))
                hinge_source = (hinge_mask > 0).astype(np.float32)
                auxiliary_rise_map += hinge_source * hinge_boost

        fan_field = np.zeros((self.size, self.size), dtype=np.float32)
        if layout.fan.enabled:
            fan_mask = self._ellipse_mask(layout.fan)
            component_masks["fan"] = fan_mask
            fan_field = self._gaussian(
                xx,
                yy,
                layout.fan.center[0],
                layout.fan.center[1],
                max(8.0, layout.fan.size[0] * 1.1),
                max(8.0, layout.fan.size[1] * 1.1),
            )
            cooling = (fan_speed / 100.0) * self.rng.uniform(*self.config.fan_cooling_strength_range)
            sink_map += fan_field * cooling

            if layout.vents and fan_speed > 0:
                flow_gain = (fan_speed / 100.0) * float(self.rng.uniform(*self.config.fan_flow_cooling_strength_range))
                for vent in layout.vents:
                    if not vent.enabled:
                        continue
                    flow_channel = self._line_heat_source_between_points(
                        xx,
                        yy,
                        start=layout.fan.center,
                        end=vent.start,
                        width=max(6.0, layout.fan.size[0] * 0.8),
                    )
                    sink_map += flow_channel * flow_gain

        if layout.vents:
            vent_union = np.zeros((self.size, self.size), dtype=np.uint8)
            for idx, vent in enumerate(layout.vents):
                if not vent.enabled:
                    continue
                vent_mask = self._line_mask_from_vent(vent)
                component_masks[f"vent_{idx}"] = vent_mask
                vent_union = np.maximum(vent_union, vent_mask)
                vent_source = self._line_heat_source_from_vent(xx, yy, vent)
                vent_boost = max(0.0, vent.temperature - ambient_temp)
                auxiliary_rise_map += vent_source * vent_boost * 0.76
            component_masks["vent"] = vent_union

        if layout.battery.enabled:
            battery_mask = self._rect_mask(layout.battery)
            component_masks["battery"] = battery_mask
            auxiliary_rise_map += battery_mask * self.rng.uniform(0.5, 1.6)

        if layout.ssd.enabled:
            ssd_mask = self._rect_mask(layout.ssd)
            component_masks["ssd"] = ssd_mask
            auxiliary_rise_map += ssd_mask * self.rng.uniform(1.2, 2.8)

        if layout.vrm.enabled:
            vrm_mask = self._rect_mask(layout.vrm)
            component_masks["vrm"] = vrm_mask
            auxiliary_rise_map += vrm_mask * self.rng.uniform(1.4, 3.2)

        if np.any(screen_mask > 0):
            # Screen/lid region is near ambient and should not host thermal sources.
            for rise_map in (
                cpu_rise_map,
                gpu_rise_map,
                heatpipe_rise_map,
                keyboard_conduction_rise_map,
                trap_rise_map,
                auxiliary_rise_map,
            ):
                rise_map[screen_mask > 0] = 0.0
            sink_map[screen_mask > 0] = 0.0

        if self.config.source_softening_sigma_scale > 0.0:
            sigma_src = max(0.35, float(self.size) * self.config.source_softening_sigma_scale)
            cpu_rise_map = cv2.GaussianBlur(cpu_rise_map, (0, 0), sigmaX=sigma_src, sigmaY=sigma_src)
            gpu_rise_map = cv2.GaussianBlur(gpu_rise_map, (0, 0), sigmaX=sigma_src, sigmaY=sigma_src)
            heatpipe_rise_map = cv2.GaussianBlur(heatpipe_rise_map, (0, 0), sigmaX=sigma_src, sigmaY=sigma_src)
            keyboard_conduction_rise_map = cv2.GaussianBlur(
                keyboard_conduction_rise_map,
                (0, 0),
                sigmaX=sigma_src,
                sigmaY=sigma_src,
            )
            trap_rise_map = cv2.GaussianBlur(trap_rise_map, (0, 0), sigmaX=sigma_src, sigmaY=sigma_src)
            auxiliary_rise_map = cv2.GaussianBlur(auxiliary_rise_map, (0, 0), sigmaX=sigma_src, sigmaY=sigma_src)
        if self.config.sink_softening_sigma_scale > 0.0:
            sigma_sink = max(0.30, float(self.size) * self.config.sink_softening_sigma_scale)
            sink_map = cv2.GaussianBlur(sink_map, (0, 0), sigmaX=sigma_sink, sigmaY=sigma_sink)

        source_map = (
            cpu_rise_map
            + gpu_rise_map
            + heatpipe_rise_map
            + keyboard_conduction_rise_map
            + trap_rise_map
            + auxiliary_rise_map
        ).astype(np.float32)

        # Add a low-frequency deck spread term so warm area footprint is closer to real IR captures.
        keyboard_bin = (keyboard_mask > 0).astype(np.float32)
        palm_bin = (component_masks.get("palm_rest", np.zeros_like(temp, dtype=np.uint8)) > 0).astype(np.float32)
        deck_mask = np.clip(keyboard_bin + palm_bin, 0.0, 1.0)
        if np.any(deck_mask > 0):
            core_source = (cpu_rise_map + gpu_rise_map + heatpipe_rise_map).astype(np.float32)
            spread_sigma_x, spread_sigma_y, spread_gain_lo, spread_gain_hi = self._deck_spread_params_for_power_class(
                layout.power_class.name
            )
            spread_sigma_x = max(4.0, self.size * spread_sigma_x)
            spread_sigma_y = max(4.0, self.size * spread_sigma_y)
            spread_field = cv2.GaussianBlur(core_source, (0, 0), sigmaX=spread_sigma_x, sigmaY=spread_sigma_y)
            spread_peak = float(np.max(spread_field))
            if spread_peak > 1e-6:
                spread_field = spread_field / spread_peak
                spread_gain = max(1.2, float(layout.cpu.temperature - ambient_temp)) * float(
                    self.rng.uniform(spread_gain_lo, spread_gain_hi)
                )
                source_map += spread_field * deck_mask * spread_gain

        conductivity_map = np.clip(conductivity_map, 0.35, 3.0)

        # Source-diffusion-sink evolution ensures continuous, physically plausible gradients.
        temp += source_map * 0.42
        alpha = float(self.config.diffusion_alpha)
        source_injection = float(self.config.source_injection)
        for step in range(self.config.diffusion_steps):
            temp = self._conductive_diffusion_step(temp=temp, conductivity=conductivity_map, alpha=alpha)
            temp += source_map * source_injection

            if layout.fan.enabled and fan_speed > 0:
                progressive_gain = 0.65 + 0.35 * (step / max(1, self.config.diffusion_steps - 1))
                temp -= sink_map * progressive_gain * 0.12

            if layout.palm_rest.enabled:
                temp -= sink_map * 0.02

            # Gentle relaxation toward ambient mimics background convection/radiation.
            temp += self.config.ambient_relaxation * (ambient_temp - temp)

            temp = np.maximum(temp, ambient_temp - 6.0)

        final_sigma = max(0.45, float(self.size) * self.config.final_smoothing_sigma_scale)
        temp = cv2.GaussianBlur(temp, (0, 0), sigmaX=final_sigma, sigmaY=final_sigma)

        cap = min(float(self.config.global_max_surface_temp), float(layout.power_class.target_max_surface_temp))
        if self.rng.random() < self.config.max_temp_overrun_probability:
            cap += float(self.config.max_temp_overrun_delta)
        temp = np.clip(temp, ambient_temp - 4.0, cap)

        if self.config.enable_key_grid_texture and keyboard_bounds is not None:
            temp = self._apply_keyboard_key_grid_texture(
                temp=temp,
                keyboard_mask=keyboard_mask,
                keyboard_bounds=keyboard_bounds,
                ambient_temp=ambient_temp,
                keyboard_layout=layout.keyboard_layout,
            )
            temp = np.clip(temp, ambient_temp - 4.0, cap)

        temp = self._apply_palm_rest_three_tier(
            temp=temp,
            palm_mask=component_masks.get("palm_rest"),
            touchpad_mask=component_masks.get("touchpad"),
            keyboard_mask=component_masks.get("keyboard_area"),
            ambient_temp=ambient_temp,
        )

        if np.any(screen_mask > 0):
            temp = self._apply_screen_region_baseline(
                temp=temp,
                screen_mask=screen_mask,
                hinge_mask=hinge_mask,
                ambient_temp=ambient_temp,
            )

        temp = np.clip(temp, ambient_temp - 4.0, cap)

        touchpad_mask = component_masks.get("touchpad")
        if touchpad_mask is not None and int(np.count_nonzero(touchpad_mask)) > 0:
            temp = self._enforce_touchpad_cool_zone(temp=temp, touchpad_mask=touchpad_mask, ambient_temp=ambient_temp)
            temp = np.clip(temp, ambient_temp - 4.0, cap)

        if self.config.enable_touchpad and np.any(touchpad_mask > 0):
            temp = self._apply_touchpad_cold_zone(temp=temp, touchpad_mask=touchpad_mask, ambient_temp=ambient_temp)
            temp = np.clip(temp, ambient_temp - 4.0, cap)

        source_numerator_map = (
            cpu_rise_map + gpu_rise_map + heatpipe_rise_map + keyboard_conduction_rise_map
        ).astype(np.float32)

        return ThermalSample(
            temperature=temp.astype(np.float32),
            cpu_mask=cpu_mask,
            component_masks=component_masks,
            source_numerator_map=source_numerator_map,
            trap_rise_map=trap_rise_map.astype(np.float32),
            trap_center=trap_center,
        )

    def _cpu_mask(self, component: ComponentSpec, shape: str) -> np.ndarray:
        """Create CPU mask with selectable geometric profile."""
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        cx, cy = int(component.center[0]), int(component.center[1])
        w, h = int(component.size[0]), int(component.size[1])
        if shape == "circle":
            r = max(4, int((w + h) * 0.25))
            cv2.circle(mask, (cx, cy), r, 255, thickness=-1)
            return mask
        axes = (max(4, int(w * 0.5)), max(4, int(h * 0.5)))
        cv2.ellipse(mask, (cx, cy), axes, float(component.angle), 0, 360, 255, thickness=-1)
        return mask

    def _ellipse_mask(self, component: ComponentSpec) -> np.ndarray:
        """Create generic ellipse mask for component footprints."""
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        cx, cy = int(component.center[0]), int(component.center[1])
        axes = (max(3, int(component.size[0] * 0.5)), max(3, int(component.size[1] * 0.5)))
        cv2.ellipse(mask, (cx, cy), axes, float(component.angle), 0, 360, 255, thickness=-1)
        return mask

    def _rect_mask(self, component: ComponentSpec) -> np.ndarray:
        """Create rectangle mask from center-size component representation."""
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        x0, y0, x1, y1 = self._rect_bounds(component)
        mask[y0:y1, x0:x1] = 1
        return mask.astype(np.uint8)

    def _rect_bounds(self, component: ComponentSpec) -> tuple[int, int, int, int]:
        """Return normalized rectangle bounds [x0, y0, x1, y1) from center-size component."""
        half_w = int(component.size[0] * 0.5)
        half_h = int(component.size[1] * 0.5)
        x0 = max(0, int(component.center[0]) - half_w)
        y0 = max(0, int(component.center[1]) - half_h)
        x1 = min(self.size, int(component.center[0]) + half_w)
        y1 = min(self.size, int(component.center[1]) + half_h)
        return x0, y0, x1, y1

    def _apply_keyboard_key_grid_texture(
        self,
        temp: np.ndarray,
        keyboard_mask: np.ndarray,
        keyboard_bounds: tuple[int, int, int, int],
        ambient_temp: float,
        keyboard_layout: str,
    ) -> np.ndarray:
        """Apply key-grid block texture and cooler gap lines for realistic keyboard appearance."""
        out = temp.copy().astype(np.float32)
        x0, y0, x1, y1 = keyboard_bounds
        if x1 - x0 < 8 or y1 - y0 < 8:
            return out

        rows = int(self.rng.integers(self.config.key_grid_rows_range[0], self.config.key_grid_rows_range[1] + 1))
        cols = int(self.rng.integers(self.config.key_grid_cols_range[0], self.config.key_grid_cols_range[1] + 1))
        blend = float(self.rng.uniform(*self.config.key_grid_blend_range))
        gap_cool = float(self.rng.uniform(*self.config.key_grid_gap_cool_range))

        base_gap = int(self.rng.integers(self.config.key_grid_gap_px_range[0], self.config.key_grid_gap_px_range[1] + 1))
        gap_px = max(1, int(round(base_gap * (self.size / 640.0))))

        y_edges = np.linspace(y0, y1, rows + 1, dtype=np.int32)

        block_ranges: list[tuple[int, int]] = [(x0, x1)]
        if keyboard_layout == "numpad":
            kb_w = max(1, x1 - x0)
            gap_w = int(np.clip(round(kb_w * 0.04), 4, 20))
            main_w = int(np.clip(round(kb_w * 0.74), 10, kb_w - gap_w - 8))
            numpad_w = max(8, kb_w - main_w - gap_w)
            main_x1 = int(np.clip(x0 + main_w, x0 + 8, x1 - numpad_w - 4))
            np_x0 = int(np.clip(main_x1 + gap_w, x0 + 12, x1 - 8))
            block_ranges = [(x0, main_x1), (np_x0, x1)]

        half_gap = max(0, gap_px // 2)

        # Per-key local flattening preserves macro thermal trend while adding key-block appearance.
        for bx0, bx1 in block_ranges:
            if bx1 - bx0 < 8:
                continue
            x_edges = np.linspace(bx0, bx1, cols + 1, dtype=np.int32)
            for r in range(rows):
                cy0 = int(y_edges[r] + half_gap)
                cy1 = int(y_edges[r + 1] - half_gap)
                if cy1 <= cy0:
                    continue
                for c in range(cols):
                    cx0 = int(x_edges[c] + half_gap)
                    cx1 = int(x_edges[c + 1] - half_gap)
                    if cx1 <= cx0:
                        continue
                    cell = out[cy0:cy1, cx0:cx1]
                    cell_mask = keyboard_mask[cy0:cy1, cx0:cx1] > 0
                    if not np.any(cell_mask):
                        continue
                    cell_avg = float(np.mean(cell[cell_mask]))
                    cell[cell_mask] = (1.0 - blend) * cell[cell_mask] + blend * cell_avg
                    out[cy0:cy1, cx0:cx1] = cell

        # Crisp cooler gap lines between keys.
        # NOTE: This is not only thermal diffusion behavior. In real IR captures,
        # keycaps and key gaps often have different emissivity/material response,
        # creating visible apparent-temperature contrast at boundaries.
        for bx0, bx1 in block_ranges:
            if bx1 - bx0 < 8:
                continue
            x_edges = np.linspace(bx0, bx1, cols + 1, dtype=np.int32)
            for edge in x_edges[1:-1]:
                gx0 = max(x0, int(edge - half_gap))
                gx1 = min(x1, gx0 + gap_px)
                if gx1 <= gx0:
                    continue
                band_mask = keyboard_mask[y0:y1, gx0:gx1] > 0
                out_band = out[y0:y1, gx0:gx1]
                out_band[band_mask] -= gap_cool
                out[y0:y1, gx0:gx1] = out_band

        for edge in y_edges[1:-1]:
            gy0 = max(y0, int(edge - half_gap))
            gy1 = min(y1, gy0 + gap_px)
            if gy1 <= gy0:
                continue
            band_mask = keyboard_mask[gy0:gy1, x0:x1] > 0
            out_band = out[gy0:gy1, x0:x1]
            out_band[band_mask] -= gap_cool
            out[gy0:gy1, x0:x1] = out_band

        # Keep gaps crisp: only very light blur within keyboard mask.
        if self.config.key_grid_light_blur_sigma > 0:
            sigma = float(self.config.key_grid_light_blur_sigma) * max(0.6, self.size / 640.0)
            blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma, sigmaY=sigma)
            k_mask = keyboard_mask > 0
            out[k_mask] = blurred[k_mask]

        out[keyboard_mask > 0] = np.maximum(out[keyboard_mask > 0], ambient_temp - 0.5)
        return out

    def _conductive_diffusion_step(self, temp: np.ndarray, conductivity: np.ndarray, alpha: float) -> np.ndarray:
        """One variable-conductivity diffusion step using local flux balance."""
        t = temp.astype(np.float32)
        k = conductivity.astype(np.float32)

        tp = cv2.copyMakeBorder(t, 1, 1, 1, 1, cv2.BORDER_REFLECT)
        kp = cv2.copyMakeBorder(k, 1, 1, 1, 1, cv2.BORDER_REFLECT)

        center = tp[1:-1, 1:-1]
        north = tp[:-2, 1:-1]
        south = tp[2:, 1:-1]
        west = tp[1:-1, :-2]
        east = tp[1:-1, 2:]

        k_center = kp[1:-1, 1:-1]
        k_n = 0.5 * (k_center + kp[:-2, 1:-1])
        k_s = 0.5 * (k_center + kp[2:, 1:-1])
        k_w = 0.5 * (k_center + kp[1:-1, :-2])
        k_e = 0.5 * (k_center + kp[1:-1, 2:])

        flux = k_n * (north - center) + k_s * (south - center) + k_w * (west - center) + k_e * (east - center)
        return (center + alpha * 0.25 * flux).astype(np.float32)

    def _line_heat_source_between_points(
        self,
        xx: np.ndarray,
        yy: np.ndarray,
        start: tuple[float, float],
        end: tuple[float, float],
        width: float,
    ) -> np.ndarray:
        """Generate elongated Gaussian channel between two points."""
        pseudo = HeatPipeSpec(
            start=start,
            end=end,
            width=width,
            conductivity=1.0,
            angle=0.0,
            length=float(np.linalg.norm(np.array(end, dtype=np.float32) - np.array(start, dtype=np.float32))),
            enabled=True,
        )
        source, _ = self._line_heat_source_with_progress(xx, yy, pseudo)
        return source

    def _enforce_touchpad_cool_zone(self, temp: np.ndarray, touchpad_mask: np.ndarray, ambient_temp: float) -> np.ndarray:
        """Keep touchpad visibly cooler than nearby palm-rest region after diffusion/noise steps."""
        out = temp.copy().astype(np.float32)
        tp = (touchpad_mask > 0).astype(np.uint8)
        if int(np.count_nonzero(tp)) == 0:
            return out

        ring = cv2.dilate(tp, np.ones((25, 25), dtype=np.uint8), iterations=1)
        ring = np.clip(ring - tp, 0, 1)
        if int(np.count_nonzero(ring)) == 0:
            return out

        ring_mean = float(np.mean(out[ring > 0]))
        delta = float(self.rng.uniform(0.8, 1.8))
        target = max(ambient_temp + 0.2, ring_mean - delta)

        tp_vals = out[tp > 0]
        if tp_vals.size == 0:
            return out

        pull = float(self.rng.uniform(0.40, 0.62))
        out[tp > 0] = (1.0 - pull) * tp_vals + pull * target

        # Preserve a rectangular and recognizable touchpad boundary.
        edge = cv2.morphologyEx(tp, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
        out[edge > 0] = np.minimum(out[edge > 0], target - 0.2)

        return out

    def _apply_touchpad_cold_zone(self, temp: np.ndarray, touchpad_mask: np.ndarray, ambient_temp: float) -> np.ndarray:
        """Create a visible but realistic touchpad cold rectangle over palm-rest area."""
        out = temp.copy().astype(np.float32)
        mask = touchpad_mask > 0
        if not np.any(mask):
            return out

        delta = float(self.rng.uniform(*self.config.touchpad_post_cool_range))
        blend = float(self.config.touchpad_uniform_blend)

        local_avg = float(np.mean(out[mask]))
        target = max(ambient_temp + 0.3, local_avg - delta)
        out[mask] = (1.0 - blend) * out[mask] + blend * target

        # Keep a subtle crisp border like real touchpad outline.
        kernel = np.ones((3, 3), dtype=np.uint8)
        eroded = cv2.erode(touchpad_mask.astype(np.uint8), kernel, iterations=1)
        border = ((touchpad_mask > 0) & (eroded == 0))
        if np.any(border):
            out[border] = np.maximum(out[border], target + float(self.config.touchpad_border_cool))

        return out

    def _line_mask(self, heatpipe: HeatPipeSpec) -> np.ndarray:
        """Rasterize heat pipe path as a thick line mask."""
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        p0 = (int(heatpipe.start[0]), int(heatpipe.start[1]))
        p1 = (int(heatpipe.end[0]), int(heatpipe.end[1]))
        cv2.line(mask, p0, p1, 255, thickness=max(1, int(heatpipe.width)))
        return mask

    def _line_heat_source(self, xx: np.ndarray, yy: np.ndarray, heatpipe: HeatPipeSpec) -> np.ndarray:
        """Generate elongated Gaussian-like source around line segment."""
        source, _ = self._line_heat_source_with_progress(xx, yy, heatpipe)
        return source

    def _line_heat_source_from_vent(self, xx: np.ndarray, yy: np.ndarray, vent: VentSpec) -> np.ndarray:
        """Generate softened vent thermal plume instead of a razor-thin hot line."""
        heatpipe_like = HeatPipeSpec(
            start=vent.start,
            end=vent.end,
            width=vent.width,
            conductivity=1.0,
            angle=0.0,
            length=vent.length,
            enabled=vent.enabled,
        )
        source, _ = self._line_heat_source_with_progress(xx, yy, heatpipe_like)

        # Raw line support tends to look like an artificial stripe after camera rotation.
        # Build an inward plume profile from the vent edge and blend with a blurred strip.
        x_mid = float((vent.start[0] + vent.end[0]) * 0.5)
        y_mid = float((vent.start[1] + vent.end[1]) * 0.5)

        if str(vent.edge).startswith("top") or str(vent.edge) == "top":
            inward_dist = np.clip((yy - y_mid) / max(6.0, self.size * 0.10), 0.0, 1.0)
        elif str(vent.edge) == "left":
            inward_dist = np.clip((xx - x_mid) / max(6.0, self.size * 0.10), 0.0, 1.0)
        else:
            inward_dist = np.clip((x_mid - xx) / max(6.0, self.size * 0.10), 0.0, 1.0)

        inward_profile = np.exp(-(inward_dist ** 2) * 2.0).astype(np.float32)
        blurred = cv2.GaussianBlur(
            source.astype(np.float32),
            (0, 0),
            sigmaX=max(1.6, float(vent.width) * 1.8),
            sigmaY=max(2.8, self.size * 0.018),
        )

        plume = (0.28 * source + 0.72 * (blurred * inward_profile)).astype(np.float32)
        return np.clip(plume, 0.0, 1.0)

    def _line_mask_from_vent(self, vent: VentSpec) -> np.ndarray:
        """Rasterize vent strip as a thin line mask."""
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        p0 = (int(vent.start[0]), int(vent.start[1]))
        p1 = (int(vent.end[0]), int(vent.end[1]))
        cv2.line(mask, p0, p1, 255, thickness=max(1, int(vent.width)))
        return mask

    def _line_mask_from_hinge(self, start: tuple[float, float], end: tuple[float, float], width: float) -> np.ndarray:
        """Rasterize hinge boundary line."""
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        p0 = (int(start[0]), int(start[1]))
        p1 = (int(end[0]), int(end[1]))
        cv2.line(mask, p0, p1, 255, thickness=max(1, int(width)))
        return mask

    def _apply_screen_region_baseline(
        self,
        temp: np.ndarray,
        screen_mask: np.ndarray,
        hinge_mask: np.ndarray,
        ambient_temp: float,
    ) -> np.ndarray:
        """Keep visible screen/lid region near ambient with light texture only."""
        out = temp.copy().astype(np.float32)
        s_mask = screen_mask > 0
        if not np.any(s_mask):
            return out

        noise_std = float(self.rng.uniform(*self.config.screen_texture_noise_std_range))
        base = ambient_temp + self.rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)
        base = cv2.GaussianBlur(base, (0, 0), sigmaX=max(0.6, self.size * 0.004), sigmaY=max(0.6, self.size * 0.004))

        blend = float(self.rng.uniform(0.78, 0.92))
        out[s_mask] = (1.0 - blend) * out[s_mask] + blend * base[s_mask]

        h_mask = hinge_mask > 0
        if np.any(h_mask):
            out[h_mask] = np.maximum(out[h_mask], ambient_temp + float(self.rng.uniform(0.6, 2.0)))
        return out

    def _apply_palm_rest_three_tier(
        self,
        temp: np.ndarray,
        palm_mask: np.ndarray | None,
        touchpad_mask: np.ndarray | None,
        keyboard_mask: np.ndarray | None,
        ambient_temp: float,
    ) -> np.ndarray:
        """Enforce keyboard->palm->touchpad three-tier contrast for realistic structure."""
        out = temp.copy().astype(np.float32)
        if palm_mask is None:
            return out

        p_mask = palm_mask > 0
        if not np.any(p_mask):
            return out

        tp_mask = (touchpad_mask > 0) if touchpad_mask is not None else np.zeros_like(p_mask, dtype=bool)
        palm_only = p_mask & (~tp_mask)

        if np.any(palm_only):
            if keyboard_mask is not None and np.any(keyboard_mask > 0):
                k_mask = keyboard_mask > 0
                k_rows = np.where(k_mask)[0]
                p_rows = np.where(p_mask)[0]
                kb_bottom = int(k_rows.max()) if k_rows.size > 0 else int(p_rows.min())
                palm_top = int(p_rows.min()) if p_rows.size > 0 else kb_bottom + 1
                palm_bottom = int(p_rows.max()) if p_rows.size > 0 else palm_top + 1
                palm_span = max(2.0, float(palm_bottom - palm_top + 1))
                yy, xx = np.indices(out.shape, dtype=np.float32)
                dist_norm = np.clip((yy - float(kb_bottom + 1)) / palm_span, 0.0, 1.0)
                # Column-wise conduction projection from keyboard lower edge.
                edge_h = max(2, int(round(out.shape[0] * 0.010)))
                y0 = max(0, kb_bottom - edge_h + 1)
                edge_band = k_mask & (yy >= float(y0)) & (yy <= float(kb_bottom))
                col_ref = np.full((out.shape[1],), float(ambient_temp), dtype=np.float32)
                for cx in range(out.shape[1]):
                    col_mask = edge_band[:, cx]
                    if np.any(col_mask):
                        col_ref[cx] = float(np.percentile(out[col_mask, cx], 60.0))

                col_ref = cv2.GaussianBlur(col_ref[None, :], (0, 0), sigmaX=max(1.2, out.shape[1] * 0.018), sigmaY=0)[0]
                col_delta = np.clip(col_ref - float(ambient_temp), 0.0, None)

                # Thermal conduction from keyboard to palm decays with vertical distance.
                decay_y = (1.0 - dist_norm) ** 1.50
                target_map = ambient_temp + (0.72 * decay_y) * col_delta[None, :]
                out[palm_only] = 0.58 * out[palm_only] + 0.42 * target_map[palm_only]

                palm_floor = ambient_temp + 0.5
                keyboard_ref = float(np.percentile(out[k_mask], 45.0))
                palm_cap = max(palm_floor + 0.1, keyboard_ref - 0.35)
                out[palm_only] = np.clip(out[palm_only], palm_floor, palm_cap)
            else:
                fallback_target = ambient_temp + float(self.rng.uniform(0.4, 1.2))
                out[palm_only] = 0.70 * out[palm_only] + 0.30 * fallback_target
                out[palm_only] = np.clip(out[palm_only], ambient_temp + 0.3, ambient_temp + 1.8)

            # Build a smooth transition strip at keyboard->palm boundary.
            if keyboard_mask is not None and np.any(keyboard_mask > 0):
                k_mask = keyboard_mask > 0
                transition = (cv2.dilate(k_mask.astype(np.uint8), np.ones((7, 7), dtype=np.uint8), iterations=1) > 0) & palm_only
                if np.any(transition):
                    edge_ref = float(np.percentile(out[k_mask], 42.0))
                    out[transition] = 0.65 * out[transition] + 0.35 * edge_ref

            # Enforce physically plausible one-way conduction in palm columns:
            # avoid a cool valley followed by re-heating further down.
            palm_rows = np.where(np.any(palm_only, axis=1))[0]
            if palm_rows.size > 1:
                ry0, ry1 = int(palm_rows.min()), int(palm_rows.max())
                for cx in range(out.shape[1]):
                    col_mask = palm_only[ry0:ry1 + 1, cx]
                    if int(np.count_nonzero(col_mask)) < 3:
                        continue
                    col_vals = out[ry0:ry1 + 1, cx].copy()
                    col_idx = np.where(col_mask)[0]
                    prof = col_vals[col_idx]
                    # Non-increasing profile from top to bottom with tiny tolerance.
                    proj = np.minimum.accumulate(prof + 0.05)
                    col_vals[col_idx] = 0.30 * prof + 0.70 * proj
                    out[ry0:ry1 + 1, cx] = col_vals

        if np.any(tp_mask):
            # NOTE: Keep touchpad colder-looking than palm rest to model material
            # emissivity contrast (glass touchpad vs. palm-rest coating), not just
            # pure conductive temperature differences.
            touch_target = ambient_temp + float(self.rng.uniform(*self.config.touchpad_target_offset_range))
            out[tp_mask] = 0.42 * out[tp_mask] + 0.58 * touch_target

            edge = cv2.morphologyEx(tp_mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
            out[edge] = np.minimum(out[edge], touch_target - 0.2)

        # Seam guard: keyboard->palm transition should not collapse into a dark band.
        if keyboard_mask is not None and np.any(keyboard_mask > 0) and np.any(p_mask):
            k_mask = keyboard_mask > 0
            ky = np.where(k_mask)[0]
            py = np.where(p_mask)[0]
            if ky.size > 0 and py.size > 0:
                kb_bottom = int(ky.max())
                palm_top = int(py.min())

                band_half = max(2, int(round(self.size * 0.008)))
                y0 = max(0, min(kb_bottom, palm_top) - band_half)
                y1 = min(self.size, max(kb_bottom, palm_top) + band_half + 1)
                seam_rows = np.zeros_like(p_mask, dtype=bool)
                seam_rows[y0:y1, :] = True

                deck_region = cv2.dilate((k_mask | p_mask).astype(np.uint8), np.ones((5, 11), dtype=np.uint8), iterations=1) > 0
                seam_region = seam_rows & deck_region
                seam_region &= (~tp_mask)

                if np.any(seam_region):
                    k_ref = float(np.percentile(out[k_mask], 38.0))
                    p_ref = float(np.percentile(out[p_mask], 58.0))
                    seam_target = float(np.clip(0.60 * k_ref + 0.40 * p_ref, ambient_temp + 2.0, ambient_temp + 10.5))
                    out[seam_region] = 0.45 * out[seam_region] + 0.55 * seam_target
                    out[seam_region] = np.maximum(out[seam_region], ambient_temp + 1.6)

        return out

    def _select_trap_zone_center(self, layout: SceneLayout) -> tuple[float, float]:
        """Pick trap center in airflow-isolated pockets, away from direct exhaust channels."""
        margin = float(self.size) * 0.10
        candidates = [
            (self.size * 0.14, self.size * 0.82),
            (self.size * 0.86, self.size * 0.82),
            (self.size * 0.18, self.size * 0.32),
            (self.size * 0.82, self.size * 0.32),
            (self.size * 0.20, self.size * 0.56),
            (self.size * 0.80, self.size * 0.56),
        ]

        avoid_points: list[tuple[float, float]] = [layout.cpu.center]
        if layout.gpu.enabled:
            avoid_points.append(layout.gpu.center)
        if layout.fan.enabled:
            avoid_points.append(layout.fan.center)
        if layout.vents:
            for vent in layout.vents:
                if vent.enabled:
                    avoid_points.append(vent.start)
                    avoid_points.append(vent.end)

        airflow_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
        if layout.fan.enabled and layout.vents:
            for vent in layout.vents:
                if not vent.enabled:
                    continue
                vent_mid = ((vent.start[0] + vent.end[0]) * 0.5, (vent.start[1] + vent.end[1]) * 0.5)
                airflow_segments.append((layout.fan.center, vent_mid))

        best = candidates[0]
        best_score = -1.0
        for cx, cy in candidates:
            dists = [float(np.hypot(cx - px, cy - py)) for px, py in avoid_points]
            score = min(dists) if dists else 0.0

            # Penalize trap points near heatpipe conduction path.
            heatpipe_dist = self._point_to_segment_distance((cx, cy), layout.heatpipe.start, layout.heatpipe.end)
            if heatpipe_dist < self.size * 0.12:
                score -= (self.size * 0.12 - heatpipe_dist) * 1.8

            # Penalize points that lie on active fan->vent airflow channels.
            for seg_start, seg_end in airflow_segments:
                flow_dist = self._point_to_segment_distance((cx, cy), seg_start, seg_end)
                if flow_dist < self.size * 0.10:
                    score -= (self.size * 0.10 - flow_dist) * 2.2

            if score > best_score:
                best_score = score
                best = (cx, cy)

        jitter = float(self.size) * 0.035
        cx = float(np.clip(best[0] + self.rng.uniform(-jitter, jitter), margin, self.size - margin))
        cy = float(np.clip(best[1] + self.rng.uniform(-jitter, jitter), margin, self.size - margin))
        return cx, cy

    def _trap_temp_range_for_power_class(self, power_class: str) -> tuple[float, float]:
        """Return trap/dead-zone absolute temperature range for selected power class."""
        if power_class == "thin":
            return self.config.thin_trap_temp_range
        if power_class == "mainstream":
            return self.config.mainstream_trap_temp_range
        return self.config.gaming_trap_temp_range

    def _deck_spread_params_for_power_class(self, power_class: str) -> tuple[float, float, float, float]:
        """Return deck-spread sigma ratios and gain range for selected power class."""
        if power_class == "thin":
            return (
                float(self.rng.uniform(*self.config.thin_deck_spread_sigma_x_ratio_range)),
                float(self.rng.uniform(*self.config.thin_deck_spread_sigma_y_ratio_range)),
                float(self.config.thin_deck_spread_gain_range[0]),
                float(self.config.thin_deck_spread_gain_range[1]),
            )
        if power_class == "mainstream":
            return (
                float(self.rng.uniform(*self.config.mainstream_deck_spread_sigma_x_ratio_range)),
                float(self.rng.uniform(*self.config.mainstream_deck_spread_sigma_y_ratio_range)),
                float(self.config.mainstream_deck_spread_gain_range[0]),
                float(self.config.mainstream_deck_spread_gain_range[1]),
            )
        return (
            float(self.rng.uniform(*self.config.gaming_deck_spread_sigma_x_ratio_range)),
            float(self.rng.uniform(*self.config.gaming_deck_spread_sigma_y_ratio_range)),
            float(self.config.gaming_deck_spread_gain_range[0]),
            float(self.config.gaming_deck_spread_gain_range[1]),
        )

    def _palm_cooling_scale_for_power_class(self, power_class: str) -> float:
        """Return palm cooling scale by power class to preserve class-specific warm area size."""
        if power_class == "thin":
            return float(self.config.thin_palm_cooling_scale)
        if power_class == "mainstream":
            return float(self.config.mainstream_palm_cooling_scale)
        return float(self.config.gaming_palm_cooling_scale)

    def _line_heat_source_with_progress(
        self,
        xx: np.ndarray,
        yy: np.ndarray,
        heatpipe: HeatPipeSpec,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate anisotropic line source and normalized progress from start to end point."""
        p0 = np.array(heatpipe.start, dtype=np.float32)
        p1 = np.array(heatpipe.end, dtype=np.float32)
        v = p1 - p0
        seg_len = max(1e-6, float(np.linalg.norm(v)))
        v_hat = v / seg_len

        px = xx - p0[0]
        py = yy - p0[1]
        t_raw = (px * v_hat[0] + py * v_hat[1]) / seg_len
        t = np.clip(t_raw, 0.0, 1.0)
        proj_x = p0[0] + t * v[0]
        proj_y = p0[1] + t * v[1]
        dist = np.sqrt((xx - proj_x) ** 2 + (yy - proj_y) ** 2)
        sigma_perp = max(3.0, heatpipe.width * 0.75)

        # Keep line support uniform along axis; directional behavior is added explicitly
        # by source-side peak + outlet-side decay in the heatpipe contribution term.
        perp_decay = np.exp(-(dist ** 2) / (2.0 * sigma_perp * sigma_perp))
        source = perp_decay.astype(np.float32)
        return source, t.astype(np.float32)

    @staticmethod
    def _point_to_segment_distance(
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> float:
        """Return Euclidean distance from point to line segment."""
        px, py = float(point[0]), float(point[1])
        x1, y1 = float(start[0]), float(start[1])
        x2, y2 = float(end[0]), float(end[1])
        vx, vy = x2 - x1, y2 - y1
        denom = vx * vx + vy * vy
        if denom <= 1e-9:
            return float(np.hypot(px - x1, py - y1))
        t = ((px - x1) * vx + (py - y1) * vy) / denom
        t = float(np.clip(t, 0.0, 1.0))
        proj_x = x1 + t * vx
        proj_y = y1 + t * vy
        return float(np.hypot(px - proj_x, py - proj_y))

    @staticmethod
    def _project_progress_on_line(
        start: tuple[float, float],
        end: tuple[float, float],
        point: tuple[float, float],
    ) -> float:
        """Project point onto segment and return normalized progress in [0, 1]."""
        p0 = np.array(start, dtype=np.float32)
        p1 = np.array(end, dtype=np.float32)
        p = np.array(point, dtype=np.float32)
        v = p1 - p0
        denom = max(1e-6, float(np.dot(v, v)))
        t = float(np.dot(p - p0, v) / denom)
        return float(np.clip(t, 0.0, 1.0))

    @staticmethod
    def _gaussian(xx: np.ndarray, yy: np.ndarray, cx: float, cy: float, sx: float, sy: float) -> np.ndarray:
        """Return normalized anisotropic 2D Gaussian field."""
        sx = max(1.0, float(sx))
        sy = max(1.0, float(sy))
        value = np.exp(-(((xx - cx) ** 2) / (2.0 * sx * sx) + ((yy - cy) ** 2) / (2.0 * sy * sy)))
        return value.astype(np.float32)
