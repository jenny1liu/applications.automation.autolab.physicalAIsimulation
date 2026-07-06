"""High-level dataset generator orchestrating simulation, labeling, and export."""

from __future__ import annotations

from dataclasses import dataclass
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .augmentation import ThermalAugmentor
from .components import ComponentSampler, SceneLayout
from .config import GeneratorConfig
from .groundtruth import cpu_min_distance_to_hinge_px, select_target_point
from .labels import BBox, LabelBuilder
from .thermal import ThermalSample, ThermalSimulator
from .hybrid import ReferenceCalibrator
from .utils import ensure_output_dirs, index_name, render_colormap, save_json
from .visualization import DebugVisualizer


@dataclass(slots=True)
class GeneratedSample:
    """In-memory representation of one generated dataset sample."""

    index: int
    matrix: np.ndarray
    rgb: np.ndarray
    cpu_mask: np.ndarray
    component_masks: dict[str, np.ndarray]
    yolo_label: str
    metadata: dict
    bbox: BBox


@dataclass(slots=True)
class SavedSampleRecord:
    """Minimal record for deterministic split-manifest export."""

    index: int
    power_class: str


class ThermalDatasetGenerator:
    """Create synthetic thermal dataset for detection, segmentation, and keypoint tasks."""

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.config.validate()
        self.rng = np.random.default_rng(config.random_seed)
        self.output_paths = ensure_output_dirs(config.output_dir)
        self._hybrid_calibrator = self._build_hybrid_calibrator()

    def _build_hybrid_calibrator(self) -> ReferenceCalibrator | None:
        """Create optional hybrid calibrator when reference-based strategy is enabled."""
        if not self.config.enable_hybrid_strategy:
            return None
        if str(self.config.hybrid_reference_dir).strip() == "":
            return None
        calibrator = ReferenceCalibrator(self.config.hybrid_reference_dir)
        if not calibrator.available:
            return None
        return calibrator

    def hybrid_available(self) -> bool:
        """Return whether hybrid reference calibrator is active and ready."""
        return self._hybrid_calibrator is not None

    def hybrid_reference_distance(self, matrix: np.ndarray) -> float:
        """Compute distance between matrix and loaded reference statistics."""
        if self._hybrid_calibrator is None:
            return float("inf")
        return float(self._hybrid_calibrator.distance_to_reference(matrix))

    def generate_sample(self, index: int, seed_offset: int = 0) -> GeneratedSample:
        """Generate one sample, regenerating until physics threshold passes or attempts are exhausted."""
        if not self.config.enable_physics_gate:
            sample = self._generate_sample_once(index=index, seed_offset=seed_offset, attempt=0)
            sample.metadata["physics_gate_enabled"] = False
            return sample

        best_sample: GeneratedSample | None = None
        best_score = -1.0

        for attempt in range(self.config.physics_max_attempts):
            sample = self._generate_sample_once(index=index, seed_offset=seed_offset, attempt=attempt)
            score = float(sample.metadata.get("physics_score", 0.0))

            sample.metadata["physics_gate_enabled"] = True
            sample.metadata["physics_gate_threshold"] = float(self.config.physics_score_threshold)
            sample.metadata["physics_gate_attempt"] = int(attempt + 1)
            sample.metadata["physics_gate_max_attempts"] = int(self.config.physics_max_attempts)

            if score >= self.config.physics_score_threshold:
                sample.metadata["physics_gate_passed"] = True
                sample.metadata["physics_gate_selected_best"] = False
                return sample

            if score > best_score:
                best_score = score
                best_sample = sample

        if best_sample is None:
            raise RuntimeError("Failed to generate any sample during physics-gate attempts")

        best_sample.metadata["physics_gate_passed"] = False
        best_sample.metadata["physics_gate_selected_best"] = True
        return best_sample

    def _generate_sample_once(self, index: int, seed_offset: int, attempt: int) -> GeneratedSample:
        """Generate a single candidate sample for one physics-gate attempt."""
        base_seed = self.config.random_seed + seed_offset + index
        local_rng = np.random.default_rng(base_seed + (attempt * 1_000_003))
        sampler = ComponentSampler(self.config, local_rng)
        layout = sampler.sample_layout()

        ambient = float(local_rng.uniform(*self.config.ambient_temp_range))
        fan_speed = int(local_rng.choice(self.config.fan_speed_values).item())

        simulator = ThermalSimulator(self.config, local_rng)
        raw: ThermalSample = simulator.generate_matrix(layout, ambient, fan_speed)

        augmentor = ThermalAugmentor(self.config, local_rng)
        matrix = augmentor.apply_temperature_noise(raw.temperature)
        matrix, cpu_mask, component_masks, rotation, translation = augmentor.apply_camera_transform(
            matrix,
            raw.cpu_mask,
            raw.component_masks,
        )
        source_numerator_map = self._warp_float_map(raw.source_numerator_map, self.config.image_size, rotation, translation)
        trap_rise_map = self._warp_float_map(
            raw.trap_rise_map,
            self.config.image_size,
            rotation,
            translation,
            border_mode=cv2.BORDER_CONSTANT,
        )

        hybrid_applied = False
        if self._hybrid_calibrator is not None:
            matrix = self._hybrid_calibrator.calibrate(
                matrix,
                self.config.hybrid_reference_weight,
                detail_preserve_strength=self.config.hybrid_detail_preserve_strength,
                lowfreq_sigma=self.config.hybrid_lowfreq_sigma,
            )
            hybrid_applied = True

        matrix = self._enforce_palm_rest_tiers_post_augmentation(
            matrix=matrix,
            component_masks=component_masks,
            ambient_temperature=ambient,
            rng=local_rng,
        )

        matrix, cpu_mask, component_masks, sx, sy = self._resize_to_sensor_output(
            matrix=matrix,
            cpu_mask=cpu_mask,
            component_masks=component_masks,
        )
        source_numerator_map = self._resize_float_map(source_numerator_map, matrix.shape)
        trap_rise_map = self._resize_float_map(trap_rise_map, matrix.shape)

        keyboard_mask_u8 = component_masks.get("keyboard_area", np.zeros_like(cpu_mask, dtype=np.uint8)).astype(np.uint8)
        hinge_mask_u8 = component_masks.get("hinge", np.zeros_like(cpu_mask, dtype=np.uint8)).astype(np.uint8)
        vent_mask_u8 = component_masks.get("vent", np.zeros_like(cpu_mask, dtype=np.uint8)).astype(np.uint8)
        gt = select_target_point(
            matrix=matrix,
            ambient_temperature=float(ambient),
            keyboard_mask=keyboard_mask_u8,
            vent_mask=vent_mask_u8,
            hinge_mask=hinge_mask_u8,
            cpu_mask=cpu_mask,
            source_numerator_map=source_numerator_map,
            trap_rise_map=trap_rise_map,
            source_attribution_threshold=float(self.config.source_attribution_threshold),
            blur_sigma=float(self.config.target_point_blur_sigma),
            exclusion_dilation_px=3,
        )

        target_x, target_y = int(gt.target_point[0]), int(gt.target_point[1])
        source_attribution_fraction = gt.source_attribution_fraction_map
        trap_mask = (trap_rise_map > max(0.30, float(np.max(trap_rise_map)) * 0.45)).astype(np.uint8)

        hottest_region_category = "genuine_source"
        excluded_hottest_region: str | None = None
        if gt.excluded_hottest_region == "vent" or gt.excluded_hottest_region == "hinge":
            hottest_region_category = "vent_hinge_structural_excluded"
            excluded_hottest_region = gt.excluded_hottest_region
        elif gt.excluded_hottest_region == "trap":
            hottest_region_category = "trap_dead_zone_excluded"
            excluded_hottest_region = "trap"
        elif gt.target_fallback_tier > 0:
            hottest_region_category = "low_source_attribution"

        if self.config.randomize_colormap:
            cmap_name = str(local_rng.choice(self.config.color_maps).item())
        else:
            cmap_name = self.config.fixed_colormap
        display_range = self.config.display_temp_range if self.config.use_fixed_display_temp_range else None
        rgb = render_colormap(matrix, cmap_name, temp_range=display_range)
        rgb = augmentor.apply_rgb_jitter(rgb)

        bbox = LabelBuilder.bbox_from_mask(cpu_mask)
        out_h, out_w = matrix.shape
        yolo_line = LabelBuilder.yolo_detection_line(bbox, out_w, out_h, class_id=0)

        cpu_center = self._mask_center(cpu_mask)
        gpu_mask = component_masks.get("gpu", np.zeros_like(cpu_mask))
        gpu_enabled = bool(np.count_nonzero(gpu_mask) > 0 and layout.gpu.enabled)
        gpu_center = self._optional_mask_center(gpu_mask) if gpu_enabled else (-1.0, -1.0)
        vent_centers = self._collect_vent_centers(component_masks)
        vent_temperature = float(max((v.temperature for v in layout.vents), default=ambient))
        fan_mask = component_masks.get("fan", np.zeros_like(cpu_mask, dtype=np.uint8))
        trap_zone_present = bool(np.count_nonzero(trap_mask) > 0)
        trap_center_pixel = (-1.0, -1.0)
        if raw.trap_center is not None:
            trap_start, _ = self._scale_line_endpoints(
                self._transform_line_endpoints(
                    raw.trap_center,
                    raw.trap_center,
                    self.config.image_size,
                    rotation,
                    translation,
                ),
                sx,
                sy,
            )
            trap_center_pixel = (float(trap_start[0]), float(trap_start[1]))

        trap_temperature: float | None = None
        if trap_zone_present:
            tx = int(np.clip(round(trap_center_pixel[0]), 0, matrix.shape[1] - 1))
            ty = int(np.clip(round(trap_center_pixel[1]), 0, matrix.shape[0] - 1))
            trap_temperature = float(matrix[ty, tx])
        heatpipe_endpoints = self._transform_line_endpoints(
            layout.heatpipe.start,
            layout.heatpipe.end,
            self.config.image_size,
            rotation,
            translation,
        )
        heatpipe_endpoints = self._scale_line_endpoints(heatpipe_endpoints, sx, sy)
        vent_endpoints = [
            self._scale_line_endpoints(
                self._transform_line_endpoints(
                    vent.start,
                    vent.end,
                    self.config.image_size,
                    rotation,
                    translation,
                ),
                sx,
                sy,
            )
            for vent in layout.vents
            if vent.enabled
        ]
        fan_center_transformed, _ = self._scale_line_endpoints(
            self._transform_line_endpoints(
                layout.fan.center,
                layout.fan.center,
                self.config.image_size,
                rotation,
                translation,
            ),
            sx,
            sy,
        )
        hinge_endpoints = self._transform_line_endpoints(
            layout.hinge.start,
            layout.hinge.end,
            self.config.image_size,
            rotation,
            translation,
        ) if layout.hinge.enabled else ((-1.0, -1.0), (-1.0, -1.0))
        hinge_endpoints = self._scale_line_endpoints(hinge_endpoints, sx, sy)
        physics_metrics = self._compute_physics_metrics(
            matrix=matrix,
            ambient_temperature=ambient,
            fan_mask=fan_mask,
            heatpipe_start=heatpipe_endpoints[0],
            heatpipe_end=heatpipe_endpoints[1],
        )
        effective_coverage = self._estimate_keyboard_effective_coverage(matrix=matrix, ambient_temperature=ambient)

        metadata = {
            "power_class": layout.power_class.name,
            "keyboard_layout": str(layout.keyboard_layout),
            "keyboard_plateau_coverage": float(layout.power_class.keyboard_plateau_coverage),
            "keyboard_plateau_target_coverage": float(layout.power_class.keyboard_plateau_coverage),
            "keyboard_plateau_effective_coverage": float(effective_coverage),
            "cpu_center_pixel": [float(cpu_center[0]), float(cpu_center[1])],
            "cpu_bbox": [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
            "cpu_temperature": float(layout.cpu.temperature),
            "cpu_power": int(layout.cpu_power),
            "gpu_enabled": gpu_enabled,
            "gpu_center": [float(gpu_center[0]), float(gpu_center[1])],
            "gpu_temperature": float(layout.gpu.temperature) if gpu_enabled else None,
            "fan_speed": fan_speed,
            "screen_visible": bool(layout.screen_visible),
            "camera_tilt_deg": float(layout.camera_tilt_deg),
            "screen_region_fraction": float(layout.screen_region_fraction),
            "framing_fill_fraction": float(layout.framing_fill_fraction),
            "hinge_line": [
                [float(hinge_endpoints[0][0]), float(hinge_endpoints[0][1])],
                [float(hinge_endpoints[1][0]), float(hinge_endpoints[1][1])],
            ],
            "vent_positions": [[float(x), float(y)] for x, y in vent_centers],
            "vent_temperature": vent_temperature,
            "ambient_temperature": ambient,
            "heatpipe_angle": float(layout.heatpipe.angle),
            "heatpipe_length": float(layout.heatpipe.length),
            "heatpipe_conductivity": float(layout.heatpipe.conductivity),
            "rotation": float(rotation),
            "translation": [int(translation[0]), int(translation[1])],
            "simulation_resolution": [int(self.config.image_size), int(self.config.image_size)],
            "output_resolution": [int(out_w), int(out_h)],
            "sensor_output_size": [int(out_w), int(out_h)] if self.config.sensor_output_size is not None else None,
            "deployment_thermal_state": str(self.config.deployment_thermal_state),
            "generator_version": str(self.config.generator_version),
            "color_map": cmap_name,
            "display_temp_range": [float(self.config.display_temp_range[0]), float(self.config.display_temp_range[1])]
            if self.config.use_fixed_display_temp_range else None,
            "frame_temp_min": float(np.min(matrix)),
            "frame_temp_max": float(np.max(matrix)),
            "hotspot_coordinate": [float(target_x), float(target_y)],
            "hotspot_temperature": float(matrix[target_y, target_x]),
            "target_point": [int(target_x), int(target_y)],
            "target_temperature": float(gt.target_temperature),
            "target_fallback_tier": int(gt.target_fallback_tier),
            "excluded_hottest_point": [int(gt.excluded_hottest_point[0]), int(gt.excluded_hottest_point[1])],
            "runner_up_point": [int(gt.runner_up_point[0]), int(gt.runner_up_point[1])],
            "runner_up_temperature": float(gt.runner_up_temperature),
            "target_source_attribution_fraction": float(source_attribution_fraction[target_y, target_x]),
            "source_attribution_threshold": float(self.config.source_attribution_threshold),
            "source_attribution_summary": {
                "keyboard_p10": float(np.percentile(source_attribution_fraction[keyboard_mask_u8 > 0], 10.0))
                if np.any(keyboard_mask_u8 > 0) else 0.0,
                "keyboard_p50": float(np.percentile(source_attribution_fraction[keyboard_mask_u8 > 0], 50.0))
                if np.any(keyboard_mask_u8 > 0) else 0.0,
                "keyboard_p90": float(np.percentile(source_attribution_fraction[keyboard_mask_u8 > 0], 90.0))
                if np.any(keyboard_mask_u8 > 0) else 0.0,
                "valid_region_ratio": float(np.mean((keyboard_mask_u8 > 0) & (~(vent_mask_u8 > 0)) & (~(hinge_mask_u8 > 0)))),
            },
            "trap_zone_present": trap_zone_present,
            "trap_zone_center": [float(trap_center_pixel[0]), float(trap_center_pixel[1])] if trap_zone_present else None,
            "trap_zone_temperature": float(trap_temperature) if trap_temperature is not None else None,
            "trap_zone_was_global_hottest": bool(gt.trap_zone_was_global_hottest),
            "hottest_region_category": hottest_region_category,
            "excluded_hottest_region": excluded_hottest_region,
            "global_hottest_coordinate": [int(gt.excluded_hottest_point[0]), int(gt.excluded_hottest_point[1])],
            "global_hottest_temperature": float(matrix[int(gt.excluded_hottest_point[1]), int(gt.excluded_hottest_point[0])]),
            "cpu_min_distance_to_hinge_px": float(cpu_min_distance_to_hinge_px(cpu_mask, hinge_mask_u8)),
            "hybrid_strategy_enabled": bool(self.config.enable_hybrid_strategy),
            "hybrid_strategy_applied": bool(hybrid_applied),
            "hybrid_reference_weight": float(self.config.hybrid_reference_weight),
            "hybrid_reference_count": int(self._hybrid_calibrator.stats.image_count)
            if self._hybrid_calibrator is not None else 0,
            "physics_metrics": physics_metrics,
            "physics_score": float(physics_metrics["physics_score"]),
        }

        sample = GeneratedSample(
            index=index,
            matrix=matrix,
            rgb=rgb,
            cpu_mask=cpu_mask,
            component_masks=component_masks,
            yolo_label=yolo_line,
            metadata=metadata,
            bbox=bbox,
        )

        if self.config.debug_mode and attempt == 0:
            DebugVisualizer.show(
                rgb_image=rgb,
                temperature_matrix=matrix,
                bbox=bbox,
                cpu_center=(cpu_center[0], cpu_center[1]),
                gpu_center=(gpu_center[0], gpu_center[1]) if gpu_enabled else None,
                heatpipe_points=heatpipe_endpoints,
                fan_center=fan_center_transformed,
                fan_radius=layout.fan.size[0],
                vent_points=vent_endpoints,
                trap_center=trap_center_pixel if trap_zone_present else None,
                target_point=(float(target_x), float(target_y)),
                hottest_point=(float(global_hot_x), float(global_hot_y)),
                hottest_category=hottest_region_category,
                target_source_attribution=float(source_attribution_fraction[target_y, target_x]),
                title=f"Sample {index_name(index)}",
            )

        return sample

    def save_sample(self, sample: GeneratedSample) -> None:
        """Persist generated sample to disk in requested output formats."""
        name = index_name(sample.index)

        image_path = self.output_paths["images"] / f"{name}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(sample.rgb, cv2.COLOR_RGB2BGR))

        label_path = self.output_paths["labels"] / f"{name}.txt"
        label_path.write_text(sample.yolo_label + "\n", encoding="utf-8")

        target = sample.metadata.get("target_point", sample.metadata.get("hotspot_coordinate", [0.0, 0.0]))
        tx = float(target[0]) if isinstance(target, (list, tuple)) and len(target) == 2 else 0.0
        ty = float(target[1]) if isinstance(target, (list, tuple)) and len(target) == 2 else 0.0

        keypoint_line = LabelBuilder.yolo_keypoint_line(
            bbox=sample.bbox,
            width=sample.matrix.shape[1],
            height=sample.matrix.shape[0],
            keypoint_x=tx,
            keypoint_y=ty,
            class_id=0,
        )
        keypoint_label_path = self.output_paths["labels_keypoint"] / f"{name}.txt"
        keypoint_label_path.write_text(keypoint_line + "\n", encoding="utf-8")

        seg_line = LabelBuilder.yolo_segmentation_line_from_mask(
            mask=sample.cpu_mask,
            width=sample.matrix.shape[1],
            height=sample.matrix.shape[0],
            class_id=0,
        )
        seg_label_path = self.output_paths["labels_seg"] / f"{name}.txt"
        seg_label_path.write_text(seg_line + "\n", encoding="utf-8")

        if self.config.save_mask:
            mask_path = self.output_paths["masks"] / f"{name}.png"
            cv2.imwrite(str(mask_path), sample.cpu_mask)

        if self.config.save_temperature:
            temp_path = self.output_paths["temperature"] / f"{name}.npy"
            np.save(temp_path, sample.matrix)

        if self.config.save_json:
            metadata_path = self.output_paths["metadata"] / f"{name}.json"
            save_json(metadata_path, sample.metadata)

    def generate_dataset(self) -> None:
        """Generate full dataset with optional multiprocessing and progress bar."""
        indices = list(range(1, self.config.dataset_size + 1))
        worker_count = self.config.workers or max(1, cpu_count() - 1)
        saved_records: list[SavedSampleRecord] = []

        if worker_count <= 1:
            for i in tqdm(indices, desc="Generating thermal dataset"):
                sample = self.generate_sample(i)
                self.save_sample(sample)
                saved_records.append(
                    SavedSampleRecord(index=int(sample.index), power_class=str(sample.metadata.get("power_class", "unknown")))
                )
            self._write_seen_heldout_split_manifests(saved_records)
            return

        tasks = [(i, self.config) for i in indices]
        with Pool(processes=worker_count) as pool:
            for sample in tqdm(pool.imap_unordered(_worker_generate_sample, tasks), total=len(tasks), desc="Generating thermal dataset"):
                self.save_sample(sample)
                saved_records.append(
                    SavedSampleRecord(index=int(sample.index), power_class=str(sample.metadata.get("power_class", "unknown")))
                )

        self._write_seen_heldout_split_manifests(saved_records)

    def _write_seen_heldout_split_manifests(self, records: list[SavedSampleRecord]) -> None:
        """Write deterministic seen-vs-held-out manifests grouped by power class."""
        if not self.config.export_seen_heldout_split or not records:
            return

        held_out_classes = set(self.config.held_out_power_classes)
        split_dir = self.config.output_dir / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, object]] = []
        for r in sorted(records, key=lambda x: int(x.index)):
            name = index_name(int(r.index))
            rows.append(
                {
                    "index": int(r.index),
                    "sample_id": name,
                    "power_class": str(r.power_class),
                    "image": f"images/{name}.png",
                    "label": f"labels/{name}.txt",
                    "label_keypoint": f"labels_keypoint/{name}.txt",
                    "label_seg": f"labels_seg/{name}.txt",
                    "mask": f"masks/{name}.png",
                    "temperature": f"temperature/{name}.npy",
                    "metadata": f"metadata/{name}.json",
                }
            )

        seen_rows = [x for x in rows if str(x["power_class"]) not in held_out_classes]
        held_out_rows = [x for x in rows if str(x["power_class"]) in held_out_classes]

        (split_dir / "all_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        (split_dir / "seen_manifest.json").write_text(json.dumps(seen_rows, indent=2), encoding="utf-8")
        (split_dir / "held_out_manifest.json").write_text(json.dumps(held_out_rows, indent=2), encoding="utf-8")

        summary = {
            "total": len(rows),
            "seen_total": len(seen_rows),
            "held_out_total": len(held_out_rows),
            "held_out_power_classes": sorted(held_out_classes),
            "counts_by_power_class": {
                c: int(sum(1 for x in rows if str(x["power_class"]) == c))
                for c in self.config.power_class_names
            },
        }
        (split_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    @staticmethod
    def _mask_center(mask: np.ndarray) -> tuple[float, float]:
        """Compute centroid of binary mask, falling back to image center."""
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            h, w = mask.shape
            return (w * 0.5, h * 0.5)
        return (float(xs.mean()), float(ys.mean()))

    @staticmethod
    def _optional_mask_center(mask: np.ndarray | None) -> tuple[float, float]:
        """Compute centroid for optional mask, returning (-1, -1) when unavailable."""
        if mask is None:
            return (-1.0, -1.0)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return (-1.0, -1.0)
        return (float(xs.mean()), float(ys.mean()))

    @staticmethod
    def _collect_vent_centers(component_masks: dict[str, np.ndarray]) -> list[tuple[float, float]]:
        """Collect vent centers from one or more vent masks."""
        centers: list[tuple[float, float]] = []
        vent_keys = [k for k in component_masks if k.startswith("vent_")]
        for key in sorted(vent_keys):
            mask = component_masks.get(key)
            if mask is None:
                continue
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            centers.append((float(xs.mean()), float(ys.mean())))
        return centers

    def _resize_to_sensor_output(
        self,
        matrix: np.ndarray,
        cpu_mask: np.ndarray,
        component_masks: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], float, float]:
        """Resize outputs to configured sensor resolution and return scale factors."""
        if self.config.sensor_output_size is None:
            return matrix, cpu_mask, component_masks, 1.0, 1.0

        target_w = int(self.config.sensor_output_size[0])
        target_h = int(self.config.sensor_output_size[1])
        src_h, src_w = matrix.shape
        if target_w == src_w and target_h == src_h:
            return matrix, cpu_mask, component_masks, 1.0, 1.0

        resized_matrix = cv2.resize(matrix, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        resized_cpu = cv2.resize(cpu_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        resized_components: dict[str, np.ndarray] = {}
        for name, mask in component_masks.items():
            resized_components[name] = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        sx = float(target_w) / float(src_w)
        sy = float(target_h) / float(src_h)
        return resized_matrix.astype(np.float32), resized_cpu.astype(np.uint8), resized_components, sx, sy

    @staticmethod
    def _scale_line_endpoints(
        endpoints: tuple[tuple[float, float], tuple[float, float]],
        sx: float,
        sy: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Scale line endpoints by x/y factors for non-square output conversion."""
        return (
            (float(endpoints[0][0]) * sx, float(endpoints[0][1]) * sy),
            (float(endpoints[1][0]) * sx, float(endpoints[1][1]) * sy),
        )

    @staticmethod
    def _argmax_with_mask(temp: np.ndarray, mask: np.ndarray | None) -> tuple[int, int]:
        """Return hottest point (y, x), preferring values inside mask when provided."""
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

    @staticmethod
    def _resize_float_map(map_data: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """Resize float map to target (h, w) shape with bilinear interpolation."""
        target_h, target_w = int(target_shape[0]), int(target_shape[1])
        if map_data.shape == (target_h, target_w):
            return map_data.astype(np.float32)
        resized = cv2.resize(map_data.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return resized.astype(np.float32)

    @staticmethod
    def _warp_float_map(
        map_data: np.ndarray,
        image_size: int,
        rotation_deg: float,
        translation: tuple[int, int],
        border_mode: int = cv2.BORDER_REFLECT,
    ) -> np.ndarray:
        """Apply camera affine transform to float attribution maps."""
        m = cv2.getRotationMatrix2D((image_size * 0.5, image_size * 0.5), float(rotation_deg), 1.0)
        m[0, 2] += int(translation[0])
        m[1, 2] += int(translation[1])
        warped = cv2.warpAffine(
            map_data.astype(np.float32),
            m,
            (image_size, image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=border_mode,
        )
        return warped.astype(np.float32)

    def _enforce_palm_rest_tiers_post_augmentation(
        self,
        matrix: np.ndarray,
        component_masks: dict[str, np.ndarray],
        ambient_temperature: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Stabilize final palm/touchpad tiers after blur/transform/hybrid steps."""
        out = matrix.astype(np.float32).copy()

        palm_mask_raw = component_masks.get("palm_rest")
        if palm_mask_raw is None:
            return out

        palm_mask = palm_mask_raw > 0
        if not np.any(palm_mask):
            return out

        touchpad_mask_raw = component_masks.get("touchpad")
        touchpad_mask = (touchpad_mask_raw > 0) if touchpad_mask_raw is not None else np.zeros_like(palm_mask, dtype=bool)

        keyboard_mask_raw = component_masks.get("keyboard_area")
        keyboard_mask = (keyboard_mask_raw > 0) if keyboard_mask_raw is not None else np.zeros_like(palm_mask, dtype=bool)

        # Avoid a non-physical dark seam between keyboard and palm-rest.
        # Real devices usually show a smooth thermal transition across this band.
        if np.any(keyboard_mask) and np.any(palm_mask):
            kb_rows = np.where(keyboard_mask)[0]
            palm_rows = np.where(palm_mask)[0]
            if kb_rows.size > 0 and palm_rows.size > 0:
                kb_bottom = int(kb_rows.max())
                palm_top = int(palm_rows.min())
                if palm_top > kb_bottom + 1:
                    y0 = kb_bottom + 1
                    y1 = palm_top
                    bridge_zone = np.zeros_like(palm_mask, dtype=bool)
                    bridge_zone[y0:y1, :] = True

                    # Keep bridge focused on deck region between keyboard and palm.
                    deck_x = cv2.dilate((keyboard_mask | palm_mask).astype(np.uint8), np.ones((5, 9), dtype=np.uint8), iterations=1) > 0
                    bridge_zone &= deck_x
                    bridge_zone &= (~keyboard_mask)
                    bridge_zone &= (~palm_mask)

                    if np.any(bridge_zone):
                        k_ref = float(np.percentile(out[keyboard_mask], 35.0))
                        p_ref = float(np.percentile(out[palm_mask], 55.0))
                        bridge_target = float(np.clip(0.58 * k_ref + 0.42 * p_ref, ambient_temperature + 2.0, ambient_temperature + 8.5))
                        out[bridge_zone] = 0.45 * out[bridge_zone] + 0.55 * bridge_target
                        out[bridge_zone] = np.maximum(out[bridge_zone], ambient_temperature + 1.6)

                # Also warm palm top-edge neighborhood even when masks are adjacent/overlapping.
                seam_half = max(3, int(round(out.shape[0] * 0.010)))
                sy0 = max(0, min(kb_bottom, palm_top) - seam_half)
                sy1 = min(out.shape[0], max(kb_bottom, palm_top) + seam_half + 1)
                seam_rows = np.zeros_like(palm_mask, dtype=bool)
                seam_rows[sy0:sy1, :] = True
                seam_zone = seam_rows & cv2.dilate((keyboard_mask | palm_mask).astype(np.uint8), np.ones((5, 11), dtype=np.uint8), iterations=1).astype(bool)
                if np.any(touchpad_mask):
                    seam_zone &= (~touchpad_mask)
                if np.any(seam_zone):
                    k_ref = float(np.percentile(out[keyboard_mask], 38.0))
                    p_ref = float(np.percentile(out[palm_mask], 58.0))
                    seam_target = float(np.clip(0.58 * k_ref + 0.42 * p_ref, ambient_temperature + 2.0, ambient_temperature + 9.2))
                    out[seam_zone] = 0.45 * out[seam_zone] + 0.55 * seam_target
                    out[seam_zone] = np.maximum(out[seam_zone], ambient_temperature + 1.6)

        # Post-augmentation guard: keep touchpad vertically below keyboard to avoid overlap artifacts.
        if np.any(touchpad_mask) and np.any(keyboard_mask):
            kb_rows = np.where(keyboard_mask)[0]
            tp_rows = np.where(touchpad_mask)[0]
            if kb_rows.size > 0 and tp_rows.size > 0:
                kb_bottom = int(kb_rows.max())
                tp_top = int(tp_rows.min())
                min_sep = max(2, int(round(out.shape[0] * 0.006)))
                min_tp_top = kb_bottom + min_sep
                if tp_top < min_tp_top:
                    dy = int(min_tp_top - tp_top)
                    shifted = np.zeros_like(touchpad_mask, dtype=bool)
                    if dy < touchpad_mask.shape[0]:
                        shifted[dy:, :] = touchpad_mask[:-dy, :]
                    shifted &= palm_mask

                    if np.any(shifted):
                        touchpad_mask = shifted
                    else:
                        yy = np.arange(touchpad_mask.shape[0])[:, None]
                        clipped = touchpad_mask & (yy >= min_tp_top) & palm_mask
                        if np.any(clipped):
                            touchpad_mask = clipped

                    component_masks["touchpad"] = touchpad_mask.astype(np.uint8)

        palm_only = palm_mask & (~touchpad_mask)

        palm_lo = float(ambient_temperature + self.config.palm_rest_warm_offset_range[0])
        palm_hi = float(ambient_temperature + self.config.palm_rest_warm_offset_range[1])

        if np.any(palm_only):
            if np.any(keyboard_mask):
                k_rows = np.where(keyboard_mask)[0]
                p_rows = np.where(palm_mask)[0]
                kb_bottom = int(k_rows.max()) if k_rows.size > 0 else int(p_rows.min())
                palm_top = int(p_rows.min()) if p_rows.size > 0 else kb_bottom + 1
                palm_bottom = int(p_rows.max()) if p_rows.size > 0 else palm_top + 1
                palm_span = max(2.0, float(palm_bottom - palm_top + 1))
                yy, xx = np.indices(out.shape, dtype=np.float32)
                dist_norm = np.clip((yy - float(kb_bottom + 1)) / palm_span, 0.0, 1.0)
                edge_h = max(2, int(round(out.shape[0] * 0.010)))
                y0 = max(0, kb_bottom - edge_h + 1)
                edge_band = keyboard_mask & (yy >= float(y0)) & (yy <= float(kb_bottom))
                col_ref = np.full((out.shape[1],), float(ambient_temperature), dtype=np.float32)
                for cx in range(out.shape[1]):
                    col_mask = edge_band[:, cx]
                    if np.any(col_mask):
                        col_ref[cx] = float(np.percentile(out[col_mask, cx], 60.0))

                col_ref = cv2.GaussianBlur(col_ref[None, :], (0, 0), sigmaX=max(1.2, out.shape[1] * 0.018), sigmaY=0)[0]
                col_delta = np.clip(col_ref - float(ambient_temperature), 0.0, None)

                decay_y = (1.0 - dist_norm) ** 1.55
                palm_target_map = ambient_temperature + (0.70 * decay_y) * col_delta[None, :]
                out[palm_only] = 0.62 * out[palm_only] + 0.38 * palm_target_map[palm_only]

                palm_floor = ambient_temperature + 0.45
                keyboard_ref = float(np.percentile(out[keyboard_mask], 46.0))
                palm_cap = max(palm_floor + 0.1, keyboard_ref - 0.30)
                out[palm_only] = np.clip(out[palm_only], palm_floor, palm_cap)
            else:
                fallback_target = float(ambient_temperature + rng.uniform(0.4, 1.2))
                out[palm_only] = 0.72 * out[palm_only] + 0.28 * fallback_target
                out[palm_only] = np.clip(out[palm_only], ambient_temperature + 0.3, ambient_temperature + 1.8)

            # Blend a narrow transition zone to avoid an abrupt keyboard->palm cliff.
            if np.any(keyboard_mask):
                bridge = (cv2.dilate(keyboard_mask.astype(np.uint8), np.ones((9, 9), dtype=np.uint8), iterations=1) > 0) & palm_only
                if np.any(bridge):
                    keyboard_edge_ref = float(np.percentile(out[keyboard_mask], 45.0))
                    out[bridge] = 0.60 * out[bridge] + 0.40 * keyboard_edge_ref

            # Post-transform monotonic guard: palm should not cool then re-heat
            # along the same column when no local heat source exists.
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
                    proj = np.minimum.accumulate(prof + 0.05)
                    col_vals[col_idx] = 0.25 * prof + 0.75 * proj
                    out[ry0:ry1 + 1, cx] = col_vals

        if np.any(touchpad_mask):
            tp_target = float(ambient_temperature + rng.uniform(*self.config.touchpad_target_offset_range))
            out[touchpad_mask] = 0.35 * out[touchpad_mask] + 0.65 * tp_target

            tp_cap = float(ambient_temperature + self.config.touchpad_target_offset_range[1] + 0.35)
            if np.any(palm_only):
                palm_mean = float(np.mean(out[palm_only]))
                tp_cap = min(tp_cap, palm_mean - 0.20)
            out[touchpad_mask] = np.minimum(out[touchpad_mask], tp_cap)

        return out

    @staticmethod
    def _transform_line_endpoints(
        start: tuple[float, float],
        end: tuple[float, float],
        image_size: int,
        rotation_deg: float,
        translation: tuple[int, int],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Apply same rotation/translation used by camera transform to line endpoints."""
        cx = image_size * 0.5
        cy = image_size * 0.5
        rad = np.deg2rad(rotation_deg)
        cos_r = float(np.cos(rad))
        sin_r = float(np.sin(rad))
        tx, ty = translation

        def transform_point(p: tuple[float, float]) -> tuple[float, float]:
            x_shift = p[0] - cx
            y_shift = p[1] - cy
            x_rot = x_shift * cos_r + y_shift * sin_r
            y_rot = -x_shift * sin_r + y_shift * cos_r
            return (x_rot + cx + tx, y_rot + cy + ty)

        return transform_point(start), transform_point(end)

    @staticmethod
    def _compute_physics_metrics(
        matrix: np.ndarray,
        ambient_temperature: float,
        fan_mask: np.ndarray,
        heatpipe_start: tuple[float, float],
        heatpipe_end: tuple[float, float],
    ) -> dict[str, float]:
        """Compute quantitative physics plausibility metrics for one thermal sample."""
        matrix_f = matrix.astype(np.float32)

        # Smoothness/continuity: lower Laplacian variance indicates fewer non-physical discontinuities.
        lap = cv2.Laplacian(matrix_f, cv2.CV_32F, ksize=3)
        laplacian_std = float(np.std(lap))
        continuity_score = float(np.exp(-laplacian_std / 4.0))

        hot_threshold = max(float(ambient_temperature + 6.0), float(np.percentile(matrix_f, 93.0)))
        hot_mask = (matrix_f >= hot_threshold).astype(np.uint8)
        peak_map = (matrix_f == cv2.dilate(matrix_f, np.ones((3, 3), dtype=np.uint8))).astype(np.uint8)
        isolated_peaks = int(np.count_nonzero((peak_map == 1) & (hot_mask == 1)))
        hot_pixels = max(1, int(np.count_nonzero(hot_mask)))
        isolated_hotspot_ratio = float(isolated_peaks / hot_pixels)
        isolated_score = float(np.clip(1.0 - (isolated_hotspot_ratio * 12.0), 0.0, 1.0))

        fan_bin = (fan_mask > 0).astype(np.uint8)
        if int(np.count_nonzero(fan_bin)) > 5:
            dilated = cv2.dilate(fan_bin, np.ones((11, 11), dtype=np.uint8), iterations=2)
            ring = np.clip(dilated - fan_bin, 0, 1)
            fan_temp = float(np.mean(matrix_f[fan_bin > 0]))
            ring_temp = float(np.mean(matrix_f[ring > 0])) if int(np.count_nonzero(ring)) > 0 else fan_temp
            fan_cooling_delta = float(ring_temp - fan_temp)
            fan_cooling_score = float(np.clip(fan_cooling_delta / 3.0, 0.0, 1.0))
        else:
            fan_cooling_delta = 0.0
            fan_cooling_score = 0.5

        h, w = matrix_f.shape
        sample_count = 24
        xs = np.linspace(heatpipe_start[0], heatpipe_end[0], sample_count)
        ys = np.linspace(heatpipe_start[1], heatpipe_end[1], sample_count)
        xi = np.clip(np.rint(xs).astype(np.int32), 0, w - 1)
        yi = np.clip(np.rint(ys).astype(np.int32), 0, h - 1)
        line_temps = matrix_f[yi, xi]
        heatpipe_direction_drop = float(line_temps[0] - line_temps[-1])
        monotonic_steps = float(np.mean((line_temps[:-1] - line_temps[1:]) >= -0.15))
        heatpipe_direction_score = float(np.clip((heatpipe_direction_drop / 5.0) * 0.7 + monotonic_steps * 0.3, 0.0, 1.0))

        physics_score = float(
            np.clip(
                0.36 * continuity_score
                + 0.24 * isolated_score
                + 0.22 * fan_cooling_score
                + 0.18 * heatpipe_direction_score,
                0.0,
                1.0,
            )
        )

        return {
            "laplacian_std": laplacian_std,
            "continuity_score": continuity_score,
            "isolated_hotspot_ratio": isolated_hotspot_ratio,
            "isolated_score": isolated_score,
            "fan_cooling_delta": fan_cooling_delta,
            "fan_cooling_score": fan_cooling_score,
            "heatpipe_direction_drop": heatpipe_direction_drop,
            "heatpipe_direction_score": heatpipe_direction_score,
            "physics_score": physics_score,
        }

    @staticmethod
    def _estimate_keyboard_effective_coverage(matrix: np.ndarray, ambient_temperature: float) -> float:
        """Estimate visible warm coverage over keyboard ROI from final matrix."""
        h, w = matrix.shape
        x0 = int(round(w * 0.08))
        x1 = int(round(w * 0.92))
        y0 = int(round(h * 0.22))
        y1 = int(round(h * 0.80))
        if x1 <= x0 or y1 <= y0:
            return 0.0

        roi = matrix[y0:y1, x0:x1].astype(np.float32)
        warm_threshold = float(ambient_temperature + 2.6)
        ratio = float(np.mean(roi >= warm_threshold))
        return float(np.clip(ratio, 0.0, 1.0))


def _worker_generate_sample(task: tuple[int, GeneratorConfig]) -> GeneratedSample:
    """Multiprocessing worker entry point for sample generation."""
    idx, cfg = task
    worker = ThermalDatasetGenerator(cfg)
    return worker.generate_sample(index=idx, seed_offset=idx * 97)
