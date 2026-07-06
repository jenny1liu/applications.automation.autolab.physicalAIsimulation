"""CLI entrypoint for synthetic thermal dataset generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import yaml

from .config import GeneratorConfig
from .generator import ThermalDatasetGenerator


def build_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser for dataset generation."""
    parser = argparse.ArgumentParser(description="Synthetic thermal dataset generator")
    parser.add_argument("--size", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--image-size", type=int, default=640, choices=[320, 640, 1024], help="Output image size")
    parser.add_argument(
        "--sensor-size",
        type=str,
        default="",
        help="Final output sensor resolution as WIDTHxHEIGHT (e.g., 640x512, 320x240)",
    )
    parser.add_argument("--workers", type=int, default=0, help="Worker process count (0 = auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug visualization")
    parser.add_argument("--randomize-colormap", action="store_true", help="Enable per-sample random colormap")
    parser.add_argument(
        "--fixed-colormap",
        type=str,
        default="INFERNO",
        choices=["JET", "HOT", "INFERNO", "MAGMA", "TURBO"],
        help="Fixed colormap when randomization is disabled",
    )
    parser.add_argument(
        "--physics-threshold",
        type=float,
        default=0.72,
        help="Minimum physics score required per sample",
    )
    parser.add_argument(
        "--physics-max-attempts",
        type=int,
        default=8,
        help="Maximum regeneration attempts when sample physics score is below threshold",
    )
    parser.add_argument(
        "--disable-physics-gate",
        action="store_true",
        help="Disable physics score gating and accept first generated sample",
    )
    parser.add_argument(
        "--hybrid-reference-dir",
        type=Path,
        default=Path(""),
        help="Directory containing real IR reference images for hybrid calibration",
    )
    parser.add_argument(
        "--hybrid-weight",
        type=float,
        default=0.35,
        help="Blend weight for reference-based hybrid calibration (0-1)",
    )
    parser.add_argument(
        "--enable-hybrid",
        action="store_true",
        help="Enable hybrid strategy: physical simulation + reference calibration",
    )
    parser.add_argument(
        "--auto-hybrid-weight",
        action="store_true",
        help="Automatically search for the best hybrid weight before generation",
    )
    parser.add_argument(
        "--hybrid-weight-candidates",
        type=str,
        default="0.20,0.35,0.50,0.65",
        help="Comma-separated candidate weights for auto search",
    )
    parser.add_argument(
        "--hybrid-probe-samples",
        type=int,
        default=4,
        help="Probe sample count per candidate during auto search",
    )
    parser.add_argument(
        "--save-hybrid-preset",
        type=Path,
        default=Path("thermal_dataset_generator/presets/auto_hybrid_weight.yaml"),
        help="Path to save the selected hybrid preset YAML",
    )
    parser.add_argument(
        "--load-hybrid-preset",
        type=Path,
        default=Path(""),
        help="Load hybrid configuration from a preset YAML before generation",
    )
    parser.add_argument(
        "--held-out-power-classes",
        type=str,
        default="gaming",
        help="Comma-separated held-out power classes used for split manifest export",
    )
    parser.add_argument(
        "--disable-seen-heldout-split",
        action="store_true",
        help="Disable seen-vs-held-out split manifest export",
    )
    return parser


def _load_hybrid_preset(path: Path) -> dict[str, float | str | bool]:
    """Load hybrid preset yaml if available, returning empty dict on failure."""
    if not path or str(path).strip() == "":
        return {}
    if not path.exists() or not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        return {}
    return {}


def _save_hybrid_preset(path: Path, config: GeneratorConfig) -> None:
    """Persist selected hybrid settings to YAML preset file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "enable_hybrid_strategy": bool(config.enable_hybrid_strategy),
        "hybrid_reference_dir": str(config.hybrid_reference_dir),
        "hybrid_reference_weight": float(config.hybrid_reference_weight),
        "generated_by": "auto_hybrid_weight_search",
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def _parse_weight_candidates(raw: str) -> list[float]:
    """Parse comma-separated candidate weights and clamp into [0, 1]."""
    values: list[float] = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        try:
            values.append(float(t))
        except ValueError:
            continue
    if not values:
        values = [0.35]
    uniq_sorted = sorted({float(max(0.0, min(1.0, v))) for v in values})
    return uniq_sorted


def _parse_sensor_size(raw: str) -> tuple[int, int] | None:
    """Parse sensor size string in WIDTHxHEIGHT format."""
    text = str(raw).strip().lower()
    if text == "":
        return None
    sep = "x" if "x" in text else ("*" if "*" in text else None)
    if sep is None:
        raise ValueError("--sensor-size must be WIDTHxHEIGHT, e.g. 640x512")
    parts = text.split(sep)
    if len(parts) != 2:
        raise ValueError("--sensor-size must be WIDTHxHEIGHT, e.g. 640x512")
    try:
        w = int(parts[0].strip())
        h = int(parts[1].strip())
    except ValueError as exc:
        raise ValueError("--sensor-size must contain integer width and height") from exc
    if w <= 0 or h <= 0:
        raise ValueError("--sensor-size width and height must be > 0")
    return (w, h)


def _parse_power_classes(raw: str, valid: tuple[str, ...]) -> tuple[str, ...]:
    """Parse comma-separated power classes and keep valid unique values in input order."""
    tokens = [t.strip() for t in str(raw).split(",") if t.strip()]
    if not tokens:
        return ("gaming",)
    seen: set[str] = set()
    parsed: list[str] = []
    for token in tokens:
        if token in valid and token not in seen:
            seen.add(token)
            parsed.append(token)
    return tuple(parsed or ["gaming"])


def _choose_best_hybrid_weight(base_config: GeneratorConfig, candidates: Iterable[float], probe_samples: int) -> float:
    """Evaluate candidate hybrid weights and return the lowest-distance option."""
    sample_count = max(1, int(probe_samples))
    best_weight = float(base_config.hybrid_reference_weight)
    best_distance = float("inf")

    for weight in candidates:
        cfg_dict = base_config.to_dict()
        cfg_dict["hybrid_reference_weight"] = float(weight)
        cfg_dict["dataset_size"] = sample_count
        cfg = GeneratorConfig.from_dict(cfg_dict)
        generator = ThermalDatasetGenerator(cfg)

        if not generator.hybrid_available():
            continue

        distances: list[float] = []
        for i in range(sample_count):
            sample = generator.generate_sample(index=i + 1, seed_offset=77_000 + i)
            distances.append(generator.hybrid_reference_distance(sample.matrix))

        mean_distance = float(sum(distances) / len(distances)) if distances else float("inf")
        print(f"[hybrid-search] weight={weight:.3f}, distance={mean_distance:.6f}")

        if mean_distance < best_distance:
            best_distance = mean_distance
            best_weight = float(weight)

    return best_weight


def main() -> None:
    """Parse arguments and trigger dataset generation."""
    args = build_parser().parse_args()
    sensor_output_size = _parse_sensor_size(args.sensor_size)
    config = GeneratorConfig(
        image_size=args.image_size,
        sensor_output_size=sensor_output_size,
        dataset_size=args.size,
        workers=args.workers,
        random_seed=args.seed,
        output_dir=args.output,
        debug_mode=args.debug,
        randomize_colormap=args.randomize_colormap,
        fixed_colormap=args.fixed_colormap,
        enable_physics_gate=not args.disable_physics_gate,
        physics_score_threshold=args.physics_threshold,
        physics_max_attempts=args.physics_max_attempts,
        enable_hybrid_strategy=args.enable_hybrid,
        hybrid_reference_dir=args.hybrid_reference_dir,
        hybrid_reference_weight=args.hybrid_weight,
        export_seen_heldout_split=not args.disable_seen_heldout_split,
        held_out_power_classes=_parse_power_classes(args.held_out_power_classes, ("thin", "mainstream", "gaming")),
    )

    preset_values = _load_hybrid_preset(args.load_hybrid_preset)
    if preset_values:
        config.enable_hybrid_strategy = bool(preset_values.get("enable_hybrid_strategy", config.enable_hybrid_strategy))
        preset_ref_dir = preset_values.get("hybrid_reference_dir")
        if isinstance(preset_ref_dir, str) and preset_ref_dir.strip():
            config.hybrid_reference_dir = Path(preset_ref_dir)
        preset_weight = preset_values.get("hybrid_reference_weight")
        if isinstance(preset_weight, (int, float)):
            config.hybrid_reference_weight = float(preset_weight)

    if args.auto_hybrid_weight and config.enable_hybrid_strategy:
        candidates = _parse_weight_candidates(args.hybrid_weight_candidates)
        chosen = _choose_best_hybrid_weight(config, candidates, args.hybrid_probe_samples)
        config.hybrid_reference_weight = float(chosen)
        print(f"[hybrid-search] selected weight={chosen:.3f}")
        _save_hybrid_preset(args.save_hybrid_preset, config)
        print(f"[hybrid-search] preset saved to {args.save_hybrid_preset}")

    generator = ThermalDatasetGenerator(config)
    generator.generate_dataset()


if __name__ == "__main__":
    main()
