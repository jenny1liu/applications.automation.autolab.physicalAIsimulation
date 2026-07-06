# Thermal Dataset Generator

Modular Python project for generating synthetic keyboard-deck thermal datasets for Physical AI model training.

## Features

- Physics-based thermal simulation for hidden CPU localization (not hottest-pixel detection).
- Surface-temperature calibration ranges aligned with real laptop IR captures.
- Power-class dependent keyboard warm-plateau coverage (`thin/mainstream/gaming`).
- Optional screen/hinge framing for open-lid viewpoints:
  - Screen visible in top region for configurable sample fraction.
  - Hinge structure is explicitly modeled and exported as a region mask.
- Three-tier palm/touchpad contrast:
  - Keyboard warm plateau.
  - Palm rest baseline around ambient +2C to +5C.
  - Touchpad near ambient as a distinct cooler rectangle.
- Keyboard key-grid texturing with per-key block flattening and crisp cooler gaps.
- Physics gate with retry sampling based on continuity/plausibility metrics.
- Optional hybrid reference calibration against real IR images.
- Exports images, YOLO labels, masks, temperature maps, and JSON metadata.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Project Structure

```text
thermal_dataset_generator/
├── config.py
├── main.py
├── generator.py
├── thermal.py
├── components.py
├── groundtruth.py
├── augmentation.py
├── visualization.py
├── evaluate_sim_to_real.py
├── labels.py
├── utils.py
├── requirements.txt
├── README.md
└── output/
   ├── images/
   ├── labels/
   ├── metadata/
   ├── masks/
   └── temperature/
```

## Sensor Resolution and Aspect Ratio

- Simulation base resolution is square via `image_size` (`320`, `640`, `1024`).
- Final exported output can be resized to a real thermal camera aspect ratio using `sensor_output_size`.
- CLI option: `--sensor-size WIDTHxHEIGHT`.

Examples:

```bash
python -m thermal_dataset_generator.main --size 1000 --image-size 640 --sensor-size 640x512
python -m thermal_dataset_generator.main --size 1000 --image-size 640 --sensor-size 320x240
```

This reduces sim-to-real gap by matching deployment sensor geometry.

## Thermal Calibration Notes

- This generator models **chassis surface temperature**, not die/junction temperature.
- Typical ranges used:
  - Ambient/background: `22C` to `30C`
  - CPU surface: `32C` to `63C`
  - GPU surface: `30C` to `60C`
  - Fan/vent surface: `28C` to `48C`
  - Frame maxima usually below `~65C`

## Emissivity Note (Important)

Some visual contrast is intentionally modeled as **apparent-temperature** differences due to material emissivity (not only conductive heat):

- Touchpad vs palm rest.
- Keyboard keycaps vs key-gap boundaries.

These effects are deliberately preserved in code comments and should not be removed as redundant.

## Ground-Truth Target Point Definition

Ground truth is not the raw hottest pixel. The target point is selected in `groundtruth.py` as:

- `keyboard_mask`
- excluding dilated vent and hinge regions
- requiring `source_attribution_fraction > SOURCE_ATTRIBUTION_THRESHOLD`

Fallback tiers are logged per sample:

- Tier 0: strict attribution rule above
- Tier 1: keyboard-only (still excluding vent/hinge)
- Tier 2: CPU mask fallback

Metadata includes `target_point`, `target_fallback_tier`, `excluded_hottest_point`, `excluded_hottest_region`, `runner_up_point`, and `cpu_min_distance_to_hinge_px`.

## No-Cheat Inference / Benchmark Architecture

The UI benchmark uses a no-cheat focus policy: no privileged simulator vent/hinge masks are used at inference time.

- Focus area is a generic geometric keyboard ROI.
- Exclusion of vent/hinge/trap must be learned from image appearance by the model itself.
- This keeps benchmark behavior representative of deployment where structural masks are unknown.

## Sim-to-Real Validation Loop

Run a synthetic-trained model on real annotated thermal frames using:

```bash
python -m thermal_dataset_generator.evaluate_sim_to_real --manifest path/to/real_manifest.json --detector opencv
```

Manifest format (JSON list):

- `image`: path to real frame (`.npy` preferred, grayscale image supported)
- `target_point`: `[x, y]`
- optional `case_type`: `normal` or reversal tags (`reversal`, `vent_reversal`, `trap_reversal`)

The script reports mean/median/p95/p99 pixel error for all, reversal, and normal groups.

## Split Strategy (Seen vs Held-Out)

Do not rely only on random shuffle. Keep at least one power-class configuration held out from training and report:

- Seen power-class metrics
- Held-out power-class metrics

The generator now exports deterministic split manifests under `output/splits/`:

- `all_manifest.json`
- `seen_manifest.json`
- `held_out_manifest.json`
- `split_summary.json`

Default held-out class is `gaming`. Override via CLI:

```bash
python -m thermal_dataset_generator.main --size 1000 --held-out-power-classes gaming
python -m thermal_dataset_generator.main --size 1000 --held-out-power-classes thin,mainstream
```

This better reflects cross-laptop generalization at deployment.

## Metadata Fields (Key)

Per-sample metadata includes:

- `power_class`
- `keyboard_plateau_target_coverage`
- `keyboard_plateau_effective_coverage`
- `cpu_center_pixel`, `cpu_bbox`, `cpu_power`
- `gpu_enabled`, `gpu_center`
- `ambient_temperature`, `fan_speed`
- `screen_visible`, `camera_tilt_deg`, `screen_region_fraction`
- `hinge_line`
- `simulation_resolution`, `output_resolution`, `sensor_output_size`
- `physics_metrics`, `physics_score`

## Generate Dataset

```bash
python -m thermal_dataset_generator.main --size 1000 --image-size 640 --workers 8
```

Enable physics gate tuning:

```bash
python -m thermal_dataset_generator.main --size 1000 --physics-threshold 0.80 --physics-max-attempts 12
```

Enable hybrid calibration:

```bash
python -m thermal_dataset_generator.main --size 1000 --enable-hybrid --hybrid-reference-dir "thermal_dataset_generator/reference_ir" --hybrid-weight 0.35
```

## Debug

```bash
python -m thermal_dataset_generator.main --size 5 --debug
```

## Output Format

- RGB image: `output/images/000001.png`
- YOLO detection label: `output/labels/000001.txt`
- CPU mask: `output/masks/000001.png`
- Temperature matrix: `output/temperature/000001.npy`
- Metadata JSON: `output/metadata/000001.json`
