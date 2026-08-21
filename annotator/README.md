# Marker-Correspondence Matcher Benchmark

Tools to benchmark three ways of matching a manually-annotated marker between a
reference image (`ce`) and a target image (`new`), and to measure accuracy,
speed, CPU and memory on **CPU / GPU / NPU**.

- `extract_pairs.py` — decode an annotator session JSON into `ce/`, `new/`, `points.csv`.
- `benchmark_matchers.py` — run/compare the matchers and write a results table.

The three matchers:

| Method | Stack | Runs on |
|--------|-------|---------|
| `sift` | pure OpenCV (SIFT + FLANN + RANSAC) | CPU only |
| `lightglue` | PyTorch (DISK features + LightGlue) | CPU or CUDA GPU |
| `openvino` | PyTorch + OpenVINO (DISK U-Net via OpenVINO, LightGlue on CPU) | CPU / Intel GPU / NPU / AUTO |

> This machine is only for validating the **program logic**. The benchmark is
> meant to be ported to a machine with better CPU/GPU/NPU to get real numbers.

---

## 1. Quick start

```powershell
# 1) turn an annotator session JSON into benchmark data
python extract_pairs.py <session>.json --out extracted

# 2) run all three matchers and print + save a comparison table
python benchmark_matchers.py --data extracted
# -> extracted/bench_out/comparison.csv
```

Each method is run in its **own subprocess** so peak memory / CPU are clean.
Results per method: median/mean position error (px), success rate
(<3px / <5px / <10px), ms per image, CPU-seconds, CPU%, peak memory (MB).

---

## 2. Parameter tuning directions and their effect

All flags of `benchmark_matchers.py` (defaults in brackets):

| Flag | Default | What it does | Raise it → | Lower it → |
|------|---------|--------------|------------|------------|
| `--max-dim` | 1280 | longest side images are downscaled to (SIFT/LightGlue) | more detail, **more accurate**, **slower**, more memory | faster, less accurate |
| `--max-features` | 4000 | SIFT keypoint cap | more matches, slower | fewer matches, faster |
| `--lg-features` | 1024 | DISK keypoint cap (LightGlue & OpenVINO) | more matches, **slower**, more memory | fewer matches, faster |
| `--min-matches` | 8 | min matches required to fit a homography | **stricter → fewer but cleaner** results (higher accuracy, more "failed" pairs) | looser → more pairs matched but **more wrong matches slip through → worse accuracy** |
| `--ratio` | 0.75 | SIFT Lowe ratio test | looser (0.8+) → more matches, more outliers | stricter (0.7-) → fewer, cleaner matches |
| `--ransac-thresh` | 5.0 | RANSAC reprojection threshold (px) for `findHomography` | looser → tolerates noisier matches, more outliers survive | **stricter (3.0)** → cleaner homography, higher accuracy, some pairs may fail |
| `--device` | cpu | torch device for LightGlue | `cuda` → much faster LightGlue (needs CUDA torch build) | — |
| `--ov-device` | AUTO | OpenVINO device for `openvino` method | `CPU`/`GPU`/`NPU` to pin a device | — |
| `--ov-devices` | (off) | sweep OpenVINO across devices, e.g. `CPU,GPU,NPU` | benchmarks each device in turn | — |
| `--ov-size` | 1024 | square input size for the OpenVINO DISK U-Net (mult. of 16) | larger → **more accurate**, **slower**, **much more memory** (1024→1280 raised peak RAM ~3.9GB→~5.9GB) | smaller → faster, less accurate |
| `--methods` | sift,lightglue,openvino | which methods to compare | — | subset, e.g. `sift,openvino` |

Also **hard-coded** in `benchmark_matchers.py` (change in code if needed):
- (none currently — the RANSAC threshold is now the `--ransac-thresh` flag.)

### Lesson from our runs (see `benchmark_runs/`)
Lowering `--min-matches` (8→6) together with raising features **increased the
number of matched pairs but made accuracy worse on every dataset** — more wrong
matches survived RANSAC. Prefer **keeping/raising `--min-matches`** and, if you
want more candidates, raise only `--lg-features` while keeping `--min-matches`
high. Accuracy-first tuning: higher `--max-dim` / `--ov-size`, stricter
`--min-matches` and tighter RANSAC threshold.

---

## 3. Porting to a new machine (CPU / GPU / NPU)

### 3.1 Python packages
Tested versions (Python 3.14, Windows x64):

```
numpy==2.4.6
opencv-python==5.0.0.93
torch==2.13.0
torchvision==0.28.0
kornia==0.8.3            # provides DISK + LightGlue
openvino==2026.2.1
onnx==1.22.0            # only needed to (re)export the DISK U-Net to ONNX
onnxscript==0.7.1       # required by torch.onnx.export on torch 2.x
psutil==7.2.2           # resource monitor (memory / CPU)
```

Minimum install:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install numpy opencv-python torch torchvision kornia openvino onnx onnxscript psutil
```

### 3.2 Enabling each device

| Target | What to install / do |
|--------|----------------------|
| **CPU** | works out of the box (the default torch + OpenVINO CPU plugin). |
| **CUDA GPU** (NVIDIA) | install the CUDA build of torch (`pip install torch --index-url https://download.pytorch.org/whl/cuXXX`). Run LightGlue with `--device cuda`. Note: OpenVINO GPU = Intel GPU, not NVIDIA. |
| **Intel GPU** (iGPU / Arc) | OpenVINO GPU plugin ships with `openvino`; needs an up-to-date Intel graphics driver. Run `--ov-device GPU`. |
| **Intel NPU** | needs the OpenVINO **NPU plugin** + Intel NPU driver installed on the OS. Run `--ov-device NPU`. The code already pastes each image onto a fixed square canvas because NPU requires a **static input shape**. |

Check what OpenVINO sees on the new machine:

```powershell
python -c "import openvino as ov; print(ov.Core().available_devices)"
# e.g. ['CPU', 'GPU', 'NPU']
```

Then sweep everything available in one run:

```powershell
python benchmark_matchers.py --data extracted --methods openvino --ov-devices CPU,GPU,NPU
# unavailable devices are skipped automatically
```

---

## 4. Offline machine (no internet) — prerequisites

On first use the code downloads model weights from the internet (via the Intel
proxy). On an **air-gapped / no-network** machine you must pre-stage everything.

### 4.1 Pre-download the Python packages (on a machine WITH network)
```powershell
# make a wheelhouse you can copy to the offline machine
mkdir wheelhouse
pip download -d wheelhouse numpy opencv-python torch torchvision kornia openvino onnx onnxscript psutil
```
Then on the offline machine:
```powershell
pip install --no-index --find-links wheelhouse numpy opencv-python torch torchvision kornia openvino onnx onnxscript psutil
```
> `torch`/`torchvision` are large and platform-specific — download the wheels
> that match the offline machine's OS, Python version and (if GPU) CUDA build.

### 4.2 Pre-stage the model weights (torch hub cache)
`kornia`'s DISK + LightGlue download these on first run into the torch hub cache.
Copy them from an online machine to the **same path** on the offline machine:

```
%USERPROFILE%\.cache\torch\hub\checkpoints\
    depth-save.pth                        (~4.2 MB)  DISK weights
    disk_lightglue_v0-1_arxiv-...pth      (~45 MB)   LightGlue (DISK) matcher
    superpoint_lightglue_v0-1_arxiv-...pth(~45 MB)   (only if you use superpoint)
```
On Linux/macOS the path is `~/.cache/torch/hub/checkpoints/`.

With these present, `KF.DISK.from_pretrained("depth")` and
`KF.LightGlueMatcher("disk")` load from disk and **need no network**.

### 4.3 Pre-export the OpenVINO ONNX model (avoids needing onnx/onnxscript online)
The `openvino` method exports the DISK U-Net to ONNX on first run, once per
`--ov-size`. You can pre-generate these on an online machine and copy the
`ov_models/` folder over:

```
annotator/ov_models/
    disk_unet_1024.onnx      + disk_unet_1024.onnx.data      (~4.2 MB)
    disk_unet_1280.onnx      + disk_unet_1280.onnx.data      (~4.2 MB)
```
Copy the `.onnx` **and** its `.onnx.data` sidecar. If the ONNX for the requested
`--ov-size` already exists, no export happens, so `onnx`/`onnxscript` and network
are not needed at run time. (If you only ever run pre-exported sizes you can even
skip installing `onnx`/`onnxscript` on the offline machine.)

### 4.4 Windows console note
ONNX export prints a unicode character that crashes the Windows cp1252 console.
The scripts already set `PYTHONUTF8=1` in the subprocess env; if you export
manually, run with `$env:PYTHONUTF8=1` first.

---

## 5. Offline checklist (summary)

- [ ] Python venv created, packages installed from `wheelhouse` (§4.1)
- [ ] `~/.cache/torch/hub/checkpoints/` populated with DISK + LightGlue weights (§4.2)
- [ ] `annotator/ov_models/` populated with `disk_unet_<size>.onnx` (+ `.data`) for every `--ov-size` you will run (§4.3)
- [ ] Device drivers present for the target: Intel GPU driver / Intel NPU driver / CUDA (§3.2)
- [ ] `python -c "import openvino as ov; print(ov.Core().available_devices)"` shows the expected devices
