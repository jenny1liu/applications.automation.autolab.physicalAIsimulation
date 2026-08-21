"""Benchmark marker-correspondence matchers on the annotator data produced by
extract_pairs.py.

Three configurations are compared:
  1. sift      -> pure OpenCV (SIFT + FLANN + RANSAC), CPU only
  2. lightglue -> PyTorch only (DISK features + LightGlue), no OpenVINO
  3. openvino  -> PyTorch + OpenVINO (DISK U-Net runs through OpenVINO on the
                  chosen device: CPU / GPU / NPU / AUTO), LightGlue match on CPU

Metrics per method:
  1. Position error (pixels)      -> mean / median
  2. Success rate                 -> % pairs with error < 3px and < 5px
  3. Processing speed             -> ms per pair (image)
  4. CPU usage                    -> CPU-seconds and CPU% (relative to one core)
  5. Peak memory                  -> peak process RSS in MB

Each method is measured in an isolated subprocess so peak memory / CPU are clean.
DISK/LightGlue weights are downloaded once via the Intel proxy and then cached,
so no network is needed here.

Device selection (portable across machines):
  --device      torch device for LightGlue features (cpu / cuda)
  --ov-device   OpenVINO device for the 'openvino' method: AUTO, CPU, GPU, NPU,
                or a specific one like GPU.0 / GPU.1
  --ov-devices  comma list to sweep the 'openvino' method across devices, e.g.
                "CPU,GPU,NPU" (unavailable devices are skipped automatically)

Usage:
  # run all three and print comparison (openvino uses --ov-device, default AUTO)
  python benchmark_matchers.py --data extracted

  # only some methods
  python benchmark_matchers.py --data extracted --methods sift,openvino

  # sweep OpenVINO across CPU/GPU/NPU on a new machine
  python benchmark_matchers.py --data extracted --methods openvino --ov-devices CPU,GPU,NPU

  # run a single method (used internally by compare, prints JSON)
  python benchmark_matchers.py --data extracted --method openvino --ov-device GPU --out-json ov.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_points(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(
                {
                    "ceFile": r["ceFile"],
                    "newFile": r["newFile"],
                    "ce": (float(r["ce_x"]), float(r["ce_y"])),
                    "new": (float(r["new_x"]), float(r["new_y"])),
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Resource monitor: peak RSS + CPU time over a code block
# --------------------------------------------------------------------------- #
class ResourceMonitor:
    def __init__(self, interval: float = 0.05):
        import psutil

        self.proc = psutil.Process()
        self.interval = interval
        self.peak_rss = 0
        self._stop = threading.Event()

    def _sample(self):
        while not self._stop.is_set():
            rss = self.proc.memory_info().rss
            if rss > self.peak_rss:
                self.peak_rss = rss
            self._stop.wait(self.interval)

    def __enter__(self):
        self.peak_rss = self.proc.memory_info().rss
        self.cpu0 = self.proc.cpu_times()
        self.t = threading.Thread(target=self._sample, daemon=True)
        self.t.start()
        self.wall0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.wall = time.perf_counter() - self.wall0
        c1 = self.proc.cpu_times()
        self._stop.set()
        self.t.join(timeout=1.0)
        self.cpu_seconds = (c1.user - self.cpu0.user) + (c1.system - self.cpu0.system)
        self.peak_mb = self.peak_rss / 1e6
        # >100% means more than one core was busy (CPU-seconds can exceed wall time).
        self.cpu_percent = 100.0 * self.cpu_seconds / self.wall if self.wall > 0 else 0.0


# --------------------------------------------------------------------------- #
# SIFT matcher
# --------------------------------------------------------------------------- #
def run_sift(points, ce_dir, new_dir, max_dim, max_features, min_matches, ratio, ransac_thresh):
    import cv2  # OpenCV: `pip install opencv-python`, imported as cv2

    # SIFT_create = classic hand-crafted feature detector+descriptor (no ML weights, no GPU).
    detector = cv2.SIFT_create(nfeatures=max_features)
    # FLANN = fast approximate nearest-neighbour matcher for the descriptor vectors.
    # algorithm=1 is KD-tree (right choice for SIFT's float descriptors).
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
    cache: dict = {}

    def feats(path):
        if path in cache:
            return cache[path]
        img = cv2.imread(str(path))  # cv2 loads images as BGR (not RGB!), shape (H,W,3)
        if img is None:
            cache[path] = None
            return None
        h, w = img.shape[:2]
        # Downscale large phone photos so detection/matching stays fast; never upscale.
        s = min(1.0, max_dim / max(h, w))
        small = cv2.resize(img, (int(w * s), int(h * s))) if s < 1.0 else img
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)  # SIFT works on a single-channel gray image
        # STEP 1 (feature extraction): keypoints (x,y) + 128-d descriptor per keypoint.
        kps, des = detector.detectAndCompute(gray, None)
        # Cache per unique image: the same photo appears in many pairs.
        cache[path] = (kps, des, s)
        return cache[path]

    t0 = time.perf_counter()
    results = []
    n = len(points)
    for i, p in enumerate(points, 1):
        f1 = feats(ce_dir / p["ceFile"])
        f2 = feats(new_dir / p["newFile"])
        if f1 is None or f2 is None or f1[1] is None or f2[1] is None:
            results.append({"error_px": None})
            continue
        kp1, des1, s1 = f1
        kp2, des2, s2 = f2
        if len(des1) < 2 or len(des2) < 2:
            results.append({"error_px": None})
            continue
        # STEP 2 (matching): for each descriptor in image1, find its 2 closest in image2.
        knn = flann.knnMatch(des1, des2, k=2)
        # Lowe ratio test: keep a match only if the best is clearly better than the 2nd best.
        good = [m for pr in knn if len(pr) == 2 for m, n in [pr] if m.distance < ratio * n.distance]
        if len(good) < min_matches:
            results.append({"error_px": None})
            continue
        # Divide by s to map keypoints from the downscaled image back to original pixels.
        # .pt is the (x,y) of a cv2.KeyPoint; queryIdx/trainIdx index image1/image2 keypoints.
        src = np.float32([kp1[m.queryIdx].pt for m in good]) / s1
        dst = np.float32([kp2[m.trainIdx].pt for m in good]) / s2
        # STEP 3 (RANSAC -> H): fit a 3x3 homography, rejecting outlier matches beyond ransac_thresh px.
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
        if H is None:
            results.append({"error_px": None})
            continue
        # STEP 4 (project): push the manual ce marker through H; error = distance to manual new marker.
        proj = cv2.perspectiveTransform(np.array([[p["ce"]]], np.float32), H)[0, 0]
        err = float(np.hypot(proj[0] - p["new"][0], proj[1] - p["new"][1]))
        results.append({"error_px": err})
        print(f"[sift {i}/{n}] {p['ceFile']}->{p['newFile']} err={err:.1f}", file=sys.stderr, flush=True)
    proc_ms = (time.perf_counter() - t0) * 1000.0
    return results, proc_ms


# --------------------------------------------------------------------------- #
# LightGlue (DISK features) matcher
# --------------------------------------------------------------------------- #
def run_lightglue(points, ce_dir, new_dir, max_dim, max_features, min_matches, device, ransac_thresh):
    import cv2
    import torch  # PyTorch runs the neural nets (CPU or CUDA GPU)
    import kornia.feature as KF  # kornia = computer-vision ops on torch tensors

    dev = torch.device(device)  # "cpu" or "cuda"; every tensor/model must live on the same device
    # DISK = learned feature detector (a CNN); weights are downloaded once, then cached on disk.
    disk = KF.DISK.from_pretrained("depth").to(dev).eval()
    # LightGlue = learned matcher (a transformer) that pairs DISK descriptors between two images.
    matcher = KF.LightGlueMatcher("disk").to(dev).eval()  # .eval() = inference mode, no training
    cache: dict = {}

    @torch.inference_mode()
    def feats(path):
        if path in cache:
            return cache[path]
        img = cv2.imread(str(path))
        if img is None:
            cache[path] = None
            return None
        h, w = img.shape[:2]
        s = min(1.0, max_dim / max(h, w))
        small = cv2.resize(img, (int(w * s), int(h * s))) if s < 1.0 else img
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)  # nets want RGB; cv2 gave us BGR
        # DISK expects a float RGB tensor in [0,1], shape (1,3,H,W).
        # permute HWC->CHW, [None] adds the batch dim, /255 scales 0..255 to 0..1.
        t = torch.from_numpy(rgb).float().permute(2, 0, 1)[None].to(dev) / 255.0
        # STEP 1 (feature extraction): DISK returns keypoints + descriptors for up to n points.
        f = disk(t, n=max_features, pad_if_not_divisible=True)[0]
        cache[path] = (f.keypoints, f.descriptors, s, small.shape[0], small.shape[1])
        return cache[path]

    t0 = time.perf_counter()
    results = []
    n = len(points)
    for i, p in enumerate(points, 1):
        f1 = feats(ce_dir / p["ceFile"])
        f2 = feats(new_dir / p["newFile"])
        if f1 is None or f2 is None:
            results.append({"error_px": None})
            continue
        kp1, des1, s1, h1, w1 = f1
        kp2, des2, s2, h2, w2 = f2
        if kp1.shape[0] < 2 or kp2.shape[0] < 2:
            results.append({"error_px": None})
            continue
        with torch.inference_mode():
            # LightGlue matches on local affine frames (LAFs); it also needs each image's size.
            laf1 = KF.laf_from_center_scale_ori(kp1[None])
            laf2 = KF.laf_from_center_scale_ori(kp2[None])
            # STEP 2 (matching): idxs = pairs of matched keypoint indices (Nx2).
            _, idxs = matcher(des1, des2, laf1, laf2, hw1=(h1, w1), hw2=(h2, w2))
        idxs = idxs.cpu().numpy()  # move tensor off GPU and into a numpy array for OpenCV
        if len(idxs) < min_matches:
            results.append({"error_px": None})
            continue
        # idxs[:,0]/[:,1] are matched keypoint indices in image1/image2; map back to original px.
        src = kp1.cpu().numpy()[idxs[:, 0]] / s1
        dst = kp2.cpu().numpy()[idxs[:, 1]] / s2
        # STEP 3 (RANSAC -> H): same homography fit as SIFT; the matcher differs, this part is shared.
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
        if H is None:
            results.append({"error_px": None})
            continue
        # STEP 4 (project): map ce marker through H, compare to manual new marker.
        proj = cv2.perspectiveTransform(np.array([[p["ce"]]], np.float32), H)[0, 0]
        err = float(np.hypot(proj[0] - p["new"][0], proj[1] - p["new"][1]))
        results.append({"error_px": err})
        print(f"[lightglue {i}/{n}] {p['ceFile']}->{p['newFile']} err={err:.1f}", file=sys.stderr, flush=True)
    proc_ms = (time.perf_counter() - t0) * 1000.0
    return results, proc_ms


# --------------------------------------------------------------------------- #
# OpenVINO (DISK U-Net via OpenVINO) matcher
# --------------------------------------------------------------------------- #
def _export_disk_unet_onnx(disk, ov_size: int, onnx_path: Path) -> None:
    """Export the DISK convolutional U-Net to ONNX at a fixed square size."""
    import torch

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    # ONNX = a portable model file format OpenVINO can load; export needs a sample input
    # so it can trace the network's shapes (values don't matter, so use random noise).
    dummy = torch.randn(1, 3, ov_size, ov_size)
    with torch.inference_mode():
        # verbose=False avoids a checkmark emoji that crashes the Windows cp1252 console.
        torch.onnx.export(
            disk.unet, dummy, str(onnx_path),
            input_names=["image"], output_names=["unet_out"],
            opset_version=18, verbose=False,
        )
    # Some torch/onnx versions write the weights into a companion "<name>.onnx.data"
    # file (external data). That sidecar is easily lost when copying/bundling the exe
    # (OpenVINO then fails with "externally stored data ... Invalid usage"). Collapse
    # everything into a single self-contained .onnx and drop the sidecar.
    try:
        import onnx
        model = onnx.load(str(onnx_path))  # resolves external data next to the file
        onnx.save_model(model, str(onnx_path), save_as_external_data=False)
        sidecar = onnx_path.with_name(onnx_path.name + ".data")
        if sidecar.exists():
            sidecar.unlink()
    except Exception as exc:
        print(f"[openvino] WARN: could not inline ONNX weights: {exc}", file=sys.stderr, flush=True)
    print(f"[openvino] exported DISK U-Net ONNX -> {onnx_path}", file=sys.stderr, flush=True)


def run_openvino(points, ce_dir, new_dir, min_matches, ov_device, ov_size, max_features, onnx_dir, ransac_thresh):
    import cv2
    import torch
    import kornia.feature as KF
    import openvino as ov  # Intel OpenVINO: runs the model on CPU / integrated GPU / NPU
    from kornia.feature.disk.detector import heatmap_to_keypoints

    if ov_size % 16 != 0:
        ov_size = ((ov_size + 15) // 16) * 16

    # torch DISK is used only to export the U-Net once and to reuse the CPU-side
    # keypoint detection + LightGlue matcher; the heavy conv runs through OpenVINO.
    disk = KF.DISK.from_pretrained("depth").eval()
    desc_dim = disk.desc_dim
    matcher = KF.LightGlueMatcher("disk").eval()

    # Export once and reuse the cached ONNX (filename encodes the input size).
    onnx_path = Path(onnx_dir) / f"disk_unet_{ov_size}.onnx"
    if not onnx_path.exists():
        _export_disk_unet_onnx(disk, ov_size, onnx_path)

    core = ov.Core()  # OpenVINO entry point: discovers devices and compiles models
    avail = core.available_devices
    print(f"[openvino] available devices: {avail}; requested: {ov_device}", file=sys.stderr, flush=True)
    # Compile the ONNX for the chosen device (CPU/GPU/NPU/AUTO); OpenVINO reads ONNX directly.
    compiled = core.compile_model(str(onnx_path), ov_device)
    out_port = compiled.output(0)  # handle to output #0; used to read the result tensor below
    cache: dict = {}

    def feats(path):
        if path in cache:
            return cache[path]
        img = cv2.imread(str(path))
        if img is None:
            cache[path] = None
            return None
        h, w = img.shape[:2]
        s = ov_size / max(h, w)
        w_r = min(int(round(w * s)), ov_size)
        h_r = min(int(round(h * s)), ov_size)
        small = cv2.resize(img, (w_r, h_r))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Paste onto a fixed ov_size x ov_size canvas: a static shape is required by NPU
        # (and faster on GPU); the image sits top-left, rest is zero padding.
        canvas = np.zeros((ov_size, ov_size, 3), np.float32)
        canvas[:h_r, :w_r] = rgb
        inp = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None])  # HWC->CHW + batch dim, C-contiguous for OV
        # STEP 1a (feature extraction): run the DISK U-Net on the OpenVINO device.
        # calling `compiled(inp)` runs inference; [out_port] pulls out the output numpy array.
        out = compiled(inp)[out_port]
        out_t = torch.from_numpy(np.ascontiguousarray(out))  # back to a torch tensor for post-processing
        # U-Net output channels: first desc_dim = dense descriptors, last = detection heatmap.
        # Crop away the padding before detecting keypoints so no spurious points appear.
        heatmaps = out_t[:, desc_dim:desc_dim + 1, :h_r, :w_r]
        descriptors = out_t[:, :desc_dim, :h_r, :w_r]
        with torch.inference_mode():
            # STEP 1b: same CPU post-processing as native DISK: NMS peaks -> sample their descriptors.
            kpl = heatmap_to_keypoints(heatmaps, n=max_features, window_size=5, score_threshold=0.0)
            feat = kpl[0].merge_with_descriptors(descriptors[0])
        cache[path] = (feat.keypoints, feat.descriptors, s, h_r, w_r)
        return cache[path]

    t0 = time.perf_counter()
    results = []
    n = len(points)
    for i, p in enumerate(points, 1):
        f1 = feats(ce_dir / p["ceFile"])
        f2 = feats(new_dir / p["newFile"])
        if f1 is None or f2 is None:
            results.append({"error_px": None})
            continue
        kp1, des1, s1, h1, w1 = f1
        kp2, des2, s2, h2, w2 = f2
        if kp1.shape[0] < 2 or kp2.shape[0] < 2:
            results.append({"error_px": None})
            continue
        with torch.inference_mode():
            laf1 = KF.laf_from_center_scale_ori(kp1[None].float())
            laf2 = KF.laf_from_center_scale_ori(kp2[None].float())
            # STEP 2 (matching): LightGlue still runs on CPU (only the U-Net went to OpenVINO).
            _, idxs = matcher(des1, des2, laf1, laf2, hw1=(h1, w1), hw2=(h2, w2))
        idxs = idxs.cpu().numpy()
        if len(idxs) < min_matches:
            results.append({"error_px": None})
            continue
        src = kp1.cpu().numpy()[idxs[:, 0]] / s1
        dst = kp2.cpu().numpy()[idxs[:, 1]] / s2
        # STEP 3 (RANSAC -> H): shared homography fit.
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
        if H is None:
            results.append({"error_px": None})
            continue
        # STEP 4 (project): map ce marker through H, compare to manual new marker.
        proj = cv2.perspectiveTransform(np.array([[p["ce"]]], np.float32), H)[0, 0]
        err = float(np.hypot(proj[0] - p["new"][0], proj[1] - p["new"][1]))
        results.append({"error_px": err})
        print(f"[openvino {i}/{n}] {p['ceFile']}->{p['newFile']} err={err:.1f}", file=sys.stderr, flush=True)
    proc_ms = (time.perf_counter() - t0) * 1000.0
    return results, proc_ms


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def summarize(method, results, proc_ms, mon) -> dict:
    # Pairs where matching failed have error_px=None and are excluded from error stats.
    errs = np.array([r["error_px"] for r in results if r["error_px"] is not None], dtype=float)
    n_total = len(results)
    n_ok = int(errs.size)
    lt3 = int((errs < 3).sum())
    lt5 = int((errs < 5).sum())
    lt10 = int((errs < 10).sum())
    return {
        "method": method,
        "n_total": n_total,
        "n_matched": n_ok,
        "err_mean_px": round(float(errs.mean()), 2) if n_ok else None,
        "err_median_px": round(float(np.median(errs)), 2) if n_ok else None,
        "succ_lt3_pct": round(100.0 * lt3 / n_total, 1),
        "succ_lt5_pct": round(100.0 * lt5 / n_total, 1),
        "succ_lt10_pct": round(100.0 * lt10 / n_total, 1),
        "succ_lt3_count": lt3,
        "succ_lt5_count": lt5,
        "succ_lt10_count": lt10,
        "ms_per_image": round(proc_ms / n_total, 1),
        "cpu_seconds": round(mon.cpu_seconds, 2),
        "cpu_percent": round(mon.cpu_percent, 1),
        "peak_mem_mb": round(mon.peak_mb, 1),
    }


def run_single(args) -> dict:
    points = load_points(args.data / "points.csv")
    ce_dir = args.data / "ce"
    new_dir = args.data / "new"
    onnx_dir = Path(__file__).resolve().parent / "ov_models"
    label = args.method
    # ResourceMonitor wraps only the matching work, so peak memory / CPU reflect this method.
    with ResourceMonitor() as mon:
        if args.method == "sift":
            results, proc_ms = run_sift(
                points, ce_dir, new_dir, args.max_dim, args.max_features, args.min_matches, args.ratio,
                args.ransac_thresh
            )
        elif args.method == "lightglue":
            results, proc_ms = run_lightglue(
                points, ce_dir, new_dir, args.max_dim, args.lg_features, args.min_matches, args.device,
                args.ransac_thresh
            )
        else:  # openvino
            label = f"openvino:{args.ov_device}"
            results, proc_ms = run_openvino(
                points, ce_dir, new_dir, args.min_matches, args.ov_device,
                args.ov_size, args.lg_features, onnx_dir, args.ransac_thresh
            )
    return summarize(label, results, proc_ms, mon)


# --------------------------------------------------------------------------- #
# Compare driver
# --------------------------------------------------------------------------- #
def _resolve_ov_devices(args) -> list[str]:
    """Which OpenVINO devices to run: --ov-devices sweep, else single --ov-device.
    Unavailable devices are dropped (AUTO is always kept)."""
    want = (
        [d.strip() for d in args.ov_devices.split(",") if d.strip()]
        if args.ov_devices else [args.ov_device]
    )
    try:
        import openvino as ov
        avail = set(ov.Core().available_devices)
    except Exception:
        avail = None
    out: list[str] = []
    for d in want:
        base = d.split(".")[0]
        if d == "AUTO" or avail is None or d in avail or base in avail:
            out.append(d)
        else:
            print(f"[skip] OpenVINO device '{d}' not available; have {sorted(avail)}",
                  file=sys.stderr, flush=True)
    return out or ["AUTO"]


def compare(args) -> None:
    out_dir = args.data / "bench_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    tasks: list[tuple[str, list[str], Path]] = []
    for m in methods:
        if m == "openvino":
            # Sweep produces one task (and one result column) per requested device.
            for dev in _resolve_ov_devices(args):
                safe = dev.replace(".", "_")
                tasks.append((m, ["--ov-device", dev], out_dir / f"openvino_{safe}.json"))
        else:
            tasks.append((m, [], out_dir / f"{m}.json"))

    metrics = []
    for method, extra, out_json in tasks:
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--data", str(args.data), "--method", method,
            "--max-dim", str(args.max_dim),
            "--max-features", str(args.max_features),
            "--lg-features", str(args.lg_features),
            "--min-matches", str(args.min_matches),
            "--ratio", str(args.ratio),
            "--ransac-thresh", str(args.ransac_thresh),
            "--device", args.device,
            "--ov-size", str(args.ov_size),
            "--out-json", str(out_json),
            *extra,
        ]
        tag = f"{method}{' ' + extra[1] if extra else ''}"
        print(f"\n=== Running {tag} in subprocess ===", flush=True)
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"  # avoid emoji-encoding crash during ONNX export on Windows
        # Weights download once through the Intel proxy, then come from cache.
        env.setdefault("HTTPS_PROXY", "http://proxy-png.intel.com:912")
        env.setdefault("HTTP_PROXY", "http://proxy-png.intel.com:912")
        subprocess.run(cmd, check=True, env=env)
        metrics.append(json.loads(out_json.read_text(encoding="utf-8")))

    _print_table(metrics)
    _write_csv(out_dir / "comparison.csv", metrics)
    print(f"\nSaved: {out_dir / 'comparison.csv'}")


def _print_table(metrics: list[dict]) -> None:
    rows = [
        ("Metric", *[m["method"] for m in metrics]),
        ("pairs matched", *[f"{m['n_matched']}/{m['n_total']}" for m in metrics]),
        ("err mean (px)", *[m["err_mean_px"] for m in metrics]),
        ("err median (px)", *[m["err_median_px"] for m in metrics]),
        ("success <3px", *[f"{m['succ_lt3_count']} ({m['succ_lt3_pct']}%)" for m in metrics]),
        ("success <5px", *[f"{m['succ_lt5_count']} ({m['succ_lt5_pct']}%)" for m in metrics]),
        ("success <10px", *[f"{m['succ_lt10_count']} ({m['succ_lt10_pct']}%)" for m in metrics]),
        ("speed (ms/image)", *[m["ms_per_image"] for m in metrics]),
        ("CPU (seconds)", *[m["cpu_seconds"] for m in metrics]),
        ("CPU (%)", *[m["cpu_percent"] for m in metrics]),
        ("peak mem (MB)", *[m["peak_mem_mb"] for m in metrics]),
    ]
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    print("\n" + "=" * (sum(widths) + 3 * len(widths)))
    for i, r in enumerate(rows):
        print("  ".join(str(v).ljust(widths[c]) for c, v in enumerate(r)))
        if i == 0:
            print("-" * (sum(widths) + 3 * len(widths)))


def _write_csv(path: Path, metrics: list[dict]) -> None:
    fields = list(metrics[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(metrics)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("extracted"))
    ap.add_argument("--method", choices=["sift", "lightglue", "openvino"],
                    help="Run a single method (omit to compare via --methods)")
    ap.add_argument("--methods", default="sift,lightglue,openvino",
                    help="Comma list of methods to compare when --method is omitted")
    ap.add_argument("--out-json", type=Path, help="Where to write single-method metrics JSON")
    ap.add_argument("--max-dim", type=int, default=1280)
    ap.add_argument("--max-features", type=int, default=4000, help="SIFT feature cap")
    ap.add_argument("--lg-features", type=int, default=1024, help="DISK feature cap (LightGlue/OpenVINO)")
    ap.add_argument("--min-matches", type=int, default=8)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--ransac-thresh", type=float, default=5.0,
                    help="RANSAC reprojection threshold in px for findHomography (lower = stricter)")
    ap.add_argument("--device", default="cpu", help="torch device for LightGlue (cpu / cuda)")
    ap.add_argument("--ov-device", default="AUTO",
                    help="OpenVINO device for 'openvino' method: AUTO / CPU / GPU / NPU / GPU.0 ...")
    ap.add_argument("--ov-devices", default=None,
                    help="Comma list to sweep 'openvino' across devices, e.g. CPU,GPU,NPU")
    ap.add_argument("--ov-size", type=int, default=1024,
                    help="Square input size for the OpenVINO DISK U-Net (multiple of 16)")
    args = ap.parse_args()

    if args.method:
        result = run_single(args)
        print(json.dumps(result, indent=2))
        if args.out_json:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        compare(args)


if __name__ == "__main__":
    main()
