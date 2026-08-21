"""Visualization-aware matcher core for the web app.

This reuses the exact matching algorithms from ``benchmark_matchers.py`` but,
in addition to the aggregate metrics, it captures the per-pair intermediate
data needed to draw an overlay image:

  * matched keypoint coordinates (in original pixels)
  * RANSAC inlier / outlier mask (from cv2.findHomography)
  * the fitted homography H
  * the projected ce-marker vs. the manual new-marker + the pixel error

It is a self-contained CLI so the web backend can run one engine per subprocess
(clean peak-memory / CPU measurement, exactly like the benchmark tool) while
also producing the side-by-side overlay PNGs and a metrics JSON.

Overlay legend:
  * yellow dot on the left image  = manual ce marker (the point you clicked)
  * green line  = RANSAC inlier match
  * red line    = RANSAC outlier match (rejected by the homography)
  * green dot on the right image  = manual new marker (ground truth)
  * red  dot on the right image   = projected ce marker (H . ce); distance to
                                    the green dot is the reported error
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the already-validated helpers so the numbers match the CLI benchmark.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark_matchers import (  # noqa: E402
    ResourceMonitor,
    load_points,
    summarize,
    _export_disk_unet_onnx,
)


# --------------------------------------------------------------------------- #
# Per-pair result container
# --------------------------------------------------------------------------- #
def _empty_pair(p):
    return {
        "ceFile": p["ceFile"],
        "newFile": p["newFile"],
        "ce": list(p["ce"]),
        "new": list(p["new"]),
        "error_px": None,
        "src": None,
        "dst": None,
        "inliers": None,
        "proj": None,
        "n_matches": 0,
        "n_inliers": 0,
    }


def _finish_pair(pair, p, src, dst, mask, H):
    import cv2

    proj = cv2.perspectiveTransform(np.array([[p["ce"]]], np.float32), H)[0, 0]
    err = float(np.hypot(proj[0] - p["new"][0], proj[1] - p["new"][1]))
    inl = mask.ravel().astype(bool) if mask is not None else np.ones(len(src), bool)
    pair["error_px"] = err
    pair["src"] = src.tolist()
    pair["dst"] = dst.tolist()
    pair["inliers"] = inl.tolist()
    pair["proj"] = [float(proj[0]), float(proj[1])]
    pair["n_matches"] = int(len(src))
    pair["n_inliers"] = int(inl.sum())
    return err


# --------------------------------------------------------------------------- #
# SIFT
# --------------------------------------------------------------------------- #
def run_sift_vis(points, ce_dir, new_dir, max_dim, max_features, min_matches, ratio, ransac_thresh):
    import cv2

    detector = cv2.SIFT_create(nfeatures=max_features)
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
    cache: dict = {}

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
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        kps, des = detector.detectAndCompute(gray, None)
        cache[path] = (kps, des, s)
        return cache[path]

    t0 = time.perf_counter()
    results = []
    n = len(points)
    for i, p in enumerate(points, 1):
        pair = _empty_pair(p)
        f1 = feats(ce_dir / p["ceFile"])
        f2 = feats(new_dir / p["newFile"])
        if f1 is None or f2 is None or f1[1] is None or f2[1] is None:
            results.append(pair)
            continue
        kp1, des1, s1 = f1
        kp2, des2, s2 = f2
        if len(des1) < 2 or len(des2) < 2:
            results.append(pair)
            continue
        knn = flann.knnMatch(des1, des2, k=2)
        good = [m for pr in knn if len(pr) == 2 for m, n in [pr] if m.distance < ratio * n.distance]
        if len(good) < min_matches:
            results.append(pair)
            continue
        src = np.float32([kp1[m.queryIdx].pt for m in good]) / s1
        dst = np.float32([kp2[m.trainIdx].pt for m in good]) / s2
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
        if H is None:
            results.append(pair)
            continue
        err = _finish_pair(pair, p, src, dst, mask, H)
        results.append(pair)
        print(f"[sift {i}/{n}] err={err:.1f}", file=sys.stderr, flush=True)
    return results, (time.perf_counter() - t0) * 1000.0


# --------------------------------------------------------------------------- #
# LightGlue (PyTorch, no OpenVINO)
# --------------------------------------------------------------------------- #
def run_lightglue_vis(points, ce_dir, new_dir, max_dim, max_features, min_matches, device, ransac_thresh):
    import cv2
    import torch
    import kornia.feature as KF

    dev = torch.device(device)
    disk = KF.DISK.from_pretrained("depth").to(dev).eval()
    matcher = KF.LightGlueMatcher("disk").to(dev).eval()
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
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).float().permute(2, 0, 1)[None].to(dev) / 255.0
        f = disk(t, n=max_features, pad_if_not_divisible=True)[0]
        cache[path] = (f.keypoints, f.descriptors, s, small.shape[0], small.shape[1])
        return cache[path]

    t0 = time.perf_counter()
    results = []
    n = len(points)
    for i, p in enumerate(points, 1):
        pair = _empty_pair(p)
        f1 = feats(ce_dir / p["ceFile"])
        f2 = feats(new_dir / p["newFile"])
        if f1 is None or f2 is None:
            results.append(pair)
            continue
        kp1, des1, s1, h1, w1 = f1
        kp2, des2, s2, h2, w2 = f2
        if kp1.shape[0] < 2 or kp2.shape[0] < 2:
            results.append(pair)
            continue
        with torch.inference_mode():
            laf1 = KF.laf_from_center_scale_ori(kp1[None])
            laf2 = KF.laf_from_center_scale_ori(kp2[None])
            _, idxs = matcher(des1, des2, laf1, laf2, hw1=(h1, w1), hw2=(h2, w2))
        idxs = idxs.cpu().numpy()
        if len(idxs) < min_matches:
            results.append(pair)
            continue
        src = kp1.cpu().numpy()[idxs[:, 0]] / s1
        dst = kp2.cpu().numpy()[idxs[:, 1]] / s2
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
        if H is None:
            results.append(pair)
            continue
        err = _finish_pair(pair, p, src, dst, mask, H)
        results.append(pair)
        print(f"[lightglue {i}/{n}] err={err:.1f}", file=sys.stderr, flush=True)
    return results, (time.perf_counter() - t0) * 1000.0


# --------------------------------------------------------------------------- #
# OpenVINO (DISK U-Net via OpenVINO on CPU / GPU / NPU)
# --------------------------------------------------------------------------- #
def run_openvino_vis(points, ce_dir, new_dir, min_matches, ov_device, ov_size, max_features, onnx_dir, ransac_thresh):
    import cv2
    import torch
    import kornia.feature as KF
    import openvino as ov
    from kornia.feature.disk.detector import heatmap_to_keypoints

    if ov_size % 16 != 0:
        ov_size = ((ov_size + 15) // 16) * 16

    disk = KF.DISK.from_pretrained("depth").eval()
    desc_dim = disk.desc_dim
    matcher = KF.LightGlueMatcher("disk").eval()

    onnx_path = Path(onnx_dir) / f"disk_unet_{ov_size}.onnx"
    if not onnx_path.exists():
        t_exp = time.perf_counter()
        _export_disk_unet_onnx(disk, ov_size, onnx_path)
        print(f"[openvino] ONNX export ({onnx_path.name}) took {time.perf_counter() - t_exp:.2f}s",
              file=sys.stderr, flush=True)

    core = ov.Core()
    print(f"[openvino] available devices: {core.available_devices}; requested: {ov_device}",
          file=sys.stderr, flush=True)
    t_comp = time.perf_counter()
    compiled = core.compile_model(str(onnx_path), ov_device)
    compile_s = time.perf_counter() - t_comp
    print(f"[openvino] compile_model on {ov_device} took {compile_s:.2f}s",
          file=sys.stderr, flush=True)
    out_port = compiled.output(0)
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
        canvas = np.zeros((ov_size, ov_size, 3), np.float32)
        canvas[:h_r, :w_r] = rgb
        inp = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None])
        out = compiled(inp)[out_port]
        out_t = torch.from_numpy(np.ascontiguousarray(out))
        heatmaps = out_t[:, desc_dim:desc_dim + 1, :h_r, :w_r]
        descriptors = out_t[:, :desc_dim, :h_r, :w_r]
        with torch.inference_mode():
            kpl = heatmap_to_keypoints(heatmaps, n=max_features, window_size=5, score_threshold=0.0)
            feat = kpl[0].merge_with_descriptors(descriptors[0])
        cache[path] = (feat.keypoints, feat.descriptors, s, h_r, w_r)
        return cache[path]

    t0 = time.perf_counter()
    results = []
    n = len(points)
    for i, p in enumerate(points, 1):
        pair = _empty_pair(p)
        f1 = feats(ce_dir / p["ceFile"])
        f2 = feats(new_dir / p["newFile"])
        if f1 is None or f2 is None:
            results.append(pair)
            continue
        kp1, des1, s1, h1, w1 = f1
        kp2, des2, s2, h2, w2 = f2
        if kp1.shape[0] < 2 or kp2.shape[0] < 2:
            results.append(pair)
            continue
        with torch.inference_mode():
            laf1 = KF.laf_from_center_scale_ori(kp1[None].float())
            laf2 = KF.laf_from_center_scale_ori(kp2[None].float())
            _, idxs = matcher(des1, des2, laf1, laf2, hw1=(h1, w1), hw2=(h2, w2))
        idxs = idxs.cpu().numpy()
        if len(idxs) < min_matches:
            results.append(pair)
            continue
        src = kp1.cpu().numpy()[idxs[:, 0]] / s1
        dst = kp2.cpu().numpy()[idxs[:, 1]] / s2
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
        if H is None:
            results.append(pair)
            continue
        err = _finish_pair(pair, p, src, dst, mask, H)
        results.append(pair)
        print(f"[openvino {i}/{n}] err={err:.1f}", file=sys.stderr, flush=True)
    return results, (time.perf_counter() - t0) * 1000.0


# --------------------------------------------------------------------------- #
# Overlay rendering
# --------------------------------------------------------------------------- #
def _load_disp(path, disp_w):
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        return None, 1.0
    h, w = img.shape[:2]
    sf = disp_w / w
    disp = cv2.resize(img, (disp_w, max(1, int(round(h * sf)))))
    return disp, sf


def render_pair(pair, ce_dir, new_dir, out_path, disp_w=720, max_outliers=60):
    """Draw the side-by-side overlay for one pair and save it to out_path."""
    import cv2

    im1, sf1 = _load_disp(ce_dir / pair["ceFile"], disp_w)
    im2, sf2 = _load_disp(new_dir / pair["newFile"], disp_w)
    if im1 is None or im2 is None:
        return False

    h1, h2 = im1.shape[0], im2.shape[0]
    H = max(h1, h2)
    canvas = np.zeros((H, disp_w * 2, 3), np.uint8)
    canvas[:h1, :disp_w] = im1
    canvas[:h2, disp_w:] = im2
    off = disp_w  # x-offset of the right image inside the canvas

    GREEN = (0, 200, 0)
    RED = (0, 0, 255)
    YELLOW = (0, 220, 220)

    # matched keypoint lines: inliers (green) + a capped number of outliers (red)
    if pair["src"] is not None:
        src = np.array(pair["src"], np.float32) * sf1
        dst = np.array(pair["dst"], np.float32) * sf2
        inl = np.array(pair["inliers"], bool)
        # outliers first (thin), so inliers draw on top
        drawn_out = 0
        for j in range(len(src)):
            if inl[j]:
                continue
            if drawn_out >= max_outliers:
                break
            drawn_out += 1
            a = (int(src[j][0]), int(src[j][1]))
            b = (int(dst[j][0]) + off, int(dst[j][1]))
            cv2.line(canvas, a, b, RED, 1, cv2.LINE_AA)
        for j in range(len(src)):
            if not inl[j]:
                continue
            a = (int(src[j][0]), int(src[j][1]))
            b = (int(dst[j][0]) + off, int(dst[j][1]))
            cv2.line(canvas, a, b, GREEN, 1, cv2.LINE_AA)
            cv2.circle(canvas, a, 2, GREEN, -1, cv2.LINE_AA)
            cv2.circle(canvas, b, 2, GREEN, -1, cv2.LINE_AA)

    # manual ce marker on the left image (yellow)
    ce = (int(pair["ce"][0] * sf1), int(pair["ce"][1] * sf1))
    cv2.drawMarker(canvas, ce, YELLOW, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
    cv2.circle(canvas, ce, 9, YELLOW, 2, cv2.LINE_AA)

    # manual new marker (green) + projected marker (red) on the right image
    gt = (int(pair["new"][0] * sf2) + off, int(pair["new"][1] * sf2))
    cv2.drawMarker(canvas, gt, GREEN, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
    cv2.circle(canvas, gt, 9, GREEN, 2, cv2.LINE_AA)
    if pair["proj"] is not None:
        pj = (int(pair["proj"][0] * sf2) + off, int(pair["proj"][1] * sf2))
        cv2.drawMarker(canvas, pj, RED, cv2.MARKER_TILTED_CROSS, 22, 2, cv2.LINE_AA)
        cv2.circle(canvas, pj, 9, RED, 2, cv2.LINE_AA)
        cv2.line(canvas, gt, pj, RED, 1, cv2.LINE_AA)

    # header text
    if pair["error_px"] is None:
        txt = "NO MATCH (matching failed)"
    else:
        txt = (f"err={pair['error_px']:.1f}px  matches={pair['n_matches']}  "
               f"inliers={pair['n_inliers']}")
    cv2.rectangle(canvas, (0, 0), (disp_w * 2, 26), (0, 0, 0), -1)
    cv2.putText(canvas, txt, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return True


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(args) -> dict:
    points = load_points(args.data / "points.csv")
    ce_dir = args.data / "ce"
    new_dir = args.data / "new"
    onnx_dir = args.onnx_dir or (Path(__file__).resolve().parent.parent / "ov_models")
    label = args.method

    with ResourceMonitor() as mon:
        if args.method == "sift":
            results, proc_ms = run_sift_vis(
                points, ce_dir, new_dir, args.max_dim, args.max_features,
                args.min_matches, args.ratio, args.ransac_thresh)
        elif args.method == "lightglue":
            results, proc_ms = run_lightglue_vis(
                points, ce_dir, new_dir, args.max_dim, args.lg_features,
                args.min_matches, args.device, args.ransac_thresh)
        else:
            label = f"openvino:{args.ov_device}"
            results, proc_ms = run_openvino_vis(
                points, ce_dir, new_dir, args.min_matches, args.ov_device,
                args.ov_size, args.lg_features, onnx_dir, args.ransac_thresh)

    # summarize() only needs error_px in each result, which we provide.
    metrics = summarize(label, results, proc_ms, mon)

    # render overlays + build a compact per-pair index for the frontend
    vis_dir = args.vis_dir
    vis_dir.mkdir(parents=True, exist_ok=True)
    pairs_index = []
    for i, pair in enumerate(results):
        name = f"pair_{i:03d}.jpg"
        render_pair(pair, ce_dir, new_dir, vis_dir / name)
        pairs_index.append({
            "idx": i,
            "ceFile": pair["ceFile"],
            "newFile": pair["newFile"],
            "error_px": pair["error_px"],
            "n_matches": pair["n_matches"],
            "n_inliers": pair["n_inliers"],
            "vis": name,
        })
    metrics["pairs"] = pairs_index
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder with ce/, new/, points.csv")
    ap.add_argument("--method", choices=["sift", "lightglue", "openvino"], required=True)
    ap.add_argument("--vis-dir", type=Path, required=True, help="where to write overlay PNGs")
    ap.add_argument("--out-json", type=Path, help="where to write metrics JSON")
    ap.add_argument("--max-dim", type=int, default=1280)
    ap.add_argument("--max-features", type=int, default=4000)
    ap.add_argument("--lg-features", type=int, default=1024)
    ap.add_argument("--min-matches", type=int, default=8)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--ransac-thresh", type=float, default=5.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ov-device", default="AUTO")
    ap.add_argument("--ov-size", type=int, default=1024)
    ap.add_argument("--onnx-dir", type=Path, default=None,
                    help="folder for the DISK-U-Net ONNX cache (default: ../ov_models)")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    result = run(args)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    # keep stdout compact: full detail is in the JSON
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
