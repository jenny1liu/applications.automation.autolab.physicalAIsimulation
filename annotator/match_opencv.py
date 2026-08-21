"""Auto-match Blue (ce) <-> Unote (new) images with OpenCV SIFT, estimate a
homography, project the annotated marker, and score it against the manual points.

No extra dependencies: uses only opencv-python (SIFT) + numpy, already installed.
This mirrors the LightGlue workflow so it can be swapped later.

Speed: full-res phone photos are downscaled before SIFT, features are cached per
unique image (4 Blue + 17 Unote, not 68x recompute), and FLANN is used for matching.

Inputs (produced by extract_pairs.py):
  <data>/ce/*.jpg        reference images
  <data>/new/*.jpg       target images
  <data>/points.csv      manual correspondences: ceFile,newFile,ce_x,ce_y,new_x,new_y

Outputs:
  <data>/match_out/results.csv   per-pair pixel error + inlier count
  <data>/match_out/*.jpg         match visualization per pair

Usage:
  python match_opencv.py --data extracted --max-dim 1280
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


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


def make_detector(max_features: int):
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=max_features), "SIFT"
    return cv2.ORB_create(max_features), "ORB"


class Features:
    """Cached SIFT features for one image (kept at downscaled resolution)."""

    def __init__(self, img_small, kps, des, scale):
        self.img_small = img_small  # downscaled BGR image (for viz)
        self.kps = kps              # KeyPoints in downscaled coords
        self.des = des              # descriptors
        self.scale = scale          # small = original * scale


def extract_features(path: Path, detector, max_dim: int, cache: dict):
    if path in cache:
        return cache[path]
    img = cv2.imread(str(path))
    if img is None:
        cache[path] = None
        return None
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    kps, des = detector.detectAndCompute(gray, None)
    feat = Features(small, kps, des, scale)
    cache[path] = feat
    return feat


def match_descriptors(des1, des2, flann, ratio: float):
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
    knn = flann.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])
    return good


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("extracted"))
    parser.add_argument("--max-dim", type=int, default=1280, help="Downscale so longest side <= this")
    parser.add_argument("--max-features", type=int, default=4000)
    parser.add_argument("--min-matches", type=int, default=8)
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test threshold")
    args = parser.parse_args()

    ce_dir = args.data / "ce"
    new_dir = args.data / "new"
    out_dir = args.data / "match_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    points = load_points(args.data / "points.csv")
    detector, det_name = make_detector(args.max_features)
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
    print(f"Detector: {det_name}; pairs: {len(points)}; max_dim={args.max_dim}", flush=True)

    cache: dict = {}
    results = []
    for i, p in enumerate(points, 1):
        f1 = extract_features(ce_dir / p["ceFile"], detector, args.max_dim, cache)
        f2 = extract_features(new_dir / p["newFile"], detector, args.max_dim, cache)
        if f1 is None or f2 is None:
            results.append({**_flat(p), "status": "missing_image", "inliers": 0, "error_px": ""})
            continue

        good = match_descriptors(f1.des, f2.des, flann, args.ratio)
        if len(good) < args.min_matches:
            results.append({**_flat(p), "status": "too_few_matches", "inliers": len(good), "error_px": ""})
            print(f"[{i}/{len(points)}] {p['ceFile']}->{p['newFile']}: too few ({len(good)})", flush=True)
            continue

        # matched points, converted back to ORIGINAL image coordinates
        src = np.float32([f1.kps[m.queryIdx].pt for m in good]) / f1.scale
        dst = np.float32([f2.kps[m.trainIdx].pt for m in good]) / f2.scale
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            results.append({**_flat(p), "status": "no_homography", "inliers": len(good), "error_px": ""})
            continue

        inliers = int(mask.sum()) if mask is not None else 0
        ce_pt = np.array([[p["ce"]]], dtype=np.float32)
        proj = cv2.perspectiveTransform(ce_pt, H)[0, 0]
        err = float(np.hypot(proj[0] - p["new"][0], proj[1] - p["new"][1]))
        results.append({**_flat(p), "status": "ok", "inliers": inliers, "error_px": round(err, 2)})
        print(f"[{i}/{len(points)}] {p['ceFile']}->{p['newFile']}: err={err:.1f}px inliers={inliers}", flush=True)

        _save_viz(out_dir, p, f1, f2, good, mask, proj)

    _write_results(out_dir / "results.csv", results)
    _print_summary(results)


def _flat(p: dict) -> dict:
    return {
        "ceFile": p["ceFile"], "newFile": p["newFile"],
        "ce_x": p["ce"][0], "ce_y": p["ce"][1],
        "new_x": p["new"][0], "new_y": p["new"][1],
    }


def _save_viz(out_dir, p, f1: Features, f2: Features, good, mask, proj):
    keep = mask.ravel().astype(bool) if mask is not None else np.ones(len(good), bool)
    draw = [g for g, k in zip(good, keep) if k]
    vis = cv2.drawMatches(f1.img_small, f1.kps, f2.img_small, f2.kps, draw, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    w1 = f1.img_small.shape[1]
    # manual (green) vs projected (red) marker on the right (img2) half, in small coords
    mx = int(round(p["new"][0] * f2.scale)) + w1
    my = int(round(p["new"][1] * f2.scale))
    px = int(round(proj[0] * f2.scale)) + w1
    py = int(round(proj[1] * f2.scale))
    cv2.circle(vis, (mx, my), 14, (0, 255, 0), 3)
    cv2.circle(vis, (px, py), 9, (0, 0, 255), 3)
    cv2.line(vis, (mx, my), (px, py), (0, 255, 255), 2)
    name = f"{Path(p['ceFile']).stem}__{Path(p['newFile']).stem}.jpg"
    cv2.imwrite(str(out_dir / name), vis)


def _write_results(path: Path, results: list[dict]) -> None:
    fields = ["ceFile", "newFile", "ce_x", "ce_y", "new_x", "new_y", "status", "inliers", "error_px"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"Wrote {path}", flush=True)


def _print_summary(results: list[dict]) -> None:
    ok = [r for r in results if r["status"] == "ok" and r["error_px"] != ""]
    print(f"\nSummary: {len(ok)}/{len(results)} pairs matched", flush=True)
    if ok:
        errs = np.array([r["error_px"] for r in ok], dtype=float)
        print(f"  pixel error  mean={errs.mean():.1f}  median={np.median(errs):.1f}  max={errs.max():.1f}")
        print(f"  within 20 px: {int((errs <= 20).sum())}/{len(ok)}")
    for status, n in Counter(r["status"] for r in results).items():
        print(f"  status[{status}] = {n}")


if __name__ == "__main__":
    main()
