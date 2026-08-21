"""Decode the annotator pair-session JSON into usable image files + correspondence table.

The annotator session JSON stores:
  - ceItems : reference images that were manually annotated (base64 data URLs)
  - newItems: target images for later LightGlue matching (base64 data URLs)
  - points  : manual correspondences {ceFile, newFile, ce:[x,y], new:[x,y]}

This script writes:
  <out>/ce/*.jpg          decoded reference images
  <out>/new/*.jpg         decoded target images
  <out>/points.csv        one row per correspondence
  <out>/viz/*.jpg         (optional) images with the annotated marker drawn on them

Usage:
  python extract_pairs.py SESSION.json --out extracted --viz
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import re
from pathlib import Path


DATA_URL_RE = re.compile(r"^data:image/[^;]+;base64,")


def decode_data_url(data_url: str) -> bytes:
    """Return raw image bytes from a `data:image/...;base64,....` string."""
    b64 = DATA_URL_RE.sub("", data_url, count=1)
    return base64.b64decode(b64)


def write_items(items: list[dict], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in items:
        name = item.get("file")
        data_url = item.get("imageDataUrl")
        if not name or not data_url:
            continue
        (out_dir / name).write_bytes(decode_data_url(data_url))
        count += 1
    return count


def write_points_csv(points: list[dict], csv_path: Path) -> int:
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ceFile", "newFile", "ce_x", "ce_y", "new_x", "new_y"])
        for p in points:
            ce = p.get("ce") or [None, None]
            new = p.get("new") or [None, None]
            writer.writerow([p.get("ceFile"), p.get("newFile"), ce[0], ce[1], new[0], new[1]])
    return len(points)


def make_viz(points: list[dict], ce_dir: Path, new_dir: Path, viz_dir: Path) -> None:
    """Draw each annotated marker location onto its image for visual verification."""
    import cv2

    viz_dir.mkdir(parents=True, exist_ok=True)

    def draw(src_dir: Path, filename: str, xy, tag: str) -> None:
        if not filename or xy is None or xy[0] is None:
            return
        img_path = src_dir / filename
        if not img_path.exists():
            return
        img = cv2.imread(str(img_path))
        if img is None:
            return
        x, y = int(round(xy[0])), int(round(xy[1]))
        cv2.circle(img, (x, y), 25, (0, 0, 255), 3)
        cv2.drawMarker(img, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 40, 2)
        cv2.imwrite(str(viz_dir / f"{tag}_{filename}"), img)

    for p in points:
        draw(ce_dir, p.get("ceFile"), p.get("ce"), "ce")
        draw(new_dir, p.get("newFile"), p.get("new"), "new")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path, help="Path to the annotator session JSON")
    parser.add_argument("--out", type=Path, default=Path("extracted"), help="Output directory")
    parser.add_argument("--viz", action="store_true", help="Also write images with annotated markers drawn on")
    args = parser.parse_args()

    data = json.loads(args.session.read_text(encoding="utf-8"))
    ce_items = data.get("ceItems", [])
    new_items = data.get("newItems", [])
    points = data.get("points", [])

    ce_dir = args.out / "ce"
    new_dir = args.out / "new"

    n_ce = write_items(ce_items, ce_dir)
    n_new = write_items(new_items, new_dir)
    n_pts = write_points_csv(points, args.out / "points.csv")

    print(f"Wrote {n_ce} reference images -> {ce_dir}")
    print(f"Wrote {n_new} target images    -> {new_dir}")
    print(f"Wrote {n_pts} correspondences  -> {args.out / 'points.csv'}")

    if args.viz:
        make_viz(points, ce_dir, new_dir, args.out / "viz")
        print(f"Wrote visualization images  -> {args.out / 'viz'}")


if __name__ == "__main__":
    main()
