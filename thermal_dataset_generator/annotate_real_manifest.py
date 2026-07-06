"""Interactive tool to annotate real IR images with target_point for sim-to-real evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".npy"}


@dataclass(slots=True)
class Entry:
    """Manifest row for one real image."""

    image: str
    target_point: list[int] | None
    case_type: str


class ManifestAnnotator:
    """OpenCV-based point annotation UI for real thermal references."""

    def __init__(self, input_dir: Path, output_manifest: Path):
        self.input_dir = input_dir
        self.output_manifest = output_manifest
        self.images = self._discover_images(input_dir)
        self.entries: list[Entry] = self._load_or_init_entries()
        self.cursor = 0
        self.pending_click: tuple[int, int] | None = None

    @staticmethod
    def _discover_images(input_dir: Path) -> list[Path]:
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"Input directory not found: {input_dir}")
        images = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in SUPPORTED_SUFFIXES]
        if not images:
            raise ValueError(f"No supported images found in: {input_dir}")
        return images

    def _load_or_init_entries(self) -> list[Entry]:
        if self.output_manifest.exists():
            with self.output_manifest.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            loaded: dict[str, Entry] = {}
            for row in raw if isinstance(raw, list) else []:
                if not isinstance(row, dict):
                    continue
                image = str(row.get("image", "")).strip()
                if not image:
                    continue
                point = row.get("target_point")
                if isinstance(point, (list, tuple)) and len(point) == 2:
                    tp = [int(point[0]), int(point[1])]
                else:
                    tp = None
                case_type = str(row.get("case_type", "normal")).strip().lower() or "normal"
                loaded[image] = Entry(image=image, target_point=tp, case_type=case_type)

            rows: list[Entry] = []
            for img in self.images:
                key = self._relative_image_path(img)
                rows.append(loaded.get(key, Entry(image=key, target_point=None, case_type="normal")))
            return rows

        rows = []
        for img in self.images:
            rows.append(Entry(image=self._relative_image_path(img), target_point=None, case_type="normal"))
        return rows

    def _relative_image_path(self, image_path: Path) -> str:
        return str(image_path.resolve().relative_to(self.output_manifest.parent.resolve())).replace("\\", "/")

    @staticmethod
    def _load_frame(path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            arr = np.load(path).astype(np.float32)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D npy thermal matrix: {path}")
            low = float(np.min(arr))
            high = float(np.max(arr))
            if high <= low:
                high = low + 1e-6
            u8 = np.clip((arr - low) / (high - low), 0.0, 1.0)
            u8 = (u8 * 255.0).astype(np.uint8)
            bgr = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
            return bgr

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")
        return image

    def _save(self) -> None:
        self.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "image": e.image,
                "target_point": e.target_point,
                "case_type": e.case_type,
            }
            for e in self.entries
        ]
        with self.output_manifest.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def run(self) -> None:
        win = "annotate_real_manifest"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1100, 760)

        def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.pending_click = (int(x), int(y))

        cv2.setMouseCallback(win, on_mouse)

        print("Controls:")
        print("  Left click: set target_point")
        print("  A / D: previous / next image")
        print("  C: clear point")
        print("  N: case_type=normal")
        print("  R: case_type=reversal")
        print("  V: case_type=vent_reversal")
        print("  T: case_type=trap_reversal")
        print("  S: save manifest")
        print("  Q or ESC: save and quit")

        while True:
            entry = self.entries[self.cursor]
            image_path = (self.output_manifest.parent / entry.image).resolve()
            frame = self._load_frame(image_path)

            if self.pending_click is not None:
                px, py = self.pending_click
                entry.target_point = [int(px), int(py)]
                self.pending_click = None

            if entry.target_point is not None:
                cv2.drawMarker(
                    frame,
                    (int(entry.target_point[0]), int(entry.target_point[1])),
                    (0, 255, 255),
                    markerType=cv2.MARKER_STAR,
                    markerSize=24,
                    thickness=2,
                )

            progress = f"{self.cursor + 1}/{len(self.entries)}"
            status = (
                f"{progress} | {entry.image} | case_type={entry.case_type} | "
                f"target_point={entry.target_point}"
            )
            cv2.putText(frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "A/D prev-next, C clear, N/R/V/T case, S save, Q quit", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 220, 220), 1, cv2.LINE_AA)

            cv2.imshow(win, frame)
            key = cv2.waitKey(30) & 0xFF

            if key in (27, ord("q")):
                self._save()
                break
            if key == ord("s"):
                self._save()
                print(f"Saved: {self.output_manifest}")
            elif key == ord("a"):
                self.cursor = max(0, self.cursor - 1)
            elif key == ord("d"):
                self.cursor = min(len(self.entries) - 1, self.cursor + 1)
            elif key == ord("c"):
                entry.target_point = None
            elif key == ord("n"):
                entry.case_type = "normal"
            elif key == ord("r"):
                entry.case_type = "reversal"
            elif key == ord("v"):
                entry.case_type = "vent_reversal"
            elif key == ord("t"):
                entry.case_type = "trap_reversal"

        cv2.destroyAllWindows()
        print(f"Saved and exit: {self.output_manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate target_point on real IR images")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("thermal_dataset_generator/reference_ir"),
        help="Directory containing real IR images (.png/.jpg/.npy)",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("thermal_dataset_generator/reference_ir/real_manifest.json"),
        help="Output manifest JSON path",
    )
    args = parser.parse_args()

    annotator = ManifestAnnotator(input_dir=args.input_dir, output_manifest=args.output_manifest)
    annotator.run()


if __name__ == "__main__":
    main()
