from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class OpenVINODetectionResult:
    center_x: float
    center_y: float
    bbox: tuple[int, int, int, int]
    confidence: float
    inference_time_ms: float
    max_temperature: float


class OpenVINOYOLODetector:
    """YOLOv8 detection using OpenVINO runtime."""

    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        conf_threshold: float = 0.12,
        iou_threshold: float = 0.50,
        imgsz: int = 640,
    ):
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.ultralytics_imgsz_hw: Optional[tuple[int, int]] = None

        self.ultralytics_model: Optional[object] = None
        self.compiled_model = None
        self.input_port = None
        self.output_port = None
        self._load_model()

    def _load_model(self) -> None:
        model_file = Path(self.model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Prefer Ultralytics runtime when model source is in a supported OpenVINO layout.
        try:
            from ultralytics import YOLO

            model_source = self._resolve_ultralytics_openvino_source(model_file)
            if model_source is not None:
                self.ultralytics_model = YOLO(model_source, task="detect")
                self.ultralytics_imgsz_hw = self._resolve_model_input_hw(model_file)
                self._warmup_ultralytics_openvino()
        except Exception:
            self.ultralytics_model = None
            self.ultralytics_imgsz_hw = None

        try:
            import openvino as ov
        except ImportError as exc:
            raise ImportError("OpenVINO requires: pip install openvino") from exc

        core = ov.Core()
        model = core.read_model(str(model_file))
        self.compiled_model = core.compile_model(model, self.device)
        self.input_port = self.compiled_model.input(0)
        self.output_port = self.compiled_model.output(0)

    def detect(self, thermal_image: np.ndarray) -> OpenVINODetectionResult:
        """Run OpenVINO YOLOv8 detection on thermal image."""
        if self.ultralytics_model is None and self.compiled_model is None:
            raise RuntimeError("Model not loaded")

        if self.ultralytics_model is not None:
            try:
                return self._detect_ultralytics_openvino(thermal_image)
            except Exception as exc:
                # Some Ultralytics versions reject xml path-based OpenVINO models at predict time.
                msg = str(exc).lower()
                if self.compiled_model is not None and (
                    "supported model format" in msg or "openvino" in msg or "autobackend" in msg
                ):
                    self.ultralytics_model = None
                else:
                    raise

        return self._detect_raw_openvino(thermal_image)

    @staticmethod
    def _resolve_ultralytics_openvino_source(model_file: Path) -> Optional[str]:
        """Return a Ultralytics-compatible OpenVINO source path when possible."""
        if model_file.is_dir():
            return str(model_file)

        # Ultralytics often expects an OpenVINO model directory rather than a single xml path.
        if model_file.suffix.lower() == ".xml":
            parent = model_file.parent
            metadata = parent / "metadata.yaml"
            if metadata.exists():
                return str(parent)

            alt_dir = parent / f"{model_file.stem}_openvino_model"
            if alt_dir.exists() and (alt_dir / "metadata.yaml").exists():
                return str(alt_dir)

            # Common layout: ./models/yolov8n.xml with export folder at project root.
            sibling_root_dir = parent.parent / f"{model_file.stem}_openvino_model"
            if sibling_root_dir.exists() and (sibling_root_dir / "metadata.yaml").exists():
                return str(sibling_root_dir)

        return None

    def _resolve_model_input_hw(self, model_file: Path) -> Optional[tuple[int, int]]:
        """Resolve static model input height/width from OpenVINO IR when available."""
        try:
            import openvino as ov
        except Exception:
            return None

        try:
            xml_path = model_file
            if model_file.is_dir():
                xml_candidates = sorted(model_file.glob("*.xml"))
                if not xml_candidates:
                    return None
                xml_path = xml_candidates[0]

            model = ov.Core().read_model(str(xml_path))
            shape = [int(dim) for dim in model.input(0).shape]
            if len(shape) != 4:
                return None
            h, w = shape[2], shape[3]
            if h > 0 and w > 0:
                return (h, w)
        except Exception:
            return None

        return None

    def _warmup_ultralytics_openvino(self) -> None:
        """Warm up Ultralytics OpenVINO backend once to avoid first-run latency spikes."""
        if self.ultralytics_model is None:
            return

        try:
            if self.ultralytics_imgsz_hw is not None:
                warm_h, warm_w = self.ultralytics_imgsz_hw
                warm_imgsz: int | tuple[int, int] = self.ultralytics_imgsz_hw
            else:
                warm_h = warm_w = int(self.imgsz)
                warm_imgsz = int(self.imgsz)

            dummy = np.zeros((max(2, warm_h), max(2, warm_w), 3), dtype=np.uint8)
            self.ultralytics_model(
                dummy,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=warm_imgsz,
                verbose=False,
                device=self.device.lower(),
            )
        except Exception:
            # Detection can still proceed via normal path; skip warmup failures.
            pass

    def _detect_ultralytics_openvino(self, thermal_image: np.ndarray) -> OpenVINODetectionResult:
        """Use Ultralytics pipeline for robust OpenVINO output decoding."""
        t0 = time.perf_counter()

        image_3channel = self._prepare_thermal_input(thermal_image)
        infer_imgsz: int | tuple[int, int]
        if self.ultralytics_imgsz_hw is not None:
            infer_imgsz = self.ultralytics_imgsz_hw
        else:
            infer_imgsz = self.imgsz

        results = self.ultralytics_model(
            image_3channel,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=infer_imgsz,
            verbose=False,
            device=self.device.lower(),
        )

        inference_time_ms = (time.perf_counter() - t0) * 1000.0
        max_temp = float(np.max(thermal_image))

        if not results or len(results[0].boxes) == 0:
            center_y, center_x = np.unravel_index(np.argmax(thermal_image), thermal_image.shape)
            center_x = float(center_x)
            center_y = float(center_y)
            box_w = max(12, int(thermal_image.shape[1] * 0.08))
            box_h = max(12, int(thermal_image.shape[0] * 0.08))
            x = int(np.clip(center_x - box_w / 2, 0, max(0, thermal_image.shape[1] - box_w)))
            y = int(np.clip(center_y - box_h / 2, 0, max(0, thermal_image.shape[0] - box_h)))
            bbox = (x, y, box_w, box_h)
            confidence = 0.05
        else:
            boxes = results[0].boxes
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()

            scores = []
            for idx, (x1, y1, x2, y2) in enumerate(xyxy):
                x1i = int(np.clip(np.floor(x1), 0, thermal_image.shape[1] - 1))
                y1i = int(np.clip(np.floor(y1), 0, thermal_image.shape[0] - 1))
                x2i = int(np.clip(np.ceil(x2), x1i + 1, thermal_image.shape[1]))
                y2i = int(np.clip(np.ceil(y2), y1i + 1, thermal_image.shape[0]))

                roi = thermal_image[y1i:y2i, x1i:x2i]
                if roi.size == 0:
                    thermal_score = 0.0
                else:
                    p20 = float(np.percentile(thermal_image, 20.0))
                    p95 = float(np.percentile(thermal_image, 95.0))
                    thermal_score = float(np.clip((np.mean(roi) - p20) / (p95 - p20 + 1e-6), 0.0, 1.0))

                score = float(confs[idx]) * (0.65 + 0.35 * thermal_score)
                scores.append(score)

            best_idx = int(np.argmax(np.asarray(scores, dtype=np.float32)))
            x1, y1, x2, y2 = xyxy[best_idx]
            x1i = int(np.clip(np.floor(x1), 0, thermal_image.shape[1] - 1))
            y1i = int(np.clip(np.floor(y1), 0, thermal_image.shape[0] - 1))
            x2i = int(np.clip(np.ceil(x2), x1i + 1, thermal_image.shape[1]))
            y2i = int(np.clip(np.ceil(y2), y1i + 1, thermal_image.shape[0]))

            center_x, center_y = self._thermal_weighted_center(thermal_image, x1i, y1i, x2i, y2i)
            bbox = (x1i, y1i, max(1, x2i - x1i), max(1, y2i - y1i))
            confidence = float(confs[best_idx])

        return OpenVINODetectionResult(
            center_x=center_x,
            center_y=center_y,
            bbox=bbox,
            confidence=confidence,
            inference_time_ms=inference_time_ms,
            max_temperature=max_temp,
        )

    def _detect_raw_openvino(self, thermal_image: np.ndarray) -> OpenVINODetectionResult:
        """Fallback raw OpenVINO decoding path for environments without Ultralytics."""
        if self.compiled_model is None:
            raise RuntimeError("Model not loaded")

        t0 = time.perf_counter()

        orig_h, orig_w = thermal_image.shape[:2]

        input_shape = [int(dim) for dim in self.input_port.shape]
        if len(input_shape) != 4:
            raise RuntimeError(f"Unsupported OpenVINO input shape: {input_shape}")

        input_h = input_shape[2] if input_shape[2] > 0 else orig_h
        input_w = input_shape[3] if input_shape[3] > 0 else orig_w

        image_3channel = np.stack([thermal_image, thermal_image, thermal_image], axis=2)
        if (orig_w, orig_h) != (input_w, input_h):
            image_3channel = cv2.resize(image_3channel, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

        image_norm = (image_3channel / 255.0).astype(np.float32)
        input_tensor = np.expand_dims(image_norm, axis=0)

        input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))

        # Use input/output ports directly to avoid name lookup failures on unnamed tensors.
        result = self.compiled_model([input_tensor])
        output = np.array(result[self.output_port], dtype=np.float32).squeeze()

        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        max_temp = float(np.max(thermal_image))

        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]

        if output.ndim == 2 and output.shape[0] in (6, 7, 84) and output.shape[1] > output.shape[0]:
            output = output.T

        if output.ndim == 1:
            output = output.reshape(1, -1)

        if output.ndim != 2 or output.shape[0] == 0 or output.shape[1] < 5:
            center_y, center_x = np.unravel_index(np.argmax(thermal_image), thermal_image.shape)
            center_x = float(center_x)
            center_y = float(center_y)
            box_w = max(12, int(thermal_image.shape[1] * 0.08))
            box_h = max(12, int(thermal_image.shape[0] * 0.08))
            x = int(np.clip(center_x - box_w / 2, 0, max(0, thermal_image.shape[1] - box_w)))
            y = int(np.clip(center_y - box_h / 2, 0, max(0, thermal_image.shape[0] - box_h)))
            bbox = (x, y, box_w, box_h)
            confidence = 0.05
        else:
            if output.shape[1] > 6:
                class_scores = output[:, 4:]
                confidences = np.max(class_scores, axis=1)
            elif output.shape[1] > 4:
                confidences = output[:, 4]
            else:
                confidences = np.zeros(output.shape[0], dtype=np.float32)

            best_idx = np.argmax(confidences)
            best_det = output[best_idx]

            x_center, y_center, w, h = [float(v) for v in best_det[:4]]
            confidence = float(confidences[best_idx]) if confidences.size else 0.0

            scale_x = orig_w / float(input_w)
            scale_y = orig_h / float(input_h)

            x_center *= scale_x
            y_center *= scale_y
            w *= scale_x
            h *= scale_y

            center_x = float(x_center)
            center_y = float(y_center)

            x = max(0, int(x_center - w / 2))
            y = max(0, int(y_center - h / 2))
            bw = max(1, min(int(w), orig_w - x))
            bh = max(1, min(int(h), orig_h - y))
            bbox = (x, y, bw, bh)

        return OpenVINODetectionResult(
            center_x=center_x,
            center_y=center_y,
            bbox=bbox,
            confidence=confidence,
            inference_time_ms=inference_time_ms,
            max_temperature=max_temp,
        )

    def _prepare_thermal_input(self, thermal_image: np.ndarray) -> np.ndarray:
        """Convert raw thermal matrix into YOLO-friendly RGB input."""
        img = thermal_image.astype(np.float32)
        p2 = float(np.percentile(img, 2.0))
        p98 = float(np.percentile(img, 98.0))
        if p98 <= p2:
            p98 = p2 + 1e-3

        norm = np.clip((img - p2) / (p98 - p2), 0.0, 1.0)
        u8 = (norm * 255.0).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        enhanced = clahe.apply(u8)

        pseudo = cv2.applyColorMap(enhanced, cv2.COLORMAP_INFERNO)
        return cv2.cvtColor(pseudo, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _thermal_weighted_center(
        thermal_image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> tuple[float, float]:
        """Use temperature-weighted centroid inside bbox for more stable localization."""
        roi = thermal_image[y1:y2, x1:x2].astype(np.float32)
        if roi.size == 0:
            return float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)

        base = float(np.percentile(roi, 65.0))
        weights = np.clip(roi - base, 0.0, None)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-6:
            return float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)

        yy, xx = np.indices(roi.shape, dtype=np.float32)
        cx = float(np.sum((xx + x1) * weights) / weight_sum)
        cy = float(np.sum((yy + y1) * weights) / weight_sum)
        return cx, cy
