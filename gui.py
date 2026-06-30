from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from metrics import MetricsCalculator
from opencv_detector import OpenCVHotspotDetector
from openvino_detector import OpenVINOYOLODetector
from robot_mapper import RobotTargetMapper
from thermal_generator import HotspotShape, ThermalImageGenerator
from yolo_detector import YOLOv8PyTorchDetector


class ThermalHotspotDemo:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Thermal Hotspot Localization - OpenVINO Performance Demo")
        self.root.geometry("1600x1000")

        self.generator = ThermalImageGenerator(width=320, height=240, noise_std=1.5)
        self.opencv_detector = OpenCVHotspotDetector()
        self.yolo_pytorch_detector: Optional[YOLOv8PyTorchDetector] = None
        self.yolo_openvino_detector: Optional[OpenVINOYOLODetector] = None
        self.robot_mapper = RobotTargetMapper()

        self.current_frame = None
        self.current_centers = []
        self.metrics_history = {"opencv": [], "pytorch": [], "openvino": []}

        self.setup_ui()
        self._initialize_default_models()

    def setup_ui(self) -> None:
        """Setup the GUI layout."""
        style = ttk.Style()
        style.theme_use("clam")

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._setup_controls(left_frame)
        self._setup_visualization(left_frame)
        self._setup_metrics_display(right_frame)

    def _setup_controls(self, parent: ttk.Frame) -> None:
        """Setup control panel."""
        ctrl_frame = ttk.LabelFrame(parent, text="Controls", padding=10)
        ctrl_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(ctrl_frame, text="Generate New Frame", command=self.generate_new_frame).pack(
            fill=tk.X, pady=2
        )

        ttk.Label(ctrl_frame, text="Hotspot Count:").pack(anchor=tk.W)
        self.hotspot_count = ttk.Scale(
            ctrl_frame, from_=1, to=5, orient=tk.HORIZONTAL, command=lambda v: None
        )
        self.hotspot_count.set(1)
        self.hotspot_count.pack(fill=tk.X, pady=2)

        ttk.Label(ctrl_frame, text="Noise Level:").pack(anchor=tk.W)
        self.noise_scale = ttk.Scale(ctrl_frame, from_=0.0, to=5.0, orient=tk.HORIZONTAL)
        self.noise_scale.set(1.5)
        self.noise_scale.pack(fill=tk.X, pady=2)

        ttk.Separator(ctrl_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Button(ctrl_frame, text="Run Detection", command=self.run_detection).pack(fill=tk.X, pady=5)

        bench_frame = ttk.Frame(ctrl_frame)
        bench_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bench_frame, text="Benchmark Samples:").pack(side=tk.LEFT)
        self.benchmark_samples = tk.IntVar(value=100)
        tk.Spinbox(
            bench_frame,
            from_=10,
            to=2000,
            increment=10,
            textvariable=self.benchmark_samples,
            width=6,
        ).pack(side=tk.RIGHT)

        ttk.Button(ctrl_frame, text="Run Benchmark", command=self.run_benchmark).pack(fill=tk.X, pady=3)

        ttk.Button(ctrl_frame, text="Clear History", command=self.clear_history).pack(fill=tk.X, pady=2)

    def _setup_visualization(self, parent: ttk.Frame) -> None:
        """Setup visualization panels."""
        vis_frame = ttk.LabelFrame(parent, text="Detection Results", padding=5)
        vis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig = Figure(figsize=(12, 8), dpi=80)
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ax_thermal = self.fig.add_subplot(2, 3, 1)
        self.ax_mask = self.fig.add_subplot(2, 3, 2)
        self.ax_opencv = self.fig.add_subplot(2, 3, 3)
        self.ax_pytorch = self.fig.add_subplot(2, 3, 4)
        self.ax_openvino = self.fig.add_subplot(2, 3, 5)
        self.ax_metrics = self.fig.add_subplot(2, 3, 6)

    def _setup_metrics_display(self, parent: ttk.Frame) -> None:
        """Setup metrics and robot target display."""
        metrics_frame = ttk.LabelFrame(parent, text="Metrics & Robot Target", padding=10)
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.metrics_text = tk.Text(metrics_frame, height=30, width=50, state=tk.DISABLED)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.metrics_text)
        self.metrics_text["yscrollcommand"] = scrollbar.set
        scrollbar["command"] = self.metrics_text.yview

    def generate_new_frame(self) -> None:
        """Generate a new thermal frame."""
        self.generator.noise_std = float(self.noise_scale.get())
        hotspot_count = int(self.hotspot_count.get())

        self.current_frame = self.generator.generate(
            hotspot_count=hotspot_count,
            shape=HotspotShape.CIRCULAR,
        )
        self._update_visualization()

    def _initialize_default_models(self) -> None:
        """Auto-enable default detectors for direct benchmarking workflow."""
        try:
            self.yolo_pytorch_detector = YOLOv8PyTorchDetector(device="cpu")
        except Exception as e:
            print(f"PyTorch default model load failed: {e}")

        self.load_openvino_model(
            model_path_raw="yolov8n_openvino_model/yolov8n.xml",
            show_success=False,
            show_errors=False,
        )

    def load_openvino_model(
        self,
        model_path_raw: Optional[str] = None,
        show_success: bool = True,
        show_errors: bool = True,
    ) -> None:
        """Load OpenVINO model."""
        from pathlib import Path

        try:
            if model_path_raw is None:
                model_path_raw = "yolov8n_openvino_model/yolov8n.xml"

            input_path = Path(model_path_raw.strip())
            project_root = Path(__file__).resolve().parent
            candidate_paths = [input_path]
            if not input_path.is_absolute():
                candidate_paths = [Path.cwd() / input_path, project_root / input_path]

            resolved_path = next((p for p in candidate_paths if p.exists()), None)
            if resolved_path is None:
                fallback_paths = [
                    Path.cwd() / "yolov8n_openvino_model" / "yolov8n.xml",
                    project_root / "yolov8n_openvino_model" / "yolov8n.xml",
                    Path.cwd() / "models" / "yolov8n.xml",
                    project_root / "models" / "yolov8n.xml",
                ]
                resolved_path = next((p for p in fallback_paths if p.exists()), None)

            if resolved_path is None:
                tried_paths = "\n".join(str(p) for p in candidate_paths)
                if show_errors:
                    messagebox.showerror(
                        "Error",
                        f"Model file not found:\n{model_path_raw}\n\nCurrent dir: {Path.cwd()}\nProject dir: {project_root}\n\nTried:\n{tried_paths}\n\nExamples:\n• models/yolov8n.xml\n• C:\\path\\to\\model.xml",
                    )
                return

            resolved_path = resolved_path.resolve()
            self.yolo_openvino_detector = OpenVINOYOLODetector(model_path=str(resolved_path), device="CPU")
            if show_success:
                messagebox.showinfo("Success", f"OpenVINO model loaded!\n\n{resolved_path}")
        except ImportError as e:
            if show_errors:
                messagebox.showerror("Error", f"OpenVINO not installed:\n{str(e)}\n\nRun: pip install openvino")
        except Exception as e:
            if show_errors:
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")

    def run_detection(self) -> None:
        """Run all detections in a background thread."""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "Generate a frame first")
            return

        thread = threading.Thread(target=self._run_detections_threaded)
        thread.daemon = True
        thread.start()

    def run_benchmark(self) -> None:
        """Run one-click benchmark across available detectors."""
        sample_count = int(self.benchmark_samples.get())
        if sample_count <= 0:
            messagebox.showwarning("Warning", "Benchmark sample count must be > 0")
            return

        thread = threading.Thread(target=self._run_benchmark_threaded, args=(sample_count,))
        thread.daemon = True
        thread.start()

    def _run_benchmark_threaded(self, sample_count: int) -> None:
        """Benchmark detectors on the same synthetic frame stream."""
        try:
            if self.yolo_pytorch_detector is None:
                try:
                    self.yolo_pytorch_detector = YOLOv8PyTorchDetector(device="cpu")
                except Exception as e:
                    print(f"PyTorch benchmark init failed: {e}")

            if self.yolo_openvino_detector is None:
                self.load_openvino_model(show_success=False, show_errors=False)

            detectors = {
                "opencv": self.opencv_detector,
                "pytorch": self.yolo_pytorch_detector,
                "openvino": self.yolo_openvino_detector,
            }

            active_detectors = {k: v for k, v in detectors.items() if v is not None}
            if not active_detectors:
                self.root.after(0, lambda: messagebox.showerror("Benchmark Error", "No detector available"))
                return

            stats = {
                key: {"error": [], "latency": [], "fps": []}
                for key in active_detectors
            }

            gen = ThermalImageGenerator(
                width=self.generator.width,
                height=self.generator.height,
                noise_std=float(self.noise_scale.get()),
                seed=20260630,
            )
            hotspot_count = int(self.hotspot_count.get())

            # Warmup one shared frame so first-run setup does not skew benchmark latency.
            warm_frame = gen.generate(hotspot_count=hotspot_count, shape=HotspotShape.CIRCULAR)
            for detector in active_detectors.values():
                try:
                    detector.detect(warm_frame.image)
                except Exception:
                    pass

            for _ in range(sample_count):
                frame = gen.generate(hotspot_count=hotspot_count, shape=HotspotShape.CIRCULAR)
                gt_x, gt_y = frame.centers[0]

                for key, detector in active_detectors.items():
                    result = detector.detect(frame.image)
                    error = MetricsCalculator.localization_error(result.center_x, result.center_y, gt_x, gt_y)
                    latency = float(result.inference_time_ms)
                    fps = 1000.0 / latency if latency > 0 else 0.0

                    stats[key]["error"].append(error)
                    stats[key]["latency"].append(latency)
                    stats[key]["fps"].append(fps)

            report = self._format_benchmark_report(stats, sample_count)
            self.root.after(0, lambda: self._apply_benchmark_report(report))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Benchmark Error", str(e)))

    def _format_benchmark_report(self, stats: dict, sample_count: int) -> str:
        """Format benchmark summary with mean/median/P95 metrics."""
        lines = [
            "Benchmark Summary:",
            "",
            f"Samples: {sample_count}",
            "Shared frame stream: Yes",
            "Ground truth source: frame.centers[0]",
            "",
        ]

        for key in ("opencv", "pytorch", "openvino"):
            if key not in stats:
                continue

            error_arr = np.asarray(stats[key]["error"], dtype=np.float32)
            latency_arr = np.asarray(stats[key]["latency"], dtype=np.float32)
            fps_arr = np.asarray(stats[key]["fps"], dtype=np.float32)

            lines.append(f"{key.upper()}:")
            lines.append(
                f"  Error (px)   mean/median/P95: {np.mean(error_arr):.2f} / {np.median(error_arr):.2f} / {np.percentile(error_arr, 95):.2f}"
            )
            lines.append(
                f"  Latency (ms) mean/median/P95: {np.mean(latency_arr):.2f} / {np.median(latency_arr):.2f} / {np.percentile(latency_arr, 95):.2f}"
            )
            lines.append(
                f"  FPS          mean/median/P95: {np.mean(fps_arr):.1f} / {np.median(fps_arr):.1f} / {np.percentile(fps_arr, 95):.1f}"
            )
            lines.append("")

        return "\n".join(lines)

    def _apply_benchmark_report(self, report: str) -> None:
        """Display benchmark summary in metrics panel."""
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, report)
        self.metrics_text.config(state=tk.DISABLED)

    def _run_detections_threaded(self) -> None:
        """Run detections (called in background thread)."""
        try:
            results = {}

            opencv_result = self.opencv_detector.detect(self.current_frame.image)
            results["opencv"] = opencv_result

            if self.yolo_pytorch_detector is None:
                try:
                    self.yolo_pytorch_detector = YOLOv8PyTorchDetector(device="cpu")
                except Exception as e:
                    print(f"PyTorch model not available: {e}")

            if self.yolo_pytorch_detector:
                pytorch_result = self.yolo_pytorch_detector.detect(self.current_frame.image)
                results["pytorch"] = pytorch_result

            if self.yolo_openvino_detector:
                openvino_result = self.yolo_openvino_detector.detect(self.current_frame.image)
                results["openvino"] = openvino_result

            self._display_results(results)
        except Exception as e:
            messagebox.showerror("Detection Error", str(e))

    def _display_results(self, results: dict) -> None:
        """Display detection results."""
        self.fig.clear()

        self.ax_thermal = self.fig.add_subplot(2, 3, 1)
        self.ax_mask = self.fig.add_subplot(2, 3, 2)
        self.ax_opencv = self.fig.add_subplot(2, 3, 3)
        self.ax_pytorch = self.fig.add_subplot(2, 3, 4)
        self.ax_openvino = self.fig.add_subplot(2, 3, 5)
        self.ax_metrics = self.fig.add_subplot(2, 3, 6)

        self.ax_thermal.imshow(self.current_frame.image, cmap="inferno")
        self.ax_thermal.set_title("Input Thermal")
        self.ax_thermal.scatter(
            [pos[0] for pos in self.current_frame.centers],
            [pos[1] for pos in self.current_frame.centers],
            c="cyan",
            s=100,
            marker="*",
        )

        self.ax_mask.imshow(self.current_frame.mask, cmap="gray")
        self.ax_mask.set_title("Ground Truth Mask")

        for ax, key in [(self.ax_opencv, "opencv"), (self.ax_pytorch, "pytorch"), (self.ax_openvino, "openvino")]:
            if key in results:
                result = results[key]
                ax.imshow(self.current_frame.image, cmap="inferno")
                ax.scatter([result.center_x], [result.center_y], c="red", s=100, marker="x")
                x, y, w, h = result.bbox
                rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor="yellow", linewidth=2)
                ax.add_patch(rect)
                ax.set_title(f"{key.upper()} | Conf: {result.confidence:.2f}")
            else:
                ax.text(0.5, 0.5, f"{key.upper()}\nN/A", ha="center", va="center")

        self.ax_metrics.axis("off")

        metrics_text = "Metrics Summary:\n\n"
        for key, result in results.items():
            gt_x, gt_y = self.current_frame.centers[0]
            error = MetricsCalculator.localization_error(result.center_x, result.center_y, gt_x, gt_y)
            fps = 1000.0 / result.inference_time_ms if result.inference_time_ms > 0 else 0.0

            metrics_text += f"{key.upper()}:\n"
            metrics_text += f"  Latency: {result.inference_time_ms:.2f} ms\n"
            metrics_text += f"  FPS: {fps:.1f}\n"
            metrics_text += f"  Error: {error:.2f} px\n"
            metrics_text += f"  Confidence: {result.confidence:.2f}\n\n"

            robot_coord = self.robot_mapper.pixel_to_robot(result.center_x, result.center_y)
            metrics_text += f"  Robot Target:\n"
            metrics_text += f"    X: {robot_coord.X:.3f}\n"
            metrics_text += f"    Y: {robot_coord.Y:.3f}\n"
            metrics_text += f"    Z: {robot_coord.Z:.3f}\n\n"

        self.ax_metrics.text(0.1, 0.9, metrics_text, transform=self.ax_metrics.transAxes, fontsize=8, verticalalignment="top", family="monospace")

        self.fig.tight_layout()
        self.canvas.draw()

        # Update Tkinter metrics text widget
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, metrics_text)
        self.metrics_text.config(state=tk.DISABLED)

    def _update_visualization(self) -> None:
        """Update main visualization after frame generation."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.imshow(self.current_frame.image, cmap="inferno")
        ax.set_title("Generated Thermal Image")
        if self.current_frame.centers:
            ax.scatter(
                [pos[0] for pos in self.current_frame.centers],
                [pos[1] for pos in self.current_frame.centers],
                c="cyan",
                s=100,
                marker="*",
            )
        self.fig.tight_layout()
        self.canvas.draw()

    def clear_history(self) -> None:
        """Clear metrics history."""
        self.metrics_history = {"opencv": [], "pytorch": [], "openvino": []}
        messagebox.showinfo("Info", "History cleared")


def main() -> None:
    root = tk.Tk()
    app = ThermalHotspotDemo(root)
    root.mainloop()


if __name__ == "__main__":
    main()
