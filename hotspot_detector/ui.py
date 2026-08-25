"""Desktop benchmark console for comparing OpenCV / PyTorch / OpenVINO hotspot
detection results on the same YOLO C-Cover detection, built with Tkinter + Matplotlib
to match the visual language used in thermal/ui.py (dark engineering theme).
"""

from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox
from pathlib import Path
import threading
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle, Polygon
from PIL import Image, ImageDraw, ImageFont, ImageTk

from hotspot_detector import data_loader
from hotspot_detector.cover_detection_mock import generate_yolo_cover_result
from hotspot_detector.model_descriptions import MODEL_DESCRIPTIONS
from thermal.detectors.opencv_detector import OpenCVHotspotDetector
from thermal.detectors.openvino_detector import OpenVINOYOLODetector
from thermal.detectors.yolo_detector import YOLOv8PyTorchDetector

# ── Design tokens (kept consistent with thermal/ui.py's dark engineering theme) ──
C: dict[str, str] = {
    "bg": "#0F1117",
    "card": "#20243A",
    "border": "#2A2F4A",
    "accent": "#60A5FA",
    "accent_dim": "#4F46E5",
    "success": "#10B981",
    "detected": "#32CD32",
    "cover_gt": "#E879F9",
    "cover_predicted": "#60A5FA",
    "thermal_surface": "#111B2A",
    "warning": "#F59E0B",
    "error": "#EF4444",
    "text": "#F1F5F9",
    "muted": "#94A3B8",
    "dim": "#475569",
    "header": "#161929",
}
FF = "Segoe UI"

MODEL_KEYS = ("opencv", "pytorch", "openvino")
SHORT_MODEL_NAMES = {"opencv": "OpenCV", "pytorch": "PyTorch", "openvino": "OpenVINO"}
CONFIDENCE_PASS_THRESHOLD = 85
IOU_PASS_THRESHOLD = 0.7
HOTSPOT_SUCCESS_DISTANCE_PX = 3.0
OPENVINO_MODEL_PATH = Path(__file__).resolve().parent.parent / "thermal" / "yolov8n_openvino_model" / "yolov8n.xml"
PYTORCH_MODEL_PATH = Path(__file__).resolve().parent.parent / "thermal" / "yolov8n.pt"


class DarkScrollbar(tk.Canvas):
    """A Canvas-drawn vertical scrollbar styled to match the dark theme, since the native
    tk.Scrollbar keeps the light OS look on Windows regardless of its color options."""

    def __init__(self, parent: tk.Misc, *, command, width: int = 12):
        super().__init__(parent, width=width, bg=C["bg"], highlightthickness=0, bd=0, cursor="arrow")
        self._command = command
        self._top = 0.0
        self._bottom = 1.0
        self._dragging = False
        self._drag_offset = 0.0

        self.bind("<Configure>", lambda _e: self._redraw())
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    def set(self, first: str, last: str) -> None:
        self._top = min(max(float(first), 0.0), 1.0)
        self._bottom = min(max(float(last), self._top), 1.0)
        self._redraw()

    def _thumb_bounds(self) -> tuple[float, float]:
        height = max(1.0, float(self.winfo_height()))
        top_y = height * self._top
        bottom_y = height * self._bottom
        min_thumb = 24.0
        if bottom_y - top_y < min_thumb:
            bottom_y = min(height, top_y + min_thumb)
            top_y = max(0.0, bottom_y - min_thumb)
        return top_y, bottom_y

    def _on_press(self, event) -> None:
        top_y, bottom_y = self._thumb_bounds()
        if top_y <= event.y <= bottom_y:
            self._dragging = True
            self._drag_offset = event.y - top_y
        else:
            self._jump_to(event.y - (bottom_y - top_y) / 2)

    def _on_drag(self, event) -> None:
        if self._dragging:
            self._jump_to(event.y - self._drag_offset)

    def _on_release(self, _event) -> None:
        self._dragging = False

    def _jump_to(self, desired_top_y: float) -> None:
        height = max(1.0, float(self.winfo_height()))
        top_y, bottom_y = self._thumb_bounds()
        thumb_height = max(1.0, bottom_y - top_y)
        clamped_top = min(max(desired_top_y, 0.0), height - thumb_height)
        fraction = clamped_top / max(1.0, height - thumb_height) if height > thumb_height else 0
        self._command("moveto", fraction)

    def _redraw(self) -> None:
        self.delete("all")
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        self.create_rectangle(0, 0, width, height, fill=C["card"], outline=C["card"])
        top_y, bottom_y = self._thumb_bounds()
        inset = 2
        self.create_rectangle(inset, top_y + inset, width - inset, bottom_y - inset,
                               fill=C["dim"], outline=C["dim"])


class Tooltip:
    """A small dark popup shown near the cursor while hovering over a widget."""

    def __init__(self, widget: tk.Widget, text: str):
        self._widget = widget
        self._text = text
        self._popup: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _event=None) -> None:
        if self._popup is not None:
            return
        x = self._widget.winfo_rootx() + self._widget.winfo_width() // 2
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 6
        self._popup = tk.Toplevel(self._widget)
        self._popup.overrideredirect(True)
        self._popup.attributes("-topmost", True)
        tk.Label(self._popup, text=self._text, bg="#0B0E16", fg=C["text"], font=(FF, 8),
                 padx=6, pady=3, highlightbackground=C["border"], highlightthickness=1).pack()
        self._popup.geometry(f"+{x}+{y}")

    def _hide(self, _event=None) -> None:
        if self._popup is not None:
            self._popup.destroy()
            self._popup = None


class RuntimeDropdown(tk.Frame):
    """Compact toolbar dropdown styled after the thermal UI's flat custom selector."""

    def __init__(self, parent: tk.Misc, *, options: tuple[str, ...], text_var: tk.StringVar, on_select) -> None:
        super().__init__(parent, bg="#111B2A", highlightbackground="#3A405A", highlightthickness=1, height=24, cursor="hand2")
        self.pack_propagate(False)
        self._menu_options = options
        self._text_var = text_var
        self._on_select = on_select
        self._menu = tk.Menu(
            self,
            tearoff=0,
            bg="#111B2A",
            fg=C["text"],
            activebackground="#1B2A40",
            activeforeground=C["text"],
            bd=0,
            relief=tk.FLAT,
            font=(FF, 8),
        )
        self._rebuild_menu()
        self._value_label = tk.Label(self, textvariable=text_var, bg="#111B2A", fg=C["text"], font=(FF, 8), padx=5, cursor="hand2")
        self._value_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._separator = tk.Frame(self, bg="#3A405A", width=1)
        self._separator.pack(side=tk.LEFT, fill=tk.Y, pady=4)
        self._arrow_label = tk.Label(self, text="▾", bg="#111B2A", fg=C["muted"], font=(FF, 10), padx=4, cursor="hand2")
        self._arrow_label.pack(side=tk.RIGHT, fill=tk.Y)
        for widget in (self, self._value_label, self._arrow_label):
            widget.bind("<Button-1>", self._open_menu)
            widget.bind("<Enter>", self._on_enter)
            widget.bind("<Leave>", self._on_leave)

    def _rebuild_menu(self) -> None:
        self._menu.delete(0, tk.END)
        for option in self._menu_options:
            self._menu.add_command(label=f" {option}", command=lambda value=option: self._select(value))

    def _select(self, value: str) -> None:
        try:
            self._text_var.set(value)
            self._on_select(value)
        except Exception as error:
            messagebox.showerror("Runtime Setting Error", str(error))

    def _open_menu(self, _event=None) -> None:
        try:
            self._menu.tk_popup(self.winfo_rootx(), self.winfo_rooty() + self.winfo_height())
        finally:
            try:
                self._menu.grab_release()
            except Exception:
                pass

    def _on_enter(self, _event=None) -> None:
        self.configure(bg="#1B2A40", highlightbackground=C["cover_predicted"])
        self._value_label.configure(bg="#1B2A40")
        self._arrow_label.configure(bg="#1B2A40", fg=C["text"])
        self._separator.configure(bg=C["cover_predicted"])

    def _on_leave(self, _event=None) -> None:
        self.configure(bg="#111B2A", highlightbackground="#3A405A")
        self._value_label.configure(bg="#111B2A")
        self._arrow_label.configure(bg="#111B2A", fg=C["muted"])
        self._separator.configure(bg="#3A405A")

    def update_options(self, options: tuple[str, ...]) -> None:
        self._menu_options = options
        self._rebuild_menu()


class HotspotBenchmarkApp:
    """Main application window: dataset navigation, per-model comparison panels,
    shared YOLO C-Cover card and a bottom benchmark summary table."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PhysicalAI  ·  Hotspot Dashboard")
        self.root.configure(bg=C["bg"])
        self.root.geometry("1920x1200")
        self.root.minsize(1200, 800)
        self._set_window_icon()

        self.image_list: list[str] = data_loader.list_images()
        self.gt_hotspot_map = data_loader.load_hotspot_ground_truth_map()
        self.image_index = 0
        self.run_counter = 0

        self.status_var = tk.StringVar(value="Ready")
        self.image_index_var = tk.StringVar(value="0 / 0")
        self.pytorchDeviceVar = tk.StringVar(value="CPU")
        self.pytorchPrecisionVar = tk.StringVar(value="FP32")
        self.openvinoDeviceVar = tk.StringVar(value="CPU")
        self.openvinoPrecisionVar = tk.StringVar(value="F32")
        self.overlay_vars = {
            "show_cover": tk.BooleanVar(value=True),
            "show_hotspot": tk.BooleanVar(value=True),
            "show_gt_cover": tk.BooleanVar(value=True),
            "show_gt_hotspot": tk.BooleanVar(value=True),
        }
        self.detectedCoverBlinkVar = tk.BooleanVar(value=True)

        self.figures: dict[str, plt.Figure] = {}
        self.axes: dict[str, plt.Axes] = {}
        self.canvases: dict[str, FigureCanvasTkAgg] = {}
        self.metrics_frames: dict[str, tk.Frame] = {}
        self.badge_labels: dict[str, tk.Label] = {}
        self.model_description_vars: dict[str, tk.StringVar] = {}
        self.detection_status_frame: Optional[tk.Frame] = None
        self.model_summary_frame: Optional[tk.Frame] = None
        self.model_cards: dict[str, tk.Frame] = {}
        self.dataset_summary: dict[str, dict] = {}
        # Populated once per "Run All Images" click (real PyTorch/OpenVINO inference across the
        # whole dataset). Browsing images afterwards (Check Results arrows) reuses this cache
        # instead of re-running inference.
        self._all_image_model_results: dict[str, dict] = {}
        self._all_image_cover_results: dict[str, dict] = {}


        # Benchmark Overview interaction state
        self.detection_canvas: Optional[tk.Canvas] = None
        self._scroll_canvas: Optional[tk.Canvas] = None
        self._detection_circle_images: list[ImageTk.PhotoImage] = []
        self._detectedLineVisible = True

        # Independent zoom/pan view state per panel (each model can be zoomed/panned on its own)
        self._view_states: dict[str, dict[str, float]] = {
            model_key: {"center_x": 0.0, "center_y": 0.0, "scale": 1.0} for model_key in MODEL_KEYS
        }
        self._min_scale = 1.0
        self._max_scale = 8.0
        self._pan_active = False
        self._pan_model_key: Optional[str] = None
        self._pan_start_px: tuple[float, float] = (0.0, 0.0)
        self._pan_start_center: tuple[float, float] = (0.0, 0.0)

        self.current_cover_box: Optional[dict] = None
        self.current_gt_hotspots: list[list[float]] = []
        self.current_cover_result: Optional[dict] = None
        self.current_model_results: dict[str, dict] = {}
        self._image_array: Optional[np.ndarray] = None
        self._openvino_detector: Optional[OpenVINOYOLODetector] = None
        self._pytorch_detector: Optional[YOLOv8PyTorchDetector] = None
        self._opencv_detector: Optional[OpenCVHotspotDetector] = None
        self._openvino_precision_dropdown: Optional[RuntimeDropdown] = None
        self._pytorch_precision_dropdown: Optional[RuntimeDropdown] = None
        self._status_blink_active = False
        self._status_blink_step = 0
        self._status_display_text = "Ready"
        self._benchmark_running = False

        self._build_header()
        self._build_body()

        if not self.image_list:
            messagebox.showerror("Dataset Error", "No thermal images were found under pseudo_ground_truth/test/images.")
            self._set_status("Error", C["error"])
        else:
            # Don't auto-run on startup - leave everything blank until the user clicks
            # "Run All Images", so it's clear that nothing has been computed yet.
            self.image_index_var.set(f"1 / {len(self.image_list)}")
            self._show_placeholder_state()

        self._schedule_detection_pulse()

    def _schedule_detection_pulse(self) -> None:
        def on_tick() -> None:
            try:
                if self.detectedCoverBlinkVar.get():
                    self._detectedLineVisible = not self._detectedLineVisible
                else:
                    self._detectedLineVisible = True
                if self.detection_canvas is not None and self.current_cover_result:
                    self._draw_detection_status_canvas()
            except Exception as error:
                print(f"[ui] Detection pulse update failed: {error}")
            finally:
                # Blink every 1 second: visible 1s, hidden 1s => full cycle ~2s.
                self.root.after(1000, on_tick)

        self.root.after(1000, on_tick)

    def _on_toggle_detected_cover_blink(self) -> None:
        if not self.detectedCoverBlinkVar.get():
            self._detectedLineVisible = True
        self._draw_detection_status_canvas()

    def _set_window_icon(self) -> None:
        """Draw a crisp hotspot/target glyph (Bootstrap Icons "bullseye"-style concentric rings)
        at high resolution and downsample it, so the window icon stays sharp at any taskbar size."""
        try:
            supersample = 8
            base_size = 32
            canvas_size = base_size * supersample
            image = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            center = canvas_size / 2
            ring_specs = [
                (0.94, (255, 107, 53, 255)),
                (0.68, (0, 0, 0, 0)),
                (0.42, (255, 107, 53, 255)),
                (0.16, (239, 68, 68, 255)),
            ]
            for radius_ratio, color in ring_specs:
                radius = center * radius_ratio
                draw.ellipse([center - radius, center - radius, center + radius, center + radius], fill=color)

            icon_sizes = (16, 24, 32, 48, 64)
            self._window_icon_images = [
                ImageTk.PhotoImage(image.resize((s, s), Image.LANCZOS)) for s in icon_sizes
            ]
            self.root.iconphoto(True, *self._window_icon_images)
        except Exception as error:
            print(f"[ui] Failed to set window icon: {error}")

    # ── Header / control panel ────────────────────────────────────────────

    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg=C["header"], height=64)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        title_text = "Hotspot Dashboard"
        try:
            title_font = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 18)
            title_bbox = title_font.getbbox(title_text)
            title_width = title_bbox[2] - title_bbox[0]
            title_height = title_bbox[3] - title_bbox[1]
            title_image = Image.new("RGBA", (title_width, title_height), (0, 0, 0, 0))
            title_mask = Image.new("L", (title_width, title_height), 0)
            ImageDraw.Draw(title_mask).text((-title_bbox[0], -title_bbox[1]), title_text, fill=255, font=title_font)
            title_gradient = Image.new("RGBA", (title_width, title_height), (0, 0, 0, 0))
            gradient_pixels = title_gradient.load()
            for x in range(title_width):
                ratio = x / max(1, title_width - 1)
                gradient_pixels[x, 0] = (
                    int(245 + (239 - 245) * ratio),
                    int(158 + (68 - 158) * ratio),
                    int(11 + (68 - 11) * ratio),
                    255,
                )
            for y in range(1, title_height):
                title_gradient.paste(title_gradient.crop((0, 0, title_width, 1)), (0, y))
            title_image.paste(title_gradient, (0, 0), title_mask)
            self._header_title_image = ImageTk.PhotoImage(title_image)
            tk.Label(header, image=self._header_title_image, bg=C["header"], bd=0).place(
                x=16, rely=0.5, y=2, anchor="w"
            )
        except Exception as error:
            print(f"[ui] Failed to render gradient header title: {error}")
            tk.Label(header, text=title_text, bg=C["header"], fg=C["warning"], font=(FF, 14, "bold")).place(
                x=16, rely=0.5, y=2, anchor="w"
            )

        right_controls = tk.Frame(header, bg=C["header"])
        right_controls.pack(side=tk.RIGHT, padx=(0, 16), pady=10)

        toolbar_font = tkfont.Font(family=FF, size=9, weight="bold")
        check_results_font = tkfont.Font(family=FF, size=8, weight="bold")
        check_segment_width = check_results_font.measure("Check Results") + check_results_font.measure("20 / 20") + 58
        status_segment_width = toolbar_font.measure("Status OpenVINO runtime updated (reload on next run)") + 16
        runtime_segment_width = 430
        run_segment_width = 122
        divider_total_width = (1 + 2 + 8) * 3
        segment_padding_width = 4 + 5 + 5 + 4
        control_strip_width = runtime_segment_width + run_segment_width + check_segment_width + status_segment_width + divider_total_width + segment_padding_width
        toolbar_width = control_strip_width + 2
        toolbar_height = 42

        toolbar_shell = tk.Canvas(
            right_controls,
            width=toolbar_width,
            height=toolbar_height,
            bg=C["header"],
            highlightthickness=0,
            bd=0,
        )
        toolbar_shell.pack(side=tk.RIGHT)

        shell_image = Image.new("RGBA", (toolbar_width, toolbar_height), (0, 0, 0, 0))
        shell_draw = ImageDraw.Draw(shell_image)
        shell_draw.rounded_rectangle(
            [0, 0, toolbar_width - 1, toolbar_height - 1],
            radius=3,
            fill=(13, 20, 32, 210),
            outline=(205, 210, 220, 52),
            width=1,
        )
        shell_draw.rounded_rectangle(
            [1, 1, toolbar_width - 2, toolbar_height // 2],
            radius=2,
            fill=(255, 255, 255, 10),
        )
        self._toolbar_shell_image = ImageTk.PhotoImage(shell_image)
        toolbar_shell.create_image(0, 0, image=self._toolbar_shell_image, anchor="nw")

        control_strip = tk.Frame(toolbar_shell, bg="#111B2A", width=control_strip_width, height=32)
        control_strip.pack_propagate(False)
        toolbar_shell.create_window(1, toolbar_height / 2, window=control_strip, anchor="w")

        runtime_segment = tk.Frame(control_strip, bg="#111B2A", width=runtime_segment_width, height=32)
        run_segment = tk.Frame(control_strip, bg="#111B2A", width=run_segment_width, height=32)
        run_segment.pack(side=tk.LEFT, padx=(4, 5))
        run_segment.pack_propagate(False)
        run_button = self._make_rounded_button(run_segment, "⚡ Run All Images", self._on_run_all)
        run_button.place(relx=0.5, rely=0.5, anchor="center")

        tk.Frame(control_strip, bg="#3A405A", width=1, height=22).pack(side=tk.LEFT, padx=5, pady=6)

        runtime_segment.pack(side=tk.LEFT, padx=0)
        runtime_segment.pack_propagate(False)
        self._build_runtime_settings(runtime_segment)

        tk.Frame(control_strip, bg="#3A405A", width=1, height=22).pack(side=tk.LEFT, padx=8, pady=6)

        check_segment = tk.Frame(control_strip, bg="#111B2A", width=check_segment_width, height=32)
        check_segment.pack(side=tk.LEFT, padx=(0, 5))
        check_segment.pack_propagate(False)

        self._make_split_button(
            check_segment, self.image_index_var, self._on_prev_image, self._on_next_image,
            left_tooltip="Previous image", right_tooltip="Next image",
            fixed_width=check_segment_width,
        ).pack(fill=tk.X, expand=True)
        tk.Frame(control_strip, bg="#3A405A", width=1, height=22).pack(side=tk.LEFT, padx=5, pady=6)

        status_segment = tk.Frame(control_strip, bg="#111B2A", width=status_segment_width, height=32)
        status_segment.pack(side=tk.LEFT, padx=0)
        status_segment.pack_propagate(False)

        info_bar = tk.Frame(status_segment, bg="#111B2A")
        info_bar.place(x=4, rely=0.5, anchor="w")
        tk.Label(info_bar, text="Status", bg="#111B2A", fg=C["muted"], font=(FF, 8, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        self._status_badge = tk.Frame(info_bar, bg="#111B2A", highlightthickness=0)
        self._status_label = tk.Label(
            self._status_badge,
            textvariable=self.status_var,
            bg="#111B2A",
            fg=C["text"],
            font=(FF, 8, "bold"),
            anchor="w",
            padx=8,
            pady=4,
        )
        self._status_label.pack(side=tk.LEFT)
        self._status_badge.pack(side=tk.LEFT)

        self._sync_check_results_height_to_status_badge()

    def _recenter_arrow_canvas(self, canvas: tk.Canvas, canvas_height: float) -> None:
        """Shift the arrow polygon so it stays vertically centered on the canvas's actual height
        (the base points were authored assuming a center-y of 15)."""
        base_points = canvas._arrow_base_points  # type: ignore[attr-defined]
        shift = canvas_height / 2 - 15
        shifted_points = [coord + shift if index % 2 == 1 else coord for index, coord in enumerate(base_points)]
        canvas.coords(canvas._arrow_item, *shifted_points)  # type: ignore[attr-defined]

    def _sync_check_results_height_to_status_badge(self) -> None:
        """Force the Check Results badge to the Status badge's real rendered pixel height, since
        font-metric estimates drift from actual layout (DPI scaling, label internal spacing)."""
        try:
            self.root.update_idletasks()
            badge_height = self._status_label.winfo_reqheight()
            if badge_height <= 1:
                return
            self._check_results_holder.configure(height=badge_height)
            for widget in (self._check_results_left, self._check_results_right):
                widget.configure(height=badge_height)
                self._recenter_arrow_canvas(widget, badge_height)
        except Exception as error:
            print(f"[ui] Failed to sync Check Results badge height: {error}")

    def _build_runtime_settings(self, parent: tk.Frame) -> None:
        """Build compact runtime selectors; changed settings take effect on the next run."""
        tk.Label(parent, text="Runtime Settings", bg="#111B2A", fg=C["muted"], font=(FF, 8, "bold")).pack(
            side=tk.LEFT, padx=(0, 10)
        )

        def add_runtime_group(
            label: str,
            device_var: tk.StringVar,
            device_options: tuple[str, ...],
            precision_var: tk.StringVar,
            precision_options: tuple[str, ...],
            callback,
        ) -> None:
            group = tk.Frame(parent, bg="#111B2A")
            group.pack(side=tk.LEFT, padx=(0, 10 if label == "PyTorch" else 0))
            tk.Label(group, text=f"{label}:", bg="#111B2A", fg=C["muted"], font=(FF, 8, "bold")).pack(side=tk.LEFT, padx=(0, 2))
            for variable, options in ((device_var, device_options), (precision_var, precision_options)):
                selector = RuntimeDropdown(group, options=options, text_var=variable, on_select=callback)
                selector.configure(width=52 if variable is device_var else 48)
                selector.pack(side=tk.LEFT, padx=0)
                if label == "OpenVINO" and variable is precision_var:
                    self._openvino_precision_dropdown = selector
                if label == "PyTorch" and variable is precision_var:
                    self._pytorch_precision_dropdown = selector

        add_runtime_group(
            "PyTorch", self.pytorchDeviceVar, ("CPU", "CUDA"), self.pytorchPrecisionVar, ("FP32", "FP16"), self._on_pytorch_runtime_change
        )
        add_runtime_group(
            "OpenVINO", self.openvinoDeviceVar, ("CPU", "GPU", "NPU", "AUTO"), self.openvinoPrecisionVar, ("F32", "F16", "Int8"), self._on_openvino_runtime_change
        )
        self._refresh_openvino_precision_options()
        self._refresh_pytorch_precision_options()

    def _on_pytorch_runtime_change(self, _selected: str) -> None:
        if self.pytorchDeviceVar.get().upper() == "CUDA":
            try:
                import torch
                if not torch.cuda.is_available():
                    self.pytorchDeviceVar.set("CPU")
                    self._refresh_pytorch_precision_options()
                    self._pytorch_detector = None
                    self._set_status("CUDA is unavailable; PyTorch reverted to CPU", C["warning"])
                    return
            except Exception as error:
                self.pytorchDeviceVar.set("CPU")
                print(f"[ui] CUDA availability check failed: {error}")
        self._refresh_pytorch_precision_options()
        self._pytorch_detector = None
        self._set_status("PyTorch runtime updated (reload on next run)", C["warning"])

    def _get_pytorch_precision_options(self, device: str) -> tuple[str, ...]:
        return ("FP32", "FP16") if device.strip().upper() == "CUDA" else ("FP32",)

    def _refresh_pytorch_precision_options(self) -> None:
        options = self._get_pytorch_precision_options(self.pytorchDeviceVar.get())
        if self.pytorchPrecisionVar.get() not in options:
            self.pytorchPrecisionVar.set(options[0])
        if self._pytorch_precision_dropdown is not None:
            self._pytorch_precision_dropdown.update_options(options)

    def _on_openvino_runtime_change(self, _selected: str) -> None:
        device = self.openvinoDeviceVar.get().strip().upper()
        self._refresh_openvino_precision_options()
        self._openvino_detector = None
        self._set_status("OpenVINO runtime updated (reload on next run)", C["warning"])

    def _get_openvino_precision_options(self, device: str) -> tuple[str, ...]:
        device = device.strip().upper()
        if device == "GPU":
            return ("F32", "F16")
        if device == "NPU":
            return ("F16", "Int8")
        if device == "AUTO":
            return ("F32", "F16")
        return ("BF16", "F32", "F16")

    def _refresh_openvino_precision_options(self) -> None:
        options = self._get_openvino_precision_options(self.openvinoDeviceVar.get())
        if self.openvinoPrecisionVar.get() not in options:
            self.openvinoPrecisionVar.set(options[0])
        if self._openvino_precision_dropdown is not None:
            self._openvino_precision_dropdown.update_options(options)

    def _make_button(self, parent: tk.Frame, text: str, command, accent: bool = False) -> tk.Label:
        bg = "#0E639C" if accent else C["header"]
        fg = "#FFFFFF" if accent else C["text"]
        btn = tk.Label(parent, text=text, bg=bg, fg=fg, font=(FF, 9, "bold"),
                       padx=14, pady=6, cursor="hand2",
                       bd=0, relief=tk.FLAT,
                       highlightbackground="#3794FF" if accent else C["header"],
                       highlightthickness=1 if accent else 0)
        hover_bg = "#1177BB" if accent else "#2A314A"
        btn.bind("<Button-1>", lambda _e: command())
        btn.bind("<Enter>", lambda _e: btn.config(bg=hover_bg))
        btn.bind("<Leave>", lambda _e: btn.config(bg=bg))
        return btn

    def _make_rounded_button(self, parent: tk.Frame, text: str, command) -> tk.Canvas:
        """Create a true rounded-corner toolbar button using Canvas + PIL raster rendering."""
        width = 120
        height = 30
        radius = 3
        bg_default = "#0E639C"
        bg_hover = "#1177BB"

        canvas = tk.Canvas(
            parent,
            width=width,
            height=height,
            bg="#232323",
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )

        def make_button_image(fill_color: str) -> ImageTk.PhotoImage:
            image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle(
                [1, 1, width - 2, height - 2],
                radius=radius,
                fill=fill_color,
            )
            return ImageTk.PhotoImage(image)

        normal_image = make_button_image(bg_default)
        hover_image = make_button_image(bg_hover)

        # Keep image references alive on the widget.
        canvas._normal_image = normal_image  # type: ignore[attr-defined]
        canvas._hover_image = hover_image  # type: ignore[attr-defined]

        bg_item = canvas.create_image(0, 0, image=normal_image, anchor="nw")
        text_item = canvas.create_text(width / 2, height / 2, text=text, fill="#FFFFFF", font=(FF, 8, "bold"))

        def on_enter(_event=None) -> None:
            canvas.itemconfigure(bg_item, image=hover_image)

        def on_leave(_event=None) -> None:
            canvas.itemconfigure(bg_item, image=normal_image)

        def on_click(_event=None) -> None:
            try:
                command()
            except Exception as error:
                print(f"[ui] Rounded button command failed: {error}")

        for target in (canvas,):
            target.bind("<Enter>", on_enter)
            target.bind("<Leave>", on_leave)
            target.bind("<Button-1>", on_click)

        canvas.tag_bind(text_item, "<Enter>", on_enter)
        canvas.tag_bind(text_item, "<Leave>", on_leave)
        canvas.tag_bind(text_item, "<Button-1>", on_click)

        return canvas

    def _make_split_button(self, parent: tk.Frame, center_var: tk.StringVar, left_command, right_command,
                            *, left_tooltip: str = "", right_tooltip: str = "", fixed_width: Optional[int] = None) -> tk.Frame:
        """Check-results toolbar segment with text/index followed by compact arrow controls."""
        # Match the Status badge's natural height exactly: same font size + same vertical padding.
        badge_font = tkfont.Font(family=FF, size=8, weight="bold")
        badge_pady = 4
        badge_height = badge_font.metrics("linespace") + badge_pady * 2
        holder = tk.Frame(parent, bg="#111B2A", height=badge_height)
        self._check_results_holder = holder
        self._check_results_bg = "#111B2A"
        self._check_results_hover_bg = "#1B2A40"
        if fixed_width is not None:
            holder.configure(width=fixed_width, height=badge_height)
            holder.pack_propagate(False)

        def make_half(direction: str, command, tooltip: str) -> tk.Canvas:
            half = tk.Canvas(
                holder,
                width=20,
                height=badge_height,
                bg="#111B2A",
                highlightthickness=0,
                bd=0,
                cursor="hand2",
            )
            # Points assume a canvas center-y of 15; _recenter_arrow_canvas() shifts them to match the real badge height.
            if direction == "left":
                arrow_points = (16, 14, 8.6, 14, 11.4, 11.2, 10, 9.8, 4.8, 15,
                                10, 20.2, 11.4, 18.8, 8.6, 16, 16, 16)
            else:
                arrow_points = (4, 14, 11.4, 14, 8.6, 11.2, 10, 9.8, 15.2, 15,
                                10, 20.2, 8.6, 18.8, 11.4, 16, 4, 16)
            arrow_item = half.create_polygon(arrow_points, fill="#D4D4D4", outline="")
            half._arrow_item = arrow_item  # type: ignore[attr-defined]
            half._arrow_base_points = arrow_points  # type: ignore[attr-defined]
            self._recenter_arrow_canvas(half, badge_height)

            def on_click(_event=None) -> None:
                try:
                    command()
                except Exception as error:
                    print(f"[ui] Image navigation failed: {error}")

            half.bind("<Button-1>", on_click)
            half.bind("<Enter>", lambda _e: half.config(bg=self._check_results_hover_bg))
            half.bind("<Leave>", lambda _e: half.config(bg=self._check_results_bg))
            if tooltip:
                Tooltip(half, tooltip)
            return half

        self._check_results_left = make_half("left", left_command, left_tooltip)
        self._check_results_label = tk.Label(holder, text="Check Results", bg="#111B2A", fg="#D4D4D4",
                      font=badge_font, pady=badge_pady)
        self._check_results_label.pack(side=tk.LEFT, padx=(0, 4))
        self._check_results_center = tk.Label(holder, textvariable=center_var, bg="#111B2A", fg="#D4D4D4",
                       font=badge_font, pady=badge_pady, width=5, anchor="w")
        self._check_results_center.pack(side=tk.LEFT, padx=(0, 3))
        self._check_results_left.pack(side=tk.LEFT)
        self._check_results_right = make_half("right", right_command, right_tooltip)
        self._check_results_right.pack(side=tk.LEFT)
        self._set_check_results_active(False)
        return holder

    def _set_check_results_active(self, active: bool) -> None:
        """Style the Previous/Next/index control as 'active' (a run has completed, browsing is
        available, filled with the same green/white treatment as the 'Completed' status badge)
        or dimmed/inactive (no run yet, so there is nothing to browse)."""
        bg_color = "#16825D" if active else "#111B2A"
        hover_color = "#1C9C6E" if active else "#1B2A40"
        text_color = "#FFFFFF" if active else C["dim"]
        self._check_results_bg = bg_color
        self._check_results_hover_bg = hover_color
        self._check_results_holder.configure(bg=bg_color)
        for widget in (self._check_results_left, self._check_results_right):
            widget.configure(bg=bg_color)
            widget.itemconfigure(widget._arrow_item, fill=text_color)
        self._check_results_label.configure(bg=bg_color, fg=text_color)
        self._check_results_center.configure(bg=bg_color, fg=text_color)


    def _make_checkbox(
        self,
        parent: tk.Frame,
        text: str,
        var: tk.BooleanVar,
        bg: str = C["header"],
        fg: str = C["text"],
        command: Optional[Callable[[], None]] = None,
    ) -> tk.Checkbutton:
        callback = command if command is not None else self._redraw_panels
        return tk.Checkbutton(
            parent, text=text, variable=var, command=callback,
            bg=bg, fg=fg, selectcolor=bg,
            activebackground=bg, activeforeground=fg,
            font=(FF, 8), highlightthickness=0, bd=0,
        )

    def _on_overlay_toggle(self) -> None:
        """Partial refresh for display options: redraw only thermal panels to reduce flicker."""
        if self._image_array is None:
            return
        for model_key in MODEL_KEYS:
            self._draw_model_panel(model_key)

    def _make_eye_toggle(self, parent: tk.Frame, label: str, var: tk.BooleanVar, *, bg: str = C["bg"]) -> tk.Button:
        button = tk.Button(
            parent,
            text="",
            command=lambda: None,
            bg=bg,
            activebackground=bg,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            font=(FF, 9),
            cursor="hand2",
            padx=8,
            pady=2,
        )

        def refresh() -> None:
            is_enabled = var.get()
            icon = "👁" if is_enabled else "🚫"
            button.configure(text=f"{icon} {label}", fg=C["text"] if is_enabled else C["dim"])

        def on_click() -> None:
            var.set(not var.get())
            refresh()
            self._redraw_panels()

        button.configure(command=on_click)
        refresh()
        return button

    # ── Body layout ────────────────────────────────────────────────────────

    def _build_body(self) -> None:
        """Wrap the whole body in a scrollable Canvas so users can scroll down to see the
        model panels when the screen height isn't tall enough to show everything at once."""
        scroll_outer = tk.Frame(self.root, bg=C["bg"])
        scroll_outer.pack(fill=tk.BOTH, expand=True)

        scroll_canvas = tk.Canvas(scroll_outer, bg=C["bg"], highlightthickness=0)
        scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._scroll_canvas = scroll_canvas

        # Plain tk.Scrollbar keeps the light OS theme on Windows regardless of color options,
        # which clashed with the dark UI - use a custom Canvas-drawn scrollbar instead.
        scrollbar = DarkScrollbar(scroll_outer, command=scroll_canvas.yview, width=12)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        body = tk.Frame(scroll_canvas, bg=C["bg"])
        body_window = scroll_canvas.create_window((10, 0), anchor="nw", window=body)

        def _sync_scrollregion(_event=None) -> None:
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        def _sync_body_width(event) -> None:
            scroll_canvas.itemconfigure(body_window, width=max(1, event.width - 20))

        body.bind("<Configure>", _sync_scrollregion)
        scroll_canvas.bind("<Configure>", _sync_body_width)

        self._build_benchmark_overview(body)

        # Keep display controls and the three thermal model panels within one shared surface.
        thermal_section = tk.Frame(
            body,
            bg=C["thermal_surface"],
            highlightbackground=C["border"],
            highlightcolor=C["border"],
            highlightthickness=1,
        )
        thermal_section.pack(fill=tk.BOTH, expand=True)

        # Overlay toggles live directly above the thermal panels since they only affect what's
        # drawn inside those images, rather than sitting in the top control panel.
        overlay_toolbar = tk.Canvas(
            thermal_section,
            height=42,
            bg=C["thermal_surface"],
            highlightthickness=0,
            bd=0,
        )
        overlay_toolbar.pack(fill=tk.X)

        overlay_row = tk.Frame(overlay_toolbar, bg="#111B2A", height=32)
        overlay_row_window = overlay_toolbar.create_window(1, 21, window=overlay_row, anchor="w", height=32)

        def draw_overlay_toolbar(event) -> None:
            try:
                width = max(2, event.width)
                height = max(2, event.height)
                toolbar_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                toolbar_draw = ImageDraw.Draw(toolbar_image)
                toolbar_draw.rounded_rectangle(
                    [0, 0, width - 1, height - 1],
                    radius=3,
                    fill=(13, 20, 32, 210),
                    outline=(205, 210, 220, 52),
                    width=1,
                )
                toolbar_draw.rounded_rectangle(
                    [1, 1, width - 2, height // 2],
                    radius=2,
                    fill=(255, 255, 255, 10),
                )
                self._overlay_toolbar_image = ImageTk.PhotoImage(toolbar_image)
                if hasattr(self, "_overlay_toolbar_image_item"):
                    overlay_toolbar.itemconfigure(self._overlay_toolbar_image_item, image=self._overlay_toolbar_image)
                else:
                    self._overlay_toolbar_image_item = overlay_toolbar.create_image(
                        0, 0, image=self._overlay_toolbar_image, anchor="nw"
                    )
                    overlay_toolbar.tag_lower(self._overlay_toolbar_image_item)
                overlay_toolbar.itemconfigure(overlay_row_window, width=max(1, width - 2))
            except Exception as error:
                print(f"[ui] Failed to draw thermal display toolbar: {error}")

        overlay_toolbar.bind("<Configure>", draw_overlay_toolbar)
        options_group = tk.Frame(overlay_row, bg="#111B2A")
        options_group.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(
            options_group,
            text="IR Thermal Display Options",
            bg="#111B2A",
            fg=C["muted"],
            font=(FF, 8, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 10), pady=6)
        tk.Frame(options_group, bg="#3A405A", width=1, height=22).pack(side=tk.LEFT, padx=(0, 10), pady=5)

        def add_overlay_group(
            label: str,
            groundTruthVar: tk.BooleanVar,
            detectedVar: tk.BooleanVar,
            groundTruthColor: str,
            predictedColor: str,
        ) -> None:
            group = tk.Frame(options_group, bg="#111B2A")
            group.pack(side=tk.LEFT)
            tk.Label(group, text=f"{label}:", bg="#111B2A", fg=C["muted"], font=(FF, 8, "bold")).pack(
                side=tk.LEFT, padx=(0, 4), pady=6
            )
            self._make_checkbox(
                group,
                "Ground Truth",
                groundTruthVar,
                bg="#111B2A",
                fg=groundTruthColor,
                command=self._on_overlay_toggle,
            ).pack(side=tk.LEFT, padx=(0, 5), pady=3)
            self._make_checkbox(
                group,
                "Predicted",
                detectedVar,
                bg="#111B2A",
                fg=predictedColor,
                command=self._on_overlay_toggle,
            ).pack(side=tk.LEFT, pady=3)

        add_overlay_group(
            "C-Cover", self.overlay_vars["show_gt_cover"], self.overlay_vars["show_cover"],
            C["cover_gt"], C["cover_predicted"]
        )
        tk.Frame(options_group, bg="#3A405A", width=1, height=22).pack(side=tk.LEFT, padx=16, pady=5)
        add_overlay_group(
            "Hotspot", self.overlay_vars["show_gt_hotspot"], self.overlay_vars["show_hotspot"],
            C["cover_gt"], C["cover_predicted"]
        )

        panels_row = tk.Frame(thermal_section, bg=C["thermal_surface"])
        panels_row.pack(fill=tk.BOTH, expand=True)
        for i in range(3):
            panels_row.columnconfigure(i, weight=1, uniform="panel")

        for col_index, model_key in enumerate(MODEL_KEYS):
            model_def = MODEL_DESCRIPTIONS[model_key]
            self._build_model_panel(panels_row, col_index, model_key, model_def)

        # Global mouse-wheel scrolling, but skip the thermal image panels so they keep
        # their own scroll-to-zoom behavior instead of paging the whole window.
        self.root.bind_all("<MouseWheel>", self._on_page_scroll)

    def _on_page_scroll(self, event) -> None:
        try:
            figure_widgets = {canvas.get_tk_widget() for canvas in self.canvases.values()}
            if event.widget in figure_widgets or self._scroll_canvas is None:
                return
            self._scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception as error:
            print(f"[ui] Page scroll failed: {error}")

    def _build_benchmark_overview(self, parent: tk.Frame) -> None:
        """Build two independent, titled panels side-by-side directly below the control panel:
        'C-Cover Detection Status' (left, the virtual predicted bounding box visualization) and
        'Execution Summary for All Images' (right, the 3 per-model cards with integrated winner badges)."""
        overview_section = tk.Frame(
            parent,
            bg=C["thermal_surface"],
            highlightbackground=C["border"],
            highlightcolor=C["border"],
            highlightthickness=1,
        )
        overview_section.pack(fill=tk.X, pady=(0, 10))

        row = tk.Frame(overview_section, bg=C["thermal_surface"])
        row.pack(fill=tk.X)
        row.columnconfigure(0, weight=1, uniform="overview_row", minsize=0)
        row.columnconfigure(1, weight=2, uniform="overview_row", minsize=0)

        self.detection_status_frame = self._make_card_grid(
            row,
            column=0,
            padx=(0, 0),
            title="C-Cover Detection Status",
            titleRightBuilder=self._build_detection_status_header_controls,
            bg=C["thermal_surface"],
            bordered=True,
        )
        self.model_summary_frame = self._make_card_grid(
            row,
            column=1,
            padx=(2, 0),
            title="Execution Summary for All Images",
            titleRightBuilder=self._build_summary_note,
            bg=C["thermal_surface"],
            bordered=True,
        )

    def _make_card_grid(
        self,
        parent: tk.Frame,
        *,
        column: int,
        padx: tuple[int, int],
        title: str,
        titleRightBuilder: Optional[Callable[[tk.Frame], None]] = None,
        bg: str = C["card"],
        bordered: bool = True,
    ) -> tk.Frame:
        """Same bordered card styling as _make_card, but gridded (for side-by-side panels)
        and topped with its own title label."""
        card = tk.Frame(
            parent,
            bg=bg,
            highlightbackground=C["border"] if bordered else bg,
            highlightthickness=1 if bordered else 0,
        )
        card.grid(row=0, column=column, sticky="nsew", padx=padx)
        inner = tk.Frame(card, bg=bg)
        inner.pack(fill=tk.BOTH, expand=True, padx=0, pady=(0, 3))
        title_row = tk.Frame(inner, bg=bg)
        title_row.pack(fill=tk.X, pady=(0, 4))
        tk.Label(title_row, text=title, bg=bg, fg=C["text"], font=(FF, 11, "bold")).pack(side=tk.LEFT, anchor=tk.N)
        if titleRightBuilder is not None:
            titleRightBuilder(title_row)
        content = tk.Frame(inner, bg=bg)
        content.pack(fill=tk.BOTH, expand=True)
        return content

    def _build_detection_status_header_controls(self, parent: tk.Frame) -> None:
        tk.Checkbutton(
            parent,
            text="Blink Predicted",
            variable=self.detectedCoverBlinkVar,
            command=self._on_toggle_detected_cover_blink,
            bg=C["thermal_surface"],
            fg=C["muted"],
            selectcolor=C["thermal_surface"],
            activebackground=C["thermal_surface"],
            activeforeground=C["muted"],
            font=(FF, 8),
            highlightthickness=0,
            bd=0,
        ).pack(side=tk.RIGHT)

    def _build_summary_note(self, parent: tk.Frame) -> None:
        guide_button = tk.Button(
            parent,
            text="Metric Guide",
            command=lambda: self._show_summary_metric_guide(guide_button),
            bg=C["thermal_surface"],
            fg=C["muted"],
            activebackground=C["thermal_surface"],
            activeforeground=C["text"],
            font=(FF, 8),
            relief=tk.FLAT,
            bd=0,
            highlightbackground=C["border"],
            highlightthickness=1,
            padx=6,
            pady=1,
            cursor="hand2",
        )
        guide_button.pack(side=tk.RIGHT, anchor=tk.N)
        guide_button.bind(
            "<Enter>",
            lambda _event: guide_button.configure(bg="#1B2A40", fg=C["text"], highlightbackground=C["cover_predicted"]),
        )
        guide_button.bind(
            "<Leave>",
            lambda _event: guide_button.configure(bg=C["thermal_surface"], fg=C["muted"], highlightbackground=C["border"]),
        )

    def _show_summary_metric_guide(self, trigger_button: tk.Button) -> None:
        try:
            guide = tk.Toplevel(self.root)
            guide.overrideredirect(True)
            guide.configure(bg=C["thermal_surface"])
            guide.resizable(False, False)
            guide.transient(self.root)

            content = tk.Frame(guide, bg=C["thermal_surface"], highlightbackground=C["border"], highlightthickness=1)
            content.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
            tk.Label(content, text="Summary Metrics", bg=C["thermal_surface"], fg=C["text"], font=(FF, 12, "bold")).pack(
                anchor=tk.W, padx=10, pady=(8, 6)
            )
            metric_items = (
                ("Overall Score", "Success Rate - Avg Error x10 - Inference Time x0.5"),
                ("Avg Error", "Mean hotspot pixel error"),
                ("Avg FPS", "1000 / Average Inference Time (ms)"),
                ("Success Rate", "Hotspots matched within 3px"),
                ("Avg IoU", "Mean C-Cover intersection over union"),
                ("mAP@3px", "Confidence-ranked average precision at 3px"),
            )
            for label, description in metric_items:
                row = tk.Frame(content, bg=C["thermal_surface"])
                row.pack(fill=tk.X, padx=10, pady=3)
                tk.Label(row, text=f"{label}: ", bg=C["thermal_surface"], fg=C["text"], font=(FF, 10, "bold")).pack(side=tk.LEFT)
                tk.Label(row, text=description, bg=C["thermal_surface"], fg=C["muted"], font=(FF, 10)).pack(side=tk.LEFT)
            tk.Button(
                content,
                text="Close",
                command=guide.destroy,
                bg=C["thermal_surface"],
                fg=C["muted"],
                activebackground=C["thermal_surface"],
                activeforeground=C["text"],
                font=(FF, 9),
                relief=tk.FLAT,
                bd=0,
                highlightbackground=C["border"],
                highlightthickness=1,
                padx=8,
                pady=2,
                cursor="hand2",
            ).pack(anchor=tk.E, padx=10, pady=(6, 8))
            guide.update_idletasks()
            popup_width = guide.winfo_reqwidth()
            popup_x = trigger_button.winfo_rootx() + trigger_button.winfo_width() - popup_width
            popup_y = trigger_button.winfo_rooty() + trigger_button.winfo_height() + 4
            popup_x = max(8, min(popup_x, guide.winfo_screenwidth() - popup_width - 8))
            guide.geometry(f"+{popup_x}+{popup_y}")
            guide.deiconify()
            guide.lift(self.root)
            guide.attributes("-topmost", True)
            guide.after_idle(lambda: guide.attributes("-topmost", False))
            guide.bind("<Escape>", lambda _event: guide.destroy())
            guide.focus_force()
        except Exception as error:
            print(f"[ui] Failed to show summary metric guide: {error}")

    def _build_model_panel(self, parent: tk.Frame, col_index: int, model_key: str, model_def: dict) -> None:
        column = tk.Frame(parent, bg=C["thermal_surface"])
        column.grid(row=0, column=col_index, sticky="nsew", padx=(0 if col_index == 0 else 2, 0))

        image_card = tk.Frame(
            column,
            bg=C["thermal_surface"],
            highlightbackground=C["border"],
            highlightthickness=1,
        )
        image_card.pack(fill=tk.BOTH, expand=False, pady=(0, 4))
        title_row = tk.Frame(image_card, bg=C["thermal_surface"])
        title_row.pack(fill=tk.X)
        tk.Label(title_row, text=model_def["title"], bg=C["thermal_surface"], fg=C["text"], font=(FF, 11, "bold")).pack(side=tk.LEFT)
        badge_label = tk.Label(
            title_row,
            text="",
            bg=C["thermal_surface"],
            fg=C["muted"],
            font=(FF, 8, "bold"),
            padx=7,
            pady=2,
            highlightthickness=1,
        )
        badge_label.pack(side=tk.RIGHT)
        self.badge_labels[model_key] = badge_label
        description_var = tk.StringVar(value=model_def["description"])
        self.model_description_vars[model_key] = description_var
        tk.Label(image_card, textvariable=description_var, bg=C["thermal_surface"], fg=C["muted"], font=(FF, 8)).pack(
            anchor=tk.W, pady=(0, 4)
        )

        figure = plt.Figure(figsize=(5.6, 4.2), dpi=100, facecolor=C["thermal_surface"])
        axes = figure.add_subplot(111)
        figure.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        axes.set_facecolor(C["thermal_surface"])
        axes.axis("off")
        image_content = tk.Frame(image_card, bg=C["thermal_surface"])
        image_content.pack(fill=tk.BOTH, expand=True)
        canvas_widget_holder = tk.Frame(image_content, bg=C["thermal_surface"])
        canvas_widget_holder.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas = FigureCanvasTkAgg(figure, master=canvas_widget_holder)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Bind with model_key captured per-canvas so each panel zooms/pans independently.
        canvas.mpl_connect("scroll_event", lambda event, mk=model_key: self._on_scroll(event, mk))
        canvas.mpl_connect("button_press_event", lambda event, mk=model_key: self._on_pan_press(event, mk))
        canvas.mpl_connect("motion_notify_event", lambda event, mk=model_key: self._on_pan_motion(event, mk))
        canvas.mpl_connect("button_release_event", lambda event, mk=model_key: self._on_pan_release(event, mk))
        self.figures[model_key] = figure
        self.axes[model_key] = axes
        self.canvases[model_key] = canvas

        self._build_zoom_control(image_content, model_key)

        metrics_card = self._make_card(column, pady=(0, 0), bg=C["thermal_surface"])
        self.metrics_frames[model_key] = metrics_card

    def _make_card(
        self,
        parent: tk.Frame,
        *,
        pady: tuple[int, int],
        inner_padx: int = 14,
        inner_pady: int = 10,
        bg: str = C["card"],
    ) -> tk.Frame:
        card = tk.Frame(parent, bg=bg, highlightbackground=C["border"], highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=False, pady=pady)
        inner = tk.Frame(card, bg=bg)
        inner.pack(fill=tk.BOTH, expand=True, padx=inner_padx, pady=inner_pady)
        return inner

    # ── Data loading / navigation ────────────────────────────────────────

    def _current_image_file(self) -> Optional[str]:
        if not self.image_list:
            return None
        return self.image_list[self.image_index]

    def _set_status(self, text: str, color: str) -> None:
        self._status_display_text = text
        self.status_var.set(text)
        self._set_status_blink_active(text == "Processing")
        badge_bg = "#111B2A" if text == "Ready" else "#007ACC"
        if color == C["warning"]:
            badge_bg = "#B89500"
        elif color == C["success"]:
            badge_bg = "#16825D"
        elif color == C["error"]:
            badge_bg = "#A1260D"
        self._status_badge.configure(bg=badge_bg)
        self._status_label.configure(bg=badge_bg, fg="#FFFFFF")

    def _set_status_blink_active(self, active: bool) -> None:
        self._status_blink_active = active
        self._status_blink_step = 0
        if active:
            self._animate_status_blink()
        else:
            self.status_var.set(self._status_display_text)
            self._status_label.configure(fg=C["text"])

    def _animate_status_blink(self) -> None:
        if not self._status_blink_active:
            return
        try:
            dot_count = self._status_blink_step % 3 + 1
            self.status_var.set(f"Processing{'.' * dot_count}")
            self._status_blink_step += 1
            self.root.after(350, self._animate_status_blink)
        except Exception as error:
            self._status_blink_active = False
            print(f"[ui] Status blink animation failed: {error}")

    def _load_current_image(self) -> None:
        image_file = self._current_image_file()
        if not image_file:
            return
        self._set_status("Processing", C["warning"])
        self.root.update_idletasks()

        try:
            image_array = data_loader.load_thermal_image(image_file)
            if image_array is None:
                raise RuntimeError(f"Could not load thermal image: {image_file}")
            height, width = image_array.shape[:2]

            cover_box = data_loader.load_yolo_cover_box(image_file, width, height)
            if cover_box is None:
                raise RuntimeError(f"Could not load YOLO C-Cover ground truth for: {image_file}")

            gt_hotspots = self.gt_hotspot_map.get(image_file, [])

            self._image_array = image_array
            self.current_cover_box = cover_box
            self.current_gt_hotspots = gt_hotspots
            # Reset each panel's independent zoom/pan back to the full image view.
            self._view_states = {
                model_key: {"center_x": width / 2, "center_y": height / 2, "scale": 1.0}
                for model_key in MODEL_KEYS
            }

            self._load_cached_results_for_image(image_file)
            self._redraw_panels()

            self.image_index_var.set(f"{self.image_index + 1} / {len(self.image_list)}")
            self._set_status("Completed", C["success"])
        except Exception as error:
            messagebox.showerror("Load Error", str(error))
            self._set_status("Error", C["error"])

    def _get_openvino_detector(self) -> OpenVINOYOLODetector:
        if self._openvino_detector is None:
            if not OPENVINO_MODEL_PATH.is_file():
                raise FileNotFoundError(f"OpenVINO model not found: {OPENVINO_MODEL_PATH}")
            self._openvino_detector = OpenVINOYOLODetector(
                model_path=str(OPENVINO_MODEL_PATH),
                device=self.openvinoDeviceVar.get(),
                inference_precision_hint=self.openvinoPrecisionVar.get(),
            )
        return self._openvino_detector

    def _get_pytorch_detector(self) -> YOLOv8PyTorchDetector:
        if self._pytorch_detector is None:
            if not PYTORCH_MODEL_PATH.is_file():
                raise FileNotFoundError(f"PyTorch model not found: {PYTORCH_MODEL_PATH}")
            self._pytorch_detector = YOLOv8PyTorchDetector(
                model_name=str(PYTORCH_MODEL_PATH),
                device=self.pytorchDeviceVar.get().lower(),
                half_precision=self.pytorchPrecisionVar.get().upper() == "FP16",
            )
        return self._pytorch_detector

    def _get_opencv_detector(self) -> OpenCVHotspotDetector:
        if self._opencv_detector is None:
            self._opencv_detector = OpenCVHotspotDetector()
        return self._opencv_detector

    @staticmethod
    def _to_thermal_matrix(image_array: np.ndarray) -> np.ndarray:
        if image_array.ndim == 2:
            return image_array.astype(np.float32)
        return np.mean(image_array[..., :3], axis=2, dtype=np.float32)

    def _run_openvino_result(self, image_array: np.ndarray, gt_hotspots: list[list[float]]) -> dict:
        detector_result = self._get_openvino_detector().detect(self._to_thermal_matrix(image_array))
        predicted_x = float(detector_result.center_x)
        predicted_y = float(detector_result.center_y)
        if not gt_hotspots:
            return {
                "hotspots": [],
                "success_rate": 0.0,
                "inference_time_ms": detector_result.inference_time_ms,
                "fps": round(1000 / max(detector_result.inference_time_ms, 0.001), 1),
                "runtime_device": self._get_openvino_detector().get_execution_device_text(),
                "runtime_precision": self._get_openvino_detector().applied_precision_hint,
            }

        nearest_index, (ground_truth_x, ground_truth_y) = min(
            enumerate(gt_hotspots),
            key=lambda item: float(np.hypot(predicted_x - item[1][0], predicted_y - item[1][1])),
        )
        error_px = round(float(np.hypot(predicted_x - ground_truth_x, predicted_y - ground_truth_y)))
        is_success = error_px <= HOTSPOT_SUCCESS_DISTANCE_PX
        return {
            "hotspots": [{
                "id": nearest_index + 1,
                "coordinate": (predicted_x, predicted_y),
                "temperature": detector_result.max_temperature,
                "ground_truth": (ground_truth_x, ground_truth_y),
                "error_px": error_px,
                "confidence": detector_result.confidence,
                "is_success": is_success,
            }],
            "success_rate": round(100 * int(is_success) / len(gt_hotspots), 1),
            "inference_time_ms": detector_result.inference_time_ms,
            "fps": round(1000 / max(detector_result.inference_time_ms, 0.001), 1),
            "runtime_device": self._get_openvino_detector().get_execution_device_text(),
            "runtime_precision": self._get_openvino_detector().applied_precision_hint,
        }

    def _run_opencv_result(self, image_array: np.ndarray, gt_hotspots: list[list[float]]) -> dict:
        detector_result = self._get_opencv_detector().detect(self._to_thermal_matrix(image_array))
        predicted_x = float(detector_result.center_x)
        predicted_y = float(detector_result.center_y)
        if not gt_hotspots:
            return {
                "hotspots": [],
                "success_rate": 0.0,
                "inference_time_ms": detector_result.inference_time_ms,
                "fps": round(1000 / max(detector_result.inference_time_ms, 0.001), 1),
            }

        nearest_index, (ground_truth_x, ground_truth_y) = min(
            enumerate(gt_hotspots),
            key=lambda item: float(np.hypot(predicted_x - item[1][0], predicted_y - item[1][1])),
        )
        error_px = round(float(np.hypot(predicted_x - ground_truth_x, predicted_y - ground_truth_y)))
        is_success = error_px <= HOTSPOT_SUCCESS_DISTANCE_PX
        return {
            "hotspots": [{
                "id": nearest_index + 1,
                "coordinate": (predicted_x, predicted_y),
                "temperature": detector_result.max_temperature,
                "ground_truth": (ground_truth_x, ground_truth_y),
                "error_px": error_px,
                "confidence": detector_result.confidence,
                "is_success": is_success,
            }],
            "success_rate": round(100 * int(is_success) / len(gt_hotspots), 1),
            "inference_time_ms": detector_result.inference_time_ms,
            "fps": round(1000 / max(detector_result.inference_time_ms, 0.001), 1),
        }

    def _run_pytorch_result(self, image_array: np.ndarray, gt_hotspots: list[list[float]]) -> dict:
        detector_result = self._get_pytorch_detector().detect(self._to_thermal_matrix(image_array))
        predicted_x = float(detector_result.center_x)
        predicted_y = float(detector_result.center_y)
        if not gt_hotspots:
            return {
                "hotspots": [],
                "success_rate": 0.0,
                "inference_time_ms": detector_result.inference_time_ms,
                "fps": round(1000 / max(detector_result.inference_time_ms, 0.001), 1),
            }

        nearest_index, (ground_truth_x, ground_truth_y) = min(
            enumerate(gt_hotspots),
            key=lambda item: float(np.hypot(predicted_x - item[1][0], predicted_y - item[1][1])),
        )
        error_px = round(float(np.hypot(predicted_x - ground_truth_x, predicted_y - ground_truth_y)))
        is_success = error_px <= HOTSPOT_SUCCESS_DISTANCE_PX
        return {
            "hotspots": [{
                "id": nearest_index + 1,
                "coordinate": (predicted_x, predicted_y),
                "temperature": detector_result.max_temperature,
                "ground_truth": (ground_truth_x, ground_truth_y),
                "error_px": error_px,
                "confidence": detector_result.confidence,
                "is_success": is_success,
            }],
            "success_rate": round(100 * int(is_success) / len(gt_hotspots), 1),
            "inference_time_ms": detector_result.inference_time_ms,
            "fps": round(1000 / max(detector_result.inference_time_ms, 0.001), 1),
        }

    def _compute_benchmark_results(self, image_file: str) -> None:
        """Run every model across the whole dataset once ('Run All Images'). Must not be called
        again just to browse images afterwards - use _load_cached_results_for_image() for that."""
        if self.current_cover_box is None:
            raise RuntimeError("Cannot compute benchmark results without a loaded YOLO C-Cover ground truth box")
        if self._image_array is None:
            raise RuntimeError("Cannot compute benchmark results without a loaded thermal image")
        self._compute_all_image_results()
        self._load_cached_results_for_image(image_file)
        self.dataset_summary = self._compute_dataset_summary()

    def _load_cached_results_for_image(self, image_file: str) -> None:
        """Look up already-computed per-image results (no inference) for Check Results browsing."""
        self.current_model_results = self._all_image_model_results.get(image_file, {})
        self.current_cover_result = self._all_image_cover_results.get(image_file, {})

    def _compute_all_image_results(self) -> None:
        """Run opencv (mock) / pytorch / openvino detection for every image in the dataset exactly
        once, caching results keyed by image file. This is the expensive step (real PyTorch and
        OpenVINO inference) and should only run from the 'Run All Images' action."""
        self._all_image_model_results = {}
        self._all_image_cover_results = {}
        for image_file in self.image_list:
            gt_hotspots = self.gt_hotspot_map.get(image_file, [])
            image_array = data_loader.load_thermal_image(image_file)
            if image_array is None:
                continue
            height, width = image_array.shape[:2]
            cover_box = data_loader.load_yolo_cover_box(image_file, width, height)
            seed_text = f"{image_file}#run{self.run_counter}"
            self._all_image_model_results[image_file] = {
                "opencv": self._run_opencv_result(image_array, gt_hotspots),
                "pytorch": self._run_pytorch_result(image_array, gt_hotspots),
                "openvino": self._run_openvino_result(image_array, gt_hotspots),
            }
            if cover_box is not None:
                self._all_image_cover_results[image_file] = generate_yolo_cover_result(cover_box, seed_text)

    def _compute_dataset_summary(self) -> dict[str, dict]:
        """Aggregate the cached per-image results (see _compute_all_image_results) across the
        whole dataset. Reads from cache only - runs no inference itself."""
        summary: dict[str, dict] = {}
        for model_key in MODEL_KEYS:
            all_errors: list[int] = []
            success_rates: list[float] = []
            inference_times: list[float] = []
            fps_values: list[float] = []
            cover_ious: list[float] = []
            map_predictions: list[tuple[float, str, float, float]] = []
            total_ground_truth = 0
            for image_file in self.image_list:
                gt_hotspots = self.gt_hotspot_map.get(image_file, [])
                if not gt_hotspots:
                    continue
                model_results = self._all_image_model_results.get(image_file)
                if model_results is None:
                    continue
                result = model_results[model_key]
                cover_result = self._all_image_cover_results.get(image_file)
                if cover_result is not None:
                    cover_ious.append(cover_result["iou"])
                all_errors.extend(h["error_px"] for h in result["hotspots"])
                success_rates.append(result["success_rate"])
                inference_times.append(result["inference_time_ms"])
                fps_values.append(result["fps"])
                total_ground_truth += len(gt_hotspots)
                map_predictions.extend(
                    (hotspot["confidence"], image_file, hotspot["coordinate"][0], hotspot["coordinate"][1])
                    for hotspot in result["hotspots"]
                )

            avg_error = round(sum(all_errors) / len(all_errors), 1) if all_errors else None
            avg_success_rate = round(sum(success_rates) / len(success_rates), 1) if success_rates else None
            avg_inference_time = round(sum(inference_times) / len(inference_times), 1) if inference_times else None
            avg_fps = round(sum(fps_values) / len(fps_values), 1) if fps_values else None
            avg_iou = round(sum(cover_ious) / len(cover_ious), 2) if cover_ious else None
            map_at_3px = self._calculate_map_at_distance(map_predictions, total_ground_truth, distance_threshold=3.0)

            overall_score = None
            if avg_error is not None and avg_success_rate is not None and avg_inference_time is not None:
                # Simple composite score: rewards high success rate, penalizes error and latency.
                overall_score = round(avg_success_rate - avg_error * 10 - avg_inference_time * 0.5, 1)

            summary[model_key] = {
                "avg_error": avg_error,
                "avg_success_rate": avg_success_rate,
                "avg_inference_time_ms": avg_inference_time,
                "avg_fps": avg_fps,
                "avg_iou": avg_iou,
                "map_at_3px": map_at_3px,
                "overall_score": overall_score,
            }
        return summary

    def _calculate_map_at_distance(
        self,
        predictions: list[tuple[float, str, float, float]],
        total_ground_truth: int,
        *,
        distance_threshold: float,
    ) -> Optional[float]:
        """Compute 101-point interpolated AP using confidence-ranked, one-to-one point matches."""
        if not predictions or total_ground_truth == 0:
            return None

        matched_ground_truth: dict[str, set[int]] = {}
        true_positives: list[int] = []
        false_positives: list[int] = []
        for _confidence, image_file, predicted_x, predicted_y in sorted(predictions, reverse=True):
            ground_truth_points = self.gt_hotspot_map.get(image_file, [])
            matched_indices = matched_ground_truth.setdefault(image_file, set())
            closest_index = None
            closest_distance = float("inf")
            for index, (ground_truth_x, ground_truth_y) in enumerate(ground_truth_points):
                if index in matched_indices:
                    continue
                distance = float(np.hypot(predicted_x - ground_truth_x, predicted_y - ground_truth_y))
                if distance < closest_distance:
                    closest_distance = distance
                    closest_index = index
            if closest_index is not None and closest_distance <= distance_threshold:
                matched_indices.add(closest_index)
                true_positives.append(1)
                false_positives.append(0)
            else:
                true_positives.append(0)
                false_positives.append(1)

        cumulative_true_positives = np.cumsum(true_positives)
        cumulative_false_positives = np.cumsum(false_positives)
        recalls = cumulative_true_positives / total_ground_truth
        precisions = cumulative_true_positives / np.maximum(cumulative_true_positives + cumulative_false_positives, 1)
        interpolated_precisions = [
            float(np.max(precisions[recalls >= recall_level])) if np.any(recalls >= recall_level) else 0.0
            for recall_level in np.linspace(0.0, 1.0, 101)
        ]
        return round(float(np.mean(interpolated_precisions) * 100), 1)

    def _on_prev_image(self) -> None:
        if self.run_counter == 0:
            return  # Nothing has been run yet - there is no result to browse to.
        if self.image_index > 0:
            self.image_index -= 1
            self._load_current_image()

    def _on_next_image(self) -> None:
        if self.run_counter == 0:
            return
        if self.image_index < len(self.image_list) - 1:
            self.image_index += 1
            self._load_current_image()

    def _on_run_all(self) -> None:
        image_file = self._current_image_file()
        if not image_file or self._benchmark_running:
            return
        self._benchmark_running = True
        self.run_counter += 1
        self._set_status("Processing", C["warning"])
        threading.Thread(target=self._run_current_image_worker, args=(image_file,), daemon=True).start()

    def _run_current_image_worker(self, image_file: str) -> None:
        try:
            image_array = data_loader.load_thermal_image(image_file)
            if image_array is None:
                raise RuntimeError(f"Could not load thermal image: {image_file}")
            height, width = image_array.shape[:2]
            cover_box = data_loader.load_yolo_cover_box(image_file, width, height)
            if cover_box is None:
                raise RuntimeError(f"Could not load YOLO C-Cover ground truth for: {image_file}")
            ground_truth_hotspots = self.gt_hotspot_map.get(image_file, [])
            self._image_array = image_array
            self.current_cover_box = cover_box
            self.current_gt_hotspots = ground_truth_hotspots
            self._compute_benchmark_results(image_file)
            self.root.after(0, lambda: self._complete_benchmark_run(image_array, cover_box, ground_truth_hotspots, image_file))
        except Exception as error:
            self.root.after(0, lambda: self._fail_benchmark_run(str(error)))

    def _complete_benchmark_run(
        self,
        image_array: np.ndarray,
        cover_box: dict,
        ground_truth_hotspots: list[list[float]],
        image_file: str,
    ) -> None:
        try:
            height, width = image_array.shape[:2]
            self._view_states = {
                model_key: {"center_x": width / 2, "center_y": height / 2, "scale": 1.0}
                for model_key in MODEL_KEYS
            }
            openvino_result = self.current_model_results.get("openvino", {})
            openvino_description = self.model_description_vars.get("openvino")
            if openvino_description is not None:
                runtime_precision = openvino_result.get("runtime_precision", "Unknown")
                if runtime_precision == "Unknown":
                    runtime_precision = f"{self.openvinoPrecisionVar.get()} (requested)"
                openvino_description.set(
                    f"Device: {openvino_result.get('runtime_device', self.openvinoDeviceVar.get())} | "
                    f"Precision: {runtime_precision} | Latency: {openvino_result.get('inference_time_ms', 0.0):.1f} ms"
                )
            pytorch_result = self.current_model_results.get("pytorch", {})
            pytorch_description = self.model_description_vars.get("pytorch")
            if pytorch_description is not None:
                pytorch_description.set(
                    f"Device: {self.pytorchDeviceVar.get()} | Precision: {self.pytorchPrecisionVar.get()} | "
                    f"Latency: {pytorch_result.get('inference_time_ms', 0.0):.1f} ms"
                )
            self._redraw_panels()
            self.image_index_var.set(f"{self.image_index + 1} / {len(self.image_list)}")
            self._set_check_results_active(True)
            self._set_status("Completed", C["success"])
        except Exception as error:
            self._fail_benchmark_run(str(error))
        finally:
            self._benchmark_running = False

    def _fail_benchmark_run(self, error_message: str) -> None:
        self._benchmark_running = False
        messagebox.showerror("Load Error", error_message)
        self._set_status("Error", C["error"])

    def _show_placeholder_state(self) -> None:
        """Initial state before any run: leave the C-Cover status and thermal panels blank
        (rather than pre-computing 20 images worth of results the user hasn't asked for yet)."""
        if self.detection_status_frame is not None:
            self._clear_frame(self.detection_status_frame)
            tk.Label(self.detection_status_frame, text="Click \u26a1 Run All Images to begin.",
                     bg=C["thermal_surface"], fg=C["muted"], font=(FF, 10)).pack(expand=True)
        if self.model_summary_frame is not None:
            self._clear_frame(self.model_summary_frame)
            tk.Label(self.model_summary_frame, text="No results yet.",
                     bg=C["thermal_surface"], fg=C["muted"], font=(FF, 10)).pack(expand=True)
        for model_key in MODEL_KEYS:
            axes = self.axes.get(model_key)
            canvas = self.canvases.get(model_key)
            if axes is not None and canvas is not None:
                axes.clear()
                axes.set_facecolor(C["thermal_surface"])
                axes.axis("off")
                axes.text(0.5, 0.5, "No image loaded", ha="center", va="center",
                          color=C["muted"], fontsize=10, transform=axes.transAxes)
                canvas.draw_idle()
            metrics_frame = self.metrics_frames.get(model_key)
            if metrics_frame is not None:
                self._clear_frame(metrics_frame)
                tk.Label(metrics_frame, text="No results yet.", bg=C["thermal_surface"], fg=C["muted"], font=(FF, 10)).pack(anchor=tk.W)

    # ── Rendering ─────────────────────────────────────────────────────────

    def _redraw_panels(self) -> None:
        if self._image_array is None:
            return
        self._update_performance_badges()
        for model_key in MODEL_KEYS:
            self._draw_model_panel(model_key)
            self._populate_metrics_card(model_key)
        self._populate_detection_status()
        self._populate_model_summary()

    def _draw_model_panel(self, model_key: str) -> None:
        axes = self.axes[model_key]
        axes.clear()
        axes.set_facecolor(C["thermal_surface"])
        axes.axis("off")
        axes.imshow(self._image_array)

        overlays = self.overlay_vars
        result = self.current_model_results.get(model_key)
        cover_result = self.current_cover_result

        if overlays["show_gt_cover"].get() and self.current_cover_box:
            self._draw_cover_polygon(axes, self.current_cover_box, color=C["cover_gt"], dashed=False)
        if overlays["show_cover"].get() and cover_result:
            self._draw_cover_polygon(axes, cover_result["detected_cover_box"], color=C["cover_predicted"], dashed=False)

        show_gt_hotspot = overlays["show_gt_hotspot"].get()
        if overlays["show_hotspot"].get() and result:
            for hotspot in result["hotspots"]:
                px, py = hotspot["coordinate"]
                gx, gy = hotspot["ground_truth"]
                error_px = hotspot["error_px"]

                if show_gt_hotspot and error_px > 0:
                    axes.plot([gx, px], [gy, py], color="#FFD54F", linewidth=1.2, linestyle="--", zorder=4)
                    # Place the error label below both points (not at the raw line midpoint) so it
                    # never overlaps the temperature badge, which sits above-right of the prediction.
                    mid_x = (gx + px) / 2
                    label_y = max(gy, py) + 16
                    axes.annotate(
                        f"Error: {error_px} px", xy=(mid_x, label_y), ha="center", va="center",
                        fontsize=7, fontweight="bold", color="#FFD54F", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor=C["bg"], edgecolor="#FFD54F", linewidth=0.8, alpha=0.9),
                    )
                if show_gt_hotspot:
                    axes.plot(gx, gy, marker="+", color=C["cover_gt"], markersize=12, markeredgewidth=1, zorder=6)
                    if error_px == 0:
                        # GT and prediction coincide exactly - call it out explicitly instead of
                        # leaving only the blue GT marker visible with no obvious match signal.
                        axes.annotate(
                            "✓ Match (0 px)", xy=(px, py), xytext=(px + 12, py + 14), textcoords="data",
                            ha="left", va="center", fontsize=7, fontweight="bold", color=C["bg"], zorder=8,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor=C["cover_predicted"], edgecolor="none", alpha=0.95),
                        )

                axes.plot(px, py, marker="+", color=C["cover_predicted"], markersize=12, markeredgewidth=1, zorder=7)
                axes.annotate(
                    f'{hotspot["temperature"]:.1f}°C', xy=(px, py), xytext=(px + 12, py - 12), textcoords="data",
                    ha="left", va="center", fontsize=8, fontweight="bold", color="white", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FF6B35", edgecolor="none", alpha=0.95),
                )
        elif show_gt_hotspot:
            for gt_x, gt_y in self.current_gt_hotspots:
                axes.plot(gt_x, gt_y, marker="+", color=C["cover_gt"], markersize=12, markeredgewidth=1, zorder=6)

        self._apply_view_to_axes(axes, model_key)
        self.canvases[model_key].draw_idle()

    @staticmethod
    def _draw_cover_polygon(axes, cover_box: dict, *, color: str, dashed: bool) -> None:
        points = [cover_box["top_left"], cover_box["top_right"], cover_box["bottom_right"], cover_box["bottom_left"]]
        polygon = Polygon(points, closed=True, fill=False, edgecolor=color, linewidth=2.4,
                           linestyle="--" if dashed else "-", zorder=3)
        axes.add_patch(polygon)

    # ── Independent per-panel zoom / pan ──────────────────────────────────

    def _apply_view_to_axes(self, axes, model_key: str) -> None:
        """Apply this panel's own zoom/pan view state to its axes instance."""
        if self._image_array is None:
            return
        height, width = self._image_array.shape[:2]
        state = self._view_states[model_key]
        half_w = (width / 2) / state["scale"]
        half_h = (height / 2) / state["scale"]
        axes.set_xlim(state["center_x"] - half_w, state["center_x"] + half_w)
        axes.set_ylim(state["center_y"] + half_h, state["center_y"] - half_h)

    def _apply_view_for_model(self, model_key: str) -> None:
        """Re-apply the given panel's view and redraw only that one canvas (no data changes)."""
        if self._image_array is None:
            return
        self._apply_view_to_axes(self.axes[model_key], model_key)
        self.canvases[model_key].draw_idle()

    def _clamp_view(self, model_key: str) -> None:
        if self._image_array is None:
            return
        height, width = self._image_array.shape[:2]
        state = self._view_states[model_key]
        state["scale"] = max(self._min_scale, min(state["scale"], self._max_scale))
        half_w = (width / 2) / state["scale"]
        half_h = (height / 2) / state["scale"]
        state["center_x"] = min(max(state["center_x"], half_w), width - half_w)
        state["center_y"] = min(max(state["center_y"], half_h), height - half_h)

    def _on_zoom_in(self, model_key: str) -> None:
        self._zoom_by_factor(model_key, 1.25)

    def _on_zoom_out(self, model_key: str) -> None:
        self._zoom_by_factor(model_key, 1 / 1.25)

    def _build_zoom_control(self, parent: tk.Frame, model_key: str) -> None:
        """A small floating Google-Maps-style +/-/reset control docked to the bottom-right
        corner of a thermal image panel; each panel gets its own, controlling only itself."""
        panel_bg = "#182033"
        control_area = tk.Frame(parent, bg=C["card"])
        control_area.pack(side=tk.RIGHT, anchor=tk.SE, padx=(8, 0), pady=4)

        def make_button(surface: tk.Canvas, text: str, command, tooltip: str) -> None:
            btn = tk.Label(surface, text=text, bg=panel_bg, fg=C["text"], font=(FF, 14, "bold"),
                           width=2, height=1, cursor="hand2")
            surface.create_window(17, 17, window=btn, width=30, height=30)
            btn.bind("<Button-1>", lambda _e: command())
            btn.bind("<Enter>", lambda _e: btn.config(bg="#283653"))
            btn.bind("<Leave>", lambda _e: btn.config(bg=panel_bg))
            Tooltip(btn, tooltip)

        reset_surface = tk.Canvas(control_area, width=34, height=34, bg=C["card"], highlightthickness=0, bd=0)
        reset_surface.pack(side=tk.TOP, pady=(0, 5))
        reset_surface.create_oval(2, 2, 32, 32, fill=panel_bg, outline=panel_bg)
        make_button(reset_surface, "\u21bb", lambda: self._on_reset_zoom(model_key), "Reset this panel's zoom")

        zoom_surface = tk.Canvas(control_area, width=34, height=66, bg=C["card"], highlightthickness=0, bd=0)
        zoom_surface.pack(side=tk.TOP)
        zoom_surface.create_arc(0, 0, 10, 10, start=90, extent=90, fill=panel_bg, outline=panel_bg)
        zoom_surface.create_arc(24, 0, 34, 10, start=0, extent=90, fill=panel_bg, outline=panel_bg)
        zoom_surface.create_arc(0, 56, 10, 66, start=180, extent=90, fill=panel_bg, outline=panel_bg)
        zoom_surface.create_arc(24, 56, 34, 66, start=270, extent=90, fill=panel_bg, outline=panel_bg)
        zoom_surface.create_rectangle(5, 0, 29, 66, fill=panel_bg, outline=panel_bg)
        zoom_surface.create_rectangle(0, 5, 34, 61, fill=panel_bg, outline=panel_bg)
        zoom_surface.create_line(2, 33, 32, 33, fill=C["border"])
        make_button(zoom_surface, "+", lambda: self._on_zoom_in(model_key), "Zoom in this panel")
        minus_btn = tk.Label(zoom_surface, text="\u2212", bg=panel_bg, fg=C["text"], font=(FF, 14, "bold"),
                             width=2, height=1, cursor="hand2")
        zoom_surface.create_window(17, 49, window=minus_btn, width=30, height=30)
        minus_btn.bind("<Button-1>", lambda _e: self._on_zoom_out(model_key))
        minus_btn.bind("<Enter>", lambda _e: minus_btn.config(bg="#283653"))
        minus_btn.bind("<Leave>", lambda _e: minus_btn.config(bg=panel_bg))
        Tooltip(minus_btn, "Zoom out this panel")

    def _on_reset_zoom(self, model_key: str) -> None:
        if self._image_array is None:
            return
        height, width = self._image_array.shape[:2]
        self._view_states[model_key] = {"center_x": width / 2, "center_y": height / 2, "scale": 1.0}
        self._apply_view_for_model(model_key)

    def _zoom_by_factor(self, model_key: str, factor: float) -> None:
        if self._image_array is None:
            return
        self._view_states[model_key]["scale"] *= factor
        self._clamp_view(model_key)
        self._apply_view_for_model(model_key)

    def _on_scroll(self, event, model_key: str) -> None:
        try:
            if self._image_array is None or event.xdata is None or event.ydata is None:
                return
            factor = 1.2 if event.button == "up" else (1 / 1.2 if event.button == "down" else 1.0)
            if factor == 1.0:
                return
            state = self._view_states[model_key]
            old_scale = state["scale"]
            new_scale = max(self._min_scale, min(old_scale * factor, self._max_scale))
            # Keep the point under the cursor fixed while zooming.
            state["center_x"] = event.xdata - (event.xdata - state["center_x"]) * (old_scale / new_scale)
            state["center_y"] = event.ydata - (event.ydata - state["center_y"]) * (old_scale / new_scale)
            state["scale"] = new_scale
            self._clamp_view(model_key)
            self._apply_view_for_model(model_key)
        except Exception as error:
            print(f"[ui] Scroll-to-zoom failed: {error}")

    def _on_pan_press(self, event, model_key: str) -> None:
        try:
            if self._image_array is None or event.button != 1 or event.x is None:
                return
            self._pan_active = True
            self._pan_model_key = model_key
            state = self._view_states[model_key]
            self._pan_start_px = (event.x, event.y)
            self._pan_start_center = (state["center_x"], state["center_y"])
        except Exception as error:
            print(f"[ui] Pan press failed: {error}")

    def _on_pan_motion(self, event, model_key: str) -> None:
        try:
            if not self._pan_active or self._pan_model_key != model_key or self._image_array is None:
                return
            if event.inaxes is None or event.x is None or event.y is None:
                return
            bbox = event.inaxes.get_window_extent()
            xlim = event.inaxes.get_xlim()
            ylim = event.inaxes.get_ylim()
            width_data = xlim[1] - xlim[0]
            height_data = ylim[0] - ylim[1]
            dx_px = event.x - self._pan_start_px[0]
            dy_px = event.y - self._pan_start_px[1]
            dx_data = -dx_px * (width_data / max(1.0, bbox.width))
            dy_data = dy_px * (height_data / max(1.0, bbox.height))
            state = self._view_states[model_key]
            state["center_x"] = self._pan_start_center[0] + dx_data
            state["center_y"] = self._pan_start_center[1] + dy_data
            self._clamp_view(model_key)
            self._apply_view_for_model(model_key)
        except Exception as error:
            print(f"[ui] Pan motion failed: {error}")

    def _on_pan_release(self, _event, model_key: str) -> None:
        if self._pan_model_key == model_key:
            self._pan_active = False
            self._pan_model_key = None

    # ── Model performance badges ─────────────────────────────────────────

    def _update_performance_badges(self) -> None:
        results = self.current_model_results
        if not all(results.get(k) for k in MODEL_KEYS):
            return
        ranked = sorted(MODEL_KEYS, key=lambda k: results[k]["inference_time_ms"])
        badge_by_rank = {
            ranked[0]: ("★ Fastest", C["success"]),
            ranked[-1]: ("○ Baseline", C["muted"]),
        }
        for model_key in MODEL_KEYS:
            text, color = badge_by_rank.get(model_key, ("◆ Balanced", C["warning"]))
            label = self.badge_labels.get(model_key)
            if label is not None:
                label.config(
                    text=text,
                    bg=C["thermal_surface"],
                    fg=color,
                    highlightbackground=color,
                )

    # ── Metrics / YOLO cover / summary cards ────────────────────────────────

    def _clear_frame(self, frame: tk.Frame) -> None:
        for child in frame.winfo_children():
            child.destroy()

    def _severity_color(self, error_px: int) -> str:
        if error_px <= 1:
            return C["success"]
        if error_px <= 3:
            return C["warning"]
        return C["error"]

    def _success_rate_color(self, success_rate: float) -> str:
        if success_rate >= 95:
            return C["success"]
        if success_rate >= 80:
            return C["warning"]
        return C["error"]

    def _populate_metrics_card(self, model_key: str) -> None:
        frame: tk.Frame = self.metrics_frames[model_key]
        self._clear_frame(frame)
        result = self.current_model_results.get(model_key)
        if not result:
            return

        tk.Label(frame, text="Detection Metrics", bg=C["thermal_surface"], fg=C["text"], font=(FF, 12, "bold")).pack(anchor=tk.W, pady=(0, 3))

        metrics_content = tk.Frame(frame, bg=C["thermal_surface"])
        metrics_content.pack(fill=tk.BOTH, expand=True)
        metrics_content.grid_columnconfigure(0, weight=1)
        metrics_content.grid_columnconfigure(1, weight=0, minsize=1)
        metrics_content.grid_columnconfigure(2, weight=1)

        hotspot_column = tk.Frame(metrics_content, bg=C["thermal_surface"])
        hotspot_column.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        tk.Frame(metrics_content, bg=C["border"], width=1).grid(row=0, column=1, sticky="ns")
        stats_column = tk.Frame(metrics_content, bg=C["thermal_surface"])
        stats_column.grid(row=0, column=2, sticky="nsew", padx=(10, 0))

        for hotspot in result["hotspots"]:
            row = tk.Frame(hotspot_column, bg=C["thermal_surface"])
            row.pack(fill=tk.X, pady=(0, 3))
            tk.Label(row, text=f'Hotspot #{hotspot["id"]}', bg=C["thermal_surface"], fg=C["text"], font=(FF, 11, "bold")).pack(anchor=tk.W)
            coord_x, coord_y = hotspot["coordinate"]
            gt_x, gt_y = hotspot["ground_truth"]
            err_row = tk.Frame(row, bg=C["thermal_surface"])
            err_row.pack(anchor=tk.W)
            tk.Label(err_row, text="+", bg=C["thermal_surface"], fg=C["cover_gt"], font=(FF, 11, "bold")).pack(side=tk.LEFT, padx=(0, 4))
            tk.Label(err_row, text=f'Ground Truth: ({gt_x:.0f}, {gt_y:.0f})', bg=C["thermal_surface"], fg=C["muted"], font=(FF, 10)).pack(side=tk.LEFT)
            tk.Label(err_row, text=f'  Error: {hotspot["error_px"]} px', bg=C["thermal_surface"],
                     fg=self._severity_color(hotspot["error_px"]), font=(FF, 10, "bold")).pack(side=tk.LEFT)
            detected_row = tk.Frame(row, bg=C["thermal_surface"])
            detected_row.pack(anchor=tk.W)
            tk.Label(detected_row, text="+", bg=C["thermal_surface"], fg=C["cover_predicted"], font=(FF, 11, "bold")).pack(side=tk.LEFT, padx=(0, 4))
            tk.Label(detected_row, text=f'Predicted: ({coord_x:.0f}, {coord_y:.0f})  Temp: {hotspot["temperature"]:.1f}°C',
                     bg=C["thermal_surface"], fg=C["muted"], font=(FF, 10)).pack(side=tk.LEFT)

        stat_row = tk.Frame(stats_column, bg=C["thermal_surface"])
        stat_row.pack(fill=tk.X)
        tk.Label(stat_row, text="•", bg=C["thermal_surface"], fg=self._success_rate_color(result["success_rate"]), font=(FF, 10, "bold")).pack(side=tk.LEFT, padx=(0, 3))
        tk.Label(stat_row, text="Success Rate", bg=C["thermal_surface"], fg=C["muted"], font=(FF, 10)).pack(side=tk.LEFT)
        tk.Label(stat_row, text=f'{result["success_rate"]}%', bg=C["thermal_surface"],
                 fg=self._success_rate_color(result["success_rate"]), font=(FF, 11, "bold")).pack(side=tk.RIGHT)

        time_row = tk.Frame(stats_column, bg=C["thermal_surface"])
        time_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(time_row, text="•", bg=C["thermal_surface"], fg=C["text"], font=(FF, 10, "bold")).pack(side=tk.LEFT, padx=(0, 3))
        tk.Label(time_row, text="Inference Time", bg=C["thermal_surface"], fg=C["muted"], font=(FF, 10)).pack(side=tk.LEFT)
        tk.Label(time_row, text=f'{result["inference_time_ms"]:.1f} ms', bg=C["thermal_surface"], fg=C["text"], font=(FF, 11)).pack(side=tk.RIGHT)

        fps_row = tk.Frame(stats_column, bg=C["thermal_surface"])
        fps_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(fps_row, text="•", bg=C["thermal_surface"], fg=C["warning"], font=(FF, 10, "bold")).pack(side=tk.LEFT, padx=(0, 3))
        tk.Label(fps_row, text="FPS", bg=C["thermal_surface"], fg=C["muted"], font=(FF, 10)).pack(side=tk.LEFT)
        tk.Label(fps_row, text=f'{result["fps"]}', bg=C["thermal_surface"], fg=C["text"], font=(FF, 11)).pack(side=tk.RIGHT)

    def _populate_detection_status(self) -> None:
        """Left column (40% width): draws a virtual predicted C-Cover rectangle as the data
        container - dashed emerald box with corner coordinate tags, and 3 circles
        (Status / Confidence / IoU), each with its label above, value inside, and - for
        Confidence/IoU - the passing threshold annotated below."""
        if self.detection_status_frame is None:
            return
        frame: tk.Frame = self.detection_status_frame
        self._clear_frame(frame)

        # Tkinter's Canvas defaults to a 200x150 requested size when no width/height is given,
        # which was silently forcing this row taller than intended - set an explicit small size.
        canvas = tk.Canvas(frame, bg=C["thermal_surface"], width=220, height=129, highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        self.detection_canvas = canvas
        canvas.bind("<Configure>", lambda _e: self._draw_detection_status_canvas())
        self._draw_detection_status_canvas()

    def _draw_detection_status_canvas(self) -> None:
        canvas = self.detection_canvas
        if canvas is None:
            return
        canvas.delete("all")
        # Keep references to the generated circle images - Canvas won't hold them alive otherwise.
        self._detection_circle_images = []
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())

        cover_result = self.current_cover_result
        if not cover_result:
            canvas.create_text(width / 2, height / 2, text="Unavailable.", fill=C["warning"], font=(FF, 9))
            return

        padding = 1
        x0, y0, x1, y1 = padding, padding, width - padding, height - 6

        box = cover_result["detected_cover_box"]
        gt_box = self.current_cover_box
        all_box_points = []
        for corner in (box["top_left"], box["top_right"], box["bottom_right"], box["bottom_left"]):
            all_box_points.append(corner)
        if gt_box:
            for corner in (gt_box["top_left"], gt_box["top_right"], gt_box["bottom_right"], gt_box["bottom_left"]):
                all_box_points.append(corner)

        min_x = min(point[0] for point in all_box_points)
        max_x = max(point[0] for point in all_box_points)
        min_y = min(point[1] for point in all_box_points)
        max_y = max(point[1] for point in all_box_points)
        span_x = max(1.0, max_x - min_x)
        span_y = max(1.0, max_y - min_y)

        def map_point(point: tuple[float, float]) -> tuple[float, float]:
            px, py = point
            mapped_x = x0 + ((px - min_x) / span_x) * (x1 - x0)
            mapped_y = y0 + ((py - min_y) / span_y) * (y1 - y0)
            return mapped_x, mapped_y

        # Draw GT first (white solid), then detected (green solid) using Matplotlib rendering,
        # so edge style matches the thermal panel renderer.
        pred_points = [map_point(point) for point in (box["top_left"], box["top_right"], box["bottom_right"], box["bottom_left"])]
        gt_points = None
        if gt_box:
            gt_points = [map_point(point) for point in (gt_box["top_left"], gt_box["top_right"], gt_box["bottom_right"], gt_box["bottom_left"])]

        overlay_photo = self._render_detection_overlay_with_matplotlib(
            width=width,
            height=height,
            gt_points=gt_points,
            pred_points=pred_points,
            show_detected=self._detectedLineVisible,
        )
        self._detection_circle_images.append(overlay_photo)
        canvas.create_image(0, 0, image=overlay_photo, anchor="nw")

        corner_map = {
            "TL": "top_left",
            "TR": "top_right",
            "BL": "bottom_left",
            "BR": "bottom_right",
        }
        detected_corner_canvas = {
            "TL": map_point(box["top_left"]),
            "TR": map_point(box["top_right"]),
            "BL": map_point(box["bottom_left"]),
            "BR": map_point(box["bottom_right"]),
        }
        center_x = sum(point[0] for point in pred_points) / 4.0
        center_y = sum(point[1] for point in pred_points) / 4.0
        label_font = tkfont.Font(family=FF, size=8)
        line_height = max(12, label_font.metrics("linespace"))

        def draw_compact_corner_label(corner_name: str) -> None:
            detected_x, detected_y = box[corner_map[corner_name]]
            gt_x, gt_y = (detected_x, detected_y)
            if gt_box:
                gt_x, gt_y = gt_box[corner_map[corner_name]]

            corner_canvas_x, corner_canvas_y = detected_corner_canvas[corner_name]

            segments = [
                (f"{corner_name}: ", C["muted"]),
                (f"({gt_x:.0f}, {gt_y:.0f})", C["cover_gt"]),
                (" / ", C["muted"]),
                (f"({detected_x:.0f}, {detected_y:.0f})", C["cover_predicted"]),
            ]

            total_width = sum(label_font.measure(text) for text, _color in segments)
            inward_ratio = 0.015
            inside_x = corner_canvas_x + (center_x - corner_canvas_x) * inward_ratio
            inside_y = corner_canvas_y + (center_y - corner_canvas_y) * inward_ratio

            if corner_name == "TL":
                base_x = inside_x
                base_y = inside_y
            elif corner_name == "TR":
                base_x = inside_x - total_width
                base_y = inside_y
            elif corner_name == "BL":
                base_x = inside_x
                base_y = inside_y - line_height
            else:  # BR
                base_x = inside_x - total_width
                base_y = inside_y - line_height

            if corner_name in ("TL", "BL"):
                base_x -= 2
            else:
                base_x += 2

            base_x = min(max(base_x, 1), max(1, width - total_width - 1))
            base_y = min(max(base_y, 1), max(1, height - line_height - 1))

            cursor_x = base_x
            for text, color in segments:
                canvas.create_text(cursor_x, base_y, text=text, fill=C["bg"], font=label_font,
                                   anchor="nw", tags=("corner_label",))
                canvas.create_text(cursor_x + 1, base_y + 1, text=text, fill=color, font=label_font,
                                   anchor="nw", tags=("corner_label",))
                cursor_x += label_font.measure(text)

        for name in ("TL", "TR", "BL", "BR"):
            draw_compact_corner_label(name)

        is_pass = cover_result["status"] == "PASS"
        status_color = C["success"] if is_pass else C["error"]
        confidence_ratio = max(0.0, min(cover_result["confidence"] / 100, 1.0))
        iou_ratio = max(0.0, min(cover_result["iou"], 1.0))

        radius = max(40, min(46, (height - 74) / 2))
        center_y = height / 2 + 5
        slot_centers = (width * 0.22, width * 0.5, width * 0.78)

        self._draw_metric_circle(canvas, slot_centers[0], center_y, radius, None,
                                  cover_result["status"], "Status", status_color)
        self._draw_metric_circle(canvas, slot_centers[1], center_y, radius, confidence_ratio,
                      f'{cover_result["confidence"]}%', "Confidence", C["accent"],
                      condition_text=f"(Pass \u2265 {CONFIDENCE_PASS_THRESHOLD}%)")
        self._draw_metric_circle(canvas, slot_centers[2], center_y, radius, iou_ratio,
                      f'{cover_result["iou"]}', "IoU", C["success"],
                                  condition_text=f"(Pass \u2265 {IOU_PASS_THRESHOLD})", label_offset_y=3)

        # Keep corner labels visible above circles and outlines.
        canvas.tag_raise("corner_label")

    def _render_detection_overlay_with_matplotlib(
        self,
        *,
        width: int,
        height: int,
        gt_points: Optional[list[tuple[float, float]]],
        pred_points: list[tuple[float, float]],
        show_detected: bool,
    ) -> ImageTk.PhotoImage:
        """Render status overlay lines with Matplotlib for smoother, panel-consistent edges."""
        try:
            dpi = 100
            fig = plt.Figure(figsize=(max(1, width) / dpi, max(1, height) / dpi), dpi=dpi)
            fig.patch.set_alpha(0.0)
            axes = fig.add_axes([0, 0, 1, 1])
            axes.set_xlim(0, width)
            axes.set_ylim(height, 0)
            axes.margins(0)
            axes.axis("off")
            axes.set_facecolor("none")

            if gt_points:
                gt_x = [point[0] for point in gt_points] + [gt_points[0][0]]
                gt_y = [point[1] for point in gt_points] + [gt_points[0][1]]
                axes.plot(gt_x, gt_y, color=C["cover_gt"], linewidth=2, solid_capstyle="round", zorder=2)
                axes.scatter(gt_x[:-1], gt_y[:-1], color=C["cover_gt"], s=20, zorder=3)

            if show_detected:
                pred_x = [point[0] for point in pred_points] + [pred_points[0][0]]
                pred_y = [point[1] for point in pred_points] + [pred_points[0][1]]
                axes.plot(pred_x, pred_y, color=C["cover_predicted"], linewidth=2.4, solid_capstyle="round", zorder=4)
                axes.scatter(pred_x[:-1], pred_y[:-1], color=C["cover_predicted"], s=20, zorder=5)

            canvas_agg = FigureCanvasAgg(fig)
            canvas_agg.draw()
            rgba = np.asarray(canvas_agg.buffer_rgba())
            image = Image.fromarray(rgba)
            plt.close(fig)
            return ImageTk.PhotoImage(image)
        except Exception as error:
            print(f"[ui] Failed to render matplotlib detection overlay: {error}")
            fallback = Image.new("RGBA", (max(1, width), max(1, height)), (0, 0, 0, 0))
            return ImageTk.PhotoImage(fallback)

    def _draw_metric_circle(self, canvas: tk.Canvas, cx: float, cy: float, radius: float,
                             ratio: Optional[float], value_text: str, label_text: str, color: str,
                             condition_text: Optional[str] = None, label_offset_y: int = 0) -> None:
        """Draw one metric as a smooth, anti-aliased ring (or solid dot) with its label above the
        circle, the value inside, and (for Confidence/IoU) the passing threshold below.
        Rendered via PIL at high resolution and downsampled, since Tkinter's native Canvas
        ovals/arcs render jagged at small sizes."""
        label_font = tkfont.Font(family=FF, size=9, weight="bold")
        condition_font = tkfont.Font(family=FF, size=9)
        label_y = cy - radius - 10 + label_offset_y
        if condition_text:
            total_width = label_font.measure(label_text) + condition_font.measure(f" {condition_text}")
            label_x = cx - total_width / 2
            canvas.create_text(label_x, label_y, text=label_text, fill=C["muted"], font=label_font, anchor="w")
            canvas.create_text(
                label_x + label_font.measure(label_text),
                label_y,
                text=f" {condition_text}",
                fill=C["muted"],
                font=condition_font,
                anchor="w",
            )
        else:
            canvas.create_text(cx, label_y, text=label_text, fill=C["muted"], font=label_font)

        photo = self._make_circle_image(radius, ratio, color)
        self._detection_circle_images.append(photo)
        canvas.create_image(cx, cy, image=photo, anchor="center")
        text_color = C["bg"] if ratio is None else C["text"]
        canvas.create_text(cx, cy, text=value_text, fill=text_color, font=(FF, 14, "bold"))

    @staticmethod
    def _make_circle_image(radius: float, ratio: Optional[float], color: str) -> ImageTk.PhotoImage:
        """Supersample a circle/ring at 8x then downscale with LANCZOS for a crisp, smooth edge
        (same technique used for the window icon), instead of a jagged native Canvas oval."""
        supersample = 8
        diameter = max(2, int(round(radius * 2)))
        size = diameter * supersample
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        if ratio is None:
            draw.ellipse([0, 0, size - 1, size - 1], fill=color)
        else:
            stroke = max(2, int(round(4 * supersample)))
            inset = stroke / 2
            draw.ellipse([inset, inset, size - 1 - inset, size - 1 - inset], outline=C["border"], width=stroke)
            draw.arc([inset, inset, size - 1 - inset, size - 1 - inset],
                      start=-90, end=-90 + 360 * max(0.0, min(ratio, 1.0)), fill=color, width=stroke)
        image = image.resize((diameter, diameter), Image.LANCZOS)
        return ImageTk.PhotoImage(image)

    def _populate_model_summary(self) -> None:
        """Right column (60% width): 3 side-by-side per-model cards showing compact metrics
        with the winner badges integrated at the bottom (no separate Winners section)."""
        if self.model_summary_frame is None:
            return
        frame: tk.Frame = self.model_summary_frame
        self._clear_frame(frame)
        self.model_cards = {}

        grid = tk.Frame(frame, bg=C["thermal_surface"])
        grid.pack(fill=tk.X, anchor=tk.N)
        for col_index in range(len(MODEL_KEYS)):
            grid.columnconfigure(col_index, weight=1, uniform="model_card")

        summary = self.dataset_summary
        has_summary = bool(summary) and all(summary.get(k, {}).get("avg_error") is not None for k in MODEL_KEYS)
        badge_keys: dict[str, set[str]] = {}
        if has_summary:
            # Use sets (not a single "best" key) so ties are all awarded the badge instead of
            # picking an arbitrary winner when two models score identically.
            best_error = min(summary[k]["avg_error"] for k in MODEL_KEYS)
            best_inference_time = min(summary[k]["avg_inference_time_ms"] for k in MODEL_KEYS)
            best_score = max(summary[k]["overall_score"] for k in MODEL_KEYS)
            badge_keys = {
                "accuracy": {k for k in MODEL_KEYS if summary[k]["avg_error"] == best_error},
                "speed": {k for k in MODEL_KEYS if summary[k]["avg_inference_time_ms"] == best_inference_time},
                "overall": {k for k in MODEL_KEYS if summary[k]["overall_score"] == best_score},
            }

        for col_index, model_key in enumerate(MODEL_KEYS):
            self._build_model_summary_card(grid, col_index, model_key, summary.get(model_key), badge_keys)

    def _build_model_summary_card(self, parent: tk.Frame, column: int, model_key: str,
                                   stats: Optional[dict], badge_keys: dict[str, set[str]]) -> None:
        card = tk.Frame(parent, bg=C["thermal_surface"], highlightbackground=C["border"], highlightthickness=1)
        card.grid(row=0, column=column, sticky="new", padx=(0 if column == 0 else 8, 0))
        self.model_cards[model_key] = card

        inner = tk.Frame(card, bg=C["thermal_surface"])
        inner.pack(fill=tk.X, padx=5, pady=3)

        title_row = tk.Frame(inner, bg=C["thermal_surface"])
        title_row.pack(fill=tk.X)
        tk.Label(title_row, text=SHORT_MODEL_NAMES[model_key], bg=C["thermal_surface"], fg=C["text"], font=(FF, 11, "bold")).pack(side=tk.LEFT)
        badge_holder = tk.Frame(title_row, bg=C["thermal_surface"])
        badge_holder.pack(side=tk.RIGHT)

        metrics_frame = tk.Frame(inner, bg=C["thermal_surface"])
        metrics_frame.pack(fill=tk.X, pady=(4, 4))
        metrics_frame.columnconfigure(0, weight=1, uniform="summary_metric")
        metrics_frame.columnconfigure(1, weight=0)
        metrics_frame.columnconfigure(2, weight=1, uniform="summary_metric")
        if stats is None:
            tk.Label(metrics_frame, text="Run all models to view metrics.", bg=C["thermal_surface"], fg=C["muted"], font=(FF, 10)).pack(anchor=tk.W)
        else:
            metric_cells = (
                ("Overall Score", f'{stats["overall_score"]}', C["accent"] if model_key in badge_keys.get("overall", set()) else C["muted"], 0, 0),
                ("Success Rate", f'{stats["avg_success_rate"]}%', C["muted"], 0, 2),
                ("Avg Error", f'{stats["avg_error"]} px', C["cover_gt"] if model_key in badge_keys.get("accuracy", set()) else C["muted"], 1, 0),
                ("Avg IoU", f'{stats["avg_iou"]}', C["muted"], 1, 2),
                ("Avg FPS", f'{stats["avg_fps"]}', C["success"] if model_key in badge_keys.get("speed", set()) else C["muted"], 2, 0),
                ("mAP@3px", f'{stats["map_at_3px"]}%', C["muted"], 2, 2),
            )
            tk.Frame(metrics_frame, bg=C["border"], width=1).grid(row=0, column=1, rowspan=3, sticky="ns", padx=6)
            for label, value, color, row, column in metric_cells:
                cell = tk.Frame(metrics_frame, bg=C["thermal_surface"])
                cell.grid(row=row, column=column, sticky="w", pady=3)
                tk.Label(cell, text="•", bg=C["thermal_surface"], fg=color, font=(FF, 10, "bold")).pack(side=tk.LEFT, padx=(0, 3))
                tk.Label(cell, text=f"{label}: ", bg=C["thermal_surface"], fg=C["muted"], font=(FF, 10)).pack(side=tk.LEFT)
                tk.Label(cell, text=value, bg=C["thermal_surface"], fg=color, font=(FF, 10, "bold")).pack(side=tk.LEFT)

        if model_key in badge_keys.get("overall", set()):
            self._make_summary_badge(badge_holder, "👑 Best Overall", C["accent"])
        if model_key in badge_keys.get("accuracy", set()):
            # Use a distinct color from "Fastest Model" so the two badges stay visually separable
            self._make_summary_badge(badge_holder, "🏆 Best Accuracy", C["cover_gt"])
        if model_key in badge_keys.get("speed", set()):
            # Keep the same color as the per-image "Fastest" badge for visual consistency
            self._make_summary_badge(badge_holder, "⚡ Fastest Model", C["success"])

        self._bind_card_hover(card)

    def _make_summary_badge(self, parent: tk.Frame, text: str, accent_color: str) -> None:
        icon_text, label_text = text.split(" ", 1)
        badge = tk.Frame(parent, bg=C["thermal_surface"], highlightbackground=accent_color, highlightthickness=1)
        badge.pack(side=tk.LEFT, padx=(0, 3))
        tk.Label(badge, text=icon_text, bg=C["thermal_surface"], fg=accent_color, font=(FF, 8, "bold"), width=2).pack(
            side=tk.LEFT, padx=(4, 1), pady=1
        )
        tk.Label(badge, text=label_text, bg=C["thermal_surface"], fg=accent_color, font=(FF, 8, "bold")).pack(
            side=tk.LEFT, padx=(0, 4), pady=1
        )

    def _bind_card_hover(self, card: tk.Frame) -> None:
        card.bind("<Enter>", lambda _e: card.config(highlightbackground=C["dim"]))
        card.bind("<Leave>", lambda _e: card.config(highlightbackground=C["border"]))






def main() -> None:
    root = tk.Tk()
    HotspotBenchmarkApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
