"""Desktop benchmark console for comparing OpenCV / PyTorch / OpenVINO hotspot
detection results on the same YOLO C-Cover detection, built with Tkinter + Matplotlib
to match the visual language used in thermal/ui.py (dark engineering theme).
"""

from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox
from pathlib import Path
import random
import threading
from typing import Callable, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle, Polygon
from PIL import Image, ImageDraw, ImageFont, ImageTk

from hotspot_detector import data_loader
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
            "show_gt_cover": tk.BooleanVar(value=True),
            "show_corner_points": tk.BooleanVar(value=False),
        }

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
        self._scroll_canvas: Optional[tk.Canvas] = None

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
        self.current_hotspot_result: Optional[dict] = None
        self._hotspot_zoom_scale = 1.0
        self._hotspot_pan_offset: tuple[float, float] = (0.0, 0.0)
        self._hotspot_drag_start: Optional[tuple[float, float]] = None
        self.current_cover_result: Optional[dict] = None
        self.current_model_results: dict[str, dict] = {}
        self._image_array: Optional[np.ndarray] = None
        self._temperature_array: Optional[np.ndarray] = None
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
        run_one_segment = tk.Frame(control_strip, bg="#111B2A", width=run_segment_width, height=32)
        run_one_segment.pack(side=tk.LEFT, padx=(4, 0))
        run_one_segment.pack_propagate(False)
        run_one_button = self._make_rounded_button(run_one_segment, "⚡ Run 1 Image", self._on_run_one_image)
        run_one_button.place(relx=0.5, rely=0.5, anchor="center")

        tk.Frame(control_strip, bg="#3A405A", width=1, height=22).pack(side=tk.LEFT, padx=4, pady=6)
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
        """Refresh C-Cover and hotspot overlays after a display toggle."""
        if self._image_array is None:
            return
        for model_key in MODEL_KEYS:
            self._draw_model_panel(model_key)
        self._populate_detection_status()

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
            title="Hotspot Detection (OpenCV)",
            titleRightBuilder=self._build_corner_points_control,
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

    def _build_corner_points_control(self, parent: tk.Frame) -> None:
        """Build the visible toggle for C-Cover corner coordinate labels."""
        self._make_checkbox(
            parent,
            "Show Corner Points",
            self.overlay_vars["show_corner_points"],
            bg=C["thermal_surface"],
            fg=C["muted"],
            command=self._on_overlay_toggle,
        ).pack(side=tk.RIGHT, padx=(8, 4), pady=2)

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
        description_var = tk.StringVar(value="Paused")
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
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        # Bind with model_key captured per-canvas so each panel zooms/pans independently.
        canvas_widget.bind("<MouseWheel>", lambda event, mk=model_key: self._on_model_mousewheel(event, mk))
        # Matplotlib registers its Windows wheel adapter directly on the root widget.
        # Remove that adapter so Python 3.14 cannot pass a widget-path string to it.
        self.root.unbind("<MouseWheel>")
        # Matplotlib adds a Windows wheel handler to the toplevel. Exclude that bindtag
        # for this widget so its callback cannot receive a Tk widget-path string.
        bindtags = list(canvas_widget.bindtags())
        toplevel_tag = str(canvas_widget.winfo_toplevel())
        if toplevel_tag in bindtags:
            bindtags.remove(toplevel_tag)
            canvas_widget.bindtags(tuple(bindtags))
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
            dot_count = self._status_blink_step % 5 + 1
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
            temperature_array = data_loader.load_temperature_matrix(image_file, image_array)
            if temperature_array is None:
                raise RuntimeError(f"Raw Celsius temperature data not found for: {image_file}")
            if temperature_array.shape != image_array.shape[:2]:
                raise RuntimeError(
                    f"Temperature shape {temperature_array.shape} does not match image shape {image_array.shape[:2]}"
                )
            height, width = image_array.shape[:2]

            cover_box = data_loader.load_yolo_cover_box(image_file, width, height)
            if cover_box is None:
                raise RuntimeError(f"Could not load YOLO C-Cover ground truth for: {image_file}")

            gt_hotspots = self.gt_hotspot_map.get(image_file, [])

            self._image_array = image_array
            self._temperature_array = temperature_array
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

    @staticmethod
    def _temperature_at_coordinate(temperature_array: np.ndarray, pixel_x: float, pixel_y: float) -> float:
        """Read the calibrated Celsius value at a predicted pixel coordinate."""
        if temperature_array.ndim != 2 or temperature_array.size == 0:
            raise ValueError("temperature_array must be a non-empty 2D matrix")
        x = int(np.clip(round(pixel_x), 0, temperature_array.shape[1] - 1))
        y = int(np.clip(round(pixel_y), 0, temperature_array.shape[0] - 1))
        temperature = float(temperature_array[y, x])
        if not np.isfinite(temperature):
            raise ValueError(f"temperature at ({x}, {y}) is not finite")
        return temperature

    def _prepare_predictions(self, detector_result, temperature_array: np.ndarray) -> list[dict]:
        """Normalize detector candidates and attach the Celsius value at each center."""
        candidates = list(getattr(detector_result, "detections", []) or [])
        if not candidates:
            candidates = [{
                "center_x": detector_result.center_x,
                "center_y": detector_result.center_y,
                "confidence": detector_result.confidence,
            }]
        predictions = []
        for candidate in candidates[:2]:
            try:
                center_x = float(candidate["center_x"])
                center_y = float(candidate["center_y"])
                predictions.append({
                    "center_x": center_x,
                    "center_y": center_y,
                    "temperature": float(candidate.get(
                        "max_temperature",
                        self._temperature_at_coordinate(temperature_array, center_x, center_y),
                    )),
                    "confidence": float(candidate.get("confidence", 0.0)),
                })
            except (KeyError, TypeError, ValueError) as error:
                print(f"[ui] Skipping invalid detector candidate: {error}")
        return predictions

    @staticmethod
    def _make_model_cover_result(detector_result, image_shape: tuple[int, ...], gt_cover_box: dict) -> dict:
        """Convert a model bbox into a four-corner C-Cover result and calculate IoU."""
        image_height, image_width = image_shape[:2]
        x, y, box_width, box_height = detector_result.bbox
        x = int(np.clip(x, 0, max(0, image_width - 1)))
        y = int(np.clip(y, 0, max(0, image_height - 1)))
        box_width = max(1, min(int(box_width), image_width - x))
        box_height = max(1, min(int(box_height), image_height - y))
        predicted_box = {
            "top_left": (float(x), float(y)),
            "top_right": (float(x + box_width), float(y)),
            "bottom_right": (float(x + box_width), float(y + box_height)),
            "bottom_left": (float(x), float(y + box_height)),
        }
        predicted_polygon = np.array([
            predicted_box["top_left"], predicted_box["top_right"],
            predicted_box["bottom_right"], predicted_box["bottom_left"],
        ], dtype=np.float32)
        gt_polygon = np.array([
            gt_cover_box["top_left"], gt_cover_box["top_right"],
            gt_cover_box["bottom_right"], gt_cover_box["bottom_left"],
        ], dtype=np.float32)
        intersection_area, _ = cv2.intersectConvexConvex(predicted_polygon, gt_polygon)
        predicted_area = abs(float(cv2.contourArea(predicted_polygon)))
        gt_area = abs(float(cv2.contourArea(gt_polygon)))
        union_area = predicted_area + gt_area - float(intersection_area)
        iou = float(intersection_area / union_area) if union_area > 0 else 0.0
        confidence = float(np.clip(detector_result.confidence * 100.0, 0.0, 100.0))
        return {
            "detected_cover_box": predicted_box,
            "confidence": round(confidence, 1),
            "iou": round(iou, 2),
            "status": "PASS" if iou >= IOU_PASS_THRESHOLD else "FAIL",
        }

    @staticmethod
    def _limit_predictions_to_gt_count(predictions: list[dict], gt_hotspots: list[list[float]]) -> list[dict]:
        """Keep at most one prediction per available ground-truth hotspot."""
        return predictions[:min(len(predictions), len(gt_hotspots), 2)]

    @staticmethod
    def _make_comparison_slots(
        predictions: list[dict],
        gt_hotspots: list[list[float]],
    ) -> list[dict]:
        """Compare each prediction with the closest ground-truth point."""
        if not gt_hotspots:
            return []
        predictions = predictions[:2]
        slots = []
        for prediction_index, prediction in enumerate(predictions):
            predicted_x = float(prediction["center_x"])
            predicted_y = float(prediction["center_y"])
            nearest_index = min(
                range(len(gt_hotspots)),
                key=lambda index: float(np.hypot(
                    predicted_x - gt_hotspots[index][0],
                    predicted_y - gt_hotspots[index][1],
                )),
            )
            ground_truth_x, ground_truth_y = gt_hotspots[nearest_index]
            error_px = round(float(np.hypot(predicted_x - ground_truth_x, predicted_y - ground_truth_y)))
            slots.append({
                "id": prediction_index + 1,
                "coordinate": (predicted_x, predicted_y),
                "temperature": float(prediction["temperature"]),
                "ground_truth": (ground_truth_x, ground_truth_y),
                "ground_truth_id": nearest_index + 1,
                "error_px": error_px,
                "confidence": float(prediction["confidence"]),
                "is_success": error_px <= HOTSPOT_SUCCESS_DISTANCE_PX,
                "has_prediction": True,
            })
        return slots

    def _run_openvino_result(self, image_array: np.ndarray, temperature_array: np.ndarray, gt_hotspots: list[list[float]], gt_cover_box: dict) -> dict:
        detector_result = self._get_openvino_detector().detect(self._to_thermal_matrix(image_array), temperature_array)
        predictions = self._prepare_predictions(detector_result, temperature_array)
        predictions = self._limit_predictions_to_gt_count(predictions, gt_hotspots)
        if not gt_hotspots:
            return {
                "hotspots": [],
                "success_rate": 0.0,
                "inference_time_ms": detector_result.inference_time_ms,
                "fps": round(1000 / max(detector_result.inference_time_ms, 0.001), 1),
                "runtime_device": self._get_openvino_detector().get_execution_device_text(),
                "runtime_precision": self._get_openvino_detector().applied_precision_hint,
            }

        comparison_slots = self._make_comparison_slots(
            predictions, gt_hotspots
        )
        return {
            "hotspots": comparison_slots,
            "cover_result": self._make_model_cover_result(detector_result, image_array.shape, gt_cover_box),
            "success_rate": round(100 * sum(slot["is_success"] for slot in comparison_slots) / len(gt_hotspots), 1),
            "inference_time_ms": detector_result.inference_time_ms,
            "fps": round(1000 / max(detector_result.inference_time_ms, 0.001), 1),
            "runtime_device": self._get_openvino_detector().get_execution_device_text(),
            "runtime_precision": self._get_openvino_detector().applied_precision_hint,
        }

    def _run_opencv_result(self, image_array: np.ndarray, temperature_array: np.ndarray, gt_hotspots: list[list[float]], gt_cover_box: dict) -> dict:
        ordered_cover_points = self._order_cover_corners(gt_cover_box)
        cover_polygon = [ordered_cover_points[corner] for corner in ("TL", "TR", "BR", "BL")]
        detector_result = self._get_opencv_detector().detect(
            self._to_thermal_matrix(image_array), temperature_array, roi_polygon=cover_polygon
        )
        predictions = self._prepare_predictions(detector_result, temperature_array)
        predictions = self._limit_predictions_to_gt_count(predictions, gt_hotspots)
        if not gt_hotspots:
            return {
                "hotspots": [],
                "success_rate": 0.0,
                "inference_time_ms": detector_result.inference_time_ms,
                "fps": round(1000 / max(detector_result.inference_time_ms, 0.001), 1),
            }

        comparison_slots = self._make_comparison_slots(
            predictions, gt_hotspots
        )
        return {
            "hotspots": comparison_slots,
            "cover_result": self._make_model_cover_result(detector_result, image_array.shape, gt_cover_box),
            "success_rate": round(100 * sum(slot["is_success"] for slot in comparison_slots) / len(gt_hotspots), 1),
            "inference_time_ms": detector_result.inference_time_ms,
            "fps": round(1000 / max(detector_result.inference_time_ms, 0.001), 1),
        }

    def _run_pytorch_result(self, image_array: np.ndarray, temperature_array: np.ndarray, gt_hotspots: list[list[float]], gt_cover_box: dict) -> dict:
        detector_result = self._get_pytorch_detector().detect(self._to_thermal_matrix(image_array), temperature_array)
        predictions = self._prepare_predictions(detector_result, temperature_array)
        predictions = self._limit_predictions_to_gt_count(predictions, gt_hotspots)
        if not gt_hotspots:
            return {
                "hotspots": [],
                "success_rate": 0.0,
                "inference_time_ms": detector_result.inference_time_ms,
                "fps": round(1000 / max(detector_result.inference_time_ms, 0.001), 1),
            }

        comparison_slots = self._make_comparison_slots(
            predictions, gt_hotspots
        )
        return {
            "hotspots": comparison_slots,
            "cover_result": self._make_model_cover_result(detector_result, image_array.shape, gt_cover_box),
            "success_rate": round(100 * sum(slot["is_success"] for slot in comparison_slots) / len(gt_hotspots), 1),
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

    def _compute_single_image_result(
        self,
        image_file: str,
        image_array: np.ndarray,
        temperature_array: np.ndarray,
        cover_box: dict,
        gt_hotspots: list[list[float]],
    ) -> None:
        """Run only OpenCV hotspot detection for one image."""
        del image_file, cover_box
        opencv_result = self._run_opencv_result(image_array, temperature_array, gt_hotspots, self.current_cover_box)
        self._all_image_model_results = {"opencv": {"opencv": opencv_result}}
        self._all_image_cover_results = {}
        self.current_model_results = {"opencv": opencv_result}
        self.current_hotspot_result = opencv_result
        self.current_cover_result = None
        self.dataset_summary = {}

    def _load_cached_results_for_image(self, image_file: str) -> None:
        """Look up already-computed per-image results (no inference) for Check Results browsing."""
        self.current_model_results = self._all_image_model_results.get(image_file, {})
        self.current_hotspot_result = self.current_model_results.get("opencv")
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
            temperature_array = data_loader.load_temperature_matrix(image_file, image_array)
            if temperature_array is None or temperature_array.shape != image_array.shape[:2]:
                raise RuntimeError(f"Raw Celsius temperature data is missing or mismatched for: {image_file}")
            height, width = image_array.shape[:2]
            cover_box = data_loader.load_yolo_cover_box(image_file, width, height)
            if cover_box is None:
                raise RuntimeError(f"Could not load YOLO C-Cover ground truth for: {image_file}")
            seed_text = f"{image_file}#run{self.run_counter}"
            self._all_image_model_results[image_file] = {
                "opencv": self._run_opencv_result(image_array, temperature_array, gt_hotspots, cover_box),
                "pytorch": self._run_pytorch_result(image_array, temperature_array, gt_hotspots, cover_box),
                "openvino": self._run_openvino_result(image_array, temperature_array, gt_hotspots, cover_box),
            }
            self._all_image_cover_results[image_file] = self._all_image_model_results[image_file]["opencv"]["cover_result"]

    def _compute_dataset_summary(self) -> dict[str, dict]:
        """Aggregate the cached per-image results (see _compute_all_image_results) across the
        whole dataset. Reads from cache only - runs no inference itself."""
        summary: dict[str, dict] = {}
        hotspot_distances: list[float] = []
        hotspot_normalized_distances: list[float] = []
        hotspot_temperature_deltas: list[float] = []
        hotspot_pck_hits = 0
        hotspot_pck_total = 0
        hotspot_result_counts = {"Perfect Hit": 0, "Acceptable Hit": 0, "Miss": 0}
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
                if model_key == "opencv":
                    image_array = data_loader.load_thermal_image(image_file)
                    if image_array is None:
                        continue
                    temperature_array = data_loader.load_temperature_matrix(image_file, image_array)
                    image_diagonal = float(np.hypot(image_array.shape[1], image_array.shape[0]))
                    predictions = [
                        hotspot for hotspot in result["hotspots"]
                        if hotspot.get("has_prediction", True) and hotspot.get("coordinate") is not None
                    ]
                    for ground_truth_point in gt_hotspots:
                        if predictions:
                            nearest_prediction = min(
                                predictions,
                                key=lambda prediction: float(np.hypot(
                                    prediction["coordinate"][0] - ground_truth_point[0],
                                    prediction["coordinate"][1] - ground_truth_point[1],
                                )),
                            )
                            predicted_x, predicted_y = nearest_prediction["coordinate"]
                            distance = float(np.hypot(
                                predicted_x - ground_truth_point[0],
                                predicted_y - ground_truth_point[1],
                            ))
                        else:
                            distance = None
                        if distance is not None:
                            hotspot_distances.append(distance)
                            hotspot_normalized_distances.append(distance / max(image_diagonal, 1e-6))
                        hotspot_pck_total += 1
                        hotspot_pck_hits += int(distance is not None and distance <= 5.0)
                        normalized_distance = distance / max(image_diagonal, 1e-6) if distance is not None else float("inf")
                        if normalized_distance < 0.01:
                            hotspot_result_counts["Perfect Hit"] += 1
                        elif normalized_distance < 0.015:
                            hotspot_result_counts["Acceptable Hit"] += 1
                        else:
                            hotspot_result_counts["Miss"] += 1
                        if temperature_array is not None:
                            if predictions:
                                gt_temperature = self._temperature_at_coordinate(
                                    temperature_array, ground_truth_point[0], ground_truth_point[1]
                                )
                                hotspot_temperature_deltas.append(
                                    float(nearest_prediction["temperature"]) - gt_temperature
                                )
                cover_result = result.get("cover_result")
                if cover_result is not None:
                    cover_ious.append(cover_result["iou"])
                matched_hotspots = [
                    hotspot for hotspot in result["hotspots"] if hotspot.get("has_prediction", True)
                ]
                all_errors.extend(hotspot["error_px"] for hotspot in matched_hotspots)
                success_rates.append(result["success_rate"])
                inference_times.append(result["inference_time_ms"])
                fps_values.append(result["fps"])
                total_ground_truth += len(gt_hotspots)
                map_predictions.extend(
                    (
                        hotspot["confidence"], image_file,
                        hotspot["coordinate"][0], hotspot["coordinate"][1]
                    )
                    for hotspot in matched_hotspots
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
        summary["hotspot"] = {
            "avg_distance": round(float(np.mean(hotspot_distances)), 2) if hotspot_distances else None,
            "avg_normalized_distance_percent": round(float(np.mean(hotspot_normalized_distances)) * 100.0, 3)
            if hotspot_normalized_distances else None,
            "avg_temperature_delta": round(float(np.mean(hotspot_temperature_deltas)), 2)
            if hotspot_temperature_deltas else None,
            "pck_at_5px": round(100.0 * hotspot_pck_hits / hotspot_pck_total, 1)
            if hotspot_pck_total else None,
            "result_counts": hotspot_result_counts,
            "image_count": len(self.image_list),
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
        threading.Thread(target=self._run_current_image_worker, args=(image_file, True), daemon=True).start()

    def _on_run_one_image(self) -> None:
        """Run all models for one randomly selected image from the dataset."""
        if not self.image_list or self._benchmark_running:
            return
        selected_index = random.randrange(len(self.image_list))
        self.image_index = selected_index
        self._benchmark_running = True
        self.run_counter += 1
        self._set_status("Processing", C["warning"])
        image_file = self.image_list[selected_index]
        threading.Thread(target=self._run_current_image_worker, args=(image_file, False), daemon=True).start()

    def _run_current_image_worker(self, image_file: str, run_all_images: bool) -> None:
        try:
            image_array = data_loader.load_thermal_image(image_file)
            if image_array is None:
                raise RuntimeError(f"Could not load thermal image: {image_file}")
            temperature_array = data_loader.load_temperature_matrix(image_file, image_array)
            if temperature_array is None:
                raise RuntimeError(f"Raw Celsius temperature data not found for: {image_file}")
            if temperature_array.shape != image_array.shape[:2]:
                raise RuntimeError(
                    f"Temperature shape {temperature_array.shape} does not match image shape {image_array.shape[:2]}"
                )
            height, width = image_array.shape[:2]
            cover_box = data_loader.load_yolo_cover_box(image_file, width, height)
            if cover_box is None:
                raise RuntimeError(f"Could not load YOLO C-Cover ground truth for: {image_file}")
            ground_truth_hotspots = self.gt_hotspot_map.get(image_file, [])
            self._image_array = image_array
            self._temperature_array = temperature_array
            self.current_cover_box = cover_box
            self.current_gt_hotspots = ground_truth_hotspots
            if run_all_images:
                self._compute_benchmark_results(image_file)
            else:
                self._compute_single_image_result(image_file, image_array, temperature_array, cover_box, ground_truth_hotspots)
            self.root.after(0, lambda: self._complete_benchmark_run(image_array, cover_box, ground_truth_hotspots, image_file))
        except Exception as error:
            error_message = str(error)
            self.root.after(0, self._fail_benchmark_run, error_message)

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
                if not openvino_result:
                    openvino_description.set("Paused")
                else:
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
                if not pytorch_result:
                    pytorch_description.set("Paused")
                else:
                    pytorch_description.set(
                        f"Device: {self.pytorchDeviceVar.get()} | Precision: {self.pytorchPrecisionVar.get()} | "
                        f"Latency: {pytorch_result.get('inference_time_ms', 0.0):.1f} ms"
                    )
            if not openvino_result or not pytorch_result:
                for model_key in MODEL_KEYS:
                    description = self.model_description_vars.get(model_key)
                    if description is not None:
                        description.set("Paused")
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
        cover_result = result.get("cover_result") if result else None

        if overlays["show_gt_cover"].get() and self.current_cover_box:
            self._draw_cover_polygon(axes, self.current_cover_box, color=C["cover_gt"], dashed=False)
        if overlays["show_cover"].get() and cover_result:
            self._draw_cover_polygon(axes, cover_result["detected_cover_box"], color=C["cover_predicted"], dashed=False)

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

    def _on_model_mousewheel(self, event, model_key: str):
        """Handle model-panel zoom with Tk directly, bypassing Matplotlib's Tk event adapter."""
        try:
            if event.delta == 0:
                return "break"
            self._zoom_by_factor(model_key, 1.2 if event.delta > 0 else 1 / 1.2)
        except Exception as error:
            print(f"[ui] Model zoom failed: {error}")
        return "break"

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

        tk.Label(frame, text="C-Cover OBB Metrics", bg=C["thermal_surface"], fg=C["text"], font=(FF, 12, "bold")).pack(anchor=tk.W, pady=(0, 3))

        cover_result = result.get("cover_result")
        if cover_result is None:
            tk.Label(frame, text="No C-Cover result.", bg=C["thermal_surface"], fg=C["muted"], font=(FF, 10)).pack(anchor=tk.W)
            return
        tk.Label(frame, text=f'GT / Predicted corners: 4 / 4', bg=C["thermal_surface"], fg=C["muted"], font=(FF, 10)).pack(anchor=tk.W)
        tk.Label(frame, text=f'Confidence: {cover_result["confidence"]}%   IoU: {cover_result["iou"]}',
                 bg=C["thermal_surface"], fg=C["cover_predicted"], font=(FF, 10, "bold")).pack(anchor=tk.W)
        tk.Label(frame, text=f'Status: {cover_result["status"]}', bg=C["thermal_surface"],
                 fg=C["success"] if cover_result["status"] == "PASS" else C["error"],
                 font=(FF, 10, "bold")).pack(anchor=tk.W, pady=(0, 5))

        metrics_content = tk.Frame(frame, bg=C["thermal_surface"])
        metrics_content.pack(fill=tk.BOTH, expand=True)
        metrics_content.grid_columnconfigure(0, weight=1)
        metrics_content.grid_columnconfigure(1, weight=0, minsize=1)
        metrics_content.grid_columnconfigure(2, weight=1)

        cover_column = tk.Frame(metrics_content, bg=C["thermal_surface"])
        cover_column.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        tk.Frame(metrics_content, bg=C["border"], width=1).grid(row=0, column=1, sticky="ns")
        stats_column = tk.Frame(metrics_content, bg=C["thermal_surface"])
        stats_column.grid(row=0, column=2, sticky="nsew", padx=(10, 0))

        tk.Label(cover_column, text="GT: 4 corners", bg=C["thermal_surface"], fg=C["cover_gt"], font=(FF, 10, "bold")).pack(anchor=tk.W)
        tk.Label(cover_column, text="Predicted: 4 corners", bg=C["thermal_surface"], fg=C["cover_predicted"], font=(FF, 10, "bold")).pack(anchor=tk.W)

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
        """Render the OpenCV hotspot card with result, heatmap, and comparison table."""
        if self.detection_status_frame is None:
            return
        frame: tk.Frame = self.detection_status_frame
        self._clear_frame(frame)

        content = tk.Frame(frame, bg=C["thermal_surface"])
        content.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        hotspot_result = self.current_hotspot_result or {}
        predictions = [
            hotspot for hotspot in hotspot_result.get("hotspots", [])
            if hotspot.get("has_prediction", True)
        ]
        image_height, image_width = self._image_array.shape[:2] if self._image_array is not None else (1, 1)
        metric_values = dict(self._calculate_hotspot_metrics(
            predictions, self.current_gt_hotspots, float(np.hypot(image_width, image_height))
        ))
        result_text = metric_values.get("Result", "N/A")
        result_label = "PERFECT HIT" if "Perfect Hit" in result_text and "Miss" not in result_text else (
            "ACCEPTABLE HIT" if "Acceptable Hit" in result_text and "Miss" not in result_text else "MISS"
        )
        result_panel = tk.Frame(
            content, bg=C["thermal_surface"],
            highlightbackground=C["border"], highlightthickness=1,
        )
        result_panel.pack(fill=tk.X, pady=(0, 5))
        result_badges = tk.Frame(result_panel, bg=C["thermal_surface"])
        result_badges.pack(side=tk.LEFT, padx=6, pady=4)
        for index, result_item in enumerate(result_text.splitlines(), start=1):
            if index > len(predictions):
                break
            result_name = result_item.split(": ", 1)[-1]
            badge_color = C["success"] if result_name == "Perfect Hit" else C["warning"] if result_name == "Acceptable Hit" else C["error"]
            card = tk.Frame(result_badges, bg=C["thermal_surface"], highlightbackground=badge_color, highlightthickness=1)
            card.pack(side=tk.LEFT, padx=(0, 6))
            tk.Label(card, text=f"Hotspot #{index}", bg=C["thermal_surface"], fg=C["muted"],
                     font=(FF, 8, "bold"), padx=6, pady=2).pack(side=tk.LEFT)
            tk.Label(card, text=result_name, bg=badge_color, fg="white",
                     font=(FF, 8, "bold"), padx=6, pady=2).pack(side=tk.LEFT)
        quality_values = metric_values.get("Match Quality", "N/A").split(" / ")
        quality_panel = tk.Frame(result_panel, bg=C["thermal_surface"])
        quality_panel.pack(side=tk.RIGHT, padx=8, pady=3)
        tk.Label(quality_panel, text="Match Quality", bg=C["thermal_surface"], fg=C["muted"],
                 font=(FF, 8, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        for index, quality in enumerate(quality_values, start=1):
            if quality == "N/A":
                continue
            try:
                quality_value = float(quality.rstrip("%"))
            except ValueError:
                quality_value = 0.0
            quality_color = (
                C["success"] if quality_value >= 99.0
                else C["warning"] if quality_value >= 97.5
                else C["error"]
            )
            quality_badge = tk.Frame(
                quality_panel, bg=quality_color,
                highlightbackground=quality_color, highlightthickness=1,
            )
            quality_badge.pack(side=tk.LEFT, padx=(0, 5))
            tk.Label(
                quality_badge, text=f"#{index} {quality}", bg=quality_color, fg="white",
                font=(FF, 8, "bold"), padx=6, pady=2,
            ).pack()

        canvas_area = tk.Frame(
            content, bg=C["thermal_surface"],
            highlightbackground=C["border"], highlightthickness=1,
        )
        canvas_area.pack(fill=tk.X, pady=(0, 1))
        canvas_area.configure(height=240)
        canvas_area.pack_propagate(False)
        heatmap_area = tk.Frame(canvas_area, bg=C["thermal_surface"])
        heatmap_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(heatmap_area, bg=C["thermal_surface"], width=520, height=240, highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        self._build_hotspot_zoom_control(heatmap_area)
        canvas.update_idletasks()
        width = max(420, canvas.winfo_width())
        height = max(220, canvas.winfo_height())
        canvas.bind("<MouseWheel>", self._on_hotspot_zoom_scroll)
        canvas.bind("<ButtonPress-1>", self._on_hotspot_pan_press)
        canvas.bind("<B1-Motion>", self._on_hotspot_pan_motion)
        canvas.bind("<ButtonRelease-1>", self._on_hotspot_pan_release)

        gt_cover = self.current_cover_box
        bottom_table = tk.Frame(
            content, bg=C["thermal_surface"],
            highlightbackground=C["border"], highlightthickness=1,
        )
        bottom_table.pack(fill=tk.X, pady=(4, 0))
        table_headers = ("Hotspot", "GT", "Pred", "Dist (px)", "Normalized Dist (%)", "Result")
        table_widths = (14, 14, 14, 14, 16, 14)
        for column, header in enumerate(table_headers):
            tk.Label(
                bottom_table, text=header, bg=C["thermal_surface"], fg=C["muted"],
                font=(FF, 8, "bold"), width=table_widths[column], anchor="w",
            ).grid(row=0, column=column, sticky="w", padx=(4, 0))
        normalized_values = metric_values.get("Normalized Dist (%)", "N/A").split(" / ")
        result_values = metric_values.get("Result", "N/A").splitlines()
        for row_index, prediction in enumerate(predictions, start=1):
            gt_x, gt_y = prediction["ground_truth"]
            pred_x, pred_y = prediction["coordinate"]
            table_values = (
                f"#{row_index}",
                f"({gt_x:.0f},{gt_y:.0f})",
                f"({pred_x:.0f},{pred_y:.0f})",
                f'{prediction["error_px"]:.1f}',
                normalized_values[row_index - 1] if row_index <= len(normalized_values) else "N/A",
                result_values[row_index - 1].split(": ", 1)[-1]
                if row_index <= len(result_values) else "N/A",
            )
            for column, value in enumerate(table_values):
                tk.Label(
                    bottom_table, text=value, bg=C["thermal_surface"],
                    fg=C["cover_gt"] if column == 1 else C["cover_predicted"] if column in (2, 3, 4, 5) else C["muted"],
                    font=(FF, 8, "bold"), width=table_widths[column], anchor="w",
                ).grid(row=row_index, column=column, sticky="w", padx=(4, 0))

        all_points = list(self.current_gt_hotspots)
        hotspot_polygon: list[tuple[float, float]] | None = None
        if gt_cover:
            all_points.extend(gt_cover[corner] for corner in ("top_left", "top_right", "bottom_right", "bottom_left"))
        all_points.extend(hotspot["coordinate"] for hotspot in predictions)
        if not all_points:
            canvas.create_text(210, 110, text="No hotspot data", fill=C["muted"], font=(FF, 10))
            return

        min_x = min(float(point[0]) for point in all_points)
        max_x = max(float(point[0]) for point in all_points)
        min_y = min(float(point[1]) for point in all_points)
        max_y = max(float(point[1]) for point in all_points)
        span_x = max(1.0, max_x - min_x)
        span_y = max(1.0, max_y - min_y)
        center_data_x = (min_x + max_x) / 2.0
        center_data_y = (min_y + max_y) / 2.0
        left_clearance = 12.0
        zoom_clearance = 56.0
        center_canvas_x = left_clearance + (width - left_clearance - zoom_clearance) / 2.0
        center_canvas_y = height / 2.0 + 4.0
        available_width = max(1.0, width - left_clearance - zoom_clearance)
        available_height = max(1.0, height - 38.0)
        scale = min(available_width / span_x, available_height / span_y) * 0.88 * self._hotspot_zoom_scale

        def map_point(point: tuple[float, float]) -> tuple[float, float]:
            return (
                center_canvas_x + (float(point[0]) - center_data_x) * scale + self._hotspot_pan_offset[0],
                center_canvas_y + (float(point[1]) - center_data_y) * scale + self._hotspot_pan_offset[1],
            )

        label_font = tkfont.Font(family=FF, size=8, weight="bold")
        label_boxes: list[tuple[float, float, float, float]] = []

        def place_label(
            x: float,
            y: float,
            text: str,
            color: str,
            candidates: tuple[tuple[float, float, str], ...],
            allowed_polygon: list[tuple[float, float]] | None = None,
            background: str | None = None,
            border: str | None = None,
        ) -> None:
            """Place a label at the first available position without overlap."""
            text_width = float(label_font.measure(text))
            text_height = float(label_font.metrics("linespace"))
            label_padding = 3.0 if background is not None else 0.0
            best_candidate = candidates[0]
            best_overlap = float("inf")
            for offset_x, offset_y, anchor in candidates:
                label_x = x + offset_x
                label_y = y + offset_y
                if anchor in ("e", "ne", "se"):
                    box_left = label_x - text_width
                elif anchor in ("center", "n", "s"):
                    box_left = label_x - text_width / 2.0
                else:
                    box_left = label_x
                if anchor in ("s", "se", "sw"):
                    box_top = label_y - text_height
                elif anchor in ("center", "e", "w"):
                    box_top = label_y - text_height / 2.0
                else:
                    box_top = label_y
                box_left -= label_padding
                box_top -= label_padding
                box_right = box_left + text_width + label_padding * 2
                box_bottom = box_top + text_height + label_padding * 2
                polygon_penalty = 0.0
                if allowed_polygon is not None:
                    polygon = np.asarray(allowed_polygon, dtype=np.float32)
                    label_corners = (
                        (box_left, box_top), (box_right, box_top),
                        (box_right, box_bottom), (box_left, box_bottom),
                    )
                    outside_count = sum(
                        cv2.pointPolygonTest(polygon, corner, False) < 0
                        for corner in label_corners
                    )
                    polygon_penalty = float(outside_count) * 10000.0
                if box_left < 2 or box_right > width - 2 or box_top < 2 or box_bottom > height - 2:
                    overlap = abs(min(box_left, 2)) + abs(max(box_right - width + 2, 0))
                    overlap += abs(min(box_top, 2)) + abs(max(box_bottom - height + 2, 0))
                else:
                    overlap = 0.0
                for old_left, old_top, old_right, old_bottom in label_boxes:
                    overlap += max(0.0, min(box_right, old_right) - max(box_left, old_left)) * max(
                        0.0, min(box_bottom, old_bottom) - max(box_top, old_top)
                    )
                overlap += polygon_penalty
                if overlap < best_overlap:
                    best_overlap = overlap
                    best_candidate = (offset_x, offset_y, anchor)
                if overlap == 0.0:
                    break

            offset_x, offset_y, anchor = best_candidate
            label_x = x + offset_x
            label_y = y + offset_y
            if anchor in ("e", "ne", "se"):
                box_left = label_x - text_width
            elif anchor in ("center", "n", "s"):
                box_left = label_x - text_width / 2.0
            else:
                box_left = label_x
            if anchor in ("s", "se", "sw"):
                box_top = label_y - text_height
            elif anchor in ("center", "e", "w"):
                box_top = label_y - text_height / 2.0
            else:
                box_top = label_y
            label_boxes.append((
                box_left - label_padding,
                box_top - label_padding,
                box_left + text_width + label_padding,
                box_top + text_height + label_padding,
            ))
            if background is not None:
                canvas.create_rectangle(
                    box_left - label_padding,
                    box_top - label_padding,
                    box_left + text_width + label_padding,
                    box_top + text_height + label_padding,
                    fill=background,
                    outline=border or background,
                    width=1,
                )
            canvas.create_text(label_x, label_y, text=text, anchor=anchor, fill=color, font=label_font)

        if gt_cover:
            ordered_cover_points = self._order_cover_corners(gt_cover)
            cover_points = [map_point(ordered_cover_points[corner]) for corner in ("TL", "TR", "BR", "BL")]
            polygon_points = [coordinate for point in cover_points for coordinate in point]
            if self._image_array is not None:
                source_points = np.asarray(
                    [ordered_cover_points[corner] for corner in ("TL", "TR", "BR", "BL")],
                    dtype=np.float32,
                )
                source_height, source_width = self._image_array.shape[:2]
                crop_left = max(0, int(np.floor(np.min(source_points[:, 0]))))
                crop_top = max(0, int(np.floor(np.min(source_points[:, 1]))))
                crop_right = min(source_width, int(np.ceil(np.max(source_points[:, 0]))) + 1)
                crop_bottom = min(source_height, int(np.ceil(np.max(source_points[:, 1]))) + 1)
                if crop_right > crop_left and crop_bottom > crop_top:
                    crop_image = Image.fromarray(self._image_array[crop_top:crop_bottom, crop_left:crop_right, :3]).convert("RGBA")
                    crop_mask = Image.new("L", crop_image.size, 0)
                    mask_points = [
                        (int(round(point[0] - crop_left)), int(round(point[1] - crop_top)))
                        for point in source_points
                    ]
                    ImageDraw.Draw(crop_mask).polygon(mask_points, fill=255)
                    crop_image.putalpha(crop_mask)
                    display_width = max(1, int(round((crop_right - crop_left) * scale)))
                    display_height = max(1, int(round((crop_bottom - crop_top) * scale)))
                    crop_image = crop_image.resize((display_width, display_height), Image.Resampling.BILINEAR)
                    crop_photo = ImageTk.PhotoImage(crop_image)
                    canvas._hotspot_crop_photo = crop_photo  # type: ignore[attr-defined]
                    crop_canvas_x, crop_canvas_y = map_point((crop_left, crop_top))
                    crop_image_item = canvas.create_image(crop_canvas_x, crop_canvas_y, image=crop_photo, anchor="nw")
                    canvas.tag_lower(crop_image_item)
            canvas.create_polygon(*polygon_points, outline=C["cover_gt"], fill="", width=2)
            hotspot_polygon = cover_points
            for corner_name in ("TL", "TR", "BL", "BR"):
                if not self.overlay_vars["show_corner_points"].get():
                    continue
                x, y = map_point(ordered_cover_points[corner_name])
                gt_x, gt_y = ordered_cover_points[corner_name]
                corner_candidates = {
                    "TL": ((-9, -9, "se"), (9, -9, "sw"), (-9, 9, "ne")),
                    "TR": ((9, -9, "sw"), (-9, -9, "se"), (9, 9, "nw")),
                    "BL": ((-9, 9, "ne"), (-9, -9, "se"), (9, 9, "nw")),
                    "BR": ((9, 9, "nw"), (9, -9, "sw"), (-9, 9, "ne")),
                }[corner_name]
                place_label(
                    x, y, f"{corner_name} ({gt_x:.0f},{gt_y:.0f})", C["cover_gt"], corner_candidates,
                )

        for index, point in enumerate(self.current_gt_hotspots, start=1):
            x, y = map_point(point)
            canvas.create_line(x - 7, y, x + 7, y, fill=C["cover_gt"], width=2)
            canvas.create_line(x, y - 7, x, y + 7, fill=C["cover_gt"], width=2)

        for index, prediction in enumerate(predictions, start=1):
            predicted_point = prediction["coordinate"]
            x, y = map_point(predicted_point)
            canvas.create_rectangle(x - 6, y - 6, x + 6, y + 6, outline=C["cover_predicted"], width=2)
            matched_gt = map_point(prediction["ground_truth"])
            canvas.create_line(matched_gt[0], matched_gt[1], x, y, fill="#FFD54F", width=1, dash=(3, 2))

    @staticmethod
    def _calculate_hotspot_metrics(
        predictions: list[dict],
        gt_hotspots: list[list[float]],
        image_diagonal: float | None = None,
    ) -> list[tuple[str, str]]:
        """Calculate distance, normalized distance, and hit classification metrics."""
        if not gt_hotspots:
            return [("Distance (px)", "N/A"), ("Normalized Dist (%)", "N/A"), ("Result", "N/A")]
        if not predictions:
            return [("Distance (px)", "N/A"), ("Normalized Dist (%)", "N/A"), ("Result", "Miss")]

        distances: list[float] = []
        normalized_distances: list[float] = []
        hit_results: list[str] = []
        for prediction in predictions:
            predicted_x, predicted_y = prediction["coordinate"]
            nearest_index = min(
                range(len(gt_hotspots)),
                key=lambda index: float(np.hypot(
                    predicted_x - gt_hotspots[index][0],
                    predicted_y - gt_hotspots[index][1],
                )),
            )
            distance = float(np.hypot(
                predicted_x - gt_hotspots[nearest_index][0],
                predicted_y - gt_hotspots[nearest_index][1],
            ))
            distances.append(distance)
            normalized_distance = distance / max(image_diagonal or 1.0, 1e-6)
            normalized_distances.append(normalized_distance)
            if normalized_distance < 0.01:
                result_text = "Perfect Hit"
            elif normalized_distance < 0.015:
                result_text = "Acceptable Hit"
            else:
                result_text = "Miss"
            hit_results.append(f"#{len(hit_results) + 1}: {result_text}")

        distance_text = " / ".join(f"{distance:.1f} px" for distance in distances)
        normalized_distance_text = " / ".join(f"{distance * 100.0:.2f}%" for distance in normalized_distances)
        return [
            ("Distance (px)", distance_text),
            ("Normalized Dist (%)", normalized_distance_text),
            ("Match Quality", " / ".join(
                f"{max(0.0, 100.0 - distance * 100.0):.2f}%"
                for distance in normalized_distances
            )),
            ("Result", "\n".join(hit_results)),
        ]

    @staticmethod
    def _order_cover_corners(cover_box: dict) -> dict[str, tuple[float, float]]:
        """Name C-Cover corners from their actual image coordinates."""
        points = [
            tuple(cover_box[corner])
            for corner in ("top_left", "top_right", "bottom_left", "bottom_right")
        ]
        points_by_y = sorted(points, key=lambda point: point[1])
        top_left, top_right = sorted(points_by_y[:2], key=lambda point: point[0])
        bottom_left, bottom_right = sorted(points_by_y[2:], key=lambda point: point[0])
        return {"TL": top_left, "TR": top_right, "BL": bottom_left, "BR": bottom_right}

    def _change_hotspot_zoom(self, factor: float) -> None:
        """Change the left hotspot canvas zoom and redraw its current contents."""
        self._hotspot_zoom_scale = max(1.0, min(self._hotspot_zoom_scale * factor, 4.0))
        self._populate_detection_status()

    def _build_hotspot_zoom_control(self, parent: tk.Frame) -> None:
        """Build the hotspot zoom control fixed to the canvas bottom-right corner."""
        panel_bg = "#182033"
        control_area = tk.Frame(parent, bg=C["thermal_surface"])
        control_area.place(relx=1.0, rely=1.0, anchor="se", x=-8, y=-6)

        def make_button(surface: tk.Canvas, text: str, command, tooltip: str) -> None:
            button = tk.Label(
                surface, text=text, bg=panel_bg, fg=C["text"], font=(FF, 14, "bold"),
                width=2, height=1, cursor="hand2"
            )
            surface.create_window(17, 17, window=button, width=30, height=30)
            button.bind("<Button-1>", lambda _event: command())
            button.bind("<Enter>", lambda _event: button.config(bg="#283653"))
            button.bind("<Leave>", lambda _event: button.config(bg=panel_bg))
            Tooltip(button, tooltip)

        reset_surface = tk.Canvas(control_area, width=34, height=34, bg=C["thermal_surface"], highlightthickness=0, bd=0)
        reset_surface.pack(side=tk.TOP, pady=(0, 5))
        reset_surface.create_oval(2, 2, 32, 32, fill=panel_bg, outline=panel_bg)
        make_button(reset_surface, "↺", self._reset_hotspot_zoom, "Reset hotspot view zoom")

        zoom_surface = tk.Canvas(control_area, width=34, height=66, bg=C["thermal_surface"], highlightthickness=0, bd=0)
        zoom_surface.pack(side=tk.TOP)
        zoom_surface.create_arc(0, 0, 10, 10, start=90, extent=90, fill=panel_bg, outline=panel_bg)
        zoom_surface.create_arc(24, 0, 34, 10, start=0, extent=90, fill=panel_bg, outline=panel_bg)
        zoom_surface.create_arc(0, 56, 10, 66, start=180, extent=90, fill=panel_bg, outline=panel_bg)
        zoom_surface.create_arc(24, 56, 34, 66, start=270, extent=90, fill=panel_bg, outline=panel_bg)
        zoom_surface.create_rectangle(5, 0, 29, 66, fill=panel_bg, outline=panel_bg)
        zoom_surface.create_rectangle(0, 5, 34, 61, fill=panel_bg, outline=panel_bg)
        zoom_surface.create_line(2, 33, 32, 33, fill=C["border"])
        make_button(zoom_surface, "+", lambda: self._change_hotspot_zoom(1.25), "Zoom in hotspot view")
        minus_button = tk.Label(
            zoom_surface, text="−", bg=panel_bg, fg=C["text"], font=(FF, 14, "bold"),
            width=2, height=1, cursor="hand2"
        )
        zoom_surface.create_window(17, 49, window=minus_button, width=30, height=30)
        minus_button.bind("<Button-1>", lambda _event: self._change_hotspot_zoom(1 / 1.25))
        minus_button.bind("<Enter>", lambda _event: minus_button.config(bg="#283653"))
        minus_button.bind("<Leave>", lambda _event: minus_button.config(bg=panel_bg))
        Tooltip(minus_button, "Zoom out hotspot view")

    def _reset_hotspot_zoom(self) -> None:
        """Reset the left hotspot canvas to its fitted view."""
        self._hotspot_zoom_scale = 1.0
        self._hotspot_pan_offset = (0.0, 0.0)
        self._populate_detection_status()

    def _on_hotspot_pan_press(self, event) -> None:
        """Start dragging the left hotspot canvas."""
        self._hotspot_drag_start = (float(event.x), float(event.y))
        event.widget.configure(cursor="fleur")

    def _on_hotspot_pan_motion(self, event) -> None:
        """Move all hotspot visualization items while dragging."""
        if self._hotspot_drag_start is None:
            return
        previous_x, previous_y = self._hotspot_drag_start
        delta_x = float(event.x) - previous_x
        delta_y = float(event.y) - previous_y
        self._hotspot_pan_offset = (
            self._hotspot_pan_offset[0] + delta_x,
            self._hotspot_pan_offset[1] + delta_y,
        )
        event.widget.move("all", delta_x, delta_y)
        self._hotspot_drag_start = (float(event.x), float(event.y))

    def _on_hotspot_pan_release(self, event) -> None:
        """Finish dragging the left hotspot canvas."""
        self._hotspot_drag_start = None
        event.widget.configure(cursor="arrow")

    def _on_hotspot_zoom_scroll(self, event) -> None:
        """Zoom the left hotspot canvas with the mouse wheel."""
        try:
            if event.delta == 0:
                return
            self._change_hotspot_zoom(1.25 if event.delta > 0 else 1 / 1.25)
        except Exception as error:
            print(f"[ui] Hotspot zoom failed: {error}")

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
        for col_index in range(len(MODEL_KEYS) + 1):
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

        self._build_hotspot_summary_card(grid, summary.get("hotspot"))
        for col_index, model_key in enumerate(MODEL_KEYS, start=1):
            self._build_model_summary_card(grid, col_index, model_key, summary.get(model_key), badge_keys)

    def _build_hotspot_summary_card(self, parent: tk.Frame, stats: Optional[dict]) -> None:
        """Build the first Execution Summary column for OpenCV hotspot results."""
        card = tk.Frame(parent, bg=C["thermal_surface"], highlightbackground=C["border"], highlightthickness=1)
        card.grid(row=0, column=0, sticky="new", padx=(0, 8))
        inner = tk.Frame(card, bg=C["thermal_surface"])
        inner.pack(fill=tk.X, padx=5, pady=3)
        tk.Label(inner, text="Hotspot Detection (OpenCV)", bg=C["thermal_surface"], fg=C["text"],
                 font=(FF, 11, "bold")).pack(anchor=tk.W)
        if not stats or stats.get("avg_distance") is None:
            tk.Label(inner, text="Run all images to view metrics.", bg=C["thermal_surface"],
                     fg=C["muted"], font=(FF, 9)).pack(anchor=tk.W, pady=(5, 3))
            return
        metric_values = (
            ("Avg Distance", f'{stats["avg_distance"]:.2f} px'),
            ("Avg Normalized Dist", f'{stats["avg_normalized_distance_percent"]:.3f}%'),
            ("Perfect Hit", str(stats["result_counts"]["Perfect Hit"])),
            ("Acceptable Hit", str(stats["result_counts"]["Acceptable Hit"])),
            ("Miss", str(stats["result_counts"]["Miss"])),
        )
        for label, value in metric_values:
            row = tk.Frame(inner, bg=C["thermal_surface"])
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=f"{label}: ", bg=C["thermal_surface"], fg=C["muted"], font=(FF, 9)).pack(side=tk.LEFT)
            tk.Label(row, text=value, bg=C["thermal_surface"], fg=C["accent"], font=(FF, 9, "bold")).pack(side=tk.LEFT)

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
