from __future__ import annotations

import json
import re
import threading
import tkinter as tk
from tkinter import messagebox
from typing import Callable, Optional
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from thermal.metrics import MetricsCalculator
from thermal.detectors.opencv_detector import OpenCVHotspotDetector
from thermal.detectors.openvino_detector import OpenVINOYOLODetector
from thermal.mapper import RobotTargetMapper
from thermal.generator import HotspotShape, ThermalImageGenerator
from thermal.detectors.yolo_detector import YOLOv8PyTorchDetector

# ── Design tokens ─────────────────────────────────────────────────────────────
C: dict[str, str] = {
    "bg":         "#0F1117",
    "surface":    "#1A1D2B",
    "card":       "#20243A",
    "border":     "#2A2F4A",
    "accent":     "#6366F1",
    "accent_dim": "#4F46E5",
    "success":    "#10B981",
    "warning":    "#F59E0B",
    "error":      "#EF4444",
    "text":       "#F1F5F9",
    "muted":      "#94A3B8",
    "dim":        "#475569",
    "sidebar":    "#131520",
    "sidebar_card": "#1B1F2E",
    "sidebar_accent": "#A5B4FC",
    "scrollbar_track": "#1B1F2E",
    "scrollbar_thumb": "#3C407F",
    "scrollbar_thumb_active": "#4A4F96",
    "header":     "#161929",
}
FF = "Segoe UI"
UI_STATE_FILE = ".physicalai_ui_state.json"
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "opencv": "OpenCV",
    "pytorch": "PyTorch (YOLOv8)",
    "openvino": "OpenVINO (YOLOv8)",
}


class SidebarSlider(tk.Frame):
    """Custom sidebar slider with explicit track/knob colors and get/set API."""

    def __init__(self,
                 parent: tk.Misc,
                 *,
                 label: str,
                 from_: float,
                 to: float,
                 default: float,
                 resolution: float = 1.0,
                 integer: bool = False,
                 width: int = 96):
        super().__init__(parent, bg=str(parent.cget("bg")))
        self._min = float(from_)
        self._max = float(to)
        self._resolution = float(resolution)
        self._integer = integer
        self._trackWidth = width
        self._knobRadius = 8
        self._value = float(default)
        self._dragging = False

        tk.Label(self, text=label, bg=str(parent.cget("bg")), fg=C["text"],
                 font=(FF, 9, "bold")).pack(anchor=tk.W)

        row = tk.Frame(self, bg=str(parent.cget("bg")))
        row.pack(fill=tk.X, pady=(5, 0))
        self._canvas = tk.Canvas(
            row,
            width=self._trackWidth,
            height=38,
            bg=str(parent.cget("bg")),
            highlightthickness=0,
            bd=0,
        )
        self._canvas.pack(fill=tk.X, expand=True)
        self._valueVar = tk.StringVar()

        self._canvas.bind("<Button-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Configure>", lambda _e: self._redraw())
        self.set(default)

    def get(self) -> float | int:
        return int(round(self._value)) if self._integer else float(self._value)

    def set(self, value: float) -> None:
        self._value = self._normalize_value(value)
        self._update_text()
        self._redraw()

    def _normalize_value(self, value: float) -> float:
        clamped = min(max(float(value), self._min), self._max)
        if self._resolution > 0:
            steps = round((clamped - self._min) / self._resolution)
            clamped = self._min + steps * self._resolution
        return round(clamped) if self._integer else clamped

    def _value_to_x(self) -> float:
        usable = max(1, self._trackWidth - (self._knobRadius * 2))
        if self._max <= self._min:
            return float(self._knobRadius)
        ratio = (self._value - self._min) / (self._max - self._min)
        return self._knobRadius + (usable * ratio)

    def _x_to_value(self, x: float) -> float:
        usable = max(1, self._trackWidth - (self._knobRadius * 2))
        ratio = min(max((x - self._knobRadius) / usable, 0.0), 1.0)
        return self._min + ((self._max - self._min) * ratio)

    def _update_text(self) -> None:
        if self._integer:
            self._valueVar.set(f"{int(round(self._value))}")
        else:
            self._valueVar.set(f"{self._value:.1f}")

    def _on_press(self, event) -> None:
        self._dragging = True
        self.set(self._x_to_value(event.x))

    def _on_drag(self, event) -> None:
        if self._dragging:
            self.set(self._x_to_value(event.x))

    def _on_release(self, _event) -> None:
        self._dragging = False

    def _redraw(self) -> None:
        self._canvas.delete("all")
        width = max(1, int(self._canvas.winfo_width() or self._trackWidth))
        self._trackWidth = width
        midY = 28
        self._canvas.create_line(
            self._knobRadius,
            midY,
            width - self._knobRadius,
            midY,
            fill="#303853",
            width=7,
            capstyle=tk.ROUND,
        )
        knobX = self._value_to_x()
        self._canvas.create_text(
            knobX,
            9,
            text=self._valueVar.get(),
            fill=C["text"],
            font=(FF, 8, "bold"),
        )
        self._canvas.create_line(
            self._knobRadius,
            midY,
            knobX,
            midY,
            fill=C["accent"],
            width=7,
            capstyle=tk.ROUND,
        )
        self._canvas.create_oval(
            knobX - self._knobRadius,
            midY - self._knobRadius,
            knobX + self._knobRadius,
            midY + self._knobRadius,
            fill=C["accent"],
            outline="#D7DDF7",
            width=1,
        )


class SidebarScrollbar(tk.Canvas):
    """Canvas-based scrollbar with deterministic colors on Windows."""

    def __init__(self,
                 parent: tk.Misc,
                 *,
                 command: Callable[..., object],
                 width: int = 10):
        self._normalWidth = max(3, int(width))
        super().__init__(
            parent,
            width=self._normalWidth,
            bg=C["scrollbar_track"],
            highlightthickness=0,
            bd=0,
            relief="flat",
            cursor="arrow",
        )
        self._scrollCommand = command
        self._thumbTop = 0.0
        self._thumbBottom = 1.0
        self._dragging = False
        self._dragOffset = 0.0
        self._hover = False

        self.bind("<Configure>", lambda _e: self._redraw())
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _on_enter(self, _event) -> None:
        self._hover = True
        self.configure(cursor="hand2")
        self._redraw()

    def _on_leave(self, _event) -> None:
        self._hover = False
        if not self._dragging:
            self.configure(cursor="arrow")
        self._redraw()

    def set(self, first, last) -> None:
        try:
            self._thumbTop = min(max(float(first), 0.0), 1.0)
            self._thumbBottom = min(max(float(last), self._thumbTop), 1.0)
        except Exception:
            self._thumbTop = 0.0
            self._thumbBottom = 1.0
        self._redraw()

    def _on_press(self, event) -> None:
        try:
            height = max(1.0, float(self.winfo_height()))
            thumbY0, thumbY1 = self._thumb_bounds(height)
            if thumbY0 <= event.y <= thumbY1:
                self._dragging = True
                self._dragOffset = float(event.y) - thumbY0
            else:
                self._move_thumb_to(event.y - ((thumbY1 - thumbY0) * 0.5))
        except Exception:
            self._dragging = False

    def _on_drag(self, event) -> None:
        if not self._dragging:
            return
        self._move_thumb_to(float(event.y) - self._dragOffset)

    def _on_release(self, _event) -> None:
        self._dragging = False
        try:
            pointerWidget = self.winfo_containing(self.winfo_pointerx(), self.winfo_pointery())
            self._hover = pointerWidget == self
        except Exception:
            self._hover = False
        if not self._hover:
            self.configure(cursor="arrow")
        else:
            self.configure(cursor="hand2")
        self._redraw()

    def _thumb_bounds(self, height: float) -> tuple[float, float]:
        topInset = 4.0
        bottomInset = 4.0
        usableHeight = max(1.0, height - topInset - bottomInset)
        thumbY0 = topInset + (usableHeight * self._thumbTop)
        thumbY1 = topInset + (usableHeight * self._thumbBottom)
        minThumbHeight = 28.0
        if (thumbY1 - thumbY0) < minThumbHeight:
            thumbY1 = min(height - bottomInset, thumbY0 + minThumbHeight)
            thumbY0 = max(topInset, thumbY1 - minThumbHeight)
        return thumbY0, thumbY1

    def _move_thumb_to(self, topY: float) -> None:
        try:
            height = max(1.0, float(self.winfo_height()))
            topInset = 4.0
            bottomInset = 4.0
            usableHeight = max(1.0, height - topInset - bottomInset)
            thumbY0, thumbY1 = self._thumb_bounds(height)
            thumbHeight = max(1.0, thumbY1 - thumbY0)
            clampedTop = min(max(float(topY), topInset), topInset + usableHeight - thumbHeight)
            fraction = (clampedTop - topInset) / max(1.0, usableHeight - thumbHeight)
            self._scrollCommand("moveto", fraction)
        except Exception:
            return

    def _redraw(self) -> None:
        self.delete("all")
        width = max(1, int(self.winfo_width()))
        height = max(1, int(self.winfo_height()))
        thumbColor = C["scrollbar_thumb_active"] if (self._hover or self._dragging) else C["scrollbar_thumb"]

        self.create_rectangle(
            0,
            0,
            width,
            height,
            fill=C["scrollbar_track"],
            outline=C["scrollbar_track"],
        )

        thumbY0, thumbY1 = self._thumb_bounds(float(height))
        if self._hover or self._dragging:
            # Keep the widget width stable; only make thumb appear fuller on hover.
            insetX = max(0, (width - 3) // 2)
        else:
            insetX = max(1, (width - 2) // 2)
        thumbWidth = max(1, width - (insetX * 2))
        thumbHeight = max(1, int(thumbY1 - thumbY0))
        radius = max(6, min(thumbWidth // 2, thumbHeight // 2))
        self._create_round_rect(
            insetX,
            int(thumbY0),
            width - insetX,
            int(thumbY1),
            radius=radius,
            fill=thumbColor,
            outline=thumbColor,
        )

    def _create_round_rect(self,
                           x0: int,
                           y0: int,
                           x1: int,
                           y1: int,
                           *,
                           radius: int,
                           fill: str,
                           outline: str) -> None:
        radius = max(0, min(radius, (x1 - x0) // 2, (y1 - y0) // 2))
        points = [
            x0 + radius, y0,
            x1 - radius, y0,
            x1, y0,
            x1, y0 + radius,
            x1, y1 - radius,
            x1, y1,
            x1 - radius, y1,
            x0 + radius, y1,
            x0, y1,
            x0, y1 - radius,
            x0, y0 + radius,
            x0, y0,
        ]
        self.create_polygon(points, smooth=True, fill=fill, outline=outline)


class ThermalHotspotDemo:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PhysicalAI  ·  Thermal Hotspot Analysis Platform")
        self.root.configure(bg=C["bg"])

        self.generator = ThermalImageGenerator(width=320, height=240, noise_std=1.5)
        self.opencv_detector = OpenCVHotspotDetector()
        self.yolo_pytorch_detector: Optional[YOLOv8PyTorchDetector] = None
        self.yolo_openvino_detector: Optional[OpenVINOYOLODetector] = None
        self.robot_mapper = RobotTargetMapper()

        self.current_frame = None
        self.metrics_history: dict = {"opencv": [], "pytorch": [], "openvino": []}

        self._status_var = tk.StringVar(value="Initializing…")
        self.openvinoDeviceVar = tk.StringVar(value="CPU")
        self._kpi: dict[str, tk.StringVar] = {
            k: tk.StringVar(value="—")
            for k in (
                "opencv_lat", "opencv_fps", "opencv_acc", "opencv_conf",
                "pytorch_lat", "pytorch_fps", "pytorch_acc", "pytorch_conf",
                "openvino_lat", "openvino_fps", "openvino_acc", "openvino_conf",
            )
        }
        self._img_figs: dict[str, Figure] = {}
        self._img_cvs: dict[str, FigureCanvasTkAgg] = {}
        self._card_title_vars: dict[str, tk.StringVar] = {}
        self._sub_vars: dict[str, tk.StringVar] = {}
        self._robot_xyz_vars: dict[str, tk.StringVar] = {
            k: tk.StringVar(value="X —  Y —  Z —")
            for k in ("opencv", "pytorch", "openvino")
        }
        self.hotspot_count: Optional[SidebarSlider] = None
        self.noise_scale: Optional[SidebarSlider] = None
        self.benchmark_samples: Optional[tk.IntVar] = None
        self.skipAreaRect: Optional[tuple[int, int, int, int]] = None
        self.targetAreaRect: Optional[tuple[int, int, int, int]] = None
        self._interactionMode: Optional[str] = None  # None | skip | target
        self._skipStart: Optional[tuple[int, int]] = None
        self._skipPreviewRect: Optional[tuple[int, int, int, int]] = None
        self._targetStart: Optional[tuple[int, int]] = None
        self._targetPreviewRect: Optional[tuple[int, int, int, int]] = None
        self._cursorPoint: Optional[tuple[int, int]] = None
        self._dot: Optional[tk.Label] = None
        self.metrics_text: Optional[tk.Text] = None
        self._vis_frame: Optional[tk.Frame] = None
        self._sidebarOuter: Optional[tk.Frame] = None
        self._sidebarCanvas: Optional[tk.Canvas] = None
        self._sidebarEdgeScroll: Optional[SidebarScrollbar] = None
        self._sidebarToggleBtn: Optional[tk.Label] = None
        self._headerToggleBtn: Optional[tk.Label] = None
        self._sidebarMinWidth = 170
        self._sidebarMaxWidth = 420
        self._sidebarResizeStartX = 0
        self._sidebarResizeStartWidth = 200
        self._sidebarResizing = False
        self._sidebarCollapsed = False

        self._setup_ui()
        self._configure_window_size()
        self._initialize_default_models()

    def _configure_window_size(self) -> None:
        """First launch fits display; later launches restore last geometry."""
        self.root.update_idletasks()
        area_x, area_y, screen_w, screen_h = self._get_display_work_area()

        # Keep margins for taskbar/window chrome to avoid off-screen bottom overflow.
        max_w = max(860, screen_w - 12)
        max_h = max(540, screen_h - 12)
        min_w = min(1100, max(860, int(screen_w * 0.68)))
        min_h = min(700, max(520, int(screen_h * 0.62)))

        state = self._load_ui_state()
        restored = False

        if state and state.get("geometry"):
            parsed = self._parse_geometry(str(state["geometry"]))
            if parsed is not None:
                w, h, x, y = parsed
                w = max(min_w, min(w, max_w))
                h = max(min_h, min(h, max_h))
                x = min(max(area_x, x), max(area_x, area_x + screen_w - w))
                y = min(max(area_y, y), max(area_y, area_y + screen_h - h))
                self.root.geometry(f"{w}x{h}+{x}+{y}")
                restored = True

        if not restored:
            target_w = min(max_w, max(min_w, int(screen_w * 0.88)))
            target_h = min(max_h, max(min_h, int(screen_h * 0.80)))
            x = area_x + max(0, (screen_w - target_w) // 2)
            y = area_y + max(0, (screen_h - target_h) // 2)
            self.root.geometry(f"{target_w}x{target_h}+{x}+{y}")

        self.root.minsize(min_w, min_h)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after_idle(self._enforce_window_fit)

    def _get_display_work_area(self) -> tuple[int, int, int, int]:
        """Return (x, y, width, height) of usable display work area."""
        try:
            import ctypes
            from ctypes import wintypes

            class RECT(ctypes.Structure):
                _fields_ = [
                    ("left", wintypes.LONG),
                    ("top", wintypes.LONG),
                    ("right", wintypes.LONG),
                    ("bottom", wintypes.LONG),
                ]

            rect = RECT()
            ok = ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)
            if ok:
                w = int(rect.right - rect.left)
                h = int(rect.bottom - rect.top)
                if w > 0 and h > 0:
                    return int(rect.left), int(rect.top), w, h
        except Exception:
            pass

        return 0, 0, int(self.root.winfo_screenwidth()), int(self.root.winfo_screenheight())

    def _enforce_window_fit(self) -> None:
        """Re-clamp size after widgets are laid out to avoid overflow on startup."""
        self.root.update_idletasks()
        area_x, area_y, area_w, area_h = self._get_display_work_area()
        parsed = self._parse_geometry(self.root.winfo_geometry())
        if parsed is None:
            return

        w, h, x, y = parsed
        max_w = max(860, area_w - 12)
        max_h = max(540, area_h - 12)
        w = min(w, max_w)
        h = min(h, max_h)
        x = min(max(area_x, x), max(area_x, area_x + area_w - w))
        y = min(max(area_y, y), max(area_y, area_y + area_h - h))
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    @staticmethod
    def _parse_geometry(geometry: str) -> Optional[tuple[int, int, int, int]]:
        match = re.match(r"^(\d+)x(\d+)([+-]\d+)([+-]\d+)$", geometry.strip())
        if not match:
            return None
        w, h, x, y = match.groups()
        return int(w), int(h), int(x), int(y)

    def _ui_state_path(self) -> Path:
        return Path(__file__).resolve().parent / UI_STATE_FILE

    def _load_ui_state(self) -> dict:
        path = self._ui_state_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_ui_state(self) -> None:
        path = self._ui_state_path()
        sidebarWidth = 200
        try:
            if self._sidebarOuter is not None:
                sidebarWidth = int(self._sidebarOuter.winfo_width())
        except Exception:
            sidebarWidth = 200
        data = {
            "geometry": self.root.winfo_geometry(),
            "sidebarWidth": int(sidebarWidth),
            "sidebarCollapsed": bool(self._sidebarCollapsed),
        }
        try:
            path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _on_close(self) -> None:
        self._save_ui_state()
        self.root.destroy()

    # ═══════════════════════════════════════════════════════════════════════
    #  UI construction
    # ═══════════════════════════════════════════════════════════════════════

    def _setup_ui(self) -> None:
        try:
            state = self._load_ui_state()
            self._sidebarCollapsed = bool(state.get("sidebarCollapsed", False)) if isinstance(state, dict) else False
        except Exception:
            self._sidebarCollapsed = False

        self.root.columnconfigure(0, weight=0, minsize=200)
        self.root.columnconfigure(1, weight=0, minsize=5)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_sidebar_edge_scrollbar()
        self._build_main()
        self._apply_sidebar_collapsed_state(self._sidebarCollapsed, saveState=False)

    # ── Sidebar ───────────────────────────────────────────────────────────

    def _build_sidebar(self) -> None:
        sidebarWidth = 200
        try:
            state = self._load_ui_state()
            savedWidth = int(state.get("sidebarWidth", 200)) if isinstance(state, dict) else 200
            sidebarWidth = max(self._sidebarMinWidth, min(savedWidth, self._sidebarMaxWidth))
        except Exception:
            sidebarWidth = 200

        sbOuter = tk.Frame(self.root, bg=C["sidebar"], width=sidebarWidth)
        sbOuter.grid(row=0, column=0, sticky="ns")
        sbOuter.grid_propagate(False)
        sbOuter.pack_propagate(False)
        self._sidebarOuter = sbOuter

        resizeHitWidth = 6

        def _start_sidebar_resize(event) -> None:
            try:
                if int(event.x) < max(0, sbOuter.winfo_width() - resizeHitWidth):
                    self._sidebarResizing = False
                    return
                self._sidebarResizeStartX = int(event.x_root)
                self._sidebarResizeStartWidth = int(sbOuter.winfo_width())
                self._sidebarResizing = True
            except Exception:
                self._sidebarResizing = False

        def _drag_sidebar_resize(event) -> None:
            if not self._sidebarResizing:
                return
            try:
                delta = int(event.x_root) - self._sidebarResizeStartX
                self._apply_sidebar_width(self._sidebarResizeStartWidth + delta)
            except Exception:
                return

        def _stop_sidebar_resize(_event=None) -> None:
            self._sidebarResizing = False

        def _update_sidebar_resize_cursor(event) -> None:
            try:
                if int(event.x) >= max(0, sbOuter.winfo_width() - resizeHitWidth):
                    sbOuter.configure(cursor="sb_h_double_arrow")
                else:
                    sbOuter.configure(cursor="")
            except Exception:
                sbOuter.configure(cursor="")

        sbOuter.bind("<ButtonPress-1>", _start_sidebar_resize, add="+")
        sbOuter.bind("<B1-Motion>", _drag_sidebar_resize, add="+")
        sbOuter.bind("<ButtonRelease-1>", _stop_sidebar_resize, add="+")
        sbOuter.bind("<Motion>", _update_sidebar_resize_cursor, add="+")
        sbOuter.bind("<Leave>", lambda _e: sbOuter.configure(cursor=""), add="+")

        sbCanvas = tk.Canvas(sbOuter, bg=C["sidebar"], highlightthickness=0, bd=0)
        sbCanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._sidebarCanvas = sbCanvas

        sb = tk.Frame(sbCanvas, bg=C["sidebar"])
        sbWindow = sbCanvas.create_window((0, 0), anchor="nw", window=sb)

        def _sync_sidebar_width(_event=None) -> None:
            sbCanvas.itemconfigure(sbWindow, width=max(1, sbCanvas.winfo_width()))

        def _sync_scrollregion(_event=None) -> None:
            sbCanvas.configure(scrollregion=sbCanvas.bbox("all"))

        def _on_sidebar_wheel(event) -> None:
            sbCanvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_canvas_scroll(first, last) -> None:
            if self._sidebarEdgeScroll is not None:
                self._sidebarEdgeScroll.set(first, last)

        def _on_sidebar_enter(_event=None) -> None:
            sbCanvas.bind_all("<MouseWheel>", _on_sidebar_wheel)

        def _on_sidebar_leave(_event=None) -> None:
            sbCanvas.unbind_all("<MouseWheel>")

        sbCanvas.configure(yscrollcommand=_on_canvas_scroll)

        sbCanvas.bind("<Configure>", _sync_sidebar_width)
        sb.bind("<Configure>", _sync_scrollregion)
        sbOuter.bind("<Enter>", _on_sidebar_enter)
        sbOuter.bind("<Leave>", _on_sidebar_leave)

        brand = tk.Frame(sb, bg=C["sidebar"])
        brand.pack(fill=tk.X, pady=(16, 0))
        tk.Label(brand, text="⬡", bg=C["sidebar"], fg=C["accent"],
               font=(FF, 15, "bold")).pack(side=tk.LEFT, padx=(8, 3))
        tk.Label(brand, text="PhysicalAI", bg=C["sidebar"], fg=C["text"],
               font=(FF, 10, "bold")).pack(side=tk.LEFT)

        brandToggleBtn = tk.Label(
            brand,
            text="◧",
            bg=C["sidebar"],
            fg=C["muted"],
            font=(FF, 11, "bold"),
            padx=6,
            pady=2,
            cursor="hand2",
        )
        brandToggleBtn.pack(side=tk.RIGHT, padx=(0, 8))
        brandToggleBtn.bind("<Button-1>", lambda _e: self._toggle_sidebar())
        brandToggleBtn.bind("<Enter>", lambda _e: brandToggleBtn.config(fg=C["text"]))
        brandToggleBtn.bind("<Leave>", lambda _e: brandToggleBtn.config(fg=C["muted"]))
        self._sidebarToggleBtn = brandToggleBtn

        tk.Frame(sb, bg=C["border"], height=1).pack(fill=tk.X, pady=(14, 10))

        tk.Label(sb, text="HOTSPOT CONFIGURATION", bg=C["sidebar"], fg=C["sidebar_accent"],
                 font=(FF, 9, "bold")).pack(anchor=tk.W, padx=10, pady=(0, 8))

        cfgCard = self._sidebar_round_card(sb, pady=(0, 10), radius=14)

        self.hotspot_count = self._sidebar_slider(
            cfgCard, "Hotspot Count", 2, 4, 2, integer=True)
        self.noise_scale = self._sidebar_slider(
            cfgCard, "Noise Level", 0.0, 5.0, 1.5, res=0.1)

        skipFrame = tk.Frame(cfgCard, bg=C["sidebar_card"])
        skipFrame.pack(fill=tk.X, padx=10, pady=(0, 0))
        tk.Label(skipFrame, text="Skip Area", bg=C["sidebar_card"], fg=C["text"],
                 font=(FF, 9, "bold")).pack(anchor=tk.W)
        self._sidebar_segmented_actions(
            skipFrame,
            leftText="Set",
            leftCmd=self.start_skip_area_selection,
            rightText="Clear",
            rightCmd=self.clear_skip_area,
        )

        targetFrame = tk.Frame(cfgCard, bg=C["sidebar_card"])
        targetFrame.pack(fill=tk.X, padx=10, pady=(20, 10))
        tk.Label(targetFrame, text="Target Area", bg=C["sidebar_card"], fg=C["text"],
                 font=(FF, 9, "bold")).pack(anchor=tk.W)
        self._sidebar_segmented_actions(
            targetFrame,
            leftText="Set",
            leftCmd=self.start_target_area_selection,
            rightText="Clear",
            rightCmd=self.clear_target_area,
        )

        tk.Label(sb, text="OPENVINO SETTINGS", bg=C["sidebar"], fg=C["sidebar_accent"],
                 font=(FF, 9, "bold")).pack(anchor=tk.W, padx=10, pady=(2, 8))

        ovCard = self._sidebar_round_card(sb, pady=(0, 10), radius=14)
        openvinoFrame = tk.Frame(ovCard, bg=C["sidebar_card"])
        openvinoFrame.pack(fill=tk.X, padx=10, pady=(6, 10))
        tk.Label(openvinoFrame, text="Execution Device", bg=C["sidebar_card"], fg=C["text"],
                 font=(FF, 9, "bold")).pack(anchor=tk.W)
        openvinoOptions = ("CPU", "GPU", "NPU", "AUTO")
        self._openvinoMenuTextVar = tk.StringVar(value=self.openvinoDeviceVar.get())

        openvinoDropdown = tk.Frame(openvinoFrame, bg=C["accent"], bd=0)
        openvinoDropdown.pack(fill=tk.X, pady=(8, 0), ipady=2)

        openvinoValueLabel = tk.Label(
            openvinoDropdown,
            textvariable=self._openvinoMenuTextVar,
            bg=C["accent"],
            fg=C["text"],
            font=(FF, 9, "bold"),
            anchor="w",
            padx=10,
            pady=5,
            cursor="hand2",
        )
        openvinoValueLabel.pack(side=tk.LEFT, fill=tk.X, expand=True)

        openvinoArrowLabel = tk.Label(
            openvinoDropdown,
            text="▾",
            bg=C["accent"],
            fg=C["text"],
            font=(FF, 10, "bold"),
            padx=10,
            pady=5,
            cursor="hand2",
        )
        openvinoArrowLabel.pack(side=tk.RIGHT)

        openvinoMenu = tk.Menu(openvinoDropdown, tearoff=0)
        openvinoMenu.config(
            bg="#23283A",
            fg=C["text"],
            activebackground=C["accent"],
            activeforeground=C["text"],
            bd=0,
            font=(FF, 9),
        )

        def on_select_openvino_device(option: str) -> None:
            self._on_openvino_device_select(option)

        for option in openvinoOptions:
            openvinoMenu.add_command(
                label=f"  {option:<8}",
                command=lambda selected=option: on_select_openvino_device(selected),
            )

        def _open_openvino_menu(_event=None) -> None:
            try:
                openvinoMenu.tk_popup(
                    openvinoDropdown.winfo_rootx(),
                    openvinoDropdown.winfo_rooty() + openvinoDropdown.winfo_height(),
                )
            finally:
                openvinoMenu.grab_release()

        def _ov_hover_enter(_event=None) -> None:
            openvinoDropdown.config(bg=C["accent_dim"])
            openvinoValueLabel.config(bg=C["accent_dim"])
            openvinoArrowLabel.config(bg=C["accent_dim"])

        def _ov_hover_leave(_event=None) -> None:
            openvinoDropdown.config(bg=C["accent"])
            openvinoValueLabel.config(bg=C["accent"])
            openvinoArrowLabel.config(bg=C["accent"])

        openvinoDropdown.bind("<Button-1>", _open_openvino_menu)
        openvinoValueLabel.bind("<Button-1>", _open_openvino_menu)
        openvinoArrowLabel.bind("<Button-1>", _open_openvino_menu)

        openvinoDropdown.bind("<Enter>", _ov_hover_enter)
        openvinoValueLabel.bind("<Enter>", _ov_hover_enter)
        openvinoArrowLabel.bind("<Enter>", _ov_hover_enter)

        openvinoDropdown.bind("<Leave>", _ov_hover_leave)
        openvinoValueLabel.bind("<Leave>", _ov_hover_leave)
        openvinoArrowLabel.bind("<Leave>", _ov_hover_leave)

        tk.Label(sb, text="BATCH RUN", bg=C["sidebar"], fg=C["sidebar_accent"],
                 font=(FF, 9, "bold")).pack(anchor=tk.W, padx=10, pady=(2, 8))

        benchCard = self._sidebar_round_card(sb, pady=(0, 10), radius=14)
        spf = tk.Frame(benchCard, bg=C["sidebar_card"])
        spf.pack(fill=tk.X, padx=10, pady=(4, 8))
        tk.Label(spf, text="Sampling Settings", bg=C["sidebar_card"], fg=C["text"],
                 font=(FF, 9, "bold")).pack(anchor=tk.W)
        self.benchmark_samples = tk.IntVar(value=100)
        tk.Spinbox(spf, from_=10, to=2000, increment=10,
                   textvariable=self.benchmark_samples, width=8,
                   bg=C["accent"], fg=C["text"], buttonbackground=C["accent"],
                   highlightthickness=0, bd=0,
                   relief="flat", font=(FF, 10), insertbackground=C["text"]).pack(fill=tk.X, pady=(4, 0), ipady=4)

        tk.Label(sb, text="v2.0  ·  OpenVINO Platform",
                 bg=C["sidebar"], fg=C["dim"],
                 font=(FF, 7)).pack(side=tk.BOTTOM, pady=10)

    def _build_sidebar_edge_scrollbar(self) -> None:
        if self._sidebarCanvas is None:
            return

        edgeScroll = SidebarScrollbar(
            self.root,
            command=self._sidebarCanvas.yview,
            width=5,
        )
        edgeScroll.grid(row=0, column=1, sticky="ns")
        self._sidebarEdgeScroll = edgeScroll

        def _sync_edge_scroll(first, last) -> None:
            if self._sidebarEdgeScroll is not None:
                self._sidebarEdgeScroll.set(first, last)

        self._sidebarCanvas.configure(yscrollcommand=_sync_edge_scroll)

    def _apply_sidebar_width(self, targetWidth: int) -> None:
        if self._sidebarOuter is None:
            return
        try:
            clamped = max(self._sidebarMinWidth, min(int(targetWidth), self._sidebarMaxWidth))
            self._sidebarOuter.configure(width=clamped)
            self.root.columnconfigure(0, minsize=clamped)
            self.root.update_idletasks()
        except Exception:
            return

    def _toggle_sidebar(self) -> None:
        self._apply_sidebar_collapsed_state(not self._sidebarCollapsed, saveState=True)

    def _apply_sidebar_collapsed_state(self, collapsed: bool, *, saveState: bool) -> None:
        self._sidebarCollapsed = bool(collapsed)

        if self._sidebarOuter is not None and self._sidebarEdgeScroll is not None:
            if self._sidebarCollapsed:
                self._sidebarOuter.grid_remove()
                self._sidebarEdgeScroll.grid_remove()
                self.root.columnconfigure(0, minsize=0)
                self.root.columnconfigure(1, minsize=0)
            else:
                self._sidebarOuter.grid()
                self._sidebarEdgeScroll.grid()
                try:
                    currentWidth = int(self._sidebarOuter.cget("width"))
                except Exception:
                    currentWidth = 200
                clampedWidth = max(self._sidebarMinWidth, min(currentWidth, self._sidebarMaxWidth))
                self.root.columnconfigure(0, minsize=clampedWidth)
                self.root.columnconfigure(1, minsize=5)

        if self._sidebarToggleBtn is not None:
            self._sidebarToggleBtn.config(text="☰" if self._sidebarCollapsed else "◧")
        if self._headerToggleBtn is not None:
            self._headerToggleBtn.config(text="☰")
            if self._sidebarCollapsed:
                self._headerToggleBtn.grid()
            else:
                self._headerToggleBtn.grid_remove()

        if saveState:
            self._save_ui_state()
        self.root.update_idletasks()

    def _sidebar_slider(self, parent: tk.Frame, label: str,
                         from_: float, to: float, default: float,
                         res: float = 1.0,
                         integer: bool = False) -> SidebarSlider:
        panelBg = str(parent.cget("bg"))
        slider = SidebarSlider(
            parent,
            label=label,
            from_=from_,
            to=to,
            default=default,
            resolution=res,
            integer=integer,
            width=96,
        )
        slider.configure(bg=panelBg)
        slider.pack(fill=tk.X, padx=10, pady=(0, 20))
        return slider

    def _canvas_round_rect(self,
                           canvas: tk.Canvas,
                           x0: int,
                           y0: int,
                           x1: int,
                           y1: int,
                           radius: int,
                           *,
                           fill: str,
                           outline: str,
                           width: int = 1,
                           tags: str = ""):
        """Draw a rounded rectangle on canvas to get visible rounded corners."""
        r = max(0, min(radius, (x1 - x0) // 2, (y1 - y0) // 2))
        points = [
            x0 + r, y0,
            x1 - r, y0,
            x1, y0,
            x1, y0 + r,
            x1, y1 - r,
            x1, y1,
            x1 - r, y1,
            x0 + r, y1,
            x0, y1,
            x0, y1 - r,
            x0, y0 + r,
            x0, y0,
        ]
        return canvas.create_polygon(
            points,
            smooth=True,
            splinesteps=20,
            fill=fill,
            outline=outline,
            width=width,
            tags=tags,
        )

    def _sidebar_round_card(self,
                            parent: tk.Frame,
                            *,
                            pady: tuple[int, int] = (0, 10),
                            radius: int = 14) -> tk.Frame:
        """Create a canvas-backed rounded card and return its inner content frame."""
        shell = tk.Frame(parent, bg=C["sidebar"])
        shell.pack(fill=tk.X, padx=6, pady=pady)

        cardCanvas = tk.Canvas(shell, bg=C["sidebar"], highlightthickness=0, bd=0, relief="flat")
        cardCanvas.pack(fill=tk.X, expand=True)

        content = tk.Frame(cardCanvas, bg=C["sidebar_card"])
        contentWindow = cardCanvas.create_window((7, 8), anchor="nw", window=content)

        def redraw_card() -> None:
            w = max(2, int(cardCanvas.winfo_width()))
            h = max(2, int(cardCanvas.winfo_height()))
            cardCanvas.delete("cardShape")
            self._canvas_round_rect(
                cardCanvas,
                1,
                1,
                w - 1,
                h - 1,
                radius,
                fill=C["sidebar_card"],
                outline=C["border"],
                width=1,
                tags="cardShape",
            )
            cardCanvas.tag_lower("cardShape")
            cardCanvas.itemconfigure(contentWindow, width=max(1, w - 14))

        def on_canvas_resize(_event) -> None:
            redraw_card()

        def on_content_resize(event) -> None:
            cardCanvas.configure(height=max(40, int(event.height) + 16))
            redraw_card()

        cardCanvas.bind("<Configure>", on_canvas_resize)
        content.bind("<Configure>", on_content_resize)
        return content

    def _sidebar_segmented_actions(self,
                                   parent: tk.Frame,
                                   *,
                                   leftText: str,
                                   leftCmd,
                                   rightText: str,
                                   rightCmd) -> tk.Frame:
        """Render Set/Clear as separated pill buttons with clear spacing."""
        panelBg = str(parent.cget("bg"))
        holder = tk.Frame(parent, bg=panelBg)
        holder.pack(fill=tk.X, pady=(8, 0))

        segFrame = tk.Frame(holder, bg=panelBg)
        segFrame.pack(fill=tk.X, padx=10)
        segFrame.columnconfigure(0, weight=1)
        segFrame.columnconfigure(1, weight=1)

        leftBtn = tk.Label(
            segFrame,
            text=leftText,
            bg=C["accent"],
            fg=C["text"],
            font=(FF, 9, "bold"),
            padx=14,
            pady=6,
            cursor="hand2",
            bd=0,
        )
        leftBtn.grid(row=0, column=0, sticky="ew")

        rightBtn = tk.Label(
            segFrame,
            text=rightText,
            bg="#303853",
            fg=C["text"],
            font=(FF, 9, "bold"),
            padx=14,
            pady=6,
            cursor="hand2",
            bd=0,
        )
        rightBtn.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        def safe_run(action, action_name: str) -> None:
            try:
                action()
            except Exception as exc:
                messagebox.showerror("Action Error", f"Failed to execute {action_name}: {exc}")

        leftBtn.bind("<Button-1>", lambda _e: safe_run(leftCmd, leftText))
        rightBtn.bind("<Button-1>", lambda _e: safe_run(rightCmd, rightText))

        leftBtn.bind("<Enter>", lambda _e: leftBtn.config(bg=C["accent_dim"]))
        leftBtn.bind("<Leave>", lambda _e: leftBtn.config(bg=C["accent"]))
        rightBtn.bind("<Enter>", lambda _e: rightBtn.config(bg="#3A4463"))
        rightBtn.bind("<Leave>", lambda _e: rightBtn.config(bg="#303853"))
        return holder

    @staticmethod
    def _normalize_openvino_device(value: str) -> str:
        normalized = str(value).strip().upper()
        if normalized in {"CPU", "GPU", "NPU", "AUTO"}:
            return normalized
        return "CPU"

    def _on_openvino_device_select(self, selected: str) -> None:
        """Apply OpenVINO device selection for next model load."""
        try:
            normalized = self._normalize_openvino_device(selected)
            self.openvinoDeviceVar.set(normalized)
            if hasattr(self, "_openvinoMenuTextVar"):
                self._openvinoMenuTextVar.set(normalized)
            # Force re-load on next run so selected device is actually used.
            self.yolo_openvino_detector = None
            self._set_status(f"OpenVINO device set to {normalized} (reload on next run)")
        except Exception as exc:
            messagebox.showerror("OpenVINO Setting Error", str(exc))

    # ── Main area ─────────────────────────────────────────────────────────

    def _build_main(self) -> None:
        self._main = tk.Frame(self.root, bg=C["bg"])
        self._main.grid(row=0, column=2, sticky="nsew")
        self._main.columnconfigure(0, weight=1)
        self._main.rowconfigure(1, weight=1)
        self._build_header()
        self._build_body()

    def _build_header(self) -> None:
        hdr = tk.Frame(self._main, bg=C["header"], height=58)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        hdr.columnconfigure(1, weight=1)

        toggleBtn = tk.Label(
            hdr,
            text="☰",
            bg=C["header"],
            fg=C["muted"],
            font=(FF, 11, "bold"),
            padx=8,
            pady=4,
            cursor="hand2",
        )
        toggleBtn.grid(row=0, column=0, padx=(10, 6), pady=12, sticky="w")
        toggleBtn.bind("<Button-1>", lambda _e: self._toggle_sidebar())
        toggleBtn.bind("<Enter>", lambda _e: toggleBtn.config(fg=C["text"]))
        toggleBtn.bind("<Leave>", lambda _e: toggleBtn.config(fg=C["muted"]))
        self._headerToggleBtn = toggleBtn

        tk.Label(hdr, text="Thermal Hotspot Analysis",
                 bg=C["header"], fg=C["text"],
                 font=(FF, 12, "bold")).grid(row=0, column=1,
                                             padx=(14, 8), pady=16, sticky="w")
        sf = tk.Frame(hdr, bg=C["header"])
        sf.grid(row=0, column=2, sticky="e", padx=8)
        self._dot = tk.Label(sf, text="●", bg=C["header"],
                              fg=C["success"], font=(FF, 10))
        self._dot.pack(side=tk.LEFT)
        tk.Label(sf, textvariable=self._status_var, bg=C["header"],
                 fg=C["muted"], font=(FF, 9)).pack(side=tk.LEFT, padx=(4, 0))

        bf = tk.Frame(hdr, bg=C["header"])
        bf.grid(row=0, column=3, padx=16)
        self._btn(bf, "▶  Single Run",
                  self.run_detection).pack(side=tk.LEFT, padx=(0, 6))
        self._btn(bf, "⟳  Batch Run",
                  self.run_benchmark).pack(side=tk.LEFT)
        toggleBtn.grid_remove()

    def _build_body(self) -> None:
        self._body = tk.Frame(self._main, bg=C["bg"])
        self._body.grid(row=1, column=0, sticky="nsew", padx=20, pady=12)
        self._body.columnconfigure(0, weight=1)
        self._body.rowconfigure(0, weight=1)
        self._body.columnconfigure(1, weight=0)
        self._build_vis_and_metrics(self._body)

    def _build_kpi_row(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent, bg=C["bg"])
        row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        for i in range(3):
            row.columnconfigure(i, weight=1)

        # Per-model Latency + FPS cards
        for i, (mkey, mlabel, color) in enumerate([
            ("opencv",   "OpenCV",   "#10B981"),
            ("pytorch",  "PyTorch",  "#F59E0B"),
            ("openvino", "OpenVINO", "#EC4899"),
        ]):
            card = tk.Frame(row, bg=C["card"],
                             highlightbackground=C["border"],
                             highlightthickness=1)
            card.grid(row=0, column=i, padx=(0 if i == 0 else 8, 0),
                      sticky="nsew")
            tk.Frame(card, bg=color, height=3).pack(fill=tk.X)
            inner = tk.Frame(card, bg=C["card"])
            inner.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)
            tk.Label(inner, text=mlabel.upper(), bg=C["card"], fg=C["muted"],
                     font=(FF, 8, "bold")).pack(anchor=tk.W)

            grid = tk.Frame(inner, bg=C["card"])
            grid.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
            grid.columnconfigure(0, weight=1)
            grid.columnconfigure(1, weight=1)
            grid.rowconfigure(0, weight=1)
            grid.rowconfigure(1, weight=1)

            for idx, (title, unit, key_suffix, value_size) in enumerate([
                ("Processing Time", " ms", "lat", 16),
                ("Frame Rate", " fps", "fps", 16),
                ("Position Error", " px", "acc", 14),
                ("Detection Confidence", " %", "conf", 14),
            ]):
                r = idx // 2
                c = idx % 2
                cell = tk.Frame(grid, bg="#1B1F33", highlightbackground=C["border"], highlightthickness=1)
                cell.grid(row=r, column=c, sticky="nsew", padx=(0 if c == 0 else 4, 0),
                          pady=(0 if r == 0 else 4, 0))

                tk.Label(cell, text=title, bg="#1B1F33", fg=C["dim"],
                         font=(FF, 7, "bold")).pack(anchor=tk.W, padx=8, pady=(6, 0))
                rowv = tk.Frame(cell, bg="#1B1F33")
                rowv.pack(anchor=tk.W, padx=8, pady=(2, 7))
                tk.Label(rowv, textvariable=self._kpi[f"{mkey}_{key_suffix}"],
                         bg="#1B1F33", fg=C["text"],
                         font=(FF, value_size, "bold")).pack(side=tk.LEFT, anchor=tk.S)
                tk.Label(rowv, text=unit, bg="#1B1F33", fg=C["muted"],
                         font=(FF, 8)).pack(side=tk.LEFT, anchor=tk.S, pady=(0, 2))

    def _build_vis_and_metrics(self, parent: tk.Frame) -> None:
        self._vis_frame = tk.Frame(parent, bg=C["bg"])
        self._vis_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        for col in (0, 1, 2):
            self._vis_frame.columnconfigure(col, weight=1, uniform="viz_col")
        for row in (0, 1):
            self._vis_frame.rowconfigure(row, weight=1)

        right = tk.Frame(parent, bg=C["surface"],
                          highlightbackground=C["border"],
                          highlightthickness=1, width=268)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_propagate(False)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        tk.Label(right, text="RUN SUMMARY", bg=C["surface"], fg=C["dim"],
                 font=(FF, 8, "bold")).grid(row=0, column=0, padx=14,
                                             pady=(12, 4), sticky="w")
        tf = tk.Frame(right, bg=C["surface"])
        tf.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        tf.columnconfigure(0, weight=1)
        tf.rowconfigure(0, weight=1)
        self.metrics_text = tk.Text(
            tf, state=tk.DISABLED, wrap=tk.NONE,
            bg=C["surface"], fg=C["text"],
            font=("Consolas", 9), relief="flat",
            selectbackground=C["accent"])
        self.metrics_text.grid(row=0, column=0, sticky="nsew")

        for tag, fg, bold in [
            ("header", C["text"],    True),
            ("key",    "#818CF8",    True),
            ("val",    C["text"],    False),
            ("good",   C["success"], False),
            ("warn",   C["warning"], False),
            ("muted",  C["muted"],   False),
            ("dim",    C["dim"],     False),
        ]:
            self.metrics_text.tag_configure(
                tag, foreground=fg,
                font=("Consolas", 9, "bold" if bold else "normal"))

        for key, title, r, c, cs, color in [
            ("thermal",  "Thermal Hotspot Baseline", 0, 0, 2, "#6366F1"),
            ("mask",     "Reference Result",    0, 2, 1, "#0EA5E9"),
            ("opencv",   MODEL_DISPLAY_NAMES["opencv"],   1, 0, 1, "#10B981"),
            ("pytorch",  MODEL_DISPLAY_NAMES["pytorch"],  1, 1, 1, "#F59E0B"),
            ("openvino", MODEL_DISPLAY_NAMES["openvino"], 1, 2, 1, "#EC4899"),
        ]:
            self._make_img_card(key, title, r, c, cs, color)

    def _make_img_card(self, key: str, title: str, row: int, col: int,
                        colspan: int, accent: str) -> None:
        card = tk.Frame(self._vis_frame, bg=C["card"],
                         highlightbackground=C["border"],
                         highlightthickness=1)
        if row == 1 and colspan == 1:
            pad_x = (4, 4)
        else:
            pad_x = (0 if col == 0 else 8, 0)
        card.grid(row=row, column=col, columnspan=colspan,
                  padx=pad_x,
                  pady=(0 if row == 0 else 8, 0),
                  sticky="nsew")
        card.columnconfigure(0, weight=1)
        card.rowconfigure(1, weight=1)

        tbar = tk.Frame(card, bg=C["card"])
        tbar.grid(row=0, column=0, sticky="ew")
        tk.Frame(tbar, bg=accent, width=3).pack(side=tk.LEFT, fill=tk.Y)
        titleVar = tk.StringVar(value=title)
        self._card_title_vars[key] = titleVar
        tk.Label(tbar, textvariable=titleVar, bg=C["card"], fg=C["text"],
                 font=(FF, 9, "bold")).pack(side=tk.LEFT, padx=10, pady=6)
        sub = tk.StringVar(value="—")
        self._sub_vars[key] = sub
        tk.Label(tbar, textvariable=sub, bg=C["card"], fg=C["muted"],
                 font=(FF, 8)).pack(side=tk.RIGHT, padx=10)

        fig = Figure(figsize=(4.2 if colspan == 2 else 2.8, 2.6), dpi=80)
        fig.patch.set_facecolor(C["card"])
        ax = fig.add_subplot(111)
        ax.set_facecolor("#161929")
        ax.text(0.5, 0.5, "—", ha="center", va="center",
                color=C["dim"], fontsize=10, transform=ax.transAxes)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

        cv = FigureCanvasTkAgg(fig, master=card)
        cv.get_tk_widget().configure(bg=C["card"], highlightthickness=0)
        cv.get_tk_widget().grid(row=1, column=0, sticky="nsew",
                                 padx=2, pady=(0, 2))
        cv.draw()
        self._img_figs[key] = fig
        self._img_cvs[key] = cv

        if key in ("opencv", "pytorch", "openvino"):
            metric_panel = tk.Frame(card, bg=C["card"])
            metric_panel.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 6))
            metric_panel.columnconfigure(0, weight=1)
            metric_panel.columnconfigure(1, weight=1)
            metric_panel.rowconfigure(0, weight=1)
            metric_panel.rowconfigure(1, weight=1)

            for idx, (title_txt, unit, key_suffix, value_size) in enumerate([
                ("Processing Time", "ms", "lat", 11),
                ("Frame Rate", "fps", "fps", 11),
                ("Position Error", "px", "acc", 11),
                ("Detection Confidence", "%", "conf", 11),
            ]):
                rr = idx // 2
                cc = idx % 2
                cell = tk.Frame(metric_panel, bg="#1B1F33", highlightbackground=C["border"],
                                highlightthickness=1)
                cell.grid(row=rr, column=cc, sticky="nsew",
                          padx=(0 if cc == 0 else 4, 0),
                          pady=(0 if rr == 0 else 4, 0))

                tk.Label(cell, text=title_txt, bg="#1B1F33", fg=C["muted"],
                         font=(FF, 8, "normal")).pack(anchor=tk.W, padx=6, pady=(4, 0))
                vrow = tk.Frame(cell, bg="#1B1F33")
                vrow.pack(anchor=tk.W, padx=6, pady=(1, 4))
                tk.Label(vrow, textvariable=self._kpi[f"{key}_{key_suffix}"],
                         bg="#1B1F33", fg=C["text"],
                         font=(FF, value_size, "bold")).pack(side=tk.LEFT)
                tk.Label(vrow, text=f" {unit}", bg="#1B1F33", fg=C["text"],
                         font=(FF, 8)).pack(side=tk.LEFT, pady=(0, 1))

                xyz_row = tk.Frame(card, bg=C["card"])
                xyz_row.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 6))
                tk.Label(xyz_row, text="Robot Target Position", bg=C["card"], fg=C["muted"],
                            font=(FF, 7, "bold")).pack(side=tk.LEFT)
                tk.Label(xyz_row, textvariable=self._robot_xyz_vars[key],
                            bg=C["card"], fg=C["muted"],
                            font=("Consolas", 8)).pack(side=tk.LEFT, padx=(8, 0))

    # ── Helpers ───────────────────────────────────────────────────────────

    def _btn(self, parent: tk.Frame, text: str, cmd,
              primary: bool = True) -> tk.Label:
        bg = C["accent"] if primary else C["surface"]
        hbg = C["accent_dim"] if primary else C["card"]
        b = tk.Label(parent, text=text, bg=bg, fg=C["text"],
                      font=(FF, 9, "bold"), padx=14, pady=6, cursor="hand2")
        b.bind("<Button-1>", lambda e: cmd())
        b.bind("<Enter>",    lambda e: b.config(bg=hbg))
        b.bind("<Leave>",    lambda e: b.config(bg=bg))
        return b

    def _set_status(self, msg: str, busy: bool = False) -> None:
        self._status_var.set(msg)
        if self._dot:
            self._dot.config(fg=C["warning"] if busy else C["success"])

    def _update_kpi(self, results: dict, gt_x: float, gt_y: float) -> None:
        for mkey in ("opencv", "pytorch", "openvino"):
            if mkey in results:
                r = results[mkey]
                lat = r.inference_time_ms
                fps = 1000.0 / lat if lat > 0 else 0.0
                err = MetricsCalculator.localization_error(
                    r.center_x, r.center_y, gt_x, gt_y)
                confidencePct = r.confidence * 100.0
                self._kpi[f"{mkey}_lat"].set(f"{lat:.1f}")
                self._kpi[f"{mkey}_fps"].set(f"{fps:.0f}")
                self._kpi[f"{mkey}_acc"].set(f"{err:.1f}")
                self._kpi[f"{mkey}_conf"].set(f"{confidencePct:.1f}")
            else:
                self._kpi[f"{mkey}_lat"].set("—")
                self._kpi[f"{mkey}_fps"].set("—")
                self._kpi[f"{mkey}_acc"].set("—")
                self._kpi[f"{mkey}_conf"].set("—")

    def _set_openvino_card_title(self, executionDeviceText: str = "") -> None:
        if "openvino" not in self._card_title_vars:
            return
        baseTitle = MODEL_DISPLAY_NAMES["openvino"]
        if executionDeviceText:
            self._card_title_vars["openvino"].set(f"{baseTitle} - {executionDeviceText}")
        else:
            self._card_title_vars["openvino"].set(baseTitle)

    def _draw(self, key: str, image: np.ndarray,
               cmap: str = "inferno",
               centers=None, result=None,
               subtitle: str = "") -> None:
        fig = self._img_figs[key]
        cv  = self._img_cvs[key]
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor("#161929")
        imgH, imgW = int(image.shape[0]), int(image.shape[1])
        if key in ("thermal", "opencv", "pytorch", "openvino"):
            # Keep all model cards on the same geometric reference and keyboard guide.
            ax.imshow(image, cmap=cmap, aspect="auto", alpha=1.0, zorder=2)
            self._draw_keyboard_c_deck_reference(ax, image.shape)
        else:
            ax.imshow(image, cmap=cmap, aspect="auto", alpha=1.0, zorder=2)
        if centers:
            ax.scatter([p[0] for p in centers], [p[1] for p in centers],
                       c="#22D3EE", s=50, marker="*", zorder=5)
        if result is not None:
            ax.scatter([result.center_x], [result.center_y],
                       c="#EF4444", s=55, marker="x", zorder=6, linewidths=2)
        if key == "thermal":
            self._draw_hotspot_constraints(ax, image.shape)

        # Lock axes to the exact thermal pixel boundaries so overlays touch all edges.
        ax.set_xlim(-0.5, imgW - 0.5)
        ax.set_ylim(imgH - 0.5, -0.5)

        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])
        ax.axis("off")
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
        cv.draw()
        self._sub_vars[key].set(subtitle)

    def _get_c_deck_bounds(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Return the interactive C-deck bounds as (x0, y0, x1, y1)."""
        x0 = 0
        y0 = 0
        x1 = max(0, int(width) - 1)
        y1 = max(0, int(height) - 1)
        return x0, y0, x1, y1

    def _draw_keyboard_c_deck_reference(self, ax, image_shape: tuple[int, int]) -> None:
        """Draw a common Windows laptop C-deck with labeled keyboard zones."""
        try:
            h, w = int(image_shape[0]), int(image_shape[1])
            if h <= 0 or w <= 0:
                return

            # Main deck frame.
            deckX0, deckY0 = -0.5, -0.5
            deckW, deckH = float(w), float(h)
            ax.add_patch(Rectangle(
                (deckX0, deckY0), deckW, deckH,
                facecolor="none", edgecolor="#D5D9E0",
                linewidth=1.1, alpha=0.42, zorder=3,
            ))

            # Integer bounds are still used for interaction-safe clamping logic.
            _, _, _, deckY1 = self._get_c_deck_bounds(w, h)

            # Keyboard zone.
            keyX0 = int(w * 0.04)
            keyX1 = int(w * 0.96)
            keyY0 = int(h * 0.20)
            keyY1 = int(h * 0.68)
            keyW = max(1, keyX1 - keyX0)
            keyH = max(1, keyY1 - keyY0)
            ax.add_patch(Rectangle(
                (keyX0, keyY0), keyW, keyH,
                facecolor="none", edgecolor="#E9ECF2",
                linewidth=1.0, alpha=0.42, zorder=3,
            ))

            # Draw a rough keycap to mimic common Windows laptop key spacing.
            def drawKey(colStart: float, colSpan: float, rowIndex: int, label: str,
                        totalCols: float = 15.0, totalRows: int = 5) -> None:
                keyGapX = keyW * 0.006
                keyGapY = keyH * 0.04
                rowH = keyH / float(totalRows)
                x0 = keyX0 + keyW * (colStart / totalCols) + keyGapX
                y0 = keyY0 + rowH * rowIndex + keyGapY
                cellW = keyW * (colSpan / totalCols) - keyGapX * 2.0
                cellH = rowH - keyGapY * 2.0
                if cellW < 1 or cellH < 1:
                    return
                ax.add_patch(Rectangle(
                    (x0, y0), cellW, cellH,
                    fill=False, edgecolor="#E2E8F0",
                    linewidth=0.55, alpha=0.58, zorder=4,
                ))
                ax.text(
                    x0 + cellW * 0.08,
                    y0 + cellH * 0.62,
                    label,
                    color="#FFFFFF",
                    fontsize=5.0,
                    alpha=0.98,
                    zorder=5,
                    bbox={"boxstyle": "round,pad=0.05", "facecolor": "#000000", "edgecolor": "none", "alpha": 0.16},
                )

            # Row 0: Esc + F row.
            fRowLabels = ["Esc", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12"]
            colCursor = 0.0
            drawKey(colCursor, 1.2, 0, "Esc")
            colCursor += 1.8
            for fLabel in fRowLabels[1:]:
                drawKey(colCursor, 0.9, 0, fLabel)
                colCursor += 1.0

            # Row 1: number row.
            row1 = ["`", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "=", "Backspace"]
            colCursor = 0.0
            for keyLabel in row1[:-1]:
                drawKey(colCursor, 1.0, 1, keyLabel)
                colCursor += 1.0
            drawKey(colCursor, 2.0, 1, row1[-1])

            # Row 2: QWERTY row.
            row2 = ["Tab", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "[", "]", "\\"]
            colCursor = 0.0
            drawKey(colCursor, 1.5, 2, row2[0])
            colCursor += 1.5
            for keyLabel in row2[1:13]:
                drawKey(colCursor, 1.0, 2, keyLabel)
                colCursor += 1.0
            drawKey(colCursor, 1.5, 2, row2[-1])

            # Row 3: home row.
            row3 = ["Caps", "A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "'", "Enter"]
            colCursor = 0.0
            drawKey(colCursor, 1.8, 3, row3[0])
            colCursor += 1.8
            for keyLabel in row3[1:12]:
                drawKey(colCursor, 1.0, 3, keyLabel)
                colCursor += 1.0
            drawKey(colCursor, 2.2, 3, row3[-1])

            # Row 4: modifiers + space.
            drawKey(0.0, 2.4, 4, "Shift")
            drawKey(2.4, 1.0, 4, "Z")
            drawKey(3.4, 1.0, 4, "X")
            drawKey(4.4, 1.0, 4, "C")
            drawKey(5.4, 1.0, 4, "V")
            drawKey(6.4, 1.0, 4, "B")
            drawKey(7.4, 1.0, 4, "N")
            drawKey(8.4, 1.0, 4, "M")
            drawKey(9.4, 1.0, 4, ",")
            drawKey(10.4, 1.0, 4, ".")
            drawKey(11.4, 1.0, 4, "/")
            drawKey(12.4, 2.6, 4, "Shift")

            # Bottom modifier strip under keyboard.
            modY = keyY1 + int(h * 0.015)
            modH = int(h * 0.055)
            modItems = [
                (0.0, 1.3, "Ctrl"),
                (1.3, 1.1, "Fn"),
                (2.4, 1.2, "Win"),
                (3.6, 1.3, "Alt"),
                (4.9, 5.6, "Space"),
                (10.5, 1.3, "Alt"),
                (13.1, 1.9, "Ctrl"),
            ]
            for colStart, colSpan, label in modItems:
                x0 = keyX0 + keyW * (colStart / 15.0)
                w0 = keyW * (colSpan / 15.0)
                ax.add_patch(Rectangle(
                    (x0, modY), max(1, w0), max(1, modH),
                    fill=False, edgecolor="#E2E8F0",
                    linewidth=0.55, alpha=0.55, zorder=4,
                ))
                ax.text(x0 + w0 * 0.08, modY + modH * 0.64, label,
                        color="#FFFFFF", fontsize=5.0, alpha=0.98, zorder=5,
                    bbox={"boxstyle": "round,pad=0.05", "facecolor": "#000000", "edgecolor": "none", "alpha": 0.16})

            # Arrow cluster (common bottom-right placement).
            arrowRegionStart = 11.82
            arrowRegionEnd = 13.08
            arrowRegionX0 = keyX0 + keyW * (arrowRegionStart / 15.0)
            arrowRegionX1 = keyX0 + keyW * (arrowRegionEnd / 15.0)
            arrowW = max(1, int(keyW * 0.020))
            arrowH = max(1, int(modH * 0.40))
            arrowGap = max(1, int(arrowW * 0.24))
            arrowBlockW = arrowW * 3 + arrowGap * 2
            arrowX0 = int(arrowRegionX0 + max(0, (arrowRegionX1 - arrowRegionX0 - arrowBlockW) * 0.5))
            arrowY0 = modY + int(modH * 0.10)
            ax.add_patch(Rectangle((arrowX0 + arrowW + arrowGap, arrowY0), arrowW, arrowH,
                                   fill=False, edgecolor="#E2E8F0", linewidth=0.5, alpha=0.55, zorder=4))
            ax.add_patch(Rectangle((arrowX0, arrowY0 + arrowH + arrowGap), arrowW, arrowH,
                                   fill=False, edgecolor="#E2E8F0", linewidth=0.5, alpha=0.55, zorder=4))
            ax.add_patch(Rectangle((arrowX0 + arrowW + arrowGap, arrowY0 + arrowH + arrowGap), arrowW, arrowH,
                                   fill=False, edgecolor="#E2E8F0", linewidth=0.5, alpha=0.55, zorder=4))
            ax.add_patch(Rectangle((arrowX0 + (arrowW + arrowGap) * 2, arrowY0 + arrowH + arrowGap), arrowW, arrowH,
                                   fill=False, edgecolor="#E2E8F0", linewidth=0.5, alpha=0.55, zorder=4))

            # Bright arrow glyphs improve readability in the thermal overlay.
            upX = arrowX0 + arrowW + arrowGap + arrowW * 0.25
            upY = arrowY0 + arrowH * 0.75
            dnX = upX
            dnY = arrowY0 + arrowH + arrowGap + arrowH * 0.75
            lfX = arrowX0 + arrowW * 0.22
            lfY = dnY
            rtX = arrowX0 + (arrowW + arrowGap) * 2 + arrowW * 0.22
            rtY = dnY
            ax.text(upX, upY, "^", color="#FFFFFF", fontsize=4.9, alpha=0.98, zorder=5,
                    bbox={"boxstyle": "round,pad=0.04", "facecolor": "#000000", "edgecolor": "none", "alpha": 0.16})
            ax.text(dnX, dnY, "v", color="#FFFFFF", fontsize=4.9, alpha=0.98, zorder=5,
                    bbox={"boxstyle": "round,pad=0.04", "facecolor": "#000000", "edgecolor": "none", "alpha": 0.16})
            ax.text(lfX, lfY, "<", color="#FFFFFF", fontsize=4.9, alpha=0.98, zorder=5,
                    bbox={"boxstyle": "round,pad=0.04", "facecolor": "#000000", "edgecolor": "none", "alpha": 0.16})
            ax.text(rtX, rtY, ">", color="#FFFFFF", fontsize=4.9, alpha=0.98, zorder=5,
                    bbox={"boxstyle": "round,pad=0.04", "facecolor": "#000000", "edgecolor": "none", "alpha": 0.16})
            # Touchpad and palm rest zones.
            padW = int(w * 0.30)
            padH = int(h * 0.18)
            padX0 = int((w - padW) * 0.5)
            padY0 = min(int(h * 0.77) + 2, max(0, deckY1 - padH + 1))
            ax.add_patch(Rectangle(
                (padX0, padY0), max(1, padW), max(1, padH),
                facecolor="none", edgecolor="#E9ECF2",
                linewidth=1.0, alpha=0.42, zorder=3,
            ))
        except Exception:
            # Fallback text prevents UI breakage if guide rendering fails.
            ax.text(3, 10, "C-Deck guide unavailable", color="#E2E8F0", fontsize=7, alpha=0.8, zorder=4)

    def _draw_hotspot_constraints(self, ax, image_shape: tuple[int, int]) -> None:
        showInteractiveArea = self._interactionMode in ("skip", "target")
        if showInteractiveArea:
            h, w = int(image_shape[0]), int(image_shape[1])
            deckX0, deckY0 = -0.5, -0.5
            deckW, deckH = float(w), float(h)
            # Highlight the drawable area before user clicks.
            ax.add_patch(Rectangle((deckX0, deckY0), deckW, deckH,
                                   fill=False, edgecolor="#38BDF8", linewidth=1.4, linestyle=":", zorder=7))
            ax.text(2, 10, "Drawable Area", color="#7DD3FC", fontsize=7, zorder=8)

        if self.skipAreaRect is not None:
            x0, y0, x1, y1 = self.skipAreaRect
            w = max(1, x1 - x0)
            h = max(1, y1 - y0)
            ax.add_patch(Rectangle((x0, y0), w, h, fill=False,
                                   edgecolor="#F97316", linewidth=1.6, linestyle="--", zorder=7))
            ax.text(x0 + 2, max(2, y0 - 4), "Skip Area", color="#FDBA74", fontsize=7, zorder=8)

        if self._interactionMode == "skip" and self._skipPreviewRect is not None:
            x0, y0, x1, y1 = self._skipPreviewRect
            w = max(1, x1 - x0)
            h = max(1, y1 - y0)
            ax.add_patch(Rectangle((x0, y0), w, h, fill=False,
                                   edgecolor="#FB923C", linewidth=1.2, linestyle="-", zorder=8))

        if self.targetAreaRect is not None:
            x0, y0, x1, y1 = self.targetAreaRect
            w = max(1, x1 - x0)
            h = max(1, y1 - y0)
            ax.add_patch(Rectangle((x0, y0), w, h, fill=False,
                                   edgecolor="#22C55E", linewidth=1.6, linestyle="-.", zorder=7))
            ax.text(x0 + 2, max(2, y0 - 4), "Target Area", color="#86EFAC", fontsize=7, zorder=8)

        if self._interactionMode == "target" and self._targetPreviewRect is not None:
            x0, y0, x1, y1 = self._targetPreviewRect
            w = max(1, x1 - x0)
            h = max(1, y1 - y0)
            ax.add_patch(Rectangle((x0, y0), w, h, fill=False,
                                   edgecolor="#4ADE80", linewidth=1.2, linestyle="-", zorder=8))

    def start_skip_area_selection(self) -> None:
        if self.current_frame is None:
            self.generate_new_frame()
        self._interactionMode = "skip"
        self._skipStart = None
        self._skipPreviewRect = None
        self._targetStart = None
        self._targetPreviewRect = None
        self._cursorPoint = None
        self._set_thermal_cursor("arrow")
        self._draw("thermal", self.current_frame.image, cmap="inferno",
                   centers=self.current_frame.centers, subtitle="raw input")
        self._set_status("Select skip area: click top-left then bottom-right", busy=True)

    def start_target_area_selection(self) -> None:
        if self.current_frame is None:
            self.generate_new_frame()
        self._interactionMode = "target"
        self._skipStart = None
        self._skipPreviewRect = None
        self._targetStart = None
        self._targetPreviewRect = None
        self._cursorPoint = None
        self._set_thermal_cursor("arrow")
        self._draw("thermal", self.current_frame.image, cmap="inferno",
                   centers=self.current_frame.centers, subtitle="raw input")
        self._set_status("Select target area: click top-left then bottom-right", busy=True)

    def clear_skip_area(self) -> None:
        self.skipAreaRect = None
        self._skipStart = None
        self._skipPreviewRect = None
        self._targetStart = None
        self._targetPreviewRect = None
        self._interactionMode = None
        self._set_thermal_cursor("arrow")
        self._refresh_thermal_preview()
        self._set_status("Skip area cleared")

    def clear_target_area(self) -> None:
        self.targetAreaRect = None
        self._targetStart = None
        self._targetPreviewRect = None
        self._skipStart = None
        self._skipPreviewRect = None
        self._interactionMode = None
        self._cursorPoint = None
        self._set_thermal_cursor("arrow")
        self._refresh_thermal_preview()
        self._set_status("Target area cleared")

    def _refresh_thermal_preview(self) -> None:
        if self.current_frame is None:
            return
        self._draw("thermal", self.current_frame.image, cmap="inferno",
                   centers=self.current_frame.centers, subtitle="raw input")

    def _set_thermal_cursor(self, cursorName: str = "arrow") -> None:
        """Set cursor style for thermal image canvas."""
        cv = self._img_cvs.get("thermal")
        if cv is None:
            return
        try:
            cv.get_tk_widget().configure(cursor=cursorName)
        except Exception:
            # Keep UI stable if cursor style is unsupported.
            pass

    def _update_skip_cursor(self, x: Optional[float], y: Optional[float], inAxes: bool) -> None:
        """Show crosshair while setting area-based interactions inside drawable region."""
        if self._interactionMode not in ("skip", "target") or not inAxes or x is None or y is None:
            self._set_thermal_cursor("arrow")
            return
        deckX0, deckY0, deckX1, deckY1 = self._get_c_deck_bounds(
            self.generator.width, self.generator.height)
        inside = deckX0 <= float(x) <= deckX1 and deckY0 <= float(y) <= deckY1
        self._set_thermal_cursor("crosshair" if inside else "arrow")

    def _handle_thermal_motion(self, x: float, y: float) -> None:
        if self._interactionMode not in ("skip", "target"):
            return
        deckX0, deckY0, deckX1, deckY1 = self._get_c_deck_bounds(
            self.generator.width, self.generator.height)
        xi = int(np.clip(round(float(x)), deckX0, deckX1))
        yi = int(np.clip(round(float(y)), deckY0, deckY1))
        self._cursorPoint = (xi, yi)

        if self._interactionMode == "skip" and self._skipStart is not None:
            x0, y0 = self._skipStart
            left = min(x0, xi)
            right = max(x0, xi)
            top = min(y0, yi)
            bottom = max(y0, yi)
            self._skipPreviewRect = (left, top, right, bottom)

        if self._interactionMode == "target" and self._targetStart is not None:
            x0, y0 = self._targetStart
            left = min(x0, xi)
            right = max(x0, xi)
            top = min(y0, yi)
            bottom = max(y0, yi)
            self._targetPreviewRect = (left, top, right, bottom)

        self._refresh_thermal_preview()

    def _handle_thermal_click(self, x: float, y: float) -> None:
        if self._interactionMode not in ("skip", "target"):
            return
        deckX0, deckY0, deckX1, deckY1 = self._get_c_deck_bounds(
            self.generator.width, self.generator.height)
        xi = int(np.clip(round(float(x)), deckX0, deckX1))
        yi = int(np.clip(round(float(y)), deckY0, deckY1))

        if self._interactionMode == "target":
            if self._targetStart is None:
                self._targetStart = (xi, yi)
                self._targetPreviewRect = (xi, yi, xi, yi)
                self._cursorPoint = (xi, yi)
                self._refresh_thermal_preview()
                self._set_status("Select target area: now click bottom-right", busy=True)
                return

            x0, y0 = self._targetStart
            left = min(x0, xi)
            right = max(x0, xi)
            top = min(y0, yi)
            bottom = max(y0, yi)
            self.targetAreaRect = (left, top, right, bottom)
            self._targetStart = None
            self._targetPreviewRect = None
            self._interactionMode = None
            self._cursorPoint = None
            self._set_thermal_cursor("arrow")
            self._refresh_thermal_preview()
            self._set_status("Target area set")
            return

        if self._skipStart is None:
            self._skipStart = (xi, yi)
            self._skipPreviewRect = (xi, yi, xi, yi)
            self._cursorPoint = (xi, yi)
            self._refresh_thermal_preview()
            self._set_status("Select skip area: now click bottom-right", busy=True)
            return

        x0, y0 = self._skipStart
        left = min(x0, xi)
        right = max(x0, xi)
        top = min(y0, yi)
        bottom = max(y0, yi)
        self.skipAreaRect = (left, top, right, bottom)
        self._skipStart = None
        self._skipPreviewRect = None
        self._interactionMode = None
        self._cursorPoint = None
        self._set_thermal_cursor("arrow")
        self._refresh_thermal_preview()
        self._set_status("Skip area set")

    def _build_generation_constraints(self) -> dict:
        constraints: dict = {}
        deckX0, deckY0, deckX1, deckY1 = self._get_c_deck_bounds(
            self.generator.width, self.generator.height)
        if self.targetAreaRect is not None:
            x0, y0, x1, y1 = self.targetAreaRect
            left = int(np.clip(min(x0, x1), deckX0, deckX1))
            right = int(np.clip(max(x0, x1), deckX0, deckX1))
            top = int(np.clip(min(y0, y1), deckY0, deckY1))
            bottom = int(np.clip(max(y0, y1), deckY0, deckY1))
            constraints["target_area"] = (left, top, right, bottom)
        return constraints

    def _build_skip_area_for_detection(self) -> Optional[tuple[int, int, int, int]]:
        if self.skipAreaRect is None:
            return None
        deckX0, deckY0, deckX1, deckY1 = self._get_c_deck_bounds(
            self.generator.width, self.generator.height)
        x0, y0, x1, y1 = self.skipAreaRect
        left = int(np.clip(min(x0, x1), deckX0, deckX1))
        right = int(np.clip(max(x0, x1), deckX0, deckX1))
        top = int(np.clip(min(y0, y1), deckY0, deckY1))
        bottom = int(np.clip(max(y0, y1), deckY0, deckY1))
        return left, top, right, bottom

    def _apply_skip_mask_for_detection(self, image: np.ndarray) -> np.ndarray:
        skip_rect = self._build_skip_area_for_detection()
        if skip_rect is None:
            return image

        left, top, right, bottom = skip_rect
        masked = image.copy()
        fill_temp = float(np.percentile(masked, 5.0))
        masked[top:bottom + 1, left:right + 1] = fill_temp
        return masked

    def _clear_card(self, key: str) -> None:
        fig = self._img_figs[key]
        cv  = self._img_cvs[key]
        if key == "openvino":
            self._set_openvino_card_title()
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor("#161929")
        ax.text(0.5, 0.5, "Single Run", ha="center", va="center",
                color=C["dim"], fontsize=9, transform=ax.transAxes)
        ax.axis("off")
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
        cv.draw()
        self._sub_vars[key].set("—")

    # ═══════════════════════════════════════════════════════════════════════
    #  Core actions
    # ═══════════════════════════════════════════════════════════════════════

    def generate_new_frame(self) -> None:
        self.generator.noise_std = float(self.noise_scale.get())
        hotspot_count = int(self.hotspot_count.get())
        constraints = self._build_generation_constraints()
        self.current_frame = self.generator.generate(
            hotspot_count=hotspot_count, shape=HotspotShape.CIRCULAR, **constraints)
        self._draw("thermal", self.current_frame.image, cmap="inferno",
                    centers=self.current_frame.centers, subtitle="raw input")
        self._draw("mask", self.current_frame.mask, cmap="gray",
                    subtitle="ground truth")
        for k in ("opencv", "pytorch", "openvino"):
            self._clear_card(k)
            self._kpi[f"{k}_lat"].set("—")
            self._kpi[f"{k}_fps"].set("—")
            self._kpi[f"{k}_acc"].set("—")
            self._kpi[f"{k}_conf"].set("—")
            self._robot_xyz_vars[k].set("X —  Y —  Z —")
        if not self._vis_frame.winfo_ismapped():
            self._vis_frame.grid()
        self._set_status("Frame generated")

    def _initialize_default_models(self) -> None:
        def _load() -> None:
            self.root.after(0, lambda: self._set_status(
                "Loading models…", busy=True))
            try:
                self.yolo_pytorch_detector = YOLOv8PyTorchDetector(
                    model_name="thermal/yolov8n.pt",
                    device="cpu",
                )
            except Exception as e:
                print(f"PyTorch load failed: {e}")
            self.load_openvino_model(
                model_path_raw="thermal/yolov8n_openvino_model/yolov8n.xml",
                show_success=False, show_errors=False)
            self.root.after(0, lambda: self._set_status("Ready"))
        threading.Thread(target=_load, daemon=True).start()

    def load_openvino_model(
        self,
        model_path_raw: Optional[str] = None,
        show_success: bool = True,
        show_errors: bool = True,
    ) -> None:
        from pathlib import Path
        try:
            if model_path_raw is None:
                model_path_raw = "thermal/yolov8n_openvino_model/yolov8n.xml"
            inp = Path(model_path_raw.strip())
            proj = Path(__file__).resolve().parent
            candidates = [inp]
            if not inp.is_absolute():
                candidates = [Path.cwd() / inp, proj / inp]
            resolved = next((p for p in candidates if p.exists()), None)
            if resolved is None:
                for fb in [
                    Path.cwd() / "thermal" / "yolov8n_openvino_model" / "yolov8n.xml",
                    proj / "yolov8n_openvino_model" / "yolov8n.xml",
                    Path.cwd() / "yolov8n_openvino_model" / "yolov8n.xml",
                    Path.cwd() / "models" / "yolov8n.xml",
                    proj / "models" / "yolov8n.xml",
                ]:
                    if fb.exists():
                        resolved = fb
                        break
            if resolved is None:
                if show_errors:
                    messagebox.showerror("Error", "OpenVINO model not found.")
                return
            selectedDevice = self._normalize_openvino_device(self.openvinoDeviceVar.get())
            self.yolo_openvino_detector = OpenVINOYOLODetector(
                model_path=str(resolved.resolve()), device=selectedDevice)
            if show_success:
                messagebox.showinfo("Success",
                                    f"OpenVINO model loaded on {selectedDevice}!\n{resolved}")
        except Exception as e:
            if show_errors:
                messagebox.showerror("Error", str(e))

    def run_detection(self) -> None:
        # Always refresh input before detection so one click completes the full flow.
        self.generate_new_frame()
        self._set_status("Single Run in progress…", busy=True)
        if not self._vis_frame.winfo_ismapped():
            self._vis_frame.grid()
        threading.Thread(target=self._run_detections_threaded,
                          daemon=True).start()

    def _run_detections_threaded(self) -> None:
        try:
            results = {}
            detection_image = self._apply_skip_mask_for_detection(self.current_frame.image)
            results["opencv"] = self.opencv_detector.detect(
                detection_image)
            if self.yolo_pytorch_detector is None:
                try:
                    self.yolo_pytorch_detector = YOLOv8PyTorchDetector(
                        model_name="thermal/yolov8n.pt",
                        device="cpu")
                except Exception as e:
                    print(f"PyTorch N/A: {e}")
            if self.yolo_pytorch_detector:
                results["pytorch"] = self.yolo_pytorch_detector.detect(
                    detection_image)
            if self.yolo_openvino_detector is None:
                # Ensure first detection can include OpenVINO even if async startup load is not ready yet.
                self.load_openvino_model(show_success=False, show_errors=False)
            if self.yolo_openvino_detector:
                results["openvino"] = self.yolo_openvino_detector.detect(
                    detection_image)
            self.root.after(0, lambda: self._display_results(results))
        except Exception as e:
            errorMessage = str(e)
            self.root.after(0,
                lambda: messagebox.showerror("Detection Error", errorMessage))

    def run_benchmark(self) -> None:
        n = int(self.benchmark_samples.get())
        if n <= 0:
            messagebox.showwarning("Warning", "Sample count must be > 0")
            return
        self._set_status(f"Batch Run in progress… ({n} samples)", busy=True)
        threading.Thread(target=self._run_benchmark_threaded,
                          args=(n,), daemon=True).start()

    def _run_benchmark_threaded(self, sample_count: int) -> None:
        try:
            if self.yolo_pytorch_detector is None:
                try:
                    self.yolo_pytorch_detector = YOLOv8PyTorchDetector(
                        model_name="thermal/yolov8n.pt",
                        device="cpu")
                except Exception as e:
                    print(f"PyTorch benchmark init failed: {e}")
            if self.yolo_openvino_detector is None:
                self.load_openvino_model(show_success=False, show_errors=False)

            detectors = {
                "opencv":   self.opencv_detector,
                "pytorch":  self.yolo_pytorch_detector,
                "openvino": self.yolo_openvino_detector,
            }
            active_detectors = {k: v for k, v in detectors.items()
                                 if v is not None}
            if not active_detectors:
                self.root.after(0, lambda: messagebox.showerror(
                    "Benchmark Error", "No detector available"))
                return

            stats = {k: {"error": [], "latency": [], "fps": [], "confidence": []}
                     for k in active_detectors}
            gen = ThermalImageGenerator(
                width=self.generator.width,
                height=self.generator.height,
                noise_std=float(self.noise_scale.get()),
                seed=20260630)
            hotspot_count = int(self.hotspot_count.get())
            constraints = self._build_generation_constraints()

            warm_frame = gen.generate(hotspot_count=hotspot_count,
                                       shape=HotspotShape.CIRCULAR,
                                       **constraints)
            warm_image = self._apply_skip_mask_for_detection(warm_frame.image)
            for detector in active_detectors.values():
                try:
                    detector.detect(warm_image)
                except Exception:
                    pass

            for i in range(sample_count):
                frame = gen.generate(hotspot_count=hotspot_count,
                                     shape=HotspotShape.CIRCULAR,
                                     **constraints)
                gt_x, gt_y = frame.centers[0]
                detection_image = self._apply_skip_mask_for_detection(frame.image)
                for key, detector in active_detectors.items():
                    result = detector.detect(detection_image)
                    error = MetricsCalculator.localization_error(
                        result.center_x, result.center_y, gt_x, gt_y)
                    latency = float(result.inference_time_ms)
                    fps = 1000.0 / latency if latency > 0 else 0.0
                    stats[key]["error"].append(error)
                    stats[key]["latency"].append(latency)
                    stats[key]["fps"].append(fps)
                    stats[key]["confidence"].append(float(result.confidence))
                if (i + 1) % 10 == 0:
                    pct = (i + 1) / sample_count * 100
                    self.root.after(0, lambda p=pct: self._set_status(
                        f"Batch Run in progress… {p:.0f}%", busy=True))

            tagged = self._format_batch_benchmark_tagged_summary(stats, sample_count)
            self.root.after(0, lambda: self._apply_tagged_summary(tagged))
            self.root.after(0, lambda: self._set_status("Batch Run complete"))
        except Exception as e:
            errorMessage = str(e)
            self.root.after(0, lambda: messagebox.showerror(
                "Benchmark Error", errorMessage))
            self.root.after(0, lambda: self._set_status("Error", busy=True))

    def _format_batch_benchmark_tagged_summary(self, stats: dict,
                                               sample_count: int) -> list[tuple[str, str]]:
        summaryByModel: dict[str, dict[str, float | str]] = {}
        for key in ("opencv", "pytorch", "openvino"):
            if key not in stats:
                continue
            err = np.asarray(stats[key]["error"], dtype=np.float32)
            lat = np.asarray(stats[key]["latency"], dtype=np.float32)
            fps = np.asarray(stats[key]["fps"], dtype=np.float32)
            confidencePct = np.asarray(stats[key]["confidence"], dtype=np.float32) * 100.0
            summaryByModel[key] = {
                "errorMean": float(np.mean(err)),
                "errorMedian": float(np.median(err)),
                "errorP95": float(np.percentile(err, 95)),
                "latencyMean": float(np.mean(lat)),
                "latencyMedian": float(np.median(lat)),
                "latencyP95": float(np.percentile(lat, 95)),
                "fpsMean": float(np.mean(fps)),
                "fpsMedian": float(np.median(fps)),
                "fpsP95": float(np.percentile(fps, 95)),
                "confidenceMean": float(np.mean(confidencePct)),
                "confidenceMedian": float(np.median(confidencePct)),
                "confidenceP95": float(np.percentile(confidencePct, 95)),
            }
            if key == "openvino" and self.yolo_openvino_detector is not None:
                summaryByModel[key]["executionDeviceText"] = self.yolo_openvino_detector.get_execution_device_text()  # type: ignore[index]

        bestError = min((v["errorMean"] for v in summaryByModel.values()), default=float("inf"))
        metricLabelWidth = 16
        metricValueWidth = 8
        metricUnitWidth = 3

        tagged: list[tuple[str, str]] = [
            ("header", "BATCH RUN SUMMARY\n\n"),
            ("muted", f"  Samples  : {sample_count}\n"),
            ("muted", "  Stream   : shared (seed=20260630)\n"),
            ("muted", "  GT       : frame.centers[0]\n\n"),
        ]

        for key in ("opencv", "pytorch", "openvino"):
            if key not in summaryByModel:
                continue
            s = summaryByModel[key]
            errorMean = float(s["errorMean"])
            isBest = abs(errorMean - bestError) < 0.01
            tagged += [
                ("key", f"{MODEL_DISPLAY_NAMES[key].upper()}\n"),
            ]
            if key == "openvino" and "executionDeviceText" in s:
                tagged += [
                    ("muted", f"    {'Execution Device':<{metricLabelWidth}}"),
                    ("val", f"{s['executionDeviceText']:>{metricValueWidth}} {'':<{metricUnitWidth}}\n"),
                ]
            tagged += [
                ("muted", "  Position Error:\n"),
                ("good" if isBest else "warn", f"    {'mean':<{metricLabelWidth}}{s['errorMean']:>{metricValueWidth}.2f} {'px':<{metricUnitWidth}}\n"),
                ("good" if isBest else "warn", f"    {'Median (P50)':<{metricLabelWidth}}{s['errorMedian']:>{metricValueWidth}.2f} {'px':<{metricUnitWidth}}\n"),
                ("good" if isBest else "warn", f"    {'P95':<{metricLabelWidth}}{s['errorP95']:>{metricValueWidth}.2f} {'px':<{metricUnitWidth}}\n"),
                ("muted", "  Processing Time:\n"),
                ("val", f"    {'mean':<{metricLabelWidth}}{s['latencyMean']:>{metricValueWidth}.2f} {'ms':<{metricUnitWidth}}\n"),
                ("val", f"    {'Median (P50)':<{metricLabelWidth}}{s['latencyMedian']:>{metricValueWidth}.2f} {'ms':<{metricUnitWidth}}\n"),
                ("val", f"    {'P95':<{metricLabelWidth}}{s['latencyP95']:>{metricValueWidth}.2f} {'ms':<{metricUnitWidth}}\n"),
                ("muted", "  Frame Rate:\n"),
                ("val", f"    {'mean':<{metricLabelWidth}}{s['fpsMean']:>{metricValueWidth}.1f} {'fps':<{metricUnitWidth}}\n"),
                ("val", f"    {'Median (P50)':<{metricLabelWidth}}{s['fpsMedian']:>{metricValueWidth}.1f} {'fps':<{metricUnitWidth}}\n"),
                ("val", f"    {'P95':<{metricLabelWidth}}{s['fpsP95']:>{metricValueWidth}.1f} {'fps':<{metricUnitWidth}}\n"),
                ("muted", "  Detection Confidence:\n"),
                ("val", f"    {'mean':<{metricLabelWidth}}{s['confidenceMean']:>{metricValueWidth}.1f} {'%':<{metricUnitWidth}}\n"),
                ("val", f"    {'Median (P50)':<{metricLabelWidth}}{s['confidenceMedian']:>{metricValueWidth}.1f} {'%':<{metricUnitWidth}}\n"),
                ("val", f"    {'P95':<{metricLabelWidth}}{s['confidenceP95']:>{metricValueWidth}.1f} {'%':<{metricUnitWidth}}\n\n"),
            ]
        return tagged

    def _apply_tagged_summary(self, tagged: list[tuple[str, str]]) -> None:
        self._vis_frame.grid_remove()
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)
        for tag, text in tagged:
            self.metrics_text.insert(tk.END, text, tag)
        self.metrics_text.config(state=tk.DISABLED)

    def _display_results(self, results: dict) -> None:
        if not self._vis_frame.winfo_ismapped():
            self._vis_frame.grid()
        gt_x, gt_y = self.current_frame.centers[0]
        self._draw("thermal", self.current_frame.image, cmap="inferno",
                    centers=self.current_frame.centers, subtitle="raw input")
        self._draw("mask", self.current_frame.mask, cmap="gray",
                    subtitle="ground truth")
        best_err = min(
            (MetricsCalculator.localization_error(
                r.center_x, r.center_y, gt_x, gt_y)
             for r in results.values()),
            default=float("inf"),
        )
        best_latency = min(
            (float(r.inference_time_ms) for r in results.values()),
            default=float("inf"),
        )
        tagged: list[tuple[str, str]] = [
            ("header", "SINGLE RUN SUMMARY\n\n")]
        for key in ("opencv", "pytorch", "openvino"):
            if key not in results:
                self._clear_card(key)
                continue
            r = results[key]
            err = MetricsCalculator.localization_error(
                r.center_x, r.center_y, gt_x, gt_y)
            fps = 1000.0 / r.inference_time_ms \
                if r.inference_time_ms > 0 else 0.0
            executionDeviceText = ""
            if key == "openvino":
                executionDevices = getattr(r, "execution_devices", ())
                if executionDevices:
                    executionDeviceText = ", ".join(str(device) for device in executionDevices)
                elif self.yolo_openvino_detector is not None:
                    executionDeviceText = self.yolo_openvino_detector.get_execution_device_text()
                self._set_openvino_card_title(executionDeviceText)
            self._draw(key, self.current_frame.image, cmap="inferno",
                        result=r,
                        subtitle=(
                            f"{r.inference_time_ms:.1f}ms  "
                            f"{err:.1f}px  detection confidence {r.confidence * 100:.1f}%"
                        ))
            robot = self.robot_mapper.pixel_to_robot(r.center_x, r.center_y)
            self._robot_xyz_vars[key].set(
                f"X {robot.X:.3f}  Y {robot.Y:.3f}  Z {robot.Z:.3f}"
            )
            is_best = abs(err - best_err) < 0.01
            is_fastest = abs(float(r.inference_time_ms) - best_latency) < 0.01
            tagged += [
                ("key",   f"{MODEL_DISPLAY_NAMES[key].upper()}\n"),
            ]
            if key == "openvino" and executionDeviceText:
                tagged += [
                    ("muted", "  Execution Device:      "),
                    ("val",   f"{executionDeviceText}\n"),
                ]
            tagged += [
                ("muted", "  Position Error:        "),
                ("good" if is_best else "warn", f"{err:.2f} px\n"),
                ("muted", "  Processing Time:       "),
                ("good" if is_fastest else "warn", f"{r.inference_time_ms:.2f} ms\n"),
                ("muted", "  Frame Rate:            "),
                ("val",   f"{fps:.0f} fps\n"),
                ("muted", "  Detection Confidence:  "),
                ("val",   f"{r.confidence * 100:.1f}%\n"),
                ("muted", "  Robot Target Position:\n"),
                ("muted", "    X: "), ("val", f"{robot.X:.3f}\n"),
                ("muted", "    Y: "), ("val", f"{robot.Y:.3f}\n"),
                ("muted", "    Z: "), ("val", f"{robot.Z:.3f}\n\n"),
            ]
        self._update_kpi(results, gt_x, gt_y)
        self._set_status("Single Run complete")
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)
        for tag, text in tagged:
            self.metrics_text.insert(tk.END, text, tag)
        self.metrics_text.config(state=tk.DISABLED)

    def clear_history(self) -> None:
        self.metrics_history = {"opencv": [], "pytorch": [], "openvino": []}


def main() -> None:
    root = tk.Tk()
    app = ThermalHotspotDemo(root)

    def _on_click(event) -> None:
        app._update_skip_cursor(event.xdata, event.ydata, event.inaxes is not None)
        if event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        app._handle_thermal_click(event.xdata, event.ydata)

    def _on_move(event) -> None:
        app._update_skip_cursor(event.xdata, event.ydata, event.inaxes is not None)
        if event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        app._handle_thermal_motion(event.xdata, event.ydata)

    def _on_leave(_event) -> None:
        app._set_thermal_cursor("arrow")

    # Bind pick behavior to thermal card once available.
    def _bind_click_later() -> None:
        fig = app._img_figs.get("thermal")
        if fig is not None:
            fig.canvas.mpl_connect("button_press_event", _on_click)
            fig.canvas.mpl_connect("motion_notify_event", _on_move)
            fig.canvas.mpl_connect("axes_leave_event", _on_leave)

    root.after(0, _bind_click_later)
    root.mainloop()


if __name__ == "__main__":
    main()
