from __future__ import annotations

from datetime import datetime
import tkinter as tk
from tkinter import messagebox

from robot.action_layer import PyBulletActionExecutor
from shared.interfaces import ActionCommand, ActionType, RobotTarget, ActionStatus, ActionResult

try:
    import pybullet as p
except Exception:
    p = None


C: dict[str, str] = {
    "bg": "#0F1117",
    "surface": "#1A1D2B",
    "card": "#20243A",
    "border": "#2A2F4A",
    "accent": "#6366F1",
    "accent_dim": "#4F46E5",
    "success": "#10B981",
    "error": "#EF4444",
    "text": "#F1F5F9",
    "muted": "#94A3B8",
    "dim": "#475569",
    "header": "#161929",
}
FF = "Segoe UI"

X_RANGE = (0.30, 0.65)
Y_RANGE = (-0.25, 0.25)
Z_RANGE = (0.10, 0.28)

BALL_SPECS = {
    "red": {"pos": (0.42, -0.08, 0.14), "rgba": (0.92, 0.20, 0.20, 1.0)},
    "yellow": {"pos": (0.45, 0.00, 0.14), "rgba": (0.95, 0.82, 0.20, 1.0)},
    "green": {"pos": (0.48, 0.08, 0.14), "rgba": (0.20, 0.78, 0.34, 1.0)},
    "orange": {"pos": (0.51, -0.02, 0.14), "rgba": (0.95, 0.52, 0.16, 1.0)},
}

BOX_CENTER = (0.58, 0.00, 0.14)
BOX_DROP_POINT = (0.58, 0.00, 0.14)
HOME_POSITION = (0.45, 0.00, 0.40)


class RobotPickPlaceDemoUI:
    """Step-by-step demo: pick selected ball and place it into a box."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Franka Panda Pick-Place Demo")
        self.root.configure(bg=C["bg"])
        self.root.minsize(640, 420)

        self.executor = PyBulletActionExecutor()
        self.ball_ids: dict[str, int] = {}
        self.box_body_ids: list[int] = []
        self.table_body_ids: list[int] = []
        self.payload_constraint_id: int | None = None
        self.active_ball_color: str | None = None
        self.active_ball_id: int | None = None
        self.ball_option_to_color: dict[str, str] = {}
        self.log_step: int = 0
        self.status_lines: list[str] = []
        self.flow_stage: str = "idle"
        self.is_executing: bool = False

        self._build_ui()

    def _build_ui(self) -> None:
        outer = tk.Frame(self.root, bg=C["bg"], padx=18, pady=16)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        hdr = tk.Frame(outer, bg=C["header"], height=58)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        tk.Label(
            hdr,
            text="Robot Demo  /  Franka Panda Pick-Place",
            bg=C["header"],
            fg=C["text"],
            font=(FF, 12, "bold"),
        ).pack(side=tk.LEFT, padx=16, pady=14)

        body = tk.Frame(outer, bg=C["surface"], highlightbackground=C["border"], highlightthickness=1)
        body.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        body.columnconfigure(0, weight=1)

        card = tk.Frame(body, bg=C["card"], highlightbackground=C["border"], highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)
        card.columnconfigure(1, weight=1)

        tk.Label(card, text="Pick Ball Color", bg=C["card"], fg=C["muted"], font=(FF, 10)).grid(
            row=0, column=0, sticky="w", padx=12, pady=(12, 6)
        )
        tk.Label(card, text="Drop Box Center", bg=C["card"], fg=C["muted"], font=(FF, 10)).grid(
            row=1, column=0, sticky="w", padx=12, pady=6
        )
        tk.Label(card, text="Ball Source Area", bg=C["card"], fg=C["muted"], font=(FF, 10)).grid(
            row=2, column=0, sticky="w", padx=12, pady=6
        )

        option_labels: list[str] = []
        self.ball_option_to_color = {}
        for color, spec in BALL_SPECS.items():
            label = self._ball_display_label(color, spec["pos"])
            option_labels.append(label)
            self.ball_option_to_color[label] = color

        self.ball_color_var = tk.StringVar(value=option_labels[0])

        card.columnconfigure(1, weight=1)

        color_menu = tk.OptionMenu(card, self.ball_color_var, *option_labels)
        color_menu.config(
            bg="#161A2C",
            fg=C["text"],
            activebackground=C["accent_dim"],
            activeforeground="white",
            relief="flat",
            highlightthickness=1,
            highlightbackground=C["border"],
            font=(FF, 10),
        )
        color_menu["menu"].config(
            bg="#161A2C",
            fg=C["text"],
            activebackground=C["accent_dim"],
            activeforeground="white",
            font=(FF, 10),
        )
        color_menu.grid(row=0, column=1, padx=(8, 12), pady=(12, 6), sticky="ew")

        tk.Label(
            card,
            text=f"({BOX_CENTER[0]:.2f}, {BOX_CENTER[1]:.2f}, {BOX_CENTER[2]:.2f}) m",
            bg=C["card"],
            fg=C["text"],
            font=(FF, 10),
            anchor="w",
        ).grid(row=1, column=1, padx=(8, 12), pady=6, sticky="ew")

        tk.Label(
            card,
            text=", ".join(self._ball_display_label(color, spec["pos"]) for color, spec in BALL_SPECS.items()),
            bg=C["card"],
            fg=C["text"],
            font=(FF, 10),
            anchor="w",
            justify=tk.LEFT,
            wraplength=520,
        ).grid(row=2, column=1, padx=(8, 12), pady=6, sticky="ew")

        self.viewer_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            card,
            text="Open PyBullet Viewer window",
            variable=self.viewer_var,
            bg=C["card"],
            fg=C["muted"],
            activebackground=C["card"],
            activeforeground=C["text"],
            selectcolor="#161A2C",
            font=(FF, 9),
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=12, pady=(2, 2))

        tk.Label(
            card,
            text=(
                f"Allowed range  X: {X_RANGE[0]:.2f}..{X_RANGE[1]:.2f} m, "
                f"Y: {Y_RANGE[0]:.2f}..{Y_RANGE[1]:.2f} m, "
                f"Z: {Z_RANGE[0]:.2f}..{Z_RANGE[1]:.2f} m"
            ),
            bg=C["card"],
            fg=C["dim"],
            font=(FF, 9),
            anchor="w",
            justify=tk.LEFT,
        ).grid(row=4, column=0, columnspan=2, sticky="ew", padx=12, pady=(2, 8))

        button_row = tk.Frame(card, bg=C["card"])
        button_row.grid(row=5, column=0, columnspan=2, sticky="ew", padx=12, pady=(12, 8))
        for i in range(3):
            button_row.columnconfigure(i, weight=1)

        self.btn_move_src = self._btn(button_row, "1) Move Above Source", self.move_above_source)
        self.btn_pick = self._btn(button_row, "2) Pick", self.pick_from_source)
        self.btn_move_tgt = self._btn(button_row, "3) Move Above Box", self.move_above_target)
        self.btn_place = self._btn(button_row, "4) Place In Box", self.place_to_target)
        self.btn_home = self._btn(button_row, "5) Go Home", self.go_home)

        self.btn_move_src.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=(0, 6))
        self.btn_pick.grid(row=0, column=1, sticky="ew", padx=6, pady=(0, 6))
        self.btn_move_tgt.grid(row=0, column=2, sticky="ew", padx=(6, 0), pady=(0, 6))
        self.btn_place.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 6))
        self.btn_home.grid(row=1, column=2, sticky="ew", padx=(6, 0))

        self.status_var = tk.StringVar(value="Initializing simulation...")
        status_card = tk.Frame(card, bg="#1B1F33", highlightbackground=C["border"], highlightthickness=1)
        status_card.grid(row=6, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 12))
        tk.Label(status_card, text="Status", bg="#1B1F33", fg=C["dim"], font=(FF, 8, "bold")).pack(anchor="w", padx=10, pady=(8, 2))
        tk.Label(
            status_card,
            textvariable=self.status_var,
            bg="#1B1F33",
            fg=C["text"],
            anchor="w",
            justify=tk.LEFT,
            font=(FF, 10),
            wraplength=560,
        ).pack(fill=tk.X, padx=10, pady=(0, 8))

        self._refresh_buttons_by_stage()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self.start_simulation)

    @staticmethod
    def _fmt_xyz(xyz: tuple[float, float, float]) -> str:
        return f"({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})"

    @staticmethod
    def _ball_display_label(color: str, xyz: tuple[float, float, float]) -> str:
        return f"{color} ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f})"

    def _set_demo_camera(self) -> None:
        if p is None:
            return
        if not self.executor.config.gui:
            return
        p.resetDebugVisualizerCamera(
            cameraDistance=1.25,
            cameraYaw=180.0,
            cameraPitch=-18.0,
            cameraTargetPosition=[0.50, 0.00, 0.14],
        )

    def _log(self, message: str) -> None:
        self.log_step += 1
        ts = datetime.now().strftime("%H:%M:%S")
        sep = "=" * 84
        line = f"[{ts}] [STEP {self.log_step:03d}] {message}"
        print(f"\n{sep}\n{line}\n{sep}", flush=True)

    def _log_lines(self, lines: list[str]) -> None:
        block = "\n".join(lines)
        print(block, flush=True)

    def _log_result(self, action_name: str, target_xyz: tuple[float, float, float], result: ActionResult) -> None:
        achieved = (
            "None"
            if result.achieved_x is None or result.achieved_y is None or result.achieved_z is None
            else self._fmt_xyz((result.achieved_x, result.achieved_y, result.achieved_z))
        )
        err = "None" if result.error_mm is None else f"{result.error_mm:.2f}"
        self._log(f"{action_name} Result")
        self._log_lines([
            f"  target_xyz : {self._fmt_xyz(target_xyz)}",
            f"  status     : {result.status.value}",
            f"  achieved   : {achieved}",
            f"  error_mm   : {err}",
            f"  message    : {result.message}",
        ])

    def _status_log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.status_lines.append(f"[{ts}] {message}")
        if len(self.status_lines) > 10:
            self.status_lines = self.status_lines[-10:]
        self.status_var.set("\n".join(self.status_lines))

    def _result_achieved_xyz(self, result: ActionResult) -> str:
        if result.achieved_x is None or result.achieved_y is None or result.achieved_z is None:
            return "None"
        return self._fmt_xyz((result.achieved_x, result.achieved_y, result.achieved_z))

    def _styled_entry(self, parent: tk.Frame, var: tk.StringVar) -> tk.Entry:
        return tk.Entry(
            parent,
            textvariable=var,
            bg="#161A2C",
            fg=C["text"],
            insertbackground=C["text"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=C["border"],
            highlightcolor=C["accent"],
            font=(FF, 10),
        )

    def _btn(self, parent: tk.Frame, text: str, command):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=C["accent"],
            fg="white",
            activebackground=C["accent_dim"],
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=12,
            pady=9,
            font=(FF, 10, "bold"),
            cursor="hand2",
        )

    @staticmethod
    def _stage_to_action_hint(stage: str) -> str:
        hints = {
            "idle": "Wait for initialization",
            "ready_to_move_source": "Move Above Source",
            "ready_to_pick": "Pick",
            "ready_to_move_target": "Move Above Box",
            "ready_to_place": "Place In Box",
        }
        return hints.get(stage, "Start Simulation")

    def _refresh_buttons_by_stage(self) -> None:
        if self.is_executing:
            for btn in (
                self.btn_move_src,
                self.btn_pick,
                self.btn_move_tgt,
                self.btn_place,
                self.btn_home,
            ):
                btn.config(state=tk.DISABLED)
            return

        self.btn_home.config(state=tk.NORMAL if self.flow_stage != "idle" else tk.DISABLED)
        self.btn_move_src.config(
            state=tk.NORMAL
            if self.flow_stage in {"ready_to_move_source", "ready_to_pick"}
            else tk.DISABLED
        )
        self.btn_pick.config(state=tk.NORMAL if self.flow_stage == "ready_to_pick" else tk.DISABLED)
        self.btn_move_tgt.config(
            state=tk.NORMAL
            if self.flow_stage in {"ready_to_move_target", "ready_to_place"}
            else tk.DISABLED
        )
        self.btn_place.config(state=tk.NORMAL if self.flow_stage == "ready_to_place" else tk.DISABLED)

    def _begin_action(self, action_name: str) -> bool:
        if self.is_executing:
            self._status_log(f"{action_name} blocked: another action is running")
            return False
        self.is_executing = True
        self._refresh_buttons_by_stage()
        return True

    def _end_action(self) -> None:
        self.is_executing = False
        self._refresh_buttons_by_stage()

    def _require_stage(self, expected_stage: str, action_name: str) -> bool:
        if self.flow_stage == expected_stage:
            return True
        next_step = self._stage_to_action_hint(self.flow_stage)
        self._log(f"{action_name} blocked | current_stage={self.flow_stage} | next={next_step}")
        self._status_log(f"{action_name} blocked. Next: {next_step}")
        return False

    def _require_any_stage(self, expected_stages: tuple[str, ...], action_name: str) -> bool:
        if self.flow_stage in expected_stages:
            return True
        next_step = self._stage_to_action_hint(self.flow_stage)
        self._log(f"{action_name} blocked | current_stage={self.flow_stage} | next={next_step}")
        self._status_log(f"{action_name} blocked. Next: {next_step}")
        return False

    def start_simulation(self) -> None:
        if not self._begin_action("Start Simulation"):
            return
        try:
            self.status_lines = []
            self._status_log("Initializing simulation...")
            self.executor.config.gui = self.viewer_var.get()
            self.executor.setup_environment()
            self.executor.configure_robot_model()
            self._spawn_demo_scene()
            self._set_demo_camera()
            self._log(f"Simulation started | viewer={self.executor.config.gui}")
            self._log("Initial scene ready: balls + box + table")
            for color, bid in self.ball_ids.items():
                if p is not None:
                    pos, _orn = p.getBasePositionAndOrientation(bid)
                    self._log(f"Ball[{color}] id={bid} at {self._fmt_xyz((float(pos[0]), float(pos[1]), float(pos[2])))}")
            self._log(f"Box drop point at {self._fmt_xyz(BOX_DROP_POINT)}")
            self.flow_stage = "ready_to_move_source"
            self._status_log("Simulation started. Initial scene ready. Next: Move Above Source")
        except Exception as ex:
            messagebox.showerror("Start failed", str(ex))
        finally:
            self._end_action()

    def _create_box(self, center_xyz: tuple[float, float, float]) -> None:
        if p is None:
            raise RuntimeError("pybullet is not installed")

        hx = 0.04
        hy = 0.04
        wall_h = 0.03
        thick = 0.005

        def _create_part(half_extents: tuple[float, float, float], pos: tuple[float, float, float]) -> int:
            cid = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half_extents))
            vid = p.createVisualShape(p.GEOM_BOX, halfExtents=list(half_extents), rgbaColor=[0.20, 0.36, 0.92, 1.0])
            return p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=cid,
                baseVisualShapeIndex=vid,
                basePosition=list(pos),
            )

        cx, cy, cz = center_xyz
        bottom_z = cz - wall_h
        self.box_body_ids.append(_create_part((hx, hy, thick), (cx, cy, bottom_z)))
        self.box_body_ids.append(_create_part((thick, hy, wall_h), (cx - hx, cy, cz)))
        self.box_body_ids.append(_create_part((thick, hy, wall_h), (cx + hx, cy, cz)))
        self.box_body_ids.append(_create_part((hx, thick, wall_h), (cx, cy - hy, cz)))
        self.box_body_ids.append(_create_part((hx, thick, wall_h), (cx, cy + hy, cz)))

    def _create_table(self) -> list[int]:
        if p is None:
            raise RuntimeError("pybullet is not installed")

        # Build a guaranteed-visible table with top + 4 legs.
        parts: list[int] = []

        top_half = (0.18, 0.14, 0.015)
        top_center = (0.45, 0.0, 0.125)
        top_cid = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(top_half))
        top_vid = p.createVisualShape(p.GEOM_BOX, halfExtents=list(top_half), rgbaColor=[0.25, 0.28, 0.33, 1.0])
        parts.append(
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=top_cid,
                baseVisualShapeIndex=top_vid,
                basePosition=list(top_center),
            )
        )

        leg_half = (0.012, 0.012, 0.11)
        leg_cid = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(leg_half))
        leg_vid = p.createVisualShape(p.GEOM_BOX, halfExtents=list(leg_half), rgbaColor=[0.16, 0.18, 0.22, 1.0])
        for sx in (-1, 1):
            for sy in (-1, 1):
                lx = top_center[0] + sx * (top_half[0] - leg_half[0] - 0.01)
                ly = top_center[1] + sy * (top_half[1] - leg_half[1] - 0.01)
                lz = top_center[2] - top_half[2] - leg_half[2]
                parts.append(
                    p.createMultiBody(
                        baseMass=0.0,
                        baseCollisionShapeIndex=leg_cid,
                        baseVisualShapeIndex=leg_vid,
                        basePosition=[lx, ly, lz],
                    )
                )

        return parts

    def _spawn_demo_scene(self) -> None:
        if p is None:
            raise RuntimeError("pybullet is not installed")

        self._detach_payload_if_needed()

        for bid in self.ball_ids.values():
            try:
                p.removeBody(bid)
            except Exception:
                pass
        self.ball_ids = {}

        for bid in self.box_body_ids:
            try:
                p.removeBody(bid)
            except Exception:
                pass
        self.box_body_ids = []

        for bid in self.table_body_ids:
            try:
                p.removeBody(bid)
            except Exception:
                pass
        self.table_body_ids = []

        self.active_ball_color = None
        self.active_ball_id = None

        # Stage colors for a cleaner, demo-friendly look.
        p.changeVisualShape(self.executor.plane_id, -1, rgbaColor=[0.92, 0.93, 0.96, 1.0])

        # Use procedural table to avoid URDF height mismatch across environments.
        self.table_body_ids = self._create_table()

        sphere_radius = 0.014
        sphere_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius)

        for color, spec in BALL_SPECS.items():
            sphere_visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=sphere_radius,
                rgbaColor=list(spec["rgba"]),
            )
            self.ball_ids[color] = p.createMultiBody(
                baseMass=0.03,
                baseCollisionShapeIndex=sphere_collision,
                baseVisualShapeIndex=sphere_visual,
                basePosition=list(spec["pos"]),
            )

        self._create_box(BOX_CENTER)

    def _selected_color(self) -> str:
        selected = self.ball_color_var.get().strip()
        color = self.ball_option_to_color.get(selected, selected.lower())
        if color not in BALL_SPECS and "(" in selected:
            color = selected.split("(", 1)[0].strip().lower()
        if color not in BALL_SPECS:
            raise ValueError("Pick Ball Color must be red, yellow, green, or orange")
        return color

    def _attach_payload_to_gripper(self, ball_id: int) -> None:
        if p is None:
            raise RuntimeError("pybullet is not installed")
        if self.executor.robot_id is None:
            raise RuntimeError("Robot is not initialized")

        if self.payload_constraint_id is not None:
            return

        ee_link = self.executor.config.end_effector_link_index
        link_state = p.getLinkState(self.executor.robot_id, ee_link)
        ee_pos, ee_orn = link_state[0], link_state[1]

        # Snap payload near gripper before constraining for a stable, visual pick result.
        p.resetBasePositionAndOrientation(ball_id, ee_pos, ee_orn)

        self.payload_constraint_id = p.createConstraint(
            parentBodyUniqueId=self.executor.robot_id,
            parentLinkIndex=ee_link,
            childBodyUniqueId=ball_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0.0, 0.0, 0.0],
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=[0.0, 0.0, 0.0, 1.0],
            childFrameOrientation=[0, 0, 0, 1],
        )

    def _detach_payload_if_needed(self) -> None:
        if p is None:
            return
        if self.payload_constraint_id is not None:
            try:
                p.removeConstraint(self.payload_constraint_id)
            finally:
                self.payload_constraint_id = None

    def _get_live_source_xyz(self) -> tuple[float, float, float]:
        if p is None or self.active_ball_id is None:
            raise RuntimeError("Please pick a valid ball and spawn scene first")
        pos, _orn = p.getBasePositionAndOrientation(self.active_ball_id)
        return float(pos[0]), float(pos[1]), float(pos[2])

    @staticmethod
    def _in_range(value: float, allowed: tuple[float, float]) -> bool:
        return allowed[0] <= value <= allowed[1]

    def _validate_xyz_range(self, x: float, y: float, z: float) -> None:
        errors: list[str] = []
        if not self._in_range(x, X_RANGE):
            errors.append(f"x must be within [{X_RANGE[0]:.2f}, {X_RANGE[1]:.2f}] m")
        if not self._in_range(y, Y_RANGE):
            errors.append(f"y must be within [{Y_RANGE[0]:.2f}, {Y_RANGE[1]:.2f}] m")
        if not self._in_range(z, Z_RANGE):
            errors.append(f"z must be within [{Z_RANGE[0]:.2f}, {Z_RANGE[1]:.2f}] m")

        if errors:
            raise ValueError("; ".join(errors))

    def _move_command(self, xyz: tuple[float, float, float], target_id: str) -> ActionResult:
        cmd = ActionCommand(
            action_type=ActionType.MOVE,
            target=RobotTarget(
                target_id=target_id,
                source_hotspot_id="demo",
                x=xyz[0],
                y=xyz[1],
                z=xyz[2],
            ),
        )
        return self.executor.execute_action(cmd)

    def move_above_source(self) -> None:
        if not self._begin_action("Move Above Source"):
            return
        try:
            if not self._require_any_stage(("ready_to_move_source", "ready_to_pick"), "Move Above Source"):
                return
            color = self._selected_color()
            if color not in self.ball_ids:
                raise RuntimeError("Scene is not ready. Press Start Simulation first")
            self.active_ball_color = color
            self.active_ball_id = self.ball_ids[color]

            sx, sy, sz = self._get_live_source_xyz()
            self._validate_xyz_range(sx, sy, sz)
            above = (sx, sy, sz + self.executor.config.approach_offset_m)
            self._status_log(
                f"Source ball[{color}] at {self._fmt_xyz((sx, sy, sz))} | move target {self._fmt_xyz(above)}"
            )
            self._log(f"Move Above Source | color={color} | source={self._fmt_xyz((sx, sy, sz))} | above={self._fmt_xyz(above)}")
            result = self._move_command(above, "source-approach")
            self._log_result("MOVE(source-approach)", above, result)
            if result.status != ActionStatus.SUCCESS:
                self._status_log(f"Move failed: {result.message}")
                return
            self.flow_stage = "ready_to_pick"
            self._status_log(
                "Move complete | achieved "
                f"{self._result_achieved_xyz(result)} | Next: Pick"
            )
        except Exception as ex:
            messagebox.showerror("Move failed", str(ex))
        finally:
            self._end_action()

    def pick_from_source(self) -> None:
        if not self._begin_action("Pick"):
            return
        try:
            if not self._require_stage("ready_to_pick", "Pick"):
                return
            if self.active_ball_id is None or self.active_ball_color is None:
                raise RuntimeError("Please run Move Above Source first")

            sx, sy, sz = self._get_live_source_xyz()

            pick_cmd = ActionCommand(
                action_type=ActionType.PICK,
                target=RobotTarget(
                    target_id="payload-source",
                    source_hotspot_id="demo",
                    x=sx,
                    y=sy,
                    z=sz,
                ),
            )
            self._log(
                f"Pick | color={self.active_ball_color} | ball_id={self.active_ball_id} | "
                f"source={self._fmt_xyz((sx, sy, sz))}"
            )
            self._status_log(
                f"Pick target source {self._fmt_xyz((sx, sy, sz))} | executing PICK"
            )
            pick_result = self.executor.execute_action(pick_cmd)
            self._log_result("PICK", (sx, sy, sz), pick_result)
            if pick_result.status != ActionStatus.SUCCESS:
                self._status_log(f"Pick failed: {pick_result.message}")
                return

            self._attach_payload_to_gripper(self.active_ball_id)

            self.flow_stage = "ready_to_move_target"
            self._status_log(
                f"Pick complete ({self.active_ball_color}) | achieved {self._result_achieved_xyz(pick_result)} | Next: Move Above Box"
            )
        except Exception as ex:
            messagebox.showerror("Pick failed", str(ex))
        finally:
            self._end_action()

    def move_above_target(self) -> None:
        if not self._begin_action("Move Above Box"):
            return
        try:
            if not self._require_any_stage(("ready_to_move_target", "ready_to_place"), "Move Above Box"):
                return
            tx, ty, tz = BOX_DROP_POINT
            self._validate_xyz_range(tx, ty, tz)

            above = (tx, ty, tz + self.executor.config.approach_offset_m)
            self._log(f"Move Above Box | box={self._fmt_xyz((tx, ty, tz))} | above={self._fmt_xyz(above)}")
            result = self._move_command(above, "target-approach")
            self._log_result("MOVE(box-approach)", above, result)
            if result.status != ActionStatus.SUCCESS:
                self._status_log(f"Move failed: {result.message}")
                return

            self.flow_stage = "ready_to_place"
            self._status_log(
                f"Move Above Box complete | achieved {self._result_achieved_xyz(result)} | Next: Place In Box"
            )
        except Exception as ex:
            messagebox.showerror("Move failed", str(ex))
        finally:
            self._end_action()

    def place_to_target(self) -> None:
        if not self._begin_action("Place In Box"):
            return
        try:
            if not self._require_stage("ready_to_place", "Place In Box"):
                return
            if self.active_ball_id is None:
                raise RuntimeError("Please pick a ball first")

            place_cmd = ActionCommand(
                action_type=ActionType.PLACE,
                target=RobotTarget(
                    target_id="hotspot-target",
                    source_hotspot_id="demo-hotspot",
                    x=BOX_DROP_POINT[0],
                    y=BOX_DROP_POINT[1],
                    z=BOX_DROP_POINT[2],
                ),
            )
            self._log(
                f"Place | color={self.active_ball_color} | ball_id={self.active_ball_id} | "
                f"drop={self._fmt_xyz(BOX_DROP_POINT)}"
            )
            place_result = self.executor.execute_action(place_cmd)
            self._log_result("PLACE", BOX_DROP_POINT, place_result)
            if place_result.status != ActionStatus.SUCCESS:
                self._status_log(f"Place failed: {place_result.message}")
                return

            self._detach_payload_if_needed()

            if p is not None:
                p.resetBasePositionAndOrientation(
                    self.active_ball_id,
                    [BOX_DROP_POINT[0], BOX_DROP_POINT[1], BOX_DROP_POINT[2]],
                    [0.0, 0.0, 0.0, 1.0],
                )
                pos, _orn = p.getBasePositionAndOrientation(self.active_ball_id)
                self._log(
                    f"Ball[{self.active_ball_color}] final position {self._fmt_xyz((float(pos[0]), float(pos[1]), float(pos[2])))}"
                )

            place_error = 0.0 if place_result.error_mm is None else place_result.error_mm
            self.flow_stage = "ready_to_move_source"
            self._status_log(
                f"Place complete: {self.active_ball_color} ball in box | place_error={place_error:.1f}mm"
            )
        except Exception as ex:
            messagebox.showerror("Place failed", str(ex))
        finally:
            self._end_action()

    def go_home(self) -> None:
        if not self._begin_action("Go Home"):
            return
        try:
            if self.flow_stage == "idle":
                self._status_log("Go Home blocked. Simulation is still initializing")
                self._log("Go Home blocked | current_stage=idle | next=Wait for initialization")
                return
            self._log(f"Go Home | target={self._fmt_xyz(HOME_POSITION)}")
            result = self._move_command(HOME_POSITION, "home")
            self._log_result("MOVE(home)", HOME_POSITION, result)
            if result.status != ActionStatus.SUCCESS:
                self._status_log(f"Go Home failed: {result.message}")
                return
            self._status_log(
                f"Go Home complete | achieved {self._result_achieved_xyz(result)}"
            )
        except Exception as ex:
            messagebox.showerror("Go Home failed", str(ex))
        finally:
            self._end_action()

    def _on_close(self) -> None:
        try:
            self._detach_payload_if_needed()
            self.executor.shutdown()
        finally:
            self.root.destroy()


def main() -> None:
    root = tk.Tk()
    RobotPickPlaceDemoUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
