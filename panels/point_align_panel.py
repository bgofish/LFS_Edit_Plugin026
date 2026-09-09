# SPDX-License-Identifier: GPL-3.0-or-later
"""3-Point Alignment Panel for Lichtfeld Studio EDIT plugin.

Workflow
--------
1. Load the REFERENCE 3DGS. Click [Pick Pt 1], [Pick Pt 2], [Pick Pt 3]
   in the Reference section — each starts the modal picker; click the model.
   The modal stays active between picks so you don't need to click the button
   for each point.  Press ESC to stop picking.  Then [Save Ref Points].

2. Switch to the ALIGN 3DGS. Repeat for the Align section. [Save Align Points].

3. Click [Calculate Transform]. Results appear below.

4. Click [Apply to Edit Panel] — writes values to settings.json.
   Then click [Read Settings] on the Edit tab to load them.

Math
----
Three-point similarity transform via SVD (Umeyama):
  s = σ_dst / σ_src   (RMS distance ratio)
  R = V·diag(1,1,det)·U^T   (SVD of cross-covariance, reflection-safe)
  t = μ_dst − s·R·μ_src
"""

from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
import lichtfeld as lf

# ── Coordinate convention ─────────────────────────────────────────────────────
# v0.5.0.x uses -Y-up; v0.5.1+ uses +Y-up.
#
# Picked points are world-space coordinates — they are already in the scene's
# native frame and must be fed to the SVD solver as-is.
#
# The Edit panel stores translation with ty and tz sign-flipped relative to
# world space in +Y-up mode (mirrors transform_panel._decompose_mat).
# So we solve in world space and only flip the OUTPUT translation before
# writing to settings.json / displaying in the panel.
def _parse_version(v: str) -> tuple:
    import re
    parts = v.lstrip("v").split(".")[:3]
    return tuple(int(re.match(r"\d+", x).group()) for x in parts)

_LFS_VER = _parse_version(lf.__version__)
Y_UP = _LFS_VER >= (0, 5, 1)   # True -> +Y-up  /  False -> -Y-up


def _world_t_to_panel(t: np.ndarray):
    """The SVD solver produces translation in world space, which matches
    the Edit panel's user-space values directly.  No sign conversion needed.
    (The ty/tz negation in transform_panel is a scene-matrix storage detail,
    not a user-space convention — it does not apply here.)
    """
    return float(t[0]), float(t[1]), float(t[2])


from ..operators.point_align_picker import (
    set_pick_callback,
    clear_pick_callback,
    was_pick_cancelled,
)

# ── Edit-gizmo tool registration ────────────────────────────────────────────
# LFS gizmos are driven by the currently ACTIVE TOOL (a ToolDef with
# gizmo="translate"), not by node selection alone -- selecting a node with
# no translate-tool active shows no gizmo at all. So Edit needs its own
# registered tool to switch into while dragging, and switches back to
# whatever was active before on Stop Editing. (Ported from CamPath_Json's
# main_panel.py edit-gizmo implementation.)

_EDIT_TOOL_ID = "LFS_Edit_Plugin.tools.point_edit"
_edit_tool_registered = False


def _ensure_edit_tool():
    global _edit_tool_registered
    if _edit_tool_registered:
        return
    try:
        from lfs_plugins.tool_defs.definition import ToolDef
        from lfs_plugins.tools import ToolRegistry
        tool = ToolDef(
            id=_EDIT_TOOL_ID,
            label="Edit Align Point",
            icon="translation",
            group="transform",
            order=500,
            description="Drag the gizmo to move the picked Reference/Align point.",
            gizmo="translate",
        )
        ToolRegistry.register_tool(tool)
        _edit_tool_registered = True
    except Exception as e:
        lf.log.warning(f"3PT-ALIGN edit-tool registration error: {e}")


def _unregister_edit_tool():
    global _edit_tool_registered
    if not _edit_tool_registered:
        return
    try:
        from lfs_plugins.tools import ToolRegistry
        ToolRegistry.unregister_tool(_EDIT_TOOL_ID)
    except Exception:
        pass
    _edit_tool_registered = False


def _flip_yz(pos):
    """Convert between pick/world-space (x, y, z) -- used by the picked-point
    tuples and world_to_screen() -- and scene-node transform space, which
    uses a Y/Z convention negated from it. Self-inverse: apply the same
    conversion both when seeding the gizmo node and when reading its
    dragged transform back."""
    x, y, z = pos
    return (x, -y, -z)

# ── Constants ─────────────────────────────────────────────────────────────────
_NUM_POINTS = 3

# Overlay colours: ref pts (cyan shades), align pts (amber/magenta shades)
_REF_COLORS = [
    (0.2, 0.9, 1.0, 1.0),   # Ref Pt 1 – cyan
    (0.1, 0.7, 1.0, 1.0),   # Ref Pt 2 – blue-cyan
    (0.0, 0.5, 1.0, 1.0),   # Ref Pt 3 – blue
]
_TBA_COLORS = [
    (1.0, 0.8, 0.1, 1.0),   # Tgt Pt 1 – amber
    (1.0, 0.5, 0.1, 1.0),   # Tgt Pt 2 – orange
    (0.9, 0.3, 1.0, 1.0),   # Tgt Pt 3 – magenta
]

_STEP_OPTIONS = [(0.001, ".001"), (0.01, ".01"), (0.1, ".1"), (1.0, "1.0")]

# ── Module-level overlay state ─────────────────────────────────────────────────
# Slots 0-2 = reference pts, slots 3-5 = align pts.
# Both sets are kept alive simultaneously so neither disappears when picking
# the other mode.
_overlay_registered = False
_ref_overlay   = [None, None, None]   # world-space ref pts shown in overlay
_tba_overlay   = [None, None, None]   # world-space align pts shown in overlay
_picking_which = 0                    # 0=idle, 1-3=waiting for that slot
_picking_mode  = ""                   # "ref" or "tba"
_pending_pick  = None                 # set by modal callback, drained in on_update


# ── File paths ────────────────────────────────────────────────────────────────

def _plugin_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _reference_json() -> Path:
    return _plugin_dir() / "align_reference.json"


def _align_json() -> Path:
    return _plugin_dir() / "align_align.json"


# ── Modal operator invocation ─────────────────────────────────────────────────

def _invoke_pick_op(pkg_prefix: str):
    """Start the pick modal operator.  pkg_prefix = 'lfs_plugins.<foldername>'."""
    op_id = f"{pkg_prefix}.operators.point_align_picker.POINTALIGN_OT_pick_point"
    lf.log.info(f"3PT-ALIGN: invoking {op_id}")
    try:
        lf.ui.ops.invoke(op_id)
        lf.ui.request_redraw()
    except Exception as e:
        lf.log.error(f"3PT-ALIGN: invoke failed: {e}")


# ── Pick relay (called from modal operator) ───────────────────────────────────

def _on_point_picked(world_pos, point_num: int):
    """Relay from POINTALIGN_OT_pick_point → panel's on_update drain."""
    global _pending_pick
    _pending_pick = (world_pos, point_num)
    lf.ui.request_redraw()


# ── Draw overlay ───────────────────────────────────────────────────────────────

def _draw_handler(ctx):
    # Picking banner
    if _picking_which > 0:
        cols = _REF_COLORS if _picking_mode == "ref" else _TBA_COLORS
        col  = cols[_picking_which - 1]
        ctx.draw_text_2d(
            (20, 50),
            f"3-PT ALIGN — PICK POINT {_picking_which} ({_picking_mode.upper()}):  "
            f"click on model  (ESC to cancel)",
            col,
        )

    # Draw reference points (always, regardless of current pick mode)
    for i, pt in enumerate(_ref_overlay):
        if pt is None:
            continue
        col = _REF_COLORS[i]
        ctx.draw_point_3d(pt, col, 6.0)
        s = ctx.world_to_screen(pt)
        if s:
            ctx.draw_circle_2d(s, 11.0, col, 2.0)
            ctx.draw_text_2d((s[0] + 15, s[1] - 6), f"R{i + 1}", col)

    # Draw align points (always, regardless of current pick mode)
    for i, pt in enumerate(_tba_overlay):
        if pt is None:
            continue
        col = _TBA_COLORS[i]
        ctx.draw_point_3d(pt, col, 6.0)
        s = ctx.world_to_screen(pt)
        if s:
            ctx.draw_circle_2d(s, 11.0, col, 2.0)
            ctx.draw_text_2d((s[0] + 15, s[1] - 6), f"A{i + 1}", col)

    # Lines between consecutive set ref points
    ref_set = [p for p in _ref_overlay if p is not None]
    for k in range(len(ref_set) - 1):
        ctx.draw_line_3d(ref_set[k], ref_set[k + 1], (0.2, 0.7, 1.0, 0.45), 1.5)

    # Lines between consecutive set align points
    tba_set = [p for p in _tba_overlay if p is not None]
    for k in range(len(tba_set) - 1):
        ctx.draw_line_3d(tba_set[k], tba_set[k + 1], (1.0, 0.6, 0.1, 0.45), 1.5)


def _ensure_overlay():
    global _overlay_registered
    if not _overlay_registered:
        try:
            lf.remove_draw_handler("edit_3pt_align_overlay")
        except Exception:
            pass
        lf.add_draw_handler("edit_3pt_align_overlay", _draw_handler, "POST_VIEW")
        _overlay_registered = True


def _remove_overlay():
    global _overlay_registered
    try:
        lf.remove_draw_handler("edit_3pt_align_overlay")
    except Exception:
        pass
    _overlay_registered = False


def _overlay_active() -> bool:
    return (
        any(p is not None for p in _ref_overlay) or
        any(p is not None for p in _tba_overlay) or
        _picking_which > 0
    )


# ── Similarity transform solver ───────────────────────────────────────────────

def _solve_similarity(src: np.ndarray, dst: np.ndarray):
    """dst ≈ s·R·src + t  via SVD (reflection-safe Umeyama)."""
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c  = src - mu_src
    dst_c  = dst - mu_dst

    sigma_src = np.sqrt((src_c ** 2).sum(axis=1).mean())
    sigma_dst = np.sqrt((dst_c ** 2).sum(axis=1).mean())
    if sigma_src < 1e-12:
        raise ValueError("Source points are coincident — cannot determine scale.")
    s = sigma_dst / sigma_src

    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = mu_dst - s * (R @ mu_src)
    return s, R, t


def _decompose_rotation(R: np.ndarray):
    if abs(R[2, 0]) < 1.0 - 1e-6:
        ry = math.asin(-R[2, 0])
        rx = math.atan2(R[2, 1], R[2, 2])
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        ry = math.pi / 2 if R[2, 0] < 0 else -math.pi / 2
        rx = math.atan2(-R[1, 2], R[1, 1])
        rz = 0.0
    # The app's Ry convention is the inverse of the SVD solver's output.
    # Rx is correct as-is; so Ry & Rz needs negating.
    return math.degrees(rx), -math.degrees(ry),-math.degrees(rz)


# ── Panel ─────────────────────────────────────────────────────────────────────

class PointAlignPanel(lf.ui.Panel):
    id                 = "edit.point_align_panel"
    label              = "3-Point Align"
    space              = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order              = 4
    template           = str(Path(__file__).resolve().with_name("point_align_panel.rml"))
    height_mode        = lf.ui.PanelHeightMode.CONTENT
    update_interval_ms = 100

    def __init__(self):
        self._handle     = None
        self._pkg_prefix = ""   # filled from __name__ on first bind

        self._ref_pts: list = [None, None, None]
        self._tba_pts: list = [None, None, None]

        # Edit gizmo (drag-to-move) state — 0=idle, 1-3=editing that slot.
        # Backed by a throwaway proxy locator node since LFS gizmos bind to
        # real scene-node transforms, not bare tuples; created on Edit,
        # deleted on Stop Editing. Mirrors CamPath_Json's implementation.
        self._editing_which    = 0
        self._editing_mode     = ""      # "ref" or "tba"
        self._editing_node_name = None
        self._prev_tool_id      = None

        self._step_size  = 0.01   # active step for nudge buttons

        self._result_tx = self._result_ty = self._result_tz = 0.0
        self._result_rx = self._result_ry = self._result_rz = 0.0
        self._result_sx = self._result_sy = self._result_sz = 1.0
        self._has_result = False

        self._status = "Pick 3 points on the REFERENCE 3DGS, then 3 on the align."

    @classmethod
    def poll(cls, context) -> bool:
        return True

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_bind_model(self, ctx):
        parts = __name__.split(".")
        self._pkg_prefix = ".".join(parts[:2])

        model = ctx.create_data_model("point_align_panel")

        # Point pick buttons & labels — reference
        for i in range(_NUM_POINTS):
            slot = i + 1
            model.bind_func(f"ref_pt{slot}_label", self._make_pt_label_fn("ref", i))
            model.bind_event(f"ref_pick{slot}", self._make_pick_handler("ref", slot))
            # Nudge buttons ref
            for ax in ("x", "y", "z"):
                model.bind_event(f"ref_pt{slot}_{ax}_minus",
                                 self._make_nudge_handler("ref", slot, ax, -1))
                model.bind_event(f"ref_pt{slot}_{ax}_plus",
                                 self._make_nudge_handler("ref", slot, ax, +1))
            model.bind_func(f"ref_pt{slot}_has_pos",
                            self._make_has_pos_fn("ref", i))
            # Edit gizmo buttons ref
            model.bind_event(f"ref_edit{slot}",
                             self._make_edit_handler("ref", slot))
            model.bind_func(f"ref_pt{slot}_editing",
                            self._make_editing_fn("ref", slot))
            model.bind_func(f"ref_pt{slot}_edit_idle",
                            self._make_edit_idle_fn("ref", i, slot))

        # Point pick buttons & labels — align
        for i in range(_NUM_POINTS):
            slot = i + 1
            model.bind_func(f"tba_pt{slot}_label", self._make_pt_label_fn("tba", i))
            model.bind_event(f"tba_pick{slot}", self._make_pick_handler("tba", slot))
            # Nudge buttons align
            for ax in ("x", "y", "z"):
                model.bind_event(f"tba_pt{slot}_{ax}_minus",
                                 self._make_nudge_handler("tba", slot, ax, -1))
                model.bind_event(f"tba_pt{slot}_{ax}_plus",
                                 self._make_nudge_handler("tba", slot, ax, +1))
            model.bind_func(f"tba_pt{slot}_has_pos",
                            self._make_has_pos_fn("tba", i))
            # Edit gizmo buttons align
            model.bind_event(f"tba_edit{slot}",
                             self._make_edit_handler("tba", slot))
            model.bind_func(f"tba_pt{slot}_editing",
                            self._make_editing_fn("tba", slot))
            model.bind_func(f"tba_pt{slot}_edit_idle",
                            self._make_edit_idle_fn("tba", i, slot))

        # Step size buttons
        for step_val, step_lbl in _STEP_OPTIONS:
            key = step_lbl.replace(".", "_")
            model.bind_event(f"step_{key}", self._make_step_handler(step_val))
            model.bind_func(f"step_{key}_active",
                            self._make_step_active_fn(step_val))

        # Save / load / clear
        model.bind_event("save_ref",  self._on_save_ref)
        model.bind_event("save_tba",  self._on_save_tba)
        model.bind_event("load_ref",  self._on_load_ref)
        model.bind_event("load_tba",  self._on_load_tba)
        model.bind_event("clear_ref", self._on_clear_ref)
        model.bind_event("clear_tba", self._on_clear_tba)

        # Edit gizmo — stop button (shared across all points)
        model.bind_event("stop_edit_point", self._on_stop_edit_point)

        # Calculate & apply
        model.bind_func("has_result", lambda: self._has_result)
        model.bind_func("result_tx",  lambda: f"{self._result_tx:+.4f}")
        model.bind_func("result_ty",  lambda: f"{self._result_ty:+.4f}")
        model.bind_func("result_tz",  lambda: f"{self._result_tz:+.4f}")
        model.bind_func("result_rx",  lambda: f"{self._result_rx:+.4f}°")
        model.bind_func("result_ry",  lambda: f"{self._result_ry:+.4f}°")
        model.bind_func("result_rz",  lambda: f"{self._result_rz:+.4f}°")
        model.bind_func("result_sx",  lambda: f"{self._result_sx:.6f}")
        model.bind_event("do_calculate",            self._on_calculate)
        model.bind_event("do_calculate_scale_fixed", self._on_calculate_scale_fixed)
        model.bind_event("do_prefill",   self._on_prefill)

        # Status
        model.bind_func("status_text",  lambda: self._status)
        model.bind_func("status_class", self._status_class)

        self._handle = model.get_handle()

    def on_update(self, doc):
        changed = self._process_pending_pick()
        self._poll_editing_point()
        if _overlay_active() or self._editing_which:
            try:
                lf.ui.request_redraw()
            except Exception:
                pass
        return changed

    def on_unmount(self, doc):
        self._stop_editing_point()
        doc.remove_data_model("point_align_panel")
        self._handle = None

    # ── Label / state factories ────────────────────────────────────────────────

    def _make_pt_label_fn(self, mode: str, idx: int):
        slot = idx + 1
        def _fn():
            pts = self._ref_pts if mode == "ref" else self._tba_pts
            p   = pts[idx]
            if _picking_which == slot and _picking_mode == mode:
                return f"◉ Picking Pt {slot}…"
            if p is None:
                return f"Pick Pt {slot}"
            return f"Pt {slot}: {p[0]:.3f}  {p[1]:.3f}  {p[2]:.3f}"
        return _fn

    def _make_has_pos_fn(self, mode: str, idx: int):
        def _fn():
            pts = self._ref_pts if mode == "ref" else self._tba_pts
            return pts[idx] is not None
        return _fn

    def _make_editing_fn(self, mode: str, slot: int):
        return lambda: self._editing_mode == mode and self._editing_which == slot

    def _make_edit_idle_fn(self, mode: str, idx: int, slot: int):
        """True when the point has a position and isn't the one being edited
        right now — i.e. when the [Edit] button (not [Stop Editing]) should
        show for this point."""
        def _fn():
            pts = self._ref_pts if mode == "ref" else self._tba_pts
            is_editing_this = self._editing_mode == mode and self._editing_which == slot
            return pts[idx] is not None and not is_editing_this
        return _fn

    def _make_step_handler(self, step_val: float):
        def _handler(handle, event, args):
            self._step_size = step_val
            self._dirty_step_labels()
        return _handler

    def _make_step_active_fn(self, step_val: float):
        return lambda: abs(self._step_size - step_val) < 1e-9

    # ── Nudge handler factory ─────────────────────────────────────────────────

    def _make_nudge_handler(self, mode: str, slot: int, axis: str, sign: int):
        idx     = slot - 1
        ax_idx  = {"x": 0, "y": 1, "z": 2}[axis]
        def _handler(handle, event, args):
            pts = self._ref_pts if mode == "ref" else self._tba_pts
            p   = pts[idx]
            if p is None:
                return
            delta   = [0.0, 0.0, 0.0]
            delta[ax_idx] = sign * self._step_size
            new_pt  = (p[0] + delta[0], p[1] + delta[1], p[2] + delta[2])
            pts[idx] = new_pt
            # Keep overlay in sync
            if mode == "ref":
                _ref_overlay[idx] = new_pt
            else:
                _tba_overlay[idx] = new_pt
            self._has_result = False
            self._dirty_all()
            try:
                lf.ui.request_redraw()
            except Exception:
                pass
        return _handler

    # ── Edit gizmo (drag-to-move) ─────────────────────────────────────────────
    # LFS gizmos bind to a real scene-node's transform, not a bare tuple, so
    # "Edit" creates a small throwaway proxy/locator node, seeds its transform
    # with the picked point, and selects it so the built-in Move gizmo
    # attaches. While editing, on_update() polls the node's transform each
    # tick and mirrors it back into self._ref_pts / self._tba_pts. "Stop
    # Editing" deletes the node -- nothing lingers in the Scene tree after.
    # Ported from CamPath_Json's Orbit-Centre/Look-Target edit gizmo.

    def _edit_node_name(self, mode: str, slot: int) -> str:
        label = "Ref" if mode == "ref" else "Align"
        return f"[3PT Align] {label} Pt {slot} (editing)"

    def _make_edit_handler(self, mode: str, slot: int):
        def _handler(handle, event, args):
            self._start_editing_point(mode, slot)
        return _handler

    def _start_editing_point(self, mode: str, slot: int):
        idx = slot - 1
        pts = self._ref_pts if mode == "ref" else self._tba_pts
        pos = pts[idx]
        if pos is None:
            self._status = "Pick this point before editing it."
            self._dirty("status_text", "status_class")
            return

        if self._editing_mode == mode and self._editing_which == slot:
            return  # already editing this exact point
        if self._editing_which:
            self._stop_editing_point()

        # Cancel any in-progress pick session -- picking and dragging a
        # gizmo at the same time would fight over clicks in the viewport.
        global _picking_which, _picking_mode
        if _picking_which > 0:
            try:
                lf.ui.ops.cancel_modal()
            except Exception:
                pass
            clear_pick_callback()
            _picking_which = 0
            _picking_mode  = ""

        if not lf.has_scene():
            self._status = "No scene loaded -- can't place an edit gizmo."
            self._dirty("status_text", "status_class")
            return

        # Capture the currently-active tool BEFORE we touch selection at all
        # -- select_node() below may itself auto-switch the active tool to
        # "Select", and if we captured after that we'd just be recording our
        # own side effect and always restoring into Select on Stop Editing.
        try:
            self._prev_tool_id = lf.ui.get_active_tool()
        except Exception:
            self._prev_tool_id = None

        name = self._edit_node_name(mode, slot)
        try:
            scene = lf.get_scene()
            try:
                # Clean up a stale node from a previous session that didn't
                # get removed (e.g. the app closed mid-edit).
                scene.remove_node(name, keep_children=False)
            except Exception:
                pass
            scene.add_group(name)
            node_pos = _flip_yz(pos)
            matrix = lf.compose_transform(
                translation=list(node_pos), euler_deg=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]
            )
            lf.set_node_transform(name, matrix)
            lf.select_node(name)

            # Switch to the translate-gizmo tool -- selection alone doesn't
            # show a gizmo, the active tool has to be one with gizmo="translate".
            _ensure_edit_tool()
            from lfs_plugins.tools import ToolRegistry
            ToolRegistry.set_active(_EDIT_TOOL_ID)
            try:
                lf.ui.request_redraw()
            except Exception:
                pass
        except Exception as e:
            lf.log.warning(f"3PT-ALIGN edit-gizmo start error: {e}")
            self._status = f"Couldn't start editing: {e}"
            self._dirty("status_text", "status_class")
            return

        self._editing_mode      = mode
        self._editing_which     = slot
        self._editing_node_name = name
        label = "REFERENCE" if mode == "ref" else "ALIGN"
        self._status = f"Drag the gizmo to move {label} Point {slot}..."
        self._dirty_all()

    def _stop_editing_point(self):
        if not self._editing_which:
            return
        name = self._editing_node_name
        # Deselect BEFORE removing the node -- removing it while still
        # selected can leave a stale selection reference behind that the
        # Select-tool's floating toolbar doesn't clear on its own.
        try:
            lf.deselect_all()
        except Exception:
            pass
        try:
            scene = lf.get_scene()
            if scene is not None and name:
                scene.remove_node(name, keep_children=False)
        except Exception as e:
            lf.log.warning(f"3PT-ALIGN edit-gizmo cleanup error: {e}")

        restored = False
        try:
            from lfs_plugins.tools import ToolRegistry
            # NOTE: '' is a legitimate "no tool was active" value, not an
            # absence of one -- `if self._prev_tool_id:` would silently skip
            # restoring it since empty string is falsy in Python.
            # `is not None` treats '' as something to actively restore.
            if self._prev_tool_id is not None:
                restored = ToolRegistry.set_active(self._prev_tool_id)
        except Exception as e:
            lf.log.warning(f"3PT-ALIGN restore-tool error: {e}")

        # Belt-and-braces: switching tools can itself re-populate a
        # selection (some tools auto-select the last-touched node when they
        # activate), which is what re-opens the Select-tool's floating
        # toolbar even though we deselected above. Deselect again *after*
        # the tool swap so nothing is selected in the tool we land back on.
        try:
            lf.deselect_all()
        except Exception:
            pass

        try:
            now_active = ToolRegistry.get_active_id()
        except Exception:
            now_active = "?"
        lf.log.info(
            f"3PT-ALIGN stop-edit: prev_tool_id={self._prev_tool_id!r} "
            f"restore_call_returned={restored} active_tool_now={now_active!r}"
        )

        self._prev_tool_id      = None
        self._editing_mode      = ""
        self._editing_which     = 0
        self._editing_node_name = None
        # Remove the edit tool from the toolbar again -- it should only be
        # visible/selectable there for the duration of an active edit
        # session, not sit around as a standing option.
        _unregister_edit_tool()
        try:
            lf.ui.request_redraw()
        except Exception:
            pass

    def _poll_editing_point(self):
        """Called every on_update() tick while an edit gizmo is active."""
        if not self._editing_which or not self._editing_node_name:
            return
        try:
            matrix = lf.get_node_transform(self._editing_node_name)
            node_pos = lf.decompose_transform(matrix)["translation"]
            x, y, z = _flip_yz((float(node_pos[0]), float(node_pos[1]), float(node_pos[2])))
        except Exception:
            return

        idx  = self._editing_which - 1
        mode = self._editing_mode
        new_pt = (x, y, z)
        if mode == "ref":
            self._ref_pts[idx] = new_pt
            _ref_overlay[idx]  = new_pt
        else:
            self._tba_pts[idx] = new_pt
            _tba_overlay[idx]  = new_pt

        self._has_result = False
        self._dirty_all()

    def _on_stop_edit_point(self, handle, event, args):
        self._stop_editing_point()
        self._status = "Editing stopped"
        self._dirty_all()
        try:
            lf.ui.request_redraw()
        except Exception:
            pass

    # ── Pick handler factory ──────────────────────────────────────────────────

    def _make_pick_handler(self, mode: str, slot: int):
        def _handler(handle, event, args):
            global _picking_which, _picking_mode

            # If already picking this exact slot/mode → cancel
            if _picking_which == slot and _picking_mode == mode:
                _picking_which = 0
                _picking_mode  = ""
                clear_pick_callback()
                try:
                    lf.ui.ops.cancel_modal()
                except Exception:
                    pass
                self._status = "Pick cancelled."
                self._dirty_all()
                lf.ui.request_redraw()
                return

            # Cancel any previous pick session cleanly
            if _picking_which > 0:
                try:
                    lf.ui.ops.cancel_modal()
                except Exception:
                    pass
                clear_pick_callback()

            # An active edit-gizmo session would fight over viewport clicks
            # with a new pick session -- stop it first.
            if self._editing_which:
                self._stop_editing_point()

            # Set up new pick — do NOT reset the other mode's overlay points
            _picking_which = slot
            _picking_mode  = mode

            # Sync both overlay arrays from current panel state (preserves both)
            for i in range(_NUM_POINTS):
                _ref_overlay[i] = self._ref_pts[i]
                _tba_overlay[i] = self._tba_pts[i]

            _ensure_overlay()
            self._status = (
                f"Click on the {'REFERENCE' if mode == 'ref' else 'align'} model "
                f"to pick Point {slot}…  (ESC to cancel)"
            )
            set_pick_callback(_on_point_picked, slot)
            _invoke_pick_op(self._pkg_prefix)
            self._dirty_all()
            lf.ui.request_redraw()
        return _handler

    # ── Pending-pick drain (called every on_update) ───────────────────────────

    def _process_pending_pick(self) -> bool:
        global _pending_pick, _picking_which, _picking_mode

        # Check for ESC / cancel from the modal operator
        if was_pick_cancelled() and _picking_which > 0:
            _picking_which = 0
            _picking_mode  = ""
            self._status = "Pick cancelled."
            self._dirty_all()
            lf.ui.request_redraw()
            return True

        if _pending_pick is None:
            return False

        world_pos, slot = _pending_pick
        _pending_pick   = None

        idx  = slot - 1
        mode = _picking_mode

        if mode == "ref":
            self._ref_pts[idx] = world_pos
            _ref_overlay[idx]  = world_pos
        else:
            self._tba_pts[idx] = world_pos
            _tba_overlay[idx]  = world_pos

        # Advance the expected slot for the NEXT click (modal stays alive)
        next_slot = slot + 1
        if next_slot <= _NUM_POINTS:
            _picking_which = next_slot
            set_pick_callback(_on_point_picked, next_slot)
            self._status = (
                f"Pt {slot} set — now click Point {next_slot} on the model."
            )
        else:
            # All 3 picked — stop modal
            _picking_which = 0
            _picking_mode  = ""
            clear_pick_callback()
            try:
                lf.ui.ops.cancel_modal()
            except Exception:
                pass
            self._status = (
                f"All 3 {'reference' if mode == 'ref' else 'align'} points set."
            )

        self._has_result = False
        self._dirty_all()
        lf.ui.request_redraw()
        return True

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_save_ref(self, handle, event, args):
        if not all(p is not None for p in self._ref_pts):
            self._status = "Pick all 3 reference points before saving."
            self._dirty("status_text", "status_class")
            return
        self._write_json(_reference_json(), self._ref_pts, "reference")

    def _on_save_tba(self, handle, event, args):
        if not all(p is not None for p in self._tba_pts):
            self._status = "Pick all 3 align points before saving."
            self._dirty("status_text", "status_class")
            return
        self._write_json(_align_json(), self._tba_pts, "align")

    def _on_load_ref(self, handle, event, args):
        pts, err = self._read_json(_reference_json())
        if err:
            self._status = f"Load failed: {err}"
        else:
            self._ref_pts    = pts
            for i in range(_NUM_POINTS):
                _ref_overlay[i] = pts[i]
            self._has_result = False
            self._status = f"Loaded reference points from {_reference_json().name}."
            _ensure_overlay()
        self._dirty_all()

    def _on_load_tba(self, handle, event, args):
        pts, err = self._read_json(_align_json())
        if err:
            self._status = f"Load failed: {err}"
        else:
            self._tba_pts    = pts
            for i in range(_NUM_POINTS):
                _tba_overlay[i] = pts[i]
            self._has_result = False
            self._status = f"Loaded align points from {_align_json().name}."
            _ensure_overlay()
        self._dirty_all()

    def _on_clear_ref(self, handle, event, args):
        if self._editing_mode == "ref":
            self._stop_editing_point()
        self._ref_pts    = [None, None, None]
        for i in range(_NUM_POINTS):
            _ref_overlay[i] = None
        self._has_result = False
        self._status = "Reference points cleared."
        self._dirty_all()

    def _on_clear_tba(self, handle, event, args):
        if self._editing_mode == "tba":
            self._stop_editing_point()
        self._tba_pts    = [None, None, None]
        for i in range(_NUM_POINTS):
            _tba_overlay[i] = None
        self._has_result = False
        self._status = "Align points cleared."
        self._dirty_all()

    def _on_calculate(self, handle, event, args):
        if not (all(p is not None for p in self._ref_pts) and
                all(p is not None for p in self._tba_pts)):
            self._status = "Need all 6 points (3 ref + 3 align) to calculate."
            self._dirty("status_text", "status_class")
            return
        try:
            # Solve in world space — picked points are already native scene coords.
            src = np.array(self._tba_pts, dtype=np.float64)
            dst = np.array(self._ref_pts, dtype=np.float64)
            s, R, t = _solve_similarity(src, dst)
            rx, ry, rz = _decompose_rotation(R)
            self._result_tx, self._result_ty, self._result_tz = _world_t_to_panel(t)
            self._result_rx = rx
            self._result_ry = ry
            self._result_rz = rz
            self._result_sx = self._result_sy = self._result_sz = float(s)
            self._has_result = True
            self._status = (
                f"Solved — T({self._result_tx:+.3f}, {self._result_ty:+.3f}, "
                f"{self._result_tz:+.3f})  R({self._result_rx:+.2f}°, "
                f"{self._result_ry:+.2f}°, {self._result_rz:+.2f}°)  "
                f"Scale={s:.5f} — press [Write to 3pAlign]."
            )
        except Exception as e:
            self._status = f"Solve error: {e}"
            lf.log.error(f"3PT-ALIGN calc error: {e}")
        self._dirty_all()

    def _on_calculate_scale_fixed(self, handle, event, args):
        """Same as _on_calculate but forces scale = 1.0.

        The rotation and translation are re-solved with the scale constrained:
          R  = same SVD rotation (scale-independent)
          t  = μ_dst − R·μ_src   (centroid alignment at unit scale)
        """
        if not (all(p is not None for p in self._ref_pts) and
                all(p is not None for p in self._tba_pts)):
            self._status = "Need all 6 points (3 ref + 3 align) to calculate."
            self._dirty("status_text", "status_class")
            return
        try:
            # Solve in world space — picked points are already native scene coords.
            src = np.array(self._tba_pts, dtype=np.float64)
            dst = np.array(self._ref_pts, dtype=np.float64)

            mu_src = src.mean(axis=0)
            mu_dst = dst.mean(axis=0)
            src_c  = src - mu_src
            dst_c  = dst - mu_dst

            H = src_c.T @ dst_c
            U, _, Vt = np.linalg.svd(H)
            d = np.linalg.det(Vt.T @ U.T)
            R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

            # Translation at unit scale: t = μ_dst − R·μ_src
            t = mu_dst - R @ mu_src

            rx, ry, rz = _decompose_rotation(R)
            self._result_tx, self._result_ty, self._result_tz = _world_t_to_panel(t)
            self._result_rx = rx
            self._result_ry = ry
            self._result_rz = rz
            self._result_sx = self._result_sy = self._result_sz = 1.0
            self._has_result = True
            self._status = (
                f"Solved (scale fixed=1) — T({self._result_tx:+.3f}, "
                f"{self._result_ty:+.3f}, {self._result_tz:+.3f})  "
                f"R({self._result_rx:+.2f}°, {self._result_ry:+.2f}°, "
                f"{self._result_rz:+.2f}°) — press [Apply to Edit Panel]."
            )
        except Exception as e:
            self._status = f"Solve error: {e}"
            lf.log.error(f"3PT-ALIGN scale-fixed calc error: {e}")
        self._dirty_all()

    def _on_prefill(self, handle, event, args):
        if not self._has_result:
            self._status = "Calculate transform first."
            self._dirty("status_text", "status_class")
            return
        try:
            settings_path = _plugin_dir() / "settings.json"
            try:
                data = json.loads(settings_path.read_text(encoding="utf-8"))
            except Exception:
                data = {}

            t = data.get("transform", {})
            t.update({
                "tx": round(self._result_tx, 4),
                "ty": round(self._result_ty, 4),
                "tz": round(self._result_tz, 4),
                "rx": round(self._result_rx, 4),
                "ry": round(self._result_ry, 4),
                "rz": round(self._result_rz, 4),
                "sx": round(self._result_sx, 6),
                "sy": round(self._result_sy, 6),
                "sz": round(self._result_sz, 6),
            })
            data["transform"] = t

            lim    = data.get("limits", {})
            t_vals = [abs(self._result_tx), abs(self._result_ty), abs(self._result_tz)]
            t_max  = max(lim.get("translation_max", 50.0), max(t_vals) * 1.2)
            t_min  = min(lim.get("translation_min", -50.0), -max(t_vals) * 1.2)
            s_max  = max(lim.get("scale_max", 5.0), self._result_sx * 1.1)
            s_min  = min(lim.get("scale_min", 0.01),
                         self._result_sx * 0.9 if self._result_sx > 1e-9 else 0.01)
            lim["translation_min"] = round(t_min, 2)
            lim["translation_max"] = round(t_max, 2)
            lim["scale_max"]       = round(s_max, 4)
            lim["scale_min"]       = round(s_min, 6)
            data["limits"] = lim

            settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            # Also write a dedicated 3pAlign.json for the Edit panel buttons
            palign_path = _plugin_dir() / "3pAlign.json"
            palign_data = {"transform": {
                "tx": round(self._result_tx, 4),
                "ty": round(self._result_ty, 4),
                "tz": round(self._result_tz, 4),
                "rx": round(self._result_rx, 4),
                "ry": round(self._result_ry, 4),
                "rz": round(self._result_rz, 4),
                "sx": round(self._result_sx, 6),
                "sy": round(self._result_sy, 6),
                "sz": round(self._result_sz, 6),
            }}
            palign_path.write_text(json.dumps(palign_data, indent=2), encoding="utf-8")

            self._status = (
                "Written to 3pAlign.json — "
                "click [Read 3pAlign] on the Edit tab to load."
            )
        except Exception as e:
            self._status = f"Prefill error: {e}"
        self._dirty("status_text", "status_class")

    # ── JSON helpers ──────────────────────────────────────────────────────────

    def _write_json(self, path: Path, pts: list, label: str):
        try:
            payload = {
                "label":  label,
                "points": [
                    {"index": i + 1, "x": p[0], "y": p[1], "z": p[2]}
                    for i, p in enumerate(pts)
                ],
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._status = f"Saved {label} points to {path.name}."
        except Exception as e:
            self._status = f"Save error: {e}"
        self._dirty("status_text", "status_class")

    def _read_json(self, path: Path):
        try:
            payload  = json.loads(path.read_text(encoding="utf-8"))
            raw_pts  = payload.get("points", [])
            if len(raw_pts) != _NUM_POINTS:
                return None, f"Expected {_NUM_POINTS} points, got {len(raw_pts)}."
            return [(p["x"], p["y"], p["z"]) for p in raw_pts], None
        except FileNotFoundError:
            return None, f"{path.name} not found."
        except Exception as e:
            return None, str(e)

    # ── Dirty helpers ─────────────────────────────────────────────────────────

    def _dirty(self, *fields):
        if not self._handle:
            return
        for f in fields:
            self._handle.dirty(f)

    def _dirty_step_labels(self):
        fields = []
        for _, step_lbl in _STEP_OPTIONS:
            key = step_lbl.replace(".", "_")
            fields.append(f"step_{key}_active")
        self._dirty(*fields)

    def _dirty_all(self):
        fields = [
            "has_result",
            "result_tx", "result_ty", "result_tz",
            "result_rx", "result_ry", "result_rz",
            "result_sx",
            "status_text", "status_class",
        ]
        for i in range(1, _NUM_POINTS + 1):
            fields += [
                f"ref_pt{i}_label", f"ref_pt{i}_has_pos",
                f"ref_pt{i}_editing", f"ref_pt{i}_edit_idle",
                f"tba_pt{i}_label", f"tba_pt{i}_has_pos",
                f"tba_pt{i}_editing", f"tba_pt{i}_edit_idle",
            ]
        for _, step_lbl in _STEP_OPTIONS:
            key = step_lbl.replace(".", "_")
            fields.append(f"step_{key}_active")
        self._dirty(*fields)

    def _status_class(self) -> str:
        s = self._status
        if any(w in s for w in ("Saved", "Loaded", "Solved", "Written",
                                 "set —", "set.", "Cleared", "points set")):
            return "text-accent"
        if s and ("error" in s.lower() or "failed" in s.lower() or
                  "not found" in s.lower()):
            return "text-muted"
        return "text-default"


# ── Called from plugin on_unload ──────────────────────────────────────────────

def remove_point_align_draw_handler():
    _remove_overlay()
    _unregister_edit_tool()
