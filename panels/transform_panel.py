# SPDX-FileCopyrightText: 2025
# SPDX-License-Identifier: GPL-3.0-or-later

"""Transform Editor Panel for Lichtfeld Studio."""

from __future__ import annotations
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import lichtfeld as lf

# ── Align: point-picker operator ─────────────────────────────────────────────
from ..operators.align_picker import (
    set_capture_callback, clear_capture_callback, was_capture_cancelled,
)

# ── Version detection ─────────────────────────────────────────────────────────
# v0.5.0.x uses -Y-up;  v0.5.1+ uses +Y-up.
# In +Y-up mode the Y and Z translation axes and the X and Z rotation axes are
# reversed relative to what the scene expects, so we flip their signs on the
# way in (_mat_from_trs) and on the way out (_decompose_mat).
def _parse_version(v: str) -> tuple:
    import re
    parts = v.lstrip("v").split(".")[:3]
    return tuple(int(re.match(r"\d+", x).group()) for x in parts)

_LFS_VER = _parse_version(lf.__version__)
Y_UP = _LFS_VER >= (0, 5, 1)   # True → +Y-up  /  False → -Y-up


# ── Helpers ───────────────────────────────────────────────────────────────────

# Sensitivity ladder used by the [-]/[+] step controls.
_STEP_LEVELS = [
    0.001, 0.002, 0.005,
    0.01,  0.02,  0.05,
    0.1,   0.2,   0.5,
    1,     2,     5,
    10,    20,    50,
    100,   200,   500,  1000,
]


def _mat_from_trs(tx, ty, tz, rx, ry, rz, sx, sy, sz):
    # In +Y-up mode, Y/Z translation and X/Z rotation are stored with the
    # opposite sign in the scene compared to what the user enters in the panel.
    if Y_UP:
        ty, tz = -ty, -tz
    rx_r = math.radians(rx); ry_r = math.radians(ry); rz_r = math.radians(rz)
    cx, sx_ = math.cos(rx_r), math.sin(rx_r)
    cy, sy_ = math.cos(ry_r), math.sin(ry_r)
    cz, sz_ = math.cos(rz_r), math.sin(rz_r)
    R = np.array([
        [ cy*cz,  cz*sx_*sy_ - cx*sz_,  cx*cz*sy_ + sx_*sz_],
        [ cy*sz_,  cx*cz + sx_*sy_*sz_, -cz*sx_ + cx*sy_*sz_],
        [-sy_,     cy*sx_,               cx*cy               ],
    ], dtype=np.float64)
    RS = R * np.array([sx, sy, sz])
    return [RS[0,0], RS[1,0], RS[2,0], 0.0,
            RS[0,1], RS[1,1], RS[2,1], 0.0,
            RS[0,2], RS[1,2], RS[2,2], 0.0,
            tx, ty, tz, 1.0]


def _decompose_mat(wt):
    M  = np.array([[wt[r][c] for c in range(4)] for r in range(4)], dtype=np.float64)
    t  = M[:3, 3]
    RS = M[:3, :3]
    sx = np.linalg.norm(RS[:, 0])
    sy = np.linalg.norm(RS[:, 1])
    sz = np.linalg.norm(RS[:, 2])
    R  = RS / np.array([sx, sy, sz])
    if abs(R[2, 0]) < 1.0 - 1e-6:
        ry = math.asin(-R[2, 0])
        rx = math.atan2(R[2, 1], R[2, 2])
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        ry = math.pi/2 if R[2, 0] < 0 else -math.pi/2
        rx = math.atan2(-R[1, 2], R[1, 1])
        rz = 0.0
    tx_, ty_, tz_ = float(t[0]), float(t[1]), float(t[2])
    rx_ = math.degrees(rx)
    ry_ = math.degrees(ry)
    rz_ = math.degrees(rz)
    # Undo the +Y-up sign flip so the panel always shows user-space values.
    if Y_UP:
        ty_, tz_ = -ty_, -tz_
    return (tx_, ty_, tz_, rx_, ry_, rz_, float(sx), float(sy), float(sz))


def _mat_to_quat(R):
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def _quat_mul_batch(q1, q2):
    w1,x1,y1,z1 = q1[0], q1[1], q1[2], q1[3]
    w2,x2,y2,z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]
    return np.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], axis=1)


def _merge_visible(name: str) -> str:
    """Merge all visible splat nodes into a single node called *name*.
    Returns an empty string on success or an error message on failure.
    """
    scene = lf.get_scene()
    if scene is None:
        return "No scene loaded."
    nodes = [n for n in scene.get_visible_nodes() if n.splat_data() is not None]
    if not nodes:
        return "No visible splat nodes to merge."
    name = name.strip() or "merged"
    try:
        group_id = scene.add_group(name)
        for n in nodes:
            scene.reparent(n.id, group_id)
        scene.merge_group(name)
        scene.invalidate_cache()
        scene.notify_changed()
        return ""
    except Exception as e:
        import traceback
        lf.log.error(f"EDIT merge error: {traceback.format_exc()}")
        return str(e)


def _unique_node_name(scene, name: str) -> str:
    """Return *name* if no node with that name exists in *scene*.
    Otherwise append an incrementing two-digit suffix (_01, _02, …).
    """
    if scene.get_node(name) is None:
        return name
    counter = 1
    while True:
        candidate = f"{name}_{counter:02d}"
        if scene.get_node(candidate) is None:
            return candidate
        counter += 1


def _bake(node_name: str) -> str:
    """Permanently write the node transform into its Gaussian data.
    Returns an empty string on success or an error message on failure.
    """
    s    = lf.get_scene()
    node = s.get_node(node_name)
    if node is None:
        return f"Node '{node_name}' not found."
    sd = node.splat_data()
    wt = node.world_transform
    M  = np.array([[wt[r][c] for c in range(4)] for r in range(4)], dtype=np.float64)
    t  = M[:3, 3]
    RS = M[:3, :3]
    sx = np.linalg.norm(RS[:, 0])
    sy = np.linalg.norm(RS[:, 1])
    sz = np.linalg.norm(RS[:, 2])
    R  = RS / np.array([sx, sy, sz])
    try:
        means = sd.means_raw.cpu().numpy().astype(np.float64)
        sd.means_raw[:] = lf.Tensor.from_numpy((means @ RS.T + t).astype(np.float32)).cuda()
        if not np.allclose([sx, sy, sz], 1.0, atol=1e-5):
            scales = sd.scaling_raw.cpu().numpy().astype(np.float64)
            sd.scaling_raw[:] = lf.Tensor.from_numpy(
                (scales + np.log([sx, sy, sz])).astype(np.float32)).cuda()
        if not np.allclose(R, np.eye(3), atol=1e-5):
            nq   = _mat_to_quat(R)
            rots = sd.rotation_raw.cpu().numpy().astype(np.float64)
            rb   = _quat_mul_batch(nq, rots).astype(np.float32)
            rb  /= np.linalg.norm(rb, axis=-1, keepdims=True)
            sd.rotation_raw[:] = lf.Tensor.from_numpy(rb).cuda()
        lf.set_node_transform(node_name, [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])
        return ""
    except Exception as e:
        import traceback
        lf.log.error(f"EDIT bake error: {traceback.format_exc()}")
        return str(e)


def _collect_splat_nodes(scene, group_node) -> list:
    """Collect all splat-bearing descendant nodes of group_node.
    node.children returns int IDs; scene.get_node() only accepts str names,
    so we build an id->node lookup from scene.get_visible_nodes() first.
    """
    # Build a complete id->node map from all visible nodes
    id_map = {n.id: n for n in scene.get_visible_nodes()}
    lf.log.info(f"EDIT bake_group: id_map has {len(id_map)} entries, "
                f"group id={group_node.id}, children={getattr(group_node, 'children', None)}")

    results = []

    def _walk(node):
        if node is None:
            return
        if node.splat_data() is not None and node.id != group_node.id:
            results.append(node)
        for child_id in (getattr(node, "children", None) or []):
            child = id_map.get(child_id)
            lf.log.info(f"EDIT bake_group: child_id={child_id} -> "
                        f"{child.name if child else 'NOT FOUND'}")
            _walk(child)

    _walk(group_node)
    return results


def _bake_group(group_name: str) -> tuple[int, list[str]]:
    """Bake world transforms of all splat nodes inside a group into their
    Gaussian data, then reset every node transform (including the group) to
    identity.  Returns (baked_count, error_list).
    """
    s     = lf.get_scene()
    group = s.get_node(group_name)
    if group is None:
        return 0, [f"Node '{group_name}' not found."]

    lf.log.info(f"EDIT bake_group: group='{group_name}' id={group.id} "
                f"has_splat={group.splat_data() is not None}")

    splat_nodes = _collect_splat_nodes(s, group)
    lf.log.info(f"EDIT bake_group: {len(splat_nodes)} splat node(s) to bake: "
                f"{[n.name for n in splat_nodes]}")

    if not splat_nodes:
        return 0, [f"No splat nodes found inside '{group_name}'."]

    errors = []
    baked  = 0
    for node in splat_nodes:
        lf.log.info(f"EDIT bake_group: baking '{node.name}'")
        err = _bake(node.name)
        if err:
            errors.append(f"{node.name}: {err}")
        else:
            baked += 1

    # Reset the group's own local transform to identity
    lf.set_node_transform(group_name, [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])

    s.invalidate_cache()
    s.notify_changed()
    return baked, errors


# ── Align helpers ─────────────────────────────────────────────────────────────

_ALIGN_AXES     = ("X-Axis", "Y-Axis", "Z-Axis")
_ALIGN_AXIS_KEYS = ("X",      "Y",      "Z")

_ALIGN_COL_X     = (1.0, 0.25, 0.25, 1.0)
_ALIGN_COL_Y     = (0.25, 1.0, 0.25, 1.0)
_ALIGN_COL_Z     = (0.25, 0.55, 1.0, 1.0)

# Module-level viewport overlay state
_align_draw_registered = False
_align_pt1_world       = None
_align_pt2_world       = None
_align_picking_which   = 0
_align_pending_pick    = None


def _calc_alignment_rotation(p1, p2, axis: str) -> tuple[float, float, float]:
    v = np.array(p2, dtype=np.float64) - np.array(p1, dtype=np.float64)
    if axis == 'X':
        angle = math.degrees(math.atan2(v[2], v[1]))
        return (-angle, 0.0, 0.0)
    elif axis == 'Y':
        angle = math.degrees(math.atan2(v[0], v[2]))
        return (0.0, -angle, 0.0)
    else:  # Z
        angle = math.degrees(math.atan2(v[1], v[0]))
        return (0.0, 0.0, -(90.0 - angle))


def _align_draw_handler(ctx):
    if _align_picking_which > 0:
        color = (0.0, 1.0, 0.5, 0.9) if _align_picking_which == 1 else (1.0, 0.8, 0.0, 0.9)
        ctx.draw_text_2d((20, 50),
                         f"ALIGN — PICK POINT {_align_picking_which}:  click on model  (ESC to cancel)",
                         color)
    if _align_pt1_world is not None:
        ctx.draw_point_3d(_align_pt1_world, (0.0, 1.0, 0.5, 1.0), 5.0)
        s1 = ctx.world_to_screen(_align_pt1_world)
        if s1:
            ctx.draw_circle_2d(s1, 10.0, (0.0, 1.0, 0.5, 1.0), 2.0)
            ctx.draw_text_2d((s1[0] + 14, s1[1] - 6), "Pt 1", (0.0, 1.0, 0.5, 1.0))
    if _align_pt2_world is not None:
        ctx.draw_point_3d(_align_pt2_world, (1.0, 0.6, 0.1, 1.0), 5.0)
        s2 = ctx.world_to_screen(_align_pt2_world)
        if s2:
            ctx.draw_circle_2d(s2, 10.0, (1.0, 0.6, 0.1, 1.0), 2.0)
            ctx.draw_text_2d((s2[0] + 14, s2[1] - 6), "Pt 2", (1.0, 0.6, 0.1, 1.0))
    if _align_pt1_world is not None and _align_pt2_world is not None:
        ctx.draw_line_3d(_align_pt1_world, _align_pt2_world, (0.8, 0.8, 0.8, 0.7), 2.0)


def _ensure_align_draw_handler():
    global _align_draw_registered
    if not _align_draw_registered:
        try:
            lf.remove_draw_handler("edit_align_overlay")
        except Exception:
            pass
        lf.add_draw_handler("edit_align_overlay", _align_draw_handler, "POST_VIEW")
        _align_draw_registered = True


def _remove_align_draw_handler():
    global _align_draw_registered
    try:
        lf.remove_draw_handler("edit_align_overlay")
    except Exception:
        pass
    _align_draw_registered = False


def _align_overlay_active() -> bool:
    """True when there is anything to draw — used to drive continuous redraws
    so overlay points correctly track the camera during pan/zoom/orbit."""
    return (_align_pt1_world is not None
            or _align_pt2_world is not None
            or _align_picking_which > 0)


def _on_align_point_picked(world_pos, point_num: int):
    global _align_pending_pick
    _align_pending_pick = (world_pos, point_num)
    lf.ui.request_redraw()


# ── Panel ─────────────────────────────────────────────────────────────────────

_T_STEP = 0.1
_R_STEP = 1.0
_S_STEP = 1.0


class TransformPanel(lf.ui.Panel):
    id                 = "edit.transform_panel"
    label              = "Edit"
    space              = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order              = 3
    template           = str(Path(__file__).resolve().with_name("transform_panel.rml"))
    height_mode        = lf.ui.PanelHeightMode.CONTENT
    update_interval_ms = 100

    def __init__(self):
        self._handle         = None
        self._node_name      = ""
        self._merge_name     = "merged"

        # ── Align sub-tool state ──────────────────────────────────────────────
        self._align_axis_idx   = 1       # default Y-Axis
        self._align_expanded   = False   # Align section collapsed by default
        self._align_pt1        = None
        self._align_pt2        = None
        self._align_picking    = 0       # 0=idle, 1=picking pt1, 2=picking pt2
        self._align_has_calc   = False
        self._align_rx         = 0.0
        self._align_ry         = 0.0
        self._align_rz         = 0.0
        self._folder_name    = "Group"
        self._move_target    = "Selection"
        self._tx = self._ty = self._tz = 0.0
        self._rx = self._ry = self._rz = 0.0
        self._sx = self._sy = self._sz = 1.0
        self._uniform_scale  = True
        self._live           = True
        self._status         = ""
        self._last_node_name = None   # dirty-detection
        self._last_logged    = None   # dedup: last transform written to session log
        self._scene_synced   = False  # guard: block _apply_to_scene until first sync

        # Slider limits — defaults; overridden by settings.json if present
        self._t_min  = -50.0
        self._t_max  =  50.0
        self._r_min  = -180.0
        self._r_max  =  180.0
        self._s_min  =  0.01
        self._s_max  =  5.0
        self._t_step =  0.1
        self._r_step =  1.0
        self._s_step =  1.0
        # Sensitivity step ladder — shared across all transform groups
        self._t_step_idx = _STEP_LEVELS.index(0.1)   if 0.1  in _STEP_LEVELS else 7
        self._r_step_idx = _STEP_LEVELS.index(1.0)   if 1.0  in _STEP_LEVELS else 9
        self._s_step_idx = _STEP_LEVELS.index(1.0)   if 1.0  in _STEP_LEVELS else 9

        self._load_settings()

    @classmethod
    def poll(cls, context) -> bool:
        return True

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("transform_panel")

        # Visibility guards
        model.bind_func("no_scene",  lambda: not lf.has_scene())
        model.bind_func("no_node",   lambda: lf.has_scene() and not self._node_name)
        model.bind_func("has_node",  lambda: bool(self._node_name))
        model.bind_func("node_name", lambda: self._node_name)

        # Live checkbox
        model.bind("live",
                   lambda: self._live,
                   self._set_live)

        # Slider limit attributes (read-only — driven from settings.json via _load_settings)
        model.bind_func("t_min",  lambda: str(self._t_min))
        model.bind_func("t_max",  lambda: str(self._t_max))
        model.bind_func("t_step", lambda: str(self._t_step))
        model.bind_func("t_step_label", lambda: str(_STEP_LEVELS[self._t_step_idx]))
        model.bind_func("r_min",  lambda: str(self._r_min))
        model.bind_func("r_max",  lambda: str(self._r_max))
        model.bind_func("r_step", lambda: str(self._r_step))
        model.bind_func("r_step_label", lambda: str(_STEP_LEVELS[self._r_step_idx]))
        model.bind_func("s_min",  lambda: str(self._s_min))
        model.bind_func("s_max",  lambda: str(self._s_max))
        model.bind_func("s_step", lambda: str(self._s_step))
        model.bind_func("s_step_label", lambda: str(_STEP_LEVELS[self._s_step_idx]))

        # Tooltip strings — built from loaded limits so they stay in sync with settings.json
        model.bind_func("tip_live",    lambda: "When enabled, every change is applied immediately. Disable to batch changes and apply with the Apply button.")
        model.bind_func("tip_uniform", lambda: "Lock X/Y/Z scale together so any axis scales all three. Uncheck to scale each axis independently.")
        model.bind_func("tip_tx",      lambda: f"Translate the node along the world X axis.  Range {self._t_min} \u2013 {self._t_max}.")
        model.bind_func("tip_ty",      lambda: f"Translate the node along the world Y axis.  Range {self._t_min} \u2013 {self._t_max}.")
        model.bind_func("tip_tz",      lambda: f"Translate the node along the world Z axis.  Range {self._t_min} \u2013 {self._t_max}.")
        model.bind_func("tip_rx",      lambda: f"Rotate the node around the world X axis (pitch).  Range {self._r_min}\u00b0 \u2013 {self._r_max}\u00b0.")
        model.bind_func("tip_ry",      lambda: f"Rotate the node around the world Y axis (yaw).    Range {self._r_min}\u00b0 \u2013 {self._r_max}\u00b0.")
        model.bind_func("tip_rz",      lambda: f"Rotate the node around the world Z axis (roll).   Range {self._r_min}\u00b0 \u2013 {self._r_max}\u00b0.")
        model.bind_func("tip_sx",      lambda: f"Scale the node along the X axis.  Range {self._s_min} \u2013 {self._s_max}.")
        model.bind_func("tip_sy",      lambda: f"Scale the node along the Y axis.  Range {self._s_min} \u2013 {self._s_max}.")
        model.bind_func("tip_sz",      lambda: f"Scale the node along the Z axis.  Range {self._s_min} \u2013 {self._s_max}.")

        # Translation
        model.bind("tx_str",
                   lambda: f"{self._tx:.3f}",
                   lambda v: self._set_trs("tx", v, self._t_min, self._t_max))
        model.bind("ty_str",
                   lambda: f"{self._ty:.3f}",
                   lambda v: self._set_trs("ty", v, self._t_min, self._t_max))
        model.bind("tz_str",
                   lambda: f"{self._tz:.3f}",
                   lambda v: self._set_trs("tz", v, self._t_min, self._t_max))

        # Rotation
        model.bind("rx_str",
                   lambda: f"{self._rx:.3f}",
                   lambda v: self._set_trs("rx", v, self._r_min, self._r_max))
        model.bind("ry_str",
                   lambda: f"{self._ry:.3f}",
                   lambda v: self._set_trs("ry", v, self._r_min, self._r_max))
        model.bind("rz_str",
                   lambda: f"{self._rz:.3f}",
                   lambda v: self._set_trs("rz", v, self._r_min, self._r_max))

        # Scale
        model.bind("uniform_scale",
                   lambda: self._uniform_scale,
                   self._set_uniform_scale)
        model.bind("sx_str",
                   lambda: f"{self._sx:.3f}",
                   lambda v: self._set_trs("sx", v, self._s_min, self._s_max))
        model.bind("sy_str",
                   lambda: f"{self._sy:.3f}",
                   lambda v: self._set_trs("sy", v, self._s_min, self._s_max))
        model.bind("sz_str",
                   lambda: f"{self._sz:.3f}",
                   lambda v: self._set_trs("sz", v, self._s_min, self._s_max))

        # Text inputs
        model.bind("merge_name",
                   lambda: self._merge_name,
                   lambda v: (setattr(self, "_merge_name", str(v)), self._save_settings()))
        model.bind("folder_name",
                   lambda: self._folder_name,
                   lambda v: (setattr(self, "_folder_name", str(v)), self._save_settings()))
        model.bind("move_target",
                   lambda: self._move_target,
                   lambda v: (setattr(self, "_move_target", str(v)), self._save_settings()))

        # Status
        model.bind_func("status_text",  lambda: self._status)
        model.bind_func("status_class", self._status_class)

        # ── Align bindings ────────────────────────────────────────────────────
        model.bind_func("align_expanded",     lambda: self._align_expanded)
        model.bind_func("align_section_label",lambda: "▼ Align" if self._align_expanded else "▶ Align")
        model.bind_event("align_toggle",      self._on_align_toggle)
        model.bind_func("align_axis_idx",   lambda: str(self._align_axis_idx))
        model.bind_func("align_btn_x", lambda: "[X]" if self._align_axis_idx == 0 else "X")
        model.bind_func("align_btn_y", lambda: "[Y]" if self._align_axis_idx == 1 else "Y")
        model.bind_func("align_btn_z", lambda: "[Z]" if self._align_axis_idx == 2 else "Z")
        model.bind_func("align_pt1_label",  lambda: (
            f"Pt 1:  {self._align_pt1[0]:.4f}  {self._align_pt1[1]:.4f}  {self._align_pt1[2]:.4f}"
            if self._align_pt1 else "Pt 1:  (not set)"))
        model.bind_func("align_pt2_label",  lambda: (
            f"Pt 2:  {self._align_pt2[0]:.4f}  {self._align_pt2[1]:.4f}  {self._align_pt2[2]:.4f}"
            if self._align_pt2 else "Pt 2:  (not set)"))
        model.bind_func("align_result_rx",  lambda: f"Rx  {self._align_rx:+.3f}°")
        model.bind_func("align_result_ry",  lambda: f"Ry  {self._align_ry:+.3f}°")
        model.bind_func("align_result_rz",  lambda: f"Rz  {self._align_rz:+.3f}°")
        model.bind_func("align_has_calc",   lambda: self._align_has_calc)
        model.bind_func("align_pick1_label",lambda: (
            "◉ Picking Pt 1…" if self._align_picking == 1 else
            ("Repick Pt 1" if self._align_pt1 else "Pick Point 1")))
        model.bind_func("align_pick2_label",lambda: (
            "◉ Picking Pt 2…" if self._align_picking == 2 else
            ("Repick Pt 2" if self._align_pt2 else "Pick Point 2")))
        model.bind_func("align_picking_active", lambda: self._align_picking > 0)
        model.bind_func("align_can_calc",   lambda: (
            self._align_pt1 is not None and self._align_pt2 is not None))

        model.bind_event("align_axis_x",     self._on_align_axis_x)
        model.bind_event("align_axis_y",     self._on_align_axis_y)
        model.bind_event("align_axis_z",     self._on_align_axis_z)
        model.bind_event("align_pick1",      self._on_align_pick1)
        model.bind_event("align_pick2",      self._on_align_pick2)
        model.bind_event("align_calc",       self._on_align_calc)

        # Events
        model.bind_event("do_refresh",         self._on_refresh)
        model.bind_event("do_grab",            self._on_grab)
        model.bind_event("do_apply",           self._on_apply)
        model.bind_event("do_reset",           self._on_reset)
        model.bind_event("do_bake",            self._on_bake)
        model.bind_event("do_merge",           self._on_merge)
        model.bind_event("do_create_folder",   self._on_create_folder)
        model.bind_event("do_move",            self._on_move)
        model.bind_event("num_step",           self._on_num_step)
        model.bind_event("step_sensitivity",   self._on_step_sensitivity)
        model.bind_event("do_reload_settings",  self._on_reload_settings)
        model.bind_event("do_open_log",         self._on_open_log)
        model.bind_event("do_open_settings",    self._on_open_settings)
        model.bind_event("do_recenter_xyz",     self._on_recenter_xyz)
        model.bind_event("do_recenter_xz_0y",  self._on_recenter_xz_0y)

        self._handle = model.get_handle()
        self._sync_from_scene()
        # Fire reset x3 to ensure scale sliders settle at 1,1,1
        self._uniform_scale = False  
        self._do_scale_reset()       
        self._do_scale_reset()
        self._do_scale_reset()     
        self._uniform_scale = True
        self._uniform_scale = False  
        self._do_scale_reset()       
        self._do_scale_reset()
        self._do_scale_reset()     
        self._uniform_scale = True


    def on_update(self, doc):
        changed = self._process_align_picks()
        if _align_overlay_active():
            try:
                lf.ui.request_redraw()
            except Exception:
                pass
        current = lf.get_selected_node_name() if lf.has_scene() else ""
        if current != self._last_node_name:
            self._last_node_name = current
            self._sync_from_scene()
            self._dirty_all()
            return True
        return changed

    def on_unmount(self, doc):
        doc.remove_data_model("transform_panel")
        self._handle = None
        self._scene_synced = False

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_refresh(self, handle, event, args):
        self._sync_from_scene()
        self._dirty_all()

    def _on_reload_settings(self, handle, event, args):
        self._load_settings()
        self._status = "Settings reloaded from settings.json."
        self._dirty_all()

    def _open_in_editor(self, path: Path) -> str:
        """Open *path* in Notepad++ or fall back to Notepad. Returns a status string."""
        npp_candidates = [
            r"C:\Program Files\Notepad++\notepad++.exe",
            r"C:\Program Files (x86)\Notepad++\notepad++.exe",
        ]
        npp = next((p for p in npp_candidates if Path(p).exists()), None)
        if npp:
            subprocess.Popen([npp, str(path)])
            return f"Opened {path.name} in Notepad++."
        else:
            subprocess.Popen(["notepad.exe", str(path)])
            return f"Notepad++ not found — opened {path.name} in Notepad."

    def _on_open_log(self, handle, event, args):
        log_path = self._log_path()
        try:
            if not log_path.exists():
                log_path.write_text("[]", encoding="utf-8")
            self._status = self._open_in_editor(log_path)
        except Exception as e:
            self._status = f"Could not open log: {e}"
        self._dirty("status_text", "status_class")

    def _on_open_settings(self, handle, event, args):
        settings_path = self._settings_path()
        try:
            if not settings_path.exists():
                self._save_settings()
            self._status = self._open_in_editor(settings_path)
        except Exception as e:
            self._status = f"Could not open settings: {e}"
        self._dirty("status_text", "status_class")

    def _on_recenter_xyz(self, handle, event, args):
        """Prefill translation fields with values that would move the bounding-box
        centroid to the world origin (0, 0, 0). Does NOT apply the transform."""
        if not self._node_name or not lf.has_scene():
            self._status = "No node selected."
            self._dirty("status_text", "status_class")
            return
        try:
            scene = lf.get_scene()
            mn, mx = scene.get_node_bounds(self._node_name)
            mn = np.array(mn, dtype=np.float64)
            mx = np.array(mx, dtype=np.float64)
            centroid = (mn + mx) / 2.0
            # Prefill — keep rotation and scale unchanged
            # _mat_from_trs negates tz again in Y_UP mode, so pre-flip to compensate
            tz_sign = 1.0 if Y_UP else -1.0
            self._tx = float(np.clip(-centroid[0], self._t_min, self._t_max))
            self._ty = - float(np.clip(-centroid[1], self._t_min, self._t_max))
            self._tz = float(np.clip(tz_sign * centroid[2], self._t_min, self._t_max))
            self._status = (f"Prefilled: X={self._tx:.4f}  Y={self._ty:.4f}  "
                            f"Z={self._tz:.4f}  (centroid → 0,0,0 — press Apply to confirm)")
            if self._live:
                self._apply_to_scene()
            self._dirty("tx_str", "ty_str", "tz_str", "status_text", "status_class")
        except Exception as e:
            self._status = f"ReCent XYZ error: {e}"
            self._dirty("status_text", "status_class")

    def _on_recenter_xz_0y(self, handle, event, args):
        """Prefill translation fields so the model is centred on X/Z and its
        top face is placed at Y=0 (floor-at-origin convention).
        Does NOT apply the transform."""
        if not self._node_name or not lf.has_scene():
            self._status = "No node selected."
            self._dirty("status_text", "status_class")
            return
        try:
            scene = lf.get_scene()
            mn, mx = scene.get_node_bounds(self._node_name)
            mn = np.array(mn, dtype=np.float64)
            mx = np.array(mx, dtype=np.float64)
            centre_x = (mn[0] + mx[0]) / 2.0
            centre_z = (mn[2] + mx[2]) / 2.0
            floor_y  = mx[1]  # top edge → Y=0
            # _mat_from_trs negates tz again in Y_UP mode, so pre-flip to compensate
            tz_sign = 1.0 if Y_UP else -1.0
            self._tx = float(np.clip(-centre_x, self._t_min, self._t_max))
            self._ty = - float(np.clip(-floor_y,  self._t_min, self._t_max))
            self._tz = float(np.clip(tz_sign * centre_z, self._t_min, self._t_max))
            self._status = (f"Prefilled: X={self._tx:.4f}  Y={self._ty:.4f}  "
                            f"Z={self._tz:.4f}  (floor at Y=0, centred X/Z — press Apply to confirm)")
            if self._live:
                self._apply_to_scene()
            self._dirty("tx_str", "ty_str", "tz_str", "status_text", "status_class")
        except Exception as e:
            self._status = f"ReCent XZ error: {e}"
            self._dirty("status_text", "status_class")

    def _on_step_sensitivity(self, handle, event, args):
        """Walk the step ladder for a transform group (t / r / s)."""
        if not args or len(args) < 2:
            return
        group     = str(args[0])   # "t", "r", or "s"
        direction = int(args[1])   # -1 or +1
        idx_attr  = f"_{group}_step_idx"
        step_attr = f"_{group}_step"
        if not hasattr(self, idx_attr):
            return
        idx = getattr(self, idx_attr)
        idx = max(0, min(len(_STEP_LEVELS) - 1, idx + direction))
        setattr(self, idx_attr, idx)
        setattr(self, step_attr, _STEP_LEVELS[idx])
        # Also persist to settings
        self._save_settings()
        self._dirty(f"{group}_step", f"{group}_step_label")

    def _do_scale_reset(self):
        """Force scale to 1,1,1 and update the scene if live."""
        self._sx = self._sy = self._sz = 1.0
        if self._live and self._node_name and self._scene_synced:
            self._apply_to_scene()
        self._dirty("sx_str", "sy_str", "sz_str")

    def _on_grab(self, handle, event, args):
        self._sync_from_scene()
        self._log_transform("grab")
        self._dirty_all()

    def _on_apply(self, handle, event, args):
        self._apply_to_scene()
        self._log_transform("apply")
        self._status = "Applied."
        self._dirty("status_text", "status_class")

    def _on_reset(self, handle, event, args):
        self._uniform_scale = False 
        self._tx = self._ty = self._tz = 0.0
        self._rx = self._ry = self._rz = 0.0
        # x3 to try & solve sticky issue
        self._sx = self._sy = self._sz = 1.0
        self._apply_to_scene()
        self._sx = self._sy = self._sz = 1.0
        self._apply_to_scene()
        self._sx = self._sy = self._sz = 1.0
        self._apply_to_scene()
        self._uniform_scale = True
        
        self._log_transform("Input.Reset")
        self._status = "Reset to identity."
        self._save_settings()
        self._dirty("tx_str", "ty_str", "tz_str",
                    "rx_str", "ry_str", "rz_str",
                    "sx_str", "sy_str", "sz_str",
                    "status_text", "status_class")

    def _on_bake(self, handle, event, args):
        self._apply_to_scene()
        self._log_transform("bake")
        self._clear_align_points()
        node = lf.get_scene().get_node(self._node_name) if lf.has_scene() else None
        if node is None:
            self._status = "Bake failed: node not found."
            self._dirty_all()
            return

        is_group = node.splat_data() is None  # group nodes carry no splat data themselves

        if is_group:
            baked, errors = _bake_group(self._node_name)
            if errors and baked == 0:
                self._status = f"Bake failed: {errors[0]}"
            elif errors:
                self._status = (f"Baked {baked} node(s); {len(errors)} error(s): "
                                f"{errors[0]}")
            else:
                self._status = (f"Baked {baked} node(s) in group "
                                f"'{self._node_name}' — transforms reset to identity.")
            self._sync_from_scene()
        else:
            err = _bake(self._node_name)
            if err:
                self._status = f"Bake failed: {err}"
            else:
                self._status = "Baked — transform reset to identity."
                self._sync_from_scene()
        self._dirty_all()

    def _on_merge(self, handle, event, args):
        err = _merge_visible(self._merge_name)
        if err:
            self._status = f"Merge failed: {err}"
        else:
            name = self._merge_name.strip() or "merged"
            self._status = f"Merged visible nodes into '{name}'."
            self._sync_from_scene()
        self._dirty("status_text", "status_class")

    def _on_create_folder(self, handle, event, args):
        name = self._folder_name.strip() or "Group"
        try:
            scene = lf.get_scene()
            if scene is None:
                self._status = "No scene loaded."
            else:
                scene.add_group(name)
                scene.notify_changed()
                self._status = f"Created group '{name}'."
        except Exception as e:
            self._status = f"Create group failed: {e}"
        self._dirty("status_text", "status_class")

    def _on_move(self, handle, event, args):
        target_name = self._move_target.strip()
        if not target_name:
            self._status = "Enter a target node name first."
            self._dirty("status_text", "status_class")
            return
        scene = lf.get_scene()
        if scene is None:
            self._status = "No scene loaded."
            self._dirty("status_text", "status_class")
            return
        if target_name != self._node_name:
            unique_target = _unique_node_name(scene, target_name)
            if unique_target != target_name:
                self._status = (
                    f"'{target_name}' already exists — "
                    f"using '{unique_target}' instead."
                )
                target_name       = unique_target
                self._move_target = target_name
                self._dirty("move_target", "status_text", "status_class")
        err = self._move_selected_splats(self._node_name, target_name)
        if err:
            self._status = f"Move failed: {err}"
        else:
            self._status     = f"Moved selected splats \u2192 '{target_name}'."
            self._move_target = "Selection"
            self._sync_from_scene()
        self._dirty_all()

    def _on_num_step(self, handle, event, args):
        if not args or len(args) < 2:
            return
        field     = str(args[0])
        direction = int(args[1])

        steps  = dict(tx=self._t_step, ty=self._t_step, tz=self._t_step,
                      rx=self._r_step, ry=self._r_step, rz=self._r_step,
                      sx=self._s_step, sy=self._s_step, sz=self._s_step)
        ranges = dict(tx=(self._t_min, self._t_max),
                      ty=(self._t_min, self._t_max),
                      tz=(self._t_min, self._t_max),
                      rx=(self._r_min, self._r_max),
                      ry=(self._r_min, self._r_max),
                      rz=(self._r_min, self._r_max),
                      sx=(self._s_min, self._s_max),
                      sy=(self._s_min, self._s_max),
                      sz=(self._s_min, self._s_max))
        if field not in steps:
            return

        lo, hi  = ranges[field]
        current = getattr(self, f"_{field}")
        new_val = round(max(lo, min(hi, current + direction * steps[field])), 4)
        if abs(new_val - current) < 1e-9:
            return
        setattr(self, f"_{field}", new_val)

        if field == "sx" and self._uniform_scale:
            self._sy = self._sz = new_val
            self._dirty("sy_str", "sz_str")

        if self._live:
            self._apply_to_scene()
        self._dirty(f"{field}_str")
        self._save_settings()

    # ── Setters ───────────────────────────────────────────────────────────────

    def _set_live(self, value):
        if isinstance(value, str):
            self._live = value.lower() not in ("false", "0", "")
        else:
            self._live = bool(value)
        self._save_settings()

    def _set_uniform_scale(self, value):
        if isinstance(value, str):
            self._uniform_scale = value.lower() not in ("false", "0", "")
        else:
            self._uniform_scale = bool(value)
        self._save_settings()

    def _set_trs(self, attr: str, value, lo: float, hi: float):
        try:
            v = max(lo, min(hi, float(value)))
        except (TypeError, ValueError):
            return
        if abs(v - getattr(self, f"_{attr}")) < 1e-9:
            return
        setattr(self, f"_{attr}", v)

        if attr == "sx" and self._uniform_scale:
            self._sy = self._sz = v
            self._dirty("sy_str", "sz_str")

        if self._live:
            self._apply_to_scene()
        self._dirty(f"{attr}_str")
        self._save_settings()

    # ── Scene sync ────────────────────────────────────────────────────────────

    def _sync_from_scene(self):
        try:
            name = lf.get_selected_node_name() if lf.has_scene() else ""
            if not name:
                self._node_name = ""
                return
            node = lf.get_scene().get_node(name)
            if node is None:
                return
            self._node_name = name
            (self._tx, self._ty, self._tz,
             self._rx, self._ry, self._rz,
             _, _, _) = _decompose_mat(node.world_transform)
            # Scale always starts at 1,1,1 on load regardless of scene state
            self._sx = self._sy = self._sz = 1.0
            self._scene_synced = True
            lf.log.info(f"EDIT synced: t=({self._tx:.3f},{self._ty:.3f},{self._tz:.3f}) "
                        f"r=({self._rx:.1f},{self._ry:.1f},{self._rz:.1f}) "
                        f"s=({self._sx:.3f},{self._sy:.3f},{self._sz:.3f})")
        except Exception as e:
            self._status = f"Sync error: {e}"

    def _apply_to_scene(self):
        if not self._node_name or not self._scene_synced:
            return
        try:
            mat = _mat_from_trs(self._tx, self._ty, self._tz,
                                 self._rx, self._ry, self._rz,
                                 self._sx, self._sy, self._sz)
            lf.set_node_transform(self._node_name, mat)
        except Exception as e:
            self._status = f"Apply error: {e}"

    def _move_selected_splats(self, source_name: str, target_name: str) -> str:
        """Move splats by manually calculating the node's offset in the global mask."""
        try:
            scene       = lf.get_scene()
            global_mask = scene.selection_mask
            if global_mask is None or not lf.has_selection():
                return "Nothing selected."

            visible_splat_nodes = [n for n in scene.get_visible_nodes()
                                   if n.splat_data() is not None]
            start_idx  = 0
            found_node = None
            for n in visible_splat_nodes:
                if n.name == source_name:
                    found_node = n
                    break
                start_idx += n.splat_data().num_points

            if not found_node:
                return f"Source node '{source_name}' not found (or not visible)."

            num_points        = found_node.splat_data().num_points
            local_mask_tensor = global_mask[start_idx : start_idx + num_points]
            mask_np           = local_mask_tensor.cpu().numpy().astype(bool)
            selected_count    = int(mask_np.sum())

            lf.log.info(f"EDIT move: source='{source_name}' total={num_points} "
                        f"selected={selected_count} start_idx={start_idx}")
            if selected_count == 0:
                return "No splats selected in this node."

            scene.clear_selection()

            source_sd    = found_node.splat_data()
            selected_idx = np.where(mask_np)[0]
            inlier_idx   = np.where(~mask_np)[0]

            def _gather(tensor, idx):
                return lf.Tensor.from_numpy(tensor.cpu().numpy()[idx]).cuda()

            sel_means    = _gather(source_sd.means_raw,    selected_idx)
            sel_sh0      = _gather(source_sd.sh0_raw,      selected_idx)
            sel_shN      = _gather(source_sd.shN_raw,      selected_idx)
            sel_scaling  = _gather(source_sd.scaling_raw,  selected_idx)
            sel_rotation = _gather(source_sd.rotation_raw, selected_idx)
            sel_opacity  = _gather(source_sd.opacity_raw,  selected_idx)

            inlier_means    = _gather(source_sd.means_raw,    inlier_idx)
            inlier_sh0      = _gather(source_sd.sh0_raw,      inlier_idx)
            inlier_shN      = _gather(source_sd.shN_raw,      inlier_idx)
            inlier_scaling  = _gather(source_sd.scaling_raw,  inlier_idx)
            inlier_rotation = _gather(source_sd.rotation_raw, inlier_idx)
            inlier_opacity  = _gather(source_sd.opacity_raw,  inlier_idx)

            active_sh = source_sd.active_sh_degree
            s_scale   = source_sd.scene_scale

            target_node = scene.get_node(target_name)
            if target_node is not None:
                target_sd = target_node.splat_data()
                def _cat(a, b):
                    return lf.Tensor.from_numpy(
                        np.concatenate([a.cpu().numpy(), b.cpu().numpy()], axis=0)
                    ).cuda()
                scene.remove_node(target_name)
                scene.add_splat(
                    target_name,
                    _cat(target_sd.means_raw,    sel_means),
                    _cat(target_sd.sh0_raw,      sel_sh0),
                    _cat(target_sd.shN_raw,      sel_shN),
                    _cat(target_sd.scaling_raw,  sel_scaling),
                    _cat(target_sd.rotation_raw, sel_rotation),
                    _cat(target_sd.opacity_raw,  sel_opacity),
                    active_sh, s_scale,
                )
            else:
                scene.add_splat(
                    target_name,
                    sel_means, sel_sh0, sel_shN,
                    sel_scaling, sel_rotation, sel_opacity,
                    active_sh, s_scale,
                )

            lf.log.info(f"EDIT move: target='{target_name}' built with {len(selected_idx)} splats")

            scene.remove_node(source_name)
            scene.add_splat(
                source_name,
                inlier_means, inlier_sh0, inlier_shN,
                inlier_scaling, inlier_rotation, inlier_opacity,
                active_sh, s_scale,
            )

            lf.log.info(f"EDIT move: source='{source_name}' rebuilt with {len(inlier_idx)} splats")

            scene.invalidate_cache()
            scene.notify_changed()
            return ""

        except Exception as e:
            import traceback
            lf.log.error(f"EDIT move error: {traceback.format_exc()}")
            return str(e)

    # ── Settings persistence ──────────────────────────────────────────────────

    @staticmethod
    def _settings_path() -> Path:
        return Path(__file__).resolve().parent.parent / "settings.json"

    @staticmethod
    def _log_path() -> Path:
        return Path(__file__).resolve().parent.parent / "session_log.json"

    def _log_transform(self, action: str = "apply"):
        """Append one transform entry to session_log.json, skipping exact duplicates."""
        try:
            snapshot = (
                action,
                self._node_name,
                round(self._tx, 4), round(self._ty, 4), round(self._tz, 4),
                round(self._rx, 4), round(self._ry, 4), round(self._rz, 4),
                round(self._sx, 4), round(self._sy, 4), round(self._sz, 4),
                self._uniform_scale, self._live,
            )
            if snapshot == self._last_logged:
                return
            self._last_logged = snapshot

            path = self._log_path()
            try:
                log = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(log, list):
                    log = []
            except Exception:
                log = []

            entry = {
                "time":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "action": action,
                "node":   self._node_name,
                "transform": {
                    "tx": round(self._tx, 4),
                    "ty": round(self._ty, 4),
                    "tz": round(self._tz, 4),
                    "rx": round(self._rx, 4),
                    "ry": round(self._ry, 4),
                    "rz": round(self._rz, 4),
                    "sx": round(self._sx, 4),
                    "sy": round(self._sy, 4),
                    "sz": round(self._sz, 4),
                    "uniform_scale": self._uniform_scale,
                    "live":          self._live,
                },
            }
            log.append(entry)
            path.write_text(json.dumps(log, indent=2), encoding="utf-8")
        except Exception as e:
            lf.log.error(f"EDIT session log error: {e}")

    def _load_settings(self):
        try:
            data = json.loads(self._settings_path().read_text(encoding="utf-8"))
            t = data.get("transform", {})
            self._tx            = float(t.get("tx",            self._tx))
            self._ty            = float(t.get("ty",            self._ty))
            self._tz            = float(t.get("tz",            self._tz))
            self._rx            = float(t.get("rx",            self._rx))
            self._ry            = float(t.get("ry",            self._ry))
            self._rz            = float(t.get("rz",            self._rz))
            self._sx = float(t.get("sx", self._sx))
            self._sy = float(t.get("sy", self._sy))
            self._sz = float(t.get("sz", self._sz))
            # Booleans: use explicit `is True` comparison to correctly read
            # JSON false (Python False) rather than truthy/falsy coercion.
            us = t.get("uniform_scale", self._uniform_scale)
            self._uniform_scale = us if isinstance(us, bool) else bool(us)
            lv = t.get("live", self._live)
            self._live          = lv if isinstance(lv, bool) else bool(lv)
            self._merge_name    = str(t.get("merge_name",      self._merge_name))
            self._folder_name   = str(t.get("folder_name",     self._folder_name))
            self._move_target   = str(t.get("move_target",     self._move_target))

            # Slider limits & steps from settings.json
            lim = data.get("limits", {})
            self._t_min  = float(lim.get("translation_min",  self._t_min))
            self._t_max  = float(lim.get("translation_max",  self._t_max))
            self._r_min  = float(lim.get("rotation_min",     self._r_min))
            self._r_max  = float(lim.get("rotation_max",     self._r_max))
            self._s_min  = float(lim.get("scale_min",        self._s_min))
            self._s_max  = float(lim.get("scale_max",        self._s_max))
            self._t_step = float(lim.get("translation_step", self._t_step))
            self._r_step = float(lim.get("rotation_step",    self._r_step))
            saved_s_step = float(lim.get("scale_step", self._s_step))
            # If settings.json has a stale sub-1.0 scale step, reset to 1.0
            self._s_step = saved_s_step if saved_s_step >= 1.0 else 1.0
            # Restore step indices to match loaded step values
            self._t_step_idx = min(range(len(_STEP_LEVELS)), key=lambda i: abs(_STEP_LEVELS[i] - self._t_step))
            self._r_step_idx = min(range(len(_STEP_LEVELS)), key=lambda i: abs(_STEP_LEVELS[i] - self._r_step))
            self._s_step_idx = min(range(len(_STEP_LEVELS)), key=lambda i: abs(_STEP_LEVELS[i] - self._s_step))
        except FileNotFoundError:
            pass  # first run — file will be created on first save
        except Exception as e:
            lf.log.error(f"EDIT settings load error: {e}")

    def _save_settings(self):
        try:
            path = self._settings_path()
            # Preserve any existing top-level keys (e.g. load_on_startup)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            data["transform"] = {
                "tx":            round(self._tx, 4),
                "ty":            round(self._ty, 4),
                "tz":            round(self._tz, 4),
                "rx":            round(self._rx, 4),
                "ry":            round(self._ry, 4),
                "rz":            round(self._rz, 4),
                "sx":            round(self._sx, 4),
                "sy":            round(self._sy, 4),
                "sz":            round(self._sz, 4),
                "uniform_scale": self._uniform_scale,
                "live":          self._live,
                "merge_name":    self._merge_name,
                "folder_name":   self._folder_name,
                "move_target":   self._move_target,
            }
            data["limits"] = {
                "translation_min":  self._t_min,
                "translation_max":  self._t_max,
                "rotation_min":     self._r_min,
                "rotation_max":     self._r_max,
                "scale_min":        self._s_min,
                "scale_max":        self._s_max,
                "translation_step": self._t_step,
                "rotation_step":    self._r_step,
                "scale_step":       self._s_step,
            }
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            lf.log.error(f"EDIT settings save error: {e}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _dirty(self, *fields):
        if not self._handle:
            return
        for f in fields:
            self._handle.dirty(f)

    def _dirty_all(self):
        self._dirty("no_scene", "no_node", "has_node", "node_name",
                    "tx_str", "ty_str", "tz_str",
                    "rx_str", "ry_str", "rz_str",
                    "sx_str", "sy_str", "sz_str",
                    "live", "uniform_scale",
                    "t_min", "t_max", "t_step", "t_step_label",
                    "r_min", "r_max", "r_step", "r_step_label",
                    "s_min", "s_max", "s_step", "s_step_label",
                    "tip_live", "tip_uniform",
                    "tip_tx", "tip_ty", "tip_tz",
                    "tip_rx", "tip_ry", "tip_rz",
                    "tip_sx", "tip_sy", "tip_sz",
                    "merge_name", "folder_name", "move_target",
                    "align_expanded", "align_section_label",
                    "align_picking_active",
                    "align_axis_idx", "align_btn_x", "align_btn_y", "align_btn_z",
                    "align_pt1_label", "align_pt2_label",
                    "align_result_rx", "align_result_ry", "align_result_rz",
                    "align_has_calc", "align_pick1_label", "align_pick2_label",
                    "align_can_calc",
                    "status_text", "status_class")

    def _status_class(self) -> str:
        s = self._status
        if any(w in s for w in ("Moved", "Merged", "Created", "Applied",
                                "Reset", "Baked", "Synced")):
            return "text-accent"
        if s and ("failed" in s.lower() or "error" in s.lower()):
            return "text-muted"
        return "text-default"


    # ── Align helpers ─────────────────────────────────────────────────────────

    def _clear_align_points(self):
        global _align_pt1_world, _align_pt2_world, _align_picking_which
        self._align_pt1       = None
        self._align_pt2       = None
        self._align_has_calc  = False
        self._align_picking   = 0
        _align_pt1_world      = None
        _align_pt2_world      = None
        _align_picking_which  = 0
        clear_capture_callback()
        self._dirty("align_pt1_label", "align_pt2_label",
                    "align_pick1_label", "align_pick2_label", "align_has_calc")
        lf.ui.request_redraw()

    # ── Align event handlers ──────────────────────────────────────────────────

    def _on_align_toggle(self, handle, event, args):
        self._align_expanded = not self._align_expanded
        self._dirty("align_expanded", "align_section_label")

    def _on_align_axis_x(self, handle, event, args):
        self._align_axis_idx = 0
        self._align_has_calc = False
        self._dirty("align_btn_x", "align_btn_y", "align_btn_z", "align_has_calc")

    def _on_align_axis_y(self, handle, event, args):
        self._align_axis_idx = 1
        self._align_has_calc = False
        self._dirty("align_btn_x", "align_btn_y", "align_btn_z", "align_has_calc")

    def _on_align_axis_z(self, handle, event, args):
        self._align_axis_idx = 2
        self._align_has_calc = False
        self._dirty("align_btn_x", "align_btn_y", "align_btn_z", "align_has_calc")

    def _on_align_pick1(self, handle, event, args):
        global _align_picking_which
        _ensure_align_draw_handler()
        if self._align_picking == 1:
            self._align_picking = 0
            _align_picking_which = 0
            clear_capture_callback()
            try:
                lf.ui.ops.cancel_modal()
            except Exception:
                pass
            self._status = "Pick cancelled."
        else:
            self._align_picking = 1
            _align_picking_which = 1
            self._align_has_calc = False
            self._status = "Click on the model to pick Point 1…  (ESC to cancel)"
            set_capture_callback(_on_align_point_picked, 1)
            lf.ui.ops.invoke("lfs_plugins.EDIT.operators.align_picker.ALIGN_OT_pick_point")
        self._dirty("align_pick1_label", "align_pick2_label",
                    "align_has_calc", "status_text", "status_class")
        lf.ui.request_redraw()

    def _on_align_pick2(self, handle, event, args):
        global _align_picking_which
        _ensure_align_draw_handler()
        if self._align_picking == 2:
            self._align_picking = 0
            _align_picking_which = 0
            clear_capture_callback()
            try:
                lf.ui.ops.cancel_modal()
            except Exception:
                pass
            self._status = "Pick cancelled."
        else:
            self._align_picking = 2
            _align_picking_which = 2
            self._align_has_calc = False
            self._status = "Click on the model to pick Point 2…  (ESC to cancel)"
            set_capture_callback(_on_align_point_picked, 2)
            lf.ui.ops.invoke("lfs_plugins.EDIT.operators.align_picker.ALIGN_OT_pick_point")
        self._dirty("align_pick1_label", "align_pick2_label",
                    "align_has_calc", "status_text", "status_class")
        lf.ui.request_redraw()

    def _on_align_calc(self, handle, event, args):
        if self._align_pt1 is None or self._align_pt2 is None:
            self._status = "Pick both points first."
            self._dirty("status_text", "status_class")
            return
        ax = _ALIGN_AXIS_KEYS[self._align_axis_idx]
        try:
            rx, ry, rz = _calc_alignment_rotation(self._align_pt1, self._align_pt2, ax)
            self._align_rx = rx
            self._align_ry = ry
            self._align_rz = rz
            self._align_has_calc = True
            # Write the delta directly into the rotation fields
            self._rx = max(self._r_min, min(self._r_max, self._rx + rx))
            self._ry = - max(self._r_min, min(self._r_max, self._ry + ry))
            self._rz = - max(self._r_min, min(self._r_max, self._rz + rz))
            self._status = (
                f"Angle calculated — Rx {rx:+.3f}°  Ry {ry:+.3f}°  Rz {rz:+.3f}°  "
                f"— values added to Rotation fields. Press Apply (or Bake) to commit."
            )
            if self._live:
                self._apply_to_scene()
            self._dirty("align_result_rx", "align_result_ry", "align_result_rz",
                        "align_has_calc", "rx_str", "ry_str", "rz_str",
                        "status_text", "status_class")
        except Exception as e:
            self._status = f"Calc angle error: {e}"
            self._dirty("status_text", "status_class")

    def _process_align_picks(self):
        """Drain the pending pick queue. Called from on_update."""
        global _align_pending_pick, _align_pt1_world, _align_pt2_world, _align_picking_which
        if _align_pending_pick is not None:
            world_pos, point_num = _align_pending_pick
            _align_pending_pick = None
            if point_num == 1:
                self._align_pt1 = world_pos
                _align_pt1_world = world_pos
                self._status = "Point 1 set — now pick Point 2."
            else:
                self._align_pt2 = world_pos
                _align_pt2_world = world_pos
                self._status = "Point 2 set — press [Calc. Angle] to calculate."
            self._align_picking = 0
            _align_picking_which = 0
            self._align_has_calc = False
            self._dirty("align_pt1_label", "align_pt2_label",
                        "align_pick1_label", "align_pick2_label",
                        "align_has_calc", "status_text", "status_class")
            lf.ui.request_redraw()
            return True

        if was_capture_cancelled() and self._align_picking > 0:
            self._align_picking = 0
            _align_picking_which = 0
            self._status = "Pick cancelled."
            self._dirty("align_pick1_label", "align_pick2_label",
                        "status_text", "status_class")
            lf.ui.request_redraw()
            return True
        return False
