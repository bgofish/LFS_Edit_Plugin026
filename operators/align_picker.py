# SPDX-License-Identifier: GPL-3.0-or-later
"""Modal point-picker operator for the Align plugin (embedded in LFS_Edit_Plugin)."""

import lichtfeld as lf
import lichtfeld.selection as sel
from lfs_plugins.types import Operator, Event

# Module-level callback state
_pick_callback  = None
_pick_point_num = 0
_pick_cancelled = False

# How many MOUSEMOVE events to wait after a LEFTMOUSE PRESS before retrying
# a failed pick — gives the render pipeline time to resolve depth.
_PICK_RETRY_TICKS = 3
_pending_pick_xy  = None   # (x, y) waiting for retry, or None
_pending_ticks    = 0


def set_pick_callback(callback, point_num: int):
    global _pick_callback, _pick_point_num, _pick_cancelled
    global _pending_pick_xy, _pending_ticks
    _pick_callback   = callback
    _pick_point_num  = point_num
    _pick_cancelled  = False
    _pending_pick_xy = None
    _pending_ticks   = 0


def clear_pick_callback():
    global _pick_callback, _pick_point_num, _pick_cancelled
    global _pending_pick_xy, _pending_ticks
    _pick_callback   = None
    _pick_point_num  = 0
    _pick_cancelled  = True
    _pending_pick_xy = None
    _pending_ticks   = 0


def was_pick_cancelled() -> bool:
    global _pick_cancelled
    if _pick_cancelled:
        _pick_cancelled = False
        return True
    return False


def _try_pick(x: float, y: float) -> bool:
    """Attempt a depth pick at (x, y).  Returns True and fires the callback on
    a hit, False on a miss (caller should retry)."""
    global _pick_callback, _pick_point_num
    if _pick_callback is None:
        return True  # callback was cleared externally — treat as done
    try:
        result = sel.pick_at_screen(x, y)
    except Exception as exc:
        lf.log.warning(f"EDIT Align pick_at_screen raised: {exc}")
        return False
    if result is None:
        return False
    _pick_callback(result.world_position, _pick_point_num)
    clear_pick_callback()
    return True


class ALIGN_OT_pick_point(Operator):
    """Modal operator: click on the viewport to pick a world-space point."""

    label       = "Pick Alignment Point"
    description = "Click on the model to pick a point for alignment"
    options     = {'BLOCKING'}

    def invoke(self, context, event: Event) -> set:
        # Register this operator as a modal handler so modal() receives events.
        # LFS may expose this on context.window_manager (Blender-style) or directly.
        try:
            context.window_manager.modal_handler_add(self)
        except AttributeError:
            try:
                context.modal_handler_add(self)
            except AttributeError:
                pass  # LFS may auto-register when invoke returns RUNNING_MODAL
        return {'RUNNING_MODAL'}

    def modal(self, context, event: Event) -> set:
        global _pending_pick_xy, _pending_ticks

        # ── Camera navigation events: keep overlay redrawn so points track ──
        if event.type in {'MOUSEMOVE', 'MIDDLEMOUSE', 'WHEELUPMOUSE',
                          'WHEELDOWNMOUSE', 'NUMPAD_0'}:
            # Flush a deferred pick if one is waiting
            if _pending_pick_xy is not None:
                _pending_ticks -= 1
                if _pending_ticks <= 0:
                    px, py = _pending_pick_xy
                    _pending_pick_xy = None
                    if _try_pick(px, py):
                        try:
                            lf.ui.request_redraw()
                        except Exception:
                            pass
                        return {'FINISHED'}
                    # Still no hit — give up and let the user click again
            try:
                lf.ui.request_redraw()
            except Exception:
                pass
            return {'PASS_THROUGH'}

        # ── Left-click: attempt pick, defer if depth not ready ───────────────
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            x, y = event.mouse_region_x, event.mouse_region_y
            if _try_pick(x, y):
                try:
                    lf.ui.request_redraw()
                except Exception:
                    pass
                return {'FINISHED'}
            # Depth miss — queue a deferred retry
            _pending_pick_xy = (x, y)
            _pending_ticks   = _PICK_RETRY_TICKS
            lf.log.info("EDIT Align: depth miss on click — deferring pick")
            return {'RUNNING_MODAL'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            _pending_pick_xy = None
            clear_pick_callback()
            try:
                lf.ui.request_redraw()
            except Exception:
                pass
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def cancel(self, context):
        global _pending_pick_xy
        _pending_pick_xy = None
        clear_pick_callback()


# ── Aliases for LFS_Edit_Plugin compatibility ──────────────────────────────────
set_capture_callback  = set_pick_callback
clear_capture_callback = clear_pick_callback
was_capture_cancelled  = was_pick_cancelled
