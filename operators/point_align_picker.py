# SPDX-License-Identifier: GPL-3.0-or-later
"""Modal point-picker operator for the 3-Point Alignment feature.

Follows the same pattern as align_picker.py (the working reference):
- invoke() returns RUNNING_MODAL with NO modal_handler_add call
- modal() stays RUNNING_MODAL after a successful pick
- pick_at_screen() is the correct API for splat picking
"""

import lichtfeld as lf
import lichtfeld.selection as sel
from lfs_plugins.types import Operator, Event

# ── Module-level callback state ───────────────────────────────────────────────
_pick_callback  = None
_pick_point_num = 0
_pick_cancelled = False


def set_pick_callback(callback, point_num: int):
    global _pick_callback, _pick_point_num, _pick_cancelled
    _pick_callback  = callback
    _pick_point_num = point_num
    _pick_cancelled = False


def clear_pick_callback():
    global _pick_callback, _pick_point_num, _pick_cancelled
    _pick_callback  = None
    _pick_point_num = 0
    _pick_cancelled = True


def was_pick_cancelled() -> bool:
    global _pick_cancelled
    if _pick_cancelled:
        _pick_cancelled = False
        return True
    return False


class POINTALIGN_OT_pick_point(Operator):
    """Modal operator: click on the viewport to pick a world-space point for
    the 3-point alignment workflow."""

    label       = "Pick 3-Point Alignment Point"
    description = "Click on the model to pick a correspondence point"
    options     = {'BLOCKING'}

    def invoke(self, context, event: Event) -> set:
        # LFS handles modal_handler_add automatically.
        return {'RUNNING_MODAL'}

    def modal(self, context, event: Event) -> set:
        global _pick_callback, _pick_point_num

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            result = sel.pick_at_screen(event.mouse_region_x, event.mouse_region_y)
            if result is not None and _pick_callback is not None:
                _pick_callback(result.world_position, _pick_point_num)
                return {'RUNNING_MODAL'}
            return {'RUNNING_MODAL'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            clear_pick_callback()
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def cancel(self, context):
        clear_pick_callback()
