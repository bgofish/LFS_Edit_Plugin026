# SPDX-License-Identifier: GPL-3.0-or-later
"""Modal point-picker operator for the Align plugin.

Modelled on the working measurement_tool reference implementation:
- invoke() returns RUNNING_MODAL with NO modal_handler_add call
- modal() stays RUNNING_MODAL after a successful pick (doesn't finish)
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


# Aliases so transform_panel imports still work
set_capture_callback   = set_pick_callback
clear_capture_callback = clear_pick_callback
was_capture_cancelled  = was_pick_cancelled


class ALIGN_OT_pick_point(Operator):
    """Modal operator: click on the viewport to pick a world-space point."""

    label       = "Pick Alignment Point"
    description = "Click on the model to pick a point for alignment"
    options     = {'BLOCKING'}

    def invoke(self, context, event: Event) -> set:
        # Do NOT call modal_handler_add — LFS handles this automatically.
        # Matches the working measurement_tool reference implementation.
        return {'RUNNING_MODAL'}

    def modal(self, context, event: Event) -> set:
        global _pick_callback, _pick_point_num

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            result = sel.pick_at_screen(event.mouse_region_x, event.mouse_region_y)
            if result is not None and _pick_callback is not None:
                _pick_callback(result.world_position, _pick_point_num)
                # Stay RUNNING_MODAL — panel cancels the modal when done
                return {'RUNNING_MODAL'}
            return {'RUNNING_MODAL'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            clear_pick_callback()
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def cancel(self, context):
        clear_pick_callback()
