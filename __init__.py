# SPDX-FileCopyrightText: 2025
# SPDX-License-Identifier: GPL-3.0-or-later

import lichtfeld as lf
from .panels.transform_panel import TransformPanel
from .panels.point_align_panel import PointAlignPanel
from .operators.align_picker import ALIGN_OT_pick_point
from .operators.point_align_picker import POINTALIGN_OT_pick_point

_classes = [TransformPanel, PointAlignPanel, ALIGN_OT_pick_point, POINTALIGN_OT_pick_point]


def on_load():
    for cls in _classes:
        lf.register_class(cls)
    lf.log.info("EDIT loaded")


def on_unload():
    from .panels.transform_panel import _remove_align_draw_handler
    from .panels.point_align_panel import remove_point_align_draw_handler
    _remove_align_draw_handler()
    remove_point_align_draw_handler()
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("EDIT unloaded")
