# SPDX-FileCopyrightText: 2025
# SPDX-License-Identifier: GPL-3.0-or-later

import lichtfeld as lf
from .panels.transform_panel import TransformPanel
from .operators.align_picker import ALIGN_OT_pick_point

_classes = [ALIGN_OT_pick_point, TransformPanel]


def on_load():
    for cls in _classes:
        lf.register_class(cls)
    lf.log.info("EDIT loaded")


def on_unload():
    from .panels.transform_panel import _remove_align_draw_handler
    _remove_align_draw_handler()
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("EDIT unloaded")
