# LFS Enhanced Edit Plugin

**Version:** 0.2.6  
**Author:** Brian Davis (bb6)  
**Requires:** Lichtfeld Studio ≥ 0.5.1

A comprehensive transform editing panel for Gaussian Splat nodes in Lichtfeld Studio. Provides precision translation, rotation, and scale controls with live preview, alignment tools, baking, merging, and splat management — all from a single panel.

---

## Installation

1. Paste address https://github.com/bgofish/LFS_Edit_Plugin  into the Plugin MarketPlace

2. The **Edit** tab will appear in the main panel sidebar (Left of Rendering)

---

## Panel Overview

The panel only activates when a splat node is selected in the scene. It is divided into the following sections:
<img width="500" height="600" alt="image" src="https://github.com/user-attachments/assets/9d9187fb-a5d2-4e09-959c-f0c29b33ee2a" />

---

## Header Controls

| Control | Description |
|---|---|
| **Node: `<name>`** | Displays the currently selected node name |
| **Grab from viewport** | Re-reads the node's current world transform from the scene and loads it into the sliders. Use this if the sliders drift out of sync after an external move |
| **Live** checkbox | When enabled, every slider or text field change is applied to the scene immediately. Disable to batch changes and commit them with **Apply** |
| **Read Settings** | Reloads `settings.json` from disk — updates slider limits, step sizes, and scene node naming with needing a reload of the plugin |
| **Open Log** | Opens `session_log.json` in Notepad++ (falls back to Notepad) |
| **Open Settings** | Opens `settings.json` in Notepad++ (falls back to Notepad) |

---

## Translation

Controls world-space position along X, Y, and Z axes.

**Default range:** −50 to +50  
**Default step:** 0.1

Each axis has a slider for coarse adjustment, a text field for direct numeric entry, and **−** / **+** step buttons that nudge the value by the current step size.

The **Step** control (−/+ buttons beside the section label) walks through a sensitivity ladder, letting you change how much each nudge moves the node:

```
0.001  0.002  0.005  0.01  0.02  0.05  0.1  0.2  0.5
1  2  5  10  20  50  100  200  500  1000
```

> **Auto-expand:** If **ReCent XYZ** or **ReCent XZ-0Y** calculates a value outside the current translation range, the limits are automatically expanded to a clean round number (with 10% margin) and `settings.json` is updated. The status bar reports what changed.

---

## Actions

Quick-fill helpers that populate the translation fields without applying the transform.

### ReCent XYZ

Calculates the bounding-box centroid of the selected node and prefills X, Y, Z so that centroid would move to the world origin `(0, 0, 0)`. Rotation and scale are left unchanged.

**Workflow:**
1. Select the node
2. Click **ReCent XYZ**
3. Verify the prefilled values in the translation fields
4. Click **Apply**  only needed if Live was not active then **Bake** to permanently write the change
4.1. or **Bake** to permanently write the change if Live is active


### ReCent XZ-0Y

Calculates the bounding-box and prefills X, Y, Z so the model is centred on the X and Z axes with its **top face placed at Y = 0** (floor-at-origin convention). Useful for aligning models to a ground plane.

**Workflow:**
1. Select the node
2. Click **ReCent XZ-0Y**
3. Verify the prefilled values
4. Click **Apply** or **Bake**

> Both recenter operations only prefill the translation fields — they do not apply immediately unless **Live** mode is on.

---

## Align ▶ / ▼

A collapsible section for aligning a node's rotation using two picked points on the model surface. Click the **▶ Align** button to expand it.

### Workflow

1. **Select an axis** — choose which world axis the vector between your two points should align to (X, Y, or Z). The active axis is shown in brackets e.g. `[Y]`
2. **Pick Point 1** — click the button, then click directly on the model surface in the viewport. The point is shown as a green dot with an overlay label
3. **Pick Point 2** — repeat for the second point (shown in orange)
4. **Calc. Angle** — computes the rotation delta needed to align the vector P1→P2 to the selected axis, and adds those values to the Rotation fields
5. **Apply** or **Bake** to commit

| Button | Description |
|---|---|
| **X / Y / Z** | Select the target alignment axis. Active axis shown as `[X]` / `[Y]` / `[Z]` |
| **Pick Point 1** / **Repick Pt 1** | Starts viewport picking for point 1. Click directly on the model surface |
| **Pick Point 2** / **Repick Pt 2** | Starts viewport picking for point 2 |
| **Calc. Angle** | Calculates rotation delta and writes to the Rx / Ry / Rz fields |

While picking is active a coloured prompt appears in the viewport. Press **ESC** or right-click to cancel picking. Picked points are shown as coloured dots with a connecting line that persist until a new pick or reset.

The calculated rotation components are displayed in colour after calculation: red Rx, green Ry, blue Rz.

---

## Rotation (°)

Controls world-space rotation around X (pitch), Y (yaw), and Z (roll) axes.

**Default range:** −180° to +180°  
**Default step:** 1.0°

The **Step** sensitivity ladder works the same way as translation. Each axis has a slider, text field, and −/+ step buttons.

---

## Scale

Controls node scale along X, Y, and Z axes.

**Default range:** 0.01 to 5.0  
**Default step:** 1.0

The **Uniform [Use X only]** checkbox locks Y and Z to follow X — changing the X slider scales all three axes together. Uncheck to scale each axis independently.

The **Step** sensitivity ladder applies the same way as translation and rotation.

---

## Apply / Reset

| Button | Description |
|---|---|
| **Apply** | Pushes the current slider values to the scene as the node's world transform. Only needed when **Live** mode is off |
| **Reset** | Sets translation to 0, rotation to 0°, scale to 1.0, and applies immediately |

---

## Bake

Permanently writes the current transform into the raw Gaussian data, then resets the node transform to identity. After baking the node sits at the origin with no visible transform — the Gaussians themselves have been repositioned.

- If a **splat node** is selected: bakes that node's transform
- If a **group node** is selected: bakes all splat nodes inside the group, then resets the group transform to identity

> ⚠️ **Bake is irreversible.** Save a backup of your `.ply` or `.spz` file before baking. There is no undo.

**Recommended workflow:**
1. Position the node using the transform controls
2. Click **Apply** to confirm the values look correct
3. Click **Bake Transform**

---

## Merge Visible Nodes

Combines all currently **visible** splat nodes into a single new node. Hidden nodes are left untouched.

| Control | Description |
|---|---|
| Name field | Name for the merged output node. Defaults to `merged` |
| **Merge Visible** | Performs the merge |

If a node with the chosen name already exists it will be overwritten.

---

## New Group Folder

Creates an empty group/folder node in the scene hierarchy for organising splat nodes.

| Control | Description |
|---|---|
| Name field | Name for the new group. Defaults to `Group` |
| **Create** | Adds the empty folder node to the scene |

---

## Move Selected Splats

Moves the **currently selected Gaussians** from the active node into a named destination node, without affecting unselected Gaussians in the source.

| Control | Description |
|---|---|
| Name field | Destination node name. If it exists, splats are appended; if not, a new node is created. Duplicate names get a suffix `_01`, `_02` … added automatically |
| **Move** | Performs the move. Make a brush or lasso selection in the viewport first |

**Workflow:**
1. Select the source node in the scene
2. Use the viewport selection tools to paint over the Gaussians you want to move
3. Type a destination node name in the field
4. Click **Move**

---

## Settings File (`settings.json`)

The plugin stores its configuration in `settings.json` inside the plugin folder. You can edit this directly to set custom slider limits and step sizes. Click **Open Settings** in the panel header to open it.

After editing, click **Read Settings** in the panel to apply changes without restarting.

### Structure

```json
{
  "transform": {
    "tx": 0.0,  "ty": 0.0,  "tz": 0.0,
    "rx": 0.0,  "ry": 0.0,  "rz": 0.0,
    "sx": 1.0,  "sy": 1.0,  "sz": 1.0,
    "uniform_scale": true,
    "live": true,
    "merge_name": "merged",
    "folder_name": "Group",
    "move_target": "Selection"
  },
  "limits": {
    "translation_min":  -50.0,
    "translation_max":   50.0,
    "rotation_min":    -180.0,
    "rotation_max":     180.0,
    "scale_min":          0.01,
    "scale_max":          5.0,
    "translation_step":   0.1,
    "rotation_step":      1.0,
    "scale_step":         1.0
  }
}
```

Translation limits are also expanded automatically when **ReCent XYZ** or **ReCent XZ-0Y** calculates a value that would exceed them — the updated values are written back to `settings.json` immediately.

---

## Session Log (`session_log.json`)

Every **Apply**, **Grab**, **Reset**, and **Bake** action is appended to `session_log.json` as a timestamped JSON record. Click **Open Log** in the panel header to view it. Exact duplicate entries are suppressed.

Each entry records:
- UTC timestamp
- Action type (`apply`, `grab`, `reset`, `bake`)
- Node name
- Full transform values (tx, ty, tz, rx, ry, rz, sx, sy, sz)
- Uniform scale and Live mode state

---

## File Structure

```
LFS_Edit_Plugin/
├── __init__.py                  Plugin entry point and class registration
├── pyproject.toml               Plugin metadata and LFS API requirements
├── settings.json                User settings (auto-created on first run)
├── session_log.json             Transform history log (auto-created)
├── operators/
│   ├── __init__.py
│   └── align_picker.py          Modal viewport point-picker operator
└── panels/
    ├── __init__.py
    ├── transform_panel.py        Main panel logic
    └── transform_panel.rml       Panel UI layout (RmlUI)
```

---

## Tips

- **Large scenes:** If your model is far from the origin, use **ReCent XYZ** first to bring it close, then fine-tune with the step controls. Translation limits auto-expand if needed.
- **Levelling a scan:** Use **Align** with the Y axis. Pick two points along a surface that should be horizontal — the calculated Rx/Rz delta will level the model.
- **Combining multiple scans:** Align and bake each scan individually, then use **Merge Visible** to produce a single consolidated node.
- **Non-destructive workflow:** Keep **Live** on while exploring positions, switch it off for precision numeric entry, then **Apply** once satisfied before baking.
- **Backup before baking:** Bake writes directly into Gaussian data and cannot be undone. Always export or copy your source files first.

---

## Changelog

### 0.2.6
- Fixed alignment point-picker — now uses correct LFS modal operator pattern (no `modal_handler_add`; `invoke` returns `RUNNING_MODAL` directly)
- Fixed operator ID resolution using Python's `__name__` — works correctly regardless of what the plugin folder is named
- Added collapsible ▶ / ▼ Align section
- Added per-section sensitivity step ladder for Translation, Rotation, and Scale (19 levels: 0.001 → 1000)
- Added auto-expand of translation limits on ReCent operations — values are never silently clipped
- Viewport overlay (pick point markers, connecting line, axis labels) now correctly tracks the camera during pan, zoom, and orbit

---

## License

GPL-3.0-or-later
