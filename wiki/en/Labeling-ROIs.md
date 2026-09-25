# Labeling & ROIs

> **PictureStudio v2.5.2** — Annotate images, assign labels, and draw regions of interest

---

# Labeling & ROIs

Annotate images, assign labels, draw regions of interest.

## Managing Labels

**Adding labels**
*Project → Manage labels…* (`Ctrl+L`)

Define a name and color. Labels can be renamed or recolored at any time.

At least 2 labels are required for training.

## Labeling Images

**Fast labeling**
1. Click an image in the list
2. Press `1`–`9` to assign label 1–9
3. `N` = next image, `P` = previous image

**Label filter:** Show only images belonging to a specific class.

## Drawing ROIs

**Rectangle** `R` – drag on the image

**Ellipse** `E` – drag on the image

**Polygon** `G` – click points, double-click to finish

## Editing ROIs

| Key | Action |
|---|---|
| Del | Delete selected ROI |
| Ctrl+C | Copy ROI |
| Ctrl+V | Paste ROI |
| Arrow keys | Move ROI by 2 px |
| Esc | Cancel drawing |

> 💡 **Copy ROIs to all images:** If all images share the same camera position, draw ROIs once and use *ROIs of this image → all images*.

## Segmentation Mask

**Tab 🎨 Segmentation Mask**

For pixel-accurate annotation:

Left-click = paint | Right-click = erase | Scroll = zoom

Select class and brush size via the toolbar.

*Save mask* saves the mask as a PNG file alongside the image.
