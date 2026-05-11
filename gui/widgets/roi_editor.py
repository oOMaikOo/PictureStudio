"""
ROI editor: Rectangle, Polygon, Ellipse drawing modes.
Permanent ROIs stay visible and are selectable/movable/deletable.

Keyboard shortcuts:
  R      – rectangle mode
  E      – ellipse mode
  G      – polygon mode
  W      – whole-image ROI (covers entire image)
  Space  – next image
  Esc    – cancel current drawing / deselect
  Del    – delete selected ROI(s)
  Ctrl+C – copy selected
  Ctrl+V – paste
  Arrows – nudge selected ROI by 2px
  1-9    – quick-assign label by index
"""
import uuid
from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPolygonItem,
    QGraphicsTextItem, QMenu, QApplication,
)
from PySide6.QtCore import Qt, QRectF, Signal, QPointF
from PySide6.QtGui import (
    QPen, QBrush, QColor, QPixmap, QPainter, QFont,
    QPolygonF, QKeyEvent, QMouseEvent, QWheelEvent,
)

DRAW_RECT    = "rect"
DRAW_ELLIPSE = "ellipse"
DRAW_POLYGON = "polygon"

NUDGE_PX = 2

# Visual constants
_BORDER_W_NORMAL = 2
_BORDER_W_HOVER  = 3
_FILL_ALPHA      = 60   # semi-transparent fill (0-255)


class _BaseROIItem:
    """Mixin: shared style + label overlay for all ROI item types."""

    def _init_roi(self, roi_data: Dict) -> None:
        self.roi_data = roi_data
        self._label_item: Optional[QGraphicsTextItem] = None
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(10)   # always above image (z=-1) and preview (z=5)
        self._apply_style()
        self._sync_label()

    def _color(self) -> QColor:
        raw = self.roi_data.get("color", "#E74C3C")
        c = QColor(raw)
        return c if c.isValid() else QColor("#E74C3C")

    def _apply_style(self, hover: bool = False) -> None:
        c = self._color()
        pen_w = _BORDER_W_HOVER if hover else _BORDER_W_NORMAL
        style = Qt.DashLine if hover else Qt.SolidLine
        pen = QPen(c, pen_w, style)
        pen.setCosmetic(False)   # scale with zoom
        self.setPen(pen)
        fill = QColor(c)
        fill.setAlpha(_FILL_ALPHA)
        self.setBrush(QBrush(fill))

    def _label_origin(self) -> tuple:
        """Top-left corner of the bounding rect in local coordinates."""
        r = self.boundingRect()
        return r.x() + 3, r.y() + 3

    def _sync_label(self) -> None:
        # Remove old label child
        if self._label_item is not None:
            self._label_item.setParentItem(None)
            self._label_item = None

        label = self.roi_data.get("label", "")
        rid   = self.roi_data.get("id", "")[:4]
        text  = label if label else f"ROI {rid}"

        item = QGraphicsTextItem(text, self)
        item.setFont(QFont(".AppleSystemUIFont", 8, QFont.Bold))
        item.setDefaultTextColor(self._color())
        lx, ly = self._label_origin()
        item.setPos(lx, ly)
        item.setZValue(11)
        self._label_item = item

    def update_roi_data(self, roi_data: Dict) -> None:
        self.roi_data = roi_data
        self._apply_style()
        self._sync_label()

    def hoverEnterEvent(self, event):
        self._apply_style(hover=True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._apply_style(hover=False)
        super().hoverLeaveEvent(event)


class RectROIItem(_BaseROIItem, QGraphicsRectItem):
    def __init__(self, roi_data: Dict, parent=None):
        QGraphicsRectItem.__init__(
            self,
            roi_data.get("x", 0), roi_data.get("y", 0),
            roi_data.get("w", 10), roi_data.get("h", 10),
            parent,
        )
        self._init_roi(roi_data)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            pos = value
            r = self.rect()
            self.roi_data["x"] = pos.x() + r.x()
            self.roi_data["y"] = pos.y() + r.y()
        return super().itemChange(change, value)


class EllipseROIItem(_BaseROIItem, QGraphicsEllipseItem):
    def __init__(self, roi_data: Dict, parent=None):
        QGraphicsEllipseItem.__init__(
            self,
            roi_data.get("x", 0), roi_data.get("y", 0),
            roi_data.get("w", 10), roi_data.get("h", 10),
            parent,
        )
        self._init_roi(roi_data)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            pos = value
            r = self.rect()
            self.roi_data["x"] = pos.x() + r.x()
            self.roi_data["y"] = pos.y() + r.y()
        return super().itemChange(change, value)


class PolygonROIItem(_BaseROIItem, QGraphicsPolygonItem):
    def __init__(self, roi_data: Dict, parent=None):
        points = roi_data.get("points", [])
        poly = QPolygonF([QPointF(p[0], p[1]) for p in points])
        QGraphicsPolygonItem.__init__(self, poly, parent)
        self._init_roi(roi_data)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            pos = value
            self.roi_data["offset_x"] = pos.x()
            self.roi_data["offset_y"] = pos.y()
        return super().itemChange(change, value)


def _make_item(roi_data: Dict) -> _BaseROIItem:
    t = roi_data.get("type", "rect")
    if t == "ellipse":
        return EllipseROIItem(roi_data)
    if t == "polygon":
        return PolygonROIItem(roi_data)
    return RectROIItem(roi_data)


# =========================================================== ROIEditor


class ROIEditor(QGraphicsView):
    """
    Full-featured ROI editor.  ROIs drawn here are permanent — they stay
    visible until explicitly deleted.  The dashed preview during drag is
    replaced by a solid, filled shape on mouse-release.
    """

    roi_added    = Signal(dict)
    roi_deleted  = Signal(str)    # roi_id
    roi_selected = Signal(str)    # roi_id
    roi_moved    = Signal(dict)   # roi_data
    mode_changed = Signal(str)    # new mode name
    label_quick_assign      = Signal(int)  # label index (0-based)
    whole_image_roi_requested = Signal()   # W key → create full-image ROI
    space_pressed             = Signal()   # Space key → next image

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = None
        self._image_rect  = QRectF()

        # Draw state
        self._mode:      str  = DRAW_RECT
        self._drawing:   bool = False
        self._start_pos: Optional[QPointF] = None
        self._drag_item  = None        # temporary dashed preview
        self._poly_points: List[QPointF] = []
        self._poly_preview = None

        # Permanent items
        self._roi_items: Dict[str, _BaseROIItem] = {}
        self._clipboard: Optional[Dict] = None

        # Current drawing colour / label
        self.current_label: str = ""
        self.current_color: str = "#E74C3C"
        self.label_list:    List[str] = []

        # View config
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QBrush(QColor(40, 40, 40)))
        self.setDragMode(QGraphicsView.NoDrag)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    # ------------------------------------------------------------------ image

    def load_image(self, image_path: str) -> bool:
        pix = QPixmap(image_path)
        if pix.isNull():
            return False
        self._cancel_drawing()
        self._scene.clear()
        self._roi_items.clear()
        self._pixmap_item = self._scene.addPixmap(pix)
        self._pixmap_item.setZValue(-1)
        self._image_rect = QRectF(pix.rect())
        self._scene.setSceneRect(self._image_rect)
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        return True

    def clear_image(self) -> None:
        self._cancel_drawing()
        self._scene.clear()
        self._roi_items.clear()
        self._pixmap_item = None

    # ------------------------------------------------------------------ ROIs

    def load_rois(self, roi_list: List[Dict]) -> None:
        """Replace all displayed ROIs with those from roi_list."""
        for item in list(self._roi_items.values()):
            self._scene.removeItem(item)
        self._roi_items.clear()
        for roi in roi_list:
            self._add_item(roi)
        self.viewport().update()

    def _add_item(self, roi_data: Dict) -> _BaseROIItem:
        item = _make_item(roi_data)
        self._scene.addItem(item)
        self._roi_items[roi_data["id"]] = item
        return item

    def add_roi_item(self, roi_data: Dict) -> None:
        """Add a single ROI item (used by undo/redo commands)."""
        if roi_data["id"] not in self._roi_items:
            self._add_item(roi_data)
            self.viewport().update()

    def update_roi_geometry(self, roi_data: Dict) -> None:
        """Update position/size of an existing ROI item (used by undo/redo)."""
        item = self._roi_items.get(roi_data["id"])
        if item:
            item.roi_data.update(roi_data)
            item.update_roi_data(item.roi_data)
            self.viewport().update()

    def update_roi_label(self, roi_id: str, label: str, color: str) -> None:
        item = self._roi_items.get(roi_id)
        if item:
            item.roi_data["label"] = label
            item.roi_data["color"] = color
            item.update_roi_data(item.roi_data)

    def delete_roi(self, roi_id: str) -> None:
        item = self._roi_items.pop(roi_id, None)
        if item:
            self._scene.removeItem(item)

    def delete_selected(self) -> None:
        for item in list(self._scene.selectedItems()):
            if hasattr(item, "roi_data"):
                roi_id = item.roi_data["id"]
                self._scene.removeItem(item)
                self._roi_items.pop(roi_id, None)
                self.roi_deleted.emit(roi_id)

    def get_all_roi_data(self) -> List[Dict]:
        """Return current roi_data for all items, flushing any pending moves."""
        result = []
        for item in self._roi_items.values():
            d = dict(item.roi_data)
            pos = item.pos()
            if pos.x() != 0 or pos.y() != 0:
                d["x"] = d.get("x", 0) + pos.x()
                d["y"] = d.get("y", 0) + pos.y()
                item.setPos(0, 0)
                item.roi_data["x"] = d["x"]
                item.roi_data["y"] = d["y"]
            result.append(d)
        return result

    def validate_rois(self) -> List[str]:
        warnings = []
        if self._image_rect.isNull():
            return warnings
        iw, ih = self._image_rect.width(), self._image_rect.height()
        for roi_id, item in self._roi_items.items():
            d = item.roi_data
            x, y = d.get("x", 0), d.get("y", 0)
            w, h = d.get("w", 0), d.get("h", 0)
            if x < 0 or y < 0 or x + w > iw or y + h > ih:
                warnings.append(f"ROI {roi_id[:4]} liegt außerhalb des Bildes.")
        return warnings

    # ------------------------------------------------------------------ mode

    def set_mode(self, mode: str) -> None:
        self._cancel_drawing()
        self._mode = mode
        self.setCursor(Qt.CrossCursor if mode in (DRAW_RECT, DRAW_ELLIPSE, DRAW_POLYGON)
                       else Qt.ArrowCursor)
        self.mode_changed.emit(mode)

    def _cancel_drawing(self) -> None:
        self._drawing = False
        self._start_pos = None
        if self._drag_item:
            self._scene.removeItem(self._drag_item)
            self._drag_item = None
        self._poly_points.clear()
        if self._poly_preview:
            self._scene.removeItem(self._poly_preview)
            self._poly_preview = None

    # ------------------------------------------------------------------ copy/paste

    def copy_selected(self) -> None:
        for item in self._scene.selectedItems():
            if hasattr(item, "roi_data"):
                self._clipboard = dict(item.roi_data)
                break

    def paste(self) -> None:
        if not self._clipboard:
            return
        roi = dict(self._clipboard)
        roi["id"] = str(uuid.uuid4())[:8]
        roi["x"]  = roi.get("x", 0) + 20
        roi["y"]  = roi.get("y", 0) + 20
        self._add_item(roi)
        self.roi_added.emit(roi)
        self.viewport().update()

    # ------------------------------------------------------------------ mouse events

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = self.mapToScene(event.pos())

        if event.button() == Qt.LeftButton:
            if self._mode in (DRAW_RECT, DRAW_ELLIPSE):
                self._drawing   = True
                self._start_pos = pos
                return              # consume event; don't pass to scene
            if self._mode == DRAW_POLYGON:
                self._poly_points.append(pos)
                return

        if event.button() == Qt.RightButton and self._mode == DRAW_POLYGON:
            self._finish_polygon()
            return

        # Default: let scene handle selection / item interaction
        super().mousePressEvent(event)
        sel = [i for i in self._scene.selectedItems() if hasattr(i, "roi_data")]
        if sel:
            self.roi_selected.emit(sel[0].roi_data["id"])

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self._mode == DRAW_POLYGON and len(self._poly_points) >= 3:
            self._finish_polygon()
        else:
            super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = self.mapToScene(event.pos())

        if self._drawing and self._start_pos is not None:
            # Rebuild the dashed drag-preview every move
            if self._drag_item:
                self._scene.removeItem(self._drag_item)
                self._drag_item = None
            rect = QRectF(self._start_pos, pos).normalized()
            pen  = QPen(QColor(self.current_color), 2, Qt.DashLine)
            fill = QColor(self.current_color)
            fill.setAlpha(30)
            if self._mode == DRAW_ELLIPSE:
                self._drag_item = self._scene.addEllipse(rect, pen, QBrush(fill))
            else:
                self._drag_item = self._scene.addRect(rect, pen, QBrush(fill))
            if self._drag_item:
                self._drag_item.setZValue(5)   # above image, below permanent ROIs
            return

        if self._mode == DRAW_POLYGON and self._poly_points:
            if self._poly_preview:
                self._scene.removeItem(self._poly_preview)
                self._poly_preview = None
            last = self._poly_points[-1]
            pen  = QPen(QColor(self.current_color), 1, Qt.DashLine)
            self._poly_preview = self._scene.addLine(
                last.x(), last.y(), pos.x(), pos.y(), pen
            )
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            pos  = self.mapToScene(event.pos())
            rect = QRectF(self._start_pos, pos).normalized()

            # Remove dashed preview
            if self._drag_item:
                self._scene.removeItem(self._drag_item)
                self._drag_item = None

            # Create permanent ROI if large enough
            if rect.width() > 5 and rect.height() > 5:
                roi = {
                    "id":    str(uuid.uuid4())[:8],
                    "type":  self._mode,
                    "x":     rect.x(),
                    "y":     rect.y(),
                    "w":     rect.width(),
                    "h":     rect.height(),
                    "label": self.current_label,
                    "color": self.current_color,
                }
                self._add_item(roi)
                self.viewport().update()    # ensure immediate repaint
                self.roi_added.emit(roi)
            return

        super().mouseReleaseEvent(event)

    def _finish_polygon(self) -> None:
        if len(self._poly_points) < 3:
            self._cancel_drawing()
            return
        if self._poly_preview:
            self._scene.removeItem(self._poly_preview)
            self._poly_preview = None
        pts = [(p.x(), p.y()) for p in self._poly_points]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        roi = {
            "id":     str(uuid.uuid4())[:8],
            "type":   DRAW_POLYGON,
            "points": pts,
            "x": min(xs), "y": min(ys),
            "w": max(xs) - min(xs),
            "h": max(ys) - min(ys),
            "label": self.current_label,
            "color": self.current_color,
        }
        self._poly_points.clear()
        self._add_item(roi)
        self.viewport().update()
        self.roi_added.emit(roi)

    # ------------------------------------------------------------------ wheel / key

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key  = event.key()
        mods = event.modifiers()

        if key == Qt.Key_Escape:
            self._cancel_drawing()
        elif key in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
        elif key == Qt.Key_R:
            self.set_mode(DRAW_RECT)
        elif key == Qt.Key_E:
            self.set_mode(DRAW_ELLIPSE)
        elif key == Qt.Key_G:
            self.set_mode(DRAW_POLYGON)
        elif mods & Qt.ControlModifier and key == Qt.Key_C:
            self.copy_selected()
        elif mods & Qt.ControlModifier and key == Qt.Key_V:
            self.paste()
        elif mods & Qt.ControlModifier and key == Qt.Key_Z:
            self.delete_selected()
        elif key == Qt.Key_W:
            self.whole_image_roi_requested.emit()
        elif key == Qt.Key_Space:
            self.space_pressed.emit()
        elif key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            self._nudge_selected(key)
        elif Qt.Key_1 <= key <= Qt.Key_9:
            self.label_quick_assign.emit(key - Qt.Key_1)
        else:
            super().keyPressEvent(event)

    def _nudge_selected(self, key: int) -> None:
        dx = dy = 0
        if key == Qt.Key_Left:  dx = -NUDGE_PX
        if key == Qt.Key_Right: dx =  NUDGE_PX
        if key == Qt.Key_Up:    dy = -NUDGE_PX
        if key == Qt.Key_Down:  dy =  NUDGE_PX
        for item in self._scene.selectedItems():
            if hasattr(item, "roi_data"):
                item.moveBy(dx, dy)
                self.roi_moved.emit(item.roi_data)

    # ------------------------------------------------------------------ context menu

    def _context_menu(self, pos) -> None:
        sel = [i for i in self._scene.selectedItems() if hasattr(i, "roi_data")]
        menu = QMenu(self)
        del_act   = menu.addAction("Löschen (Del)")
        del_act.setEnabled(bool(sel))
        copy_act  = menu.addAction("Kopieren (Ctrl+C)")
        copy_act.setEnabled(bool(sel))
        paste_act = menu.addAction("Einfügen (Ctrl+V)")
        paste_act.setEnabled(self._clipboard is not None)
        action = menu.exec(self.mapToGlobal(pos))
        if action == del_act:
            self.delete_selected()
        elif action == copy_act:
            self.copy_selected()
        elif action == paste_act:
            self.paste()

    # ------------------------------------------------------------------ zoom

    def reset_zoom(self) -> None:
        self.resetTransform()
        if self._scene.sceneRect().isValid():
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
