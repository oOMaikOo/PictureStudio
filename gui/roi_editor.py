"""
Interactive ROI editor built on QGraphicsView.
Supports drawing, selecting, and deleting rectangle ROIs.
"""
import uuid
from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QGraphicsTextItem, QMenu, QApplication,
)
from PySide6.QtCore import Qt, QRectF, Signal, QPointF
from PySide6.QtGui import (
    QPen, QBrush, QColor, QPixmap, QPainter, QFont,
    QWheelEvent, QMouseEvent, QKeyEvent,
)


class ROIGraphicsItem(QGraphicsRectItem):
    """Visual representation of a single ROI."""

    def __init__(self, roi_data: Dict, parent=None):
        x = roi_data.get("x", 0)
        y = roi_data.get("y", 0)
        w = roi_data.get("w", 10)
        h = roi_data.get("h", 10)
        super().__init__(x, y, w, h, parent)
        self.roi_data = roi_data
        self.setFlags(
            QGraphicsRectItem.ItemIsSelectable
            | QGraphicsRectItem.ItemIsMovable
            | QGraphicsRectItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self._apply_style()
        self._label_item = None
        self._update_label()

    def _apply_style(self, selected: bool = False) -> None:
        color_str = self.roi_data.get("color", "#E74C3C")
        color = QColor(color_str)
        pen_width = 3 if selected else 2
        pen = QPen(color, pen_width)
        pen.setStyle(Qt.DashLine if selected else Qt.SolidLine)
        self.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(40)
        self.setBrush(QBrush(fill))

    def _update_label(self) -> None:
        if self._label_item:
            self.scene().removeItem(self._label_item) if self.scene() else None
            self._label_item = None
        label = self.roi_data.get("label", "")
        roi_id = self.roi_data.get("id", "")
        display = f"{label}" if label else f"ROI {roi_id[:4]}"
        self._label_item = QGraphicsTextItem(display, self)
        font = QFont("Arial", 9, QFont.Bold)
        self._label_item.setFont(font)
        self._label_item.setDefaultTextColor(QColor(self.roi_data.get("color", "#E74C3C")))
        self._label_item.setPos(self.rect().x() + 2, self.rect().y() + 2)

    def update_roi(self, roi_data: Dict) -> None:
        self.roi_data = roi_data
        r = self.rect()
        self.setRect(roi_data.get("x", r.x()), roi_data.get("y", r.y()),
                     roi_data.get("w", r.width()), roi_data.get("h", r.height()))
        self._apply_style()
        self._update_label()

    def itemChange(self, change, value):
        if change == QGraphicsRectItem.ItemPositionHasChanged and self.scene():
            new_pos = value
            new_rect = self.rect()
            self.roi_data["x"] = new_pos.x() + new_rect.x()
            self.roi_data["y"] = new_pos.y() + new_rect.y()
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event):
        self._apply_style(selected=True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._apply_style(selected=False)
        super().hoverLeaveEvent(event)


class ROIEditor(QGraphicsView):
    """
    Full ROI editing widget.
    Signals:
        roi_added(roi_data dict)
        roi_selected(roi_id str)
        roi_deleted(roi_id str)
        roi_moved(roi_data dict)
    """

    roi_added = Signal(dict)
    roi_selected = Signal(str)
    roi_deleted = Signal(str)
    roi_moved = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item = None
        self._drawing = False
        self._draw_mode = False
        self._start_pos: Optional[QPointF] = None
        self._drag_rect = None  # temporary dashed rect while drawing
        self._roi_items: Dict[str, ROIGraphicsItem] = {}

        self.current_label: str = ""
        self.current_color: str = "#E74C3C"

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(Qt.darkGray)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    # ------------------------------------------------------------------ image

    def load_image(self, image_path: str) -> bool:
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return False
        self._scene.clear()
        self._roi_items.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(-1)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        return True

    def clear_image(self) -> None:
        self._scene.clear()
        self._roi_items.clear()
        self._pixmap_item = None

    # ------------------------------------------------------------------ ROIs

    def load_rois(self, roi_list: List[Dict]) -> None:
        """Remove old ROIs from scene and load a new list."""
        for item in list(self._roi_items.values()):
            self._scene.removeItem(item)
        self._roi_items.clear()
        for roi in roi_list:
            self._add_roi_item(roi)

    def _add_roi_item(self, roi_data: Dict) -> ROIGraphicsItem:
        item = ROIGraphicsItem(roi_data)
        self._scene.addItem(item)
        self._roi_items[roi_data["id"]] = item
        return item

    def update_roi_label(self, roi_id: str, label: str, color: str) -> None:
        item = self._roi_items.get(roi_id)
        if item:
            item.roi_data["label"] = label
            item.roi_data["color"] = color
            item.update_roi(item.roi_data)

    def delete_roi(self, roi_id: str) -> None:
        item = self._roi_items.pop(roi_id, None)
        if item:
            self._scene.removeItem(item)

    def delete_selected_rois(self) -> None:
        for item in self._scene.selectedItems():
            if isinstance(item, ROIGraphicsItem):
                roi_id = item.roi_data["id"]
                self._scene.removeItem(item)
                self._roi_items.pop(roi_id, None)
                self.roi_deleted.emit(roi_id)

    def get_all_roi_data(self) -> List[Dict]:
        result = []
        for item in self._roi_items.values():
            d = dict(item.roi_data)
            pos = item.pos()
            if pos.x() != 0 or pos.y() != 0:
                d["x"] = item.roi_data.get("x", 0) + pos.x()
                d["y"] = item.roi_data.get("y", 0) + pos.y()
                item.setPos(0, 0)
                item.roi_data["x"] = d["x"]
                item.roi_data["y"] = d["y"]
            result.append(d)
        return result

    # ------------------------------------------------------------------ mode

    def set_draw_mode(self, enabled: bool) -> None:
        self._draw_mode = enabled
        if enabled:
            self.setCursor(Qt.CrossCursor)
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            self.setCursor(Qt.ArrowCursor)
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    # ------------------------------------------------------------------ mouse events

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._draw_mode and event.button() == Qt.LeftButton:
            self._drawing = True
            self._start_pos = self.mapToScene(event.pos())
        else:
            super().mousePressEvent(event)
            selected = [i for i in self._scene.selectedItems() if isinstance(i, ROIGraphicsItem)]
            if selected:
                self.roi_selected.emit(selected[0].roi_data["id"])

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drawing and self._start_pos is not None:
            pos = self.mapToScene(event.pos())
            if self._drag_rect:
                self._scene.removeItem(self._drag_rect)
            rect = QRectF(self._start_pos, pos).normalized()
            pen = QPen(QColor(self.current_color), 2, Qt.DashLine)
            self._drag_rect = self._scene.addRect(rect, pen)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drawing and event.button() == Qt.LeftButton:
            self._drawing = False
            pos = self.mapToScene(event.pos())
            rect = QRectF(self._start_pos, pos).normalized()

            if self._drag_rect:
                self._scene.removeItem(self._drag_rect)
                self._drag_rect = None

            if rect.width() > 5 and rect.height() > 5:
                roi_data = {
                    "id": str(uuid.uuid4())[:8],
                    "x": rect.x(),
                    "y": rect.y(),
                    "w": rect.width(),
                    "h": rect.height(),
                    "label": self.current_label,
                    "color": self.current_color,
                    "type": "rect",
                }
                self._add_roi_item(roi_data)
                self.roi_added.emit(roi_data)
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Delete:
            self.delete_selected_rois()
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    # ------------------------------------------------------------------ context menu

    def _context_menu(self, pos) -> None:
        selected = [i for i in self._scene.selectedItems() if isinstance(i, ROIGraphicsItem)]
        if not selected:
            return
        menu = QMenu(self)
        delete_action = menu.addAction("ROI löschen")
        action = menu.exec(self.mapToGlobal(pos))
        if action == delete_action:
            for item in selected:
                roi_id = item.roi_data["id"]
                self._scene.removeItem(item)
                self._roi_items.pop(roi_id, None)
                self.roi_deleted.emit(roi_id)

    # ------------------------------------------------------------------ zoom

    def reset_zoom(self) -> None:
        self.resetTransform()
        if self._scene.sceneRect().isValid():
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
