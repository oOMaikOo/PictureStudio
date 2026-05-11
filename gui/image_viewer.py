"""
Scrollable image viewer with zoom and pan support.
Used inside the ROI editor as the background widget.
"""
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PySide6.QtCore import Qt, QRectF, Signal, QPointF
from PySide6.QtGui import QPixmap, QPainter, QWheelEvent


class ImageViewer(QGraphicsView):
    """A simple zoomable/pannable image view (no ROI editing)."""

    zoom_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem = None
        self._zoom_factor = 1.0

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(Qt.darkGray)

    def load_image(self, image_path: str) -> bool:
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return False
        self._set_pixmap(pixmap)
        return True

    def load_pixmap(self, pixmap: QPixmap) -> None:
        self._set_pixmap(pixmap)

    def _set_pixmap(self, pixmap: QPixmap) -> None:
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.reset_zoom()

    def reset_zoom(self) -> None:
        self.resetTransform()
        self._zoom_factor = 1.0
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.angleDelta().y() > 0:
            factor = 1.15
        else:
            factor = 1.0 / 1.15
        self._zoom_factor *= factor
        self.scale(factor, factor)
        self.zoom_changed.emit(self._zoom_factor)

    def get_scene(self) -> QGraphicsScene:
        return self._scene

    def get_pixmap_item(self) -> QGraphicsPixmapItem:
        return self._pixmap_item
