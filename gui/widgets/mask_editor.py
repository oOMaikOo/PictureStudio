"""
Pixel-level segmentation mask editor.
Left-click = paint, Right-click = erase. Mask stored as PNG beside the image.
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSlider, QPushButton,
    QLabel, QComboBox, QSizePolicy, QCheckBox,
)
from PySide6.QtCore import Qt, Signal, QPoint, QRect, QSize
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QCursor,
    QMouseEvent, QWheelEvent,
)

_COLORS = [
    ("#E74C3C", "Defekt"),
    ("#3498DB", "Bereich 1"),
    ("#2ECC71", "Bereich 2"),
    ("#F39C12", "Bereich 3"),
    ("#9B59B6", "Bereich 4"),
]


def mask_path_for(image_path: str, label: str = "mask") -> str:
    base, _ = os.path.splitext(image_path)
    return f"{base}__{label}_mask.png"


class MaskEditorWidget(QWidget):
    """
    Overlay mask editor on top of an image.
    Emits `mask_saved(path)` when the mask is written to disk.
    """

    mask_saved = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_path: Optional[str] = None
        self._image_bgr: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None       # H×W uint8, 0=bg, 1..5=class
        self._brush_size = 18
        self._active_class = 1
        self._erasing = False
        self._last_pos: Optional[QPoint] = None
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(300, 300)

    # ── public API ────────────────────────────────────────────────────────────

    def load_image(self, image_path: str) -> None:
        self._image_path = image_path
        img = cv2.imread(image_path)
        if img is None:
            return
        self._image_bgr = img
        h, w = img.shape[:2]

        # Load existing mask if present
        mpath = mask_path_for(image_path)
        if os.path.exists(mpath):
            loaded = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
            if loaded is not None and loaded.shape == (h, w):
                self._mask = loaded.astype(np.uint8)
            else:
                self._mask = np.zeros((h, w), dtype=np.uint8)
        else:
            self._mask = np.zeros((h, w), dtype=np.uint8)

        self._fit_to_view()
        self.update()

    def save_mask(self) -> Optional[str]:
        if self._image_path is None or self._mask is None:
            return None
        mpath = mask_path_for(self._image_path)
        cv2.imwrite(mpath, self._mask)
        self.mask_saved.emit(mpath)
        return mpath

    def clear_mask(self) -> None:
        if self._mask is not None:
            self._mask[:] = 0
            self.update()

    def set_brush_size(self, size: int) -> None:
        self._brush_size = max(1, size)
        self.update()

    def set_active_class(self, cls: int) -> None:
        self._active_class = cls

    # ── painting ──────────────────────────────────────────────────────────────

    def _img_to_widget(self, p: QPoint) -> QPoint:
        """Map image-space point → widget-space."""
        return QPoint(int(p.x() * self._scale) + self._offset.x(),
                      int(p.y() * self._scale) + self._offset.y())

    def _widget_to_img(self, p: QPoint) -> QPoint:
        """Map widget-space point → image-space."""
        ix = (p.x() - self._offset.x()) / self._scale
        iy = (p.y() - self._offset.y()) / self._scale
        return QPoint(int(ix), int(iy))

    def _paint_at(self, widget_pos: QPoint) -> None:
        if self._mask is None:
            return
        ip = self._widget_to_img(widget_pos)
        r = max(1, int(self._brush_size / self._scale / 2))
        h, w = self._mask.shape
        x1 = max(0, ip.x() - r)
        y1 = max(0, ip.y() - r)
        x2 = min(w, ip.x() + r)
        y2 = min(h, ip.y() + r)
        val = 0 if self._erasing else self._active_class
        cv2.circle(self._mask, (ip.x(), ip.y()), r, val, -1)
        self.update()

    # ── Qt events ────────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._mask is None:
            return
        self._erasing = event.button() == Qt.RightButton
        self._paint_at(event.position().toPoint())
        self._last_pos = event.position().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._mask is None:
            return
        if event.buttons() & (Qt.LeftButton | Qt.RightButton):
            self._paint_at(event.position().toPoint())
        self._last_pos = event.position().toPoint()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._last_pos = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        self._scale = max(0.1, min(8.0, self._scale * factor))
        self.update()

    def resizeEvent(self, event) -> None:
        self._fit_to_view()
        super().resizeEvent(event)

    def _fit_to_view(self) -> None:
        if self._image_bgr is None:
            return
        h, w = self._image_bgr.shape[:2]
        vw, vh = self.width(), self.height()
        self._scale = min(vw / w, vh / h)
        self._offset = QPoint(
            int((vw - w * self._scale) / 2),
            int((vh - h * self._scale) / 2),
        )

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        if self._image_bgr is not None:
            # Draw image
            img_rgb = cv2.cvtColor(self._image_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            qimg = QImage(img_rgb.data, w, h, w * 3, QImage.Format_RGB888)
            dest = QRect(
                self._offset.x(), self._offset.y(),
                int(w * self._scale), int(h * self._scale),
            )
            painter.drawPixmap(dest, QPixmap.fromImage(qimg))

            # Draw mask overlay
            if self._mask is not None:
                overlay = np.zeros((*self._mask.shape, 4), dtype=np.uint8)
                for cls_idx, (hex_color, _) in enumerate(_COLORS, start=1):
                    c = QColor(hex_color)
                    mask_px = self._mask == cls_idx
                    overlay[mask_px] = [c.red(), c.green(), c.blue(), 160]
                ov_img = QImage(overlay.data, w, h, w * 4, QImage.Format_RGBA8888)
                painter.drawPixmap(dest, QPixmap.fromImage(ov_img))

        # Draw brush cursor
        if self._last_pos is not None:
            r = int(self._brush_size / 2)
            color = QColor("#888888") if self._erasing else QColor(
                _COLORS[self._active_class - 1][0]
            )
            pen = QPen(color, 1, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(self._last_pos, r, r)

        painter.end()


class MaskEditorPanel(QWidget):
    """Wraps MaskEditorWidget with controls (brush size, class, save, clear)."""

    mask_saved = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        vl = QVBoxLayout(self)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(4)

        # Controls
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Klasse:"))
        self._class_combo = QComboBox()
        for hex_c, name in _COLORS:
            self._class_combo.addItem(name)
        self._class_combo.currentIndexChanged.connect(
            lambda i: self.editor.set_active_class(i + 1)
        )
        ctrl.addWidget(self._class_combo)

        ctrl.addWidget(QLabel("Pinsel:"))
        self._brush_slider = QSlider(Qt.Horizontal)
        self._brush_slider.setRange(2, 80)
        self._brush_slider.setValue(18)
        self._brush_slider.setFixedWidth(100)
        self._brush_slider.valueChanged.connect(self.editor_brush_changed)
        ctrl.addWidget(self._brush_slider)
        self._brush_lbl = QLabel("18 px")
        ctrl.addWidget(self._brush_lbl)

        ctrl.addStretch()

        save_btn = QPushButton("Maske speichern")
        save_btn.setStyleSheet(
            "background:#1976D2;color:white;padding:4px 10px;border-radius:3px;"
        )
        save_btn.clicked.connect(self._save)
        ctrl.addWidget(save_btn)

        clear_btn = QPushButton("Löschen")
        clear_btn.setStyleSheet("padding:4px 8px;")
        clear_btn.clicked.connect(self._clear)
        ctrl.addWidget(clear_btn)

        hint = QLabel("LMT = malen · RMT = löschen · Scroll = Zoom")
        hint.setStyleSheet("color:#888;font-size:10px;")
        ctrl.addWidget(hint)

        vl.addLayout(ctrl)

        self.editor = MaskEditorWidget()
        self.editor.mask_saved.connect(self.mask_saved)
        vl.addWidget(self.editor, stretch=1)

        # Wire brush slider (after editor is created)
        self._brush_slider.valueChanged.connect(self._on_brush_changed)

    def _on_brush_changed(self, val: int) -> None:
        self.editor.set_brush_size(val)
        self._brush_lbl.setText(f"{val} px")

    def editor_brush_changed(self, val: int) -> None:
        self.editor.set_brush_size(val)

    def load_image(self, path: str) -> None:
        self.editor.load_image(path)

    def _save(self) -> None:
        self.editor.save_mask()

    def _clear(self) -> None:
        self.editor.clear_mask()
