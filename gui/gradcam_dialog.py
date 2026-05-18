"""
Grad-CAM dialog: shows original image + heatmap overlay side by side.
Lets the user pick any class from a dropdown to see where the model looks.
"""
import os
from typing import Optional, List

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSplitter, QProgressBar, QMessageBox, QFileDialog,
    QGroupBox,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage

try:
    from PIL.Image import Image as PILImage
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class GradCAMWorker(QThread):
    finished = Signal(object, object)   # (original_pil, overlay_pil)
    error    = Signal(str)

    def __init__(self, model, model_type: str, image_path: str,
                 class_idx: Optional[int], image_size: int = 224,
                 roi: Optional[dict] = None):
        super().__init__()
        self.model      = model
        self.model_type = model_type
        self.image_path = image_path
        self.class_idx  = class_idx
        self.image_size = image_size
        self.roi        = roi

    def run(self) -> None:
        try:
            from core.gradcam import compute_gradcam_overlay
            orig, overlay = compute_gradcam_overlay(
                self.model, self.model_type, self.image_path,
                self.class_idx, self.image_size, roi=self.roi,
            )
            self.finished.emit(orig, overlay)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# PIL image → QPixmap helper
# ---------------------------------------------------------------------------

def _pil_to_qpixmap(img) -> QPixmap:
    """Convert a PIL RGB image to a QPixmap."""
    img_rgb = img.convert("RGB")
    w, h = img_rgb.size
    data = img_rgb.tobytes("raw", "RGB")
    qimg = QImage(data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class GradCAMDialog(QDialog):
    """
    Shows original image alongside its Grad-CAM overlay.

    Parameters
    ----------
    model        : loaded PyTorch model
    model_type   : str  (resnet18, mobilenet_v2, …)
    image_path   : absolute path to the image file
    class_names  : list of class label strings
    class_idx    : initially selected class index (None = argmax)
    image_size   : model input size (default 224)
    """

    def __init__(self, model, model_type: str, image_path: str,
                 class_names: List[str], class_idx: Optional[int] = None,
                 image_size: int = 224, roi: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.model      = model
        self.model_type = model_type
        self.image_path = image_path
        self.class_names = class_names
        self._class_idx  = class_idx
        self.image_size  = image_size
        self._roi        = roi
        self._overlay_pil = None
        self._worker: Optional[GradCAMWorker] = None

        title = "Grad-CAM – Aktivierungskarte"
        if roi is not None:
            title += f"  [ROI {int(roi.get('x',0))},{int(roi.get('y',0))}  {int(roi.get('w',0))}×{int(roi.get('h',0))} px]"
        self.setWindowTitle(title)
        self.resize(900, 520)
        self._build_ui()
        self._run()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # Top bar
        top = QHBoxLayout()
        top.addWidget(QLabel("Ziel-Klasse:"))
        self._class_combo = QComboBox()
        self._class_combo.addItem("Vorhergesagte Klasse (auto)")
        for name in self.class_names:
            self._class_combo.addItem(name)
        if self._class_idx is not None and self._class_idx < len(self.class_names):
            self._class_combo.setCurrentIndex(self._class_idx + 1)
        self._class_combo.currentIndexChanged.connect(self._on_class_changed)
        top.addWidget(self._class_combo)
        top.addStretch()
        self._save_btn = QPushButton("Overlay speichern…")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_overlay)
        top.addWidget(self._save_btn)
        root.addLayout(top)

        # Image panels
        splitter = QSplitter(Qt.Horizontal)
        left_box  = QGroupBox("Original")
        right_box = QGroupBox("Grad-CAM Overlay")
        lv = QVBoxLayout(left_box)
        rv = QVBoxLayout(right_box)

        self._orig_label    = QLabel()
        self._overlay_label = QLabel()
        for lbl in (self._orig_label, self._overlay_label):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(300, 300)
            lbl.setStyleSheet("background:#1E2837;border-radius:4px;")
        lv.addWidget(self._orig_label)
        rv.addWidget(self._overlay_label)
        splitter.addWidget(left_box)
        splitter.addWidget(right_box)
        splitter.setSizes([1, 1])
        root.addWidget(splitter, stretch=1)

        # Progress / status
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self._status_label)

        # Close
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Schließen")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------ compute

    def _run(self) -> None:
        idx = self._combo_to_class_idx()
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()

        self._progress.setVisible(True)
        self._status_label.setText("Berechne Grad-CAM …")
        self._save_btn.setEnabled(False)
        self._class_combo.setEnabled(False)

        self._worker = GradCAMWorker(
            self.model, self.model_type, self.image_path, idx, self.image_size,
            roi=self._roi,
        )
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _combo_to_class_idx(self) -> Optional[int]:
        i = self._class_combo.currentIndex()
        return None if i == 0 else (i - 1)

    # ------------------------------------------------------------------ slots

    def _on_class_changed(self) -> None:
        self._run()

    def _on_done(self, orig_pil, overlay_pil) -> None:
        self._progress.setVisible(False)
        self._class_combo.setEnabled(True)
        self._overlay_pil = overlay_pil

        self._show_image(self._orig_label, orig_pil)
        self._show_image(self._overlay_label, overlay_pil)

        cls_name = self._combo_to_class_name()
        self._status_label.setText(f"Aktivierungskarte für Klasse: {cls_name}")
        self._save_btn.setEnabled(True)

    def _on_error(self, msg: str) -> None:
        self._progress.setVisible(False)
        self._class_combo.setEnabled(True)
        self._status_label.setText(f"Fehler: {msg}")
        QMessageBox.critical(self, "Grad-CAM Fehler", msg)

    def _show_image(self, label: QLabel, pil_img) -> None:
        pix = _pil_to_qpixmap(pil_img)
        pix = pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pix)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Re-scale images when dialog is resized
        if self._overlay_pil:
            pass  # images are rescaled on next paint via scaled()

    def _combo_to_class_name(self) -> str:
        i = self._class_combo.currentIndex()
        if i == 0:
            return "vorhergesagt (auto)"
        idx = i - 1
        return self.class_names[idx] if idx < len(self.class_names) else str(idx)

    def _save_overlay(self) -> None:
        if self._overlay_pil is None:
            return
        base = os.path.splitext(os.path.basename(self.image_path))[0]
        default = f"{base}_gradcam.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Overlay speichern", default, "PNG (*.png);;JPEG (*.jpg)"
        )
        if path:
            try:
                self._overlay_pil.save(path)
                QMessageBox.information(self, "Gespeichert", f"Gespeichert:\n{path}")
            except Exception as exc:
                QMessageBox.critical(self, "Fehler", str(exc))
