"""
Augmentation preview dialog.
Shows the original image alongside N randomly augmented versions
so the user can see what the training pipeline will do to their data.
"""
import os
from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QCheckBox, QSpinBox, QGroupBox, QScrollArea,
    QWidget, QComboBox, QFileDialog, QProgressBar, QSizePolicy,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class AugWorker(QThread):
    done  = Signal(object, list)   # (original_pil, [augmented_pil, ...])
    error = Signal(str)

    def __init__(self, image_path: str, aug_cfg: Dict,
                 image_size: int, n_samples: int):
        super().__init__()
        self.image_path = image_path
        self.aug_cfg    = aug_cfg
        self.image_size = image_size
        self.n_samples  = n_samples

    def run(self) -> None:
        try:
            from torchvision import transforms

            img = PILImage.open(self.image_path).convert("RGB")

            # Original — just resize
            orig = img.resize((self.image_size, self.image_size), PILImage.BILINEAR)

            # Build augmentation pipeline (PIL in → PIL out, no ToTensor)
            t_list = []
            if self.aug_cfg.get("flip"):
                t_list.append(transforms.RandomHorizontalFlip(p=0.5))
                t_list.append(transforms.RandomVerticalFlip(p=0.15))
            if self.aug_cfg.get("rotation"):
                t_list.append(transforms.RandomRotation(15))
            if self.aug_cfg.get("brightness") or self.aug_cfg.get("contrast"):
                t_list.append(transforms.ColorJitter(
                    brightness=0.35 if self.aug_cfg.get("brightness") else 0,
                    contrast=0.35 if self.aug_cfg.get("contrast") else 0,
                    saturation=0.15,
                ))
            if self.aug_cfg.get("scale"):
                t_list.append(transforms.RandomResizedCrop(
                    self.image_size, scale=(0.75, 1.0)
                ))
            else:
                t_list.append(transforms.Resize((self.image_size, self.image_size)))

            pipeline = transforms.Compose(t_list) if t_list else None

            augmented = []
            for _ in range(self.n_samples):
                if pipeline is not None:
                    augmented.append(pipeline(img))
                else:
                    augmented.append(orig.copy())

            self.done.emit(orig, augmented)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# PIL → QPixmap helper
# ---------------------------------------------------------------------------

def _pil_to_qpixmap(img) -> QPixmap:
    img_rgb = img.convert("RGB")
    w, h = img_rgb.size
    data = img_rgb.tobytes("raw", "RGB")
    qimg = QImage(data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# Thumbnail cell widget
# ---------------------------------------------------------------------------

class _ThumbCell(QLabel):
    SIZE = 170

    def __init__(self, caption: str, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.SIZE + 8, self.SIZE + 26)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "background:#1A1A2E; border:1px solid #333; border-radius:4px;"
        )
        self._caption = caption
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        self.setText(f"<small>{self._caption}</small><br><span style='color:#555'>…</span>")

    def set_image(self, pil_img) -> None:
        pix = _pil_to_qpixmap(pil_img)
        pix = pix.scaled(self.SIZE, self.SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # Render caption below image
        self.setPixmap(pix)
        self.setToolTip(self._caption)


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class AugmentationPreviewDialog(QDialog):
    """
    Shows original + N augmented samples for a chosen image.

    Parameters
    ----------
    project     : current Project (for picking images)
    aug_cfg     : dict with keys flip/rotation/brightness/contrast/scale
    image_size  : model input size in pixels
    """

    COLS      = 3
    N_SAMPLES = 8   # augmented samples (original is shown separately)

    def __init__(self, project, aug_cfg: Dict,
                 image_size: int = 224, parent=None):
        super().__init__(parent)
        self._project    = project
        self._image_path = ""
        self._worker: Optional[AugWorker] = None

        self.setWindowTitle("Augmentierungs-Vorschau")
        self.resize(680, 660)
        self._build_ui(aug_cfg, image_size)

        # Pre-select first project image
        if project and project.images:
            first = project.images[0]
            if os.path.isfile(first):
                self._image_path = first
                idx = self._img_combo.findData(first)
                if idx >= 0:
                    self._img_combo.setCurrentIndex(idx)
                self._generate()

    # ------------------------------------------------------------------ UI

    def _build_ui(self, aug_cfg: Dict, image_size: int) -> None:
        root = QVBoxLayout(self)

        # ── Controls ────────────────────────────────────────────────────────
        ctrl = QGroupBox("Einstellungen")
        cv   = QVBoxLayout(ctrl)

        # Image picker
        img_row = QHBoxLayout()
        img_row.addWidget(QLabel("Bild:"))
        self._img_combo = QComboBox()
        self._img_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        if self._project:
            for p in self._project.images:
                if os.path.isfile(p):
                    self._img_combo.addItem(os.path.basename(p), p)
        self._img_combo.currentIndexChanged.connect(self._on_combo_changed)
        img_row.addWidget(self._img_combo)
        browse_btn = QPushButton("Datei…")
        browse_btn.setFixedWidth(60)
        browse_btn.clicked.connect(self._browse_image)
        img_row.addWidget(browse_btn)
        cv.addLayout(img_row)

        # Augmentation toggles + image size
        aug_row = QHBoxLayout()
        self._cb_flip       = QCheckBox("Flip")
        self._cb_rotation   = QCheckBox("Rotation")
        self._cb_brightness = QCheckBox("Helligkeit")
        self._cb_scale      = QCheckBox("Skalierung")
        self._cb_flip.setChecked(aug_cfg.get("flip", True))
        self._cb_rotation.setChecked(aug_cfg.get("rotation", True))
        self._cb_brightness.setChecked(
            aug_cfg.get("brightness", True) or aug_cfg.get("contrast", True)
        )
        self._cb_scale.setChecked(aug_cfg.get("scale", False))
        for cb in [self._cb_flip, self._cb_rotation, self._cb_brightness, self._cb_scale]:
            cb.stateChanged.connect(self._generate)
            aug_row.addWidget(cb)
        aug_row.addSpacing(16)
        aug_row.addWidget(QLabel("Größe:"))
        self._size_spin = QSpinBox()
        self._size_spin.setRange(64, 512)
        self._size_spin.setSingleStep(32)
        self._size_spin.setValue(image_size)
        self._size_spin.valueChanged.connect(self._generate)
        aug_row.addWidget(self._size_spin)
        aug_row.addStretch()
        self._regen_btn = QPushButton("↺ Neu generieren")
        self._regen_btn.clicked.connect(self._generate)
        aug_row.addWidget(self._regen_btn)
        cv.addLayout(aug_row)

        root.addWidget(ctrl)

        # ── Progress ────────────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        self._progress.setMaximumHeight(6)
        root.addWidget(self._progress)

        # ── Image grid ──────────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        grid_widget = QWidget()
        self._grid = QGridLayout(grid_widget)
        self._grid.setSpacing(6)
        scroll.setWidget(grid_widget)
        root.addWidget(scroll, stretch=1)

        # Create cells: 1 original + N_SAMPLES augmented
        total = 1 + self.N_SAMPLES
        self._cells: List[_ThumbCell] = []
        for i in range(total):
            caption = "Original" if i == 0 else f"Augmentiert {i}"
            cell = _ThumbCell(caption)
            row, col = divmod(i, self.COLS)
            self._grid.addWidget(cell, row, col)
            self._cells.append(cell)

        # ── Close button ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Schließen")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------ helpers

    def _current_aug_cfg(self) -> Dict:
        return {
            "flip":       self._cb_flip.isChecked(),
            "rotation":   self._cb_rotation.isChecked(),
            "brightness": self._cb_brightness.isChecked(),
            "contrast":   self._cb_brightness.isChecked(),
            "scale":      self._cb_scale.isChecked(),
        }

    # ------------------------------------------------------------------ events

    def _on_combo_changed(self) -> None:
        path = self._img_combo.currentData()
        if path and os.path.isfile(path):
            self._image_path = path
            self._generate()

    def _browse_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Bild wählen", "",
            "Bilder (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;Alle (*)"
        )
        if path:
            self._image_path = path
            # Add to combo if not already there
            if self._img_combo.findData(path) < 0:
                self._img_combo.insertItem(0, os.path.basename(path), path)
            self._img_combo.setCurrentIndex(self._img_combo.findData(path))
            self._generate()

    def _generate(self) -> None:
        if not self._image_path or not os.path.isfile(self._image_path):
            return
        if not HAS_PIL:
            for cell in self._cells:
                cell.setText("PIL nicht installiert")
            return
        try:
            from torchvision import transforms as _  # noqa – check availability
        except ImportError:
            for cell in self._cells:
                cell.setText("torchvision fehlt")
            return

        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()

        self._progress.setVisible(True)
        self._regen_btn.setEnabled(False)
        for cell in self._cells:
            cell._show_placeholder()

        self._worker = AugWorker(
            self._image_path,
            self._current_aug_cfg(),
            self._size_spin.value(),
            self.N_SAMPLES,
        )
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, orig, augmented) -> None:
        self._progress.setVisible(False)
        self._regen_btn.setEnabled(True)
        self._cells[0].set_image(orig)
        for i, aug_img in enumerate(augmented[:self.N_SAMPLES], start=1):
            if i < len(self._cells):
                self._cells[i].set_image(aug_img)

    def _on_error(self, msg: str) -> None:
        self._progress.setVisible(False)
        self._regen_btn.setEnabled(True)
        self._cells[0].setText(f"Fehler:\n{msg}")
