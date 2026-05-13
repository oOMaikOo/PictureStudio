"""
Augmentation preview dialog.
Shows the original image alongside N randomly augmented versions
so the user can see what the training pipeline will do to their data.
"""
import os
from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox, QGroupBox,
    QScrollArea, QWidget, QComboBox, QFileDialog, QProgressBar,
    QSizePolicy, QSlider,
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
            orig = img.resize((self.image_size, self.image_size), PILImage.BILINEAR)

            t_list = []
            cfg = self.aug_cfg
            if cfg.get("flip"):
                t_list.append(transforms.RandomHorizontalFlip(p=0.5))
                t_list.append(transforms.RandomVerticalFlip(p=0.15))
            if cfg.get("rotation"):
                deg = float(cfg.get("rotation_degrees", 15))
                t_list.append(transforms.RandomRotation(deg))
            if cfg.get("brightness") or cfg.get("contrast"):
                bri = float(cfg.get("brightness_strength", 0.35))
                t_list.append(transforms.ColorJitter(
                    brightness=bri if cfg.get("brightness") else 0,
                    contrast=bri if cfg.get("contrast") else 0,
                    saturation=0.15,
                ))
            if cfg.get("blur"):
                radius = int(cfg.get("blur_radius", 3))
                radius = radius if radius % 2 == 1 else radius + 1
                t_list.append(transforms.GaussianBlur(radius, sigma=(0.1, 2.0)))
            if cfg.get("scale"):
                scale_lo = float(cfg.get("scale_min", 0.75))
                t_list.append(transforms.RandomResizedCrop(
                    self.image_size, scale=(scale_lo, 1.0)
                ))
            else:
                t_list.append(transforms.Resize((self.image_size, self.image_size)))

            pipeline = transforms.Compose(t_list) if t_list else None
            augmented = [
                pipeline(img) if pipeline else orig.copy()
                for _ in range(self.n_samples)
            ]
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
    Augmentation editor + live preview dialog.
    Shows original + N randomly augmented versions of a chosen image.
    Emits `config_accepted` with the final aug_cfg dict when the user
    clicks "Einstellungen übernehmen".

    Parameters
    ----------
    project     : current Project (for picking images)
    aug_cfg     : dict with keys flip/rotation/brightness/contrast/scale/blur + intensity params
    image_size  : model input size in pixels
    """

    config_accepted = Signal(dict)   # emitted when user clicks "Übernehmen"

    COLS      = 3
    N_SAMPLES = 8

    def __init__(self, project, aug_cfg: Dict,
                 image_size: int = 224, parent=None):
        super().__init__(parent)
        self._project    = project
        self._image_path = ""
        self._worker: Optional[AugWorker] = None

        self.setWindowTitle("Augmentierungs-Editor")
        self.resize(780, 740)
        self._build_ui(aug_cfg, image_size)

        if project and project.images:
            first = project.images[0]
            if os.path.isfile(first):
                self._image_path = first
                idx = self._img_combo.findData(first)
                if idx >= 0:
                    self._img_combo.setCurrentIndex(idx)
                self._generate()

    # ------------------------------------------------------------------ UI

    def _slider_row(self, label: str, lo: int, hi: int, val: int,
                    unit: str = "") -> tuple:
        """Return (QHBoxLayout, QSlider, QLabel) for a labeled slider."""
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setFixedWidth(130)
        lbl.setStyleSheet("color:#ADBAC7;font-size:10px;")
        row.addWidget(lbl)
        sld = QSlider(Qt.Horizontal)
        sld.setRange(lo, hi)
        sld.setValue(val)
        row.addWidget(sld)
        val_lbl = QLabel(f"{val}{unit}")
        val_lbl.setFixedWidth(44)
        val_lbl.setStyleSheet("color:#388BFD;font-size:10px;")
        row.addWidget(val_lbl)
        sld.valueChanged.connect(lambda v, l=val_lbl, u=unit: l.setText(f"{v}{u}"))
        return row, sld, val_lbl

    def _build_ui(self, aug_cfg: Dict, image_size: int) -> None:
        root = QVBoxLayout(self)

        # ── Image picker ────────────────────────────────────────────────────
        img_grp = QGroupBox("Vorschau-Bild")
        img_v = QVBoxLayout(img_grp)
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
        img_v.addLayout(img_row)
        root.addWidget(img_grp)

        # ── Augmentation editor ──────────────────────────────────────────────
        aug_grp = QGroupBox("Augmentierungen")
        av = QVBoxLayout(aug_grp)

        def _cb(text: str, key: str, default: bool) -> QCheckBox:
            cb = QCheckBox(text)
            cb.setChecked(aug_cfg.get(key, default))
            cb.stateChanged.connect(self._generate)
            return cb

        # Row 1: toggles
        tog_row = QHBoxLayout()
        self._cb_flip       = _cb("Flip",        "flip",       True)
        self._cb_rotation   = _cb("Rotation",    "rotation",   True)
        self._cb_brightness = _cb("Helligkeit",  "brightness", True)
        self._cb_scale      = _cb("Skalierung",  "scale",      False)
        self._cb_blur       = _cb("Blur",        "blur",       False)
        for cb in [self._cb_flip, self._cb_rotation, self._cb_brightness,
                   self._cb_scale, self._cb_blur]:
            tog_row.addWidget(cb)
        tog_row.addStretch()
        av.addLayout(tog_row)

        # Sliders
        row, self._rot_sld, _ = self._slider_row(
            "Rotation (±°):", 1, 45, int(aug_cfg.get("rotation_degrees", 15)), "°"
        )
        self._rot_sld.valueChanged.connect(self._generate)
        av.addLayout(row)

        row, self._bri_sld, _ = self._slider_row(
            "Helligkeit:", 5, 80, int(aug_cfg.get("brightness_strength", 0.3) * 100), "%"
        )
        self._bri_sld.valueChanged.connect(self._generate)
        av.addLayout(row)

        row, self._blur_sld, _ = self._slider_row(
            "Blur Radius:", 1, 11, int(aug_cfg.get("blur_radius", 3)), "px"
        )
        self._blur_sld.valueChanged.connect(self._generate)
        av.addLayout(row)

        row, self._scale_sld, _ = self._slider_row(
            "Skalierung min:", 50, 95, int(aug_cfg.get("scale_min", 0.8) * 100), "%"
        )
        self._scale_sld.valueChanged.connect(self._generate)
        av.addLayout(row)

        # Image size + regen
        bottom_row = QHBoxLayout()
        bottom_row.addWidget(QLabel("Bildgröße:"))
        self._size_spin = QSpinBox()
        self._size_spin.setRange(64, 512)
        self._size_spin.setSingleStep(32)
        self._size_spin.setValue(image_size)
        self._size_spin.valueChanged.connect(self._generate)
        bottom_row.addWidget(self._size_spin)
        bottom_row.addStretch()
        self._regen_btn = QPushButton("↺ Neu generieren")
        self._regen_btn.clicked.connect(self._generate)
        bottom_row.addWidget(self._regen_btn)
        av.addLayout(bottom_row)
        root.addWidget(aug_grp)

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

        total = 1 + self.N_SAMPLES
        self._cells: List[_ThumbCell] = []
        for i in range(total):
            caption = "Original" if i == 0 else f"Aug {i}"
            cell = _ThumbCell(caption)
            row_i, col = divmod(i, self.COLS)
            self._grid.addWidget(cell, row_i, col)
            self._cells.append(cell)

        # ── Buttons ─────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Schließen")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        accept_btn = QPushButton("Einstellungen übernehmen")
        accept_btn.setStyleSheet(
            "background:#1F6FEB;color:white;font-weight:bold;padding:5px 14px;"
        )
        accept_btn.setToolTip(
            "Aktuelle Augmentierungs-Einstellungen in die Trainingsseite übernehmen"
        )
        accept_btn.clicked.connect(self._accept_config)
        btn_row.addWidget(accept_btn)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------ helpers

    def _current_aug_cfg(self) -> Dict:
        return {
            "flip":                self._cb_flip.isChecked(),
            "rotation":            self._cb_rotation.isChecked(),
            "rotation_degrees":    self._rot_sld.value(),
            "brightness":          self._cb_brightness.isChecked(),
            "contrast":            self._cb_brightness.isChecked(),
            "brightness_strength": self._bri_sld.value() / 100.0,
            "scale":               self._cb_scale.isChecked(),
            "scale_min":           self._scale_sld.value() / 100.0,
            "blur":                self._cb_blur.isChecked(),
            "blur_radius":         self._blur_sld.value(),
        }

    def _accept_config(self) -> None:
        self.config_accepted.emit(self._current_aug_cfg())
        self.accept()

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
