"""
Dialog showing thumbnails for one cell of the confusion matrix.
Used for both misclassified (off-diagonal) and correctly classified (diagonal) cells.
"""
import os
from typing import List, Dict

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QAbstractItemView, QSizePolicy,
    QMessageBox,
)
from PySide6.QtCore import Qt, QSize, QRunnable, QThreadPool, QObject, Signal
from PySide6.QtGui import QPixmap, QIcon


# ---------------------------------------------------------------------------
# Async thumbnail loader (reused from thumbnail_list.py pattern)
# ---------------------------------------------------------------------------

class _ThumbSignals(QObject):
    loaded = Signal(str, QPixmap)


class _ThumbLoader(QRunnable):
    def __init__(self, path: str, size: int = 120):
        super().__init__()
        self._path   = path
        self._size   = size
        self.signals = _ThumbSignals()
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            pix = QPixmap(self._path)
            if not pix.isNull():
                pix = pix.scaled(
                    self._size, self._size,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.signals.loaded.emit(self._path, pix)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class MisclassifiedDialog(QDialog):
    """
    Shows thumbnails + filenames for all images in a confusion matrix cell.

    Parameters
    ----------
    samples      : list of dicts with keys 'path', 'true_label', 'pred_label'
    true_label   : true class name (row)
    pred_label   : predicted class name (column)
    """

    THUMB_SIZE = 120

    def __init__(
        self,
        samples: List[Dict],
        true_label: str,
        pred_label: str,
        parent=None,
    ):
        super().__init__(parent)
        self._samples   = samples
        self._true_label = true_label
        self._pred_label = pred_label
        self._pool = QThreadPool.globalInstance()
        self._cache: dict = {}

        is_correct = (true_label == pred_label)
        if is_correct:
            title = f"Korrekt klassifiziert: {true_label}  ({len(samples)} Bilder)"
        else:
            title = (
                f"Fehlklassifizierungen: {true_label} → {pred_label}"
                f"  ({len(samples)} Bilder)"
            )
        self.setWindowTitle(title)
        self.resize(860, 560)
        self._build_ui(title, is_correct)
        self._populate()

    # ------------------------------------------------------------------ UI

    def _build_ui(self, title: str, is_correct: bool) -> None:
        root = QVBoxLayout(self)

        # Banner
        banner = QLabel(title)
        banner.setWordWrap(True)
        color = "#1b5e20" if is_correct else "#6d1212"
        banner.setStyleSheet(
            f"background:{color};color:white;padding:8px 12px;"
            "border-radius:4px;font-weight:bold;font-size:12px;"
        )
        root.addWidget(banner)

        # Thumbnail grid
        self._list = QListWidget()
        self._list.setViewMode(QListWidget.IconMode)
        self._list.setIconSize(QSize(self.THUMB_SIZE, self.THUMB_SIZE))
        self._list.setSpacing(6)
        self._list.setResizeMode(QListWidget.Adjust)
        self._list.setMovement(QListWidget.Static)
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.setWordWrap(True)
        self._list.itemDoubleClicked.connect(self._open_image)
        root.addWidget(self._list, stretch=1)

        # Status
        self._status = QLabel(f"{len(self._samples)} Bilder  |  Doppelklick zum Öffnen")
        self._status.setStyleSheet("color:#888;font-size:10px;")
        root.addWidget(self._status)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        open_btn = QPushButton("Bild öffnen")
        open_btn.clicked.connect(lambda: self._open_image(self._list.currentItem()))
        btn_row.addWidget(open_btn)
        close_btn = QPushButton("Schließen")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------ populate

    def _populate(self) -> None:
        for sample in self._samples:
            path = sample.get("path", "")
            fname = os.path.basename(path)
            item = QListWidgetItem(fname)
            item.setData(Qt.UserRole, path)
            item.setTextAlignment(Qt.AlignCenter)
            item.setSizeHint(QSize(self.THUMB_SIZE + 20, self.THUMB_SIZE + 30))
            self._list.addItem(item)

            if path in self._cache:
                item.setIcon(QIcon(self._cache[path]))
            elif os.path.isfile(path):
                loader = _ThumbLoader(path, self.THUMB_SIZE)
                loader.signals.loaded.connect(self._on_loaded)
                self._pool.start(loader)

    def _on_loaded(self, path: str, pix: QPixmap) -> None:
        self._cache[path] = pix
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.UserRole) == path:
                item.setIcon(QIcon(pix))
                break

    # ------------------------------------------------------------------ open

    def _open_image(self, item: QListWidgetItem) -> None:
        if item is None:
            item = self._list.currentItem()
        if item is None:
            return
        path = item.data(Qt.UserRole)
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "Nicht gefunden", f"Datei nicht gefunden:\n{path}")
            return
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))
