from __future__ import annotations
import os
import shutil
import tempfile
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox, QListWidget, QListWidgetItem, QGroupBox,
    QSplitter, QProgressBar, QFileDialog, QMessageBox, QSizePolicy,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QPixmap, QImage

_CV2_AVAILABLE = False
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    pass


class _FrameExtractThread(QThread):
    progress = Signal(int, int)
    frame_ready = Signal(int, str)
    finished = Signal(int)
    error = Signal(str)

    def __init__(self, video_path: str, interval: int, output_dir: str, parent=None):
        super().__init__(parent)
        self._video_path = video_path
        self._interval = max(1, interval)
        self._output_dir = output_dir
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        if not _CV2_AVAILABLE:
            self.error.emit("opencv-python ist nicht installiert (pip install opencv-python).")
            return
        try:
            cap = cv2.VideoCapture(self._video_path)
            if not cap.isOpened():
                self.error.emit(f"Kann Video nicht öffnen: {self._video_path}")
                return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            extracted = 0
            frame_no = 0
            os.makedirs(self._output_dir, exist_ok=True)
            while not self._stop:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_no % self._interval == 0:
                    path = os.path.join(self._output_dir, f"frame_{frame_no:06d}.jpg")
                    cv2.imwrite(path, frame)
                    self.frame_ready.emit(frame_no, path)
                    extracted += 1
                    if total > 0:
                        self.progress.emit(frame_no, total)
                frame_no += 1
            cap.release()
            self.finished.emit(extracted)
        except Exception as exc:
            self.error.emit(str(exc))


class VideoAnnotationPage(QWidget):
    """
    Seite für Frame-für-Frame Video-Beschriftung.
    Workflow: Video laden → Frames extrahieren → Label vergeben → In Projekt speichern.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.project = None
        self._frame_paths: list[str] = []
        self._frame_labels: dict[int, str] = {}
        self._extract_thread: Optional[_FrameExtractThread] = None
        self._temp_dir: Optional[str] = None
        self._current_frame_idx: int = 0
        self._build_ui()

    def set_project(self, project, audit=None) -> None:
        self.project = project
        self._refresh_label_list()

    def closeEvent(self, event) -> None:
        self._cleanup_temp()
        super().closeEvent(event)

    def hideEvent(self, event) -> None:
        if self._extract_thread and self._extract_thread.isRunning():
            self._extract_thread.stop()
            self._extract_thread.wait(2000)
        super().hideEvent(event)

    def _cleanup_temp(self) -> None:
        if self._temp_dir and os.path.isdir(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass
        self._temp_dir = None

    def _build_ui(self) -> None:
        from utils.i18n import tr
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        title = QLabel(tr("videoannotation.title"))
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #E6EDF3;")
        root.addWidget(title)

        top_bar = QHBoxLayout()
        self._load_btn = QPushButton(tr("videoannotation.load_btn"))
        self._load_btn.setStyleSheet("background: #1F6FEB; color: white; border-radius: 4px; padding: 5px 12px; font-weight: bold;")
        self._load_btn.clicked.connect(self._load_video)
        top_bar.addWidget(self._load_btn)

        self._video_lbl = QLabel(tr("videoannotation.no_video"))
        self._video_lbl.setStyleSheet("color: #8B949E;")
        top_bar.addWidget(self._video_lbl, stretch=1)

        top_bar.addWidget(QLabel(tr("videoannotation.frame_interval")))
        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(1, 100)
        self._interval_spin.setValue(5)
        self._interval_spin.setToolTip("1 = jeder Frame, 5 = jeder 5. Frame")
        self._interval_spin.setFixedWidth(70)
        top_bar.addWidget(self._interval_spin)
        root.addLayout(top_bar)

        self._extract_progress = QProgressBar()
        self._extract_progress.setRange(0, 100)
        self._extract_progress.setVisible(False)
        self._extract_progress.setFixedHeight(8)
        root.addWidget(self._extract_progress)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)

        self._frame_view = QLabel("Kein Frame")
        self._frame_view.setAlignment(Qt.AlignCenter)
        self._frame_view.setMinimumSize(320, 240)
        self._frame_view.setStyleSheet("background: #0D1117; border: 1px solid #30363D; border-radius: 4px; color: #4A5568;")
        self._frame_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self._frame_view)

        self._frame_info = QLabel("Frame – / –")
        self._frame_info.setAlignment(Qt.AlignCenter)
        self._frame_info.setStyleSheet("color: #8B949E; font-size: 11px;")
        left_layout.addWidget(self._frame_info)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self._on_slider_changed)
        left_layout.addWidget(self._slider)

        nav_row = QHBoxLayout()
        prev_btn = QPushButton(tr("videoannotation.prev_btn"))
        prev_btn.clicked.connect(lambda: self._slider.setValue(max(0, self._slider.value() - 1)))
        next_btn = QPushButton(tr("videoannotation.next_btn"))
        next_btn.clicked.connect(lambda: self._slider.setValue(min(self._slider.maximum(), self._slider.value() + 1)))
        nav_row.addWidget(prev_btn)
        nav_row.addStretch()
        nav_row.addWidget(next_btn)
        left_layout.addLayout(nav_row)

        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)

        lbl_grp = QGroupBox(tr("videoannotation.label_group"))
        lbl_grp.setStyleSheet("QGroupBox { font-weight: bold; color: #8B949E; border: 1px solid #30363D; border-radius: 6px; margin-top: 8px; padding-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 8px; }")
        lbl_v = QVBoxLayout(lbl_grp)

        self._label_list = QListWidget()
        self._label_list.setStyleSheet("QListWidget { background: #0D1117; border: 1px solid #30363D; color: #E6EDF3; border-radius: 4px; } QListWidget::item:selected { background: #1F6FEB; }")
        lbl_v.addWidget(self._label_list)

        assign_btn = QPushButton(tr("videoannotation.assign_btn"))
        assign_btn.setStyleSheet("background: #238636; color: white; border-radius: 4px; padding: 5px 12px; font-weight: bold;")
        assign_btn.clicked.connect(self._assign_label)
        lbl_v.addWidget(assign_btn)

        self._current_label_lbl = QLabel(tr("videoannotation.no_label"))
        self._current_label_lbl.setStyleSheet("color: #8B949E; font-size: 11px;")
        lbl_v.addWidget(self._current_label_lbl)

        right_layout.addWidget(lbl_grp)

        self._status_lbl = QLabel("Kein Frame gelabelt.")
        self._status_lbl.setStyleSheet("color: #8B949E; font-size: 11px;")
        right_layout.addWidget(self._status_lbl)
        right_layout.addStretch()

        splitter.addWidget(right)
        splitter.setSizes([480, 220])
        root.addWidget(splitter, stretch=1)

        save_btn = QPushButton(tr("videoannotation.save_btn"))
        save_btn.setStyleSheet("background: #6A1B9A; color: white; border-radius: 4px; padding: 7px 16px; font-weight: bold;")
        save_btn.clicked.connect(self._save_to_project)
        root.addWidget(save_btn)

    def _refresh_label_list(self) -> None:
        self._label_list.clear()
        if self.project and self.project.labels:
            for name in self.project.labels:
                self._label_list.addItem(name)

    @Slot()
    def _load_video(self) -> None:
        from utils.i18n import tr
        path, _ = QFileDialog.getOpenFileName(
            self, tr("videoannotation.video_dlg"),
            "",
            "Video-Dateien (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv);;Alle Dateien (*)"
        )
        if not path:
            return
        self._cleanup_temp()
        self._frame_paths.clear()
        self._frame_labels.clear()
        self._slider.setRange(0, 0)
        self._frame_view.setText(tr("videoannotation.extracting"))

        self._temp_dir = tempfile.mkdtemp(prefix="va_frames_")
        interval = self._interval_spin.value()
        self._extract_thread = _FrameExtractThread(path, interval, self._temp_dir, self)
        self._extract_thread.progress.connect(self._on_extract_progress)
        self._extract_thread.frame_ready.connect(self._on_frame_ready)
        self._extract_thread.finished.connect(self._on_extract_finished)
        self._extract_thread.error.connect(self._on_extract_error)
        self._extract_progress.setVisible(True)
        self._extract_progress.setValue(0)
        self._load_btn.setEnabled(False)
        self._video_lbl.setText(os.path.basename(path))
        self._extract_thread.start()

    @Slot(int, int)
    def _on_extract_progress(self, current: int, total: int) -> None:
        if total > 0:
            self._extract_progress.setValue(int(current / total * 100))

    @Slot(int, str)
    def _on_frame_ready(self, frame_no: int, path: str) -> None:
        self._frame_paths.append(path)
        n = len(self._frame_paths)
        self._slider.setRange(0, max(0, n - 1))
        if n == 1:
            self._show_frame(0)

    @Slot(int)
    def _on_extract_finished(self, total: int) -> None:
        from utils.i18n import tr
        self._extract_progress.setVisible(False)
        self._load_btn.setEnabled(True)
        self._status_lbl.setText(tr("videoannotation.extracted", n=total))

    @Slot(str)
    def _on_extract_error(self, msg: str) -> None:
        from utils.i18n import tr
        self._extract_progress.setVisible(False)
        self._load_btn.setEnabled(True)
        QMessageBox.critical(self, tr("common.error"), msg)

    @Slot(int)
    def _on_slider_changed(self, idx: int) -> None:
        self._show_frame(idx)

    def _show_frame(self, idx: int) -> None:
        from utils.i18n import tr
        if not self._frame_paths or idx >= len(self._frame_paths):
            return
        self._current_frame_idx = idx
        path = self._frame_paths[idx]
        pix = QPixmap(path)
        if not pix.isNull():
            vw = self._frame_view.width() or 320
            vh = self._frame_view.height() or 240
            self._frame_view.setPixmap(pix.scaled(vw, vh, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        n = len(self._frame_paths)
        self._frame_info.setText(f"Frame {idx + 1} / {n}")
        lbl = self._frame_labels.get(idx, "")
        self._current_label_lbl.setText(f"Label: {lbl}" if lbl else tr("videoannotation.no_label"))

    @Slot()
    def _assign_label(self) -> None:
        items = self._label_list.selectedItems()
        if not items:
            return
        lbl = items[0].text()
        idx = self._current_frame_idx
        self._frame_labels[idx] = lbl
        self._current_label_lbl.setText(f"Label: {lbl}")
        labeled = len(self._frame_labels)
        self._status_lbl.setText(f"{labeled} Frame(s) gelabelt.")

    @Slot()
    def _save_to_project(self) -> None:
        from utils.i18n import tr
        if not self.project:
            QMessageBox.warning(self, tr("common.warning"), tr("common.no_project"))
            return
        if not self._frame_labels:
            QMessageBox.information(self, tr("common.info"), tr("videoannotation.no_labels_msg"))
            return
        save_dir = os.path.join(os.path.dirname(self.project.project_path or "."), "video_frames")
        os.makedirs(save_dir, exist_ok=True)
        added = 0
        for idx, label in self._frame_labels.items():
            if idx >= len(self._frame_paths):
                continue
            src = self._frame_paths[idx]
            if not os.path.isfile(src):
                continue
            fname = os.path.basename(src)
            dst = os.path.join(save_dir, fname)
            try:
                shutil.copy2(src, dst)
                if self.project.add_image(dst):
                    self.project.image_labels[dst] = label
                    added += 1
            except Exception:
                pass
        try:
            self.project.save()
        except Exception:
            pass
        QMessageBox.information(
            self, tr("videoannotation.saved_title"),
            tr("videoannotation.saved", n=added) + f":\n{save_dir}"
        )
        self._status_lbl.setText(f"{added} Frames ins Projekt übertragen.")
