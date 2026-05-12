"""
Dialog for importing frames from a video file into the project.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QProgressBar,
    QFileDialog, QMessageBox, QSizePolicy,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap


VIDEO_FILTERS = "Video-Dateien (*.mp4 *.avi *.mov *.mkv *.webm *.m4v)"
_THUMB_W = 320
_THUMB_H = 200


class _ExtractThread(QThread):
    progress = Signal(int, int)   # current, total
    finished = Signal(list)       # extracted paths
    error    = Signal(str)

    def __init__(self, video_path: str, out_dir: str, every_n: int):
        super().__init__()
        self.video_path = video_path
        self.out_dir = out_dir
        self.every_n = every_n
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error.emit(f"Video konnte nicht geöffnet werden:\n{self.video_path}")
            return

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            os.makedirs(self.out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(self.video_path))[0]
            extracted: list[str] = []
            idx = 0
            saved = 0

            while True:
                if self._cancel:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % self.every_n == 0:
                    fname = f"{base}_frame{idx:06d}.jpg"
                    fpath = os.path.join(self.out_dir, fname)
                    cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    extracted.append(fpath)
                    saved += 1
                idx += 1
                if total_frames > 0 and idx % 10 == 0:
                    self.progress.emit(idx, total_frames)

            self.progress.emit(total_frames, total_frames)
            self.finished.emit(extracted)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            cap.release()


class VideoImportDialog(QDialog):
    """
    Pick a video file, configure frame extraction, run extraction in background.
    After accept(), `extracted_paths` contains the saved frame paths.
    """

    def __init__(self, default_out_dir: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video-Frames importieren")
        self.setMinimumWidth(520)
        self.extracted_paths: list[str] = []
        self._video_path = ""
        self._default_out_dir = default_out_dir or os.path.expanduser("~")
        self._thread: _ExtractThread | None = None
        self._total_frames = 0
        self._fps = 0.0
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)

        # ── Video file ────────────────────────────────────────────────────
        file_box = QGroupBox("Videodatei")
        fl = QHBoxLayout(file_box)
        self._file_lbl = QLabel("(keine Datei gewählt)")
        self._file_lbl.setWordWrap(True)
        self._file_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        fl.addWidget(self._file_lbl)
        pick_btn = QPushButton("Wählen…")
        pick_btn.setFixedWidth(90)
        pick_btn.clicked.connect(self._pick_file)
        fl.addWidget(pick_btn)
        root.addWidget(file_box)

        # ── Preview ───────────────────────────────────────────────────────
        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setFixedSize(_THUMB_W, _THUMB_H)
        self._preview.setStyleSheet("background:#111; border-radius:6px; color:#555;")
        self._preview.setText("Vorschau")
        root.addWidget(self._preview, alignment=Qt.AlignHCenter)

        self._info_lbl = QLabel("")
        self._info_lbl.setStyleSheet("color:#aaa; font-size:11px;")
        self._info_lbl.setAlignment(Qt.AlignCenter)
        root.addWidget(self._info_lbl)

        # ── Extraction settings ────────────────────────────────────────────
        cfg_box = QGroupBox("Extraktionseinstellungen")
        cf = QFormLayout(cfg_box)

        self._every_n = QSpinBox()
        self._every_n.setRange(1, 300)
        self._every_n.setValue(10)
        self._every_n.setSuffix("  (jedes N-te Frame)")
        self._every_n.valueChanged.connect(self._update_estimate)
        cf.addRow("Intervall:", self._every_n)

        self._est_lbl = QLabel("")
        self._est_lbl.setStyleSheet("color:#888; font-size:10px;")
        cf.addRow("", self._est_lbl)

        out_row = QHBoxLayout()
        self._out_lbl = QLabel(self._default_out_dir)
        self._out_lbl.setWordWrap(True)
        self._out_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        out_row.addWidget(self._out_lbl)
        out_btn = QPushButton("…")
        out_btn.setFixedWidth(36)
        out_btn.clicked.connect(self._pick_out_dir)
        out_row.addWidget(out_btn)
        cf.addRow("Ausgabeordner:", out_row)

        root.addWidget(cfg_box)

        # ── Progress ──────────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color:#aaa; font-size:11px;")
        root.addWidget(self._status_lbl)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._extract_btn = QPushButton("Frames extrahieren")
        self._extract_btn.setStyleSheet(
            "background:#1976D2; color:white; font-weight:bold; padding:8px; border-radius:4px;"
        )
        self._extract_btn.setEnabled(False)
        self._extract_btn.clicked.connect(self._start_extraction)
        btn_row.addWidget(self._extract_btn)

        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        root.addLayout(btn_row)

    # ── file picking ──────────────────────────────────────────────────────

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Videodatei wählen", "", VIDEO_FILTERS)
        if not path:
            return
        self._video_path = path
        self._file_lbl.setText(os.path.basename(path))
        self._load_preview(path)
        self._extract_btn.setEnabled(True)

    def _load_preview(self, path: str) -> None:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._info_lbl.setText("Video konnte nicht gelesen werden.")
            cap.release()
            return

        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = self._total_frames / self._fps if self._fps > 0 else 0

        self._info_lbl.setText(
            f"{w}×{h} px  ·  {self._fps:.1f} fps  ·  "
            f"{self._total_frames} Frames  ·  {duration:.1f}s"
        )

        ret, frame = cap.read()
        cap.release()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rh, rw = rgb.shape[:2]
            img = QImage(rgb.data, rw, rh, rw * 3, QImage.Format_RGB888)
            pix = QPixmap.fromImage(img).scaled(
                _THUMB_W, _THUMB_H, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self._preview.setPixmap(pix)

        self._update_estimate()

    def _pick_out_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Ausgabeordner wählen", self._out_lbl.text())
        if folder:
            self._out_lbl.setText(folder)

    def _update_estimate(self) -> None:
        if self._total_frames <= 0:
            return
        n = max(1, self._every_n.value())
        est = self._total_frames // n
        fps_eff = self._fps / n if self._fps > 0 else 0
        self._est_lbl.setText(f"≈ {est} Frames werden extrahiert  ({fps_eff:.1f} eff. fps)")

    # ── extraction ────────────────────────────────────────────────────────

    def _start_extraction(self) -> None:
        if not self._video_path:
            return
        out_dir = self._out_lbl.text() or self._default_out_dir
        every_n = self._every_n.value()

        self._extract_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status_lbl.setText("Extrahiere Frames …")

        self._thread = _ExtractThread(self._video_path, out_dir, every_n)
        self._thread.progress.connect(self._on_progress)
        self._thread.finished.connect(self._on_finished)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    @Slot(int, int)
    def _on_progress(self, current: int, total: int) -> None:
        if total > 0:
            self._progress.setValue(int(current / total * 100))

    @Slot(list)
    def _on_finished(self, paths: list) -> None:
        self.extracted_paths = paths
        self._progress.setValue(100)
        self._status_lbl.setText(f"✓ {len(paths)} Frames extrahiert.")
        self._extract_btn.setEnabled(True)
        self.accept()

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._progress.setVisible(False)
        self._extract_btn.setEnabled(True)
        QMessageBox.critical(self, "Extraktionsfehler", msg)

    def closeEvent(self, event) -> None:
        if self._thread and self._thread.isRunning():
            self._thread.cancel()
            self._thread.wait(2000)
        super().closeEvent(event)
