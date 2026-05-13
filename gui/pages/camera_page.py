"""
Live-Kamera-Seite: eingebetteter Livestream mit Schnellzugriff auf den Aufnahme-Dialog.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from core.camera import apply_timestamp
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QCheckBox, QGroupBox, QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont


class CameraPage(QWidget):
    """Embedded live camera view with quick-launch button for full capture dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project = None
        self._cap: cv2.VideoCapture | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(50)   # ~20 fps
        self._timer.timeout.connect(self._grab_frame)

        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # ── Titel ──────────────────────────────────────────────────────────
        title = QLabel("📷  Live-Kamera")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #5DADE2;")
        root.addWidget(title)

        # ── Steuerleiste ───────────────────────────────────────────────────
        bar = QHBoxLayout()
        bar.setSpacing(10)

        bar.addWidget(QLabel("Kamera-Index:"))
        self._cam_spin = QSpinBox()
        self._cam_spin.setRange(0, 9)
        self._cam_spin.setValue(0)
        self._cam_spin.setFixedWidth(60)
        bar.addWidget(self._cam_spin)

        self._start_btn = QPushButton("▶  Kamera starten")
        self._start_btn.setFixedHeight(36)
        self._start_btn.setCursor(Qt.PointingHandCursor)
        self._start_btn.setStyleSheet(
            "QPushButton { background:#1976D2; color:white; border-radius:6px;"
            " padding:0 14px; font-weight:bold; }"
            "QPushButton:hover { background:#1565C0; }"
            "QPushButton:checked { background:#B71C1C; }"
        )
        self._start_btn.setCheckable(True)
        self._start_btn.toggled.connect(self._on_toggle)
        bar.addWidget(self._start_btn)

        self._ts_cb = QCheckBox("Zeitstempel einblenden")
        bar.addWidget(self._ts_cb)

        bar.addStretch()

        self._open_btn = QPushButton("⚙  Aufnahme & Anomalie-Erkennung …")
        self._open_btn.setFixedHeight(36)
        self._open_btn.setCursor(Qt.PointingHandCursor)
        self._open_btn.setStyleSheet(
            "QPushButton { background:#2E7D32; color:white; border-radius:6px;"
            " padding:0 14px; font-weight:bold; }"
            "QPushButton:hover { background:#1B5E20; }"
        )
        self._open_btn.clicked.connect(self._open_full_dialog)
        bar.addWidget(self._open_btn)

        root.addLayout(bar)

        # ── Preview ────────────────────────────────────────────────────────
        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._preview.setStyleSheet(
            "background:#111; border-radius:8px; color:#555; font-size:14px;"
        )
        self._preview.setText("Kamera noch nicht gestartet.\nKamera-Index wählen und ▶ drücken.")
        root.addWidget(self._preview, stretch=1)

        # ── Statuszeile ────────────────────────────────────────────────────
        self._status = QLabel("Kamera: inaktiv")
        self._status.setStyleSheet("color:#888; font-size:11px;")
        root.addWidget(self._status)

    # ── Projekt ────────────────────────────────────────────────────────────

    def set_project(self, project, audit=None) -> None:
        self._project = project

    # ── Kamera starten/stoppen ─────────────────────────────────────────────

    def _on_toggle(self, checked: bool) -> None:
        if checked:
            self._start_camera()
        else:
            self._stop_camera()

    def _start_camera(self) -> None:
        idx = self._cam_spin.value()
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            self._start_btn.setChecked(False)
            self._status.setText(f"Fehler: Kamera {idx} konnte nicht geöffnet werden.")
            return
        for _ in range(3):   # warm-up
            cap.read()
        self._cap = cap
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._start_btn.setText("⏹  Kamera stoppen")
        self._cam_spin.setEnabled(False)
        self._status.setText(f"Kamera {idx}  |  {w}×{h} px  |  live")
        self._timer.start()

    def _stop_camera(self) -> None:
        self._timer.stop()
        if self._cap:
            self._cap.release()
            self._cap = None
        self._start_btn.setText("▶  Kamera starten")
        self._start_btn.setChecked(False)
        self._cam_spin.setEnabled(True)
        self._preview.setPixmap(QPixmap())
        self._preview.setText("Kamera noch nicht gestartet.\nKamera-Index wählen und ▶ drücken.")
        self._status.setText("Kamera: inaktiv")

    def _grab_frame(self) -> None:
        if not self._cap:
            return
        ret, frame = self._cap.read()
        if not ret:
            self._stop_camera()
            self._status.setText("Kamera getrennt.")
            return

        if self._ts_cb.isChecked():
            frame = self._apply_timestamp(frame)

        self._show_frame(frame)

    # ── Hilfsmethoden ──────────────────────────────────────────────────────

    @staticmethod
    def _apply_timestamp(frame: np.ndarray) -> np.ndarray:
        return apply_timestamp(frame)

    def _show_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        pw = self._preview.width()
        ph = self._preview.height()
        self._preview.setPixmap(
            pix.scaled(pw, ph, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _open_full_dialog(self) -> None:
        was_running = self._timer.isActive()
        if was_running:
            self._stop_camera()

        from gui.camera_capture_dialog import CameraCaptureDialog
        save_dir = None
        if self._project and getattr(self._project, "project_path", None):
            save_dir = os.path.join(
                os.path.dirname(self._project.project_path), "camera_captures"
            )
        dlg = CameraCaptureDialog(save_dir=save_dir, parent=self)
        dlg.exec()

        if was_running:
            # Kurze Pause damit der Dialog-Thread die Kamera vollständig freigibt
            QTimer.singleShot(600, lambda: self._start_btn.setChecked(True))

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def hideEvent(self, event) -> None:
        """Kamera anhalten wenn die Seite verlassen wird."""
        if self._timer.isActive():
            self._stop_camera()
        super().hideEvent(event)
