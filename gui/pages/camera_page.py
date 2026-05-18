"""
Live monitoring page: load a trained anomaly model and observe a live camera feed.

This page is the "control room" — for frame collection and model training use
Datei → Kamera aufnehmen… (CameraCaptureDialog).
"""
from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

from core.camera import list_usb_cameras, apply_timestamp, CameraFrameThread
from core.anomaly_detector import AnomalyDetector
from core.alarm_notifier import AlarmNotifier
from core.industrial_notifier import IndustrialNotifier

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QSizePolicy,
    QComboBox, QSplitter, QScrollArea, QFrame, QProgressBar,
    QFileDialog, QMessageBox, QDialog, QTextBrowser, QInputDialog,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap


_SMOOTH_DEFAULT = 5
_DEDUP_DEFAULT  = 30   # seconds


# ── background camera scanner ─────────────────────────────────────────────────

class _ScanThread(QThread):
    """Background thread that enumerates connected USB cameras without blocking the UI."""

    done = Signal(list)   # list of (index, name)

    def run(self) -> None:
        """Call ``list_usb_cameras()`` and emit the result."""
        self.done.emit(list_usb_cameras())


# ── page ──────────────────────────────────────────────────────────────────────

class CameraPage(QWidget):
    """
    Embedded live monitoring view for anomaly detection with a pre-trained model.
    Workflow:
        1. Load trained .pth model
        2. Choose camera source
        3. Click Verbinden
        4. Enable Scoring
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project = None
        self._detector: Optional[AnomalyDetector] = None
        self._camera_thread: Optional[CameraFrameThread] = None
        self._model_path: Optional[str] = None
        self._scan_thread: Optional[_ScanThread] = None
        self._rest_server = None
        self._notifier: Optional[AlarmNotifier] = None
        self._industrial_notifier: Optional[IndustrialNotifier] = None
        self._last_frame: Optional[np.ndarray] = None

        # Scoring state
        self._smooth_buf: list[float] = []
        self._score_history: list[float] = []
        self._last_alarm_t: float = 0.0
        self._event_count: int = 0
        self._log_path: Optional[str] = None
        self._roi: Optional[list] = None  # [x1,y1,x2,y2] normalized, from model metadata

        self._build_ui()
        self._scan_cameras()

    # ── project ───────────────────────────────────────────────────────────────

    def set_project(self, project, audit=None) -> None:
        """Accept the active project and set up the anomaly-event CSV log path."""
        self._project = project
        if project and getattr(project, "project_path", None):
            log_dir = os.path.join(
                os.path.dirname(project.project_path), "anomaly_events"
            )
            os.makedirs(log_dir, exist_ok=True)
            self._log_path = os.path.join(log_dir, "monitoring_events.csv")
            if self._rest_server:
                self._rest_server.set_event_log_path(self._log_path)
                self._rest_server.set_alarm_frame_dir(log_dir)

    def set_rest_server(self, server) -> None:
        """Wire in the REST API server so live scores and alarm frames are pushed."""
        self._rest_server = server
        if self._log_path:
            server.set_event_log_path(self._log_path)
            server.set_alarm_frame_dir(os.path.dirname(self._log_path))

    def set_notifier(self, notifier: AlarmNotifier) -> None:
        """Inject the ``AlarmNotifier`` instance for e-mail/webhook alarm notifications."""
        self._notifier = notifier

    def set_industrial_notifier(self, notifier: IndustrialNotifier) -> None:
        """Inject the ``IndustrialNotifier`` instance for OPC-UA / Modbus TCP notifications."""
        self._industrial_notifier = notifier

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ── Title row ─────────────────────────────────────────────────────────
        title_row = QHBoxLayout()
        title = QLabel("📷  Live-Monitoring")
        title.setStyleSheet("font-size:18px; font-weight:bold; color:#5DADE2;")
        title_row.addWidget(title)
        title_row.addStretch()
        open_dlg_btn = QPushButton("⚙  Training & Aufnahme…")
        open_dlg_btn.setToolTip(
            "Vollständigen Kamera-Dialog öffnen:\n"
            "Frames sammeln · Autoencoder trainieren · Batch-Analyse · Aufnehmen"
        )
        open_dlg_btn.setStyleSheet(
            "QPushButton{background:#2E7D32;color:white;border-radius:6px;"
            "padding:5px 14px;font-weight:bold;}"
            "QPushButton:hover{background:#1B5E20;}"
        )
        open_dlg_btn.clicked.connect(self._open_capture_dialog)
        title_row.addWidget(open_dlg_btn)
        root.addLayout(title_row)

        # ── Model row ─────────────────────────────────────────────────────────
        model_row = QHBoxLayout()
        model_row.setSpacing(6)
        model_load_btn = QPushButton("Modell laden…")
        model_load_btn.setStyleSheet(
            "background:#1565C0;color:white;padding:5px 10px;"
            "border-radius:4px;font-weight:bold;"
        )
        model_load_btn.setToolTip("Trainierten Autoencoder (.pth) laden")
        model_load_btn.clicked.connect(self._load_model)
        model_row.addWidget(model_load_btn)

        self._model_lbl = QLabel("Kein Modell geladen  –  Bitte zuerst ein .pth Modell laden")
        self._model_lbl.setStyleSheet("color:#7F8C8D;")
        model_row.addWidget(self._model_lbl, stretch=1)

        self._model_info_btn = QPushButton("ℹ")
        self._model_info_btn.setFixedWidth(28)
        self._model_info_btn.setEnabled(False)
        self._model_info_btn.setToolTip("Modell-Metadaten anzeigen (Trainingszeit, Frames, SHA256 …)")
        self._model_info_btn.clicked.connect(self._show_model_info)
        model_row.addWidget(self._model_info_btn)

        self._onnx_export_btn = QPushButton("Als ONNX exportieren")
        self._onnx_export_btn.setEnabled(False)
        self._onnx_export_btn.setToolTip(
            "Trainiertes Modell als ONNX exportieren (.onnx + .meta.json)"
        )
        self._onnx_export_btn.setStyleSheet(
            "QPushButton{background:#6A1B9A;color:white;padding:5px 10px;"
            "border-radius:4px;font-weight:bold;}"
            "QPushButton:hover:enabled{background:#4A148C;}"
            "QPushButton:disabled{background:#2C3E50;color:#555;}"
        )
        self._onnx_export_btn.clicked.connect(self._export_onnx)
        model_row.addWidget(self._onnx_export_btn)

        root.addLayout(model_row)

        # ── Camera row ────────────────────────────────────────────────────────
        cam_row = QHBoxLayout()
        cam_row.setSpacing(8)
        cam_row.addWidget(QLabel("Kamera:"))
        self._cam_combo = QComboBox()
        self._cam_combo.setMinimumWidth(200)
        self._cam_combo.addItem("Suche läuft…")
        cam_row.addWidget(self._cam_combo)

        self._refresh_btn = QPushButton("↺")
        self._refresh_btn.setFixedWidth(30)
        self._refresh_btn.setToolTip("Kameras neu suchen")
        self._refresh_btn.setEnabled(False)
        self._refresh_btn.clicked.connect(self._scan_cameras)
        cam_row.addWidget(self._refresh_btn)

        self._connect_btn = QPushButton("Verbinden")
        self._connect_btn.setCheckable(True)
        self._connect_btn.setStyleSheet(
            "QPushButton{background:#27AE60;color:white;padding:5px 14px;"
            "border-radius:4px;font-weight:bold;}"
            "QPushButton:hover:!checked{background:#1E8449;}"
            "QPushButton:checked{background:#E74C3C;}"
            "QPushButton:checked:hover{background:#C0392B;}"
        )
        self._connect_btn.toggled.connect(self._on_connect_toggled)
        cam_row.addWidget(self._connect_btn)

        self._ts_cb = QCheckBox("Zeitstempel")
        self._ts_cb.setToolTip("Datum/Uhrzeit ins Vorschaubild einblenden")
        cam_row.addWidget(self._ts_cb)

        self._conn_status_lbl = QLabel("Nicht verbunden")
        self._conn_status_lbl.setStyleSheet("color:#E74C3C;")
        cam_row.addWidget(self._conn_status_lbl)
        cam_row.addStretch()
        root.addLayout(cam_row)

        # ── Alarm banner ──────────────────────────────────────────────────────
        self._alarm_banner = QLabel("  ⚠  ANOMALIE ERKANNT")
        self._alarm_banner.setAlignment(Qt.AlignCenter)
        self._alarm_banner.setStyleSheet(
            "background:#C0392B;color:white;font-weight:bold;"
            "font-size:15px;padding:6px;border-radius:4px;"
        )
        self._alarm_banner.setVisible(False)
        root.addWidget(self._alarm_banner)

        # ── Main splitter ─────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)

        # Left control panel
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(260)
        left_scroll.setStyleSheet("QScrollArea{border:none;}")

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 4, 8, 4)
        lv.setSpacing(8)

        # Score display
        score_grp = QGroupBox("Anomalie-Score")
        sg = QVBoxLayout(score_grp)

        self._score_bar = QProgressBar()
        self._score_bar.setRange(0, 100)
        self._score_bar.setValue(0)
        self._score_bar.setTextVisible(False)
        self._score_bar.setFixedHeight(12)
        self._score_bar.setStyleSheet(
            "QProgressBar{background:#1A252F;border-radius:5px;}"
            "QProgressBar::chunk{background:#27AE60;border-radius:5px;}"
        )
        sg.addWidget(self._score_bar)

        self._score_lbl = QLabel("Score: –")
        self._score_lbl.setAlignment(Qt.AlignCenter)
        self._score_lbl.setStyleSheet(
            "font-weight:bold;font-size:18px;padding:6px;"
            "border-radius:5px;background:#1A252F;color:#7F8C8D;"
        )
        sg.addWidget(self._score_lbl)
        lv.addWidget(score_grp)

        # Detection controls
        det_grp = QGroupBox("Erkennung")
        df = QVBoxLayout(det_grp)

        self._scoring_btn = QPushButton("Scoring aktivieren")
        self._scoring_btn.setCheckable(True)
        self._scoring_btn.setEnabled(False)
        self._scoring_btn.setToolTip(
            "Anomalie-Scoring aktivieren.\n"
            "Voraussetzung: Modell geladen + Kamera verbunden."
        )
        self._scoring_btn.setStyleSheet(
            "QPushButton{background:#2C3E50;color:#BDC3C7;border:1px solid #34495E;"
            "border-radius:4px;padding:6px;font-weight:bold;}"
            "QPushButton:enabled:!checked:hover{background:#34495E;color:white;}"
            "QPushButton:enabled:checked{background:#27AE60;color:white;border:none;}"
            "QPushButton:enabled:checked:hover{background:#1E8449;}"
        )
        self._scoring_btn.toggled.connect(self._on_scoring_toggled)
        df.addWidget(self._scoring_btn)

        self._heatmap_cb = QCheckBox("Heatmap-Overlay anzeigen")
        self._heatmap_cb.setToolTip(
            "Überlagert das Live-Bild mit einer Fehlerwärmekarte.\n"
            "Rot = hoher Rekonstruktionsfehler = potenzielle Anomalie."
        )
        df.addWidget(self._heatmap_cb)

        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Schwellwert:"))
        self._thr_spin = QDoubleSpinBox()
        self._thr_spin.setRange(0.00001, 1.0)
        self._thr_spin.setDecimals(5)
        self._thr_spin.setSingleStep(0.001)
        self._thr_spin.setValue(0.02)
        self._thr_spin.setEnabled(False)
        self._thr_spin.setToolTip(
            "Automatisch gesetzt beim Laden des Modells.\n"
            "Erhöhen = weniger sensitiv · Senken = sensitiver."
        )
        self._thr_spin.valueChanged.connect(self._on_threshold_changed)
        thr_row.addWidget(self._thr_spin)
        df.addLayout(thr_row)

        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("Glättung:"))
        self._smooth_spin = QSpinBox()
        self._smooth_spin.setRange(1, 20)
        self._smooth_spin.setValue(_SMOOTH_DEFAULT)
        self._smooth_spin.setSuffix(" Fr.")
        self._smooth_spin.setToolTip(
            "Alarm erst nach N aufeinanderfolgenden Frames über dem Schwellwert.\n"
            "Verhindert Fehlalarme durch kurze Störungen."
        )
        smooth_row.addWidget(self._smooth_spin)
        df.addLayout(smooth_row)

        dedup_row = QHBoxLayout()
        dedup_row.addWidget(QLabel("Alarm-Pause:"))
        self._dedup_spin = QSpinBox()
        self._dedup_spin.setRange(0, 3600)
        self._dedup_spin.setValue(_DEDUP_DEFAULT)
        self._dedup_spin.setSuffix(" s")
        self._dedup_spin.setToolTip(
            "Mindestabstand in Sekunden zwischen zwei protokollierten Alarm-Events.\n"
            "0 = alle Frames loggen."
        )
        dedup_row.addWidget(self._dedup_spin)
        df.addLayout(dedup_row)

        lv.addWidget(det_grp)

        # Event log
        ev_grp = QGroupBox("Ereignisse")
        ev = QVBoxLayout(ev_grp)
        self._event_lbl = QLabel("0 Alarme in dieser Sitzung")
        self._event_lbl.setStyleSheet("color:#7F8C8D; font-size:11px;")
        ev.addWidget(self._event_lbl)
        self._log_btn = QPushButton("Log öffnen")
        self._log_btn.setEnabled(False)
        self._log_btn.setToolTip("CSV-Ereignislog öffnen")
        self._log_btn.clicked.connect(self._open_log)
        ev.addWidget(self._log_btn)
        lv.addWidget(ev_grp)

        # Score chart (optional widget)
        self._score_chart = None
        try:
            from gui.widgets.score_chart import ScoreChart
            self._score_chart = ScoreChart()
            self._score_chart.setToolTip("Live-Verlauf der Anomalie-Scores")
            lv.addWidget(self._score_chart)
        except Exception:
            pass

        lv.addStretch()
        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)

        # Right — live preview
        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._preview.setStyleSheet(
            "background:#111; border-radius:8px; color:#555; font-size:14px;"
        )
        self._preview.setText(
            "Kein Signal\n\n"
            "① Modell laden (.pth)\n"
            "② Kamera wählen → Verbinden\n"
            "③ Scoring aktivieren"
        )
        splitter.addWidget(self._preview)
        splitter.setSizes([290, 720])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter, stretch=1)

        # Status bar
        self._status_bar = QLabel("Bereit  –  Modell laden und Kamera verbinden um zu starten.")
        self._status_bar.setStyleSheet("color:#888; font-size:11px;")
        root.addWidget(self._status_bar)

    # ── Camera scan ───────────────────────────────────────────────────────────

    def _scan_cameras(self) -> None:
        """Start a background ``_ScanThread`` to enumerate available cameras."""
        self._prev_cam = self._cam_combo.currentData()  # save BEFORE clearing
        self._cam_combo.clear()
        self._cam_combo.addItem("Suche läuft…")
        self._refresh_btn.setEnabled(False)
        self._scan_thread = _ScanThread(self)
        self._scan_thread.done.connect(self._on_scan_done)
        self._scan_thread.start()

    @Slot(list)
    def _on_scan_done(self, cams: list) -> None:
        """Populate the camera combo box with the discovered camera list."""
        prev = self._prev_cam
        self._cam_combo.clear()
        if cams:
            for idx, name in cams:
                self._cam_combo.addItem(name, userData=idx)
        else:
            self._cam_combo.addItem("Keine USB-Kamera gefunden")
        self._cam_combo.addItem("IP-Kamera (URL eingeben…)", userData="ip")
        self._refresh_btn.setEnabled(True)
        # Restore previous selection so a refresh doesn't silently reset the user's choice
        if prev is not None:
            for i in range(self._cam_combo.count()):
                if self._cam_combo.itemData(i) == prev:
                    self._cam_combo.setCurrentIndex(i)
                    break

    # ── Connect / Disconnect ──────────────────────────────────────────────────

    def _on_connect_toggled(self, checked: bool) -> None:
        """Start or stop the camera stream when the connect button is toggled."""
        if checked:
            self._start_stream()
        else:
            self._stop_stream()

    def _start_stream(self) -> None:
        """Open the selected camera source and start the ``CameraFrameThread``."""
        source = self._cam_combo.currentData()

        if source == "ip":
            url, ok = QInputDialog.getText(
                self, "IP-Kamera URL",
                "Kamera-URL (RTSP / HTTP):",
                text="rtsp://user:pass@192.168.1.100:554/stream",
            )
            if not ok or not url.strip():
                self._connect_btn.setChecked(False)
                return
            source = url.strip()
            _valid_schemes = ("rtsp://", "rtsps://", "http://", "https://")
            if not any(source.lower().startswith(s) for s in _valid_schemes):
                QMessageBox.warning(
                    self, "Ungültige URL",
                    "Die URL hat kein unterstütztes Schema.\n\n"
                    "Erlaubte Formate:\n"
                    "  rtsp://user:pass@192.168.1.100:554/stream\n"
                    "  http://192.168.1.100:8080/video\n"
                    "  https://...\n\n"
                    "Bitte URL korrigieren und erneut versuchen."
                )
                self._connect_btn.setChecked(False)
                return
        elif source is None:
            self._connect_btn.setChecked(False)
            return

        self._camera_thread = CameraFrameThread(source, fps=15.0, parent=self)
        self._camera_thread.frame_ready.connect(self._on_frame)
        self._camera_thread.error.connect(self._on_camera_error)
        self._camera_thread.start()

        self._connect_btn.setText("Trennen")
        self._conn_status_lbl.setText("Verbunden")
        self._conn_status_lbl.setStyleSheet("color:#2ECC71;")
        cam_name = self._cam_combo.currentText()
        self._status_bar.setText(f"Verbunden: {cam_name}")

        # Enable scoring if model already loaded
        if self._detector and self._detector.trained:
            self._scoring_btn.setEnabled(True)

    def _stop_stream(self) -> None:
        """Stop the camera thread and reset all live-monitoring UI elements."""
        self._scoring_btn.setChecked(False)
        self._scoring_btn.setEnabled(False)
        if self._camera_thread:
            self._camera_thread.stop()
            self._camera_thread = None
        self._connect_btn.setText("Verbinden")
        self._connect_btn.setChecked(False)
        self._conn_status_lbl.setText("Nicht verbunden")
        self._conn_status_lbl.setStyleSheet("color:#E74C3C;")
        self._preview.setPixmap(QPixmap())
        self._preview.setText(
            "Kein Signal\n\n"
            "① Modell laden (.pth)\n"
            "② Kamera wählen → Verbinden\n"
            "③ Scoring aktivieren"
        )
        self._score_lbl.setText("Score: –")
        self._score_lbl.setStyleSheet(
            "font-weight:bold;font-size:18px;padding:6px;"
            "border-radius:5px;background:#1A252F;color:#7F8C8D;"
        )
        self._score_bar.setValue(0)
        self._alarm_banner.setVisible(False)
        self._smooth_buf.clear()
        self._score_history.clear()
        self._status_bar.setText("Bereit")

    @Slot(str)
    def _on_camera_error(self, msg: str) -> None:
        """Handle camera errors by disabling scoring and updating the status label."""
        self._scoring_btn.setEnabled(False)
        self._connect_btn.setChecked(False)
        self._conn_status_lbl.setText("Fehler")
        self._conn_status_lbl.setStyleSheet("color:#E74C3C;")
        self._status_bar.setText(f"Kamera-Fehler: {msg}")

    # ── Scoring toggle ────────────────────────────────────────────────────────

    def _on_scoring_toggled(self, active: bool) -> None:
        """Reset score history and update button text when scoring is toggled."""
        self._scoring_btn.setText("Scoring aktiv  ●" if active else "Scoring aktivieren")
        self._smooth_buf.clear()
        self._score_history.clear()
        if not active:
            self._alarm_banner.setVisible(False)
            self._score_lbl.setText("Score: –")
            self._score_lbl.setStyleSheet(
                "font-weight:bold;font-size:18px;padding:6px;"
                "border-radius:5px;background:#1A252F;color:#7F8C8D;"
            )
            self._score_bar.setValue(0)

    # ── Frame handling ────────────────────────────────────────────────────────

    @Slot(object)
    def _on_frame(self, frame: np.ndarray) -> None:
        """
        Process each incoming camera frame.

        When scoring is active: compute the anomaly score, optionally render the
        heatmap overlay, draw the ROI rectangle, and call ``_update_score``.
        Always displays the (possibly annotated) frame in the preview label.
        """
        self._last_frame = frame
        display = frame

        if self._scoring_btn.isChecked() and self._detector and self._detector.trained:
            analysis = self._apply_roi_crop(frame)
            score, _rec, overlay, _bbox = self._detector.score_detailed(analysis)
            if self._heatmap_cb.isChecked():
                if self._roi:
                    display = frame.copy()
                    h, w = display.shape[:2]
                    x1 = int(self._roi[0] * w); y1 = int(self._roi[1] * h)
                    x2 = int(self._roi[2] * w); y2 = int(self._roi[3] * h)
                    rw, rh = max(1, x2 - x1), max(1, y2 - y1)
                    display[y1:y2, x1:x2] = cv2.resize(
                        overlay, (rw, rh), interpolation=cv2.INTER_LINEAR
                    )
                else:
                    display = overlay
            self._update_score(score)

        # Draw cyan ROI rectangle so the monitored region is always visible
        if self._roi is not None:
            if display is frame:
                display = frame.copy()
            h, w = display.shape[:2]
            x1 = int(self._roi[0] * w); y1 = int(self._roi[1] * h)
            x2 = int(self._roi[2] * w); y2 = int(self._roi[3] * h)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

        if self._ts_cb.isChecked():
            display = apply_timestamp(display)

        self._show_frame(display)

    def _apply_roi_crop(self, frame: np.ndarray) -> np.ndarray:
        """
        Crop *frame* to the ROI region stored in ``self._roi`` (normalised coords).

        Returns the full frame unchanged when no ROI is configured or the
        computed pixel dimensions are too small.
        """
        if self._roi is None:
            return frame
        h, w = frame.shape[:2]
        x1 = int(self._roi[0] * w); y1 = int(self._roi[1] * h)
        x2 = int(self._roi[2] * w); y2 = int(self._roi[3] * h)
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return frame
        return frame[y1:y2, x1:x2]

    def _update_score(self, score: float) -> None:
        """
        Update the score bar, score label, chart, and alarm banner for *score*.

        Applies a rolling-average smoothing buffer. When the smoothed score
        exceeds the threshold for N consecutive frames (configured by the
        smoothing spin) an alarm event is logged (with deduplication).
        """
        thr = self._thr_spin.value()

        self._smooth_buf.append(score)
        n = self._smooth_spin.value()
        if len(self._smooth_buf) > n:
            self._smooth_buf = self._smooth_buf[-n:]
        avg = sum(self._smooth_buf) / len(self._smooth_buf)
        is_anomaly = avg > thr

        # Score bar (capped at 3× threshold for visual range)
        bar_pct = int(min(score / max(3 * thr, 1e-9), 1.0) * 100)
        self._score_bar.setValue(bar_pct)
        bar_color = "#E74C3C" if is_anomaly else "#27AE60"
        self._score_bar.setStyleSheet(
            f"QProgressBar{{background:#1A252F;border-radius:5px;}}"
            f"QProgressBar::chunk{{background:{bar_color};border-radius:5px;}}"
        )

        label_color = "#E74C3C" if is_anomaly else "#27AE60"
        state_text  = "⚠  ANOMALIE" if is_anomaly else "✓  Normal"
        self._score_lbl.setText(f"Score: {score:.5f}    {state_text}")
        self._score_lbl.setStyleSheet(
            f"font-weight:bold;font-size:14px;padding:6px;"
            f"border-radius:5px;background:#1A252F;color:{label_color};"
        )

        if self._score_chart:
            self._score_history.append(score)
            self._score_chart.update_data(self._score_history, thr)

        if self._rest_server:
            self._rest_server.push_score(score, thr)

        self._alarm_banner.setVisible(is_anomaly)

        if self._industrial_notifier:
            self._industrial_notifier.on_alarm(is_anomaly, avg, thr)

        # Log with dedup
        if is_anomaly:
            now = time.perf_counter()
            dedup = self._dedup_spin.value()
            if dedup == 0 or (now - self._last_alarm_t) >= dedup:
                self._last_alarm_t = now
                self._event_count += 1
                plural = "e" if self._event_count != 1 else ""
                self._event_lbl.setText(f"{self._event_count} Alarm{plural} in dieser Sitzung")
                self._write_event(score, thr)

    def _write_event(self, score: float, threshold: float) -> None:
        """
        Append one alarm event to the CSV log and save a JPEG snapshot.

        The CSV header is written automatically on the first call. A JPEG
        snapshot of the current frame is saved to the same directory and its
        filename is pushed to the REST API server.
        """
        if not self._log_path:
            return
        ts = datetime.now(timezone.utc)
        log_dir = os.path.dirname(self._log_path)

        # Save the current frame as a JPEG alarm snapshot
        frame_filename = ""
        if self._last_frame is not None:
            frame_filename = f"alarm_{ts.strftime('%Y%m%dT%H%M%SZ')}.jpg"
            frame_path = os.path.join(log_dir, frame_filename)
            try:
                cv2.imwrite(frame_path, self._last_frame)
                if self._rest_server:
                    self._rest_server.push_latest_alarm(frame_path, score, threshold)
            except Exception:
                frame_filename = ""

        if self._notifier and frame_filename:
            model_name = os.path.basename(self._model_path) if self._model_path else ""
            self._notifier.notify(
                score, threshold,
                frame_path=os.path.join(log_dir, frame_filename) if frame_filename else "",
                model_name=model_name,
            )

        score_pct = int(score / threshold * 100) if threshold > 0 else 0
        write_header = not os.path.exists(self._log_path)
        try:
            with open(self._log_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["timestamp_utc", "score", "threshold", "score_pct", "model", "frame_path"])
                w.writerow([
                    ts.isoformat(),
                    f"{score:.6f}",
                    f"{threshold:.6f}",
                    str(score_pct),
                    os.path.basename(self._model_path) if self._model_path else "",
                    frame_filename,
                ])
            self._log_btn.setEnabled(True)
        except Exception:
            pass

    # ── Frame display ─────────────────────────────────────────────────────────

    def _show_frame(self, frame: np.ndarray) -> None:
        """Convert a BGR numpy frame to a QPixmap and display it in the preview label."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        pw, ph = self._preview.width(), self._preview.height()
        self._preview.setPixmap(
            pix.scaled(pw, ph, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    # ── Model ─────────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Open a file chooser to select and load an autoencoder .pth checkpoint."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Autoencoder-Modell laden", "", "PyTorch-Modell (*.pth)"
        )
        if not path:
            return
        try:
            det = AnomalyDetector()
            det.load(path)
        except Exception as e:
            QMessageBox.critical(self, "Fehler beim Laden", str(e))
            return

        self._detector = det
        self._model_path = path
        self._roi = det.metadata.get("roi")  # [x1,y1,x2,y2] normalized, or None
        name = os.path.basename(path)
        roi_tag = "  ·  ROI aktiv" if self._roi else ""
        self._model_lbl.setText(f"{name}  (Schwellwert: {det.threshold:.5f}{roi_tag})")
        self._model_lbl.setStyleSheet("color:#2ECC71;")
        self._thr_spin.blockSignals(True)
        self._thr_spin.setValue(det.threshold)
        self._thr_spin.blockSignals(False)
        self._thr_spin.setEnabled(True)
        self._model_info_btn.setEnabled(True)
        self._onnx_export_btn.setEnabled(True)

        # Enable scoring button only when camera is also connected
        if self._camera_thread:
            self._scoring_btn.setEnabled(True)

        self._status_bar.setText(
            f"Modell geladen: {name}  |  Schwellwert: {det.threshold:.5f}{roi_tag}"
        )

    def _show_model_info(self) -> None:
        """Display an HTML dialog with the loaded model's metadata dictionary."""
        if not self._detector:
            return
        meta = self._detector.metadata
        rows = "".join(
            f"<tr><td style='color:#85C1E9;padding:3px 10px 3px 0'><b>{k}</b></td>"
            f"<td style='color:#ECF0F1'>{v}</td></tr>"
            for k, v in meta.items()
        )
        html = (
            "<style>body{font-family:-apple-system,sans-serif;font-size:12px;"
            "background:#0D1117;color:#E0E0E0;}</style>"
            f"<table>{rows}</table>"
        ) if rows else "<p style='color:#aaa'>Keine Metadaten gespeichert.</p>"

        dlg = QDialog(self)
        dlg.setWindowTitle("Modell-Informationen")
        dlg.resize(500, 380)
        dv = QVBoxLayout(dlg)
        tb = QTextBrowser()
        tb.setHtml(html)
        tb.setStyleSheet("background:#0D1117; border:none;")
        dv.addWidget(tb)
        close_btn = QPushButton("Schließen")
        close_btn.clicked.connect(dlg.accept)
        dv.addWidget(close_btn)
        dlg.exec()

    def _on_threshold_changed(self, val: float) -> None:
        """Propagate a spin-box threshold change to the loaded ``AnomalyDetector``."""
        if self._detector:
            self._detector.threshold = val

    # ── Event log ─────────────────────────────────────────────────────────────

    def _open_log(self) -> None:
        if not self._log_path or not os.path.exists(self._log_path):
            return
        import subprocess
        import platform
        if platform.system() == "Darwin":
            subprocess.Popen(["open", self._log_path])
        elif platform.system() == "Windows":
            os.startfile(self._log_path)
        else:
            subprocess.Popen(["xdg-open", self._log_path])

    # ── Capture dialog ────────────────────────────────────────────────────────

    def _open_capture_dialog(self) -> None:
        was_streaming = bool(self._camera_thread)
        if was_streaming:
            self._stop_stream()

        from gui.camera_capture_dialog import CameraCaptureDialog
        save_dir = None
        if self._project and getattr(self._project, "project_path", None):
            save_dir = os.path.join(
                os.path.dirname(self._project.project_path), "camera_captures"
            )
        dlg = CameraCaptureDialog(save_dir=save_dir, parent=self)
        dlg.exec()

        # If a model was trained during the session, offer to load it here
        model_path = dlg.trained_model_path
        if model_path and os.path.exists(model_path):
            reply = QMessageBox.question(
                self,
                "Modell in Live-Monitoring laden?",
                f"Ein Autoencoder wurde trainiert und gespeichert:\n"
                f"{os.path.basename(model_path)}\n\n"
                f"Jetzt ins Live-Monitoring laden?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self._load_model_from_path(model_path)

        if was_streaming:
            QTimer.singleShot(600, lambda: self._connect_btn.setChecked(True))

    def _load_model_from_path(self, path: str) -> None:
        """Load a model directly by path (no file dialog)."""
        try:
            det = AnomalyDetector()
            det.load(path)
        except Exception as e:
            QMessageBox.critical(self, "Fehler beim Laden", str(e))
            return
        self._detector = det
        self._model_path = path
        self._roi = det.metadata.get("roi")  # [x1,y1,x2,y2] normalized, or None
        name = os.path.basename(path)
        roi_tag = "  ·  ROI aktiv" if self._roi else ""
        self._model_lbl.setText(f"{name}  (Schwellwert: {det.threshold:.5f}{roi_tag})")
        self._model_lbl.setStyleSheet("color:#2ECC71;")
        self._thr_spin.blockSignals(True)
        self._thr_spin.setValue(det.threshold)
        self._thr_spin.blockSignals(False)
        self._thr_spin.setEnabled(True)
        self._model_info_btn.setEnabled(True)
        self._onnx_export_btn.setEnabled(True)
        if self._camera_thread:
            self._scoring_btn.setEnabled(True)
        self._status_bar.setText(
            f"Modell geladen: {name}  |  Schwellwert: {det.threshold:.5f}{roi_tag}"
        )

    def _export_onnx(self) -> None:
        """Export the currently loaded model to ONNX format with a metadata sidecar."""
        if self._detector is None or not self._detector.trained:
            return
        base = os.path.splitext(self._model_path or "anomalie_modell")[0]
        default_path = base + ".onnx"
        path, _ = QFileDialog.getSaveFileName(
            self, "ONNX exportieren", default_path, "ONNX (*.onnx)"
        )
        if not path:
            return
        try:
            self._detector.export_onnx_with_meta(path)
            QMessageBox.information(
                self,
                "ONNX exportiert",
                f"Modell exportiert:\n{path}\n\nMetadaten:\n{path}.meta.json",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Fehler", str(exc))

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def hideEvent(self, event) -> None:
        if self._camera_thread:
            self._stop_stream()
        super().hideEvent(event)
