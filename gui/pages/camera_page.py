"""
Live monitoring page: load a trained anomaly model and observe a live camera feed.

This page is the "control room" — for frame collection and model training use
Datei → Kamera aufnehmen… (CameraCaptureDialog).
"""
from __future__ import annotations

import csv
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

from core.camera import (
    list_usb_cameras, apply_timestamp, CameraFrameThread,
    apply_cam_props, apply_frame_filter, check_camera_permission,
)
from core.anomaly_detector import AnomalyDetector
from core.alarm_notifier import AlarmNotifier
from core.industrial_notifier import IndustrialNotifier

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QSizePolicy,
    QComboBox, QSplitter, QScrollArea, QFrame, QProgressBar,
    QFileDialog, QMessageBox, QDialog, QTextBrowser, QInputDialog,
    QSlider, QFormLayout,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, Slot
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut

from utils.i18n import tr

log = logging.getLogger(__name__)

_SMOOTH_DEFAULT = 5
_DEDUP_DEFAULT  = 30   # seconds
_RETRAIN_THRESHOLD = 20  # logged alarms before auto-retrain banner appears


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

        # Auto-reconnect state
        self._reconnect_source = None   # source string/int to reconnect to; None = don't reconnect
        self._reconnect_attempts: int = 0
        self._is_video_file: bool = False
        self._reconnect_timer = QTimer(self)
        self._reconnect_timer.setSingleShot(True)
        self._reconnect_timer.timeout.connect(self._try_reconnect)

        # Scoring state
        self._smooth_buf: list[float] = []
        self._score_history: list[float] = []
        self._last_alarm_t: float = 0.0
        self._event_count: int = 0
        self._log_path: Optional[str] = None
        self._roi: Optional[list] = None  # [x1,y1,x2,y2] normalized, from model metadata

        # USP 2 — auto-retrain suggestion
        self._session_alarm_count: int = 0

        # USP 3 — shadow / A-B model comparison
        self._shadow_detector: Optional[AnomalyDetector] = None
        self._shadow_model_path: Optional[str] = None
        self._shadow_log_path: Optional[str] = None

        self._build_ui()
        self._scan_cameras()
        QShortcut(QKeySequence(Qt.Key_Space), self,
                  activated=lambda: self._scoring_btn.toggle()
                  if self._scoring_btn.isEnabled() else None)

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
            self._shadow_log_path = os.path.join(log_dir, "shadow_divergences.csv")
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
        title = QLabel(tr("camera.title"))
        title.setStyleSheet("font-size:18px; font-weight:bold; color:#5DADE2;")
        title_row.addWidget(title)
        title_row.addStretch()
        open_dlg_btn = QPushButton(tr("camera.open_dialog_btn"))
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
        model_load_btn = QPushButton(tr("camera.model_load_btn"))
        model_load_btn.setStyleSheet(
            "background:#1565C0;color:white;padding:5px 10px;"
            "border-radius:4px;font-weight:bold;"
        )
        model_load_btn.setToolTip("Trainierten Autoencoder (.pth) laden")
        model_load_btn.clicked.connect(self._load_model)
        model_row.addWidget(model_load_btn)

        self._model_lbl = QLabel(tr("camera.no_model"))
        self._model_lbl.setStyleSheet("color:#7F8C8D;")
        model_row.addWidget(self._model_lbl, stretch=1)

        self._model_info_btn = QPushButton("ℹ")
        self._model_info_btn.setFixedWidth(28)
        self._model_info_btn.setEnabled(False)
        self._model_info_btn.setToolTip("Modell-Metadaten anzeigen (Trainingszeit, Frames, SHA256 …)")
        self._model_info_btn.clicked.connect(self._show_model_info)
        model_row.addWidget(self._model_info_btn)

        self._onnx_export_btn = QPushButton(tr("camera.onnx_export_btn"))
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

        # ── Shadow model row (USP 3) ──────────────────────────────────────────
        shadow_row = QHBoxLayout()
        shadow_row.setSpacing(6)
        shadow_load_btn = QPushButton(tr("camera.shadow_load_btn"))
        shadow_load_btn.setStyleSheet(
            "background:#7D3C98;color:white;padding:5px 10px;"
            "border-radius:4px;font-weight:bold;"
        )
        shadow_load_btn.setToolTip(
            "Zweites Anomalie-Modell (Shadow) laden.\n"
            "Beide Modelle laufen parallel — Abweichungen werden geloggt."
        )
        shadow_load_btn.clicked.connect(self._load_shadow_model)
        shadow_row.addWidget(shadow_load_btn)

        self._shadow_lbl = QLabel(tr("camera.shadow_no_model"))
        self._shadow_lbl.setStyleSheet("color:#7F8C8D;")
        shadow_row.addWidget(self._shadow_lbl, stretch=1)

        self._shadow_clear_btn = QPushButton("✕")
        self._shadow_clear_btn.setFixedWidth(28)
        self._shadow_clear_btn.setEnabled(False)
        self._shadow_clear_btn.setToolTip("Shadow-Modell entfernen")
        self._shadow_clear_btn.clicked.connect(self._clear_shadow_model)
        shadow_row.addWidget(self._shadow_clear_btn)

        root.addLayout(shadow_row)

        # ── Camera row ────────────────────────────────────────────────────────
        cam_row = QHBoxLayout()
        cam_row.setSpacing(8)
        cam_row.addWidget(QLabel(tr("camera.cam_label")))
        self._cam_combo = QComboBox()
        self._cam_combo.setMinimumWidth(200)
        self._cam_combo.addItem(tr("camera.search_running"))
        cam_row.addWidget(self._cam_combo)

        self._refresh_btn = QPushButton(tr("camera.refresh_btn"))
        self._refresh_btn.setFixedWidth(30)
        self._refresh_btn.setToolTip(tr("camera.refresh_tooltip"))
        self._refresh_btn.setEnabled(False)
        self._refresh_btn.clicked.connect(self._scan_cameras)
        cam_row.addWidget(self._refresh_btn)

        self._connect_btn = QPushButton(tr("camera.connect_btn"))
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

        self._ts_cb = QCheckBox(tr("camera.timestamp_cb"))
        self._ts_cb.setToolTip("Datum/Uhrzeit ins Vorschaubild einblenden")
        cam_row.addWidget(self._ts_cb)

        self._conn_status_lbl = QLabel("Nicht verbunden")
        self._conn_status_lbl.setStyleSheet("color:#E74C3C;")
        cam_row.addWidget(self._conn_status_lbl)
        cam_row.addStretch()
        root.addLayout(cam_row)

        # ── Alarm banner ──────────────────────────────────────────────────────
        self._alarm_banner = QLabel(tr("camera.alarm_banner"))
        self._alarm_banner.setAlignment(Qt.AlignCenter)
        self._alarm_banner.setStyleSheet(
            "background:#C0392B;color:white;font-weight:bold;"
            "font-size:15px;padding:6px;border-radius:4px;"
        )
        self._alarm_banner.setVisible(False)
        root.addWidget(self._alarm_banner)

        # ── Retrain suggestion banner (USP 2) ─────────────────────────────────
        self._retrain_banner = QFrame()
        self._retrain_banner.setStyleSheet(
            "QFrame{background:#1A5276;border-radius:4px;}"
        )
        rb_lay = QHBoxLayout(self._retrain_banner)
        rb_lay.setContentsMargins(10, 4, 6, 4)
        rb_lay.setSpacing(8)
        self._retrain_lbl = QLabel("")
        self._retrain_lbl.setStyleSheet("color:white;font-weight:bold;font-size:12px;")
        rb_lay.addWidget(self._retrain_lbl)
        rb_lay.addStretch()
        retrain_go_btn = QPushButton(tr("camera.retrain_go_btn"))
        retrain_go_btn.setStyleSheet(
            "QPushButton{background:#2E86C1;color:white;border-radius:4px;"
            "padding:3px 10px;font-weight:bold;}"
            "QPushButton:hover{background:#1A5276;}"
        )
        retrain_go_btn.clicked.connect(self._on_retrain_suggested)
        rb_lay.addWidget(retrain_go_btn)
        retrain_dismiss_btn = QPushButton("✕")
        retrain_dismiss_btn.setFixedWidth(24)
        retrain_dismiss_btn.setToolTip("Banner schließen (Zähler zurücksetzen)")
        retrain_dismiss_btn.clicked.connect(self._dismiss_retrain_banner)
        rb_lay.addWidget(retrain_dismiss_btn)
        self._retrain_banner.setVisible(False)
        root.addWidget(self._retrain_banner)

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
        score_grp = QGroupBox(tr("camera.score_group"))
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

        # Shadow score display (USP 3) — hidden until a shadow model is loaded
        self._shadow_bar = QProgressBar()
        self._shadow_bar.setRange(0, 100)
        self._shadow_bar.setValue(0)
        self._shadow_bar.setTextVisible(False)
        self._shadow_bar.setFixedHeight(8)
        self._shadow_bar.setStyleSheet(
            "QProgressBar{background:#1A252F;border-radius:5px;}"
            "QProgressBar::chunk{background:#E67E22;border-radius:5px;}"
        )
        self._shadow_bar.setVisible(False)
        sg.addWidget(self._shadow_bar)

        self._shadow_score_lbl = QLabel("")
        self._shadow_score_lbl.setAlignment(Qt.AlignCenter)
        self._shadow_score_lbl.setStyleSheet(
            "font-size:11px;padding:2px;color:#E67E22;"
        )
        self._shadow_score_lbl.setVisible(False)
        sg.addWidget(self._shadow_score_lbl)

        self._divergence_lbl = QLabel("")
        self._divergence_lbl.setAlignment(Qt.AlignCenter)
        self._divergence_lbl.setStyleSheet("font-size:11px;color:#85929E;padding:2px;")
        self._divergence_lbl.setVisible(False)
        sg.addWidget(self._divergence_lbl)

        lv.addWidget(score_grp)

        # Detection controls
        det_grp = QGroupBox(tr("camera.detection_group"))
        df = QVBoxLayout(det_grp)

        self._scoring_btn = QPushButton(tr("camera.scoring_btn"))
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

        self._heatmap_cb = QCheckBox(tr("camera.heatmap_cb"))
        self._heatmap_cb.setToolTip(
            "Überlagert das Live-Bild mit einer Fehlerwärmekarte.\n"
            "Rot = hoher Rekonstruktionsfehler = potenzielle Anomalie."
        )
        df.addWidget(self._heatmap_cb)

        self._gradcam_cb = QCheckBox(tr("camera.gradcam_cb"))
        self._gradcam_cb.setToolTip(
            "Grad-CAM: zeigt welche Bildregionen den Anomalie-Score verursachen.\n"
            "Langsamer als Heatmap — deaktiviert Heatmap automatisch wenn aktiv."
        )
        df.addWidget(self._gradcam_cb)

        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel(tr("camera.threshold_label")))
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
        smooth_row.addWidget(QLabel(tr("camera.smooth_label")))
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
        dedup_row.addWidget(QLabel(tr("camera.dedup_label")))
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
        ev_grp = QGroupBox(tr("camera.events_group"))
        ev = QVBoxLayout(ev_grp)
        self._event_lbl = QLabel(tr("camera.events_label"))
        self._event_lbl.setStyleSheet("color:#7F8C8D; font-size:11px;")
        ev.addWidget(self._event_lbl)
        self._log_btn = QPushButton(tr("camera.log_btn"))
        self._log_btn.setEnabled(False)
        self._log_btn.setToolTip("CSV-Ereignislog öffnen")
        self._log_btn.clicked.connect(self._open_log)
        ev.addWidget(self._log_btn)
        lv.addWidget(ev_grp)

        # Alarm statistics
        stats_grp = QGroupBox(tr("camera.stats_group"))
        stats_layout = QVBoxLayout(stats_grp)
        self._stats_session_lbl = QLabel(tr("camera.stats_session", n=0))
        self._stats_session_lbl.setStyleSheet("color:#7F8C8D; font-size:11px;")
        self._stats_today_lbl = QLabel(tr("camera.stats_today", n="–"))
        self._stats_today_lbl.setStyleSheet("color:#7F8C8D; font-size:11px;")
        self._stats_week_lbl = QLabel(tr("camera.stats_week", n="–"))
        self._stats_week_lbl.setStyleSheet("color:#7F8C8D; font-size:11px;")
        for lbl in (self._stats_session_lbl, self._stats_today_lbl, self._stats_week_lbl):
            stats_layout.addWidget(lbl)
        lv.addWidget(stats_grp)

        # Score chart (optional widget)
        self._score_chart = None
        try:
            from gui.widgets.score_chart import ScoreChart
            self._score_chart = ScoreChart()
            self._score_chart.setToolTip("Live-Verlauf der Anomalie-Scores")
            lv.addWidget(self._score_chart)
        except Exception:
            pass

        # ── Camera settings ───────────────────────────────────────────────────
        cam_settings_grp = QGroupBox(tr("camera.settings_group"))
        cam_settings_grp.setCheckable(True)
        cam_settings_grp.setChecked(False)  # collapsed by default
        cs = QFormLayout(cam_settings_grp)
        cs.setSpacing(4)

        def _make_prop_slider(minimum, maximum, default, prop_name):
            row = QHBoxLayout()
            sl = QSlider(Qt.Horizontal)
            sl.setRange(minimum, maximum)
            sl.setValue(default)
            sl.setFixedHeight(18)
            val_lbl = QLabel(str(default))
            val_lbl.setFixedWidth(30)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(sl)
            row.addWidget(val_lbl)
            def _on_change(v):
                val_lbl.setText(str(v))
                self._apply_cam_prop(prop_name, v)
            sl.valueChanged.connect(_on_change)
            return row, sl

        br_row, self._brightness_sl = _make_prop_slider(-64, 64, 0, "brightness")
        cs.addRow(tr("camera.brightness_label"), br_row)
        ct_row, self._contrast_sl = _make_prop_slider(0, 95, 0, "contrast")
        cs.addRow(tr("camera.contrast_label"), ct_row)
        sat_row, self._saturation_sl = _make_prop_slider(0, 100, 0, "saturation")
        cs.addRow(tr("camera.saturation_label"), sat_row)
        sh_row, self._sharpness_sl = _make_prop_slider(0, 7, 0, "sharpness")
        cs.addRow(tr("camera.sharpness_label"), sh_row)
        exp_row, self._exposure_sl = _make_prop_slider(-13, -1, -6, "exposure")
        cs.addRow(tr("camera.exposure_label"), exp_row)

        reset_cam_btn = QPushButton(tr("camera.reset_btn"))
        reset_cam_btn.setFixedHeight(24)
        reset_cam_btn.clicked.connect(self._reset_cam_settings)
        cs.addRow("", reset_cam_btn)

        lv.addWidget(cam_settings_grp)
        self._cam_settings_grp = cam_settings_grp

        # ── Preprocessing filter ──────────────────────────────────────────────
        filter_grp = QGroupBox(tr("camera.filter_group"))
        ff = QFormLayout(filter_grp)
        ff.setSpacing(4)

        self._filter_combo = QComboBox()
        self._filter_combo.addItem(tr("camera.filter_none"), "none")
        self._filter_combo.addItem(tr("camera.filter_grayscale"), "grayscale")
        self._filter_combo.addItem(tr("camera.filter_canny"), "canny")
        self._filter_combo.addItem(tr("camera.filter_sobel"), "sobel")
        self._filter_combo.addItem(tr("camera.filter_laplacian"), "laplacian")
        ff.addRow(tr("camera.filter_label"), self._filter_combo)

        self._filter_scoring_cb = QCheckBox(tr("camera.filter_scoring_cb"))
        self._filter_scoring_cb.setToolTip(
            "Wenn aktiv, sieht der Autoencoder den gefilterten Frame.\n"
            "Nur sinnvoll wenn das Modell auch auf gefilterten Frames trainiert wurde."
        )
        ff.addRow("", self._filter_scoring_cb)

        lv.addWidget(filter_grp)

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
        self._status_bar = QLabel(tr("camera.status_ready"))
        self._status_bar.setStyleSheet("color:#888; font-size:11px;")
        root.addWidget(self._status_bar)

    # ── Camera scan ───────────────────────────────────────────────────────────

    def _scan_cameras(self) -> None:
        """Start a background ``_ScanThread`` to enumerate available cameras."""
        self._prev_cam = self._cam_combo.currentData()  # save BEFORE clearing
        self._cam_combo.clear()
        self._cam_combo.addItem(tr("camera.search_running"))
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
            self._cam_combo.addItem(tr("camera.no_camera"))
        self._cam_combo.addItem(tr("camera.ip_option"), userData="ip")
        self._cam_combo.addItem(tr("camera.video_file_option"), userData="video")
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

        is_video = False

        if source == "ip":
            url, ok = QInputDialog.getText(
                self, tr("camera.ip_dialog_title"),
                tr("camera.ip_dialog_prompt"),
                text="rtsp://user:pass@192.168.1.100:554/stream",
            )
            if not ok or not url.strip():
                self._connect_btn.setChecked(False)
                return
            source = url.strip()
            _valid_schemes = ("rtsp://", "rtsps://", "http://", "https://")
            if not any(source.lower().startswith(s) for s in _valid_schemes):
                QMessageBox.warning(
                    self, tr("camera.invalid_url_title"),
                    "Die URL hat kein unterstütztes Schema.\n\n"
                    "Erlaubte Formate:\n"
                    "  rtsp://user:pass@192.168.1.100:554/stream\n"
                    "  http://192.168.1.100:8080/video\n"
                    "  https://...\n\n"
                    "Bitte URL korrigieren und erneut versuchen."
                )
                self._connect_btn.setChecked(False)
                return
        elif source == "video":
            path, _ = QFileDialog.getOpenFileName(
                self, tr("camera.video_file_dlg"), "",
                "Video-Dateien (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv *.flv);;"
                "Alle Dateien (*)"
            )
            if not path:
                self._connect_btn.setChecked(False)
                return
            source = path
            is_video = True
        elif source is None:
            self._connect_btn.setChecked(False)
            return
        elif isinstance(source, int):
            ok, msg = check_camera_permission(source)
            if not ok:
                QMessageBox.warning(self, tr("camera.permission_error_title"), msg)
                self._connect_btn.setChecked(False)
                return

        self._is_video_file = is_video
        # Only live streams get auto-reconnect
        self._reconnect_source = None if is_video else source
        self._reconnect_attempts = 0
        self._reconnect_timer.stop()

        # Detect native FPS for video files; fall back to 25 fps
        fps = 15.0
        if is_video:
            cap = cv2.VideoCapture(source)
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            fps = native_fps if native_fps and native_fps > 0 else 25.0

        self._camera_thread = CameraFrameThread(source, fps=fps, parent=self)
        self._camera_thread.frame_ready.connect(self._on_frame)
        self._camera_thread.error.connect(self._on_camera_error)
        self._camera_thread.start()

        self._connect_btn.setText(tr("camera.disconnect_btn"))
        if is_video:
            fname = os.path.basename(source)
            self._conn_status_lbl.setText(f"Wiedergabe: {fname}")
            self._conn_status_lbl.setStyleSheet("color:#2ECC71;")
            self._status_bar.setText(f"Video: {source}  ({fps:.1f} fps)")
        else:
            self._conn_status_lbl.setText("Verbunden")
            self._conn_status_lbl.setStyleSheet("color:#2ECC71;")
            cam_name = self._cam_combo.currentText()
            self._status_bar.setText(f"Verbunden: {cam_name}")

        # Enable scoring if model already loaded
        if self._detector and self._detector.trained:
            self._scoring_btn.setEnabled(True)

    def _stop_stream(self) -> None:
        """Stop the camera thread and reset all live-monitoring UI elements."""
        # Disable reconnect so the error handler doesn't restart the stream
        self._reconnect_source = None
        self._reconnect_timer.stop()
        self._is_video_file = False

        self._scoring_btn.setChecked(False)
        self._scoring_btn.setEnabled(False)
        if self._camera_thread:
            self._camera_thread.stop()
            self._camera_thread = None
        self._connect_btn.setText(tr("camera.connect_btn"))
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
        self._shadow_bar.setValue(0)
        self._shadow_score_lbl.setText("")
        self._divergence_lbl.setText("")
        self._status_bar.setText("Bereit")

    @Slot(str)
    def _on_camera_error(self, msg: str) -> None:
        """Handle camera errors; auto-reconnect for live streams (not video files)."""
        self._scoring_btn.setEnabled(False)
        self._connect_btn.blockSignals(True)
        self._connect_btn.setChecked(False)
        self._connect_btn.blockSignals(False)

        if self._reconnect_source is not None and not self._is_video_file:
            self._conn_status_lbl.setText(tr("camera.reconnecting"))
            self._conn_status_lbl.setStyleSheet("color:#D29922;")
            self._status_bar.setText(f"Kamera-Fehler: {msg} — Reconnect läuft…")
            self._reconnect_timer.start(5000)
        else:
            label = "Video beendet" if self._is_video_file else "Fehler"
            self._conn_status_lbl.setText(label)
            self._conn_status_lbl.setStyleSheet("color:#E74C3C;")
            self._status_bar.setText(
                "Video abgespielt." if self._is_video_file else f"Kamera-Fehler: {msg}"
            )

    def _try_reconnect(self) -> None:
        """Attempt to restart the stream after a connection error."""
        source = self._reconnect_source
        if source is None:
            return
        self._reconnect_attempts += 1
        self._conn_status_lbl.setText(f"Reconnect #{self._reconnect_attempts}…")
        self._conn_status_lbl.setStyleSheet("color:#D29922;")
        self._status_bar.setText(f"Verbinde erneut… (Versuch {self._reconnect_attempts})")

        if self._camera_thread:
            self._camera_thread.stop()
            self._camera_thread = None

        self._camera_thread = CameraFrameThread(source, fps=15.0, parent=self)
        self._camera_thread.frame_ready.connect(self._on_frame)
        self._camera_thread.error.connect(self._on_camera_error)
        self._camera_thread.start()

        self._connect_btn.blockSignals(True)
        self._connect_btn.setChecked(True)
        self._connect_btn.setText(tr("camera.disconnect_btn"))
        self._connect_btn.blockSignals(False)

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

        # Apply preprocessing filter to display frame
        filter_key = self._filter_combo.currentData()
        display = apply_frame_filter(frame, filter_key) if filter_key != "none" else frame
        # Optionally also filter the frame used for scoring
        score_input = display if self._filter_scoring_cb.isChecked() else frame

        # First frame after reconnect — restore green status
        if self._reconnect_attempts > 0:
            self._reconnect_attempts = 0
            self._conn_status_lbl.setText("Verbunden")
            self._conn_status_lbl.setStyleSheet("color:#2ECC71;")
            self._status_bar.setText("Verbindung wiederhergestellt.")
            if self._detector and self._detector.trained:
                self._scoring_btn.setEnabled(True)

        if self._scoring_btn.isChecked() and self._detector and self._detector.trained:
            analysis = self._apply_roi_crop(score_input)
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
            # Grad-CAM overlay (mutually exclusive with heatmap for performance)
            if self._gradcam_cb.isChecked():
                from core.gradcam import compute_gradcam_anomaly
                try:
                    display = compute_gradcam_anomaly(self._detector, display)
                except Exception:
                    pass
            self._update_score(score)

            # USP 3 — shadow model comparison (same ROI-cropped input for fair comparison)
            if self._shadow_detector and self._shadow_detector.trained:
                shadow_score, _, _, _ = self._shadow_detector.score_detailed(analysis)
                self._update_shadow_score(score, shadow_score)

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
                self._stats_session_lbl.setText(tr("camera.stats_session", n=self._event_count))
                self._write_event(score, thr)
                # USP 2 — auto-retrain suggestion
                self._session_alarm_count += 1
                if self._session_alarm_count >= _RETRAIN_THRESHOLD:
                    self._retrain_lbl.setText(
                        f"⚠  {self._session_alarm_count} Alarme in dieser Sitzung  —  "
                        "Retraining empfohlen"
                    )
                    self._retrain_banner.setVisible(True)

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

    # ── USP 2: Auto-retrain suggestion ───────────────────────────────────────

    def _on_retrain_suggested(self) -> None:
        """Navigate to the Training page and dismiss the retrain banner."""
        self._dismiss_retrain_banner()
        mw = self.window()
        if hasattr(mw, "_switch_page"):
            mw._switch_page(3)

    def _dismiss_retrain_banner(self) -> None:
        """Hide the retrain banner and reset the session alarm counter."""
        self._session_alarm_count = 0
        self._retrain_banner.setVisible(False)

    # ── USP 3: Shadow model / A-B comparison ─────────────────────────────────

    def _load_shadow_model(self) -> None:
        """Load a second AnomalyDetector to run in parallel for A-B comparison."""
        path, _ = QFileDialog.getOpenFileName(
            self, tr("camera.shadow_load_title"), "", "PyTorch-Modell (*.pth)"
        )
        if not path:
            return
        try:
            det = AnomalyDetector()
            det.load(path)
        except Exception as exc:
            QMessageBox.critical(self, tr("common.error"), str(exc))
            return
        self._shadow_detector = det
        self._shadow_model_path = path
        name = os.path.basename(path)
        self._shadow_lbl.setText(f"Shadow: {name}  (Thr: {det.threshold:.5f})")
        self._shadow_lbl.setStyleSheet("color:#E67E22;")
        self._shadow_clear_btn.setEnabled(True)
        self._shadow_bar.setVisible(True)
        self._shadow_score_lbl.setVisible(True)
        self._divergence_lbl.setVisible(True)
        self._status_bar.setText(f"Shadow-Modell geladen: {name}")

    def _clear_shadow_model(self) -> None:
        """Unload the shadow model and hide comparison widgets."""
        self._shadow_detector = None
        self._shadow_model_path = None
        self._shadow_lbl.setText(tr("camera.shadow_no_model"))
        self._shadow_lbl.setStyleSheet("color:#7F8C8D;")
        self._shadow_clear_btn.setEnabled(False)
        self._shadow_bar.setValue(0)
        self._shadow_bar.setVisible(False)
        self._shadow_score_lbl.setText("")
        self._shadow_score_lbl.setVisible(False)
        self._divergence_lbl.setText("")
        self._divergence_lbl.setVisible(False)

    def _update_shadow_score(self, primary: float, shadow: float) -> None:
        """Update the shadow score bar and divergence indicator."""
        thr = self._thr_spin.value()
        bar_pct = int(min(shadow / max(3 * thr, 1e-9), 1.0) * 100)
        self._shadow_bar.setValue(bar_pct)
        shadow_alarm = shadow > thr
        bar_color = "#E74C3C" if shadow_alarm else "#E67E22"
        self._shadow_bar.setStyleSheet(
            f"QProgressBar{{background:#1A252F;border-radius:5px;}}"
            f"QProgressBar::chunk{{background:{bar_color};border-radius:5px;}}"
        )
        self._shadow_score_lbl.setText(f"Shadow: {shadow:.5f}")

        primary_alarm = primary > thr
        diff = abs(primary - shadow)
        if primary_alarm != shadow_alarm:
            self._divergence_lbl.setText(f"⚡ Divergenz Δ{diff:.5f}")
            self._divergence_lbl.setStyleSheet(
                "font-size:11px;color:#E67E22;font-weight:bold;padding:2px;"
            )
            self._log_shadow_divergence(primary, shadow, thr)
        else:
            self._divergence_lbl.setText(f"Δ{diff:.5f}")
            self._divergence_lbl.setStyleSheet("font-size:11px;color:#85929E;padding:2px;")

    def _log_shadow_divergence(self, primary: float, shadow: float, thr: float) -> None:
        """Append a divergence event to the shadow-divergences CSV."""
        if not self._shadow_log_path:
            return
        ts = datetime.now(timezone.utc)
        write_header = not os.path.exists(self._shadow_log_path)
        try:
            with open(self._shadow_log_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow([
                        "timestamp_utc", "primary_score", "shadow_score",
                        "threshold", "primary_model", "shadow_model",
                    ])
                w.writerow([
                    ts.isoformat(),
                    f"{primary:.6f}",
                    f"{shadow:.6f}",
                    f"{thr:.6f}",
                    os.path.basename(self._model_path) if self._model_path else "",
                    os.path.basename(self._shadow_model_path) if self._shadow_model_path else "",
                ])
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
            QMessageBox.critical(self, tr("common.error"), str(e))
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

        # Collect current camera properties from sliders
        current_cam_props = {}
        prop_names = ["brightness", "contrast", "saturation", "sharpness", "exposure"]
        for prop in prop_names:
            sl = getattr(self, f"_{prop}_sl", None)
            if sl is not None:
                current_cam_props[prop] = sl.value()

        # Current preprocessing filter
        current_filter = getattr(self._filter_combo, "currentData", lambda: "none")() or "none"

        save_dir = None
        if self._project and getattr(self._project, "project_path", None):
            save_dir = os.path.join(
                os.path.dirname(self._project.project_path), "camera_captures"
            )

        dlg = CameraCaptureDialog(
            save_dir=save_dir,
            cam_props=current_cam_props,
            filter_name=current_filter,
            parent=self,
        )
        dlg.exec()

        # Auto-load trained model without asking
        model_path = dlg.trained_model_path
        if model_path and os.path.exists(model_path):
            self._load_model_from_path(model_path)

        if was_streaming:
            QTimer.singleShot(600, lambda: self._connect_btn.setChecked(True))

    def _load_model_from_path(self, path: str) -> None:
        """Load a model directly by path (no file dialog)."""
        try:
            det = AnomalyDetector()
            det.load(path)
        except Exception as e:
            QMessageBox.critical(self, tr("common.error"), str(e))
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
            QMessageBox.critical(self, tr("common.error"), str(exc))

    # ── Camera properties ─────────────────────────────────────────────────────

    def _apply_cam_prop(self, prop_name: str, value: int) -> None:
        """Forward a single camera property update to the running camera thread."""
        if self._camera_thread and self._camera_thread.isRunning():
            self._camera_thread.set_cam_props({prop_name: value})

    def _reset_cam_settings(self) -> None:
        """Reset all camera sliders to neutral values and send reset to camera."""
        defaults = {"brightness": 0, "contrast": 0, "saturation": 0, "sharpness": 0, "exposure": -6}
        for prop_name, default_val in defaults.items():
            sl = getattr(self, f"_{prop_name}_sl", None)
            if sl:
                sl.blockSignals(True)
                sl.setValue(default_val)
                sl.blockSignals(False)
        if self._camera_thread and self._camera_thread.isRunning():
            self._camera_thread.set_cam_props(defaults)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh_alarm_stats()

    def _refresh_alarm_stats(self) -> None:
        """Read the alarm CSV and update the today/7-day stat labels."""
        if not self._log_path or not os.path.isfile(self._log_path):
            return
        from utils.i18n import tr
        today_count = 0
        week_count = 0
        now = datetime.now(timezone.utc)
        try:
            with open(self._log_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts_str = row.get("timestamp", "")
                    if not ts_str:
                        continue
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        delta = now - ts
                        if delta.days < 1:
                            today_count += 1
                        if delta.days < 7:
                            week_count += 1
                    except ValueError:
                        continue
        except Exception:
            return
        self._stats_today_lbl.setText(tr("camera.stats_today", n=today_count))
        self._stats_week_lbl.setText(tr("camera.stats_week", n=week_count))

    def hideEvent(self, event) -> None:
        if self._camera_thread:
            self._stop_stream()
        if self._scan_thread and self._scan_thread.isRunning():
            self._scan_thread.wait(2000)
        super().hideEvent(event)

    def closeEvent(self, event) -> None:
        self._reconnect_timer.stop()
        if self._camera_thread:
            self._camera_thread.stop()
            self._camera_thread = None
        if self._scan_thread and self._scan_thread.isRunning():
            self._scan_thread.wait(2000)
        super().closeEvent(event)
