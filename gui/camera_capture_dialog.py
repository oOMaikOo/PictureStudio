"""
Camera capture dialog: live preview from USB or IP cameras, single/burst capture,
optional timestamp overlay, and unsupervised autoencoder anomaly detection.
"""
import os
import time
import tempfile
from typing import Optional

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox,
    QListWidget, QListWidgetItem, QSplitter, QFileDialog,
    QMessageBox, QTabWidget, QWidget, QProgressBar, QCheckBox,
    QScrollArea,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QEvent, QRect, QPoint
from PySide6.QtGui import QPixmap, QFont, QKeySequence, QShortcut, QCursor, QPainter, QPen, QColor

from core.camera import list_usb_cameras, frame_to_qimage, CameraFrameThread
from utils.logging_utils import get_logger

log = get_logger()

_PREVIEW_W = 640
_PREVIEW_H = 480

# ── background threads ────────────────────────────────────────────────────────


class _UsbScanThread(QThread):
    """Enumerates USB cameras in background (may take a second)."""
    finished = Signal(list)

    def run(self):
        self.finished.emit(list_usb_cameras())


class _AETrainThread(QThread):
    """Trains the autoencoder on collected normal frames."""
    epoch_done = Signal(int, int, float)   # epoch, total, loss
    finished = Signal(float)               # computed threshold
    error = Signal(str)

    def __init__(self, detector, epochs: int, parent=None):
        super().__init__(parent)
        self._detector = detector
        self._epochs = epochs

    def run(self):
        try:
            threshold = self._detector.train(
                epochs=self._epochs,
                progress_cb=lambda e, t, l: self.epoch_done.emit(e, t, l),
            )
            self.finished.emit(threshold)
        except Exception as exc:
            self.error.emit(str(exc))


# ── main dialog ───────────────────────────────────────────────────────────────


class CameraCaptureDialog(QDialog):
    """
    Full-featured camera capture dialog.

    After exec(), check `.captured_paths` for the list of saved PNG files.
    The caller is responsible for adding them to the project.
    """

    def __init__(self, save_dir: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Kamera aufnehmen")
        self.setMinimumSize(1200, 720)

        self._save_dir = save_dir or tempfile.gettempdir()
        self._frame_thread: Optional[CameraFrameThread] = None
        self._current_frame: Optional[np.ndarray] = None
        self._capture_index = 0
        self.captured_paths: list[str] = []

        # anomaly detection state
        self._detector = None          # AnomalyDetector, lazy-initialized
        self._ae_collecting = False
        self._ae_collect_remaining = 0
        self._ae_train_thread: Optional[_AETrainThread] = None
        self._ae_score_counter = 0     # skip counter so we score every 3rd frame
        self._ae_score_streak = 0      # consecutive anomaly frames (for smoothing)

        # ROI state (normalized 0–1 relative to frame): (x1, y1, x2, y2) or None
        self._roi: Optional[tuple[float, float, float, float]] = None
        self._roi_drawing = False
        self._roi_start: Optional[QPoint] = None   # in label coords
        self._roi_end: Optional[QPoint] = None     # in label coords (live drag)
        self._frame_shape: tuple[int, int] = (480, 640)  # (h, w) of last frame

        self._build_ui()
        self._scan_usb_cameras()

    # ================================================================== UI BUILD

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ── Left panel (scrollable) ──────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(360)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(8, 8, 8, 8)
        lv.setSpacing(6)

        # ── Camera source tabs ───────────────────────────────────────────────
        self._src_tabs = QTabWidget()

        usb_w = QWidget()
        usb_l = QVBoxLayout(usb_w)
        usb_l.addWidget(QLabel("Verfügbare USB-Kameras:"))
        self._usb_combo = QComboBox()
        self._usb_combo.addItem("Suche läuft…")
        usb_l.addWidget(self._usb_combo)
        refresh_btn = QPushButton("Kameras neu suchen")
        refresh_btn.clicked.connect(self._scan_usb_cameras)
        usb_l.addWidget(refresh_btn)
        usb_l.addStretch()
        self._src_tabs.addTab(usb_w, "USB Kamera")

        ip_w = QWidget()
        ip_l = QVBoxLayout(ip_w)
        ip_l.addWidget(QLabel("Kamera-URL (RTSP / HTTP):"))
        self._ip_edit = QLineEdit()
        self._ip_edit.setPlaceholderText("rtsp://user:pass@192.168.1.100:554/stream")
        ip_l.addWidget(self._ip_edit)
        examples = QLabel(
            "<small>rtsp://192.168.1.100:554/live<br>"
            "http://192.168.1.100:8080/video</small>"
        )
        examples.setWordWrap(True)
        ip_l.addWidget(examples)
        ip_l.addStretch()
        self._src_tabs.addTab(ip_w, "IP Kamera")
        lv.addWidget(self._src_tabs)

        # ── Connect / status ─────────────────────────────────────────────────
        conn_row = QHBoxLayout()
        self._connect_btn = QPushButton("Verbinden")
        self._connect_btn.setStyleSheet("background:#2ECC71;color:white;padding:6px;font-weight:bold;")
        self._connect_btn.clicked.connect(self._toggle_connection)
        conn_row.addWidget(self._connect_btn)
        self._status_lbl = QLabel("Nicht verbunden")
        self._status_lbl.setStyleSheet("color:#E74C3C;")
        conn_row.addWidget(self._status_lbl)
        lv.addLayout(conn_row)

        # ── Capture controls ─────────────────────────────────────────────────
        cap_group = QGroupBox("Aufnahme")
        cg = QVBoxLayout(cap_group)

        self._single_btn = QPushButton("Bild aufnehmen  [Leertaste]")
        self._single_btn.setStyleSheet("background:#3498DB;color:white;padding:8px;font-weight:bold;")
        self._single_btn.clicked.connect(self._capture_single)
        cg.addWidget(self._single_btn)

        burst_row = QHBoxLayout()
        burst_row.addWidget(QLabel("Burst:"))
        self._burst_count = QSpinBox()
        self._burst_count.setRange(2, 500)
        self._burst_count.setValue(10)
        burst_row.addWidget(self._burst_count)
        burst_row.addWidget(QLabel("Bilder,"))
        self._burst_interval = QDoubleSpinBox()
        self._burst_interval.setRange(0.1, 10.0)
        self._burst_interval.setValue(0.5)
        self._burst_interval.setSuffix(" s")
        burst_row.addWidget(self._burst_interval)
        cg.addLayout(burst_row)

        self._burst_btn = QPushButton("Burst starten")
        self._burst_btn.clicked.connect(self._start_burst)
        cg.addWidget(self._burst_btn)

        self._burst_progress = QProgressBar()
        self._burst_progress.setVisible(False)
        cg.addWidget(self._burst_progress)
        lv.addWidget(cap_group)

        # ── Timestamp ────────────────────────────────────────────────────────
        ts_group = QGroupBox("Zeitstempel")
        tg = QVBoxLayout(ts_group)
        self._ts_preview_cb = QCheckBox("Im Vorschaubild anzeigen")
        self._ts_preview_cb.setToolTip("Blendet Systemzeit und -datum im Live-Bild ein")
        tg.addWidget(self._ts_preview_cb)
        self._ts_save_cb = QCheckBox("In gespeichertes Bild einbrennen")
        self._ts_save_cb.setToolTip("Brennt den Zeitstempel dauerhaft in die PNG-Datei ein")
        tg.addWidget(self._ts_save_cb)
        lv.addWidget(ts_group)

        # ── Anomaly detection ─────────────────────────────────────────────────
        lv.addWidget(self._build_anomaly_group())

        # ── Save location ────────────────────────────────────────────────────
        save_group = QGroupBox("Speicherort")
        sg = QVBoxLayout(save_group)
        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit(self._save_dir)
        self._dir_edit.setReadOnly(True)
        dir_row.addWidget(self._dir_edit)
        dir_btn = QPushButton("…")
        dir_btn.setMaximumWidth(30)
        dir_btn.clicked.connect(self._choose_save_dir)
        dir_row.addWidget(dir_btn)
        sg.addLayout(dir_row)
        lv.addWidget(save_group)

        # ── Captured list ────────────────────────────────────────────────────
        list_group = QGroupBox("Aufgenommene Bilder")
        lg = QVBoxLayout(list_group)
        self._captured_list = QListWidget()
        self._captured_list.setMaximumHeight(130)
        lg.addWidget(self._captured_list)
        del_btn = QPushButton("Markierte entfernen")
        del_btn.clicked.connect(self._delete_selected)
        lg.addWidget(del_btn)
        clear_btn = QPushButton("Alle löschen")
        clear_btn.clicked.connect(self._clear_all)
        lg.addWidget(clear_btn)
        lv.addWidget(list_group)

        lv.addStretch()
        scroll.setWidget(left)
        splitter.addWidget(scroll)

        # ── Right panel: preview ─────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)

        # Anomaly alarm banner (hidden until triggered)
        self._alarm_banner = QLabel("  ⚠  ANOMALIE ERKANNT")
        self._alarm_banner.setAlignment(Qt.AlignCenter)
        self._alarm_banner.setStyleSheet(
            "background:#C0392B;color:white;font-weight:bold;font-size:15px;padding:6px;"
        )
        self._alarm_banner.setVisible(False)
        rv.addWidget(self._alarm_banner)

        self._preview_lbl = QLabel("Kein Signal")
        self._preview_lbl.setAlignment(Qt.AlignCenter)
        self._preview_lbl.setMinimumSize(_PREVIEW_W, _PREVIEW_H)
        self._preview_lbl.setStyleSheet(
            "background:#111;color:#555;font-size:18px;border:3px solid #333;"
        )
        self._preview_lbl.setMouseTracking(True)
        self._preview_lbl.installEventFilter(self)
        rv.addWidget(self._preview_lbl, stretch=1)

        self._frame_info = QLabel("")
        self._frame_info.setAlignment(Qt.AlignRight)
        self._frame_info.setFont(QFont("Courier New", 8))
        rv.addWidget(self._frame_info)

        splitter.addWidget(right)
        splitter.setSizes([360, 840])

        # ── Bottom buttons ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        self._accept_btn = QPushButton("In Projekt übernehmen (0)")
        self._accept_btn.setStyleSheet("background:#2ECC71;color:white;padding:8px;font-weight:bold;")
        self._accept_btn.setEnabled(False)
        self._accept_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._accept_btn)
        root.addLayout(btn_row)

        sc = QShortcut(QKeySequence(Qt.Key_Space), self)
        sc.activated.connect(self._capture_single)

    def _build_anomaly_group(self) -> QGroupBox:
        """Build and return the anomaly-detection GroupBox."""
        grp = QGroupBox("Anomalie-Erkennung (Autoencoder)")
        g = QVBoxLayout(grp)
        g.setSpacing(5)

        self._ae_enabled_cb = QCheckBox("Aktiv (Live-Scoring nach Training)")
        self._ae_enabled_cb.setToolTip(
            "Bewertet jeden Frame mit dem trainierten Autoencoder.\n"
            "Hoher Rekonstruktionsfehler = unbekanntes / anomales Ereignis."
        )
        g.addWidget(self._ae_enabled_cb)

        # ── ROI ──────────────────────────────────────────────────────────────
        roi_grp = QGroupBox("0 · Analysebereich (ROI) – optional")
        rl = QVBoxLayout(roi_grp)
        roi_hint = QLabel("Im Vorschaubild einen Bereich aufziehen.\n"
                          "Nur dieser Bereich fließt in Training & Scoring ein.")
        roi_hint.setStyleSheet("color:#7F8C8D; font-size:10px;")
        roi_hint.setWordWrap(True)
        rl.addWidget(roi_hint)
        roi_btn_row = QHBoxLayout()
        self._roi_draw_btn = QPushButton("ROI aufziehen")
        self._roi_draw_btn.setCheckable(True)
        self._roi_draw_btn.setToolTip("Klicken, dann im Vorschaubild Rechteck ziehen")
        self._roi_draw_btn.toggled.connect(self._on_roi_draw_toggled)
        roi_btn_row.addWidget(self._roi_draw_btn)
        self._roi_clear_btn = QPushButton("ROI löschen")
        self._roi_clear_btn.setEnabled(False)
        self._roi_clear_btn.clicked.connect(self._clear_roi)
        roi_btn_row.addWidget(self._roi_clear_btn)
        rl.addLayout(roi_btn_row)
        self._roi_lbl = QLabel("Kein ROI – ganzes Bild wird analysiert")
        self._roi_lbl.setStyleSheet("color:#7F8C8D; font-size:10px;")
        rl.addWidget(self._roi_lbl)
        g.addWidget(roi_grp)

        # ── Collection ───────────────────────────────────────────────────────
        coll_grp = QGroupBox("1 · Normalframes aufnehmen")
        cl = QVBoxLayout(coll_grp)
        cnt_row = QHBoxLayout()
        cnt_row.addWidget(QLabel("Anzahl:"))
        self._ae_collect_n = QSpinBox()
        self._ae_collect_n.setRange(20, 2000)
        self._ae_collect_n.setValue(150)
        self._ae_collect_n.setSuffix(" Frames")
        cnt_row.addWidget(self._ae_collect_n)
        cl.addLayout(cnt_row)

        collect_row = QHBoxLayout()
        self._ae_collect_btn = QPushButton("Aufnehmen starten")
        self._ae_collect_btn.clicked.connect(self._start_collecting)
        collect_row.addWidget(self._ae_collect_btn)
        ae_clear_btn = QPushButton("Löschen")
        ae_clear_btn.setMaximumWidth(60)
        ae_clear_btn.clicked.connect(self._clear_ae_frames)
        collect_row.addWidget(ae_clear_btn)
        cl.addLayout(collect_row)

        self._ae_collect_bar = QProgressBar()
        self._ae_collect_bar.setVisible(False)
        cl.addWidget(self._ae_collect_bar)

        self._ae_collect_lbl = QLabel("0 Frames gesammelt")
        self._ae_collect_lbl.setStyleSheet("color:#7F8C8D; font-size:10px;")
        cl.addWidget(self._ae_collect_lbl)
        g.addWidget(coll_grp)

        # ── Training ─────────────────────────────────────────────────────────
        train_grp = QGroupBox("2 · Autoencoder trainieren")
        tl = QVBoxLayout(train_grp)
        ep_row = QHBoxLayout()
        ep_row.addWidget(QLabel("Epochen:"))
        self._ae_epochs = QSpinBox()
        self._ae_epochs.setRange(5, 200)
        self._ae_epochs.setValue(40)
        ep_row.addWidget(self._ae_epochs)
        tl.addLayout(ep_row)

        self._ae_train_btn = QPushButton("Training starten")
        self._ae_train_btn.setEnabled(False)
        self._ae_train_btn.clicked.connect(self._train_autoencoder)
        tl.addWidget(self._ae_train_btn)

        self._ae_train_bar = QProgressBar()
        self._ae_train_bar.setVisible(False)
        tl.addWidget(self._ae_train_bar)

        self._ae_train_lbl = QLabel("")
        self._ae_train_lbl.setStyleSheet("color:#7F8C8D; font-size:10px;")
        tl.addWidget(self._ae_train_lbl)
        g.addWidget(train_grp)

        # ── Live scoring ─────────────────────────────────────────────────────
        score_grp = QGroupBox("3 · Live-Erkennung")
        sl = QVBoxLayout(score_grp)

        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("Glättung:"))
        self._ae_smooth_n = QSpinBox()
        self._ae_smooth_n.setRange(1, 20)
        self._ae_smooth_n.setValue(5)
        self._ae_smooth_n.setSuffix(" Frames")
        self._ae_smooth_n.setToolTip(
            "Alarm erst nach N aufeinanderfolgenden Frames über dem Schwellwert.\n"
            "Verhindert Fehlalarme durch kurze Erschütterungen oder Kamerawackler.\n"
            "1 = sofortiger Alarm, 5 = robuster gegen Einzelstörer."
        )
        smooth_row.addWidget(self._ae_smooth_n)
        smooth_row.addStretch()
        sl.addLayout(smooth_row)

        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Schwellwert:"))
        self._ae_threshold_spin = QDoubleSpinBox()
        self._ae_threshold_spin.setRange(0.00001, 1.0)
        self._ae_threshold_spin.setDecimals(5)
        self._ae_threshold_spin.setSingleStep(0.001)
        self._ae_threshold_spin.setValue(0.02)
        self._ae_threshold_spin.setToolTip(
            "Nach dem Training automatisch gesetzt (Mittelwert + 2,5 × Std-Abw).\n"
            "Erhöhen = weniger sensitiv, senken = sensitiver."
        )
        self._ae_threshold_spin.valueChanged.connect(self._on_threshold_changed)
        thr_row.addWidget(self._ae_threshold_spin)
        sl.addLayout(thr_row)

        self._ae_score_lbl = QLabel("Score: –")
        self._ae_score_lbl.setAlignment(Qt.AlignCenter)
        self._ae_score_lbl.setStyleSheet(
            "font-weight:bold;font-size:12px;padding:5px;"
            "border-radius:5px;background:#1A252F;color:#7F8C8D;"
        )
        sl.addWidget(self._ae_score_lbl)

        self._ae_save_anomaly_cb = QCheckBox("Anomalie-Frames automatisch speichern")
        self._ae_save_anomaly_cb.setToolTip("Speichert jeden Frame bei dem der Score den Schwellwert überschreitet.")
        sl.addWidget(self._ae_save_anomaly_cb)
        g.addWidget(score_grp)

        # ── Model save / load ─────────────────────────────────────────────────
        io_grp = QGroupBox("Modell speichern / laden")
        il = QHBoxLayout(io_grp)
        save_ae_btn = QPushButton("Speichern…")
        save_ae_btn.clicked.connect(self._save_ae_model)
        load_ae_btn = QPushButton("Laden…")
        load_ae_btn.clicked.connect(self._load_ae_model)
        il.addWidget(save_ae_btn)
        il.addWidget(load_ae_btn)
        g.addWidget(io_grp)

        return grp

    # ================================================================== CAMERA SCAN

    def _scan_usb_cameras(self) -> None:
        self._usb_combo.clear()
        self._usb_combo.addItem("Suche läuft…")
        self._usb_combo.setEnabled(False)
        self._scan_thread = _UsbScanThread(self)
        self._scan_thread.finished.connect(self._on_usb_scan_done)
        self._scan_thread.start()

    @Slot(list)
    def _on_usb_scan_done(self, cameras: list) -> None:
        self._usb_combo.clear()
        self._usb_combo.setEnabled(True)
        if cameras:
            for idx, label in cameras:
                self._usb_combo.addItem(label, userData=idx)
        else:
            self._usb_combo.addItem("Keine USB-Kamera gefunden")

    # ================================================================== CONNECTION

    def _toggle_connection(self) -> None:
        if self._frame_thread and self._frame_thread.isRunning():
            self._disconnect()
        else:
            self._connect()

    def _connect(self) -> None:
        source = self._resolve_source()
        if source is None:
            return
        self._frame_thread = CameraFrameThread(source, fps=15.0, parent=self)
        self._frame_thread.frame_ready.connect(self._on_frame)
        self._frame_thread.error.connect(self._on_camera_error)
        self._frame_thread.start()
        self._connect_btn.setText("Trennen")
        self._connect_btn.setStyleSheet("background:#E74C3C;color:white;padding:6px;font-weight:bold;")
        self._status_lbl.setText("Verbinde…")
        self._status_lbl.setStyleSheet("color:#F39C12;")

    def _disconnect(self) -> None:
        if self._frame_thread:
            self._frame_thread.stop()
            self._frame_thread = None
        self._connect_btn.setText("Verbinden")
        self._connect_btn.setStyleSheet("background:#2ECC71;color:white;padding:6px;font-weight:bold;")
        self._status_lbl.setText("Nicht verbunden")
        self._status_lbl.setStyleSheet("color:#E74C3C;")
        self._preview_lbl.setText("Kein Signal")
        self._current_frame = None

    def _resolve_source(self):
        if self._src_tabs.currentIndex() == 0:
            data = self._usb_combo.currentData()
            if data is None:
                QMessageBox.warning(self, "Keine Kamera", "Keine USB-Kamera ausgewählt.")
                return None
            return int(data)
        url = self._ip_edit.text().strip()
        if not url:
            QMessageBox.warning(self, "Keine URL", "Bitte eine Kamera-URL eingeben.")
            return None
        return url

    # ================================================================== FRAME HANDLING

    @Slot(object)
    def _on_frame(self, frame: np.ndarray) -> None:
        self._current_frame = frame
        h, w = frame.shape[:2]
        self._frame_shape = (h, w)

        # Region used for anomaly detection (full frame or ROI crop)
        analysis_frame = self._crop_roi(frame)

        # ── Frame collection for autoencoder training ─────────────────────────
        if self._ae_collecting and self._ae_collect_remaining > 0:
            self._detector.collect_frame(analysis_frame)
            self._ae_collect_remaining -= 1
            n = self._detector.n_collected()
            self._ae_collect_bar.setValue(n)
            self._ae_collect_lbl.setText(f"{n} Frames gesammelt")
            if self._ae_collect_remaining <= 0:
                self._ae_collecting = False
                self._ae_collect_bar.setVisible(False)
                self._ae_collect_btn.setEnabled(True)
                self._ae_train_btn.setEnabled(True)

        # ── Anomaly scoring (every 3rd frame to reduce CPU load) ──────────────
        if self._ae_enabled_cb.isChecked() and self._detector and self._detector.trained:
            self._ae_score_counter += 1
            if self._ae_score_counter >= 3:
                self._ae_score_counter = 0
                score, is_anomaly = self._detector.is_anomaly(analysis_frame)
                # Score smoothing: alarm only after N consecutive anomaly frames
                if is_anomaly:
                    self._ae_score_streak += 1
                else:
                    self._ae_score_streak = 0
                smooth_n = self._ae_smooth_n.value()
                smoothed_alarm = self._ae_score_streak >= smooth_n
                self._update_score_display(score, smoothed_alarm)
                if smoothed_alarm and self._ae_save_anomaly_cb.isChecked():
                    path = self._save_frame(frame)
                    if path:
                        self._add_to_list(path)
        else:
            if not self._ae_enabled_cb.isChecked():
                self._ae_score_streak = 0
                self._set_preview_border_normal()
                self._alarm_banner.setVisible(False)

        # ── Display (timestamp + ROI overlay) ────────────────────────────────
        display = self._apply_timestamp(frame) if self._ts_preview_cb.isChecked() else frame.copy()
        display = self._draw_roi_overlay(display)
        pix = QPixmap.fromImage(frame_to_qimage(display))
        pix = pix.scaled(self._preview_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._preview_lbl.setPixmap(pix)
        self._status_lbl.setText("Verbunden")
        self._status_lbl.setStyleSheet("color:#2ECC71;")
        self._frame_info.setText(f"{w}×{h} px")

    @Slot(str)
    def _on_camera_error(self, msg: str) -> None:
        self._disconnect()
        QMessageBox.warning(self, "Kamerafehler", msg)

    # ================================================================== SCORE DISPLAY

    def _update_score_display(self, score: float, is_anomaly: bool) -> None:
        thr = self._detector.threshold
        pct = min(100, int(score / thr * 100)) if thr > 0 else 0
        label_text = f"Score: {score:.5f}  ({pct}% des Schwellwerts)"

        if is_anomaly:
            self._ae_score_lbl.setText(label_text)
            self._ae_score_lbl.setStyleSheet(
                "font-weight:bold;font-size:12px;padding:5px;"
                "border-radius:5px;background:#922B21;color:white;"
            )
            self._preview_lbl.setStyleSheet(
                "background:#111;color:#555;font-size:18px;border:4px solid #E74C3C;"
            )
            self._alarm_banner.setVisible(True)
        else:
            self._ae_score_lbl.setText(label_text)
            self._ae_score_lbl.setStyleSheet(
                "font-weight:bold;font-size:12px;padding:5px;"
                "border-radius:5px;background:#1A4D2E;color:#58D68D;"
            )
            self._set_preview_border_normal()
            self._alarm_banner.setVisible(False)

    def _set_preview_border_normal(self) -> None:
        self._preview_lbl.setStyleSheet(
            "background:#111;color:#555;font-size:18px;border:3px solid #333;"
        )

    # ================================================================== TIMESTAMP

    def _apply_timestamp(self, frame: np.ndarray) -> np.ndarray:
        from datetime import datetime as _dt
        out = frame.copy()
        text = _dt.now().strftime("%Y-%m-%d  %H:%M:%S")
        h, w = out.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, w / 1280)
        thickness = max(1, int(scale * 1.5))
        _, baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = 10, h - 10 - baseline
        cv2.putText(out, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(out, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return out

    # ================================================================== CAPTURE

    def _capture_single(self) -> None:
        if self._current_frame is None:
            return
        path = self._save_frame(self._current_frame)
        if path:
            self._add_to_list(path)

    def _save_frame(self, frame: np.ndarray) -> Optional[str]:
        os.makedirs(self._save_dir, exist_ok=True)
        self._capture_index += 1
        filename = f"capture_{int(time.time())}_{self._capture_index:04d}.png"
        path = os.path.join(self._save_dir, filename)
        save_frame = self._apply_timestamp(frame) if self._ts_save_cb.isChecked() else frame
        if not cv2.imwrite(path, save_frame):
            QMessageBox.critical(self, "Speicherfehler", f"Konnte nicht speichern:\n{path}")
            return None
        return path

    def _add_to_list(self, path: str) -> None:
        self.captured_paths.append(path)
        item = QListWidgetItem(os.path.basename(path))
        item.setData(Qt.UserRole, path)
        self._captured_list.addItem(item)
        self._captured_list.scrollToBottom()
        self._accept_btn.setText(f"In Projekt übernehmen ({len(self.captured_paths)})")
        self._accept_btn.setEnabled(True)

    # ================================================================== BURST

    def _start_burst(self) -> None:
        if self._current_frame is None:
            QMessageBox.warning(self, "Kein Signal", "Bitte zuerst Kamera verbinden.")
            return
        count = self._burst_count.value()
        interval = self._burst_interval.value()
        self._burst_btn.setEnabled(False)
        self._single_btn.setEnabled(False)
        self._burst_progress.setRange(0, count)
        self._burst_progress.setValue(0)
        self._burst_progress.setVisible(True)
        self._burst_remaining = count
        self._burst_timer = QTimer(self)
        self._burst_timer.timeout.connect(self._burst_tick)
        self._burst_timer.start(int(interval * 1000))
        self._burst_tick()

    def _burst_tick(self) -> None:
        if self._burst_remaining <= 0 or self._current_frame is None:
            self._burst_timer.stop()
            self._burst_btn.setEnabled(True)
            self._single_btn.setEnabled(True)
            self._burst_progress.setVisible(False)
            return
        self._capture_single()
        self._burst_remaining -= 1
        self._burst_progress.setValue(self._burst_progress.maximum() - self._burst_remaining)
        if self._burst_remaining <= 0:
            self._burst_timer.stop()
            self._burst_btn.setEnabled(True)
            self._single_btn.setEnabled(True)
            self._burst_progress.setVisible(False)

    # ================================================================== LIST

    def _delete_selected(self) -> None:
        for item in self._captured_list.selectedItems():
            path = item.data(Qt.UserRole)
            if path in self.captured_paths:
                self.captured_paths.remove(path)
            self._captured_list.takeItem(self._captured_list.row(item))
        self._accept_btn.setText(f"In Projekt übernehmen ({len(self.captured_paths)})")
        self._accept_btn.setEnabled(len(self.captured_paths) > 0)

    def _clear_all(self) -> None:
        self.captured_paths.clear()
        self._captured_list.clear()
        self._accept_btn.setText("In Projekt übernehmen (0)")
        self._accept_btn.setEnabled(False)

    # ================================================================== SAVE DIR

    def _choose_save_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Speicherordner wählen", self._save_dir)
        if folder:
            self._save_dir = folder
            self._dir_edit.setText(folder)

    # ================================================================== ANOMALY DETECTION

    def _ensure_detector(self) -> None:
        if self._detector is None:
            from core.anomaly_detector import AnomalyDetector
            self._detector = AnomalyDetector()

    def _start_collecting(self) -> None:
        if self._current_frame is None:
            QMessageBox.warning(self, "Kein Signal", "Bitte zuerst Kamera verbinden.")
            return
        self._ensure_detector()
        n = self._ae_collect_n.value()
        self._ae_collecting = True
        self._ae_collect_remaining = n
        self._ae_collect_bar.setRange(0, n)
        self._ae_collect_bar.setValue(self._detector.n_collected())
        self._ae_collect_bar.setVisible(True)
        self._ae_collect_btn.setEnabled(False)

    def _clear_ae_frames(self) -> None:
        if self._detector:
            self._detector.clear_frames()
        self._ae_collecting = False
        self._ae_collect_remaining = 0
        self._ae_collect_bar.setVisible(False)
        self._ae_collect_lbl.setText("0 Frames gesammelt")
        self._ae_collect_btn.setEnabled(True)
        self._ae_train_btn.setEnabled(False)

    def _train_autoencoder(self) -> None:
        self._ensure_detector()
        if self._detector.n_collected() < 10:
            QMessageBox.warning(self, "Zu wenig Frames",
                                "Bitte mindestens 10 Normalframes sammeln.")
            return
        epochs = self._ae_epochs.value()
        self._ae_train_btn.setEnabled(False)
        self._ae_collect_btn.setEnabled(False)
        self._ae_train_bar.setRange(0, epochs)
        self._ae_train_bar.setValue(0)
        self._ae_train_bar.setVisible(True)
        self._ae_train_lbl.setText("Training läuft…")

        self._ae_train_thread = _AETrainThread(self._detector, epochs, parent=self)
        self._ae_train_thread.epoch_done.connect(self._on_ae_epoch)
        self._ae_train_thread.finished.connect(self._on_ae_trained)
        self._ae_train_thread.error.connect(self._on_ae_error)
        self._ae_train_thread.start()

    @Slot(int, int, float)
    def _on_ae_epoch(self, epoch: int, total: int, loss: float) -> None:
        self._ae_train_bar.setValue(epoch)
        self._ae_train_lbl.setText(f"Epoche {epoch}/{total}  |  Loss: {loss:.5f}")

    @Slot(float)
    def _on_ae_trained(self, threshold: float) -> None:
        self._ae_train_bar.setVisible(False)
        self._ae_train_btn.setEnabled(True)
        self._ae_collect_btn.setEnabled(True)
        self._ae_threshold_spin.blockSignals(True)
        self._ae_threshold_spin.setValue(threshold)
        self._ae_threshold_spin.blockSignals(False)
        self._ae_train_lbl.setText(
            f"Fertig! Schwellwert: {threshold:.5f}"
        )
        self._ae_score_lbl.setText("Score: – (bereit)")
        self._ae_score_lbl.setStyleSheet(
            "font-weight:bold;font-size:12px;padding:5px;"
            "border-radius:5px;background:#1A252F;color:#58D68D;"
        )
        log.info(f"Autoencoder trained, threshold={threshold:.5f}")

    @Slot(str)
    def _on_ae_error(self, msg: str) -> None:
        self._ae_train_bar.setVisible(False)
        self._ae_train_btn.setEnabled(True)
        self._ae_collect_btn.setEnabled(True)
        self._ae_train_lbl.setText(f"Fehler: {msg}")
        QMessageBox.critical(self, "Trainingsfehler", msg)

    def _on_threshold_changed(self, value: float) -> None:
        if self._detector:
            self._detector.threshold = value

    def _save_ae_model(self) -> None:
        self._ensure_detector()
        if not self._detector.trained:
            QMessageBox.warning(self, "Kein Modell", "Erst Autoencoder trainieren.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Autoencoder-Modell speichern", self._save_dir, "PyTorch (*.pth)"
        )
        if path:
            self._detector.save(path)
            QMessageBox.information(self, "Gespeichert", f"Modell gespeichert:\n{path}")

    def _load_ae_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Autoencoder-Modell laden", self._save_dir, "PyTorch (*.pth)"
        )
        if not path:
            return
        self._ensure_detector()
        try:
            self._detector.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Ladefehler", str(exc))
            return
        self._ae_threshold_spin.blockSignals(True)
        self._ae_threshold_spin.setValue(self._detector.threshold)
        self._ae_threshold_spin.blockSignals(False)
        self._ae_train_lbl.setText(f"Modell geladen | Schwellwert: {self._detector.threshold:.5f}")
        self._ae_score_lbl.setText("Score: – (Modell geladen, bereit)")

    # ================================================================== ROI

    def _on_roi_draw_toggled(self, checked: bool) -> None:
        if checked:
            self._roi_drawing = True
            self._preview_lbl.setCursor(Qt.CrossCursor)
            self._roi_draw_btn.setText("ROI aufziehen … (Klick + Ziehen)")
        else:
            self._roi_drawing = False
            self._roi_start = None
            self._roi_end = None
            self._preview_lbl.setCursor(Qt.ArrowCursor)
            self._roi_draw_btn.setText("ROI aufziehen")

    def _clear_roi(self) -> None:
        self._roi = None
        self._roi_start = None
        self._roi_end = None
        self._roi_clear_btn.setEnabled(False)
        self._roi_lbl.setText("Kein ROI – ganzes Bild wird analysiert")
        self._roi_lbl.setStyleSheet("color:#7F8C8D; font-size:10px;")

    def _crop_roi(self, frame: np.ndarray) -> np.ndarray:
        """Return the ROI crop of frame, or the full frame if no ROI is set."""
        if self._roi is None:
            return frame
        h, w = frame.shape[:2]
        x1 = int(self._roi[0] * w)
        y1 = int(self._roi[1] * h)
        x2 = int(self._roi[2] * w)
        y2 = int(self._roi[3] * h)
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return frame
        return frame[y1:y2, x1:x2]

    def _draw_roi_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw the ROI rectangle (and live drag preview) onto a display copy."""
        out = frame if frame.flags['WRITEABLE'] else frame.copy()
        h, w = out.shape[:2]

        # Committed ROI
        if self._roi is not None:
            x1 = int(self._roi[0] * w)
            y1 = int(self._roi[1] * h)
            x2 = int(self._roi[2] * w)
            y2 = int(self._roi[3] * h)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(out, "ROI", (x1 + 4, y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        # Live drag preview
        if self._roi_drawing and self._roi_start and self._roi_end:
            p1 = self._label_to_frame(self._roi_start, w, h)
            p2 = self._label_to_frame(self._roi_end, w, h)
            if p1 and p2:
                cv2.rectangle(out, p1, p2, (255, 200, 0), 2)

        return out

    def _label_to_frame(self, lpos: QPoint, fw: int, fh: int) -> Optional[tuple[int, int]]:
        """Convert a label-pixel position to frame-pixel coordinates."""
        lw = self._preview_lbl.width()
        lh = self._preview_lbl.height()
        if lw <= 0 or lh <= 0 or fw <= 0 or fh <= 0:
            return None
        scale = min(lw / fw, lh / fh)
        img_w = fw * scale
        img_h = fh * scale
        ox = (lw - img_w) / 2
        oy = (lh - img_h) / 2
        fx = (lpos.x() - ox) / scale
        fy = (lpos.y() - oy) / scale
        return (int(max(0, min(fw - 1, fx))), int(max(0, min(fh - 1, fy))))

    def eventFilter(self, obj, event) -> bool:
        if obj is self._preview_lbl and self._roi_drawing:
            t = event.type()
            if t == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._roi_start = event.pos()
                self._roi_end = event.pos()
                return True
            if t == QEvent.MouseMove and self._roi_start:
                self._roi_end = event.pos()
                return True
            if t == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and self._roi_start:
                self._roi_end = event.pos()
                self._finalise_roi()
                return True
        return super().eventFilter(obj, event)

    def _finalise_roi(self) -> None:
        if not (self._roi_start and self._roi_end):
            return
        h, w = self._frame_shape
        p1 = self._label_to_frame(self._roi_start, w, h)
        p2 = self._label_to_frame(self._roi_end, w, h)
        if not (p1 and p2):
            return
        x1n = min(p1[0], p2[0]) / w
        y1n = min(p1[1], p2[1]) / h
        x2n = max(p1[0], p2[0]) / w
        y2n = max(p1[1], p2[1]) / h
        if (x2n - x1n) < 0.02 or (y2n - y1n) < 0.02:
            return   # too small, ignore
        self._roi = (x1n, y1n, x2n, y2n)
        self._roi_draw_btn.setChecked(False)   # exits draw mode via toggled signal
        self._roi_clear_btn.setEnabled(True)
        pw = int((x2n - x1n) * 100)
        ph = int((y2n - y1n) * 100)
        self._roi_lbl.setText(f"ROI: {pw}% × {ph}% des Bildes")
        self._roi_lbl.setStyleSheet("color:#00E5FF; font-size:10px; font-weight:bold;")

    # ================================================================== CLEANUP

    def closeEvent(self, event) -> None:
        self._ae_collecting = False
        if self._ae_train_thread and self._ae_train_thread.isRunning():
            self._ae_train_thread.wait(3000)
        self._disconnect()
        super().closeEvent(event)

    def reject(self) -> None:
        self._ae_collecting = False
        self._disconnect()
        super().reject()
