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
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QPixmap, QFont, QKeySequence, QShortcut

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

        # ── Frame collection for autoencoder training ─────────────────────────
        if self._ae_collecting and self._ae_collect_remaining > 0:
            self._detector.collect_frame(frame)
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
                score, is_anomaly = self._detector.is_anomaly(frame)
                self._update_score_display(score, is_anomaly)
                if is_anomaly and self._ae_save_anomaly_cb.isChecked():
                    path = self._save_frame(frame)
                    if path:
                        self._add_to_list(path)
        else:
            # Reset display when detection is disabled
            if not self._ae_enabled_cb.isChecked():
                self._set_preview_border_normal()
                self._alarm_banner.setVisible(False)

        # ── Display ───────────────────────────────────────────────────────────
        display = self._apply_timestamp(frame) if self._ts_preview_cb.isChecked() else frame
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
