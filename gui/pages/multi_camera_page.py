"""
Multi-Kamera-Monitoring: bis zu 4 Kamera-Kanäle gleichzeitig überwachen.
Jeder Kanal hat eine eigene Kamera-Quelle und ein eigenes Anomalie-Modell.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

from core.anomaly_detector import AnomalyDetector
from core.camera import list_usb_cameras, CameraFrameThread
from core.alarm_notifier import AlarmNotifier
from core.industrial_notifier import IndustrialNotifier

try:
    from core.onnx_anomaly_scorer import OnnxAnomalyScorer, HAS_ORT
except Exception:
    OnnxAnomalyScorer = None  # type: ignore[assignment,misc]
    HAS_ORT = False

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QProgressBar,
    QDialog, QComboBox, QLineEdit, QFileDialog,
    QDialogButtonBox, QPlainTextEdit, QSizePolicy, QSpinBox,
)
from PySide6.QtCore import Qt, Signal, QMetaObject, Q_ARG, Slot
from PySide6.QtGui import QImage, QPixmap, QFont

_MAX_PER_PAGE = 4   # channels visible at once (2 × 2 grid)

# ---------------------------------------------------------------------------
# Channel state (plain Python object, no Qt)
# ---------------------------------------------------------------------------

class _ChannelState:
    """All runtime state for a single monitoring channel."""

    def __init__(self) -> None:
        self.camera_idx: int = 0
        self.model_path: str = ""
        self.detector: Optional[AnomalyDetector] = None
        self.camera_thread: Optional[CameraFrameThread] = None
        self.roi: Optional[list] = None          # [x1,y1,x2,y2] normalised
        self.threshold: float = 0.001
        self.running: bool = False
        self.score: float = 0.0
        self.is_anomaly: bool = False
        self.event_count: int = 0
        self.last_alarm_t: float = 0.0
        self.frame_counter: int = 0              # for every-3rd-frame scoring


# ---------------------------------------------------------------------------
# Per-channel widget
# ---------------------------------------------------------------------------

class _ChannelWidget(QGroupBox):
    """Visual widget for one monitoring channel in the 2×2 grid."""

    configure_requested = Signal(int)
    start_requested = Signal(int)
    stop_requested = Signal(int)

    def __init__(self, channel_idx: int, parent=None) -> None:
        super().__init__(f"Kanal {channel_idx + 1}", parent)
        self._channel_idx = channel_idx
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setStyleSheet(
            "QGroupBox {"
            "  background: #1C2A3A;"
            "  border: 1px solid #2C3E50;"
            "  border-radius: 8px;"
            "  margin-top: 10px;"
            "  color: #BDC3C7;"
            "  font-weight: bold;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  left: 10px;"
            "  padding: 0 4px;"
            "  color: #5DADE2;"
            "}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 14, 6, 6)
        root.setSpacing(4)

        # Video label
        self._video_lbl = QLabel()
        self._video_lbl.setMinimumSize(280, 210)
        self._video_lbl.setAlignment(Qt.AlignCenter)
        self._video_lbl.setStyleSheet(
            "background: #0D1117; border-radius: 4px; color: #4A5568; font-size: 12px;"
        )
        self._video_lbl.setText("Kein Signal")
        self._video_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self._video_lbl)

        # Score bar
        self._score_bar = QProgressBar()
        self._score_bar.setRange(0, 100)
        self._score_bar.setValue(0)
        self._score_bar.setTextVisible(False)
        self._score_bar.setFixedHeight(8)
        self._score_bar.setStyleSheet(
            "QProgressBar { background: #1A252F; border-radius: 4px; }"
            "QProgressBar::chunk { background: #27AE60; border-radius: 4px; }"
        )
        root.addWidget(self._score_bar)

        # Status label
        self._status_lbl = QLabel("Nicht konfiguriert")
        self._status_lbl.setAlignment(Qt.AlignCenter)
        self._status_lbl.setStyleSheet("color: #7F8C8D; font-size: 11px; padding: 2px;")
        root.addWidget(self._status_lbl)

        # Info label (camera + model name)
        self._info_lbl = QLabel("")
        self._info_lbl.setAlignment(Qt.AlignCenter)
        self._info_lbl.setStyleSheet(
            "color: #566573; font-size: 10px; padding: 1px;"
        )
        self._info_lbl.setWordWrap(True)
        root.addWidget(self._info_lbl)

        # Button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self._cfg_btn = QPushButton("Konfigurieren")
        self._cfg_btn.setStyleSheet(
            "QPushButton { background: #2C3E50; color: #BDC3C7; border-radius: 4px;"
            "  padding: 4px 8px; font-size: 11px; }"
            "QPushButton:hover { background: #34495E; }"
        )
        self._cfg_btn.clicked.connect(lambda: self.configure_requested.emit(self._channel_idx))
        btn_row.addWidget(self._cfg_btn)

        self._start_btn = QPushButton("Starten")
        self._start_btn.setEnabled(False)
        self._start_btn.setStyleSheet(
            "QPushButton { background: #1E8449; color: white; border-radius: 4px;"
            "  padding: 4px 8px; font-size: 11px; }"
            "QPushButton:hover:enabled { background: #27AE60; }"
            "QPushButton:disabled { background: #1A252F; color: #4A5568; }"
        )
        self._start_btn.clicked.connect(lambda: self.start_requested.emit(self._channel_idx))
        btn_row.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stoppen")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(
            "QPushButton { background: #922B21; color: white; border-radius: 4px;"
            "  padding: 4px 8px; font-size: 11px; }"
            "QPushButton:hover:enabled { background: #E74C3C; }"
            "QPushButton:disabled { background: #1A252F; color: #4A5568; }"
        )
        self._stop_btn.clicked.connect(lambda: self.stop_requested.emit(self._channel_idx))
        btn_row.addWidget(self._stop_btn)

        root.addLayout(btn_row)

    # ── Public API ────────────────────────────────────────────────────────────

    def update_frame(
        self,
        frame: np.ndarray,
        score: float,
        threshold: float,
        is_anomaly: bool,
    ) -> None:
        """Convert frame to QPixmap, update score bar and status label."""
        # Display frame
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(img)
            vw = self._video_lbl.width()
            vh = self._video_lbl.height()
            if vw > 0 and vh > 0:
                self._video_lbl.setPixmap(
                    pix.scaled(vw, vh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
            else:
                self._video_lbl.setPixmap(pix)
        except Exception:
            pass

        # Score bar
        bar_pct = int(min(score / max(3.0 * threshold, 1e-9), 1.0) * 100)
        self._score_bar.setValue(bar_pct)
        bar_color = "#E74C3C" if is_anomaly else "#27AE60"
        self._score_bar.setStyleSheet(
            f"QProgressBar {{ background: #1A252F; border-radius: 4px; }}"
            f"QProgressBar::chunk {{ background: {bar_color}; border-radius: 4px; }}"
        )

        # Status label
        if is_anomaly:
            self.set_status("ANOMALIE", "#E74C3C")
        else:
            self.set_status("Normal", "#27AE60")

    def set_configured(self, camera_name: str, model_name: str) -> None:
        """Update info label and enable the start button."""
        self._info_lbl.setText(f"{camera_name}  |  {model_name}")
        self._start_btn.setEnabled(True)
        self._status_lbl.setText("Gestoppt")
        self._status_lbl.setStyleSheet("color: #F39C12; font-size: 11px; padding: 2px;")

    def set_running(self, running: bool) -> None:
        """Toggle start / stop buttons."""
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        if not running:
            self._video_lbl.setText("Kein Signal")
            self._video_lbl.setPixmap(QPixmap())
            self._score_bar.setValue(0)
            self._score_bar.setStyleSheet(
                "QProgressBar { background: #1A252F; border-radius: 4px; }"
                "QProgressBar::chunk { background: #27AE60; border-radius: 4px; }"
            )
            self._status_lbl.setText("Gestoppt")
            self._status_lbl.setStyleSheet("color: #F39C12; font-size: 11px; padding: 2px;")

    def set_status(self, text: str, color: str) -> None:
        """Update the status label text and colour."""
        self._status_lbl.setText(text)
        self._status_lbl.setStyleSheet(
            f"color: {color}; font-size: 11px; padding: 2px; font-weight: bold;"
        )


# ---------------------------------------------------------------------------
# Configuration dialog
# ---------------------------------------------------------------------------

class _ConfigDialog(QDialog):
    """Dialog to configure a single channel's camera source and model."""

    def __init__(
        self,
        channel_idx: int,
        current_camera_idx: int,
        current_model_path: str,
        cameras: list,          # list of (idx, name) tuples
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Kanal {channel_idx + 1} konfigurieren")
        self.setMinimumWidth(460)
        self.setStyleSheet(
            "QDialog { background: #1C2A3A; color: #BDC3C7; }"
            "QLabel { color: #BDC3C7; }"
            "QComboBox, QLineEdit {"
            "  background: #0D1117; color: #E0E0E0;"
            "  border: 1px solid #2C3E50; border-radius: 4px; padding: 4px;"
            "}"
            "QPushButton {"
            "  background: #2C3E50; color: #BDC3C7;"
            "  border-radius: 4px; padding: 5px 12px;"
            "}"
            "QPushButton:hover { background: #34495E; }"
        )

        self._cameras = cameras

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Camera selector
        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Kamera:"))
        self._cam_combo = QComboBox()
        for idx, name in cameras:
            self._cam_combo.addItem(name, userData=idx)
        self._cam_combo.addItem("IP-Kamera...", userData="ip")
        # Restore previous selection
        for i in range(self._cam_combo.count()):
            if self._cam_combo.itemData(i) == current_camera_idx:
                self._cam_combo.setCurrentIndex(i)
                break
        cam_row.addWidget(self._cam_combo, stretch=1)
        layout.addLayout(cam_row)

        # Model path
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Modell:"))
        self._model_edit = QLineEdit(current_model_path)
        self._model_edit.setPlaceholderText("Modell-Datei (.pt / .onnx) wählen…")
        model_row.addWidget(self._model_edit, stretch=1)
        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(32)
        browse_btn.clicked.connect(self._browse_model)
        model_row.addWidget(browse_btn)
        layout.addLayout(model_row)

        # OK / Cancel
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Anomalie-Modell wählen",
            "",
            "Modell-Dateien (*.pth *.onnx);;Alle Dateien (*)",
        )
        if path:
            self._model_edit.setText(path)

    @property
    def selected_camera_idx(self) -> int:
        """Return the integer camera index chosen in the combo box."""
        data = self._cam_combo.currentData()
        if isinstance(data, int):
            return data
        return 0

    @property
    def selected_camera_name(self) -> str:
        """Return the display name for the selected camera entry."""
        return self._cam_combo.currentText()

    @property
    def selected_model_path(self) -> str:
        """Return the model file path typed / chosen in the dialog."""
        return self._model_edit.text().strip()


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

class MultiCameraPage(QWidget):
    """
    2×2 grid of independent camera+model monitoring channels.

    Each channel runs its own CameraFrameThread and AnomalyDetector.
    Frame callbacks from the background threads are dispatched to the UI
    thread via _frame_signal so Qt widget updates are always on the main thread.
    """

    # Signal emitted from the camera thread; handled on the UI thread.
    # Args: channel_idx, frame (np.ndarray), score, threshold, is_anomaly
    _frame_signal = Signal(int, object, float, float, bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._channels: list[_ChannelState] = []
        self._widgets: list[_ChannelWidget] = []
        self._notifier: Optional[AlarmNotifier] = None
        self._industrial_notifier: Optional[IndustrialNotifier] = None
        self._rest_server = None
        self._alarm_cooldown: float = 30.0
        self._cached_cameras: list = []   # [(idx, name), ...]
        self._num_channels: int = 2
        self._current_page: int = 0
        self._alarm_output_dir: str = "monitor_logs/multi_cam"

        self._frame_signal.connect(self._on_frame_ui)
        self._build_ui()
        self._apply_channel_count(2)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_notifier(self, n: AlarmNotifier) -> None:
        """Inject the AlarmNotifier used for email/webhook alarm notifications."""
        self._notifier = n

    def set_industrial_notifier(self, n: IndustrialNotifier) -> None:
        """Inject the IndustrialNotifier for OPC-UA / Modbus TCP alarm forwarding."""
        self._industrial_notifier = n

    def set_rest_server(self, server) -> None:
        """Wire in the REST API server."""
        self._rest_server = server
        if server is not None:
            try:
                server.set_mc_channel_count(self._num_channels)
            except Exception:
                pass

    # ── Camera count + pagination ─────────────────────────────────────────────

    @Slot(int)
    def _on_count_changed(self, count: int) -> None:
        self._apply_channel_count(count)

    def _apply_channel_count(self, count: int) -> None:
        """Resize channel and widget lists, then rebuild the page view."""
        self._on_stop_all()
        self._num_channels = count

        # Grow or shrink _channels (preserve existing configs)
        while len(self._channels) < count:
            self._channels.append(_ChannelState())
        while len(self._channels) > count:
            self._channels.pop()

        # Rebuild widget list
        for w in self._widgets:
            w.setParent(None)
        self._widgets.clear()

        for i in range(count):
            w = _ChannelWidget(i, self)
            w.configure_requested.connect(self._on_configure)
            w.start_requested.connect(self._on_start)
            w.stop_requested.connect(self._on_stop)
            self._widgets.append(w)

        # Clamp page index
        total_pages = max(1, (count + _MAX_PER_PAGE - 1) // _MAX_PER_PAGE)
        if self._current_page >= total_pages:
            self._current_page = total_pages - 1

        # Sync channel count to REST API
        if self._rest_server is not None:
            try:
                self._rest_server.set_mc_channel_count(count)
            except Exception:
                pass

        self._refresh_page_view()

    def _refresh_page_view(self) -> None:
        """Repopulate the grid with the widgets for the current page."""
        layout = self._grid_layout
        # Detach existing widgets without destroying them
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        start = self._current_page * _MAX_PER_PAGE
        end = min(start + _MAX_PER_PAGE, len(self._widgets))
        for slot, idx in enumerate(range(start, end)):
            self._grid_layout.addWidget(self._widgets[idx], slot // 2, slot % 2)
            self._widgets[idx].setParent(self._grid_widget)

        # Update page nav
        count = self._num_channels
        total_pages = max(1, (count + _MAX_PER_PAGE - 1) // _MAX_PER_PAGE)
        self._page_nav.setVisible(count > _MAX_PER_PAGE)
        self._page_label.setText(f"Seite {self._current_page + 1} / {total_pages}")
        self._prev_btn.setEnabled(self._current_page > 0)
        self._next_btn.setEnabled(self._current_page < total_pages - 1)

    def _on_page_prev(self) -> None:
        if self._current_page > 0:
            self._current_page -= 1
            self._refresh_page_view()

    def _on_page_next(self) -> None:
        total_pages = max(1, (self._num_channels + _MAX_PER_PAGE - 1) // _MAX_PER_PAGE)
        if self._current_page < total_pages - 1:
            self._current_page += 1
            self._refresh_page_view()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setStyleSheet(
            "MultiCameraPage { background: #0D1117; }"
            "QWidget { background: #0D1117; color: #BDC3C7; }"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ── Top bar ───────────────────────────────────────────────────────────
        top_bar = QHBoxLayout()

        title_lbl = QLabel("Multi-Kamera-Monitoring")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_lbl.setFont(title_font)
        title_lbl.setStyleSheet("color: #5DADE2;")
        top_bar.addWidget(title_lbl)

        top_bar.addSpacing(20)
        count_lbl = QLabel("Kanäle:")
        count_lbl.setStyleSheet("color: #BDC3C7; font-size: 13px;")
        top_bar.addWidget(count_lbl)

        self._count_spin = QSpinBox()
        self._count_spin.setRange(1, 9)
        self._count_spin.setValue(2)
        self._count_spin.setFixedWidth(60)
        self._count_spin.setToolTip(
            "Anzahl der Kamera-Kanäle (1–9). "
            "Bei mehr als 4 Kanälen wird in Seiten à 4 geblättert."
        )
        self._count_spin.setStyleSheet(
            "QSpinBox { background: #0D1117; color: #E0E0E0;"
            "  border: 1px solid #2C3E50; border-radius: 4px; padding: 3px; }"
        )
        self._count_spin.valueChanged.connect(self._on_count_changed)
        top_bar.addWidget(self._count_spin)

        top_bar.addStretch()

        self._start_all_btn = QPushButton("Alle starten")
        self._start_all_btn.setStyleSheet(
            "QPushButton { background: #1E8449; color: white; border-radius: 5px;"
            "  padding: 6px 16px; font-weight: bold; }"
            "QPushButton:hover { background: #27AE60; }"
        )
        self._start_all_btn.clicked.connect(self._on_start_all)
        top_bar.addWidget(self._start_all_btn)

        self._stop_all_btn = QPushButton("Alle stoppen")
        self._stop_all_btn.setStyleSheet(
            "QPushButton { background: #922B21; color: white; border-radius: 5px;"
            "  padding: 6px 16px; font-weight: bold; }"
            "QPushButton:hover { background: #E74C3C; }"
        )
        self._stop_all_btn.clicked.connect(self._on_stop_all)
        top_bar.addWidget(self._stop_all_btn)

        root.addLayout(top_bar)

        # ── Channel grid container ─────────────────────────────────────────────
        self._grid_widget = QWidget()
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setSpacing(8)
        root.addWidget(self._grid_widget, stretch=1)

        # ── Page navigation (visible only when count > _MAX_PER_PAGE) ─────────
        self._page_nav = QWidget()
        nav_row = QHBoxLayout(self._page_nav)
        nav_row.setContentsMargins(0, 0, 0, 0)
        nav_row.setSpacing(8)

        self._prev_btn = QPushButton("◀ Vorherige")
        self._prev_btn.setFixedWidth(120)
        self._prev_btn.setStyleSheet(
            "QPushButton { background: #2C3E50; color: #BDC3C7; border-radius: 4px;"
            "  padding: 4px 12px; }"
            "QPushButton:hover:enabled { background: #34495E; }"
            "QPushButton:disabled { background: #1A252F; color: #4A5568; }"
        )
        self._prev_btn.clicked.connect(self._on_page_prev)
        nav_row.addWidget(self._prev_btn)

        self._page_label = QLabel("Seite 1 / 1")
        self._page_label.setAlignment(Qt.AlignCenter)
        self._page_label.setStyleSheet("color: #BDC3C7; font-size: 12px;")
        nav_row.addWidget(self._page_label, stretch=1)

        self._next_btn = QPushButton("Nächste ▶")
        self._next_btn.setFixedWidth(120)
        self._next_btn.setStyleSheet(
            "QPushButton { background: #2C3E50; color: #BDC3C7; border-radius: 4px;"
            "  padding: 4px 12px; }"
            "QPushButton:hover:enabled { background: #34495E; }"
            "QPushButton:disabled { background: #1A252F; color: #4A5568; }"
        )
        self._next_btn.clicked.connect(self._on_page_next)
        nav_row.addWidget(self._next_btn)

        self._page_nav.setVisible(False)
        root.addWidget(self._page_nav)

        # ── Alarm log ─────────────────────────────────────────────────────────
        alarm_grp = QGroupBox("Alarm-Ereignisse")
        alarm_grp.setStyleSheet(
            "QGroupBox { background: #1C2A3A; border: 1px solid #2C3E50;"
            "  border-radius: 6px; margin-top: 8px; color: #BDC3C7; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px;"
            "  padding: 0 4px; color: #E67E22; }"
        )
        alarm_layout = QVBoxLayout(alarm_grp)
        alarm_layout.setContentsMargins(6, 14, 6, 6)

        self._alarm_log = QPlainTextEdit()
        self._alarm_log.setReadOnly(True)
        self._alarm_log.setMaximumBlockCount(200)
        self._alarm_log.setFixedHeight(100)
        self._alarm_log.setStyleSheet(
            "QPlainTextEdit {"
            "  background: #0D1117; color: #E67E22;"
            "  border: none; border-radius: 4px;"
            "  font-family: monospace; font-size: 11px;"
            "}"
        )
        alarm_layout.addWidget(self._alarm_log)
        root.addWidget(alarm_grp)

    # ── Channel configuration ─────────────────────────────────────────────────

    @Slot(int)
    def _on_configure(self, channel_idx: int) -> None:
        """Open the configuration dialog for the given channel."""
        # Scan cameras if we haven't yet (or always use cached list for speed)
        if not self._cached_cameras:
            self._cached_cameras = list_usb_cameras()

        state = self._channels[channel_idx]
        dlg = _ConfigDialog(
            channel_idx=channel_idx,
            current_camera_idx=state.camera_idx,
            current_model_path=state.model_path,
            cameras=self._cached_cameras,
            parent=self,
        )
        if dlg.exec() != QDialog.Accepted:
            return

        model_path = dlg.selected_model_path
        if not model_path:
            return

        # Load detector — try AnomalyDetector (.pth) first, then OnnxAnomalyScorer
        detector = None
        try:
            if model_path.lower().endswith(".onnx") and HAS_ORT and OnnxAnomalyScorer is not None:
                detector = OnnxAnomalyScorer.from_path(model_path)
            else:
                det = AnomalyDetector()
                det.load(model_path)
                detector = det
        except Exception:
            # Fallback: try the other loader
            try:
                if OnnxAnomalyScorer is not None and HAS_ORT:
                    detector = OnnxAnomalyScorer.from_path(model_path)
                else:
                    det = AnomalyDetector()
                    det.load(model_path)
                    detector = det
            except Exception:
                detector = None

        if detector is None:
            return

        # Read ROI and threshold from model metadata
        try:
            meta = detector.metadata
            roi = meta.get("roi")          # [x1,y1,x2,y2] normalised, or None
            threshold = float(detector.threshold)
        except Exception:
            roi = None
            threshold = 0.001

        # Update state
        state.detector = detector
        state.model_path = model_path
        state.camera_idx = dlg.selected_camera_idx
        state.roi = roi
        state.threshold = threshold

        camera_name = dlg.selected_camera_name
        model_name = os.path.basename(model_path)
        self._widgets[channel_idx].set_configured(camera_name, model_name)

    # ── Channel start / stop ──────────────────────────────────────────────────

    @Slot(int)
    def _on_start(self, channel_idx: int) -> None:
        """Start the camera thread for the given channel."""
        state = self._channels[channel_idx]
        if state.running or state.detector is None:
            return

        thread = CameraFrameThread(state.camera_idx, fps=15.0, parent=None)
        # Use a closure that captures channel_idx by value
        def _on_raw_frame(frame: np.ndarray, idx: int = channel_idx) -> None:
            self._on_frame(idx, frame)

        thread.frame_ready.connect(_on_raw_frame)
        thread.start()

        state.camera_thread = thread
        state.running = True
        state.frame_counter = 0
        self._widgets[channel_idx].set_running(True)
        self._widgets[channel_idx].set_status("Verbunden", "#2ECC71")

    @Slot(int)
    def _on_stop(self, channel_idx: int) -> None:
        """Stop the camera thread for the given channel."""
        state = self._channels[channel_idx]
        if state.camera_thread is not None:
            state.camera_thread.stop()
            state.camera_thread = None
        state.running = False
        self._widgets[channel_idx].set_running(False)

    def _on_start_all(self) -> None:
        """Start all configured (but not yet running) channels."""
        for i in range(self._num_channels):
            if not self._channels[i].running and self._channels[i].detector is not None:
                self._on_start(i)

    def _on_stop_all(self) -> None:
        """Stop all running channels."""
        for i in range(self._num_channels):
            if i < len(self._channels) and self._channels[i].running:
                self._on_stop(i)

    # ── Frame processing (camera thread) ─────────────────────────────────────

    def _on_frame(self, channel_idx: int, frame: np.ndarray) -> None:
        """
        Called from the camera background thread for each incoming frame.

        Scoring is performed every 3rd frame for performance. The result is
        dispatched to the UI thread via _frame_signal.
        """
        state = self._channels[channel_idx]
        if not state.running or state.detector is None:
            return

        state.frame_counter += 1

        score = state.score
        is_anomaly = state.is_anomaly

        # Score only every 3rd frame
        if state.frame_counter % 3 == 0:
            try:
                cropped = self._apply_roi_crop(frame, state.roi)
                result = state.detector.score_detailed(cropped)
                score = float(result[0])
                is_anomaly = score > state.threshold
                state.score = score
                state.is_anomaly = is_anomaly
            except Exception:
                pass

        # Push score to REST API (every scored frame)
        if self._rest_server is not None and state.frame_counter % 3 == 0:
            try:
                self._rest_server.push_mc_score(channel_idx, score, state.threshold)
            except Exception:
                pass

        # Fire alarm if anomaly and cooldown has expired
        if is_anomaly:
            now = time.perf_counter()
            if (now - state.last_alarm_t) >= self._alarm_cooldown:
                state.last_alarm_t = now
                state.event_count += 1
                ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                thr = state.threshold

                # Save alarm JPEG
                fname = self._save_alarm_frame(frame, channel_idx)

                log_msg = (
                    f"[{ts}] Kanal {channel_idx + 1}:"
                    f" Score={score:.5f} > Thr={thr:.5f}"
                    + (f"  → {fname}" if fname else "")
                )

                # Push alarm to REST API
                if self._rest_server is not None:
                    try:
                        self._rest_server.push_mc_alarm(
                            channel_idx, score, thr, fname
                        )
                    except Exception:
                        pass

                # Dispatch log + notifier call via signal (reaches UI thread)
                self._frame_signal.emit(
                    channel_idx, frame, score, state.threshold, is_anomaly
                )
                # Notifier can be called from any thread
                if self._notifier is not None:
                    model_name = os.path.basename(state.model_path)
                    fpath = os.path.join(self._alarm_output_dir, fname) if fname else ""
                    try:
                        self._notifier.notify(
                            score, thr,
                            frame_path=fpath,
                            model_name=model_name,
                        )
                    except Exception:
                        pass
                # Industrial protocol (OPC-UA / Modbus TCP)
                if self._industrial_notifier is not None:
                    try:
                        self._industrial_notifier.on_alarm(True, score, thr)
                    except Exception:
                        pass
                # We already emitted the signal above; return to avoid double emit
                QMetaObject.invokeMethod(
                    self._alarm_log,
                    "appendPlainText",
                    Qt.QueuedConnection,
                    Q_ARG(str, log_msg),
                )
                return

        # Normal frame update (no alarm)
        self._frame_signal.emit(
            channel_idx, frame, score, state.threshold, is_anomaly
        )

    @Slot(int, object, float, float, bool)
    def _on_frame_ui(
        self,
        channel_idx: int,
        frame: object,
        score: float,
        threshold: float,
        is_anomaly: bool,
    ) -> None:
        """UI-thread handler: update the channel widget with the latest frame data."""
        if not isinstance(frame, np.ndarray):
            return
        self._widgets[channel_idx].update_frame(frame, score, threshold, is_anomaly)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _save_alarm_frame(self, frame: np.ndarray, channel_idx: int) -> str:
        """Save *frame* as a JPEG alarm snapshot; return the filename or ''."""
        try:
            os.makedirs(self._alarm_output_dir, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            fname = f"mc_ch{channel_idx + 1}_{ts}.jpg"
            fpath = os.path.join(self._alarm_output_dir, fname)
            cv2.imwrite(fpath, frame)
            # Tell REST server where to serve alarm frames from
            if self._rest_server is not None:
                try:
                    self._rest_server.set_alarm_frame_dir(self._alarm_output_dir)
                except Exception:
                    pass
            return fname
        except Exception:
            return ""

    @staticmethod
    def _apply_roi_crop(frame: np.ndarray, roi: Optional[list]) -> np.ndarray:
        """
        Crop frame to the normalised ROI region [x1,y1,x2,y2].

        Returns the full frame unchanged when roi is None or the crop is too small.
        """
        if roi is None:
            return frame
        h, w = frame.shape[:2]
        x1 = int(roi[0] * w); y1 = int(roi[1] * h)
        x2 = int(roi[2] * w); y2 = int(roi[3] * h)
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return frame
        return frame[y1:y2, x1:x2]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def hideEvent(self, event) -> None:
        """Stop all channel threads when the page is hidden."""
        self._on_stop_all()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:
        """Stop all channel threads on close."""
        self._on_stop_all()
        super().closeEvent(event)
