from __future__ import annotations
import io
import json
import urllib.request
import urllib.error
from typing import Optional

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QLabel, QCheckBox, QDialog, QFormLayout,
    QLineEdit, QDialogButtonBox, QTextEdit, QMessageBox, QAbstractItemView,
    QTabWidget, QProgressBar, QSpinBox, QDoubleSpinBox, QSizePolicy,
    QGroupBox,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSettings
from PySide6.QtGui import QColor, QFont, QPixmap, QImage


class _PollThread(QThread):
    """Pollt /api/status für alle registrierten Geräte."""
    result = Signal(str, dict)   # url, result_dict (oder {"error": msg})

    def __init__(self, devices: list, parent=None) -> None:
        super().__init__(parent)
        self._devices = list(devices)

    def run(self) -> None:
        for device in self._devices:
            url = device.get("url", "").rstrip("/")
            api_key = device.get("api_key", "")
            status_url = f"{url}/api/status"
            try:
                req = urllib.request.Request(status_url, method="GET")
                if api_key:
                    req.add_header("X-Api-Key", api_key)
                req.add_header("Accept", "application/json")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                self.result.emit(url, data)
            except urllib.error.HTTPError as exc:
                self.result.emit(url, {"error": f"HTTP {exc.code}"})
            except Exception as exc:
                self.result.emit(url, {"error": str(exc)[:80]})


class _AddDeviceDialog(QDialog):
    def __init__(self, parent=None) -> None:
        from utils.i18n import tr
        super().__init__(parent)
        self.setWindowTitle(tr("fleet.add_dlg_title"))
        self.setMinimumWidth(380)
        layout = QFormLayout(self)

        self._name = QLineEdit()
        self._name.setPlaceholderText("z.B. Kamera Nord")
        layout.addRow(tr("fleet.device_name_label"), self._name)

        self._url = QLineEdit()
        self._url.setPlaceholderText("http://192.168.1.100:8766")
        layout.addRow(tr("fleet.device_url_label"), self._url)

        self._key = QLineEdit()
        self._key.setPlaceholderText("optional")
        self._key.setEchoMode(QLineEdit.Password)
        layout.addRow(tr("fleet.device_apikey_label"), self._key)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _validate_and_accept(self) -> None:
        from utils.i18n import tr
        url = self._url.text().strip()
        if not url.startswith(("http://", "https://")):
            QMessageBox.warning(self, tr("common.warning"), tr("fleet.invalid_url_msg"))
            return
        if not self._name.text().strip():
            QMessageBox.warning(self, tr("common.warning"), tr("fleet.missing_name_msg"))
            return
        self.accept()

    @property
    def device(self) -> dict:
        return {
            "name": self._name.text().strip(),
            "url": self._url.text().strip().rstrip("/"),
            "api_key": self._key.text(),
        }


# ---------------------------------------------------------------------------
# Remote-Training helpers
# ---------------------------------------------------------------------------

class _FrameDownloadThread(QThread):
    """Downloads frames ZIP from GET /api/frames and feeds them to AnomalyDetector.

    One HTTP request replaces 150 individual frame polls.
    Signals: progress(int 0-100), finished(int frame_count), error(str).
    """

    progress = Signal(int)
    finished = Signal(int)
    error    = Signal(str)

    def __init__(self, device_url: str, api_key: str, count: int,
                 detector, parent=None) -> None:
        super().__init__(parent)
        self._url      = device_url.rstrip("/")
        self._api_key  = api_key
        self._count    = count
        self._detector = detector

    def run(self) -> None:
        import zipfile, cv2
        try:
            req = urllib.request.Request(
                f"{self._url}/api/frames?n={self._count}")
            if self._api_key:
                req.add_header("X-Api-Key", self._api_key)
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
        except Exception as exc:
            self.error.emit(f"Download fehlgeschlagen: {exc}")
            return

        try:
            zf = zipfile.ZipFile(io.BytesIO(raw))
        except Exception as exc:
            self.error.emit(f"ZIP ungültig: {exc}")
            return

        names = zf.namelist()
        if not names:
            self.error.emit("ZIP enthält keine Frames.")
            return

        loaded = 0
        for i, name in enumerate(names):
            jpg = zf.read(name)
            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                self._detector.collect_frame(frame)
                loaded += 1
            self.progress.emit(int((i + 1) / len(names) * 100))

        zf.close()
        self.finished.emit(loaded)


class _LocalTrainThread(QThread):
    """
    Runs AnomalyDetector.train() in a background thread.

    Signals
    -------
    progress(int, int, float):
        Emitted per epoch: (epoch, total_epochs, loss).
    finished(float):
        Emitted when training completes with the computed threshold.
    error(str):
        Emitted when training raises an exception.
    """

    progress = Signal(int, int, float)
    finished = Signal(float)
    error = Signal(str)

    def __init__(self, detector, epochs: int, parent=None) -> None:
        super().__init__(parent)
        self._detector = detector
        self._epochs = epochs

    def run(self) -> None:
        """Invoke detector.train() and relay progress via signals."""
        try:
            def _cb(epoch: int, total: int, loss: float) -> None:
                self.progress.emit(epoch, total, loss)

            threshold = self._detector.train(
                epochs=self._epochs,
                progress_cb=_cb,
            )
            self.finished.emit(float(threshold))
        except Exception as exc:
            self.error.emit(str(exc))


class _RemoteTrainDialog(QDialog):
    """
    Guides the user through remote training for one monitor.py device.

    Tab 1 — Frames & Training: download buffered frames via GET /api/frames,
                               train an AnomalyDetector locally.
    Tab 2 — Deployen:          hot-swap the .pth model via POST /api/deploy.
    """

    def __init__(self, device: dict, parent=None) -> None:
        super().__init__(parent)
        self._device     = device
        self._device_url = device.get("url", "").rstrip("/")
        self._api_key    = device.get("api_key", "")
        self._detector   = None
        self._dl_thread: Optional[_FrameDownloadThread] = None
        self._train_thread: Optional[_LocalTrainThread] = None
        self._trained_threshold: float = 0.0

        from utils.i18n import tr
        self.setWindowTitle(tr("fleet.remote_train_dlg", device=device.get("name", "")))
        self.setMinimumSize(480, 460)
        self._build_ui()
        self._check_device_status()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self._tabs = QTabWidget()
        root.addWidget(self._tabs)
        self._build_tab_frames()
        self._build_tab_deploy()
        self._tabs.setTabEnabled(1, False)

    def _build_tab_frames(self) -> None:
        from utils.i18n import tr
        tab = QWidget()
        lay = QVBoxLayout(tab)

        self._status_label = QLabel("Verbinde mit Gerät…")
        self._status_label.setStyleSheet("color: #8B949E; font-size: 11px;")
        self._status_label.setWordWrap(True)
        lay.addWidget(self._status_label)

        dl_grp = QGroupBox("Frames herunterladen")
        dl_lay = QVBoxLayout(dl_grp)

        cnt_row = QHBoxLayout()
        cnt_row.addWidget(QLabel("Anzahl:"))
        self._frame_count_spin = QSpinBox()
        self._frame_count_spin.setRange(30, 500)
        self._frame_count_spin.setValue(150)
        cnt_row.addWidget(self._frame_count_spin)
        cnt_row.addStretch()
        dl_lay.addLayout(cnt_row)

        self._dl_progress = QProgressBar()
        self._dl_progress.setRange(0, 100)
        self._dl_progress.setValue(0)
        dl_lay.addWidget(self._dl_progress)

        self._dl_label = QLabel("Bereit")
        self._dl_label.setStyleSheet("color: #8B949E; font-size: 11px;")
        dl_lay.addWidget(self._dl_label)

        self._dl_btn = QPushButton(tr("fleet.collect_btn"))
        self._dl_btn.setStyleSheet("background: #238636; color: white; font-weight: bold;")
        self._dl_btn.clicked.connect(self._start_download)
        dl_lay.addWidget(self._dl_btn)
        lay.addWidget(dl_grp)

        train_grp = QGroupBox("Modell trainieren")
        train_lay = QVBoxLayout(train_grp)

        ep_row = QHBoxLayout()
        ep_row.addWidget(QLabel("Epochen:"))
        self._epoch_spin = QSpinBox()
        self._epoch_spin.setRange(10, 100)
        self._epoch_spin.setValue(40)
        ep_row.addWidget(self._epoch_spin)
        ep_row.addStretch()
        train_lay.addLayout(ep_row)

        self._train_progress = QProgressBar()
        self._train_progress.setRange(0, 100)
        self._train_progress.setValue(0)
        train_lay.addWidget(self._train_progress)

        self._train_status_label = QLabel("Bereit")
        self._train_status_label.setStyleSheet("color: #8B949E; font-size: 11px;")
        train_lay.addWidget(self._train_status_label)

        self._train_btn = QPushButton(tr("fleet.train_btn"))
        self._train_btn.setStyleSheet("background: #1C6EA4; color: white; font-weight: bold;")
        self._train_btn.setEnabled(False)
        self._train_btn.clicked.connect(self._start_training)
        train_lay.addWidget(self._train_btn)
        lay.addWidget(train_grp)

        self._tabs.addTab(tab, tr("fleet.tab.training"))

    def _build_tab_deploy(self) -> None:
        from utils.i18n import tr
        tab = QWidget()
        lay = QVBoxLayout(tab)

        self._deploy_info = QLabel("Modell noch nicht trainiert.")
        self._deploy_info.setStyleSheet("color: #8B949E; font-size: 12px;")
        self._deploy_info.setWordWrap(True)
        lay.addWidget(self._deploy_info)

        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Schwellwert:"))
        self._thr_spin = QDoubleSpinBox()
        self._thr_spin.setDecimals(6)
        self._thr_spin.setRange(0.0, 100.0)
        self._thr_spin.setSingleStep(0.0001)
        thr_row.addWidget(self._thr_spin)
        thr_row.addStretch()
        lay.addLayout(thr_row)

        self._deploy_btn = QPushButton(tr("fleet.deploy_btn"))
        self._deploy_btn.setStyleSheet("background: #1A8754; color: white; font-weight: bold; padding: 8px;")
        self._deploy_btn.setEnabled(False)
        self._deploy_btn.clicked.connect(self._deploy_model)
        lay.addWidget(self._deploy_btn)

        self._deploy_result = QLabel("")
        self._deploy_result.setWordWrap(True)
        self._deploy_result.setStyleSheet("font-size: 12px;")
        lay.addWidget(self._deploy_result)

        lay.addStretch()
        self._tabs.addTab(tab, tr("fleet.tab.deploy"))

    # ── Status check ──────────────────────────────────────────────────────────

    def _check_device_status(self) -> None:
        """Query GET /api/status and show frame_count + model info."""
        try:
            req = urllib.request.Request(f"{self._device_url}/api/status")
            if self._api_key:
                req.add_header("X-Api-Key", self._api_key)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            fc    = data.get("frame_count", 0)
            model = data.get("model_name", "—")
            self._status_label.setText(
                f"Gerät erreichbar  |  {fc} Frames im Puffer  |  Modell: {model}")
            self._status_label.setStyleSheet("color: #2ECC71; font-size: 11px;")
        except Exception as exc:
            self._status_label.setText(f"Gerät nicht erreichbar: {exc}")
            self._status_label.setStyleSheet("color: #E74C3C; font-size: 11px;")

    # ── Tab 1: Frames & Training ───────────────────────────────────────────────

    def _start_download(self) -> None:
        if self._dl_thread and self._dl_thread.isRunning():
            return

        if self._detector is None:
            try:
                from core.anomaly_detector import AnomalyDetector
                self._detector = AnomalyDetector()
            except Exception as exc:
                QMessageBox.critical(self, "Fehler",
                                     f"AnomalyDetector konnte nicht erstellt werden:\n{exc}")
                return

        count = self._frame_count_spin.value()
        self._dl_btn.setEnabled(False)
        self._dl_progress.setValue(0)
        self._dl_label.setText("Lade Frames herunter…")

        self._dl_thread = _FrameDownloadThread(
            device_url=self._device_url,
            api_key=self._api_key,
            count=count,
            detector=self._detector,
            parent=self,
        )
        self._dl_thread.progress.connect(self._on_dl_progress)
        self._dl_thread.finished.connect(self._on_dl_finished)
        self._dl_thread.error.connect(self._on_dl_error)
        self._dl_thread.start()

    def _on_dl_progress(self, pct: int) -> None:
        self._dl_progress.setValue(pct)

    def _on_dl_finished(self, count: int) -> None:
        self._dl_label.setText(f"{count} Frames geladen.")
        self._dl_btn.setEnabled(True)
        if count >= 30:
            self._train_btn.setEnabled(True)
            self._train_status_label.setText(f"{count} Frames bereit — Training möglich.")

    def _on_dl_error(self, msg: str) -> None:
        self._dl_label.setText(f"Fehler: {msg}")
        self._dl_btn.setEnabled(True)

    def _start_training(self) -> None:
        if self._detector is None:
            return
        epochs = self._epoch_spin.value()
        self._train_btn.setEnabled(False)
        self._train_status_label.setText("Training läuft…")
        self._train_thread = _LocalTrainThread(self._detector, epochs, parent=self)
        self._train_thread.progress.connect(self._on_train_progress)
        self._train_thread.finished.connect(self._on_train_finished)
        self._train_thread.error.connect(self._on_train_error)
        self._train_thread.start()

    def _on_train_progress(self, epoch: int, total: int, loss: float) -> None:
        pct = int(epoch / max(total, 1) * 100)
        self._train_progress.setValue(pct)
        self._train_status_label.setText(f"Epoch {epoch}/{total} — Loss: {loss:.6f}")

    def _on_train_finished(self, threshold: float) -> None:
        self._trained_threshold = threshold
        self._train_progress.setValue(100)
        self._train_status_label.setText(
            f"Training abgeschlossen. Schwellwert: {threshold:.6f}")
        self._thr_spin.setValue(threshold)
        self._deploy_info.setText(f"Modell trainiert.\nAuto-Schwellwert: {threshold:.6f}")
        self._deploy_btn.setEnabled(True)
        self._tabs.setTabEnabled(1, True)
        self._tabs.setCurrentIndex(1)

    def _on_train_error(self, msg: str) -> None:
        self._train_btn.setEnabled(True)
        self._train_status_label.setText(f"Fehler: {msg}")
        QMessageBox.critical(self, "Trainingsfehler", msg)

    # ── Tab 2: Deploy ─────────────────────────────────────────────────────────

    def _deploy_model(self) -> None:
        """Save model to a temp file and POST to POST /api/deploy (hot-swap)."""
        if self._detector is None:
            return
        import tempfile, os
        try:
            with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
                tmp_path = tmp.name
            self._detector.threshold = self._thr_spin.value()
            self._detector.save(tmp_path)
            with open(tmp_path, "rb") as fh:
                model_data = fh.read()
            os.unlink(tmp_path)
        except Exception as exc:
            self._deploy_result.setText(f"Fehler beim Speichern: {exc}")
            self._deploy_result.setStyleSheet("color: #E74C3C;")
            return

        boundary = "PictureStudioBoundary42"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="model"; filename="model.pth"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode() + model_data + f"\r\n--{boundary}--\r\n".encode()

        try:
            req = urllib.request.Request(
                f"{self._device_url}/api/deploy", data=body, method="POST")
            req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
            req.add_header("Content-Length", str(len(body)))
            if self._api_key:
                req.add_header("X-Api-Key", self._api_key)
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())
            from utils.i18n import tr
            self._deploy_result.setText(
                tr("fleet.deploy_success", path=result.get("model_path", "?")))
            self._deploy_result.setStyleSheet("color: #2ECC71;")
            self._deploy_btn.setEnabled(False)
        except Exception as exc:
            from utils.i18n import tr
            self._deploy_result.setText(tr("fleet.deploy_failed", msg=exc))
            self._deploy_result.setStyleSheet("color: #E74C3C;")

    def closeEvent(self, event) -> None:
        if self._dl_thread and self._dl_thread.isRunning():
            self._dl_thread.terminate()
            self._dl_thread.wait(1000)
        if self._train_thread and self._train_thread.isRunning():
            self._train_thread.terminate()
        super().closeEvent(event)


class FleetPage(QWidget):
    """
    Fleet-Management: überwacht mehrere remote monitor.py Instanzen.
    Geräte werden persistent in QSettings gespeichert.
    """

    _SETTINGS_KEY = "fleet/devices"
    _COL_NAME, _COL_URL, _COL_STATUS, _COL_SCORE, _COL_ALARM, _COL_ACTIONS = range(6)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._devices: list[dict] = []
        self._poll_thread: Optional[_PollThread] = None
        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(30_000)
        self._auto_timer.timeout.connect(self._poll_all)
        self._build_ui()
        self._load_devices()

    def set_project(self, project, audit=None) -> None:
        pass   # no project dependency

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        from utils.i18n import tr
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        title = QLabel(tr("fleet.title"))
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #E6EDF3;")
        root.addWidget(title)

        # Top bar
        top = QHBoxLayout()
        add_btn = QPushButton(tr("fleet.add_device_btn"))
        add_btn.setStyleSheet("background: #238636; color: white; border-radius: 4px; padding: 5px 12px; font-weight: bold;")
        add_btn.clicked.connect(self._add_device)
        top.addWidget(add_btn)

        refresh_btn = QPushButton(tr("fleet.refresh_all_btn"))
        refresh_btn.setStyleSheet("background: #1F6FEB; color: white; border-radius: 4px; padding: 5px 12px;")
        refresh_btn.clicked.connect(self._poll_all)
        top.addWidget(refresh_btn)

        self._auto_cb = QCheckBox(tr("fleet.auto_refresh_cb"))
        self._auto_cb.toggled.connect(self._on_auto_toggled)
        top.addWidget(self._auto_cb)
        top.addStretch()
        root.addLayout(top)

        # Table
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            tr("fleet.col.name"), tr("fleet.col.url"), tr("fleet.col.status"),
            tr("fleet.col.score"), tr("fleet.col.alarm"), tr("fleet.col.actions"),
        ])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for col in (0, 2, 3, 4, 5):
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            "QTableWidget { background: #0D1117; color: #E6EDF3; gridline-color: #21262D; border: 1px solid #30363D; }"
            "QTableWidget::item:selected { background: #1F3A5F; }"
            "QHeaderView::section { background: #161B22; color: #8B949E; border: 1px solid #21262D; padding: 4px; }"
        )
        root.addWidget(self.table)

        # Log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(120)
        self._log.setStyleSheet("background: #0D1117; color: #8B949E; border: 1px solid #30363D; font-family: monospace; font-size: 11px;")
        root.addWidget(self._log)

    # ── Device management ─────────────────────────────────────────────────────

    def _load_devices(self) -> None:
        s = QSettings("ImageLabelingStudio", "ILS")
        raw = s.value(self._SETTINGS_KEY, "[]")
        try:
            self._devices = json.loads(raw) if isinstance(raw, str) else list(raw or [])
        except Exception:
            self._devices = []
        self._rebuild_table()

    def _save_devices(self) -> None:
        s = QSettings("ImageLabelingStudio", "ILS")
        s.setValue(self._SETTINGS_KEY, json.dumps(self._devices))

    def _add_device(self) -> None:
        dlg = _AddDeviceDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        self._devices.append(dlg.device)
        self._save_devices()
        self._rebuild_table()
        self._log.append(f"Gerät hinzugefügt: {dlg.device['name']} — {dlg.device['url']}")

    def _remove_device(self, idx: int) -> None:
        if 0 <= idx < len(self._devices):
            name = self._devices[idx].get("name", "?")
            del self._devices[idx]
            self._save_devices()
            self._rebuild_table()
            self._log.append(f"Gerät entfernt: {name}")

    def _open_dashboard(self, url: str) -> None:
        import webbrowser
        webbrowser.open(f"{url}/dashboard")

    def _open_setup(self, url: str) -> None:
        """Open the monitor.py setup web interface in the default browser."""
        import webbrowser
        webbrowser.open(f"{url}/setup")

    def _open_remote_training(self, device: dict) -> None:
        """Open the remote training dialog for the given device."""
        from core.anomaly_detector import AnomalyDetector  # lazy import
        dlg = _RemoteTrainDialog(device, parent=self)
        dlg.exec()

    def _rebuild_table(self) -> None:
        from utils.i18n import tr
        self.table.setRowCount(len(self._devices))
        for row, dev in enumerate(self._devices):
            self.table.setItem(row, self._COL_NAME, QTableWidgetItem(dev.get("name", "")))
            self.table.setItem(row, self._COL_URL, QTableWidgetItem(dev.get("url", "")))
            status_item = QTableWidgetItem(tr("fleet.status_unknown"))
            status_item.setForeground(QColor("#8B949E"))
            self.table.setItem(row, self._COL_STATUS, status_item)
            self.table.setItem(row, self._COL_SCORE, QTableWidgetItem("–"))
            self.table.setItem(row, self._COL_ALARM, QTableWidgetItem("–"))

            # Action buttons widget
            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(2, 2, 2, 2)
            btn_layout.setSpacing(4)

            url = dev.get("url", "")

            dash_btn = QPushButton("Dashboard")
            dash_btn.setFixedHeight(22)
            dash_btn.setStyleSheet("background: #1F6FEB; color: white; border-radius: 3px; font-size: 10px; padding: 0 6px;")
            dash_btn.clicked.connect(lambda _, u=url: self._open_dashboard(u))
            btn_layout.addWidget(dash_btn)

            setup_btn = QPushButton(tr("fleet.action_setup"))
            setup_btn.setFixedHeight(22)
            setup_btn.setStyleSheet("background: #6C3483; color: white; border-radius: 3px; font-size: 10px; padding: 0 6px;")
            setup_btn.clicked.connect(lambda _, u=url: self._open_setup(u))
            btn_layout.addWidget(setup_btn)

            train_btn = QPushButton(tr("fleet.action_training"))
            train_btn.setFixedHeight(22)
            train_btn.setStyleSheet("background: #1A5276; color: white; border-radius: 3px; font-size: 10px; padding: 0 6px;")
            train_btn.clicked.connect(lambda _, d=dict(dev): self._open_remote_training(d))
            btn_layout.addWidget(train_btn)

            del_btn = QPushButton(tr("fleet.action_delete"))
            del_btn.setFixedHeight(22)
            del_btn.setStyleSheet("background: #6E2C2C; color: white; border-radius: 3px; font-size: 10px; padding: 0 6px;")
            del_btn.clicked.connect(lambda _, r=row: self._remove_device(r))
            btn_layout.addWidget(del_btn)

            self.table.setCellWidget(row, self._COL_ACTIONS, btn_widget)

    # ── Polling ───────────────────────────────────────────────────────────────

    def _on_auto_toggled(self, enabled: bool) -> None:
        if enabled:
            self._auto_timer.start()
            self._poll_all()
        else:
            self._auto_timer.stop()

    def _poll_all(self) -> None:
        if not self._devices:
            return
        if self._poll_thread and self._poll_thread.isRunning():
            return
        from utils.i18n import tr
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self._COL_STATUS)
            if item:
                item.setText(tr("fleet.status_checking"))
                item.setForeground(QColor("#F39C12"))

        self._poll_thread = _PollThread(self._devices, self)
        self._poll_thread.result.connect(self._on_poll_result)
        self._poll_thread.start()

    def _on_poll_result(self, url: str, data: dict) -> None:
        # Find row by URL
        for row, dev in enumerate(self._devices):
            if dev.get("url", "") == url:
                status_item = self.table.item(row, self._COL_STATUS)
                score_item = self.table.item(row, self._COL_SCORE)
                alarm_item = self.table.item(row, self._COL_ALARM)
                if status_item is None:
                    return
                from utils.i18n import tr
                if "error" in data:
                    status_item.setText(f"{tr('fleet.status_offline')} ({data['error']})")
                    status_item.setForeground(QColor("#E74C3C"))
                else:
                    status_item.setText(tr("fleet.status_online"))
                    status_item.setForeground(QColor("#2ECC71"))
                    if score_item:
                        score = data.get("score", data.get("last_score", "–"))
                        score_item.setText(f"{score:.4f}" if isinstance(score, float) else str(score))
                    if alarm_item:
                        alarm = data.get("latest_alarm", {})
                        if isinstance(alarm, dict) and alarm.get("timestamp"):
                            alarm_item.setText(str(alarm["timestamp"])[:19])
                        else:
                            alarm_item.setText("–")
                self._log.append(f"[{url}] {'OK' if 'error' not in data else 'FEHLER: ' + data['error']}")
                break
