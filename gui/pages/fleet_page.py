from __future__ import annotations
import json
import urllib.request
import urllib.error
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QLabel, QCheckBox, QDialog, QFormLayout,
    QLineEdit, QDialogButtonBox, QTextEdit, QMessageBox, QAbstractItemView,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSettings
from PySide6.QtGui import QColor, QFont


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
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                self.result.emit(url, data)
            except urllib.error.HTTPError as exc:
                self.result.emit(url, {"error": f"HTTP {exc.code}"})
            except Exception as exc:
                self.result.emit(url, {"error": str(exc)[:80]})


class _AddDeviceDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gerät hinzufügen")
        self.setMinimumWidth(380)
        layout = QFormLayout(self)

        self._name = QLineEdit()
        self._name.setPlaceholderText("z.B. Kamera Nord")
        layout.addRow("Name:", self._name)

        self._url = QLineEdit()
        self._url.setPlaceholderText("http://192.168.1.100:8765")
        layout.addRow("URL:", self._url)

        self._key = QLineEdit()
        self._key.setPlaceholderText("optional")
        self._key.setEchoMode(QLineEdit.Password)
        layout.addRow("API-Key:", self._key)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _validate_and_accept(self) -> None:
        url = self._url.text().strip()
        if not url.startswith(("http://", "https://")):
            QMessageBox.warning(self, "Ungültige URL", "URL muss mit http:// oder https:// beginnen.")
            return
        if not self._name.text().strip():
            QMessageBox.warning(self, "Name fehlt", "Bitte einen Namen eingeben.")
            return
        self.accept()

    @property
    def device(self) -> dict:
        return {
            "name": self._name.text().strip(),
            "url": self._url.text().strip().rstrip("/"),
            "api_key": self._key.text(),
        }


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
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        title = QLabel("🌐 Fleet-Management")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #E6EDF3;")
        root.addWidget(title)

        # Top bar
        top = QHBoxLayout()
        add_btn = QPushButton("+ Gerät hinzufügen")
        add_btn.setStyleSheet("background: #238636; color: white; border-radius: 4px; padding: 5px 12px; font-weight: bold;")
        add_btn.clicked.connect(self._add_device)
        top.addWidget(add_btn)

        refresh_btn = QPushButton("Alle aktualisieren")
        refresh_btn.setStyleSheet("background: #1F6FEB; color: white; border-radius: 4px; padding: 5px 12px;")
        refresh_btn.clicked.connect(self._poll_all)
        top.addWidget(refresh_btn)

        self._auto_cb = QCheckBox("Auto-Refresh (30 s)")
        self._auto_cb.toggled.connect(self._on_auto_toggled)
        top.addWidget(self._auto_cb)
        top.addStretch()
        root.addLayout(top)

        # Table
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Name", "URL", "Status", "Score", "Letzter Alarm", "Aktionen"])
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

    def _rebuild_table(self) -> None:
        self.table.setRowCount(len(self._devices))
        for row, dev in enumerate(self._devices):
            self.table.setItem(row, self._COL_NAME, QTableWidgetItem(dev.get("name", "")))
            self.table.setItem(row, self._COL_URL, QTableWidgetItem(dev.get("url", "")))
            status_item = QTableWidgetItem("⟳ Unbekannt")
            status_item.setForeground(QColor("#8B949E"))
            self.table.setItem(row, self._COL_STATUS, status_item)
            self.table.setItem(row, self._COL_SCORE, QTableWidgetItem("–"))
            self.table.setItem(row, self._COL_ALARM, QTableWidgetItem("–"))

            # Action buttons widget
            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(2, 2, 2, 2)
            btn_layout.setSpacing(4)
            dash_btn = QPushButton("Dashboard")
            dash_btn.setFixedHeight(22)
            dash_btn.setStyleSheet("background: #1F6FEB; color: white; border-radius: 3px; font-size: 10px; padding: 0 6px;")
            url = dev.get("url", "")
            dash_btn.clicked.connect(lambda _, u=url: self._open_dashboard(u))
            btn_layout.addWidget(dash_btn)
            del_btn = QPushButton("Entfernen")
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
        # Set all to "Prüfe..."
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self._COL_STATUS)
            if item:
                item.setText("⟳ Prüfe…")
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
                if "error" in data:
                    status_item.setText(f"● Offline ({data['error']})")
                    status_item.setForeground(QColor("#E74C3C"))
                else:
                    status_item.setText("● Online")
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
