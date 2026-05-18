"""
Settings page: theme, autosave, inference defaults, SSH profiles.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QPushButton,
    QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QListWidget, QListWidgetItem, QHBoxLayout, QMessageBox,
    QLineEdit, QApplication, QScrollArea, QFrame,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPalette, QColor

from core.alarm_notifier import AlarmNotifier


class SettingsPage(QWidget):
    """
    Application settings page (stack index 7).

    Groups of settings:
    - Appearance: dark/light theme toggle and font size.
    - Project & Autosave: interval and backup-before-save.
    - Labeling: thumbnail size and ROI-label display.
    - Inference: low-confidence threshold and default Top-K.
    - REST API: start/stop, port, URL copy, live dashboard link.
    - MQTT: broker connection details for anomaly alarm publishing.
    - SSH profiles: add/remove profiles used by remote training.

    Signals
    -------
    theme_changed    : Emitted immediately when the theme combo changes.
    autosave_changed : Emitted when "Einstellungen speichern" is clicked;
                       carries (interval_seconds, enabled).
    """

    theme_changed = Signal(str)
    autosave_changed = Signal(int, bool)
    alarm_notifier_config_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = None
        self._api_server = None
        self._notifier: AlarmNotifier | None = None
        self._build_ui()

    def set_settings(self, settings) -> None:
        """Accept the ``AppSettings`` instance and populate all form fields."""
        self._settings = settings
        self._load_values()

    def set_notifier(self, n: AlarmNotifier) -> None:
        """Inject the ``AlarmNotifier`` instance used by test buttons."""
        self._notifier = n

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        layout = QVBoxLayout(container)
        scroll.setWidget(container)
        outer.addWidget(scroll)

        title = QLabel("Einstellungen")
        title.setStyleSheet("font-size:20px;font-weight:bold;color:#3498DB;")
        layout.addWidget(title)

        # Appearance
        app_group = QGroupBox("Erscheinungsbild")
        af = QFormLayout(app_group)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light"])
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        af.addRow("Design:", self.theme_combo)

        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(7, 16)
        self.font_size_spin.setValue(9)
        af.addRow("Schriftgröße:", self.font_size_spin)
        layout.addWidget(app_group)

        # Project
        proj_group = QGroupBox("Projekt & Autosave")
        pf = QFormLayout(proj_group)

        self.autosave_cb = QCheckBox("Autosave aktiviert")
        self.autosave_cb.setChecked(True)
        pf.addRow("", self.autosave_cb)

        self.autosave_spin = QSpinBox()
        self.autosave_spin.setRange(30, 3600)
        self.autosave_spin.setValue(300)
        self.autosave_spin.setSuffix(" s")
        pf.addRow("Autosave-Intervall:", self.autosave_spin)

        self.backup_cb = QCheckBox("Backup vor Speichern")
        self.backup_cb.setChecked(True)
        pf.addRow("", self.backup_cb)
        layout.addWidget(proj_group)

        # Labeling
        lbl_group = QGroupBox("Labeling")
        lf = QFormLayout(lbl_group)

        self.thumb_size_spin = QSpinBox()
        self.thumb_size_spin.setRange(60, 240)
        self.thumb_size_spin.setValue(100)
        self.thumb_size_spin.setSuffix(" px")
        lf.addRow("Thumbnail-Größe:", self.thumb_size_spin)

        self.show_roi_labels_cb = QCheckBox("ROI-Labels im Editor anzeigen")
        self.show_roi_labels_cb.setChecked(True)
        lf.addRow("", self.show_roi_labels_cb)
        layout.addWidget(lbl_group)

        # Inference
        inf_group = QGroupBox("Inferenz")
        inf_f = QFormLayout(inf_group)

        self.low_conf_spin = QDoubleSpinBox()
        self.low_conf_spin.setRange(0.0, 1.0)
        self.low_conf_spin.setValue(0.70)
        self.low_conf_spin.setSingleStep(0.05)
        inf_f.addRow("Schwelle 'unsicher':", self.low_conf_spin)

        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 5)
        self.top_k_spin.setValue(3)
        inf_f.addRow("Standard Top-K:", self.top_k_spin)
        layout.addWidget(inf_group)

        # REST API
        api_group = QGroupBox("REST-API Server")
        ag = QVBoxLayout(api_group)

        api_form = QFormLayout()
        self.api_port_spin = QSpinBox()
        self.api_port_spin.setRange(1024, 65535)
        self.api_port_spin.setValue(8765)
        api_form.addRow("Port:", self.api_port_spin)
        ag.addLayout(api_form)

        self.api_status_label = QLabel("Gestoppt")
        self.api_status_label.setStyleSheet("color:#E74C3C; font-size:10px;")
        self.api_status_label.setWordWrap(True)
        ag.addWidget(self.api_status_label)

        api_btn_row = QHBoxLayout()
        self.api_toggle_btn = QPushButton("API starten")
        self.api_toggle_btn.setStyleSheet(
            "background:#2ECC71;color:white;padding:5px 10px;font-weight:bold;border-radius:4px;"
        )
        self.api_toggle_btn.clicked.connect(self._toggle_api)
        api_btn_row.addWidget(self.api_toggle_btn)

        self.api_copy_btn = QPushButton("URL kopieren")
        self.api_copy_btn.setEnabled(False)
        self.api_copy_btn.setToolTip("Basis-URL in Zwischenablage kopieren")
        self.api_copy_btn.clicked.connect(self._copy_api_url)
        api_btn_row.addWidget(self.api_copy_btn)
        self.api_dashboard_btn = QPushButton("📊 Dashboard")
        self.api_dashboard_btn.setEnabled(False)
        self.api_dashboard_btn.setToolTip("Live-Monitoring-Dashboard im Browser öffnen")
        self.api_dashboard_btn.clicked.connect(self._open_dashboard)
        api_btn_row.addWidget(self.api_dashboard_btn)
        ag.addLayout(api_btn_row)

        api_hint = QLabel(
            "<small>"
            "<b>Endpunkte (alle GET außer label):</b><br>"
            "GET &nbsp;/api/status &nbsp;&nbsp;&nbsp; Server-Status<br>"
            "GET &nbsp;/api/project &nbsp;&nbsp; Projektübersicht<br>"
            "GET &nbsp;/api/labels &nbsp;&nbsp;&nbsp; Label-Definitionen<br>"
            "GET &nbsp;/api/images &nbsp;&nbsp;&nbsp; Alle Bilder mit Labels<br>"
            "GET &nbsp;/api/images/&lt;name&gt; &nbsp;Einzelbild + ROIs<br>"
            "POST /api/images/label &nbsp;&nbsp;Label zuweisen<br>"
            "&nbsp;&nbsp;&nbsp;Body: {\"path\":\"...\",\"label\":\"...\"}<br>"
            "POST /api/images/multilabel &nbsp;Multi-Label<br>"
            "&nbsp;&nbsp;&nbsp;Body: {\"path\":\"...\",\"labels\":[...]}"
            "</small>"
        )
        api_hint.setStyleSheet("color:#666; padding:4px;")
        api_hint.setWordWrap(True)
        ag.addWidget(api_hint)
        layout.addWidget(api_group)

        # MQTT
        mqtt_group = QGroupBox("MQTT-Alarm (Anomalie-Erkennung)")
        mf = QFormLayout(mqtt_group)
        self.mqtt_enabled_cb = QCheckBox("MQTT-Publishing aktiviert")
        mf.addRow("", self.mqtt_enabled_cb)
        self.mqtt_host_edit = QLineEdit("localhost")
        mf.addRow("Broker-Host:", self.mqtt_host_edit)
        self.mqtt_port_spin = QSpinBox()
        self.mqtt_port_spin.setRange(1, 65535)
        self.mqtt_port_spin.setValue(1883)
        mf.addRow("Port:", self.mqtt_port_spin)
        self.mqtt_topic_edit = QLineEdit("picture_studio/anomaly")
        mf.addRow("Topic:", self.mqtt_topic_edit)
        self.mqtt_user_edit = QLineEdit()
        self.mqtt_user_edit.setPlaceholderText("optional")
        mf.addRow("Benutzername:", self.mqtt_user_edit)
        self.mqtt_pass_edit = QLineEdit()
        self.mqtt_pass_edit.setEchoMode(QLineEdit.Password)
        self.mqtt_pass_edit.setPlaceholderText("optional")
        mf.addRow("Passwort:", self.mqtt_pass_edit)
        self.mqtt_status_lbl = QLabel("Nicht verbunden")
        self.mqtt_status_lbl.setStyleSheet("color:#7F8C8D;font-size:10px;")
        mf.addRow("Status:", self.mqtt_status_lbl)
        mqtt_hint = QLabel(
            "<small>Publiziert JSON-Events bei jedem Anomalie-Alarm.<br>"
            "paho-mqtt muss installiert sein: <tt>pip install paho-mqtt</tt></small>"
        )
        mqtt_hint.setStyleSheet("color:#7F8C8D;")
        mqtt_hint.setWordWrap(True)
        mf.addRow(mqtt_hint)
        layout.addWidget(mqtt_group)

        # Alarm notifier (E-Mail & Webhook)
        alarm_group = QGroupBox("Alarmierung (E-Mail & Webhook)")
        af2 = QFormLayout(alarm_group)

        # --- E-Mail section ---
        email_lbl = QLabel("<b>E-Mail</b>")
        af2.addRow(email_lbl)

        self._email_enabled_cb = QCheckBox("E-Mail-Benachrichtigung aktivieren")
        af2.addRow("", self._email_enabled_cb)

        self._smtp_host_edit = QLineEdit()
        self._smtp_host_edit.setPlaceholderText("smtp.gmail.com")
        af2.addRow("SMTP-Host:", self._smtp_host_edit)

        self._smtp_port_spin = QSpinBox()
        self._smtp_port_spin.setRange(1, 65535)
        self._smtp_port_spin.setValue(587)
        af2.addRow("SMTP-Port:", self._smtp_port_spin)

        self._smtp_user_edit = QLineEdit()
        af2.addRow("Benutzername:", self._smtp_user_edit)

        self._smtp_pass_edit = QLineEdit()
        self._smtp_pass_edit.setEchoMode(QLineEdit.Password)
        af2.addRow("Passwort:", self._smtp_pass_edit)

        self._smtp_tls_cb = QCheckBox("TLS verwenden")
        self._smtp_tls_cb.setChecked(True)
        af2.addRow("", self._smtp_tls_cb)

        self._email_from_edit = QLineEdit()
        af2.addRow("Absender:", self._email_from_edit)

        self._email_to_edit = QLineEdit()
        self._email_to_edit.setPlaceholderText("emp1@domain.de, emp2@domain.de")
        af2.addRow("Empfänger:", self._email_to_edit)

        test_email_btn = QPushButton("Test-E-Mail senden")
        test_email_btn.clicked.connect(self._test_email)
        af2.addRow("", test_email_btn)

        # --- Webhook section ---
        webhook_lbl = QLabel("<b>Webhook</b>")
        af2.addRow(webhook_lbl)

        self._webhook_enabled_cb = QCheckBox("Webhook aktivieren")
        af2.addRow("", self._webhook_enabled_cb)

        self._webhook_url_edit = QLineEdit()
        self._webhook_url_edit.setPlaceholderText("https://hooks.example.com/...")
        af2.addRow("URL:", self._webhook_url_edit)

        test_webhook_btn = QPushButton("Test-Webhook senden")
        test_webhook_btn.clicked.connect(self._test_webhook)
        af2.addRow("", test_webhook_btn)

        # --- Cooldown ---
        cooldown_lbl = QLabel("<b>Allgemein</b>")
        af2.addRow(cooldown_lbl)

        self._notify_cooldown_spin = QSpinBox()
        self._notify_cooldown_spin.setRange(10, 3600)
        self._notify_cooldown_spin.setValue(60)
        self._notify_cooldown_spin.setSuffix(" s")
        af2.addRow("Mindestabstand zwischen Alarmen:", self._notify_cooldown_spin)

        layout.addWidget(alarm_group)

        # Connect live-save signals for all new alarm widgets
        self._email_enabled_cb.toggled.connect(self._save_alarm_notifier_settings)
        self._smtp_host_edit.editingFinished.connect(self._save_alarm_notifier_settings)
        self._smtp_port_spin.valueChanged.connect(self._save_alarm_notifier_settings)
        self._smtp_user_edit.editingFinished.connect(self._save_alarm_notifier_settings)
        self._smtp_pass_edit.editingFinished.connect(self._save_alarm_notifier_settings)
        self._smtp_tls_cb.toggled.connect(self._save_alarm_notifier_settings)
        self._email_from_edit.editingFinished.connect(self._save_alarm_notifier_settings)
        self._email_to_edit.editingFinished.connect(self._save_alarm_notifier_settings)
        self._webhook_enabled_cb.toggled.connect(self._save_alarm_notifier_settings)
        self._webhook_url_edit.editingFinished.connect(self._save_alarm_notifier_settings)
        self._notify_cooldown_spin.valueChanged.connect(self._save_alarm_notifier_settings)

        # SSH profiles
        ssh_group = QGroupBox("SSH-Profile")
        ssh_v = QVBoxLayout(ssh_group)
        self.ssh_list = QListWidget()
        ssh_v.addWidget(self.ssh_list)
        ssh_btn_row = QHBoxLayout()
        add_ssh = QPushButton("Profil hinzufügen")
        add_ssh.clicked.connect(self._add_ssh_profile)
        del_ssh = QPushButton("Profil löschen")
        del_ssh.clicked.connect(self._del_ssh_profile)
        ssh_btn_row.addWidget(add_ssh)
        ssh_btn_row.addWidget(del_ssh)
        ssh_v.addLayout(ssh_btn_row)
        layout.addWidget(ssh_group)

        # Save
        save_btn = QPushButton("Einstellungen speichern")
        save_btn.setStyleSheet("background:#2ECC71;color:white;padding:8px;font-weight:bold;")
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn)

        layout.addStretch()

    def _load_values(self) -> None:
        """Populate all form widgets from the current ``AppSettings`` values."""
        if not self._settings:
            return
        self.theme_combo.setCurrentText(self._settings.get_theme())
        self.font_size_spin.setValue(self._settings.get_font_size())
        self.autosave_cb.setChecked(self._settings.get_autosave_enabled())
        self.autosave_spin.setValue(self._settings.get_autosave_interval())
        self.backup_cb.setChecked(self._settings.get_backup_enabled())
        self.thumb_size_spin.setValue(self._settings.get_thumbnail_size())
        self.low_conf_spin.setValue(self._settings.get_low_confidence_threshold())
        self.top_k_spin.setValue(self._settings.get_show_top_k())
        mqtt = self._settings.get_mqtt_config()
        self.mqtt_enabled_cb.setChecked(bool(mqtt.get("enabled", False)))
        self.mqtt_host_edit.setText(mqtt.get("host", "localhost"))
        self.mqtt_port_spin.setValue(int(mqtt.get("port", 1883)))
        self.mqtt_topic_edit.setText(mqtt.get("topic", "picture_studio/anomaly"))
        self.mqtt_user_edit.setText(mqtt.get("username", ""))
        self.mqtt_pass_edit.setText(mqtt.get("password", ""))
        from core.mqtt_client import HAS_MQTT
        if not HAS_MQTT:
            self.mqtt_status_lbl.setText("paho-mqtt nicht installiert")
            self.mqtt_status_lbl.setStyleSheet("color:#F85149;font-size:10px;")
        self._load_alarm_notifier_settings()
        self._refresh_ssh_list()

    def _load_alarm_notifier_settings(self) -> None:
        """Populate alarm notifier UI fields from AppSettings."""
        if not self._settings:
            return
        cfg = self._settings.get_alarm_notifier_config()
        # Block signals to avoid triggering auto-save during load
        for w in [self._email_enabled_cb, self._smtp_tls_cb, self._webhook_enabled_cb]:
            w.blockSignals(True)
        self._email_enabled_cb.setChecked(bool(cfg.get("email_enabled", False)))
        self._smtp_host_edit.setText(cfg.get("smtp_host", ""))
        self._smtp_port_spin.setValue(int(cfg.get("smtp_port", 587)))
        self._smtp_user_edit.setText(cfg.get("smtp_user", ""))
        self._smtp_pass_edit.setText(cfg.get("smtp_pass", ""))
        self._smtp_tls_cb.setChecked(bool(cfg.get("smtp_tls", True)))
        self._email_from_edit.setText(cfg.get("email_from", ""))
        self._email_to_edit.setText(cfg.get("email_to", ""))
        self._webhook_enabled_cb.setChecked(bool(cfg.get("webhook_enabled", False)))
        self._webhook_url_edit.setText(cfg.get("webhook_url", ""))
        self._notify_cooldown_spin.setValue(int(cfg.get("cooldown", 60)))
        for w in [self._email_enabled_cb, self._smtp_tls_cb, self._webhook_enabled_cb]:
            w.blockSignals(False)

    def _save_alarm_notifier_settings(self) -> None:
        """Read alarm notifier UI fields, persist via AppSettings, and emit signal."""
        if not self._settings:
            return
        cfg = {
            "email_enabled":  self._email_enabled_cb.isChecked(),
            "smtp_host":      self._smtp_host_edit.text().strip(),
            "smtp_port":      self._smtp_port_spin.value(),
            "smtp_user":      self._smtp_user_edit.text().strip(),
            "smtp_pass":      self._smtp_pass_edit.text(),
            "smtp_tls":       self._smtp_tls_cb.isChecked(),
            "email_from":     self._email_from_edit.text().strip(),
            "email_to":       self._email_to_edit.text().strip(),
            "webhook_enabled": self._webhook_enabled_cb.isChecked(),
            "webhook_url":    self._webhook_url_edit.text().strip(),
            "cooldown":       self._notify_cooldown_spin.value(),
        }
        self._settings.save_alarm_notifier_config(cfg)
        if self._notifier:
            self._notifier.update_config(cfg)
        self.alarm_notifier_config_changed.emit(cfg)

    def _test_email(self) -> None:
        """Send a test e-mail via the notifier and show the result."""
        if not self._notifier:
            QMessageBox.warning(self, "Kein Notifier", "Notifier nicht initialisiert.")
            return
        self._save_alarm_notifier_settings()
        success, msg = self._notifier.test_email()
        if success:
            QMessageBox.information(self, "Test-E-Mail", "Test-E-Mail erfolgreich gesendet.")
        else:
            QMessageBox.critical(self, "Test-E-Mail fehlgeschlagen", msg)

    def _test_webhook(self) -> None:
        """Send a test webhook call via the notifier and show the result."""
        if not self._notifier:
            QMessageBox.warning(self, "Kein Notifier", "Notifier nicht initialisiert.")
            return
        self._save_alarm_notifier_settings()
        success, msg = self._notifier.test_webhook()
        if success:
            QMessageBox.information(self, "Test-Webhook", "Test-Webhook erfolgreich gesendet.")
        else:
            QMessageBox.critical(self, "Test-Webhook fehlgeschlagen", msg)

    def _refresh_ssh_list(self) -> None:
        """Rebuild the SSH profile list widget from saved settings."""
        self.ssh_list.clear()
        if not self._settings:
            return
        for p in self._settings.get_ssh_profiles():
            self.ssh_list.addItem(f"{p.get('name', '?')}  —  {p.get('host', '?')}")

    def _on_theme_changed(self, theme: str) -> None:
        """Forward the selected theme to ``MainWindow`` via the ``theme_changed`` signal."""
        self.theme_changed.emit(theme)

    def _add_ssh_profile(self) -> None:
        """Open a dialog to collect SSH profile details and save the new entry."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        if not self._settings:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("SSH-Profil hinzufügen")
        dlg.setMinimumWidth(360)
        form = QFormLayout(dlg)
        name_edit = QLineEdit()
        host_edit = QLineEdit()
        user_edit = QLineEdit()
        key_edit  = QLineEdit()
        form.addRow("Profilname:", name_edit)
        form.addRow("Host:", host_edit)
        form.addRow("Benutzername:", user_edit)
        form.addRow("SSH-Key-Pfad:", key_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec():
            profiles = self._settings.get_ssh_profiles()
            profiles.append({
                "name": name_edit.text().strip(),
                "host": host_edit.text().strip(),
                "username": user_edit.text().strip(),
                "key_path": key_edit.text().strip(),
            })
            self._settings.save_ssh_profiles(profiles)
            self._refresh_ssh_list()

    def _del_ssh_profile(self) -> None:
        """Delete the currently selected SSH profile from settings."""
        if not self._settings:
            return
        row = self.ssh_list.currentRow()
        if row < 0:
            return
        profiles = self._settings.get_ssh_profiles()
        if row < len(profiles):
            del profiles[row]
            self._settings.save_ssh_profiles(profiles)
            self._refresh_ssh_list()

    # ------------------------------------------------------------------ REST API

    def set_api_server(self, server) -> None:
        """Inject the ``RestApiServer`` instance and sync the toggle button state."""
        self._api_server = server
        if server:
            server.set_status_callback(self._on_api_status)
            # Sync button states in case the API was already running
            if server.is_running:
                self.api_toggle_btn.setText("API stoppen")
                self.api_toggle_btn.setStyleSheet(
                    "background:#E74C3C;color:white;padding:5px 10px;"
                    "font-weight:bold;border-radius:4px;"
                )
                self.api_copy_btn.setEnabled(True)
                self.api_dashboard_btn.setEnabled(True)
                self.api_port_spin.setEnabled(False)

    def _toggle_api(self) -> None:
        """Start or stop the REST API server and update button appearance accordingly."""
        if not self._api_server:
            return
        if self._api_server.is_running:
            self._api_server.stop()
            self.api_toggle_btn.setText("API starten")
            self.api_toggle_btn.setStyleSheet(
                "background:#2ECC71;color:white;padding:5px 10px;"
                "font-weight:bold;border-radius:4px;"
            )
            self.api_copy_btn.setEnabled(False)
            self.api_dashboard_btn.setEnabled(False)
            self.api_port_spin.setEnabled(True)
        else:
            port = self.api_port_spin.value()
            ok = self._api_server.start(port)
            if ok:
                self.api_toggle_btn.setText("API stoppen")
                self.api_toggle_btn.setStyleSheet(
                    "background:#E74C3C;color:white;padding:5px 10px;"
                    "font-weight:bold;border-radius:4px;"
                )
                self.api_copy_btn.setEnabled(True)
                self.api_dashboard_btn.setEnabled(True)
                self.api_port_spin.setEnabled(False)

    def _on_api_status(self, msg: str) -> None:
        """Update the API status label text and colour (green = running, red = stopped)."""
        self.api_status_label.setText(msg)
        running = self._api_server and self._api_server.is_running
        self.api_status_label.setStyleSheet(
            "color:#2ECC71; font-size:10px;" if running else "color:#E74C3C; font-size:10px;"
        )

    def _copy_api_url(self) -> None:
        """Copy the running API base URL to the system clipboard."""
        if self._api_server and self._api_server.is_running:
            QApplication.clipboard().setText(self._api_server.url)

    def _open_dashboard(self) -> None:
        """Open the live-monitoring HTML dashboard in the default browser."""
        if self._api_server and self._api_server.is_running:
            import webbrowser
            port = self._api_server.port
            webbrowser.open(f"http://localhost:{port}/dashboard")

    def _save(self) -> None:
        """Persist all settings, emit ``autosave_changed``, and show a confirmation dialog."""
        if not self._settings:
            return
        self._settings.set_theme(self.theme_combo.currentText())
        self._settings.set_font_size(self.font_size_spin.value())
        self._settings.set_autosave_enabled(self.autosave_cb.isChecked())
        self._settings.set_autosave_interval(self.autosave_spin.value())
        self._settings.set_backup_enabled(self.backup_cb.isChecked())
        self._settings.set_thumbnail_size(self.thumb_size_spin.value())
        self._settings.set_low_confidence_threshold(self.low_conf_spin.value())
        self._settings.set_show_top_k(self.top_k_spin.value())
        self._settings.save_mqtt_config({
            "enabled":  self.mqtt_enabled_cb.isChecked(),
            "host":     self.mqtt_host_edit.text().strip(),
            "port":     self.mqtt_port_spin.value(),
            "topic":    self.mqtt_topic_edit.text().strip(),
            "username": self.mqtt_user_edit.text().strip(),
            "password": self.mqtt_pass_edit.text(),
        })
        self._settings.sync()
        self.autosave_changed.emit(self.autosave_spin.value(), self.autosave_cb.isChecked())
        QMessageBox.information(self, "Gespeichert", "Einstellungen wurden gespeichert.")
