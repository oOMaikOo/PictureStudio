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
from core.industrial_notifier import IndustrialNotifier


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
    industrial_config_changed = Signal(dict)
    api_key_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = None
        self._api_server = None
        self._notifier: AlarmNotifier | None = None
        self._industrial_notifier: IndustrialNotifier | None = None
        self._build_ui()

    def set_settings(self, settings) -> None:
        """Accept the ``AppSettings`` instance and populate all form fields."""
        self._settings = settings
        self._load_values()

    def set_notifier(self, n: AlarmNotifier) -> None:
        """Inject the ``AlarmNotifier`` instance used by test buttons."""
        self._notifier = n

    def set_industrial_notifier(self, n: IndustrialNotifier) -> None:
        """Inject the ``IndustrialNotifier`` instance for OPC-UA / Modbus."""
        self._industrial_notifier = n

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

        from utils.i18n import tr
        title = QLabel(tr("settings.title"))
        title.setStyleSheet("font-size:20px;font-weight:bold;color:#3498DB;")
        layout.addWidget(title)

        # Language
        lang_group = QGroupBox(tr("settings.lang.group"))
        lf = QFormLayout(lang_group)
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("Deutsch", "de")
        self.lang_combo.addItem("English", "en")
        lf.addRow(tr("settings.lang.label"), self.lang_combo)
        lang_hint = QLabel(tr("settings.lang.hint"))
        lang_hint.setStyleSheet("color:#888; font-size:10px;")
        lf.addRow("", lang_hint)
        layout.addWidget(lang_group)

        # Appearance
        app_group = QGroupBox(tr("settings.appearance_group"))
        af = QFormLayout(app_group)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light"])
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        af.addRow(tr("settings.theme_label"), self.theme_combo)

        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(7, 16)
        self.font_size_spin.setValue(9)
        af.addRow(tr("settings.font_size_label"), self.font_size_spin)
        layout.addWidget(app_group)

        # Project
        proj_group = QGroupBox(tr("settings.project_group"))
        pf = QFormLayout(proj_group)

        self.autosave_cb = QCheckBox(tr("settings.autosave_cb"))
        self.autosave_cb.setChecked(True)
        pf.addRow("", self.autosave_cb)

        self.autosave_spin = QSpinBox()
        self.autosave_spin.setRange(30, 3600)
        self.autosave_spin.setValue(300)
        self.autosave_spin.setSuffix(" s")
        pf.addRow(tr("settings.autosave_interval"), self.autosave_spin)

        self.backup_cb = QCheckBox(tr("settings.backup_cb"))
        self.backup_cb.setChecked(True)
        pf.addRow("", self.backup_cb)
        layout.addWidget(proj_group)

        # Labeling
        lbl_group = QGroupBox(tr("settings.labeling_group"))
        lf = QFormLayout(lbl_group)

        self.thumb_size_spin = QSpinBox()
        self.thumb_size_spin.setRange(60, 240)
        self.thumb_size_spin.setValue(100)
        self.thumb_size_spin.setSuffix(" px")
        lf.addRow(tr("settings.thumb_size_label"), self.thumb_size_spin)

        self.show_roi_labels_cb = QCheckBox(tr("settings.roi_labels_cb"))
        self.show_roi_labels_cb.setChecked(True)
        lf.addRow("", self.show_roi_labels_cb)
        layout.addWidget(lbl_group)

        # Inference
        inf_group = QGroupBox(tr("settings.inference_group"))
        inf_f = QFormLayout(inf_group)

        self.low_conf_spin = QDoubleSpinBox()
        self.low_conf_spin.setRange(0.0, 1.0)
        self.low_conf_spin.setValue(0.70)
        self.low_conf_spin.setSingleStep(0.05)
        inf_f.addRow(tr("settings.low_conf_label"), self.low_conf_spin)

        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 5)
        self.top_k_spin.setValue(3)
        inf_f.addRow(tr("settings.top_k_label"), self.top_k_spin)
        layout.addWidget(inf_group)

        # REST API
        api_group = QGroupBox(tr("settings.api_group"))
        ag = QVBoxLayout(api_group)

        api_form = QFormLayout()
        self.api_port_spin = QSpinBox()
        self.api_port_spin.setRange(1024, 65535)
        self.api_port_spin.setValue(8765)
        api_form.addRow(tr("settings.api_port_label"), self.api_port_spin)

        # API Key
        key_row = QHBoxLayout()
        self._api_key_edit = QLineEdit()
        self._api_key_edit.setReadOnly(True)
        self._api_key_edit.setPlaceholderText(tr("settings.api_key_placeholder"))
        self._api_key_edit.setEchoMode(QLineEdit.Password)
        key_row.addWidget(self._api_key_edit)
        gen_key_btn = QPushButton(tr("settings.api_generate_btn"))
        gen_key_btn.setToolTip("Neuen zufälligen API-Key erstellen")
        gen_key_btn.clicked.connect(self._generate_api_key)
        key_row.addWidget(gen_key_btn)
        show_key_btn = QPushButton(tr("settings.api_show_btn"))
        show_key_btn.setCheckable(True)
        show_key_btn.toggled.connect(
            lambda on: self._api_key_edit.setEchoMode(
                QLineEdit.Normal if on else QLineEdit.Password
            )
        )
        key_row.addWidget(show_key_btn)
        clear_key_btn = QPushButton(tr("settings.api_clear_btn"))
        clear_key_btn.setToolTip("API-Key entfernen — Authentifizierung deaktivieren")
        clear_key_btn.clicked.connect(self._clear_api_key)
        key_row.addWidget(clear_key_btn)
        api_form.addRow(tr("settings.api_key_label"), key_row)
        ag.addLayout(api_form)

        self.api_status_label = QLabel(tr("settings.api_status_stopped"))
        self.api_status_label.setStyleSheet("color:#E74C3C; font-size:10px;")
        self.api_status_label.setWordWrap(True)
        ag.addWidget(self.api_status_label)

        api_btn_row = QHBoxLayout()
        self.api_toggle_btn = QPushButton(tr("settings.api_start_btn"))
        self.api_toggle_btn.setStyleSheet(
            "background:#2ECC71;color:white;padding:5px 10px;font-weight:bold;border-radius:4px;"
        )
        self.api_toggle_btn.clicked.connect(self._toggle_api)
        api_btn_row.addWidget(self.api_toggle_btn)

        self.api_copy_btn = QPushButton(tr("settings.api_copy_url_btn"))
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
        mqtt_group = QGroupBox(tr("settings.mqtt_group"))
        mf = QFormLayout(mqtt_group)
        self.mqtt_enabled_cb = QCheckBox(tr("settings.mqtt_enabled_cb"))
        mf.addRow("", self.mqtt_enabled_cb)
        self.mqtt_host_edit = QLineEdit("localhost")
        mf.addRow(tr("settings.mqtt_broker_label"), self.mqtt_host_edit)
        self.mqtt_port_spin = QSpinBox()
        self.mqtt_port_spin.setRange(1, 65535)
        self.mqtt_port_spin.setValue(1883)
        mf.addRow(tr("settings.mqtt_port_label"), self.mqtt_port_spin)
        self.mqtt_topic_edit = QLineEdit("picture_studio/anomaly")
        mf.addRow(tr("settings.mqtt_topic_label"), self.mqtt_topic_edit)
        self.mqtt_user_edit = QLineEdit()
        self.mqtt_user_edit.setPlaceholderText("optional")
        mf.addRow(tr("settings.mqtt_user_label"), self.mqtt_user_edit)
        self.mqtt_pass_edit = QLineEdit()
        self.mqtt_pass_edit.setEchoMode(QLineEdit.Password)
        self.mqtt_pass_edit.setPlaceholderText("optional")
        mf.addRow(tr("settings.mqtt_pass_label"), self.mqtt_pass_edit)
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
        alarm_group = QGroupBox(tr("settings.alarm_group"))
        af2 = QFormLayout(alarm_group)

        # --- E-Mail section ---
        email_lbl = QLabel("<b>E-Mail</b>")
        af2.addRow(email_lbl)

        self._email_enabled_cb = QCheckBox(tr("settings.email_enabled_cb"))
        af2.addRow("", self._email_enabled_cb)

        self._smtp_host_edit = QLineEdit()
        self._smtp_host_edit.setPlaceholderText("smtp.gmail.com")
        af2.addRow(tr("settings.smtp_host_label"), self._smtp_host_edit)

        self._smtp_port_spin = QSpinBox()
        self._smtp_port_spin.setRange(1, 65535)
        self._smtp_port_spin.setValue(587)
        af2.addRow(tr("settings.smtp_port_label"), self._smtp_port_spin)

        self._smtp_user_edit = QLineEdit()
        af2.addRow(tr("settings.smtp_user_label"), self._smtp_user_edit)

        self._smtp_pass_edit = QLineEdit()
        self._smtp_pass_edit.setEchoMode(QLineEdit.Password)
        af2.addRow(tr("settings.smtp_pass_label"), self._smtp_pass_edit)

        self._smtp_tls_cb = QCheckBox(tr("settings.smtp_tls_cb"))
        self._smtp_tls_cb.setChecked(True)
        af2.addRow("", self._smtp_tls_cb)

        self._email_from_edit = QLineEdit()
        af2.addRow(tr("settings.email_from_label"), self._email_from_edit)

        self._email_to_edit = QLineEdit()
        self._email_to_edit.setPlaceholderText("emp1@domain.de, emp2@domain.de")
        af2.addRow(tr("settings.email_to_label"), self._email_to_edit)

        test_email_btn = QPushButton(tr("settings.test_email_btn"))
        test_email_btn.clicked.connect(self._test_email)
        af2.addRow("", test_email_btn)

        # --- Webhook section ---
        webhook_lbl = QLabel("<b>Webhook</b>")
        af2.addRow(webhook_lbl)

        self._webhook_enabled_cb = QCheckBox(tr("settings.webhook_enabled_cb"))
        af2.addRow("", self._webhook_enabled_cb)

        self._webhook_url_edit = QLineEdit()
        self._webhook_url_edit.setPlaceholderText("https://hooks.example.com/...")
        af2.addRow(tr("settings.webhook_url_label"), self._webhook_url_edit)

        test_webhook_btn = QPushButton(tr("settings.test_webhook_btn"))
        test_webhook_btn.clicked.connect(self._test_webhook)
        af2.addRow("", test_webhook_btn)

        # --- Cooldown ---
        cooldown_lbl = QLabel("<b>Allgemein</b>")
        af2.addRow(cooldown_lbl)

        self._notify_cooldown_spin = QSpinBox()
        self._notify_cooldown_spin.setRange(10, 3600)
        self._notify_cooldown_spin.setValue(60)
        self._notify_cooldown_spin.setSuffix(" s")
        af2.addRow(tr("settings.cooldown_label"), self._notify_cooldown_spin)

        layout.addWidget(alarm_group)

        # Industrieanbindung (OPC-UA & Modbus)
        industrial_group = QGroupBox(tr("settings.industrial_group"))
        ind_f = QFormLayout(industrial_group)

        # OPC-UA section
        opcua_lbl = QLabel("<b>OPC-UA</b>")
        ind_f.addRow(opcua_lbl)

        self._opcua_enabled_cb = QCheckBox(tr("settings.opcua_enabled_cb"))
        ind_f.addRow("", self._opcua_enabled_cb)

        self._opcua_url_edit = QLineEdit()
        self._opcua_url_edit.setPlaceholderText("opc.tcp://192.168.1.10:4840")
        ind_f.addRow(tr("settings.opcua_url_label"), self._opcua_url_edit)

        self._opcua_node_edit = QLineEdit()
        self._opcua_node_edit.setPlaceholderText("ns=2;i=1001")
        ind_f.addRow(tr("settings.opcua_node_label"), self._opcua_node_edit)

        self._test_opcua_btn = QPushButton(tr("settings.opcua_test_btn"))
        self._test_opcua_btn.clicked.connect(self._test_opcua_connection)
        ind_f.addRow("", self._test_opcua_btn)

        # Modbus TCP section
        modbus_lbl = QLabel("<b>Modbus TCP</b>")
        ind_f.addRow(modbus_lbl)

        self._modbus_enabled_cb = QCheckBox(tr("settings.modbus_enabled_cb"))
        ind_f.addRow("", self._modbus_enabled_cb)

        self._modbus_host_edit = QLineEdit()
        self._modbus_host_edit.setPlaceholderText("192.168.1.20")
        ind_f.addRow(tr("settings.modbus_host_label"), self._modbus_host_edit)

        self._modbus_port_spin = QSpinBox()
        self._modbus_port_spin.setRange(1, 65535)
        self._modbus_port_spin.setValue(502)
        ind_f.addRow(tr("settings.modbus_port_label"), self._modbus_port_spin)

        self._modbus_coil_spin = QSpinBox()
        self._modbus_coil_spin.setRange(0, 65535)
        self._modbus_coil_spin.setValue(0)
        ind_f.addRow(tr("settings.modbus_coil_label"), self._modbus_coil_spin)

        self._modbus_unit_spin = QSpinBox()
        self._modbus_unit_spin.setRange(1, 247)
        self._modbus_unit_spin.setValue(1)
        ind_f.addRow(tr("settings.modbus_unit_label"), self._modbus_unit_spin)

        self._test_modbus_btn = QPushButton(tr("settings.modbus_test_btn"))
        self._test_modbus_btn.clicked.connect(self._test_modbus_connection)
        ind_f.addRow("", self._test_modbus_btn)

        layout.addWidget(industrial_group)

        # Connect live-save signals for industrial widgets
        self._opcua_enabled_cb.toggled.connect(self._save_industrial_settings)
        self._opcua_url_edit.editingFinished.connect(self._save_industrial_settings)
        self._opcua_node_edit.editingFinished.connect(self._save_industrial_settings)
        self._modbus_enabled_cb.toggled.connect(self._save_industrial_settings)
        self._modbus_host_edit.editingFinished.connect(self._save_industrial_settings)
        self._modbus_port_spin.valueChanged.connect(self._save_industrial_settings)
        self._modbus_coil_spin.valueChanged.connect(self._save_industrial_settings)
        self._modbus_unit_spin.valueChanged.connect(self._save_industrial_settings)

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
        ssh_group = QGroupBox(tr("settings.ssh_group"))
        ssh_v = QVBoxLayout(ssh_group)
        self.ssh_list = QListWidget()
        ssh_v.addWidget(self.ssh_list)
        ssh_btn_row = QHBoxLayout()
        add_ssh = QPushButton(tr("settings.ssh_add_btn"))
        add_ssh.clicked.connect(self._add_ssh_profile)
        del_ssh = QPushButton(tr("settings.ssh_del_btn"))
        del_ssh.clicked.connect(self._del_ssh_profile)
        ssh_btn_row.addWidget(add_ssh)
        ssh_btn_row.addWidget(del_ssh)
        ssh_v.addLayout(ssh_btn_row)
        layout.addWidget(ssh_group)

        # Save
        from utils.i18n import tr as _tr
        save_btn = QPushButton(_tr("settings.save_btn"))
        save_btn.setStyleSheet("background:#2ECC71;color:white;padding:8px;font-weight:bold;")
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn)

        layout.addStretch()

    def _load_values(self) -> None:
        """Populate all form widgets from the current ``AppSettings`` values."""
        if not self._settings:
            return
        # Language
        saved_lang = self._settings.get_language()
        idx = self.lang_combo.findData(saved_lang)
        if idx >= 0:
            self.lang_combo.setCurrentIndex(idx)
        self.theme_combo.setCurrentText(self._settings.get_theme())
        self.font_size_spin.setValue(self._settings.get_font_size())
        self.autosave_cb.setChecked(self._settings.get_autosave_enabled())
        self.autosave_spin.setValue(self._settings.get_autosave_interval())
        self.backup_cb.setChecked(self._settings.get_backup_enabled())
        self.thumb_size_spin.setValue(self._settings.get_thumbnail_size())
        self.show_roi_labels_cb.setChecked(self._settings.get_show_roi_labels())
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
        from utils.i18n import tr
        if not HAS_MQTT:
            self.mqtt_status_lbl.setText(tr("settings.mqtt_not_installed"))
            self.mqtt_status_lbl.setStyleSheet("color:#F85149;font-size:10px;")
        self._load_alarm_notifier_settings()
        self._load_industrial_settings()
        self._refresh_ssh_list()
        # API key
        if hasattr(self, "_api_key_edit"):
            self._api_key_edit.setText(self._settings.get_api_key())

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

        webhook_url = self._webhook_url_edit.text().strip()
        email_to = self._email_to_edit.text().strip()

        # Inline validation: red border when format is wrong
        _invalid = "border: 1px solid #F85149;"
        _ok = ""
        url_ok = not webhook_url or webhook_url.startswith(("http://", "https://"))
        self._webhook_url_edit.setStyleSheet(_ok if url_ok else _invalid)

        emails_ok = True
        if email_to:
            emails_ok = all("@" in e.strip() for e in email_to.split(",") if e.strip())
        self._email_to_edit.setStyleSheet(_ok if emails_ok else _invalid)

        cfg = {
            "email_enabled":  self._email_enabled_cb.isChecked(),
            "smtp_host":      self._smtp_host_edit.text().strip(),
            "smtp_port":      self._smtp_port_spin.value(),
            "smtp_user":      self._smtp_user_edit.text().strip(),
            "smtp_pass":      self._smtp_pass_edit.text(),
            "smtp_tls":       self._smtp_tls_cb.isChecked(),
            "email_from":     self._email_from_edit.text().strip(),
            "email_to":       email_to,
            "webhook_enabled": self._webhook_enabled_cb.isChecked(),
            "webhook_url":    webhook_url,
            "cooldown":       self._notify_cooldown_spin.value(),
        }
        self._settings.save_alarm_notifier_config(cfg)
        if self._notifier:
            self._notifier.update_config(cfg)
        self.alarm_notifier_config_changed.emit(cfg)

    def _load_industrial_settings(self) -> None:
        """Populate industrial notifier UI fields from AppSettings."""
        if not self._settings:
            return
        cfg = self._settings.get_industrial_config()
        opcua = cfg.get("opcua", {})
        modbus = cfg.get("modbus", {})
        for w in [self._opcua_enabled_cb, self._modbus_enabled_cb]:
            w.blockSignals(True)
        self._opcua_enabled_cb.setChecked(bool(opcua.get("enabled", False)))
        self._opcua_url_edit.setText(opcua.get("url", ""))
        self._opcua_node_edit.setText(opcua.get("node_id", ""))
        self._modbus_enabled_cb.setChecked(bool(modbus.get("enabled", False)))
        self._modbus_host_edit.setText(modbus.get("host", ""))
        self._modbus_port_spin.setValue(int(modbus.get("port", 502)))
        self._modbus_coil_spin.setValue(int(modbus.get("coil_addr", 0)))
        self._modbus_unit_spin.setValue(int(modbus.get("unit_id", 1)))
        for w in [self._opcua_enabled_cb, self._modbus_enabled_cb]:
            w.blockSignals(False)

    def _save_industrial_settings(self) -> None:
        """Read industrial UI fields, persist via AppSettings, and emit signal."""
        if not self._settings:
            return
        cfg = {
            "opcua": {
                "enabled":   self._opcua_enabled_cb.isChecked(),
                "url":       self._opcua_url_edit.text().strip(),
                "node_id":   self._opcua_node_edit.text().strip(),
            },
            "modbus": {
                "enabled":   self._modbus_enabled_cb.isChecked(),
                "host":      self._modbus_host_edit.text().strip(),
                "port":      self._modbus_port_spin.value(),
                "coil_addr": self._modbus_coil_spin.value(),
                "unit_id":   self._modbus_unit_spin.value(),
            },
        }
        self._settings.save_industrial_config(cfg)
        if self._industrial_notifier:
            self._industrial_notifier.update_config(cfg)
        self.industrial_config_changed.emit(cfg)

    def _test_opcua_connection(self) -> None:
        """Test the OPC-UA connection and show a QMessageBox with the result."""
        from utils.i18n import tr
        if not self._industrial_notifier:
            QMessageBox.warning(self, tr("common.warning"), "Industrial Notifier nicht initialisiert.")
            return
        self._save_industrial_settings()
        success, msg = self._industrial_notifier.test_opcua()
        if success:
            QMessageBox.information(self, "OPC-UA Test", tr("settings.test_success"))
        else:
            QMessageBox.critical(self, tr("common.error"), msg)

    def _test_modbus_connection(self) -> None:
        """Test the Modbus TCP connection and show a QMessageBox with the result."""
        from utils.i18n import tr
        if not self._industrial_notifier:
            QMessageBox.warning(self, tr("common.warning"), "Industrial Notifier nicht initialisiert.")
            return
        self._save_industrial_settings()
        success, msg = self._industrial_notifier.test_modbus()
        if success:
            QMessageBox.information(self, "Modbus Test", tr("settings.test_success"))
        else:
            QMessageBox.critical(self, tr("common.error"), msg)

    def _test_email(self) -> None:
        """Send a test e-mail via the notifier and show the result."""
        from utils.i18n import tr
        if not self._notifier:
            QMessageBox.warning(self, tr("common.warning"), "Notifier nicht initialisiert.")
            return
        self._save_alarm_notifier_settings()
        success, msg = self._notifier.test_email()
        if success:
            QMessageBox.information(self, tr("common.info"), tr("settings.email_sent"))
        else:
            QMessageBox.critical(self, tr("common.error"), msg)

    def _test_webhook(self) -> None:
        """Send a test webhook call via the notifier and show the result."""
        from utils.i18n import tr
        if not self._notifier:
            QMessageBox.warning(self, tr("common.warning"), "Notifier nicht initialisiert.")
            return
        self._save_alarm_notifier_settings()
        success, msg = self._notifier.test_webhook()
        if success:
            QMessageBox.information(self, tr("common.info"), tr("settings.webhook_sent"))
        else:
            QMessageBox.critical(self, tr("common.error"), msg)

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
        from utils.i18n import tr
        if not self._settings:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("settings.ssh_dlg_title"))
        dlg.setMinimumWidth(360)
        form = QFormLayout(dlg)
        name_edit = QLineEdit()
        host_edit = QLineEdit()
        user_edit = QLineEdit()
        key_edit  = QLineEdit()
        form.addRow(tr("settings.ssh_name_label"), name_edit)
        form.addRow(tr("settings.ssh_host_label"), host_edit)
        form.addRow(tr("settings.ssh_user_label"), user_edit)
        form.addRow(tr("settings.ssh_key_label"), key_edit)
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
        from utils.i18n import tr
        self._api_server = server
        if server:
            server.set_status_callback(self._on_api_status)
            # Sync button states in case the API was already running
            if server.is_running:
                self.api_toggle_btn.setText(tr("settings.api_stop_btn"))
                self.api_toggle_btn.setStyleSheet(
                    "background:#E74C3C;color:white;padding:5px 10px;"
                    "font-weight:bold;border-radius:4px;"
                )
                self.api_copy_btn.setEnabled(True)
                self.api_dashboard_btn.setEnabled(True)
                self.api_port_spin.setEnabled(False)

    def _toggle_api(self) -> None:
        """Start or stop the REST API server and update button appearance accordingly."""
        from utils.i18n import tr
        if not self._api_server:
            return
        if self._api_server.is_running:
            self._api_server.stop()
            self.api_toggle_btn.setText(tr("settings.api_start_btn"))
            self.api_toggle_btn.setStyleSheet(
                "background:#2ECC71;color:white;padding:5px 10px;"
                "font-weight:bold;border-radius:4px;"
            )
            self.api_copy_btn.setEnabled(False)
            self.api_dashboard_btn.setEnabled(False)
            self.api_port_spin.setEnabled(True)
        else:
            port = self.api_port_spin.value()
            import socket as _socket
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
                if _s.connect_ex(("127.0.0.1", port)) == 0:
                    from utils.i18n import tr as _tr
                    QMessageBox.warning(
                        self, _tr("common.warning"),
                        f"Port {port} ist bereits belegt. Bitte einen anderen Port wählen."
                    )
                    return
            ok = self._api_server.start(port)
            if ok:
                self.api_toggle_btn.setText(tr("settings.api_stop_btn"))
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

    def _generate_api_key(self) -> None:
        """Generate a random 32-byte hex API key, save it, and apply it immediately."""
        import secrets
        key = secrets.token_hex(32)
        self._api_key_edit.setText(key)
        self._apply_api_key(key)

    def _clear_api_key(self) -> None:
        """Remove the API key, disabling authentication."""
        self._api_key_edit.clear()
        self._apply_api_key("")

    def _apply_api_key(self, key: str) -> None:
        """Persist and broadcast the new API key immediately."""
        if self._settings:
            self._settings.save_api_key(key)
        if self._api_server:
            self._api_server.set_api_key(key)
        self.api_key_changed.emit(key)

    def _save(self) -> None:
        """Persist all settings, emit ``autosave_changed``, and show a confirmation dialog."""
        if not self._settings:
            return
        self._settings.set_language(self.lang_combo.currentData())
        self._settings.set_theme(self.theme_combo.currentText())
        self._settings.set_font_size(self.font_size_spin.value())
        self._settings.set_autosave_enabled(self.autosave_cb.isChecked())
        self._settings.set_autosave_interval(self.autosave_spin.value())
        self._settings.set_backup_enabled(self.backup_cb.isChecked())
        self._settings.set_thumbnail_size(self.thumb_size_spin.value())
        self._settings.set_show_roi_labels(self.show_roi_labels_cb.isChecked())
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
        from utils.i18n import tr
        QMessageBox.information(self, tr("settings.saved.title"), tr("settings.saved.msg"))
