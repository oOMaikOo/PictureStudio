"""
Main window: sidebar navigation + stacked pages + autosave + theming.
"""
import os
from datetime import datetime
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QStackedWidget,
    QStatusBar, QFileDialog, QMessageBox,
    QApplication,
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QAction, QFont, QKeySequence, QShortcut

from utils.config import APP_NAME, APP_VERSION
from utils.logging_utils import get_logger
from utils.settings import AppSettings
from gui.sidebar import Sidebar

log = get_logger()


class MainWindow(QMainWindow):
    """
    Top-level application window.

    Owns a ``Sidebar`` (navigation) and a ``QStackedWidget`` (10 pages).
    Responsibilities:
    - Project lifecycle: new / open / save / save-as / backup / crash-recovery.
    - Page orchestration: distributes the ``Project`` instance to every page
      after load via ``_load_project()``.
    - Autosave: configurable interval via ``AppSettings``; triggered by a
      ``QTimer``.
    - Menu bar with Datei / Projekt / Ansicht / Audit / Hilfe menus.
    - REST API server lifecycle (start/stop, project injection).
    - Global drag-and-drop fallback for images dropped anywhere in the window.
    - Active-Learning integration: relays ``al_queue_updated`` and
      ``labels_applied`` signals from ``InferencePage`` to ``LabelingPage``.
    """

    def __init__(self):
        super().__init__()
        self.project = None
        self.audit = None
        self._settings = AppSettings()

        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1280, 780)

        # Restore geometry
        geo = self._settings.get_window_geometry()
        if geo:
            self.restoreGeometry(geo)

        self._build_ui()
        self._build_menu()
        self._build_statusbar()
        self._apply_theme(self._settings.get_theme())
        self.dashboard_page.set_recent_projects(self._settings.get_recent_projects())
        self._check_crash_recovery()

        # Autosave timer
        self._autosave_timer = QTimer(self)
        self._autosave_timer.timeout.connect(self._autosave)
        self._reset_autosave_timer()

        # Periodic status update
        QTimer(self).timeout.connect(self._update_status)
        self.findChild(QTimer).start(8000) if self.findChild(QTimer) else None

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        """Construct the central widget: sidebar + stacked page area + REST server."""
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.sidebar = Sidebar()
        self.sidebar.set_locked(True)   # locked until a project is loaded
        self.sidebar.page_requested.connect(self._switch_page)
        self.sidebar.help_requested.connect(lambda: self._show_help())
        self.sidebar.tour_requested.connect(self._start_tour)
        root.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        root.addWidget(self.stack)

        # Guided tour (created after stack so it floats above everything)
        from gui.guide_tour import GuideTour
        self._tour = GuideTour(self)

        # Instantiate all pages
        from gui.pages.dashboard_page       import DashboardPage
        from gui.pages.data_page            import DataPage
        from gui.pages.camera_page          import CameraPage
        from gui.pages.labeling_page        import LabelingPage
        from gui.pages.training_page        import TrainingPage
        from gui.pages.models_page          import ModelsPage
        from gui.pages.inference_page       import InferencePage
        from gui.pages.export_page          import ExportPage
        from gui.pages.settings_page        import SettingsPage
        from gui.pages.batch_inference_page import BatchInferencePage

        self.dashboard_page    = DashboardPage()
        self.data_page         = DataPage()
        self.camera_page       = CameraPage()
        self.labeling_page     = LabelingPage()
        self.training_page     = TrainingPage()
        self.models_page       = ModelsPage()
        self.inference_page    = InferencePage()
        self.export_page       = ExportPage()
        self.settings_page     = SettingsPage()
        self.batch_page        = BatchInferencePage()

        for page in [
            self.dashboard_page, self.data_page, self.labeling_page,
            self.training_page, self.models_page, self.inference_page,
            self.export_page, self.settings_page, self.camera_page,
            self.batch_page,   # index 9
        ]:
            self.stack.addWidget(page)

        # REST API server
        from api.rest_server import RestApiServer
        self._rest_server = RestApiServer()

        # Cross-page signals
        self.dashboard_page.new_project_requested.connect(self._new_project)
        self.dashboard_page.open_project_requested.connect(self._open_project)
        self.dashboard_page.open_recent_requested.connect(self._open_recent)
        self.data_page.images_loaded.connect(self._on_images_loaded)
        self.training_page.training_finished.connect(self._on_training_finished)
        self.models_page.model_loaded.connect(self.inference_page.load_model_path)
        self.models_page.model_loaded.connect(self._on_model_loaded_api)
        self.settings_page.set_settings(self._settings)
        self.settings_page.set_api_server(self._rest_server)
        self.camera_page.set_rest_server(self._rest_server)
        self.training_page.set_settings(self._settings)
        self.settings_page.theme_changed.connect(self._apply_theme)
        self.settings_page.autosave_changed.connect(self._on_autosave_changed)

        # Active Learning: inference → labeling queue → retrain
        self.inference_page.al_queue_updated.connect(self._on_al_queue_updated)
        self.inference_page.labels_applied.connect(self._on_labels_applied)
        self.labeling_page.al_retrain_requested.connect(
            lambda: self._switch_page(3)  # 3 = Training page
        )

        # Global Drag & Drop fallback on the stacked-widget area
        from gui.widgets.drop_mixin import ImageDropFilter
        self._global_drop = ImageDropFilter(self.stack)
        self._global_drop.files_dropped.connect(self._on_global_drop)

    def _build_menu(self) -> None:
        """Create the top-level menu bar with all menus and actions."""
        mb = self.menuBar()

        # File
        fm = mb.addMenu("Datei")
        for label, shortcut, slot in [
            ("Neues Projekt",            "Ctrl+N", self._new_project),
            ("Projekt öffnen…",           "Ctrl+O", self._open_project),
            ("Projekt speichern",         "Ctrl+S", self._save_project),
            ("Projekt speichern unter…",  "Ctrl+Shift+S", self._save_project_as),
            ("Backup erstellen",          "",       self._create_backup),
        ]:
            a = QAction(label, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(slot)
            fm.addAction(a)
        cam_a = QAction("Kamera aufnehmen…", self)
        cam_a.setShortcut("Ctrl+K")
        cam_a.triggered.connect(self._open_camera_dialog)
        fm.addAction(cam_a)
        fm.addSeparator()
        for label, page_idx in [("Zuletzt geöffnet", None)]:
            pass  # submenu for recent projects added separately
        recent_menu = fm.addMenu("Zuletzt geöffnet")
        self._recent_menu = recent_menu
        self._refresh_recent_menu()
        fm.addSeparator()
        quit_a = QAction("Beenden", self)
        quit_a.setShortcut("Ctrl+Q")
        quit_a.triggered.connect(self.close)
        fm.addAction(quit_a)

        # Project
        pm = mb.addMenu("Projekt")
        labels_a = QAction("Labels verwalten…", self)
        labels_a.setShortcut("Ctrl+L")
        labels_a.triggered.connect(self._manage_labels)
        pm.addAction(labels_a)

        validate_a = QAction("Bilddateien prüfen", self)
        validate_a.triggered.connect(self._validate_files)
        pm.addAction(validate_a)

        info_a = QAction("Projektinfo…", self)
        info_a.triggered.connect(self._show_project_info)
        pm.addAction(info_a)

        pm.addSeparator()
        report_a = QAction("Bericht erstellen…", self)
        report_a.setShortcut("Ctrl+R")
        report_a.triggered.connect(self._create_report)
        pm.addAction(report_a)

        # View
        vm = mb.addMenu("Ansicht")
        for label, idx in [
            ("Dashboard",      0),
            ("Daten",          1),
            ("Labeling",       2),
            ("Training",       3),
            ("Modelle",        4),
            ("Klassifikation", 5),
            ("Batch-Inferenz", 9),
            ("Export",         6),
            ("Einstellungen",  7),
        ]:
            a = QAction(label, self)
            a.triggered.connect(lambda _, i=idx: self._switch_page(i))
            vm.addAction(a)

        # Audit
        audit_m = mb.addMenu("Audit")
        audit_view_a = QAction("Änderungsprotokoll anzeigen…", self)
        audit_view_a.triggered.connect(self._show_audit)
        audit_m.addAction(audit_view_a)

        # Help
        hm = mb.addMenu("Hilfe")
        manual_a = QAction("Handbuch öffnen…  (F1)", self)
        manual_a.triggered.connect(lambda: self._show_help())
        hm.addAction(manual_a)
        tour_a = QAction("Geführte Tour starten", self)
        tour_a.triggered.connect(self._start_tour)
        hm.addAction(tour_a)
        hm.addSeparator()
        for label, page_idx in [
            ("Dashboard – Hilfe",       0),
            ("Daten – Hilfe",           1),
            ("Labeling – Hilfe",        2),
            ("Training – Hilfe",        3),
            ("Modelle – Hilfe",         4),
            ("Klassifikation – Hilfe",  5),
            ("Export – Hilfe",          6),
            ("Einstellungen – Hilfe",   7),
            ("Kamera – Hilfe",          10),
            ("Tastenkürzel",            11),
            ("Fehlerbehebung",          12),
        ]:
            a = QAction(label, self)
            a.triggered.connect(lambda _, i=page_idx: self._show_help(i))
            hm.addAction(a)
        hm.addSeparator()
        log_a = QAction("Fehlerlog anzeigen…", self)
        log_a.triggered.connect(self._show_log_viewer)
        hm.addAction(log_a)
        hm.addSeparator()
        about_a = QAction("Über…", self)
        about_a.triggered.connect(self._show_about)
        hm.addAction(about_a)

        # Global F1 shortcut – works regardless of which widget has focus
        f1 = QShortcut(QKeySequence("F1"), self)
        f1.setContext(Qt.ApplicationShortcut)
        f1.activated.connect(lambda: self._show_help())

    def _build_statusbar(self) -> None:
        """Create the status bar with a project-stats label and an autosave indicator."""
        sb = QStatusBar()
        self.setStatusBar(sb)
        from PySide6.QtWidgets import QLabel
        self._status_label = QLabel("Bereit – kein Projekt geladen")
        sb.addWidget(self._status_label, 1)
        self._autosave_label = QLabel("")
        self._autosave_label.setStyleSheet("color: #3FB950; font-size: 10px; padding-right: 8px;")
        sb.addPermanentWidget(self._autosave_label)

    # ------------------------------------------------------------------ page switching

    @Slot(int)
    def _switch_page(self, idx: int) -> None:
        """Show the stacked-widget page at *idx* and sync the sidebar highlight."""
        self.stack.setCurrentIndex(idx)
        self.sidebar.set_page(idx)

    # ------------------------------------------------------------------ active learning

    def _on_global_drop(self, paths: list) -> None:
        """Global fallback: add dropped images regardless of which page is shown."""
        if not self.project:
            QMessageBox.warning(self, "Kein Projekt",
                                "Bitte zuerst ein Projekt öffnen, dann Bilder hineinziehen.")
            return
        added = 0
        for path in paths:
            if self.project.add_image(path):
                added += 1
        if added == 0:
            return
        # Refresh whichever page is visible
        cur = self.stack.currentWidget()
        if cur is self.labeling_page:
            self.labeling_page._refresh_thumb_list()
        elif cur is self.data_page:
            self.data_page.images_loaded.emit(added)
        n = len(self.project.images)
        self._status_label.setText(
            f"{added} Bild(er) per Drag & Drop hinzugefügt  |  {n} Bilder im Projekt"
        )

    def _on_al_queue_updated(self) -> None:
        """Refresh labeling page queue panel and notify user."""
        self.labeling_page.refresh_al_queue_panel()
        n = len(self.project.get_al_queue()) if self.project else 0
        self._status_label.setText(
            f"AL-Queue aktualisiert — {n} Bilder warten auf Labeling. "
            "Wechsle zum Labeling-Reiter."
        )

    def _on_model_loaded_api(self, model_path: str) -> None:
        """Push the loaded inferencer into the REST API for live classify."""
        self._rest_server.set_inferencer(self.inference_page.inferencer)

    def _on_labels_applied(self, count: int) -> None:
        """Refresh labeling page after semi-automatic labeling."""
        if hasattr(self.labeling_page, "_refresh_thumb_list"):
            self.labeling_page._refresh_thumb_list()
        if hasattr(self.labeling_page, "_update_stats"):
            self.labeling_page._update_stats()
        self._status_label.setText(
            f"✓ {count} Label(s) automatisch übernommen. "
            "Wechsle zum Labeling-Reiter zur Kontrolle."
        )

    # ------------------------------------------------------------------ project actions

    def _new_project(self) -> None:
        """Show the new-project dialog, ask for a save path, then load the project."""
        if not self._confirm_save():
            return
        from gui.new_project_dialog import NewProjectDialog
        dlg = NewProjectDialog(self)
        if dlg.exec() != NewProjectDialog.Accepted:
            return
        name = dlg.project_name
        project_type = dlg.project_type
        desc = dlg.description
        path, _ = QFileDialog.getSaveFileName(
            self, "Projekt speichern", f"{name}.json", "Projektdatei (*.json)"
        )
        if not path:
            return
        from core.project import Project
        project = Project()
        project.config.name = name
        project.config.description = desc
        project.config.project_type = project_type
        project.config.created_at = datetime.now().isoformat()
        project.save(path)
        self._load_project(project)
        log.info("Neues Projekt (%s): %s", project_type, path)

    def _open_project(self) -> None:
        """Open a file-chooser dialog to select and load a project JSON file."""
        if not self._confirm_save():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Projekt öffnen", "", "Projektdatei (*.json);;Alle Dateien (*)"
        )
        if not path:
            return
        try:
            from core.project import Project
            project = Project.load(path)
            self._load_project(project)
        except Exception as exc:
            QMessageBox.critical(self, "Ladefehler", str(exc))

    def _load_project(self, project) -> None:
        """
        Wire a freshly-loaded (or newly-created) project into the whole UI.

        Creates a new ``AuditTrail``, unlocks the sidebar, propagates the
        project to every page, updates the REST API, adds the path to recent
        projects, and navigates to the Dashboard.
        """
        self.project = project
        from core.audit import AuditTrail
        self.audit = AuditTrail(project.get_project_dir(), project.config.name)
        self.audit.log_project_saved(project.project_path)

        # Unlock sidebar and configure for project type
        self.sidebar.set_project_type(getattr(project.config, "project_type", "image"))
        self.sidebar.set_locked(False)

        # Propagate to all pages
        self.dashboard_page.set_project(project)
        self.data_page.set_project(project)
        self.camera_page.set_project(project)
        self.labeling_page.set_project(project, self.audit)
        self.training_page.set_project(project, self.audit)
        self.models_page.set_project(project)
        self.inference_page.set_project(project, self.audit)
        self.export_page.set_project(project)
        self.batch_page.set_project(project)

        self._rest_server.set_project(project)
        self._settings.add_recent_project(project.project_path)
        self._refresh_recent_menu()
        self.dashboard_page.set_recent_projects(self._settings.get_recent_projects())
        self.setWindowTitle(f"{APP_NAME} – {project.config.name or project.project_path}")
        self._update_status()
        self._switch_page(0)  # Go to dashboard

    def _open_camera_dialog(self) -> None:
        """
        Open the camera-capture dialog, optionally injecting MQTT and REST clients.

        Captured images are added to the project after the dialog closes.
        """
        if not self.project:
            QMessageBox.warning(self, "Kein Projekt", "Bitte zuerst ein Projekt öffnen.")
            return
        import os
        save_dir = os.path.join(
            os.path.dirname(self.project.project_path or ""),
            "camera_captures"
        ) if self.project.project_path else None
        from gui.camera_capture_dialog import CameraCaptureDialog
        dlg = CameraCaptureDialog(save_dir=save_dir, parent=self)
        # Inject MQTT client if configured
        mqtt_cfg = self._settings.get_mqtt_config()
        if mqtt_cfg.get("enabled"):
            from core.mqtt_client import MQTTAlarmClient
            mqtt = MQTTAlarmClient(
                host=mqtt_cfg.get("host", "localhost"),
                port=int(mqtt_cfg.get("port", 1883)),
                topic=mqtt_cfg.get("topic", "picture_studio/anomaly"),
                username=mqtt_cfg.get("username", ""),
                password=mqtt_cfg.get("password", ""),
            )
            if mqtt.connect():
                dlg.set_mqtt_client(mqtt)
        # Inject REST API server for live dashboard scores
        if self._rest_server.is_running:
            dlg.set_api_server(self._rest_server)
        if dlg.exec() and dlg.captured_paths:
            added = sum(1 for p in dlg.captured_paths if self.project.add_image(p))
            if added:
                self.labeling_page.set_project(self.project)
                self.dashboard_page.refresh()
                self._update_status()
            self._status_label.setText(
                f"{added} Kamerabild(er) zum Projekt hinzugefügt"
            )

    def _save_project(self) -> None:
        """Save the current project to its existing path."""
        if not self.project:
            return
        self._sync_and_save()

    def _save_project_as(self) -> None:
        """Prompt for a new path, then save a copy of the project there."""
        if not self.project:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Speichern unter", "", "Projektdatei (*.json)"
        )
        if path:
            self._sync_and_save(path)

    def _sync_and_save(self, path: str = None) -> None:
        """
        Flush unsaved ROIs from the labeling page, optionally create a backup,
        then persist the project. Shows a critical dialog on error.
        """
        self.labeling_page._save_current_rois()
        try:
            if self._settings.get_backup_enabled() and not path:
                backup = self.project.create_backup()
                if backup and self.audit:
                    self.audit.log_project_backup(backup)
            self.project.save(path)
            if self.audit:
                self.audit.log_project_saved(self.project.project_path)
            self._status_label.setText(
                f"Gespeichert: {os.path.basename(self.project.project_path)}  "
                f"({datetime.now().strftime('%H:%M:%S')})"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Speicherfehler", str(exc))

    def _create_backup(self) -> None:
        """Manually trigger a project backup and notify the user of its path."""
        if not self.project:
            return
        backup = self.project.create_backup()
        if backup:
            QMessageBox.information(self, "Backup", f"Backup gespeichert:\n{backup}")

    def _autosave(self) -> None:
        """Timer slot: save the project silently if autosave is enabled in settings."""
        if self.project and self._settings.get_autosave_enabled():
            try:
                self.labeling_page._save_current_rois()
                self.project.save()
                from datetime import datetime as _dt
                self._autosave_label.setText(
                    f"✓ Auto-gespeichert {_dt.now().strftime('%H:%M:%S')}"
                )
                log.debug("Autosave erfolgreich")
            except Exception as exc:
                self._autosave_label.setText("⚠ Autosave fehlgeschlagen")
                log.warning("Autosave fehlgeschlagen: %s", exc)

    def _reset_autosave_timer(self) -> None:
        """(Re)start the autosave timer using the interval from settings."""
        interval = self._settings.get_autosave_interval() * 1000
        self._autosave_timer.start(interval)

    @Slot(int, bool)
    def _on_autosave_changed(self, interval: int, enabled: bool) -> None:
        """Update the autosave timer when the user changes settings."""
        if enabled:
            self._autosave_timer.start(interval * 1000)
        else:
            self._autosave_timer.stop()

    def _confirm_save(self) -> bool:
        """
        Ask the user whether to save before performing a destructive action.

        Returns ``True`` to proceed, ``False`` if the user pressed Cancel.
        """
        if not self.project:
            return True
        reply = QMessageBox.question(
            self, "Speichern?",
            "Aktuelles Projekt speichern?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
        )
        if reply == QMessageBox.Cancel:
            return False
        if reply == QMessageBox.Yes:
            self._save_project()
        return True

    # ------------------------------------------------------------------ label management

    def _manage_labels(self) -> None:
        """Open the label-manager dialog; propagate label changes to affected pages."""
        if not self.project:
            QMessageBox.warning(self, "Kein Projekt", "Bitte zuerst ein Projekt öffnen.")
            return
        from gui.label_manager import LabelManagerDialog
        dlg = LabelManagerDialog(self.project, self)
        dlg.labels_changed.connect(self.labeling_page.on_labels_changed)
        dlg.labels_changed.connect(self.dashboard_page.refresh)
        dlg.exec()

    # ------------------------------------------------------------------ training callback

    @Slot(dict)
    def _on_images_loaded(self, count: int) -> None:
        """Refresh labeling page and dashboard after new images are added via DataPage."""
        if not self.project:
            return
        self.labeling_page.set_project(self.project)
        self.dashboard_page.refresh()
        self._update_status()

    def _on_training_finished(self, result: dict) -> None:
        """Persist the training result, refresh models/dashboard, and autosave."""
        if not self.project:
            return
        self.project.add_training_run(result)
        self.project.current_model_path = result.get("best_model_path", "")
        self.models_page.refresh()
        self.dashboard_page.refresh()
        self._sync_and_save()

    # ------------------------------------------------------------------ file validation

    def _validate_files(self) -> None:
        """Check all project images for missing/unreadable files and report results."""
        if not self.project:
            return
        v = self.project.validate_image_files()
        msg = (
            f"OK: {len(v['ok'])}\n"
            f"Fehlend: {len(v['missing'])}\n"
            f"Unlesbar: {len(v['unreadable'])}"
        )
        if v["missing"]:
            msg += "\n\nFehlende Dateien:\n" + "\n".join(
                os.path.basename(p) for p in v["missing"][:15]
            )
        QMessageBox.information(self, "Datei-Validierung", msg)

    # ------------------------------------------------------------------ audit

    def _show_audit(self) -> None:
        """Open a read-only dialog showing the last 200 audit-trail entries."""
        if not self.audit:
            QMessageBox.information(self, "Audit", "Kein Audit-Trail vorhanden.")
            return
        entries = self.audit.get_entries(200)
        text = self.audit.format_entries(entries)
        from PySide6.QtWidgets import QDialog, QTextEdit, QVBoxLayout
        dlg = QDialog(self)
        dlg.setWindowTitle("Änderungsprotokoll")
        dlg.resize(760, 480)
        v = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setFont(QFont("Courier New", 9))
        te.setPlainText(text)
        v.addWidget(te)
        dlg.exec()

    # ------------------------------------------------------------------ info / about

    def _create_report(self) -> None:
        """Open the report-generation dialog for the current project."""
        if not self.project:
            QMessageBox.warning(self, "Kein Projekt", "Bitte zuerst ein Projekt öffnen.")
            return
        from gui.report_dialog import ReportDialog
        dlg = ReportDialog(self.project, self)
        dlg.exec()

    def _show_project_info(self) -> None:
        """Display a summary dialog with project metadata and latest training metrics."""
        if not self.project:
            QMessageBox.information(self, "Kein Projekt", "Kein Projekt geladen.")
            return
        p = self.project
        runs = len(p.training_runs)
        last = p.get_last_training_run() or {}
        m = last.get("metrics", {})
        info = (
            f"Name:           {p.config.name}\n"
            f"Pfad:           {p.project_path}\n"
            f"Erstellt:       {p.config.created_at[:19]}\n"
            f"Geändert:       {p.config.modified_at[:19]}\n\n"
            f"Bilder:         {len(p.images)}\n"
            f"Gelabelt:       {p.get_labeled_image_count()}\n"
            f"Klassen:        {', '.join(p.labels.keys())}\n"
            f"ROIs:           {p.get_roi_count()}\n"
            f"Trainingsläufe: {runs}\n"
        )
        if last:
            info += (
                f"\nLetztes Training:\n"
                f"  Accuracy: {m.get('accuracy', 0)*100:.2f}%\n"
                f"  F1:       {m.get('macro_f1', 0)*100:.2f}%\n"
                f"  Modell:   {last.get('model_type', '?')}\n"
            )
        QMessageBox.information(self, "Projektinfo", info)

    def _show_help(self, page_idx: int = None) -> None:
        """Open the help dialog on the page matching *page_idx* (defaults to current page)."""
        from gui.help_dialog import HelpDialog
        idx = page_idx if page_idx is not None else self.stack.currentIndex()
        dlg = HelpDialog(page_index=idx, parent=self)
        dlg.exec()

    def _start_tour(self) -> None:
        """Launch the guided tour overlaid on the currently-visible page."""
        idx = self.stack.currentIndex()
        page_widget = self.stack.currentWidget()
        self._tour.start(idx, page_widget)

    def resizeEvent(self, event) -> None:
        """Keep the floating tour overlay positioned correctly on window resize."""
        super().resizeEvent(event)
        if hasattr(self, "_tour") and self._tour.isVisible():
            self._tour._reposition()

    def _check_crash_recovery(self) -> None:
        """Scan recent project paths for orphaned .tmp files from a crashed write."""
        for path in self._settings.get_recent_projects():
            tmp = path + ".tmp"
            if os.path.exists(tmp) and not os.path.exists(path):
                reply = QMessageBox.question(
                    self, "Crash-Wiederherstellung",
                    f"Eine unvollständige Speicherung wurde gefunden:\n{tmp}\n\n"
                    "Soll die Datei als Projektdatei wiederhergestellt werden?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    try:
                        os.replace(tmp, path)
                        from core.project import Project
                        self._load_project(Project.load(path))
                    except Exception as exc:
                        QMessageBox.critical(self, "Wiederherstellungsfehler", str(exc))
                else:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

    def _show_log_viewer(self) -> None:
        """Open a scrollable dialog showing the current session log file."""
        from utils.logging_utils import get_log_file, get_log_dir
        log_file = get_log_file()
        log_dir = get_log_dir()
        from PySide6.QtWidgets import QDialog, QTextEdit, QVBoxLayout, QPushButton, QHBoxLayout
        dlg = QDialog(self)
        dlg.setWindowTitle("Fehlerlog")
        dlg.resize(860, 520)
        v = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setFont(QFont("Courier New", 9))
        if log_file and os.path.exists(log_file):
            try:
                with open(log_file, encoding="utf-8") as fh:
                    content = fh.read()
                te.setPlainText(content)
                from PySide6.QtGui import QTextCursor
                te.moveCursor(QTextCursor.End)
            except Exception as exc:
                te.setPlainText(f"Log konnte nicht geladen werden: {exc}")
        else:
            te.setPlainText(f"Kein Log für diese Sitzung gefunden.\nLog-Ordner: {log_dir}")
        v.addWidget(te)
        btn_row = QHBoxLayout()
        if log_dir and os.path.exists(log_dir):
            open_btn = QPushButton("Log-Ordner öffnen")
            open_btn.clicked.connect(lambda: __import__("subprocess").Popen(
                ["open" if __import__("platform").system() == "Darwin"
                 else "xdg-open", log_dir]
            ))
            btn_row.addWidget(open_btn)
        btn_row.addStretch()
        close_btn = QPushButton("Schließen")
        close_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(close_btn)
        v.addLayout(btn_row)
        dlg.exec()

    def _show_about(self) -> None:
        """Display the About dialog with application version and library versions."""
        from utils.reproducibility import get_software_versions
        vers = get_software_versions()
        text = f"{APP_NAME} v{APP_VERSION}\n\nBibliotheken:\n"
        for k, v in vers.items():
            text += f"  {k}: {v}\n"
        QMessageBox.about(self, f"Über {APP_NAME}", text)

    # ------------------------------------------------------------------ theming

    def _apply_theme(self, theme: str) -> None:
        """Switch the application palette between dark and the system default."""
        app = QApplication.instance()
        if theme == "dark":
            from gui.theme import apply_dark_theme
            apply_dark_theme(app)
            self.sidebar.setStyleSheet(
                "QWidget#Sidebar { background: #161B22; border-right: 1px solid #30363D; }"
            )
        else:
            app.setPalette(app.style().standardPalette())
            app.setStyleSheet("")
            self.sidebar.setStyleSheet("")

    # ------------------------------------------------------------------ recent projects

    def _refresh_recent_menu(self) -> None:
        """Rebuild the 'Zuletzt geöffnet' submenu from the recent-projects list."""
        self._recent_menu.clear()
        for path in self._settings.get_recent_projects():
            a = QAction(os.path.basename(path) or path, self)
            a.setToolTip(path)
            a.triggered.connect(lambda _, p=path: self._open_recent(p))
            self._recent_menu.addAction(a)

    def _open_recent(self, path: str) -> None:
        """Load a project from the recent-projects list; warn if the file is missing."""
        if not os.path.exists(path):
            QMessageBox.warning(self, "Nicht gefunden", f"Datei nicht gefunden:\n{path}")
            return
        if not self._confirm_save():
            return
        try:
            from core.project import Project
            self._load_project(Project.load(path))
        except Exception as exc:
            QMessageBox.critical(self, "Ladefehler", str(exc))

    # ------------------------------------------------------------------ status

    def _update_status(self) -> None:
        """Refresh the status-bar label with current project statistics."""
        if not self.project:
            self._status_label.setText("Bereit – kein Projekt geladen")
            return
        self._status_label.setText(
            f"Projekt: {self.project.config.name or '–'}  │  "
            f"Bilder: {len(self.project.images)}  │  "
            f"Gelabelt: {self.project.get_labeled_image_count()}  │  "
            f"ROIs: {self.project.get_roi_count()}  │  "
            f"Klassen: {len(self.project.labels)}"
        )

    # ------------------------------------------------------------------ close

    def closeEvent(self, event) -> None:
        """Save window geometry, stop the REST server, and optionally save the project."""
        self._settings.save_window_geometry(self.saveGeometry())
        if self._rest_server.is_running:
            self._rest_server.stop()
        if self.project:
            reply = QMessageBox.question(
                self, "Beenden",
                "Projekt vor dem Beenden speichern?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                event.ignore()
                return
            if reply == QMessageBox.Yes:
                self._save_project()
        event.accept()
