"""
ProjectController — handles project lifecycle outside of MainWindow.

Extracted from MainWindow to separate concerns:
  - File I/O: new / open / save / save-as / backup / crash-recovery
  - Audit trail management
  - Recent-project persistence

MainWindow instantiates this controller, connects its signals, and delegates
all project-file operations to it. UI state (sidebar, status-bar, page wiring)
stays in MainWindow; pure project logic lives here.
"""
import os
from datetime import datetime
from typing import Optional

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox

from utils.logging_utils import get_logger
from utils.settings import AppSettings

log = get_logger()


class ProjectController(QObject):
    """
    Manages project CRUD operations for MainWindow.

    Signals
    -------
    project_loaded(project)   — emitted after a project is fully loaded
    project_saved(path)       — emitted after a successful save
    backup_created(path)      — emitted after a manual backup
    error_occurred(title, msg)— emitted when an operation fails
    status_message(msg)       — short status-bar text
    """

    project_loaded  = Signal(object)        # Project instance
    project_saved   = Signal(str)           # save path
    backup_created  = Signal(str)           # backup path
    error_occurred  = Signal(str, str)      # title, detail
    status_message  = Signal(str)

    def __init__(self, settings: AppSettings, parent: QObject = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self.project = None
        self.audit = None

    # ── public project operations ─────────────────────────────────────────────

    def new_project(self, name: str, project_type: str, description: str,
                    save_path: str) -> None:
        """Create a blank project, save it, and emit project_loaded."""
        from core.project import Project
        project = Project()
        project.config.name = name
        project.config.description = description
        project.config.project_type = project_type
        project.config.created_at = datetime.now().isoformat()
        try:
            project.save(save_path)
        except Exception as exc:
            self.error_occurred.emit("Speicherfehler", str(exc))
            return
        self._finalize_load(project)
        log.info("Neues Projekt (%s): %s", project_type, save_path)

    def open_project(self, path: str) -> None:
        """Load a project from *path*, handling crash-recovery if needed."""
        from core.project import Project
        try:
            recovery_available, tmp_path = Project.check_tmp_recovery(path)
            if recovery_available:
                reply = QMessageBox.question(
                    None,
                    "Crash-Recovery",
                    "Eine ungespeicherte Sicherungskopie wurde gefunden "
                    "(die App wurde möglicherweise unerwartet beendet).\n\n"
                    "Soll diese Sicherungskopie wiederhergestellt werden?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    project = Project.load(tmp_path)
                    project.project_path = path
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                if reply == QMessageBox.Yes:
                    self._finalize_load(project)
                    return
            project = Project.load(path)
            self._finalize_load(project)
        except Exception as exc:
            self.error_occurred.emit("Ladefehler", str(exc))

    def save_project(self, path: str = None, is_autosave: bool = False) -> bool:
        """Save the current project; creates a backup first if configured (not during autosave)."""
        if not self.project:
            return False
        if path:
            self.project.project_path = path
        if not self.project.project_path:
            return False
        try:
            if self._settings.get_backup_enabled() and not path and not is_autosave:
                backup = self.project.create_backup()
                if backup:
                    if self.audit:
                        self.audit.log_project_backup(backup)
                    self.backup_created.emit(backup)
            self.project.save(path)
            if self.audit:
                self.audit.log_project_saved(self.project.project_path)
            self.project_saved.emit(self.project.project_path)
            self.status_message.emit(
                f"Gespeichert: {os.path.basename(self.project.project_path)}  "
                f"({datetime.now().strftime('%H:%M:%S')})"
            )
            return True
        except Exception as exc:
            self.error_occurred.emit("Speicherfehler", str(exc))
            return False

    def create_backup(self) -> None:
        """Manually create a timestamped backup and emit backup_created."""
        if not self.project:
            return
        backup = self.project.create_backup()
        if backup:
            if self.audit:
                self.audit.log_project_backup(backup)
            self.backup_created.emit(backup)

    def check_crash_recovery(self, recent_paths: list) -> None:
        """Scan recent project paths for orphaned .tmp files from a crashed write."""
        from core.project import Project
        for path in recent_paths:
            tmp = path + ".tmp"
            if os.path.exists(tmp) and not os.path.exists(path):
                reply = QMessageBox.question(
                    None,
                    "Crash-Wiederherstellung",
                    f"Eine unvollständige Speicherung wurde gefunden:\n{tmp}\n\n"
                    "Soll die Datei als Projektdatei wiederhergestellt werden?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    try:
                        os.replace(tmp, path)
                        project = Project.load(path)
                        self._finalize_load(project)
                    except Exception as exc:
                        self.error_occurred.emit("Wiederherstellungsfehler", str(exc))
                else:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

    # ── internal helpers ──────────────────────────────────────────────────────

    def _finalize_load(self, project) -> None:
        """Wire audit trail, persist to recent list, then emit project_loaded."""
        from core.audit import AuditTrail
        self.project = project
        self.audit = AuditTrail(project.get_project_dir(), project.config.name)
        self.audit.log_project_saved(project.project_path)
        self._settings.add_recent_project(project.project_path)
        self.project_loaded.emit(project)
