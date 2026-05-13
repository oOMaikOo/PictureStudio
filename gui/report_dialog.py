"""
Report dialog: configure and generate an HTML project report.
"""
import os
import sys
import subprocess

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QCheckBox, QGroupBox, QLineEdit, QMessageBox,
)
from PySide6.QtCore import Qt


class ReportDialog(QDialog):
    """Generates and opens a self-contained HTML project report."""

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Projektbericht erstellen")
        self.setMinimumWidth(460)
        self._build_ui()
        self._set_default_path()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(12)

        info = QLabel(
            "Erstellt einen selbstständigen HTML-Bericht mit Datensatzstatistiken,\n"
            "Klassenverteilung, Trainingsverlauf und Warnungen."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#7F8C8D;font-size:11px;")
        root.addWidget(info)

        # Output path
        path_grp = QGroupBox("Ausgabedatei")
        pv = QVBoxLayout(path_grp)
        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setReadOnly(True)
        path_row.addWidget(self._path_edit)
        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(32)
        browse_btn.clicked.connect(self._browse)
        path_row.addWidget(browse_btn)
        pv.addLayout(path_row)
        root.addWidget(path_grp)

        # Options
        opt_grp = QGroupBox("Optionen")
        ov = QVBoxLayout(opt_grp)
        self._open_cb = QCheckBox("Bericht nach dem Erstellen im Browser öffnen")
        self._open_cb.setChecked(True)
        ov.addWidget(self._open_cb)
        root.addWidget(opt_grp)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        self._gen_btn = QPushButton("Bericht erstellen")
        self._gen_btn.setStyleSheet(
            "background:#1F6FEB;color:white;font-weight:bold;padding:6px 16px;"
        )
        self._gen_btn.clicked.connect(self._generate)
        btn_row.addWidget(self._gen_btn)
        root.addLayout(btn_row)

    def _set_default_path(self) -> None:
        base = ""
        if self.project and self.project.project_path:
            base = os.path.dirname(self.project.project_path)
        name = (self.project.config.name or "projekt").replace(" ", "_")
        self._path_edit.setText(
            os.path.join(base or os.path.expanduser("~"), f"{name}_bericht.html")
        )

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Bericht speichern unter", self._path_edit.text(),
            "HTML (*.html);;Alle Dateien (*)"
        )
        if path:
            self._path_edit.setText(path)

    def _generate(self) -> None:
        path = self._path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Kein Pfad", "Bitte Ausgabepfad wählen.")
            return
        try:
            from core.report_generator import generate_html_report
            generate_html_report(self.project, path)
        except Exception as exc:
            QMessageBox.critical(self, "Fehler", str(exc))
            return

        if self._open_cb.isChecked():
            if sys.platform == "darwin":
                subprocess.Popen(["open", path])
            elif sys.platform == "win32":
                os.startfile(path)
            else:
                subprocess.Popen(["xdg-open", path])

        QMessageBox.information(
            self, "Erstellt", f"Bericht gespeichert:\n{path}"
        )
        self.accept()
