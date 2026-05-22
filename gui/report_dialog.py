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
        from utils.i18n import tr
        super().__init__(parent)
        self.project = project
        self.setWindowTitle(tr("report.title"))
        self.setMinimumWidth(460)
        self._build_ui()
        self._set_default_path()

    def _build_ui(self) -> None:
        from utils.i18n import tr
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
        path_grp = QGroupBox(tr("report.output_group"))
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
        opt_grp = QGroupBox(tr("report.options_group"))
        ov = QVBoxLayout(opt_grp)
        self._open_cb = QCheckBox(tr("report.open_cb"))
        self._open_cb.setChecked(True)
        ov.addWidget(self._open_cb)
        root.addWidget(opt_grp)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton(tr("common.cancel"))
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        self._gen_btn = QPushButton(tr("report.generate_btn"))
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
        from utils.i18n import tr
        path, _ = QFileDialog.getSaveFileName(
            self, tr("report.save_dlg"), self._path_edit.text(),
            "HTML (*.html);;Alle Dateien (*)"
        )
        if path:
            self._path_edit.setText(path)

    def _generate(self) -> None:
        from utils.i18n import tr
        path = self._path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, tr("report.no_path_title"), tr("report.no_path_msg"))
            return
        try:
            from core.report_generator import generate_html_report
            generate_html_report(self.project, path)
        except Exception as exc:
            QMessageBox.critical(self, tr("common.error"), str(exc))
            return

        if self._open_cb.isChecked():
            if sys.platform == "darwin":
                subprocess.Popen(["open", path])
            elif sys.platform == "win32":
                os.startfile(path)
            else:
                subprocess.Popen(["xdg-open", path])

        QMessageBox.information(
            self, tr("report.success_title"), tr("report.success_msg", path=path)
        )
        self.accept()
