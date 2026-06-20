"""
Startup mode selection: Beginner (label + train only) vs Expert (everything).

Shown once at application start (before MainWindow). The choice is persisted as
the default for the next launch via AppSettings.set_ui_mode().
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
)

from utils.i18n import tr


class ModeSelectDialog(QDialog):
    """Modal dialog returning 'beginner' or 'expert' in ``selected_mode``."""

    def __init__(self, default_mode: str = "expert", parent=None) -> None:
        super().__init__(parent)
        self.selected_mode: str = default_mode
        self.setWindowTitle(tr("mode.title"))
        self.setModal(True)
        self._build_ui(default_mode)

    def _build_ui(self, default_mode: str) -> None:
        layout = QVBoxLayout(self)

        prompt = QLabel(tr("mode.prompt"))
        prompt.setWordWrap(True)
        prompt.setStyleSheet("font-size:14px;font-weight:bold;")
        layout.addWidget(prompt)

        row = QHBoxLayout()
        row.addWidget(self._mode_button(
            "beginner", "👤", tr("mode.beginner"), tr("mode.beginner_desc"),
            default_mode == "beginner"))
        row.addWidget(self._mode_button(
            "expert", "🛠", tr("mode.expert"), tr("mode.expert_desc"),
            default_mode != "beginner"))
        layout.addLayout(row)

    def _mode_button(self, mode: str, icon: str, title: str, desc: str,
                     is_default: bool) -> QPushButton:
        btn = QPushButton(f"{icon}  {title}\n{desc}")
        btn.setMinimumSize(220, 110)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setDefault(is_default)
        btn.clicked.connect(lambda: self._choose(mode))
        return btn

    def _choose(self, mode: str) -> None:
        self.selected_mode = mode
        self.accept()

    @staticmethod
    def choose(default_mode: str = "expert", parent=None) -> str:
        """Show the dialog modally and return the chosen mode."""
        dlg = ModeSelectDialog(default_mode, parent)
        dlg.exec()
        return dlg.selected_mode
