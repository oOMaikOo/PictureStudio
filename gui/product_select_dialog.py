"""First-launch product line: Studio (image) vs Monitor (video/fleet)."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
)

from utils.i18n import tr


class ProductSelectDialog(QDialog):
    """Modal dialog returning ``studio`` or ``monitor`` in ``selected_line``."""

    def __init__(self, default_line: str = "studio", parent=None) -> None:
        super().__init__(parent)
        self.selected_line: str = (
            "monitor" if default_line == "monitor" else "studio"
        )
        self.setWindowTitle(tr("product.title"))
        self.setModal(True)
        self._build_ui(self.selected_line)

    def _build_ui(self, default_line: str) -> None:
        layout = QVBoxLayout(self)
        prompt = QLabel(tr("product.prompt"))
        prompt.setWordWrap(True)
        prompt.setStyleSheet("font-size:14px;font-weight:bold;")
        layout.addWidget(prompt)

        row = QHBoxLayout()
        row.addWidget(self._btn(
            "studio", "🖼", tr("product.studio"), tr("product.studio_desc"),
            default_line == "studio"))
        row.addWidget(self._btn(
            "monitor", "📹", tr("product.monitor"), tr("product.monitor_desc"),
            default_line == "monitor"))
        layout.addLayout(row)

    def _btn(self, line: str, icon: str, title: str, desc: str,
             is_default: bool) -> QPushButton:
        btn = QPushButton(f"{icon}  {title}\n{desc}")
        btn.setMinimumSize(240, 120)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setDefault(is_default)
        btn.clicked.connect(lambda: self._choose(line))
        return btn

    def _choose(self, line: str) -> None:
        self.selected_line = "monitor" if line == "monitor" else "studio"
        self.accept()

    @staticmethod
    def choose(default_line: str = "studio", parent=None) -> str:
        dlg = ProductSelectDialog(default_line, parent)
        dlg.exec()
        return dlg.selected_line
