"""
Navigation sidebar with icon + text buttons.
"""
import platform

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QSizePolicy
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont


def _ui_font(size: int = 11) -> QFont:
    f = QFont()
    if platform.system() == "Darwin":
        f.setFamily(".AppleSystemUIFont")
    f.setPointSize(size)
    return f


PAGES = [
    ("Dashboard",     "🏠"),
    ("Daten",         "📁"),
    ("Labeling",      "🏷"),
    ("Training",      "🧠"),
    ("Modelle",       "📊"),
    ("Klassifikation","🔍"),
    ("Export",        "📤"),
    ("Einstellungen", "⚙"),
]


class Sidebar(QWidget):
    """Vertical navigation sidebar."""

    page_requested = Signal(int)
    help_requested = Signal()
    tour_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(168)
        self.setObjectName("Sidebar")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 8)
        layout.setSpacing(2)

        # App name
        title = QLabel("Image\nLabeling\nStudio")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "color: #3498DB; font-size: 13px; font-weight: bold;"
            "padding: 8px 0 12px 0; border-bottom: 1px solid #333;"
        )
        layout.addWidget(title)

        self._buttons = []
        for i, (name, icon) in enumerate(PAGES):
            btn = QPushButton(f"{icon}  {name}")
            btn.setCheckable(True)
            btn.setFixedHeight(42)
            btn.setFont(_ui_font(12))
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 6px 12px;
                    border: none;
                    border-radius: 6px;
                    color: #ccc;
                    background: transparent;
                }
                QPushButton:hover {
                    background: #1565C0;
                    color: white;
                }
                QPushButton:checked {
                    background: #1976D2;
                    color: white;
                    font-weight: bold;
                }
            """)
            btn.clicked.connect(lambda checked, idx=i: self._on_click(idx))
            layout.addWidget(btn)
            self._buttons.append(btn)

        layout.addStretch()

        btn_style = """
            QPushButton {{
                text-align: center;
                padding: 4px 8px;
                border: 1px solid {border};
                border-radius: 6px;
                color: {color};
                background: transparent;
                margin: 2px 4px;
            }}
            QPushButton:hover {{ background: {hover}; color: white; }}
        """

        # Tour button
        self._tour_btn = QPushButton("▶  Tour starten")
        self._tour_btn.setFixedHeight(34)
        self._tour_btn.setFont(_ui_font(10))
        self._tour_btn.setCursor(Qt.PointingHandCursor)
        self._tour_btn.setStyleSheet(
            btn_style.format(border="#F39C12", color="#F0B27A", hover="#D68910")
        )
        self._tour_btn.clicked.connect(self.tour_requested.emit)
        layout.addWidget(self._tour_btn)

        # Help button
        self._help_btn = QPushButton("?  Hilfe (F1)")
        self._help_btn.setFixedHeight(34)
        self._help_btn.setFont(_ui_font(10))
        self._help_btn.setCursor(Qt.PointingHandCursor)
        self._help_btn.setStyleSheet(
            btn_style.format(border="#1976D2", color="#5DADE2", hover="#1565C0")
        )
        self._help_btn.clicked.connect(self.help_requested.emit)
        layout.addWidget(self._help_btn)

        # Version label
        from utils.config import APP_VERSION
        ver_lbl = QLabel(f"v{APP_VERSION}")
        ver_lbl.setAlignment(Qt.AlignCenter)
        ver_lbl.setStyleSheet("color: #555; font-size: 9px;")
        layout.addWidget(ver_lbl)

        # Select first
        self._select(0)

    def _on_click(self, idx: int) -> None:
        self._select(idx)
        self.page_requested.emit(idx)

    def _select(self, idx: int) -> None:
        for i, btn in enumerate(self._buttons):
            btn.setChecked(i == idx)

    def set_page(self, idx: int) -> None:
        self._select(idx)
