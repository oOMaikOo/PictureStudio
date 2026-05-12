"""
Navigation sidebar with icon + text buttons.
Supports two page configurations (image / video) and a locked state.
"""
import platform
from typing import List, Tuple

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont


def _ui_font(size: int = 11) -> QFont:
    f = QFont()
    if platform.system() == "Darwin":
        f.setFamily(".AppleSystemUIFont")
    f.setPointSize(size)
    return f


# (label, icon, stack_idx)
_IMAGE_PAGES: List[Tuple[str, str, int]] = [
    ("Dashboard",      "🏠", 0),
    ("Daten",          "📁", 1),
    ("Labeling",       "🏷", 2),
    ("Training",       "🧠", 3),
    ("Modelle",        "📊", 4),
    ("Klassifikation", "🔍", 5),
    ("Export",         "📤", 6),
    ("Einstellungen",  "⚙",  7),
]

_VIDEO_PAGES: List[Tuple[str, str, int]] = [
    ("Dashboard",       "🏠", 0),
    ("Daten",           "📁", 1),
    ("Live & Anomalie", "🎥", 8),
    ("Modelle",         "📊", 4),
    ("Export",          "📤", 6),
    ("Einstellungen",   "⚙",  7),
]

_BTN_STYLE = """
    QPushButton {
        text-align: left;
        padding: 6px 12px;
        border: none;
        border-radius: 6px;
        color: #8B949E;
        background: transparent;
        font-size: 12px;
    }
    QPushButton:hover:enabled {
        background: #21262D;
        color: #E6EDF3;
    }
    QPushButton:checked {
        background: #1F6FEB;
        color: white;
        font-weight: bold;
    }
    QPushButton:disabled {
        color: #30363D;
        background: transparent;
    }
"""


class Sidebar(QWidget):
    """Vertical navigation sidebar."""

    page_requested = Signal(int)   # emits stack_idx
    help_requested = Signal()
    tour_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(168)
        self.setObjectName("Sidebar")

        self._locked = True
        self._page_config = _IMAGE_PAGES
        # List of (QPushButton, stack_idx)
        self._buttons: List[Tuple[QPushButton, int]] = []

        self._build_outer_ui()
        self._rebuild_buttons()

    # ------------------------------------------------------------------ build

    def _build_outer_ui(self) -> None:
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 8, 4, 8)
        self._layout.setSpacing(2)

        title_area = QWidget()
        title_area.setStyleSheet(
            "QWidget { border-bottom: 1px solid #30363D; padding-bottom: 8px; }"
        )
        title_vl = QVBoxLayout(title_area)
        title_vl.setContentsMargins(4, 8, 4, 10)
        title_vl.setSpacing(2)

        app_lbl = QLabel("Picture Studio")
        app_lbl.setAlignment(Qt.AlignCenter)
        app_lbl.setStyleSheet(
            "color: #388BFD; font-size: 14px; font-weight: bold;"
            "border: none; padding: 0;"
        )
        title_vl.addWidget(app_lbl)

        self._type_badge = QLabel("")
        self._type_badge.setAlignment(Qt.AlignCenter)
        self._type_badge.setStyleSheet(
            "color: #484F58; font-size: 10px; border: none; padding: 0;"
        )
        self._type_badge.setVisible(False)
        title_vl.addWidget(self._type_badge)

        self._layout.addWidget(title_area)

        # Container for nav buttons — rebuilt on type change
        self._btn_container = QWidget()
        self._btn_layout = QVBoxLayout(self._btn_container)
        self._btn_layout.setContentsMargins(0, 0, 0, 0)
        self._btn_layout.setSpacing(2)
        self._layout.addWidget(self._btn_container)

        self._layout.addStretch()

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

        self._tour_btn = QPushButton("▶  Tour starten")
        self._tour_btn.setFixedHeight(34)
        self._tour_btn.setFont(_ui_font(10))
        self._tour_btn.setCursor(Qt.PointingHandCursor)
        self._tour_btn.setStyleSheet(
            btn_style.format(border="#F39C12", color="#F0B27A", hover="#D68910")
        )
        self._tour_btn.clicked.connect(self.tour_requested.emit)
        self._layout.addWidget(self._tour_btn)

        self._help_btn = QPushButton("?  Hilfe (F1)")
        self._help_btn.setFixedHeight(34)
        self._help_btn.setFont(_ui_font(10))
        self._help_btn.setCursor(Qt.PointingHandCursor)
        self._help_btn.setStyleSheet(
            btn_style.format(border="#1976D2", color="#5DADE2", hover="#1565C0")
        )
        self._help_btn.clicked.connect(self.help_requested.emit)
        self._layout.addWidget(self._help_btn)

        from utils.config import APP_VERSION
        ver_lbl = QLabel(f"v{APP_VERSION}")
        ver_lbl.setAlignment(Qt.AlignCenter)
        ver_lbl.setStyleSheet("color: #555; font-size: 9px;")
        self._layout.addWidget(ver_lbl)

    def _rebuild_buttons(self) -> None:
        # Remove and delete all existing nav buttons
        while self._btn_layout.count():
            item = self._btn_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._buttons.clear()

        for label, icon, stack_idx in self._page_config:
            btn = QPushButton(f"{icon}  {label}")
            btn.setCheckable(True)
            btn.setFixedHeight(42)
            btn.setFont(_ui_font(12))
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(_BTN_STYLE)
            is_dashboard = stack_idx == 0
            btn.setEnabled(not self._locked or is_dashboard)
            btn.clicked.connect(lambda checked, i=stack_idx: self._on_click(i))
            self._btn_layout.addWidget(btn)
            self._buttons.append((btn, stack_idx))

        self._select_by_stack(0)

    # ------------------------------------------------------------------ public API

    def set_locked(self, locked: bool) -> None:
        """Disable all nav buttons except Dashboard when locked."""
        self._locked = locked
        for btn, stack_idx in self._buttons:
            is_dashboard = stack_idx == 0
            btn.setEnabled(not locked or is_dashboard)

    def set_project_type(self, type_str: str) -> None:
        """Switch between image and video page configurations."""
        new_config = _VIDEO_PAGES if type_str == "video" else _IMAGE_PAGES
        badge_text = "📸 Bildprojekt" if type_str == "image" else "🎬 Videoprojekt"
        self._type_badge.setText(badge_text)
        self._type_badge.setVisible(True)
        if new_config is self._page_config:
            return
        self._page_config = new_config
        self._rebuild_buttons()

    def set_page(self, stack_idx: int) -> None:
        """Highlight the button that corresponds to the given stack index."""
        self._select_by_stack(stack_idx)

    # ------------------------------------------------------------------ internals

    def _on_click(self, stack_idx: int) -> None:
        self._select_by_stack(stack_idx)
        self.page_requested.emit(stack_idx)

    def _select_by_stack(self, stack_idx: int) -> None:
        for btn, si in self._buttons:
            btn.setChecked(si == stack_idx)
