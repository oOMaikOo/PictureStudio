"""
Modern dark theme: palette + stylesheet applied globally.
"""
import platform


DARK_QSS = """
/* ── MenuBar ─────────────────────────────────────────────────────── */
QMenuBar {
    background: #161B22;
    color: #E6EDF3;
    border-bottom: 1px solid #30363D;
    padding: 2px 4px;
    spacing: 2px;
}
QMenuBar::item {
    background: transparent;
    padding: 4px 10px;
    border-radius: 4px;
}
QMenuBar::item:selected { background: #21262D; }
QMenuBar::item:pressed  { background: #1F6FEB; color: white; }

/* ── Menu ────────────────────────────────────────────────────────── */
QMenu {
    background: #1C2128;
    color: #E6EDF3;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 4px;
}
QMenu::item { padding: 6px 20px; border-radius: 4px; }
QMenu::item:selected { background: #21262D; }
QMenu::separator { height: 1px; background: #30363D; margin: 4px 8px; }

/* ── ToolTip ─────────────────────────────────────────────────────── */
QToolTip {
    background: #1C2128;
    color: #E6EDF3;
    border: 1px solid #30363D;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 11px;
}

/* ── StatusBar ───────────────────────────────────────────────────── */
QStatusBar {
    background: #161B22;
    color: #8B949E;
    border-top: 1px solid #30363D;
    font-size: 11px;
}
QStatusBar::item { border: none; }

/* ── StackedWidget ───────────────────────────────────────────────── */
QStackedWidget { border-left: 1px solid #30363D; }

/* ── GroupBox ────────────────────────────────────────────────────── */
QGroupBox {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    margin-top: 16px;
    padding-top: 8px;
    color: #8B949E;
    font-size: 11px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: -1px;
    padding: 0 6px;
    color: #8B949E;
}

/* ── Buttons ─────────────────────────────────────────────────────── */
QPushButton {
    background: #21262D;
    color: #E6EDF3;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 5px 14px;
    min-height: 28px;
    font-size: 12px;
}
QPushButton:hover { background: #30363D; border-color: #8B949E; }
QPushButton:pressed { background: #0D1117; }
QPushButton:checked {
    background: #1F6FEB;
    border-color: #388BFD;
    color: white;
    font-weight: bold;
}
QPushButton:disabled { color: #484F58; background: #161B22; border-color: #21262D; }

/* ── Inputs ──────────────────────────────────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit {
    background: #0D1117;
    color: #E6EDF3;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 4px 8px;
    selection-background-color: #1F6FEB;
    selection-color: white;
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #388BFD;
}
QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {
    color: #8B949E;
    background: #161B22;
}

/* ── SpinBox ─────────────────────────────────────────────────────── */
QSpinBox, QDoubleSpinBox {
    background: #0D1117;
    color: #E6EDF3;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 4px 6px;
    min-height: 26px;
}
QSpinBox:focus, QDoubleSpinBox:focus { border-color: #388BFD; }
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background: #21262D;
    border: none;
    width: 18px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: #30363D;
}

/* ── ComboBox ────────────────────────────────────────────────────── */
QComboBox {
    background: #0D1117;
    color: #E6EDF3;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 4px 10px;
    min-height: 28px;
}
QComboBox:focus { border-color: #388BFD; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView {
    background: #1C2128;
    color: #E6EDF3;
    border: 1px solid #30363D;
    border-radius: 4px;
    selection-background-color: #1F6FEB;
    outline: none;
    padding: 2px;
}

/* ── CheckBox / RadioButton ──────────────────────────────────────── */
QCheckBox, QRadioButton {
    color: #E6EDF3;
    background: transparent;
    spacing: 6px;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #484F58;
    border-radius: 3px;
    background: #0D1117;
}
QCheckBox::indicator:checked {
    background: #1F6FEB;
    border-color: #388BFD;
}
QCheckBox::indicator:hover { border-color: #8B949E; }
QRadioButton::indicator { border-radius: 8px; }
QRadioButton::indicator:checked { background: #1F6FEB; border-color: #388BFD; }

/* ── Slider ──────────────────────────────────────────────────────── */
QSlider::groove:horizontal {
    background: #21262D;
    height: 4px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #388BFD;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover { background: #1F6FEB; }
QSlider::sub-page:horizontal { background: #1F6FEB; border-radius: 2px; }

/* ── ProgressBar ─────────────────────────────────────────────────── */
QProgressBar {
    background: #21262D;
    border: none;
    border-radius: 4px;
    min-height: 8px;
    max-height: 8px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1F6FEB, stop:1 #388BFD);
    border-radius: 4px;
}

/* ── TabWidget ───────────────────────────────────────────────────── */
QTabWidget::pane {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 0 6px 6px 6px;
    top: -1px;
}
QTabBar {
    background: transparent;
}
QTabBar::tab {
    background: #0D1117;
    color: #8B949E;
    padding: 7px 16px;
    border: 1px solid #30363D;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    margin-right: 2px;
    font-size: 12px;
}
QTabBar::tab:selected {
    background: #161B22;
    color: #E6EDF3;
    border-bottom: 2px solid #1F6FEB;
    font-weight: bold;
}
QTabBar::tab:hover:!selected { background: #1C2128; color: #C9D1D9; }

/* ── Tables ──────────────────────────────────────────────────────── */
QTableWidget, QTableView {
    background: #161B22;
    alternate-background-color: #1C2128;
    color: #E6EDF3;
    border: 1px solid #30363D;
    border-radius: 6px;
    gridline-color: #21262D;
    outline: none;
}
QTableWidget::item, QTableView::item {
    padding: 4px 8px;
    border: none;
}
QTableWidget::item:selected, QTableView::item:selected {
    background: #1F6FEB;
    color: white;
}
QHeaderView::section {
    background: #21262D;
    color: #8B949E;
    padding: 6px 8px;
    border: none;
    border-bottom: 1px solid #30363D;
    border-right: 1px solid #30363D;
    font-size: 11px;
    font-weight: bold;
}
QHeaderView::section:first { border-top-left-radius: 5px; }
QHeaderView::section:last  { border-top-right-radius: 5px; border-right: none; }

/* ── Lists ───────────────────────────────────────────────────────── */
QListWidget {
    background: #161B22;
    color: #E6EDF3;
    border: 1px solid #30363D;
    border-radius: 6px;
    outline: none;
}
QListWidget::item { padding: 4px 8px; border-radius: 4px; }
QListWidget::item:selected { background: #1F6FEB; color: white; }
QListWidget::item:hover:!selected { background: #21262D; }

/* ── TreeWidget ──────────────────────────────────────────────────── */
QTreeWidget, QTreeView {
    background: #161B22;
    color: #E6EDF3;
    border: 1px solid #30363D;
    border-radius: 6px;
    outline: none;
}
QTreeWidget::item:selected, QTreeView::item:selected { background: #1F6FEB; }
QTreeWidget::item:hover,    QTreeView::item:hover    { background: #21262D; }

/* ── ScrollBars ──────────────────────────────────────────────────── */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #30363D;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #484F58; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #30363D;
    border-radius: 4px;
    min-width: 24px;
}
QScrollBar::handle:horizontal:hover { background: #484F58; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

/* ── Splitter ────────────────────────────────────────────────────── */
QSplitter::handle { background: #30363D; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical   { height: 1px; }
QSplitter::handle:hover { background: #388BFD; }

/* ── ScrollArea ──────────────────────────────────────────────────── */
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
"""


def apply_dark_theme(app) -> None:
    """Apply palette + QSS to the application. Call once on startup."""
    from PySide6.QtGui import QPalette, QColor, QFont

    palette = QPalette()
    bg        = QColor(13,  17,  23)   # #0D1117
    surface   = QColor(22,  27,  34)   # #161B22
    elevated  = QColor(28,  33,  40)   # #1C2128
    text_hi   = QColor(230, 237, 243)  # #E6EDF3
    text_lo   = QColor(139, 148, 158)  # #8B949E
    disabled  = QColor(72,  79,  88)   # #484F58
    primary   = QColor(31,  111, 235)  # #1F6FEB
    bright_r  = QColor(248, 81,  73)   # #F85149
    link      = QColor(56,  139, 253)  # #388BFD

    palette.setColor(QPalette.Window,          bg)
    palette.setColor(QPalette.WindowText,      text_hi)
    palette.setColor(QPalette.Base,            surface)
    palette.setColor(QPalette.AlternateBase,   elevated)
    palette.setColor(QPalette.ToolTipBase,     elevated)
    palette.setColor(QPalette.ToolTipText,     text_hi)
    palette.setColor(QPalette.Text,            text_hi)
    palette.setColor(QPalette.Button,          QColor(33, 38, 45))
    palette.setColor(QPalette.ButtonText,      text_hi)
    palette.setColor(QPalette.BrightText,      bright_r)
    palette.setColor(QPalette.Link,            link)
    palette.setColor(QPalette.Highlight,       primary)
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.Disabled, QPalette.Text,       disabled)
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, disabled)
    palette.setColor(QPalette.Disabled, QPalette.WindowText, disabled)
    app.setPalette(palette)

    font = QFont()
    if platform.system() == "Darwin":
        font.setFamily(".AppleSystemUIFont")
    elif platform.system() == "Windows":
        font.setFamily("Segoe UI")
    font.setPointSize(12)
    app.setFont(font)

    app.setStyleSheet(DARK_QSS)
