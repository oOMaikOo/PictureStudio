"""
Modern dark theme: palette + stylesheet applied globally.
"""
import platform


DARK_QSS = """
/* ── MenuBar ─────────────────────────────────────────────────────── */
QMenuBar {
    background: #333C4E;
    color: #E6EDF3;
    border-bottom: 1px solid #4B5465;
    padding: 2px 4px;
    spacing: 2px;
}
QMenuBar::item {
    background: transparent;
    padding: 4px 10px;
    border-radius: 4px;
}
QMenuBar::item:selected { background: #3D4657; }
QMenuBar::item:pressed  { background: #1F6FEB; color: white; }

/* ── Menu ────────────────────────────────────────────────────────── */
QMenu {
    background: #3D4657;
    color: #E6EDF3;
    border: 1px solid #4B5465;
    border-radius: 6px;
    padding: 4px;
}
QMenu::item { padding: 6px 20px; border-radius: 4px; }
QMenu::item:selected { background: #4B5465; }
QMenu::separator { height: 1px; background: #4B5465; margin: 4px 8px; }

/* ── ToolTip ─────────────────────────────────────────────────────── */
QToolTip {
    background: #3D4657;
    color: #E6EDF3;
    border: 1px solid #4B5465;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 11px;
}

/* ── StatusBar ───────────────────────────────────────────────────── */
QStatusBar {
    background: #333C4E;
    color: #8B949E;
    border-top: 1px solid #4B5465;
    font-size: 11px;
}
QStatusBar::item { border: none; }

/* ── StackedWidget ───────────────────────────────────────────────── */
QStackedWidget { border-left: 1px solid #4B5465; }

/* ── GroupBox ────────────────────────────────────────────────────── */
QGroupBox {
    background: #333C4E;
    border: 1px solid #4B5465;
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
    background: #3D4657;
    color: #E6EDF3;
    border: 1px solid #4B5465;
    border-radius: 6px;
    padding: 5px 14px;
    min-height: 28px;
    font-size: 12px;
}
QPushButton:hover { background: #4B5465; border-color: #8B949E; }
QPushButton:pressed { background: #28303F; }
QPushButton:checked {
    background: #1F6FEB;
    border-color: #388BFD;
    color: white;
    font-weight: bold;
}
QPushButton:disabled { color: #545D68; background: #333C4E; border-color: #3D4657; }

/* ── Inputs ──────────────────────────────────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit {
    background: #28303F;
    color: #E6EDF3;
    border: 1px solid #4B5465;
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
    background: #333C4E;
}

/* ── SpinBox ─────────────────────────────────────────────────────── */
QSpinBox, QDoubleSpinBox {
    background: #28303F;
    color: #E6EDF3;
    border: 1px solid #4B5465;
    border-radius: 6px;
    padding: 4px 6px;
    min-height: 26px;
}
QSpinBox:focus, QDoubleSpinBox:focus { border-color: #388BFD; }
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background: #3D4657;
    border: none;
    width: 18px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: #4B5465;
}

/* ── ComboBox ────────────────────────────────────────────────────── */
QComboBox {
    background: #28303F;
    color: #E6EDF3;
    border: 1px solid #4B5465;
    border-radius: 6px;
    padding: 4px 10px;
    min-height: 28px;
}
QComboBox:focus { border-color: #388BFD; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView {
    background: #3D4657;
    color: #E6EDF3;
    border: 1px solid #4B5465;
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
    border: 1px solid #545D68;
    border-radius: 3px;
    background: #28303F;
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
    background: #3D4657;
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
    background: #3D4657;
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
    background: #333C4E;
    border: 1px solid #4B5465;
    border-radius: 0 6px 6px 6px;
    top: -1px;
}
QTabBar {
    background: transparent;
}
QTabBar::tab {
    background: #28303F;
    color: #8B949E;
    padding: 7px 16px;
    border: 1px solid #4B5465;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    margin-right: 2px;
    font-size: 12px;
}
QTabBar::tab:selected {
    background: #333C4E;
    color: #E6EDF3;
    border-bottom: 2px solid #1F6FEB;
    font-weight: bold;
}
QTabBar::tab:hover:!selected { background: #3D4657; color: #C9D1D9; }

/* ── Tables ──────────────────────────────────────────────────────── */
QTableWidget, QTableView {
    background: #333C4E;
    alternate-background-color: #3D4657;
    color: #E6EDF3;
    border: 1px solid #4B5465;
    border-radius: 6px;
    gridline-color: #3D4657;
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
    background: #3D4657;
    color: #8B949E;
    padding: 6px 8px;
    border: none;
    border-bottom: 1px solid #4B5465;
    border-right: 1px solid #4B5465;
    font-size: 11px;
    font-weight: bold;
}
QHeaderView::section:first { border-top-left-radius: 5px; }
QHeaderView::section:last  { border-top-right-radius: 5px; border-right: none; }

/* ── Lists ───────────────────────────────────────────────────────── */
QListWidget {
    background: #333C4E;
    color: #E6EDF3;
    border: 1px solid #4B5465;
    border-radius: 6px;
    outline: none;
}
QListWidget::item { padding: 4px 8px; border-radius: 4px; }
QListWidget::item:selected { background: #1F6FEB; color: white; }
QListWidget::item:hover:!selected { background: #3D4657; }

/* ── TreeWidget ──────────────────────────────────────────────────── */
QTreeWidget, QTreeView {
    background: #333C4E;
    color: #E6EDF3;
    border: 1px solid #4B5465;
    border-radius: 6px;
    outline: none;
}
QTreeWidget::item:selected, QTreeView::item:selected { background: #1F6FEB; }
QTreeWidget::item:hover,    QTreeView::item:hover    { background: #3D4657; }

/* ── ScrollBars ──────────────────────────────────────────────────── */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #4B5465;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #606878; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #4B5465;
    border-radius: 4px;
    min-width: 24px;
}
QScrollBar::handle:horizontal:hover { background: #606878; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

/* ── Splitter ────────────────────────────────────────────────────── */
QSplitter::handle { background: #4B5465; }
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
    bg        = QColor(40,  48,  63)   # #28303F  softer dark navy
    surface   = QColor(51,  60,  78)   # #333C4E
    elevated  = QColor(61,  70,  87)   # #3D4657
    text_hi   = QColor(173, 186, 199)  # #ADBAC7
    text_lo   = QColor(118, 131, 144)  # #768390
    disabled  = QColor(84,  93,  104)  # #545D68
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
    palette.setColor(QPalette.Button,          QColor(45, 54, 71))
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
