"""
Image Labeling Studio – Entry point.
"""
import sys
import os
import traceback

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from gui.main_window import MainWindow
from utils.logging_utils import setup_logging, get_logger
from utils.config import APP_NAME

_LOG_DIR = os.path.join(os.path.expanduser("~"), ".image_labeling_studio", "logs")


def _install_exception_hook(log_dir: str) -> None:
    """Catch unhandled exceptions, write to log, show user-friendly dialog."""
    import logging
    logger = logging.getLogger("ImageLabelingStudio")

    def _hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.critical("Unbehandelter Fehler:\n%s", msg)
        app = QApplication.instance()
        if app:
            dlg = QMessageBox()
            dlg.setIcon(QMessageBox.Critical)
            dlg.setWindowTitle("Unerwarteter Fehler")
            dlg.setText(
                "Ein unerwarteter Fehler ist aufgetreten.\n"
                f"Details wurden in das Fehlerlog geschrieben:\n{log_dir}"
            )
            dlg.setDetailedText(msg)
            dlg.exec()

    sys.excepthook = _hook


def main() -> None:
    log_dir = _LOG_DIR
    setup_logging(log_dir=log_dir)
    _install_exception_hook(log_dir)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("ImageLabelingStudio")
    app.setStyle("Fusion")

    # Dark palette
    from PySide6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(45, 45, 45))
    palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
    palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    import platform
    font = QFont()
    if platform.system() == "Darwin":
        font.setFamily(".AppleSystemUIFont")
    elif platform.system() == "Windows":
        font.setFamily("Segoe UI")
    else:
        font.setFamily("Ubuntu")
    font.setPointSize(13)
    font.setHintingPreference(QFont.PreferFullHinting)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
