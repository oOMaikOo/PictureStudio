"""
ImageDropFilter — install on any QWidget to receive drag-dropped
image files and folders as a list of resolved image paths.

Usage:
    self._drop = ImageDropFilter(some_widget)
    self._drop.files_dropped.connect(self._handle_paths)
"""
import os
from typing import List

from PySide6.QtCore import QObject, QEvent, Signal
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDropEvent

from utils.config import IMAGE_FORMATS

# Styles applied to the target widget during an active drag
_DRAG_OVER_STYLE = (
    "border: 2px dashed #2ECC71 !important;"
    "background: rgba(46, 204, 113, 0.08);"
)
_DRAG_NORMAL_STYLE = ""   # restore (empty = revert to stylesheet)


class ImageDropFilter(QObject):
    """
    Event filter that makes any QWidget a drop target for image files/folders.

    Signals
    -------
    files_dropped(list[str])
        Emitted with a deduplicated, sorted list of absolute image paths.
    """

    files_dropped = Signal(list)

    def __init__(self, widget, parent: QObject = None):
        super().__init__(parent or widget)
        self._widget = widget
        self._saved_style: str = widget.styleSheet()
        widget.setAcceptDrops(True)
        widget.installEventFilter(self)

    # ------------------------------------------------------------------ filter

    def eventFilter(self, obj, event) -> bool:
        t = event.type()
        if t == QEvent.DragEnter:
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
                self._highlight(True)
                return True
        elif t == QEvent.DragMove:
            event.acceptProposedAction()
            return True
        elif t == QEvent.DragLeave:
            self._highlight(False)
            return False
        elif t == QEvent.Drop:
            self._highlight(False)
            paths = self._collect(event.mimeData().urls())
            if paths:
                self.files_dropped.emit(paths)
                event.acceptProposedAction()
            return True
        return False

    # ------------------------------------------------------------------ helpers

    def _highlight(self, active: bool) -> None:
        try:
            if active:
                self._saved_style = self._widget.styleSheet()
                self._widget.setStyleSheet(
                    self._saved_style + ";" + _DRAG_OVER_STYLE
                )
            else:
                self._widget.setStyleSheet(self._saved_style)
        except RuntimeError:
            pass  # widget already deleted

    @staticmethod
    def _collect(urls) -> List[str]:
        seen: set = set()
        paths: List[str] = []
        for url in urls:
            local = url.toLocalFile()
            if not local:
                continue
            if os.path.isdir(local):
                for fname in sorted(os.listdir(local)):
                    if os.path.splitext(fname)[1].lower() in IMAGE_FORMATS:
                        p = os.path.join(local, fname)
                        if p not in seen:
                            seen.add(p)
                            paths.append(p)
            elif os.path.isfile(local):
                if os.path.splitext(local)[1].lower() in IMAGE_FORMATS:
                    if local not in seen:
                        seen.add(local)
                        paths.append(local)
        return paths
