"""
Lazy-loading thumbnail list using QThreadPool.
"""
import os
from collections import OrderedDict
from typing import Dict, Optional, List

from PySide6.QtWidgets import (
    QListWidget, QListWidgetItem, QAbstractItemView, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QRunnable, QThreadPool, QObject, QSize
from PySide6.QtGui import QPixmap, QIcon, QColor

from utils.config import IMAGE_FORMATS


class _LRUPixmapCache:
    """OrderedDict-based LRU cache for QPixmap thumbnails with a fixed capacity."""

    def __init__(self, maxsize: int = 500) -> None:
        self._d: OrderedDict = OrderedDict()
        self._max = maxsize

    def __contains__(self, key: str) -> bool:
        return key in self._d

    def __getitem__(self, key: str):
        self._d.move_to_end(key)
        return self._d[key]

    def __setitem__(self, key: str, value) -> None:
        if key in self._d:
            self._d.move_to_end(key)
        self._d[key] = value
        while len(self._d) > self._max:
            self._d.popitem(last=False)

    def clear(self) -> None:
        self._d.clear()

    def __len__(self) -> int:
        return len(self._d)


class ThumbnailSignals(QObject):
    """Carrier object so ``ThumbnailLoader`` (a QRunnable) can emit Qt signals."""

    loaded = Signal(str, QPixmap)  # path, pixmap


class ThumbnailLoader(QRunnable):
    """Loads one thumbnail in a background thread."""

    def __init__(self, image_path: str, size: int = 100):
        """
        Parameters
        ----------
        image_path : Absolute path to the image file to load.
        size       : Maximum width (and 3/4 of that as height) for the scaled thumbnail.
        """
        super().__init__()
        self.image_path = image_path
        self.size = size
        self.signals = ThumbnailSignals()
        self.setAutoDelete(True)

    def run(self) -> None:
        """Load, scale, and emit the thumbnail pixmap; silently skip on error."""
        try:
            pix = QPixmap(self.image_path)
            if not pix.isNull():
                pix = pix.scaled(self.size, self.size * 3 // 4,
                                 Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.signals.loaded.emit(self.image_path, pix)
        except Exception:
            pass


class LazyThumbnailList(QListWidget):
    """
    QListWidget that loads thumbnails lazily via a thread pool.
    Emits:
      image_selected(path: str)
    """

    image_selected   = Signal(str)        # single selection
    selection_changed = Signal(list)      # list[str] of all selected paths

    def __init__(self, thumb_size: int = 100, parent=None):
        """
        Parameters
        ----------
        thumb_size : Width in pixels for each thumbnail icon; height is 3/4 of this.
        """
        super().__init__(parent)
        self.thumb_size = thumb_size
        self._pool = QThreadPool.globalInstance()
        self._pool.setMaxThreadCount(4)
        self._items: Dict[str, QListWidgetItem] = {}   # path -> item
        self._cache: _LRUPixmapCache = _LRUPixmapCache(500)
        self._pending: set = set()

        self.setIconSize(QSize(thumb_size, thumb_size * 3 // 4))
        self.setSpacing(2)
        self.setResizeMode(QListWidget.Adjust)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.itemSelectionChanged.connect(self._on_selection_changed)

    # ------------------------------------------------------------------ public

    def add_image(self, image_path: str, label: str = "", label_color: str = "") -> None:
        """
        Add *image_path* to the list, loading its thumbnail asynchronously.

        If the path is already present the call is a no-op. The thumbnail is
        served from the in-memory cache when available, otherwise a
        ``ThumbnailLoader`` task is submitted to the global thread pool.
        """
        if image_path in self._items:
            return
        fname = os.path.basename(image_path)
        item = QListWidgetItem(fname + (f"\n[{label}]" if label else ""))
        item.setData(Qt.UserRole, image_path)
        item.setData(Qt.UserRole + 1, label)
        if label_color:
            item.setForeground(QColor(label_color))
        self.addItem(item)
        self._items[image_path] = item

        if image_path in self._cache:
            item.setIcon(QIcon(self._cache[image_path]))
        elif image_path not in self._pending:
            self._pending.add(image_path)
            loader = ThumbnailLoader(image_path, self.thumb_size)
            loader.signals.loaded.connect(self._on_thumbnail_loaded)
            self._pool.start(loader)

    def update_label(self, image_path: str, label: str, color: str = "") -> None:
        """Update the displayed label text and foreground colour for *image_path*."""
        item = self._items.get(image_path)
        if item:
            fname = os.path.basename(image_path)
            item.setText(fname + (f"\n[{label}]" if label else ""))
            item.setData(Qt.UserRole + 1, label)
            if color:
                item.setForeground(QColor(color))

    def update_flag(self, image_path: str, uncertain: bool) -> None:
        """
        Set or clear the QA-uncertain visual indicator for *image_path*.

        When *uncertain* is True the item background turns dark orange and a
        tooltip is added; clearing restores a transparent background.
        """
        item = self._items.get(image_path)
        if item:
            if uncertain:
                item.setBackground(QColor(61, 32, 0))
                item.setToolTip("Label unsicher – benötigt Review")
            else:
                item.setBackground(QColor(0, 0, 0, 0))
                item.setToolTip("")

    def remove_image(self, image_path: str) -> None:
        """Remove the item for *image_path* from the list. No-op if not present."""
        item = self._items.pop(image_path, None)
        if item:
            row = self.row(item)
            if row >= 0:
                self.takeItem(row)

    def clear_all(self) -> None:
        """Remove all items and flush the thumbnail cache and pending-load set."""
        self.clear()
        self._items.clear()
        self._cache.clear()
        self._pending.clear()

    def filter(
        self,
        text: str = "",
        label_filter: str = "",
        label_set: set = None,
        only_unlabeled: bool = False,
        roi_paths: set = None,
        uncertain_paths: set = None,
    ) -> None:
        """
        Show/hide items.

        Parameters
        ----------
        text            : substring search on filename + label (case-insensitive)
        label_filter    : single label match (legacy; ignored when label_set given)
        label_set       : set of labels to show (OR logic); None = show all
        only_unlabeled  : show only items with no label
        roi_paths       : when given, show only paths in this set
        uncertain_paths : when given, show only paths in this set (uncertain-flagged)
        """
        q = text.lower()
        for path, item in self._items.items():
            item_label = item.data(Qt.UserRole + 1) or ""
            hide = False
            if q and q not in os.path.basename(path).lower() and q not in item_label.lower():
                hide = True
            if label_set is not None:
                if item_label not in label_set:
                    hide = True
            elif label_filter and item_label != label_filter:
                hide = True
            if only_unlabeled and item_label:
                hide = True
            if roi_paths is not None and path not in roi_paths:
                hide = True
            if uncertain_paths is not None and path not in uncertain_paths:
                hide = True
            item.setHidden(hide)

    def sort_items(self, key_func, reverse: bool = False) -> None:
        """Re-order list items according to key_func(path) without clearing the cache."""
        paths = sorted(self._items.keys(), key=key_func, reverse=reverse)
        for i, path in enumerate(paths):
            item = self._items[path]
            row  = self.row(item)
            if row != i:
                taken = self.takeItem(row)
                self.insertItem(i, taken)

    def get_selected_path(self) -> str:
        """Return the image path of the currently focused item, or ``""``."""
        item = self.currentItem()
        return item.data(Qt.UserRole) if item else ""

    def get_selected_paths(self) -> List[str]:
        """Return image paths for all highlighted items (multi-select)."""
        return [
            item.data(Qt.UserRole)
            for item in self.selectedItems()
            if item.data(Qt.UserRole)
        ]

    def select_path(self, image_path: str) -> None:
        """Programmatically focus and select the item for *image_path*."""
        item = self._items.get(image_path)
        if item:
            self.setCurrentItem(item)

    def get_all_paths(self) -> List[str]:
        """Return all image paths currently in the list (visible or hidden)."""
        return list(self._items.keys())

    def count_visible(self) -> int:
        """Return the number of items that are not hidden by the current filter."""
        return sum(1 for i in range(self.count()) if not self.item(i).isHidden())

    # ------------------------------------------------------------------ slots

    def _on_thumbnail_loaded(self, path: str, pix: QPixmap) -> None:
        """Store the finished pixmap in the cache and apply it to the list item."""
        self._cache[path] = pix
        self._pending.discard(path)
        item = self._items.get(path)
        if item:
            item.setIcon(QIcon(pix))

    def _on_selection_changed(self) -> None:
        """Emit ``selection_changed`` and, for single selections, ``image_selected``."""
        paths = self.get_selected_paths()
        self.selection_changed.emit(paths)
        # Single-selection path: load the image in the editor
        if len(paths) == 1:
            self.image_selected.emit(paths[0])
