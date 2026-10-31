"""
Review-queue hub: uncertain QA flags, Active Learning candidates, and
low-confidence batch leftovers in one place.
"""
from __future__ import annotations

import os
from typing import List, Optional

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget,
    QListWidget, QListWidgetItem, QPushButton, QAbstractItemView,
)

from utils.i18n import tr


class ReviewQueuePage(QWidget):
    """
    Stack page aggregating items that need human review.

    Signals
    -------
    open_in_labeling(str)
        Absolute image path the user wants to open on the Labeling page.
    """

    open_in_labeling = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._batch_low: List[dict] = []
        self._build_ui()

    def set_project(self, project, audit=None) -> None:
        self.project = project
        self.refresh()

    def set_batch_low_confidence(self, items: List[dict]) -> None:
        """Store last batch low-confidence rows ``{path, confidence, predicted}``."""
        self._batch_low = list(items or [])
        self.refresh()

    def refresh(self) -> None:
        self._fill_uncertain()
        self._fill_al()
        self._fill_batch()
        n = (
            self._uncertain_list.count()
            + self._al_list.count()
            + self._batch_list.count()
        )
        self._summary.setText(tr("review.summary", n=n))

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)

        title = QLabel(tr("review.title"))
        title.setStyleSheet("font-size:18px;font-weight:bold;color:#E6EDF3;")
        root.addWidget(title)

        self._summary = QLabel(tr("review.summary", n=0))
        self._summary.setStyleSheet("color:#8B949E;font-size:12px;")
        root.addWidget(self._summary)

        hint = QLabel(tr("review.hint"))
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#ADBAC7;font-size:11px;padding-bottom:8px;")
        root.addWidget(hint)

        self._tabs = QTabWidget()
        self._uncertain_list = self._make_list()
        self._al_list = self._make_list()
        self._batch_list = self._make_list()
        self._tabs.addTab(self._uncertain_list, tr("review.tab.uncertain"))
        self._tabs.addTab(self._al_list, tr("review.tab.al"))
        self._tabs.addTab(self._batch_list, tr("review.tab.batch"))
        root.addWidget(self._tabs, stretch=1)

        row = QHBoxLayout()
        refresh_btn = QPushButton(tr("review.refresh_btn"))
        refresh_btn.clicked.connect(self.refresh)
        row.addWidget(refresh_btn)
        open_btn = QPushButton(tr("review.open_btn"))
        open_btn.clicked.connect(self._open_selected)
        row.addWidget(open_btn)
        row.addStretch(1)
        root.addLayout(row)

    def _make_list(self) -> QListWidget:
        lw = QListWidget()
        lw.setSelectionMode(QAbstractItemView.SingleSelection)
        lw.itemDoubleClicked.connect(lambda _i: self._open_selected())
        return lw

    def _fill_uncertain(self) -> None:
        self._uncertain_list.clear()
        if not self.project:
            return
        for path in self.project.get_uncertain_images():
            flag = self.project.get_label_flag(path)
            comment = (flag.get("comment") or "").strip()
            label = self.project.image_labels.get(path, "—")
            text = f"{os.path.basename(path)}  [{label}]"
            if comment:
                text += f"  — {comment}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, path)
            self._uncertain_list.addItem(item)

    def _fill_al(self) -> None:
        self._al_list.clear()
        if not self.project:
            return
        for entry in self.project.get_al_queue():
            path = entry.get("path", "")
            conf = entry.get("confidence", 0.0)
            pred = entry.get("predicted_label") or entry.get("predicted") or "?"
            text = f"{os.path.basename(path)}  → {pred}  ({conf:.0%})"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, path)
            self._al_list.addItem(item)

    def _fill_batch(self) -> None:
        self._batch_list.clear()
        for row in self._batch_low:
            path = row.get("path", "")
            conf = float(row.get("confidence", 0.0) or 0.0)
            pred = row.get("predicted", "?")
            text = f"{os.path.basename(path)}  → {pred}  ({conf:.0%})"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, path)
            self._batch_list.addItem(item)

    def _open_selected(self) -> None:
        lw = self._tabs.currentWidget()
        if not isinstance(lw, QListWidget):
            return
        item = lw.currentItem()
        if not item:
            return
        path = item.data(Qt.UserRole)
        if path:
            self.open_in_labeling.emit(str(path))
