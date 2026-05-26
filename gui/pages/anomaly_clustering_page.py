"""
Anomalie-Clustering-Seite (Stack-Index 11).

Gruppiert Alarm-Bilder automatisch nach visueller Ähnlichkeit, zeigt einen
Cluster-Browser mit repräsentativen Beispielen und ermöglicht den CSV-Export
der Cluster-Zuordnungen.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QSpinBox, QProgressBar,
    QListWidget, QListWidgetItem, QFileDialog,
    QMessageBox, QScrollArea, QFrame, QSplitter,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QPixmap, QFont, QColor

from core.anomaly_clustering import ClusteringThread, AnomalyClustering


# ---------------------------------------------------------------------------
# Small helper: cluster card widget
# ---------------------------------------------------------------------------

class _ClusterCard(QWidget):
    """Compact card showing cluster ID, image count and representative thumbnail."""

    clicked = Signal(int)   # emits cluster_id

    _CARD_STYLE = """
        QWidget#ClusterCard {
            background: #1C2A3A;
            border: 1px solid #2C3E50;
            border-radius: 8px;
        }
        QWidget#ClusterCard:hover {
            border: 1px solid #388BFD;
        }
    """
    _CARD_STYLE_SELECTED = """
        QWidget#ClusterCard {
            background: #1F3A5F;
            border: 2px solid #1F6FEB;
            border-radius: 8px;
        }
    """

    def __init__(self, cluster_id: int, paths: List[str], representative: str, parent=None):
        super().__init__(parent)
        self.setObjectName("ClusterCard")
        self._cluster_id = cluster_id
        self._selected = False
        self._build_ui(cluster_id, paths, representative)
        self.setStyleSheet(self._CARD_STYLE)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedWidth(140)
        self.setFixedHeight(160)

    def _build_ui(self, cluster_id: int, paths: List[str], representative: str) -> None:
        from utils.i18n import tr
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Thumbnail
        self._thumb_lbl = QLabel()
        self._thumb_lbl.setAlignment(Qt.AlignCenter)
        self._thumb_lbl.setFixedSize(120, 90)
        self._thumb_lbl.setStyleSheet("border: none; background: #111;")
        if representative and os.path.isfile(representative):
            pix = QPixmap(representative)
            if not pix.isNull():
                pix = pix.scaled(120, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self._thumb_lbl.setPixmap(pix)
            else:
                self._thumb_lbl.setText("?")
        else:
            self._thumb_lbl.setText("?")
        layout.addWidget(self._thumb_lbl, alignment=Qt.AlignCenter)

        # Cluster number
        id_lbl = QLabel(tr("clustering.cluster_title", n=cluster_id))
        id_lbl.setAlignment(Qt.AlignCenter)
        id_lbl.setStyleSheet("color: #E6EDF3; font-weight: bold; font-size: 11px; border: none;")
        layout.addWidget(id_lbl)

        # Count
        cnt_lbl = QLabel(tr("clustering.images_count", n=len(paths)))
        cnt_lbl.setAlignment(Qt.AlignCenter)
        cnt_lbl.setStyleSheet("color: #8B949E; font-size: 10px; border: none;")
        layout.addWidget(cnt_lbl)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.setStyleSheet(
            self._CARD_STYLE_SELECTED if selected else self._CARD_STYLE
        )

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self._cluster_id)
        super().mousePressEvent(event)

    @property
    def cluster_id(self) -> int:
        return self._cluster_id


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

class AnomalyClusteringPage(QWidget):
    """
    Seite für Anomalie-Clustering (Stack-Index 11).

    Öffentliche API (für Tests und MainWindow):
        AnomalyClusteringPage()            — ohne Argumente instanziierbar
        page.set_project(project)          — nimmt ein Project-Objekt entgegen
        page.btn_start                     — "Clustering starten"-Button
        page.btn_export                    — "CSV exportieren"-Button
        page.spin_clusters                 — Spinbox für Cluster-Anzahl
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._clustering: Optional[AnomalyClustering] = None
        self._thread: Optional[QThread] = None
        self._cluster_cards: Dict[int, _ClusterCard] = {}
        self._selected_cluster_id: Optional[int] = None
        self._alarm_paths_cache: Optional[List[str]] = None
        self._build_ui()

    # ------------------------------------------------------------------ project

    def set_project(self, project, audit=None) -> None:
        """Accept a Project instance and reset the clustering state."""
        self.project = project
        self._clustering = None
        self._cluster_cards.clear()
        self._selected_cluster_id = None
        self._alarm_paths_cache = None
        self._clear_cards()
        self._clear_image_list()
        self._status_lbl.setText("Projekt geladen. Clustering starten, um Bilder zu gruppieren.")
        self.btn_export.setEnabled(False)
        self._update_image_count_label()

    # ------------------------------------------------------------------ UI construction

    def _build_ui(self) -> None:
        from utils.i18n import tr
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # Left panel: controls + cluster cards
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_widget = QWidget()
        left_scroll.setWidget(left_widget)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)
        splitter.addWidget(left_scroll)

        # Title
        title_lbl = QLabel(tr("clustering.title"))
        title_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #E6EDF3;")
        left_layout.addWidget(title_lbl)

        # Image count info
        self._img_count_lbl = QLabel(tr("clustering.no_images_msg"))
        self._img_count_lbl.setStyleSheet("color: #8B949E; font-size: 11px;")
        left_layout.addWidget(self._img_count_lbl)

        # Controls group
        ctrl_group = QGroupBox(tr("clustering.settings_group"))
        ctrl_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #8B949E; border: 1px solid #30363D;"
            " border-radius: 6px; margin-top: 8px; padding-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )
        ctrl_layout = QVBoxLayout(ctrl_group)
        ctrl_layout.setSpacing(8)

        # Cluster count row
        cluster_row = QHBoxLayout()
        cluster_row.addWidget(QLabel(tr("clustering.count_label")))
        self.spin_clusters = QSpinBox()
        self.spin_clusters.setRange(2, 20)
        self.spin_clusters.setValue(5)
        self.spin_clusters.setToolTip("Anzahl der Cluster (2–20)")
        cluster_row.addWidget(self.spin_clusters)
        cluster_row.addStretch()
        ctrl_layout.addLayout(cluster_row)

        # Start button
        self.btn_start = QPushButton(tr("clustering.start_btn"))
        self.btn_start.setFixedHeight(36)
        self.btn_start.setStyleSheet(
            "QPushButton { background: #1F6FEB; color: white; border-radius: 6px;"
            " font-weight: bold; }"
            "QPushButton:hover { background: #388BFD; }"
            "QPushButton:disabled { background: #30363D; color: #484F58; }"
        )
        self.btn_start.clicked.connect(self._start_clustering)
        ctrl_layout.addWidget(self.btn_start)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        ctrl_layout.addWidget(self._progress)

        left_layout.addWidget(ctrl_group)

        # Status label
        self._status_lbl = QLabel(tr("clustering.no_project"))
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet("color: #8B949E; font-size: 11px;")
        left_layout.addWidget(self._status_lbl)

        # Cluster cards area label
        cards_lbl = QLabel("Cluster")
        cards_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #E6EDF3;")
        left_layout.addWidget(cards_lbl)

        # Scrollable area for cluster cards (flow layout via QWidget + wrapping HBoxes)
        self._cards_scroll = QScrollArea()
        self._cards_scroll.setWidgetResizable(True)
        self._cards_scroll.setFrameShape(QFrame.NoFrame)
        self._cards_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._cards_container = QWidget()
        self._cards_layout = _FlowLayout(self._cards_container)
        self._cards_scroll.setWidget(self._cards_container)
        self._cards_scroll.setMinimumHeight(200)
        left_layout.addWidget(self._cards_scroll)

        # Export button
        self.btn_export = QPushButton(tr("common.export_csv"))
        self.btn_export.setFixedHeight(34)
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet(
            "QPushButton { background: #238636; color: white; border-radius: 6px; }"
            "QPushButton:hover { background: #2EA043; }"
            "QPushButton:disabled { background: #30363D; color: #484F58; }"
        )
        self.btn_export.clicked.connect(self._export_csv)
        left_layout.addWidget(self.btn_export)

        left_layout.addStretch()

        # Right panel: image list for selected cluster
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(8)
        splitter.addWidget(right_widget)

        self._detail_title = QLabel(tr("clustering.detail_title"))
        self._detail_title.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #E6EDF3;"
        )
        right_layout.addWidget(self._detail_title)

        self._img_list = QListWidget()
        self._img_list.setStyleSheet(
            "QListWidget { background: #0D1117; border: 1px solid #30363D;"
            " border-radius: 4px; color: #E6EDF3; }"
            "QListWidget::item:selected { background: #1F6FEB; }"
        )
        right_layout.addWidget(self._img_list)

        splitter.setSizes([360, 640])

    # ------------------------------------------------------------------ clustering

    def _get_alarm_paths(self) -> List[str]:
        """Return image paths that are labeled as 'anomalie' / 'alarm' / 'Anomalie' etc."""
        if not self.project:
            return []
        if self._alarm_paths_cache is not None:
            return self._alarm_paths_cache
        alarm_keywords = {"anomalie", "alarm", "fehler", "defekt", "error"}
        paths: List[str] = []
        for path in self.project.images:
            label = (self.project.image_labels.get(path) or "").lower()
            if any(kw in label for kw in alarm_keywords):
                paths.append(path)
        # If no anomaly-labeled images found, fall back to all labeled images
        if not paths:
            paths = [p for p in self.project.images
                     if self.project.image_labels.get(p)]
        # Ultimate fallback: all images
        if not paths:
            paths = list(self.project.images)
        self._alarm_paths_cache = paths
        return paths

    @Slot()
    def _start_clustering(self) -> None:
        """Start the ClusteringThread on alarm-labeled images."""
        from utils.i18n import tr
        if not self.project:
            QMessageBox.warning(self, tr("common.warning"), tr("clustering.no_project"))
            return

        paths = self._get_alarm_paths()
        if not paths:
            QMessageBox.information(
                self, tr("common.info"),
                tr("clustering.no_images_msg")
            )
            return

        n_clusters = min(self.spin_clusters.value(), max(2, len(paths)))
        self.spin_clusters.setMaximum(max(2, min(20, len(paths))))
        if self._thread and self._thread.isRunning():
            return  # already running

        self.btn_start.setEnabled(False)
        self.btn_export.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status_lbl.setText(tr("clustering.in_progress", n=len(paths)))
        self._clear_cards()
        self._clear_image_list()

        try:
            self._thread = ClusteringThread(paths, n_clusters, parent=self)
            self._thread.progress.connect(self._on_progress)
            self._thread.finished.connect(self._on_finished)
            self._thread.error.connect(self._on_error)
            self._thread.start()
        except Exception as exc:
            self._thread = None
            self._on_error(str(exc))

    @Slot(int, int)
    def _on_progress(self, current: int, total: int) -> None:
        if total > 0:
            pct = int(current / total * 100)
            self._progress.setValue(pct)

    @Slot(dict)
    def _on_finished(self, clusters: dict) -> None:
        from utils.i18n import tr
        if self._thread and hasattr(self._thread, "clustering"):
            self._clustering = self._thread.clustering
        self._thread = None
        self._progress.setValue(100)
        self._progress.setVisible(False)
        self.btn_start.setEnabled(True)
        if not clusters:
            self._status_lbl.setText(tr("clustering.failed"))
            return
        total_images = sum(len(v) for v in clusters.values())
        self._status_lbl.setText(
            tr("clustering.done", clusters=len(clusters), images=total_images)
        )
        self.btn_export.setEnabled(True)
        self._build_cards(clusters)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        from utils.i18n import tr
        self._thread = None
        self._progress.setVisible(False)
        self.btn_start.setEnabled(True)
        self._status_lbl.setText(f"Fehler: {msg}")
        QMessageBox.critical(self, tr("common.error"), msg)

    def hideEvent(self, event) -> None:
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)
        super().hideEvent(event)

    # ------------------------------------------------------------------ cards

    def _clear_cards(self) -> None:
        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._cluster_cards.clear()
        self._selected_cluster_id = None

    def _build_cards(self, clusters: dict) -> None:
        self._clear_cards()
        for cid in sorted(clusters.keys()):
            paths = clusters[cid]
            rep = ""
            if self._clustering:
                rep = self._clustering.get_representative(cid)
            card = _ClusterCard(cid, paths, rep, parent=self._cards_container)
            card.clicked.connect(self._on_card_clicked)
            self._cluster_cards[cid] = card
            self._cards_layout.addWidget(card)

    @Slot(int)
    def _on_card_clicked(self, cluster_id: int) -> None:
        # Deselect previously selected card
        if self._selected_cluster_id is not None:
            prev_card = self._cluster_cards.get(self._selected_cluster_id)
            if prev_card:
                prev_card.set_selected(False)

        self._selected_cluster_id = cluster_id
        card = self._cluster_cards.get(cluster_id)
        if card:
            card.set_selected(True)

        if not self._clustering:
            return

        paths = self._clustering.clusters.get(cluster_id, [])
        self._detail_title.setText(f"Cluster {cluster_id} – {len(paths)} Bilder")
        self._clear_image_list()
        rep = self._clustering.get_representative(cluster_id)
        for path in paths:
            item = QListWidgetItem(os.path.basename(path))
            item.setToolTip(path)
            if path == rep:
                item.setForeground(QColor("#F39C12"))
                item.setText(f"★ {os.path.basename(path)}")
            self._img_list.addItem(item)

    def _clear_image_list(self) -> None:
        from utils.i18n import tr
        self._img_list.clear()
        self._detail_title.setText(tr("clustering.detail_title"))

    # ------------------------------------------------------------------ export

    @Slot()
    def _export_csv(self) -> None:
        from utils.i18n import tr
        if not self._clustering:
            QMessageBox.information(self, tr("common.info"), tr("clustering.start_btn"))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, tr("common.export_csv"), "cluster_export.csv",
            "CSV-Datei (*.csv);;Alle Dateien (*)"
        )
        if not path:
            return
        try:
            self._clustering.export_csv(path)
            QMessageBox.information(
                self, tr("common.info"),
                tr("clustering.export_success", path=path)
            )
            self._status_lbl.setText(f"CSV gespeichert: {os.path.basename(path)}")
        except Exception as exc:
            QMessageBox.critical(self, tr("common.error"), str(exc))

    # ------------------------------------------------------------------ helpers

    def _update_image_count_label(self) -> None:
        if not self.project:
            self._img_count_lbl.setText("Kein Projekt geladen.")
            return
        total = len(self.project.images)
        alarm_count = len(self._get_alarm_paths())
        self._img_count_lbl.setText(
            f"{alarm_count} Bilder für Clustering verfügbar (gesamt: {total})"
        )


# ---------------------------------------------------------------------------
# Simple flow layout (wraps widgets like CSS flex-wrap)
# ---------------------------------------------------------------------------

from PySide6.QtWidgets import QLayout
from PySide6.QtCore import QRect, QPoint, QSize


class _FlowLayout(QLayout):
    """A simple flow layout that wraps widgets left-to-right."""

    def __init__(self, parent=None, h_spacing: int = 8, v_spacing: int = 8):
        super().__init__(parent)
        self._items = []
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        m = self.contentsMargins()
        x = rect.x() + m.left()
        y = rect.y() + m.top()
        line_height = 0
        right = rect.right() - m.right()

        for item in self._items:
            w = item.sizeHint()
            next_x = x + w.width()
            if next_x > right and line_height > 0:
                x = rect.x() + m.left()
                y += line_height + self._v_spacing
                next_x = x + w.width()
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), w))
            x = next_x + self._h_spacing
            line_height = max(line_height, w.height())

        return y + line_height - rect.y() + m.bottom()
