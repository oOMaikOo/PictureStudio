from __future__ import annotations
import logging
import os
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QProgressBar, QListWidget, QScrollArea, QFrame,
    QSplitter, QFormLayout, QMessageBox,
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont

log = logging.getLogger(__name__)


class DatasetStatsPage(QWidget):
    """
    Datensatz-Analyse-Seite (Stack-Index wird von MainWindow vergeben).
    Zeigt: Klassenverteilung, Dateiformat-Statistik, Bildgrößen, Duplikate, Label-Rate.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.project = None
        self._build_ui()

    def set_project(self, project, audit=None) -> None:
        self.project = project
        self.refresh()

    def refresh(self) -> None:
        """Neu berechnen und alle Widgets aktualisieren."""
        try:
            if not self.project:
                self._set_empty_state()
                return
            self._update_label_distribution()
            self._update_format_stats()
            self._update_size_stats()
            self._update_label_rate()
            self._dup_list.clear()
            self._dup_status_lbl.setText("Duplikate noch nicht gesucht.")
        except Exception as exc:
            log.error("Fehler in DatasetStatsPage.refresh: %s", exc)

    def _build_ui(self) -> None:
        from utils.i18n import tr
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel(tr("dataset_stats.title"))
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #E6EDF3;")
        root.addWidget(title)

        btn_row = QHBoxLayout()
        refresh_btn = QPushButton(tr("common.refresh"))
        refresh_btn.setStyleSheet("background: #1F6FEB; color: white; border-radius: 4px; padding: 5px 12px;")
        refresh_btn.clicked.connect(self.refresh)
        btn_row.addWidget(refresh_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(10)
        scroll.setWidget(container)
        root.addWidget(scroll)

        lbl_grp = QGroupBox(tr("dataset_stats.label_dist"))
        lbl_grp.setStyleSheet("QGroupBox { font-weight: bold; color: #8B949E; border: 1px solid #30363D; border-radius: 6px; margin-top: 8px; padding-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 8px; }")
        self._lbl_layout = QVBoxLayout(lbl_grp)
        self._lbl_no_labels = QLabel(tr("dataset_stats.no_labels"))
        self._lbl_no_labels.setStyleSheet("color: #8B949E;")
        self._lbl_layout.addWidget(self._lbl_no_labels)
        layout.addWidget(lbl_grp)

        stats_grp = QGroupBox(tr("dataset_stats.general_group"))
        stats_grp.setStyleSheet(lbl_grp.styleSheet())
        sf = QFormLayout(stats_grp)
        self._total_lbl = QLabel("–")
        self._labeled_lbl = QLabel("–")
        self._label_rate_lbl = QLabel("–")
        self._format_lbl = QLabel("–")
        self._size_lbl = QLabel("–")
        for caption, widget in [
            (tr("dataset_stats.total_images"), self._total_lbl),
            (tr("dataset_stats.labeled"), self._labeled_lbl),
            (tr("dataset_stats.label_rate"), self._label_rate_lbl),
            (tr("dataset_stats.formats"), self._format_lbl),
            (tr("dataset_stats.avg_res"), self._size_lbl),
        ]:
            sf.addRow(QLabel(caption), widget)
            widget.setStyleSheet("color: #E6EDF3;")
        layout.addWidget(stats_grp)

        dup_grp = QGroupBox(tr("dataset_stats.dup_group"))
        dup_grp.setStyleSheet(lbl_grp.styleSheet())
        dv = QVBoxLayout(dup_grp)
        dup_btn_row = QHBoxLayout()
        self._dup_btn = QPushButton(tr("dataset_stats.find_dups_btn"))
        self._dup_btn.setStyleSheet("background: #238636; color: white; border-radius: 4px; padding: 5px 12px;")
        self._dup_btn.clicked.connect(self._find_duplicates)
        dup_btn_row.addWidget(self._dup_btn)
        dup_btn_row.addStretch()
        dv.addLayout(dup_btn_row)
        self._dup_status_lbl = QLabel("Duplikate noch nicht gesucht.")
        self._dup_status_lbl.setStyleSheet("color: #8B949E; font-size: 11px;")
        dv.addWidget(self._dup_status_lbl)
        self._dup_list = QListWidget()
        self._dup_list.setStyleSheet("QListWidget { background: #0D1117; border: 1px solid #30363D; color: #E6EDF3; border-radius: 4px; } QListWidget::item:selected { background: #1F6FEB; }")
        self._dup_list.setMaximumHeight(150)
        dv.addWidget(self._dup_list)
        layout.addWidget(dup_grp)
        layout.addStretch()

    def _set_empty_state(self) -> None:
        self._total_lbl.setText("–")
        self._labeled_lbl.setText("–")
        self._label_rate_lbl.setText("–")
        self._format_lbl.setText("–")
        self._size_lbl.setText("–")

    def _update_label_distribution(self) -> None:
        while self._lbl_layout.count():
            item = self._lbl_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        counts: dict = {}
        for path in self.project.images:
            lbl = self.project.image_labels.get(path, "")
            if lbl:
                counts[lbl] = counts.get(lbl, 0) + 1

        if not counts:
            from utils.i18n import tr
            lbl = QLabel(tr("dataset_stats.no_labels"))
            lbl.setStyleSheet("color: #8B949E;")
            self._lbl_layout.addWidget(lbl)
            return

        total = sum(counts.values())
        color_map = {name: d.get("color", "#1F6FEB") for name, d in self.project.labels.items()} if self.project.labels else {}

        for lbl_name, count in sorted(counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            row = QHBoxLayout()
            name_lbl = QLabel(f"{lbl_name}")
            name_lbl.setFixedWidth(120)
            name_lbl.setStyleSheet("color: #E6EDF3; font-size: 11px;")
            row.addWidget(name_lbl)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(pct))
            bar.setFixedHeight(16)
            bar.setTextVisible(False)
            color = color_map.get(lbl_name, "#1F6FEB")
            bar.setStyleSheet(f"QProgressBar {{ background: #21262D; border-radius: 3px; }} QProgressBar::chunk {{ background: {color}; border-radius: 3px; }}")
            row.addWidget(bar, stretch=1)
            cnt_lbl = QLabel(f"{count} ({pct:.1f}%)")
            cnt_lbl.setFixedWidth(90)
            cnt_lbl.setStyleSheet("color: #8B949E; font-size: 11px;")
            row.addWidget(cnt_lbl)
            w = QWidget()
            w.setLayout(row)
            self._lbl_layout.addWidget(w)

    def _update_format_stats(self) -> None:
        fmt_counts: dict = {}
        for p in self.project.images:
            ext = os.path.splitext(p)[1].lower() or "(kein)"
            fmt_counts[ext] = fmt_counts.get(ext, 0) + 1
        parts = [f"{ext}: {n}" for ext, n in sorted(fmt_counts.items(), key=lambda x: -x[1])]
        self._format_lbl.setText("  ".join(parts) or "–")

    def _update_size_stats(self) -> None:
        """Sample max 200 Bilder für Performance."""
        images = self.project.images
        if not images:
            self._size_lbl.setText("–")
            return
        import random
        sample = random.sample(images, min(200, len(images)))
        widths, heights = [], []
        for p in sample:
            if not os.path.isfile(p):
                continue
            try:
                from PIL import Image as _PIL
                with _PIL.open(p) as img:
                    w, h = img.size
                widths.append(w)
                heights.append(h)
            except Exception:
                pass
        if widths:
            self._size_lbl.setText(
                f"Ø {sum(widths)//len(widths)} × {sum(heights)//len(heights)} px "
                f"(Sample: {len(widths)} Bilder)"
            )
        else:
            from utils.i18n import tr
            self._size_lbl.setText(tr("dataset_stats.not_readable"))

    def _update_label_rate(self) -> None:
        total = len(self.project.images)
        labeled = sum(1 for p in self.project.images if self.project.image_labels.get(p))
        self._total_lbl.setText(str(total))
        self._labeled_lbl.setText(str(labeled))
        rate = labeled / total * 100 if total else 0
        self._label_rate_lbl.setText(f"{rate:.1f}%")

    @Slot()
    def _find_duplicates(self) -> None:
        from utils.i18n import tr
        if not self.project or not self.project.images:
            QMessageBox.information(self, tr("common.info"), tr("common.no_project"))
            return
        try:
            import imagehash
            from PIL import Image as _PIL
        except ImportError:
            self._dup_btn.setText(tr("dataset_stats.imagehash_missing"))
            self._dup_btn.setEnabled(False)
            return

        self._dup_status_lbl.setText("Suche läuft…")
        hash_map: dict = {}
        for p in self.project.images:
            if not os.path.isfile(p):
                continue
            try:
                with _PIL.open(p) as img:
                    h = str(imagehash.average_hash(img))
                hash_map.setdefault(h, []).append(p)
            except Exception:
                pass

        groups = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
        self._dup_list.clear()
        if not groups:
            self._dup_status_lbl.setText(tr("dataset_stats.no_dups"))
            return
        total_dups = sum(len(v) for v in groups.values())
        self._dup_status_lbl.setText(tr("dataset_stats.dups_found", groups=len(groups), images=total_dups))
        for i, (h, paths) in enumerate(groups.items(), 1):
            self._dup_list.addItem(f"Gruppe {i}: " + ", ".join(os.path.basename(p) for p in paths))
