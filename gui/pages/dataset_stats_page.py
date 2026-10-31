from __future__ import annotations
import logging
import os
from typing import Optional, List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QProgressBar, QListWidget, QScrollArea, QFrame,
    QFormLayout, QMessageBox,
)
from PySide6.QtCore import Qt, Slot, QThread, Signal
from PySide6.QtGui import QKeySequence, QShortcut

log = logging.getLogger(__name__)


class DatasetStatsWorker(QThread):
    """Background worker for PIL-heavy dataset stats (sizes / duplicates).

    Analogous to ``AnalysisThread`` on the Data page — keeps the UI responsive
    when sampling image dimensions or computing perceptual hashes.
    """

    finished = Signal(str, object)  # kind ("sizes"|"dups"), payload
    error = Signal(str)

    def __init__(self, images: List[str], kind: str, parent=None) -> None:
        super().__init__(parent)
        self._images = list(images)
        self._kind = kind  # "sizes" | "dups"

    def run(self) -> None:
        try:
            if self._kind == "sizes":
                self.finished.emit("sizes", self._compute_sizes())
            elif self._kind == "dups":
                self.finished.emit("dups", self._compute_dups())
            else:
                self.error.emit(f"Unknown kind: {self._kind}")
        except Exception as exc:
            self.error.emit(str(exc))

    def _compute_sizes(self) -> dict:
        import random
        images = self._images
        if not images:
            return {"text": "–", "ok": False}
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
                log.debug("Could not read image dimensions for %s", p)
        if not widths:
            return {"text": "", "ok": False}
        return {
            "ok": True,
            "avg_w": sum(widths) // len(widths),
            "avg_h": sum(heights) // len(heights),
            "n": len(widths),
        }

    def _compute_dups(self) -> dict:
        import imagehash
        from PIL import Image as _PIL

        hash_map: dict = {}
        for p in self._images:
            if not os.path.isfile(p):
                continue
            try:
                with _PIL.open(p) as img:
                    h = str(imagehash.average_hash(img))
                hash_map.setdefault(h, []).append(p)
            except Exception:
                log.debug("Could not hash image %s", p)

        groups = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
        return {
            "groups": [
                [os.path.basename(p) for p in paths]
                for paths in groups.values()
            ],
            "n_groups": len(groups),
            "n_images": sum(len(v) for v in groups.values()),
        }


class DatasetStatsPage(QWidget):
    """
    Datensatz-Analyse-Seite (Stack-Index wird von MainWindow vergeben).
    Zeigt: Klassenverteilung, Dateiformat-Statistik, Bildgrößen, Duplikate, Label-Rate.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.project = None
        self._worker: Optional[DatasetStatsWorker] = None
        self._build_ui()
        QShortcut(QKeySequence("F5"), self, activated=self.refresh)

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
            self._update_label_rate()
            self._dup_list.clear()
            from utils.i18n import tr as _tr
            self._dup_status_lbl.setText(_tr("dataset_stats.dups_not_searched"))
            # Size stats (PIL) off the UI thread — skip when nothing is on disk
            # (avoids QThread teardown races in tests with fake paths).
            if any(os.path.isfile(p) for p in self.project.images[:50]):
                self._size_lbl.setText("…")
                self._start_worker("sizes")
            else:
                self._size_lbl.setText("–")
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
        refresh_btn.setToolTip(tr("dataset_stats.refresh_tip"))
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
        self._dup_status_lbl = QLabel(tr("dataset_stats.dups_not_searched"))
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

    def _update_label_rate(self) -> None:
        total = len(self.project.images)
        labeled = sum(1 for p in self.project.images if self.project.image_labels.get(p))
        self._total_lbl.setText(str(total))
        self._labeled_lbl.setText(str(labeled))
        rate = labeled / total * 100 if total else 0
        self._label_rate_lbl.setText(f"{rate:.1f}%")

    def _start_worker(self, kind: str) -> None:
        if not self.project:
            return
        if self._worker and self._worker.isRunning():
            # One worker at a time — avoid dropping the running reference.
            return
        worker = DatasetStatsWorker(self.project.images, kind, parent=self)
        self._worker = worker
        worker.finished.connect(self._on_worker_finished)
        worker.error.connect(self._on_worker_error)
        worker.start()

    @Slot(str, object)
    def _on_worker_finished(self, kind: str, payload: object) -> None:
        from utils.i18n import tr
        self._worker = None
        if kind == "sizes":
            data = payload or {}
            if data.get("ok"):
                self._size_lbl.setText(
                    tr("dataset_stats.size_sample",
                       w=data["avg_w"], h=data["avg_h"], n=data["n"])
                )
            else:
                self._size_lbl.setText(tr("dataset_stats.not_readable"))
            return

        if kind == "dups":
            self._dup_btn.setEnabled(True)
            data = payload or {}
            groups = data.get("groups", [])
            self._dup_list.clear()
            if not groups:
                self._dup_status_lbl.setText(tr("dataset_stats.no_dups"))
                return
            self._dup_status_lbl.setText(
                tr("dataset_stats.dups_found",
                   groups=data.get("n_groups", 0),
                   images=data.get("n_images", 0))
            )
            for i, names in enumerate(groups, 1):
                self._dup_list.addItem(
                    tr("dataset_stats.dup_group_row", i=i, names=", ".join(names))
                )

    @Slot(str)
    def _on_worker_error(self, msg: str) -> None:
        from utils.i18n import tr
        self._worker = None
        self._dup_btn.setEnabled(True)
        self._size_lbl.setText(tr("dataset_stats.not_readable"))
        log.error("DatasetStatsWorker: %s", msg)

    @Slot()
    def _find_duplicates(self) -> None:
        from utils.i18n import tr
        if not self.project or not self.project.images:
            QMessageBox.information(self, tr("common.info"), tr("common.no_project"))
            return
        try:
            import imagehash  # noqa: F401
            from PIL import Image as _PIL  # noqa: F401
        except ImportError:
            self._dup_btn.setText(tr("dataset_stats.imagehash_missing"))
            self._dup_btn.setEnabled(False)
            return

        self._dup_status_lbl.setText(tr("dataset_stats.search_running"))
        self._dup_btn.setEnabled(False)
        self._dup_list.clear()
        self._start_worker("dups")

    def hideEvent(self, event) -> None:
        self._stop_worker()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:
        self._stop_worker()
        super().closeEvent(event)

    def _stop_worker(self) -> None:
        w = self._worker
        self._worker = None
        if w is None:
            return
        try:
            w.finished.disconnect()
        except (RuntimeError, TypeError):
            pass
        try:
            w.error.disconnect()
        except (RuntimeError, TypeError):
            pass
        if w.isRunning():
            w.wait(3000)
