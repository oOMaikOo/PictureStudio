"""
Data Drift Page (stack index 16) — compare production images to training distribution.

Workflow:
  1. Build a baseline from training images (all project images, or a specific folder).
  2. Score a new folder of production images against the baseline.
  3. Drifted images are highlighted; export CSV for further analysis.
"""
from __future__ import annotations

import csv
import os
from typing import List, Dict, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QPushButton, QLabel, QDoubleSpinBox, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QMessageBox, QCheckBox, QSizePolicy, QScrollArea, QFrame,
    QSpinBox,
)
from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QColor, QFont

from core.data_drift import DriftDetector


_SCORE_OK     = QColor("#4caf50")  # green   < threshold
_SCORE_WARN   = QColor("#ff9800")  # orange  threshold .. 2×threshold
_SCORE_DRIFT  = QColor("#f44336")  # red     > 2×threshold
_TEXT_WHITE   = QColor("#ffffff")


def _row_color(score: float, threshold: float) -> Optional[QColor]:
    if score < 0:
        return None
    if score <= threshold:
        return _SCORE_OK
    if score <= threshold * 2:
        return _SCORE_WARN
    return _SCORE_DRIFT


class DataDriftPage(QWidget):
    """Data drift monitoring — stack index 16 (image projects)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._detector = DriftDetector()
        self._baseline_thread = None
        self._scoring_thread  = None
        self._results: List[Dict] = []
        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setSizes([300, 700])

    # ---- Left panel: baseline + threshold config ----

    def _build_left(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(8)

        # Info
        info_lbl = QLabel(
            "<b>Was ist Data Drift?</b><br>"
            "Wenn sich Produktionsbilder statistisch von den Trainingsbildern unterscheiden "
            "(andere Beleuchtung, Kamerawinkel, Bildqualität), kann das Modell schlechter werden. "
            "Dieses Werkzeug erkennt solche Verschiebungen automatisch."
        )
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        v.addWidget(info_lbl)

        from utils.i18n import tr
        # ---- Baseline group ----
        grp_base = QGroupBox(tr("datadrift.baseline_group"))
        gb = QVBoxLayout(grp_base)

        self._baseline_lbl = QLabel(tr("datadrift.no_baseline"))
        self._baseline_lbl.setWordWrap(True)
        self._baseline_lbl.setStyleSheet("color: #ccc;")
        gb.addWidget(self._baseline_lbl)

        self._btn_build = QPushButton(tr("datadrift.build_btn"))
        self._btn_build.setToolTip(
            "Analysiert alle Bilder des aktuellen Projekts und speichert deren statistische Verteilung."
        )
        self._btn_build.clicked.connect(self._build_baseline_from_project)
        gb.addWidget(self._btn_build)

        row_bl = QHBoxLayout()
        self._btn_build_folder = QPushButton(tr("datadrift.build_folder_btn"))
        self._btn_build_folder.setToolTip("Baseline aus einem benutzerdefinierten Ordner erstellen.")
        self._btn_build_folder.clicked.connect(self._build_baseline_from_folder)
        row_bl.addWidget(self._btn_build_folder)

        self._btn_save_bl = QPushButton(tr("datadrift.save_btn"))
        self._btn_save_bl.clicked.connect(self._save_baseline)
        self._btn_save_bl.setEnabled(False)
        row_bl.addWidget(self._btn_save_bl)

        self._btn_load_bl = QPushButton(tr("datadrift.load_btn"))
        self._btn_load_bl.clicked.connect(self._load_baseline)
        row_bl.addWidget(self._btn_load_bl)
        gb.addLayout(row_bl)

        self._baseline_prog = QProgressBar()
        self._baseline_prog.setVisible(False)
        gb.addWidget(self._baseline_prog)

        v.addWidget(grp_base)

        # ---- Threshold ----
        grp_thr = QGroupBox(tr("datadrift.threshold_group"))
        gt = QVBoxLayout(grp_thr)

        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel(tr("datadrift.threshold_label")))
        self._thr_spin = QDoubleSpinBox()
        self._thr_spin.setRange(1.0, 20.0)
        self._thr_spin.setSingleStep(0.5)
        self._thr_spin.setValue(3.0)
        self._thr_spin.setDecimals(1)
        self._thr_spin.setToolTip(
            "Z-Score-Schwellwert: Ein Wert > Schwellwert bedeutet Drift. "
            "3.0 = statistisch unwahrscheinlich unter der Trainingsdistribution."
        )
        thr_row.addWidget(self._thr_spin)
        gt.addLayout(thr_row)

        thr_note = QLabel(
            "Z-Score = maximale Standardabweichung über alle Bildmerkmale.\n"
            "Grün ≤ Schwellwert  •  Orange ≤ 2×  •  Rot > 2×"
        )
        thr_note.setStyleSheet("color: #aaa; font-size: 10px;")
        thr_note.setWordWrap(True)
        gt.addWidget(thr_note)
        v.addWidget(grp_thr)

        # ---- Production folder ----
        grp_prod = QGroupBox(tr("datadrift.production_group"))
        gp = QVBoxLayout(grp_prod)

        folder_row = QHBoxLayout()
        self._folder_lbl = QLabel(tr("datadrift.no_folder"))
        self._folder_lbl.setStyleSheet("color: #ccc; font-size: 11px;")
        folder_row.addWidget(self._folder_lbl, 1)
        self._btn_pick = QPushButton(tr("datadrift.pick_folder_btn"))
        self._btn_pick.setFixedWidth(32)
        self._btn_pick.clicked.connect(self._pick_folder)
        folder_row.addWidget(self._btn_pick)
        gp.addLayout(folder_row)

        self._recursive_cb = QCheckBox(tr("datadrift.recursive_cb"))
        gp.addWidget(self._recursive_cb)

        self._btn_score = QPushButton(tr("datadrift.score_btn"))
        self._btn_score.setEnabled(False)
        self._btn_score.clicked.connect(self._start_scoring)
        gp.addWidget(self._btn_score)

        self._score_prog = QProgressBar()
        self._score_prog.setVisible(False)
        gp.addWidget(self._score_prog)

        self._score_summary = QLabel("")
        self._score_summary.setWordWrap(True)
        gp.addWidget(self._score_summary)

        v.addWidget(grp_prod)

        # Export
        self._btn_export = QPushButton(tr("datadrift.export_csv_btn"))
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._export_csv)
        v.addWidget(self._btn_export)

        v.addStretch()
        return w

    # ---- Right panel: results table ----

    def _build_right(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)

        hdr = QLabel("Ergebnisse")
        hdr.setFont(QFont("", 11, QFont.Weight.Bold))
        v.addWidget(hdr)

        from utils.i18n import tr
        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels([
            tr("datadrift.col.file"), tr("datadrift.col.zscore"),
            tr("datadrift.col.color"), tr("datadrift.col.sharpness"),
            tr("datadrift.col.edges"), tr("datadrift.col.histogram"),
        ])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, 6):
            self._table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        v.addWidget(self._table)

        return w

    # ------------------------------------------------------------------ project

    def set_project(self, project):
        self.project = project
        n = len(getattr(project, "images", []))
        self._baseline_lbl.setText(
            f"Projekt: {n} Bilder verfügbar.\n"
            f"{'Baseline vorhanden: ' + str(self._detector.n_baseline) + ' Bilder.' if self._detector.is_ready() else 'Keine Baseline vorhanden.'}"
        )
        self._btn_build.setEnabled(n > 0)

    # ------------------------------------------------------------------ baseline

    @Slot()
    def _build_baseline_from_project(self):
        from utils.i18n import tr
        if not self.project or not self.project.images:
            QMessageBox.warning(self, tr("common.no_images"), tr("datadrift.no_images_msg"))
            return
        self._run_baseline(list(self.project.images))

    @Slot()
    def _build_baseline_from_folder(self):
        from utils.i18n import tr
        folder = QFileDialog.getExistingDirectory(self, tr("datadrift.pick_folder_dlg"))
        if not folder:
            return
        from core.data_drift import IMAGE_EXTS
        paths = [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ]
        if not paths:
            QMessageBox.warning(self, tr("common.no_images"), tr("datadrift.no_folder_msg"))
            return
        self._run_baseline(paths)

    def _run_baseline(self, paths: list):
        from core.data_drift import DriftBaselineThread
        self._baseline_prog.setVisible(True)
        self._baseline_prog.setMaximum(len(paths))
        self._baseline_prog.setValue(0)
        self._btn_build.setEnabled(False)
        self._btn_build_folder.setEnabled(False)

        t = DriftBaselineThread(self._detector, paths, self)
        t.progress.connect(lambda c, tot: self._baseline_prog.setValue(c))
        t.finished.connect(self._on_baseline_done)
        t.error.connect(self._on_baseline_error)
        self._baseline_thread = t
        t.start()

    @Slot(dict)
    def _on_baseline_done(self, stats: dict):
        self._baseline_prog.setVisible(False)
        self._btn_build.setEnabled(True)
        self._btn_build_folder.setEnabled(True)
        self._btn_save_bl.setEnabled(True)
        from utils.i18n import tr
        n_ok  = stats.get("n_ok", 0)
        n_err = stats.get("n_errors", 0)
        self._baseline_lbl.setText(
            tr("datadrift.baseline_done", n=n_ok)
            + (f" ({n_err} übersprungen)" if n_err else "")
        )
        self._update_score_btn()

    @Slot(str)
    def _on_baseline_error(self, msg: str):
        from utils.i18n import tr
        self._baseline_prog.setVisible(False)
        self._btn_build.setEnabled(True)
        self._btn_build_folder.setEnabled(True)
        QMessageBox.critical(self, tr("common.error"), msg)

    @Slot()
    def _save_baseline(self):
        from utils.i18n import tr
        path, _ = QFileDialog.getSaveFileName(
            self, tr("datadrift.save_dlg"), "drift_baseline.json", "JSON (*.json)"
        )
        if path:
            try:
                self._detector.save(path)
            except Exception as exc:
                QMessageBox.critical(self, tr("common.error"), str(exc))

    @Slot()
    def _load_baseline(self):
        from utils.i18n import tr
        path, _ = QFileDialog.getOpenFileName(
            self, tr("datadrift.load_dlg"), "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            self._detector.load(path)
            self._baseline_lbl.setText(
                tr("datadrift.baseline_loaded", n=self._detector.n_baseline) + f"\n({path})"
            )
            self._btn_save_bl.setEnabled(True)
            self._update_score_btn()
        except Exception as exc:
            QMessageBox.critical(self, tr("common.error"), str(exc))

    # ------------------------------------------------------------------ folder

    @Slot()
    def _pick_folder(self):
        from utils.i18n import tr
        folder = QFileDialog.getExistingDirectory(self, tr("datadrift.pick_production_dlg"))
        if folder:
            self._folder_lbl.setText(folder)
            self._update_score_btn()

    def _update_score_btn(self):
        folder_ok   = os.path.isdir(self._folder_lbl.text())
        baseline_ok = self._detector.is_ready()
        self._btn_score.setEnabled(folder_ok and baseline_ok)

    # ------------------------------------------------------------------ scoring

    @Slot()
    def _start_scoring(self):
        folder = self._folder_lbl.text()
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "Ordner nicht gefunden", folder)
            return

        from core.data_drift import DriftScoringThread
        self._score_prog.setVisible(True)
        self._score_prog.setValue(0)
        self._btn_score.setEnabled(False)
        self._score_summary.setText("")

        t = DriftScoringThread(
            self._detector, folder,
            recursive=self._recursive_cb.isChecked(),
            threshold=self._thr_spin.value(),
            parent=self,
        )
        t.progress.connect(lambda c, tot: (
            self._score_prog.setMaximum(tot),
            self._score_prog.setValue(c),
        ))
        t.finished.connect(self._on_scoring_done)
        t.error.connect(self._on_scoring_error)
        self._scoring_thread = t
        t.start()

    @Slot(list)
    def _on_scoring_done(self, results: list):
        self._score_prog.setVisible(False)
        self._btn_score.setEnabled(True)
        self._results = results
        self._populate_table(results)
        self._btn_export.setEnabled(bool(results))

        n_drifted = sum(1 for r in results if r.get("drifted"))
        n_total   = len(results)
        threshold = self._thr_spin.value()
        pct       = round(100 * n_drifted / n_total) if n_total else 0
        color     = "#f44336" if pct > 20 else ("#ff9800" if pct > 5 else "#4caf50")
        self._score_summary.setText(
            f"<span style='color:{color}'><b>{n_drifted}/{n_total} Bilder ({pct}%) "
            f"über Schwellwert {threshold:.1f}</b></span>"
        )

    @Slot(str)
    def _on_scoring_error(self, msg: str):
        from utils.i18n import tr
        self._score_prog.setVisible(False)
        self._btn_score.setEnabled(True)
        QMessageBox.critical(self, tr("common.error"), msg)

    def _populate_table(self, results: list):
        threshold = self._thr_spin.value()
        self._table.setRowCount(0)
        self._table.setRowCount(len(results))

        for row, r in enumerate(results):
            score = r.get("score", -1.0)
            det   = r.get("details", {})
            err   = r.get("error")

            def _item(text, align=Qt.AlignmentFlag.AlignCenter):
                it = QTableWidgetItem(str(text))
                it.setTextAlignment(align)
                return it

            fn_item = _item(r["filename"], Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            if err:
                fn_item.setToolTip(f"Fehler: {err}")

            score_txt = f"{score:.2f}" if score >= 0 else "Fehler"
            col_mean  = det.get("z_color_mean",  0.0)
            sharp     = det.get("z_sharpness",   0.0)
            edges     = det.get("z_edges",       0.0)
            hist      = det.get("z_histogram",   0.0)

            items = [
                fn_item,
                _item(score_txt),
                _item(f"{col_mean:.2f}"),
                _item(f"{sharp:.2f}"),
                _item(f"{edges:.2f}"),
                _item(f"{hist:.2f}"),
            ]
            for col, it in enumerate(items):
                self._table.setItem(row, col, it)

            bg = _row_color(score, threshold)
            if bg:
                for col in range(6):
                    cell = self._table.item(row, col)
                    if cell:
                        cell.setBackground(bg)
                        cell.setForeground(_TEXT_WHITE)

    # ------------------------------------------------------------------ export

    @Slot()
    def _export_csv(self):
        from utils.i18n import tr
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, tr("datadrift.export_dlg"), "data_drift_report.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Datei", "Pfad", "Z-Score", "Drifted",
                             "Z-Farbe", "Z-Schärfe", "Z-Kanten", "Z-Histogramm", "Fehler"])
                for r in self._results:
                    det = r.get("details", {})
                    w.writerow([
                        r["filename"], r["path"],
                        r["score"], r["drifted"],
                        round(det.get("z_color_mean", 0.0), 3),
                        round(det.get("z_sharpness",  0.0), 3),
                        round(det.get("z_edges",      0.0), 3),
                        round(det.get("z_histogram",  0.0), 3),
                        r.get("error", ""),
                    ])
            from utils.i18n import tr
            QMessageBox.information(self, tr("common.exported"), tr("common.saved") + f"\n{path}")
        except Exception as exc:
            from utils.i18n import tr
            QMessageBox.critical(self, tr("common.error"), str(exc))
