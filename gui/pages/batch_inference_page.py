"""
Batch inference page: classify or anomaly-score an entire folder in one run.
Results shown in a sortable table; exportable as CSV.
"""
import csv
import os
from typing import List, Dict, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QMessageBox, QAbstractItemView,
    QComboBox, QDoubleSpinBox, QSplitter,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont


# ── background thread ──────────────────────────────────────────────────────────

class _BatchThread(QThread):
    progress = Signal(int, int)     # current, total
    finished = Signal(list)         # list of result dicts
    error    = Signal(str)

    def __init__(self, worker, paths: List[str], parent=None):
        super().__init__(parent)
        self._worker = worker
        self._paths = paths

    def run(self) -> None:
        try:
            results = self._worker.run(self._paths)
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))

    def cancel(self) -> None:
        self._worker.cancel()


# ── page ───────────────────────────────────────────────────────────────────────

class BatchInferencePage(QWidget):
    """Folder-level batch classification with CSV export."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Optional[object] = None
        self._model = None
        self._class_names: List[str] = []
        self._image_size: int = 224
        self._results: List[Dict] = []
        self._thread: Optional[_BatchThread] = None
        self._build_ui()

    # ------------------------------------------------------------------ project

    def set_project(self, project, audit=None) -> None:
        self.project = project
        self._refresh_model_combo()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setSizes([300, 800])

    def _build_left(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        # ── Model ──────────────────────────────────────────────────────────────
        mg = QGroupBox("Modell")
        mv = QVBoxLayout(mg)
        self._model_combo = QComboBox()
        self._model_combo.setToolTip("Trainiertes Modell aus dem Projekt wählen")
        mv.addWidget(self._model_combo)
        load_ext_btn = QPushButton("Externe .pth laden…")
        load_ext_btn.clicked.connect(self._load_external_model)
        mv.addWidget(load_ext_btn)
        self._model_info_lbl = QLabel("Kein Modell geladen")
        self._model_info_lbl.setWordWrap(True)
        self._model_info_lbl.setStyleSheet("color:#7F8C8D;font-size:10px;")
        mv.addWidget(self._model_info_lbl)
        load_sel_btn = QPushButton("Ausgewähltes Modell laden")
        load_sel_btn.setStyleSheet(
            "background:#1F6FEB;color:white;font-weight:bold;padding:5px;"
        )
        load_sel_btn.clicked.connect(self._load_selected_model)
        mv.addWidget(load_sel_btn)
        v.addWidget(mg)

        # ── Input folder ───────────────────────────────────────────────────────
        ig = QGroupBox("Eingabe-Ordner")
        iv = QVBoxLayout(ig)
        self._folder_lbl = QLabel("Kein Ordner gewählt")
        self._folder_lbl.setWordWrap(True)
        self._folder_lbl.setStyleSheet("color:#7F8C8D;font-size:10px;")
        iv.addWidget(self._folder_lbl)
        folder_btn = QPushButton("Ordner wählen…")
        folder_btn.clicked.connect(self._choose_folder)
        iv.addWidget(folder_btn)
        use_proj_btn = QPushButton("Projektbilder verwenden")
        use_proj_btn.setToolTip("Alle Bilder aus dem geöffneten Projekt verarbeiten")
        use_proj_btn.clicked.connect(self._use_project_images)
        iv.addWidget(use_proj_btn)
        self._image_count_lbl = QLabel("")
        self._image_count_lbl.setStyleSheet("color:#7F8C8D;font-size:10px;")
        iv.addWidget(self._image_count_lbl)
        v.addWidget(ig)

        # ── Options ────────────────────────────────────────────────────────────
        og = QGroupBox("Filter")
        ov = QVBoxLayout(og)
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Min. Confidence:"))
        self._min_conf_spin = QDoubleSpinBox()
        self._min_conf_spin.setRange(0.0, 1.0)
        self._min_conf_spin.setSingleStep(0.05)
        self._min_conf_spin.setValue(0.0)
        conf_row.addWidget(self._min_conf_spin)
        ov.addLayout(conf_row)
        cls_row = QHBoxLayout()
        cls_row.addWidget(QLabel("Klassen-Filter:"))
        self._cls_filter_combo = QComboBox()
        self._cls_filter_combo.addItem("Alle")
        cls_row.addWidget(self._cls_filter_combo)
        ov.addLayout(cls_row)
        apply_btn = QPushButton("Filter anwenden")
        apply_btn.clicked.connect(self._apply_filter)
        ov.addWidget(apply_btn)
        v.addWidget(og)

        # ── Run ────────────────────────────────────────────────────────────────
        self._run_btn = QPushButton("▶  Batch starten")
        self._run_btn.setFixedHeight(40)
        self._run_btn.setStyleSheet(
            "background:#3FB950;color:white;font-weight:bold;font-size:13px;padding:6px;"
        )
        self._run_btn.clicked.connect(self._start_batch)
        v.addWidget(self._run_btn)

        self._cancel_btn = QPushButton("Abbrechen")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._cancel_batch)
        v.addWidget(self._cancel_btn)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        v.addWidget(self._progress)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet("color:#7F8C8D;font-size:10px;")
        v.addWidget(self._status_lbl)

        v.addStretch()

        # ── Export ─────────────────────────────────────────────────────────────
        export_btn = QPushButton("CSV exportieren…")
        export_btn.clicked.connect(self._export_csv)
        v.addWidget(export_btn)

        return w

    def _build_right(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 8, 8, 8)

        # Summary row
        self._summary_lbl = QLabel("Keine Ergebnisse")
        self._summary_lbl.setStyleSheet("font-size:12px;color:#ADBAC7;padding:4px;")
        v.addWidget(self._summary_lbl)

        # Results table
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["Dateiname", "Klasse", "Confidence", "Fehler"]
        )
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        v.addWidget(self._table)

        return w

    # ------------------------------------------------------------------ model

    def _refresh_model_combo(self) -> None:
        self._model_combo.clear()
        if not self.project:
            return
        for run in self.project.training_runs:
            mp = run.get("best_model_path", "")
            if mp and os.path.exists(mp):
                label = f"{os.path.basename(mp)}  ({run.get('model_type', '?')})"
                self._model_combo.addItem(label, userData=mp)
        if self._model_combo.count() == 0:
            self._model_combo.addItem("Kein trainiertes Modell vorhanden")

    def _load_selected_model(self) -> None:
        path = self._model_combo.currentData()
        if not path:
            QMessageBox.warning(self, "Kein Modell", "Bitte Modell aus Liste wählen.")
            return
        self._load_model_from_path(path)

    def _load_external_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Modell laden", "", "PyTorch (*.pth)"
        )
        if path:
            self._load_model_from_path(path)

    def _load_model_from_path(self, path: str) -> None:
        try:
            from core.inference import Inferencer
            inf = Inferencer()
            meta = inf.load_model(path)
            self._model = inf.model
            self._class_names = meta.get("class_names", [])
            self._image_size = meta.get("image_size", 224)
            self._model_info_lbl.setText(
                f"✓ {os.path.basename(path)}\n"
                f"Klassen: {', '.join(self._class_names)}\n"
                f"Bildgröße: {self._image_size}px"
            )
            self._model_info_lbl.setStyleSheet("color:#3FB950;font-size:10px;")
            self._cls_filter_combo.clear()
            self._cls_filter_combo.addItem("Alle")
            for cls in self._class_names:
                self._cls_filter_combo.addItem(cls)
        except Exception as exc:
            QMessageBox.critical(self, "Modellfehler", str(exc))

    # ------------------------------------------------------------------ folder

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Ordner wählen")
        if folder:
            self._set_folder(folder)

    def _use_project_images(self) -> None:
        if not self.project or not self.project.images:
            QMessageBox.information(self, "Kein Projekt", "Kein Projekt mit Bildern geladen.")
            return
        self._image_paths = list(self.project.images)
        self._folder_lbl.setText(f"Projektbilder ({len(self._image_paths)} Bilder)")
        self._image_count_lbl.setText(f"{len(self._image_paths)} Bilder bereit")

    def _set_folder(self, folder: str) -> None:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        paths = [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if os.path.splitext(f)[1].lower() in exts
        ]
        self._image_paths = paths
        self._folder_lbl.setText(folder)
        self._image_count_lbl.setText(f"{len(paths)} Bilder gefunden")

    # ------------------------------------------------------------------ batch run

    def _start_batch(self) -> None:
        if self._model is None:
            QMessageBox.warning(self, "Kein Modell", "Bitte zuerst ein Modell laden.")
            return
        if not getattr(self, "_image_paths", None):
            QMessageBox.warning(self, "Keine Bilder", "Bitte Ordner wählen oder Projektbilder verwenden.")
            return

        from core.batch_inference import BatchInferenceWorker
        worker = BatchInferenceWorker(
            model=self._model,
            class_names=self._class_names,
            image_size=self._image_size,
        )
        self._thread = _BatchThread(worker, self._image_paths, self)
        self._thread.progress.connect(self._on_progress)
        self._thread.finished.connect(self._on_finished)
        self._thread.error.connect(self._on_error)

        total = len(self._image_paths)
        self._progress.setRange(0, total)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._run_btn.setVisible(False)
        self._cancel_btn.setVisible(True)
        self._status_lbl.setText(f"Verarbeite 0 / {total} Bilder …")
        self._results = []
        self._table.setRowCount(0)
        self._thread.start()

    @Slot(int, int)
    def _on_progress(self, current: int, total: int) -> None:
        self._progress.setValue(current)
        self._status_lbl.setText(f"Verarbeite {current} / {total} Bilder …")

    @Slot(list)
    def _on_finished(self, results: List[Dict]) -> None:
        self._results = results
        self._progress.setVisible(False)
        self._run_btn.setVisible(True)
        self._cancel_btn.setVisible(False)
        self._show_results(results)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._progress.setVisible(False)
        self._run_btn.setVisible(True)
        self._cancel_btn.setVisible(False)
        QMessageBox.critical(self, "Fehler", msg)

    def _cancel_batch(self) -> None:
        if self._thread:
            self._thread.cancel()
            self._status_lbl.setText("Wird abgebrochen …")

    # ------------------------------------------------------------------ display

    def _show_results(self, results: List[Dict]) -> None:
        min_conf = self._min_conf_spin.value()
        cls_filter = self._cls_filter_combo.currentText()

        filtered = [
            r for r in results
            if r["confidence"] >= min_conf
            and (cls_filter == "Alle" or r["predicted"] == cls_filter)
        ]

        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(filtered))

        errors = sum(1 for r in filtered if r["error"])
        counts: Dict[str, int] = {}
        for r in filtered:
            counts[r["predicted"]] = counts.get(r["predicted"], 0) + 1

        for row, res in enumerate(filtered):
            self._table.setItem(row, 0, QTableWidgetItem(res["filename"]))
            self._table.setItem(row, 1, QTableWidgetItem(res["predicted"]))

            conf_item = QTableWidgetItem(f"{res['confidence']:.1%}")
            conf_item.setData(Qt.UserRole, res["confidence"])
            if res["confidence"] >= 0.9:
                conf_item.setForeground(QColor("#3FB950"))
            elif res["confidence"] >= 0.7:
                conf_item.setForeground(QColor("#D29922"))
            else:
                conf_item.setForeground(QColor("#F85149"))
            self._table.setItem(row, 2, conf_item)

            err_item = QTableWidgetItem(res["error"] or "")
            if res["error"]:
                err_item.setForeground(QColor("#F85149"))
            self._table.setItem(row, 3, err_item)

            for col in range(4):
                item = self._table.item(row, col)
                if item:
                    item.setData(Qt.UserRole + 1, res["path"])

        self._table.setSortingEnabled(True)

        summary_parts = [f"{len(filtered)} Bilder"]
        for cls, n in sorted(counts.items()):
            summary_parts.append(f"{cls}: {n}")
        if errors:
            summary_parts.append(f"⚠ {errors} Fehler")
        self._summary_lbl.setText("  |  ".join(summary_parts))
        self._status_lbl.setText(f"Fertig — {len(results)} Bilder verarbeitet")

    def _apply_filter(self) -> None:
        if self._results:
            self._show_results(self._results)

    # ------------------------------------------------------------------ export

    def _export_csv(self) -> None:
        if not self._results:
            QMessageBox.information(self, "Keine Daten", "Bitte zuerst Batch starten.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "CSV exportieren", "batch_results.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Header: fixed columns + one per class
                writer.writerow(
                    ["filename", "path", "predicted", "confidence", "error"]
                    + [f"p_{cls}" for cls in self._class_names]
                )
                for r in self._results:
                    prob_cols = [
                        f"{r['probabilities'].get(cls, 0):.4f}"
                        for cls in self._class_names
                    ]
                    writer.writerow([
                        r["filename"], r["path"], r["predicted"],
                        f"{r['confidence']:.4f}", r["error"] or "",
                        *prob_cols,
                    ])
            QMessageBox.information(self, "Exportiert", f"CSV gespeichert:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Exportfehler", str(exc))
