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
    QComboBox, QDoubleSpinBox, QSplitter, QScrollArea, QFrame,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont

from utils.i18n import tr


# ── background thread ──────────────────────────────────────────────────────────

class _BatchThread(QThread):
    """QThread wrapper that runs ``BatchInferenceWorker`` in the background."""

    progress = Signal(int, int)     # current, total
    finished = Signal(list)         # list of result dicts
    error    = Signal(str)

    def __init__(self, worker, paths: List[str], parent=None):
        """
        Parameters
        ----------
        worker : ``BatchInferenceWorker`` instance (model + class names already set).
        paths  : Ordered list of absolute image file paths to process.
        """
        super().__init__(parent)
        self._worker = worker
        self._paths = paths

    def run(self) -> None:
        """Run the worker and emit ``finished`` with the result list, or ``error``."""
        try:
            results = self._worker.run(self._paths)
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))

    def cancel(self) -> None:
        """Forward a cancellation request to the underlying worker."""
        self._worker.cancel()


# ── page ───────────────────────────────────────────────────────────────────────

class BatchInferencePage(QWidget):
    """
    Batch inference page (stack index 9).

    Classifies an entire folder of images (or all project images) using a
    ``BatchInferenceWorker`` running in a background ``_BatchThread``.

    Features:
    - Load a model from the project's training runs or from an external .pth file.
    - Folder selection or use of all project images as input.
    - Confidence threshold and class-label filter applied to the results table.
    - Red-highlighted rows for predictions below the warning threshold.
    - Export all results or only low-confidence results as CSV.
    - Cancel support: forwards cancellation to the worker's cancel flag.
    """

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
        """Accept the active project and populate the model combo from its training runs."""
        self.project = project
        self._refresh_model_combo()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self._build_left())
        splitter.addWidget(scroll)

        splitter.addWidget(self._build_right())
        splitter.setSizes([320, 780])

    def _build_left(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        # ── Model ──────────────────────────────────────────────────────────────
        mg = QGroupBox(tr("batch.model_group"))
        mv = QVBoxLayout(mg)
        self._model_combo = QComboBox()
        self._model_combo.setToolTip("Trainiertes Modell aus dem Projekt wählen")
        mv.addWidget(self._model_combo)
        load_ext_btn = QPushButton(tr("batch.external_model_btn"))
        load_ext_btn.clicked.connect(self._load_external_model)
        mv.addWidget(load_ext_btn)
        self._model_info_lbl = QLabel(tr("common.no_model"))
        self._model_info_lbl.setWordWrap(True)
        self._model_info_lbl.setStyleSheet("color:#7F8C8D;font-size:10px;")
        mv.addWidget(self._model_info_lbl)
        load_sel_btn = QPushButton(tr("batch.load_btn"))
        load_sel_btn.setStyleSheet(
            "background:#1F6FEB;color:white;font-weight:bold;padding:5px;"
        )
        load_sel_btn.clicked.connect(self._load_selected_model)
        mv.addWidget(load_sel_btn)
        v.addWidget(mg)

        # ── Input folder ───────────────────────────────────────────────────────
        ig = QGroupBox(tr("batch.input_group"))
        iv = QVBoxLayout(ig)
        self._folder_lbl = QLabel(tr("batch.no_folder"))
        self._folder_lbl.setWordWrap(True)
        self._folder_lbl.setStyleSheet("color:#7F8C8D;font-size:10px;")
        iv.addWidget(self._folder_lbl)
        folder_btn = QPushButton(tr("batch.folder_btn"))
        folder_btn.clicked.connect(self._choose_folder)
        iv.addWidget(folder_btn)
        use_proj_btn = QPushButton(tr("batch.use_project_btn"))
        use_proj_btn.setToolTip("Alle Bilder aus dem geöffneten Projekt verarbeiten")
        use_proj_btn.clicked.connect(self._use_project_images)
        iv.addWidget(use_proj_btn)
        self._image_count_lbl = QLabel("")
        self._image_count_lbl.setStyleSheet("color:#7F8C8D;font-size:10px;")
        iv.addWidget(self._image_count_lbl)
        v.addWidget(ig)

        # ── Options ────────────────────────────────────────────────────────────
        og = QGroupBox(tr("batch.filter_group"))
        ov = QVBoxLayout(og)
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel(tr("batch.min_conf_label")))
        self._min_conf_spin = QDoubleSpinBox()
        self._min_conf_spin.setRange(0.0, 1.0)
        self._min_conf_spin.setSingleStep(0.05)
        self._min_conf_spin.setValue(0.0)
        self._min_conf_spin.setToolTip(
            "Ergebnisse unterhalb dieser Konfidenz werden ausgeblendet.\n"
            "0.0 = alle anzeigen."
        )
        conf_row.addWidget(self._min_conf_spin)
        ov.addLayout(conf_row)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel(tr("batch.warn_thresh_label")))
        self._warn_thresh_spin = QDoubleSpinBox()
        self._warn_thresh_spin.setRange(0.0, 1.0)
        self._warn_thresh_spin.setSingleStep(0.05)
        self._warn_thresh_spin.setValue(0.70)
        self._warn_thresh_spin.setToolTip(
            "Ergebnisse unterhalb dieser Konfidenz werden\n"
            "rot hinterlegt und als 'Unsicher' markiert."
        )
        thresh_row.addWidget(self._warn_thresh_spin)
        ov.addLayout(thresh_row)

        cls_row = QHBoxLayout()
        cls_row.addWidget(QLabel(tr("batch.class_filter_label")))
        self._cls_filter_combo = QComboBox()
        self._cls_filter_combo.addItem("Alle")
        cls_row.addWidget(self._cls_filter_combo)
        ov.addLayout(cls_row)
        apply_btn = QPushButton(tr("batch.apply_filter_btn"))
        apply_btn.clicked.connect(self._apply_filter)
        ov.addWidget(apply_btn)

        export_low_btn = QPushButton(tr("batch.export_low_btn"))
        export_low_btn.setToolTip(
            "Alle Ergebnisse unterhalb der Warnschwelle als CSV exportieren.\n"
            "Nützlich für gezielte manuelle Nachprüfung."
        )
        export_low_btn.clicked.connect(self._export_low_confidence)
        ov.addWidget(export_low_btn)
        v.addWidget(og)

        # ── Run ────────────────────────────────────────────────────────────────
        self._run_btn = QPushButton(tr("batch.run_btn"))
        self._run_btn.setFixedHeight(40)
        self._run_btn.setStyleSheet(
            "background:#3FB950;color:white;font-weight:bold;font-size:13px;padding:6px;"
        )
        self._run_btn.clicked.connect(self._start_batch)
        v.addWidget(self._run_btn)

        self._cancel_btn = QPushButton(tr("batch.cancel_btn"))
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
        export_btn = QPushButton(tr("batch.export_csv_btn"))
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
        self._table.setHorizontalHeaderLabels([
            tr("batch.col.filename"), tr("batch.col.class"),
            tr("batch.col.confidence"), tr("batch.col.error"),
        ])
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
        """Populate the model combo with all training-run checkpoints that exist on disk."""
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
        """Load the model currently selected in the combo box."""
        path = self._model_combo.currentData()
        if not path:
            QMessageBox.warning(self, tr("common.no_model"), "Bitte Modell aus Liste wählen.")
            return
        self._load_model_from_path(path)

    def _load_external_model(self) -> None:
        """Open a file chooser to load a .pth model from outside the project."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Modell laden", "", "PyTorch (*.pth)"
        )
        if path:
            self._load_model_from_path(path)

    def _load_model_from_path(self, path: str) -> None:
        """
        Load a checkpoint at *path*, update instance attributes, and refresh the
        class-filter combo. Shows a critical dialog on failure.
        """
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
            QMessageBox.critical(self, tr("common.error"), str(exc))

    # ------------------------------------------------------------------ folder

    def _choose_folder(self) -> None:
        """Open a folder chooser and scan it for supported image files."""
        folder = QFileDialog.getExistingDirectory(self, "Ordner wählen")
        if folder:
            self._set_folder(folder)

    def _use_project_images(self) -> None:
        """Use all images from the currently loaded project as the input list."""
        if not self.project or not self.project.images:
            QMessageBox.information(self, tr("common.no_project"), tr("batch.no_project_msg"))
            return
        self._image_paths = list(self.project.images)
        self._folder_lbl.setText(f"Projektbilder ({len(self._image_paths)} Bilder)")
        self._image_count_lbl.setText(f"{len(self._image_paths)} Bilder bereit")

    def _set_folder(self, folder: str) -> None:
        """Scan *folder* for supported image files and store their paths."""
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
        """Validate model and image list, then start the background batch thread."""
        if self._model is None:
            QMessageBox.warning(self, tr("common.no_model"), "Bitte zuerst ein Modell laden.")
            return
        if not getattr(self, "_image_paths", None):
            QMessageBox.warning(self, tr("common.warning"), tr("batch.no_images_msg"))
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
        """Update the progress bar and status label each time an image is processed."""
        self._progress.setValue(current)
        self._status_lbl.setText(f"Verarbeite {current} / {total} Bilder …")

    @Slot(list)
    def _on_finished(self, results: List[Dict]) -> None:
        """Store results, restore buttons, and populate the results table."""
        self._results = results
        self._progress.setVisible(False)
        self._run_btn.setVisible(True)
        self._cancel_btn.setVisible(False)
        self._show_results(results)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        """Show a critical dialog and restore the UI after a batch error."""
        self._progress.setVisible(False)
        self._run_btn.setVisible(True)
        self._cancel_btn.setVisible(False)
        QMessageBox.critical(self, tr("common.error"), msg)

    def _cancel_batch(self) -> None:
        """Request cancellation of the running batch and update the status label."""
        if self._thread:
            self._thread.cancel()
            self._status_lbl.setText("Wird abgebrochen …")

    # ------------------------------------------------------------------ display

    def _show_results(self, results: List[Dict]) -> None:
        """Populate the results table applying the current confidence/class filters."""
        min_conf = self._min_conf_spin.value()
        warn_thresh = self._warn_thresh_spin.value()
        cls_filter = self._cls_filter_combo.currentText()

        filtered = [
            r for r in results
            if r["confidence"] >= min_conf
            and (cls_filter == "Alle" or r["predicted"] == cls_filter)
        ]

        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(filtered))

        errors = sum(1 for r in filtered if r["error"])
        uncertain = sum(1 for r in filtered if r["confidence"] < warn_thresh and not r["error"])
        counts: Dict[str, int] = {}
        for r in filtered:
            counts[r["predicted"]] = counts.get(r["predicted"], 0) + 1

        warn_bg = QColor(80, 30, 20)
        for row, res in enumerate(filtered):
            is_uncertain = res["confidence"] < warn_thresh and not res["error"]

            self._table.setItem(row, 0, QTableWidgetItem(res["filename"]))
            self._table.setItem(row, 1, QTableWidgetItem(res["predicted"]))

            conf_item = QTableWidgetItem(f"{res['confidence']:.1%}")
            conf_item.setData(Qt.UserRole, res["confidence"])
            if res["confidence"] >= 0.9:
                conf_item.setForeground(QColor("#3FB950"))
            elif res["confidence"] >= warn_thresh:
                conf_item.setForeground(QColor("#D29922"))
            else:
                conf_item.setForeground(QColor("#F85149"))
            self._table.setItem(row, 2, conf_item)

            err_item = QTableWidgetItem(res["error"] or ("⚠ Niedrig" if is_uncertain else ""))
            if res["error"]:
                err_item.setForeground(QColor("#F85149"))
            elif is_uncertain:
                err_item.setForeground(QColor("#D29922"))
            self._table.setItem(row, 3, err_item)

            for col in range(4):
                item = self._table.item(row, col)
                if item:
                    item.setData(Qt.UserRole + 1, res["path"])
                    if is_uncertain:
                        item.setBackground(warn_bg)

        self._table.setSortingEnabled(True)

        summary_parts = [f"{len(filtered)} Bilder"]
        for cls, n in sorted(counts.items()):
            summary_parts.append(f"{cls}: {n}")
        if uncertain:
            summary_parts.append(f"⚠ {uncertain} unsicher (<{warn_thresh:.0%})")
        if errors:
            summary_parts.append(f"✗ {errors} Fehler")
        self._summary_lbl.setText("  |  ".join(summary_parts))
        self._status_lbl.setText(f"Fertig — {len(results)} Bilder verarbeitet")

    def _export_low_confidence(self) -> None:
        """Export only results below the warning threshold to a CSV file."""
        if not self._results:
            QMessageBox.information(self, tr("common.ok"), "Bitte zuerst Batch starten.")
            return
        warn_thresh = self._warn_thresh_spin.value()
        low = [r for r in self._results if r["confidence"] < warn_thresh and not r["error"]]
        if not low:
            QMessageBox.information(
                self, "Keine unsicheren Ergebnisse",
                f"Alle Ergebnisse liegen über der Warnschwelle ({warn_thresh:.0%})."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Unsichere Ergebnisse exportieren",
            "low_confidence.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "path", "predicted", "confidence"])
                for r in low:
                    writer.writerow([r["filename"], r["path"], r["predicted"],
                                     f"{r['confidence']:.4f}"])
            QMessageBox.information(
                self, "Exportiert",
                f"{len(low)} unsichere Ergebnisse gespeichert:\n{path}"
            )
        except Exception as exc:
            QMessageBox.critical(self, tr("common.error"), str(exc))

    def _apply_filter(self) -> None:
        """Re-apply the current filter to the stored results and refresh the table."""
        if self._results:
            self._show_results(self._results)

    # ------------------------------------------------------------------ export

    def _export_csv(self) -> None:
        """Export all batch results (including per-class probabilities) to a CSV file."""
        if not self._results:
            QMessageBox.information(self, tr("common.ok"), "Bitte zuerst Batch starten.")
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
            QMessageBox.information(self, tr("common.ok"), f"CSV gespeichert:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, tr("common.error"), str(exc))
