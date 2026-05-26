"""
Model library page: list, compare, export, delete trained models.
"""
import os
from typing import Optional, List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QMessageBox, QTextEdit, QFileDialog, QInputDialog,
    QTabWidget,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QBrush

from utils.i18n import tr


class _AccuracyChart(QWidget):
    """
    Simple bar chart showing validation accuracy per training run.

    Each run is one vertical bar (blue for normal, green for the best model).
    A thin yellow horizontal line at the top of each bar indicates the F1 score.
    Rendered with QPainter; no external charting library required.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: List[dict] = []   # list of {label, acc, f1, is_best}
        self.setMinimumHeight(160)

    def set_data(self, runs: List[dict]) -> None:
        """Update the chart data and trigger a repaint."""
        self._data = runs
        self.update()

    def paintEvent(self, event) -> None:
        """Draw bars, axis, labels, and the F1 overlay line."""
        if not self._data:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        pad_l, pad_r, pad_t, pad_b = 48, 16, 16, 36
        chart_w = w - pad_l - pad_r
        chart_h = h - pad_t - pad_b

        # Background
        painter.fillRect(0, 0, w, h, QColor("#0D1117"))

        # Axis lines
        pen = QPen(QColor("#30363D"))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(pad_l, pad_t, pad_l, pad_t + chart_h)
        painter.drawLine(pad_l, pad_t + chart_h, pad_l + chart_w, pad_t + chart_h)

        # Y-axis labels (0%, 50%, 100%)
        painter.setPen(QColor("#8B949E"))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        for pct in [0, 50, 100]:
            y = pad_t + chart_h - int(pct / 100 * chart_h)
            painter.drawText(2, y + 4, 42, 12, Qt.AlignRight, f"{pct}%")
            if pct > 0:
                painter.setPen(QPen(QColor("#21262D")))
                painter.drawLine(pad_l, y, pad_l + chart_w, y)
                painter.setPen(QColor("#8B949E"))

        # Bars
        n = len(self._data)
        if n == 0:
            return
        bar_w = max(8, min(40, (chart_w - 4) // n - 4))
        spacing = (chart_w - n * bar_w) // (n + 1)

        for i, entry in enumerate(self._data):
            x = pad_l + spacing + i * (bar_w + spacing)
            acc = min(max(entry.get("acc", 0), 0), 1)
            f1 = min(max(entry.get("f1", 0), 0), 1)
            bar_h_acc = int(acc * chart_h)
            bar_h_f1 = int(f1 * chart_h)

            # Accuracy bar
            color = QColor("#1F6FEB") if not entry.get("is_best") else QColor("#2ECC71")
            painter.fillRect(x, pad_t + chart_h - bar_h_acc, bar_w, bar_h_acc, color)

            # F1 overlay (thin line)
            if f1 > 0:
                painter.setPen(QPen(QColor("#D29922"), 2))
                y_f1 = pad_t + chart_h - bar_h_f1
                painter.drawLine(x, y_f1, x + bar_w, y_f1)
                painter.setPen(Qt.NoPen)

            # Value label on bar
            painter.setPen(QColor("#FFFFFF"))
            if bar_h_acc > 16:
                painter.drawText(
                    x, pad_t + chart_h - bar_h_acc, bar_w, 14,
                    Qt.AlignHCenter, f"{acc*100:.0f}%"
                )

            # X-axis label (run index or short ID)
            painter.setPen(QColor("#8B949E"))
            lbl = entry.get("label", str(i + 1))
            painter.drawText(
                x - 4, pad_t + chart_h + 4, bar_w + 8, 28,
                Qt.AlignHCenter | Qt.AlignTop, lbl
            )

        # Legend
        painter.setPen(QColor("#1F6FEB"))
        painter.fillRect(pad_l, 2, 10, 10, QColor("#1F6FEB"))
        painter.setPen(QColor("#8B949E"))
        painter.drawText(pad_l + 14, 2, 80, 12, Qt.AlignLeft, "Accuracy")
        painter.setPen(QPen(QColor("#D29922"), 2))
        y_leg = 2 + 5
        painter.drawLine(pad_l + 100, y_leg, pad_l + 110, y_leg)
        painter.setPen(QColor("#8B949E"))
        painter.drawText(pad_l + 114, 2, 40, 12, Qt.AlignLeft, "F1")


class ModelsPage(QWidget):
    """
    Model library page (stack index 4).

    Two tabs:
    - Modellbibliothek: table of registered models with detail panel and action
      buttons (load, ONNX export, TorchScript export, rename, archive, delete,
      multi-model comparison).
    - Run-History: chronological table of all training runs plus an
      ``_AccuracyChart`` bar chart comparing accuracy across runs.

    Signals
    -------
    model_loaded : Emitted with the model file path when the user clicks
                   "In Inferenz laden". Consumed by ``InferencePage`` and the
                   REST API server in ``MainWindow``.
    """

    model_loaded = Signal(str)   # model_path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._manager = None
        self._build_ui()

    def set_project(self, project) -> None:
        """Accept a project, initialise the ``ModelManager``, and refresh the view."""
        self.project = project
        self._init_manager()
        self.refresh()

    def _init_manager(self) -> None:
        """Create a ``ModelManager`` pointing at the project's models directory."""
        if self.project:
            from core.model_manager import ModelManager
            self._manager = ModelManager(self.project.get_models_dir())

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        # ── Tab 1: Model Library ──────────────────────────────────────────────
        lib_widget = self._build_library_tab()
        self._tabs.addTab(lib_widget, tr("models.tab.library"))

        # ── Tab 2: Run History / Comparison ──────────────────────────────────
        hist_widget = self._build_history_tab()
        self._tabs.addTab(hist_widget, tr("models.tab.history"))

    def _build_library_tab(self) -> QWidget:
        from PySide6.QtWidgets import QWidget
        w = QWidget()
        splitter = QSplitter(Qt.Horizontal)
        hl = QHBoxLayout(w)
        hl.addWidget(splitter)

        # Left: table
        left = QGroupBox(tr("models.library_group"))
        lv = QVBoxLayout(left)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            tr("models.col.name"), tr("models.col.arch"), tr("models.col.accuracy"),
            tr("models.col.f1"), tr("models.col.classes"),
            tr("models.col.created"), tr("models.col.best")
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.doubleClicked.connect(self._load_selected)
        lv.addWidget(self.table)

        btn_row = QHBoxLayout()
        _btn_tips = {
            tr("models.refresh_btn"): "Modellliste aus dem Projekt neu laden",
            tr("models.mark_best_btn"): "Dieses Modell als Standard für das Projekt setzen.\nWird beim nächsten Öffnen automatisch geladen.",
            tr("models.load_btn"): "Modell auf die Klassifikations-Seite laden\num neue Bilder damit zu bewerten.",
        }
        for label, slot in [
            (tr("models.refresh_btn"), self.refresh),
            (tr("models.mark_best_btn"), self._mark_best),
            (tr("models.load_btn"), self._load_selected),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            if label in _btn_tips:
                btn.setToolTip(_btn_tips[label])
            btn_row.addWidget(btn)
        lv.addLayout(btn_row)
        splitter.addWidget(left)

        # Right: details + actions
        right = QGroupBox(tr("models.detail_group"))
        rv = QVBoxLayout(right)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setFont(QFont("Courier New", 9))
        rv.addWidget(self.detail_text)

        _action_tips = {
            tr("models.export_onnx_btn"): (
                "Exportiert das Modell als ONNX (Opset 17).\n"
                "Einsetzbar in: ONNX Runtime, OpenCV DNN, TensorRT,\n"
                "C++, C#, JavaScript (ONNX.js) und vielen anderen Frameworks."
            ),
            tr("models.export_ts_btn"): (
                "Exportiert als TorchScript (.pt).\n"
                "Für PyTorch C++ API oder mobile Apps (Android/iOS).\n"
                "Kein Python-Import nötig zur Laufzeit."
            ),
            tr("models.rename_btn"): "Modell-Alias im Projekt umbenennen (Dateiname bleibt gleich).",
            tr("models.archive_btn"): "Modell in Unterordner 'archive' verschieben — bleibt erhalten aber\nerscheint nicht mehr in der Hauptliste.",
            tr("models.delete_btn"): "Modell dauerhaft löschen (kann nicht rückgängig gemacht werden).",
        }
        for label, slot in [
            (tr("models.export_onnx_btn"), self._export_onnx),
            (tr("models.export_ts_btn"), self._export_torchscript),
            (tr("models.rename_btn"), self._rename_model),
            (tr("models.archive_btn"), self._archive_model),
            (tr("models.delete_btn"), self._delete_model),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            if label in _action_tips:
                btn.setToolTip(_action_tips[label])
            rv.addWidget(btn)

        rv.addWidget(QLabel("Modell vergleichen:"))
        self.compare_btn = QPushButton(tr("models.compare_btn"))
        self.compare_btn.setToolTip(
            "Mehrere Modelle auswählen (Strg+Klick in der Tabelle)\n"
            "und Accuracy, F1 sowie Architektur nebeneinander vergleichen."
        )
        self.compare_btn.clicked.connect(self._compare_models)
        rv.addWidget(self.compare_btn)

        rv.addWidget(QLabel("Kalibrierung & Edge-Deployment:"))
        for label, slot, tip in [
            (tr("models.calibrate_btn"), self._calibrate_model,
             "Post-hoc Konfidenz-Kalibrierung via Temperature Scaling.\n"
             "Verbessert die Zuverlässigkeit von Konfidenz-Werten.\n"
             "Benötigt: scipy (pip install scipy)"),
            (tr("models.edge_onnx_btn"), self._export_edge_onnx,
             "Exportiert das Modell als INT8-quantisiertes ONNX für Edge-Deployment.\n"
             "Typisch 2–4× kleiner und schneller als FP32 ONNX.\n"
             "Benötigt: onnxruntime.quantization"),
            (tr("models.coreml_btn"), self._export_coreml,
             "Exportiert als Apple CoreML (.mlpackage) für macOS/iOS.\n"
             "Benötigt: coremltools (nur macOS, pip install coremltools)"),
            (tr("models.docker_btn"), self._generate_docker,
             "Erstellt Dockerfile, docker-compose.yml und Startskript\n"
             "für den containerisierten Betrieb von monitor.py."),
        ]:
            btn = QPushButton(label)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            rv.addWidget(btn)

        splitter.addWidget(right)
        splitter.setSizes([600, 400])
        return w

    def _build_history_tab(self) -> QWidget:
        from PySide6.QtWidgets import QWidget, QSplitter
        w = QWidget()
        vl = QVBoxLayout(w)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("Alle Trainingsläufe im Vergleich:"))
        hdr.addStretch()
        refresh_btn = QPushButton(tr("common.refresh"))
        refresh_btn.clicked.connect(self._refresh_history)
        hdr.addWidget(refresh_btn)
        vl.addLayout(hdr)

        splitter = QSplitter(Qt.Vertical)

        self._history_table = QTableWidget(0, 8)
        self._history_table.setHorizontalHeaderLabels([
            "Datum", "Run-ID", "Architektur", "Accuracy", "F1",
            "Train-Acc", "Epochen", "Gerät"
        ])
        self._history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._history_table.setAlternatingRowColors(True)
        splitter.addWidget(self._history_table)

        self._history_chart = _AccuracyChart()
        splitter.addWidget(self._history_chart)
        splitter.setSizes([300, 200])
        vl.addWidget(splitter)

        return w

    # ------------------------------------------------------------------ refresh

    def refresh(self) -> None:
        """Reload the model table and register any new training runs from the project."""
        if not self._manager:
            return
        models = self._manager.get_all(include_archived=False)
        self.table.setRowCount(len(models))
        for row, m in enumerate(models):
            items = [
                m.name, m.architecture,
                m.accuracy_str(), m.f1_str(),
                ", ".join(m.class_names[:3]) + ("…" if len(m.class_names) > 3 else ""),
                m.created_at[:10],
                "★" if m.is_best else "",
            ]
            for col, val in enumerate(items):
                item = QTableWidgetItem(str(val))
                if col == 6 and m.is_best:
                    item.setForeground(QColor("#F39C12"))
                self.table.setItem(row, col, item)
                item.setData(Qt.UserRole, m.model_id)

        # Register any new run results not yet in registry
        if self.project and self.project.training_runs:
            existing_run_ids = {m.run_id for m in self._manager.get_all(include_archived=True)}
            newly_registered = False
            for run in self.project.training_runs:
                if run.get("run_id") not in existing_run_ids:
                    self._manager.register(run)
                    newly_registered = True
            if newly_registered:
                self.refresh()
                return

        self._refresh_history()

    def _refresh_history(self) -> None:
        """Repopulate the run-history table and update the accuracy bar chart."""
        if not self._manager:
            return
        all_models = self._manager.get_all(include_archived=True)
        sorted_models = sorted(all_models, key=lambda x: x.created_at)
        self._history_table.setRowCount(len(sorted_models))
        best_acc = max((m.metrics.get("accuracy", 0) for m in sorted_models), default=0)
        chart_data = []
        for row, m in enumerate(sorted(sorted_models, key=lambda x: x.created_at, reverse=True)):
            acc = m.metrics.get("accuracy", 0)
            f1 = m.metrics.get("macro_f1", 0)
            items = [
                m.created_at[:16],
                m.run_id[:8],
                m.architecture,
                f"{acc*100:.2f}%",
                f"{f1*100:.2f}%",
                f"{m.metrics.get('train_acc', 0)*100:.1f}%" if "train_acc" in m.metrics else "–",
                str(m.hyperparameters.get("epochs", "–")),
                m.hyperparameters.get("device", "–"),
            ]
            for col, val in enumerate(items):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                if col == 3 and acc >= best_acc and acc > 0:
                    item.setForeground(QColor("#2ECC71"))
                self._history_table.setItem(row, col, item)

        # Build chart data (chronological order)
        for i, m in enumerate(sorted_models):
            chart_data.append({
                "label": m.created_at[5:10],   # MM-DD
                "acc":   m.metrics.get("accuracy", 0),
                "f1":    m.metrics.get("macro_f1", 0),
                "is_best": m.is_best,
            })
        if hasattr(self, "_history_chart"):
            self._history_chart.set_data(chart_data)

    def _selected_model_id(self) -> Optional[str]:
        """Return the model ID stored in ``UserRole`` of the currently focused row."""
        row = self.table.currentRow()
        item = self.table.item(row, 0)
        if item:
            return item.data(Qt.UserRole)
        return None

    def _get_selected_model(self):
        """Return the ModelRecord for the currently selected row, or None."""
        mid = self._selected_model_id()
        if not mid or not self._manager:
            return None
        return self._manager.get_by_id(mid)

    def _on_selection_changed(self) -> None:
        """Slot for ``itemSelectionChanged``; delegates to ``_on_row_selected``."""
        self._on_row_selected(self.table.currentRow())

    def _on_row_selected(self, row: int) -> None:
        """Populate the detail text area with metadata for the model at *row*."""
        if not self._manager:
            return
        item = self.table.item(row, 0)
        if not item:
            return
        model_id = item.data(Qt.UserRole)
        m = self._manager.get_by_id(model_id)
        if not m:
            return
        lines = [
            f"Name:         {m.name}",
            f"Architektur:  {m.architecture}",
            f"Version:      {m.version}",
            f"Run-ID:       {m.run_id}",
            f"Erstellt:     {m.created_at[:19]}",
            f"Accuracy:     {m.accuracy_str()}",
            f"F1 (Macro):   {m.f1_str()}",
            f"Klassen:      {', '.join(m.class_names)}",
            f"Bildgröße:    {m.image_size}px",
            f"Train/Val/Test: {m.train_size}/{m.val_size}/{m.test_size}",
            f"Best:         {'Ja' if m.is_best else 'Nein'}",
            f"Archiviert:   {'Ja' if m.archived else 'Nein'}",
            f"Modelldatei:  {m.model_path}",
            f"ONNX:         {m.onnx_path or '–'}",
            "",
            "Hyperparameter:",
        ]
        for k, v in list(m.hyperparameters.items())[:10]:
            lines.append(f"  {k}: {v}")
        self.detail_text.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------ actions

    def _mark_best(self) -> None:
        """Mark the selected model as the project's best and update its current model path."""
        mid = self._selected_model_id()
        if mid and self._manager:
            self._manager.mark_as_best(mid)
            # Update project's current model
            m = self._manager.get_by_id(mid)
            if m and self.project:
                self.project.current_model_path = m.model_path
            self.refresh()

    def _load_selected(self) -> None:
        """Emit ``model_loaded`` with the path of the selected model file."""
        mid = self._selected_model_id()
        if not mid or not self._manager:
            return
        m = self._manager.get_by_id(mid)
        if not m:
            return
        if not os.path.exists(m.model_path):
            QMessageBox.critical(self, tr("common.error"),
                                 tr("models.file_not_found", path=m.model_path))
            return
        self.model_loaded.emit(m.model_path)
        QMessageBox.information(self, tr("models.loaded_title"),
                                tr("models.loaded_msg", name=m.name))

    def _export_onnx(self) -> None:
        mid = self._selected_model_id()
        if not mid or not self._manager:
            return
        try:
            path = self._manager.export_onnx(mid)
            QMessageBox.information(self, tr("models.onnx_success"), tr("models.onnx_saved", path=path))
            self.refresh()
        except Exception as exc:
            QMessageBox.critical(self, tr("common.error"), str(exc))

    def _export_torchscript(self) -> None:
        mid = self._selected_model_id()
        if not mid or not self._manager:
            return
        try:
            path = self._manager.export_torchscript(mid)
            QMessageBox.information(self, tr("models.ts_success"), f"Gespeichert:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, tr("common.error"), str(exc))

    def _rename_model(self) -> None:
        mid = self._selected_model_id()
        if not mid or not self._manager:
            return
        m = self._manager.get_by_id(mid)
        if not m:
            return
        new_name, ok = QInputDialog.getText(self, tr("models.rename_dlg"), tr("models.rename_prompt"), text=m.name)
        if ok and new_name.strip():
            self._manager.update_metadata(mid, name=new_name.strip())
            self.refresh()

    def _archive_model(self) -> None:
        mid = self._selected_model_id()
        if not mid or not self._manager:
            return
        reply = QMessageBox.question(self, tr("models.archive_title"),
                                     tr("models.archive_msg"),
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self._manager.archive(mid)
            self.refresh()

    def _delete_model(self) -> None:
        mid = self._selected_model_id()
        if not mid or not self._manager:
            return
        m = self._manager.get_by_id(mid)
        if not m:
            return
        reply = QMessageBox.question(
            self, tr("models.delete_title"),
            tr("models.delete_msg", name=m.name),
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._manager.delete(mid, delete_file=False)
            self.refresh()

    def _compare_models(self) -> None:
        """Compare selected models in ModelComparisonDialog."""
        if not self._manager:
            return
        selected = []
        for item in self.table.selectedItems():
            if item.column() == 0:
                selected.append(item.data(Qt.UserRole))

        if len(selected) < 2:
            QMessageBox.information(self, tr("models.compare_btn"),
                                    tr("models.compare_prompt"))
            return
        runs = self._manager.compare(selected)
        from gui.dialogs.model_comparison_dialog import ModelComparisonDialog
        dlg = ModelComparisonDialog(runs, parent=self)
        dlg.exec()

    def _calibrate_model(self) -> None:
        """Calibrate the selected model's confidence with Temperature Scaling."""
        mid = self._selected_model_id()
        if not mid or not self._manager:
            return
        m = self._manager.get_by_id(mid)
        if not m or not os.path.exists(m.model_path):
            QMessageBox.warning(self, tr("models.calibrate_btn"), "Modelldatei nicht gefunden.")
            return
        if not self.project:
            QMessageBox.warning(self, tr("models.calibrate_btn"), "Kein Projekt geladen.")
            return
        try:
            from core.calibration import TemperatureScaler
        except ImportError:
            QMessageBox.warning(self, tr("models.scipy_missing"),
                                tr("models.scipy_install"))
            return
        import json, os as _os
        cal_path = _os.path.splitext(m.model_path)[0] + "_calibration.json"
        scaler = TemperatureScaler()
        if _os.path.exists(cal_path):
            scaler.load(cal_path)
            QMessageBox.information(
                self, tr("models.calibrate_btn"),
                f"Bestehende Kalibrierung geladen:\nTemperatur = {scaler.temperature:.4f}\n\n"
                f"Datei: {cal_path}\n\n"
                "Um neu zu kalibrieren, löschen Sie die Kalibrierungsdatei und\n"
                "starten Sie nach einem Inferenz-Lauf erneut."
            )
        else:
            QMessageBox.information(
                self, tr("models.calibrate_btn"),
                "Temperature Scaling kalibriert das Modell auf Validierungsdaten.\n\n"
                "Voraussetzung: Führen Sie zuerst eine Batch-Inferenz durch,\n"
                "damit Logits und Labels verfügbar sind.\n\n"
                f"Kalibrierungsdatei wird gespeichert unter:\n{cal_path}"
            )

    def _export_edge_onnx(self) -> None:
        """Export selected model as INT8-quantised ONNX."""
        mid = self._selected_model_id()
        if not mid or not self._manager:
            return
        m = self._manager.get_by_id(mid)
        if not m or not os.path.exists(m.model_path):
            QMessageBox.warning(self, tr("common.warning"), "Modelldatei nicht gefunden.")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, tr("models.edge_onnx_btn"), m.name + "_int8.onnx", "ONNX (*.onnx)"
        )
        if not out_path:
            return
        try:
            from core.edge_export import EdgeExporter
            exporter = EdgeExporter()
            result = exporter.export_quantized_onnx(m.model_path, out_path,
                                                    image_size=m.image_size or 224)
            QMessageBox.information(self, tr("models.onnx_success"), tr("models.onnx_saved", path=result))
        except Exception as exc:
            QMessageBox.critical(self, tr("common.error"), str(exc))

    def _export_coreml(self) -> None:
        """Export selected model as CoreML (macOS only)."""
        mid = self._selected_model_id()
        if not mid or not self._manager:
            return
        m = self._manager.get_by_id(mid)
        if not m or not os.path.exists(m.model_path):
            QMessageBox.warning(self, tr("common.warning"), "Modelldatei nicht gefunden.")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, tr("models.coreml_btn"), m.name + ".mlpackage", "CoreML (*.mlpackage)"
        )
        if not out_path:
            return
        try:
            from core.edge_export import EdgeExporter
            exporter = EdgeExporter()
            result = exporter.export_coreml(m.model_path, out_path,
                                            image_size=m.image_size or 224)
            QMessageBox.information(self, "CoreML exportiert", f"Gespeichert:\n{result}")
        except ImportError as exc:
            QMessageBox.warning(self, tr("models.coreml_missing"), str(exc))
        except Exception as exc:
            QMessageBox.critical(self, tr("common.error"), str(exc))

    def _generate_docker(self) -> None:
        """Generate Docker deployment files for monitor.py."""
        if not self._manager:
            return
        mid = self._selected_model_id()
        model_path = ""
        if mid:
            m = self._manager.get_by_id(mid)
            if m:
                model_path = m.model_path

        out_dir = QFileDialog.getExistingDirectory(
            self, tr("models.docker_btn")
        )
        if not out_dir:
            return
        try:
            from core.docker_generator import DockerGenerator
            files = DockerGenerator().generate(out_dir, model_path=model_path)
            QMessageBox.information(
                self, tr("models.docker_success"),
                f"Folgende Dateien wurden erstellt:\n" + "\n".join(
                    f"  • {os.path.basename(f)}" for f in files
                ) + f"\n\nOrdner: {out_dir}"
            )
        except Exception as exc:
            QMessageBox.critical(self, tr("common.error"), str(exc))
