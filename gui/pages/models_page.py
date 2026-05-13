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


class _AccuracyChart(QWidget):
    """Simple bar chart showing validation accuracy per training run."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: List[dict] = []   # list of {label, acc, f1, is_best}
        self.setMinimumHeight(160)

    def set_data(self, runs: List[dict]) -> None:
        self._data = runs
        self.update()

    def paintEvent(self, event) -> None:
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
    model_loaded = Signal(str)   # model_path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._manager = None
        self._build_ui()

    def set_project(self, project) -> None:
        self.project = project
        self._init_manager()
        self.refresh()

    def _init_manager(self) -> None:
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
        self._tabs.addTab(lib_widget, "📦 Modellbibliothek")

        # ── Tab 2: Run History / Comparison ──────────────────────────────────
        hist_widget = self._build_history_tab()
        self._tabs.addTab(hist_widget, "📊 Run-History")

    def _build_library_tab(self) -> QWidget:
        from PySide6.QtWidgets import QWidget
        w = QWidget()
        splitter = QSplitter(Qt.Horizontal)
        hl = QHBoxLayout(w)
        hl.addWidget(splitter)

        # Left: table
        left = QGroupBox("Modellbibliothek")
        lv = QVBoxLayout(left)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "Name", "Architektur", "Accuracy", "F1", "Klassen",
            "Erstellt", "Best"
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
            "Aktualisieren": "Modellliste aus dem Projekt neu laden",
            "Als Best markieren": "Dieses Modell als Standard für das Projekt setzen.\nWird beim nächsten Öffnen automatisch geladen.",
            "In Inferenz laden": "Modell auf die Klassifikations-Seite laden\num neue Bilder damit zu bewerten.",
        }
        for label, slot in [
            ("Aktualisieren", self.refresh),
            ("Als Best markieren", self._mark_best),
            ("In Inferenz laden", self._load_selected),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            if label in _btn_tips:
                btn.setToolTip(_btn_tips[label])
            btn_row.addWidget(btn)
        lv.addLayout(btn_row)
        splitter.addWidget(left)

        # Right: details + actions
        right = QGroupBox("Modelldetails & Aktionen")
        rv = QVBoxLayout(right)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setFont(QFont("Courier New", 9))
        rv.addWidget(self.detail_text)

        _action_tips = {
            "Als ONNX exportieren": (
                "Exportiert das Modell als ONNX (Opset 17).\n"
                "Einsetzbar in: ONNX Runtime, OpenCV DNN, TensorRT,\n"
                "C++, C#, JavaScript (ONNX.js) und vielen anderen Frameworks."
            ),
            "Als TorchScript exportieren": (
                "Exportiert als TorchScript (.pt).\n"
                "Für PyTorch C++ API oder mobile Apps (Android/iOS).\n"
                "Kein Python-Import nötig zur Laufzeit."
            ),
            "Umbenennen": "Modell-Alias im Projekt umbenennen (Dateiname bleibt gleich).",
            "Archivieren": "Modell in Unterordner 'archive' verschieben — bleibt erhalten aber\nerscheint nicht mehr in der Hauptliste.",
            "Löschen": "Modell dauerhaft löschen (kann nicht rückgängig gemacht werden).",
        }
        for label, slot in [
            ("Als ONNX exportieren", self._export_onnx),
            ("Als TorchScript exportieren", self._export_torchscript),
            ("Umbenennen", self._rename_model),
            ("Archivieren", self._archive_model),
            ("Löschen", self._delete_model),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            if label in _action_tips:
                btn.setToolTip(_action_tips[label])
            rv.addWidget(btn)

        rv.addWidget(QLabel("Modell vergleichen:"))
        self.compare_btn = QPushButton("Ausgewählte vergleichen")
        self.compare_btn.setToolTip(
            "Mehrere Modelle auswählen (Strg+Klick in der Tabelle)\n"
            "und Accuracy, F1 sowie Architektur nebeneinander vergleichen."
        )
        self.compare_btn.clicked.connect(self._compare_models)
        rv.addWidget(self.compare_btn)
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
        refresh_btn = QPushButton("Aktualisieren")
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
        row = self.table.currentRow()
        item = self.table.item(row, 0)
        if item:
            return item.data(Qt.UserRole)
        return None

    def _on_selection_changed(self) -> None:
        self._on_row_selected(self.table.currentRow())

    def _on_row_selected(self, row: int) -> None:
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
        mid = self._selected_model_id()
        if mid and self._manager:
            self._manager.mark_as_best(mid)
            # Update project's current model
            m = self._manager.get_by_id(mid)
            if m and self.project:
                self.project.current_model_path = m.model_path
            self.refresh()

    def _load_selected(self) -> None:
        mid = self._selected_model_id()
        if not mid or not self._manager:
            return
        m = self._manager.get_by_id(mid)
        if m and os.path.exists(m.model_path):
            self.model_loaded.emit(m.model_path)
            QMessageBox.information(self, "Geladen",
                                    f"Modell in Inferenz-Panel geladen:\n{m.name}")

    def _export_onnx(self) -> None:
        mid = self._selected_model_id()
        if not mid or not self._manager:
            return
        try:
            path = self._manager.export_onnx(mid)
            QMessageBox.information(self, "ONNX exportiert", f"Gespeichert:\n{path}")
            self.refresh()
        except Exception as exc:
            QMessageBox.critical(self, "ONNX-Fehler", str(exc))

    def _export_torchscript(self) -> None:
        mid = self._selected_model_id()
        if not mid or not self._manager:
            return
        try:
            path = self._manager.export_torchscript(mid)
            QMessageBox.information(self, "TorchScript exportiert", f"Gespeichert:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "TorchScript-Fehler", str(exc))

    def _rename_model(self) -> None:
        mid = self._selected_model_id()
        if not mid or not self._manager:
            return
        m = self._manager.get_by_id(mid)
        if not m:
            return
        new_name, ok = QInputDialog.getText(self, "Umbenennen", "Neuer Name:", text=m.name)
        if ok and new_name.strip():
            self._manager.update_metadata(mid, name=new_name.strip())
            self.refresh()

    def _archive_model(self) -> None:
        mid = self._selected_model_id()
        if not mid or not self._manager:
            return
        reply = QMessageBox.question(self, "Archivieren",
                                     "Modell archivieren? (Es bleibt erhalten, wird aber nicht mehr angezeigt.)",
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
            self, "Löschen",
            f"Modell '{m.name}' aus der Bibliothek entfernen?\n"
            f"(Modelldatei auf Disk bleibt erhalten)",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._manager.delete(mid, delete_file=False)
            self.refresh()

    def _compare_models(self) -> None:
        if not self._manager:
            return
        selected = []
        for item in self.table.selectedItems():
            if item.column() == 0:
                selected.append(item.data(Qt.UserRole))

        if len(selected) < 2:
            QMessageBox.information(self, "Vergleich",
                                    "Bitte mindestens 2 Modelle auswählen (Strg+Klick).")
            return
        rows = self._manager.compare(selected)
        lines = ["Modellvergleich:\n"]
        for r in rows:
            lines.append(
                f"  {r['name']:<25} Acc={r['accuracy']*100:.2f}%  "
                f"F1={r['f1']*100:.2f}%  Arch={r['architecture']}  "
                f"{'★ BEST' if r['is_best'] else ''}"
            )
        QMessageBox.information(self, "Vergleich", "\n".join(lines))
