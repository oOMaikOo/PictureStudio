"""
Model library page: list, compare, export, delete trained models.
"""
import os
from typing import Optional, List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QMessageBox, QTextEdit, QFileDialog, QInputDialog,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont


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
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

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
        for label, slot in [
            ("Aktualisieren", self.refresh),
            ("Als Best markieren", self._mark_best),
            ("In Inferenz laden", self._load_selected),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
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

        for label, slot in [
            ("Als ONNX exportieren", self._export_onnx),
            ("Umbenennen", self._rename_model),
            ("Archivieren", self._archive_model),
            ("Löschen", self._delete_model),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            rv.addWidget(btn)

        rv.addWidget(QLabel("Modell vergleichen:"))
        self.compare_btn = QPushButton("Ausgewählte vergleichen")
        self.compare_btn.clicked.connect(self._compare_models)
        rv.addWidget(self.compare_btn)
        splitter.addWidget(right)
        splitter.setSizes([600, 400])

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

        # Also register any new run results not yet in registry
        if self.project and self.project.training_runs:
            # Register runs not in manager
            existing_run_ids = {m.run_id for m in self._manager.get_all(include_archived=True)}
            for run in self.project.training_runs:
                if run.get("run_id") not in existing_run_ids:
                    self._manager.register(run)
            # Refresh again if new models were added
            self.refresh()

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
