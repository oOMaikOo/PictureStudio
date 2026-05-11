"""
Excel export page: custom column mapping, append/overwrite, sheet selection.
"""
import os
from typing import List, Dict

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QCheckBox, QLineEdit, QMessageBox, QTextEdit,
)
from PySide6.QtCore import Qt

from core.export import DEFAULT_COLUMNS


class ExportPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._results: List[Dict] = []
        self._col_defs: List[Dict] = [dict(c) for c in DEFAULT_COLUMNS]
        self._build_ui()

    def set_project(self, project) -> None:
        self.project = project
        if project and project.inference_results:
            self._results = project.inference_results
            self.count_label.setText(f"{len(self._results)} Ergebnisse aus letzter Inferenz")

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Source
        src_group = QGroupBox("Datenquelle")
        sv = QVBoxLayout(src_group)
        self.count_label = QLabel("Keine Ergebnisse vorhanden.")
        sv.addWidget(self.count_label)
        load_btn = QPushButton("Ergebnisse aus letzter Inferenz laden")
        load_btn.clicked.connect(self._load_from_project)
        sv.addWidget(load_btn)
        layout.addWidget(src_group)

        # Target
        tgt_group = QGroupBox("Zieldatei")
        tv = QHBoxLayout(tgt_group)
        self.file_label = QLabel("Keine Datei gewählt")
        self.file_label.setWordWrap(True)
        tv.addWidget(self.file_label)
        choose_btn = QPushButton("Datei wählen…")
        choose_btn.clicked.connect(self._choose_file)
        tv.addWidget(choose_btn)
        new_btn = QPushButton("Neue Datei erstellen")
        new_btn.clicked.connect(self._new_file)
        tv.addWidget(new_btn)
        layout.addWidget(tgt_group)

        # Sheet + mode
        opt_group = QGroupBox("Optionen")
        ov = QHBoxLayout(opt_group)
        ov.addWidget(QLabel("Tabellenblatt:"))
        self.sheet_edit = QLineEdit("Ergebnisse")
        self.sheet_edit.setFixedWidth(160)
        ov.addWidget(self.sheet_edit)
        self.append_cb = QCheckBox("Anhängen (nicht überschreiben)")
        ov.addWidget(self.append_cb)
        ov.addStretch()
        layout.addWidget(opt_group)

        # Column mapping
        col_group = QGroupBox("Spaltenzuordnung")
        cv = QVBoxLayout(col_group)
        self.col_table = QTableWidget(len(self._col_defs), 3)
        self.col_table.setHorizontalHeaderLabels(["Aktiv", "Datenwert", "Spaltenname"])
        self.col_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._populate_col_table()
        cv.addWidget(self.col_table)
        layout.addWidget(col_group)

        # Export button
        exp_btn = QPushButton("Excel exportieren")
        exp_btn.setStyleSheet("background:#2ECC71;color:white;font-weight:bold;padding:8px;")
        exp_btn.clicked.connect(self._export)
        layout.addWidget(exp_btn)

        # Protocol
        proto_group = QGroupBox("Export-Protokoll")
        pv = QVBoxLayout(proto_group)
        self.proto_text = QTextEdit()
        self.proto_text.setReadOnly(True)
        self.proto_text.setMaximumHeight(120)
        pv.addWidget(self.proto_text)
        layout.addWidget(proto_group)

    def _populate_col_table(self) -> None:
        for row, col in enumerate(self._col_defs):
            # Active checkbox
            cb = QCheckBox()
            cb.setChecked(col.get("enabled", True))
            cb_widget = QWidget()
            cb_layout = QHBoxLayout(cb_widget)
            cb_layout.addWidget(cb)
            cb_layout.setAlignment(Qt.AlignCenter)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            self.col_table.setCellWidget(row, 0, cb_widget)
            cb.toggled.connect(lambda checked, r=row: self._col_defs.__setitem__(
                r, {**self._col_defs[r], "enabled": checked}
            ))

            key_item = QTableWidgetItem(col["key"])
            key_item.setFlags(Qt.ItemIsEnabled)
            self.col_table.setItem(row, 1, key_item)

            header_item = QTableWidgetItem(col["header"])
            self.col_table.setItem(row, 2, header_item)

    def _get_col_defs(self) -> List[Dict]:
        defs = []
        for row in range(self.col_table.rowCount()):
            cb_widget = self.col_table.cellWidget(row, 0)
            cb = cb_widget.findChild(QCheckBox)
            key_item = self.col_table.item(row, 1)
            header_item = self.col_table.item(row, 2)
            if key_item and header_item:
                defs.append({
                    "key": key_item.text(),
                    "header": header_item.text(),
                    "enabled": cb.isChecked() if cb else True,
                })
        return defs

    # ------------------------------------------------------------------ actions

    def _load_from_project(self) -> None:
        if not self.project or not self.project.inference_results:
            QMessageBox.information(self, "Keine Daten",
                                    "Zuerst im Inferenz-Panel Bilder klassifizieren.")
            return
        self._results = self.project.inference_results
        self.count_label.setText(f"{len(self._results)} Ergebnisse geladen")

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Excel-Datei wählen", "", "Excel (*.xlsx *.xls)"
        )
        if path:
            self.file_label.setText(path)
            self.append_cb.setChecked(True)

    def _new_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Neue Excel-Datei", "ergebnisse.xlsx", "Excel (*.xlsx)"
        )
        if path:
            self.file_label.setText(path)
            self.append_cb.setChecked(False)

    def _export(self) -> None:
        path = self.file_label.text()
        if not path or path == "Keine Datei gewählt":
            QMessageBox.warning(self, "Keine Datei", "Bitte zuerst eine Zieldatei wählen.")
            return
        if not self._results:
            QMessageBox.warning(self, "Keine Daten", "Keine Ergebnisse zum Exportieren.")
            return

        col_defs = self._get_col_defs()
        sheet = self.sheet_edit.text().strip() or "Ergebnisse"
        append = self.append_cb.isChecked()

        try:
            from core.export import export_results_to_excel
            from core.model_manager import ModelManager
            model_name = ""
            if self.project:
                model_name = os.path.basename(self.project.current_model_path)

            export_results_to_excel(
                self._results, path,
                model_name=model_name,
                column_defs=col_defs,
                sheet_name=sheet,
                append_mode=append,
            )
            msg = (
                f"Erfolgreich exportiert:\n"
                f"  Datei: {path}\n"
                f"  Blatt: {sheet}\n"
                f"  Zeilen: {len(self._results)}\n"
                f"  Modus: {'Anhängen' if append else 'Überschreiben'}"
            )
            self.proto_text.append(msg)
            QMessageBox.information(self, "Exportiert", msg)
        except Exception as exc:
            err = f"Exportfehler: {exc}"
            self.proto_text.append(err)
            QMessageBox.critical(self, "Exportfehler", str(exc))
