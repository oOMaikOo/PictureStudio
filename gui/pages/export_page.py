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
from utils.i18n import tr


class ExportPage(QWidget):
    """
    Excel export page (stack index 6).

    Lets the user:
    - Pull inference results from the project's last classification run.
    - Choose a target Excel file (new or existing).
    - Configure sheet name and append vs. overwrite mode.
    - Toggle and rename individual result columns via a mapping table.
    - Trigger ``core.export.export_results_to_excel`` and view a log.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._results: List[Dict] = []
        self._col_defs: List[Dict] = [dict(c) for c in DEFAULT_COLUMNS]
        self._build_ui()

    def set_project(self, project) -> None:
        """Accept the active project and pre-fill the result count if results exist."""
        self.project = project
        if project and project.inference_results:
            self._results = project.inference_results
            self.count_label.setText(f"{len(self._results)} Ergebnisse aus letzter Inferenz")  # log

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Source
        src_group = QGroupBox(tr("export.source_group"))
        sv = QVBoxLayout(src_group)
        self.count_label = QLabel(tr("export.no_results"))
        sv.addWidget(self.count_label)
        load_btn = QPushButton(tr("export.load_btn"))
        load_btn.setToolTip(
            "Übernimmt die Klassifikationsergebnisse der letzten\n"
            "Batch-Klassifikation von der Inferenz-Seite."
        )
        load_btn.clicked.connect(self._load_from_project)
        sv.addWidget(load_btn)
        layout.addWidget(src_group)

        # Target
        tgt_group = QGroupBox(tr("export.target_group"))
        tv = QHBoxLayout(tgt_group)
        self.file_label = QLabel(tr("export.no_file"))
        self.file_label.setWordWrap(True)
        tv.addWidget(self.file_label)
        choose_btn = QPushButton(tr("export.choose_file_btn"))
        choose_btn.clicked.connect(self._choose_file)
        tv.addWidget(choose_btn)
        new_btn = QPushButton(tr("export.new_file_btn"))
        new_btn.clicked.connect(self._new_file)
        tv.addWidget(new_btn)
        layout.addWidget(tgt_group)

        # Sheet + mode
        opt_group = QGroupBox(tr("export.options_group"))
        ov = QHBoxLayout(opt_group)
        ov.addWidget(QLabel(tr("export.sheet_label")))
        self.sheet_edit = QLineEdit("Ergebnisse")
        self.sheet_edit.setFixedWidth(160)
        ov.addWidget(self.sheet_edit)
        self.append_cb = QCheckBox(tr("export.append_cb"))
        self.append_cb.setToolTip(
            "Aktiviert: neue Zeilen werden unterhalb vorhandener Daten eingefügt.\n"
            "Deaktiviert: Datei wird neu erstellt / Tabellenblatt überschrieben."
        )
        ov.addWidget(self.append_cb)
        ov.addStretch()
        layout.addWidget(opt_group)

        # Column mapping
        col_group = QGroupBox(tr("export.columns_group"))
        cv = QVBoxLayout(col_group)
        self.col_table = QTableWidget(len(self._col_defs), 3)
        self.col_table.setHorizontalHeaderLabels(["Aktiv", "Datenwert", "Spaltenname"])
        self.col_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._populate_col_table()
        cv.addWidget(self.col_table)
        layout.addWidget(col_group)

        # Format selector + export button
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel(tr("export.format_label")))
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(["Excel (.xlsx)", "CSV (.csv)", "JSON (.json)"])
        self.fmt_combo.setFixedWidth(160)
        fmt_row.addWidget(self.fmt_combo)
        fmt_row.addStretch()
        layout.addLayout(fmt_row)

        exp_btn = QPushButton(tr("export.export_btn"))
        exp_btn.setStyleSheet("background:#2ECC71;color:white;font-weight:bold;padding:8px;")
        exp_btn.setToolTip(
            "Exportiert die Klassifikationsergebnisse im gewählten Format.\n"
            "Excel benötigt: pip install openpyxl"
        )
        exp_btn.clicked.connect(self._export)
        layout.addWidget(exp_btn)

        # Protocol
        proto_group = QGroupBox(tr("export.protocol_group"))
        pv = QVBoxLayout(proto_group)
        self.proto_text = QTextEdit()
        self.proto_text.setReadOnly(True)
        self.proto_text.setMaximumHeight(120)
        pv.addWidget(self.proto_text)
        layout.addWidget(proto_group)

    def _populate_col_table(self) -> None:
        """Fill the column-mapping table with the default column definitions."""
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
        """Read the current column-mapping table and return a list of column definition dicts."""
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
        """Copy the project's stored inference results into the page for export."""
        if not self.project or not self.project.inference_results:
            QMessageBox.information(self, tr("common.info"),
                                    tr("export.no_data_msg"))
            return
        self._results = self.project.inference_results
        self.count_label.setText(f"{len(self._results)} Ergebnisse geladen")  # log

    def _choose_file(self) -> None:
        """Select an existing file to append data to (format matches combo selection)."""
        fmt = self.fmt_combo.currentIndex()
        if fmt == 1:
            f, _ = QFileDialog.getOpenFileName(self, "CSV-Datei wählen", "", "CSV (*.csv)")
        elif fmt == 2:
            f, _ = QFileDialog.getOpenFileName(self, "JSON-Datei wählen", "", "JSON (*.json)")
        else:
            f, _ = QFileDialog.getOpenFileName(self, "Excel-Datei wählen", "", "Excel (*.xlsx *.xls)")
        if f:
            self.file_label.setText(f)
            self.append_cb.setChecked(fmt == 0)  # append only makes sense for Excel

    def _new_file(self) -> None:
        """Prompt for a new file path and switch to overwrite mode."""
        fmt = self.fmt_combo.currentIndex()
        if fmt == 1:
            f, _ = QFileDialog.getSaveFileName(self, "Neue CSV-Datei", "ergebnisse.csv", "CSV (*.csv)")
        elif fmt == 2:
            f, _ = QFileDialog.getSaveFileName(self, "Neue JSON-Datei", "ergebnisse.json", "JSON (*.json)")
        else:
            f, _ = QFileDialog.getSaveFileName(self, "Neue Excel-Datei", "ergebnisse.xlsx", "Excel (*.xlsx)")
        if f:
            self.file_label.setText(f)
            self.append_cb.setChecked(False)

    def _export(self) -> None:
        """Validate inputs, call the appropriate export function, and log the outcome."""
        path = self.file_label.text()
        if not path or path == tr("export.no_file"):
            QMessageBox.warning(self, tr("common.warning"), tr("export.no_file_msg"))
            return
        if not self._results:
            QMessageBox.warning(self, tr("common.warning"), tr("export.no_data_msg"))
            return

        col_defs = self._get_col_defs()
        model_name = os.path.basename(self.project.current_model_path) if self.project else ""
        fmt = self.fmt_combo.currentIndex()

        try:
            if fmt == 1:
                from core.export import export_results_to_csv
                export_results_to_csv(self._results, path, model_name=model_name, column_defs=col_defs)
                msg = f"CSV exportiert:\n  Datei: {path}\n  Zeilen: {len(self._results)}"
            elif fmt == 2:
                from core.export import export_results_to_json
                export_results_to_json(self._results, path, model_name=model_name, column_defs=col_defs)
                msg = f"JSON exportiert:\n  Datei: {path}\n  Einträge: {len(self._results)}"
            else:
                from core.export import export_results_to_excel
                sheet = self.sheet_edit.text().strip() or "Ergebnisse"
                append = self.append_cb.isChecked()
                export_results_to_excel(
                    self._results, path,
                    model_name=model_name,
                    column_defs=col_defs,
                    sheet_name=sheet,
                    append_mode=append,
                )
                msg = (
                    f"Excel exportiert:\n  Datei: {path}\n"
                    f"  Blatt: {sheet}\n  Zeilen: {len(self._results)}\n"
                    f"  Modus: {'Anhängen' if append else 'Überschreiben'}"
                )
            self.proto_text.append(msg)
            QMessageBox.information(self, tr("common.exported"), msg)
        except Exception as exc:
            err = f"Exportfehler: {exc}"
            self.proto_text.append(err)
            QMessageBox.critical(self, tr("common.error"), str(exc))
