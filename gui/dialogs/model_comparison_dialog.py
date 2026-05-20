from __future__ import annotations
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QLabel, QPushButton, QDialogButtonBox,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont


class ModelComparisonDialog(QDialog):
    """
    Zeigt ausgewählte Trainings-Runs in einer sortierbaren Tabelle nebeneinander.
    runs: list[dict] mit keys: name, accuracy, macro_f1, architecture, is_best
    """

    def __init__(self, runs: list, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Modellvergleich")
        self.setMinimumSize(640, 320)
        self._runs = runs
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        hdr = QLabel(f"Vergleich: {len(self._runs)} Modelle")
        hdr.setStyleSheet("font-size: 14px; font-weight: bold; color: #388BFD; padding: 4px;")
        layout.addWidget(hdr)

        self.table = QTableWidget(len(self._runs), 5)
        self.table.setHorizontalHeaderLabels(["Modell", "Acc %", "F1 %", "Architektur", "★"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 5):
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.setSortingEnabled(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        GOLD = QColor("#3D2B00")
        for row, run in enumerate(self._runs):
            acc = run.get("accuracy", 0.0) * 100
            f1  = run.get("macro_f1", 0.0) * 100
            is_best = bool(run.get("is_best", False))
            items = [
                QTableWidgetItem(run.get("name", f"Run {row}")),
                QTableWidgetItem(f"{acc:.2f}"),
                QTableWidgetItem(f"{f1:.2f}"),
                QTableWidgetItem(run.get("architecture", "?")),
                QTableWidgetItem("★" if is_best else ""),
            ]
            for col_idx in (1, 2):
                items[col_idx].setData(Qt.UserRole, float(items[col_idx].text()))
            for col_idx, item in enumerate(items):
                if is_best:
                    item.setBackground(GOLD)
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                item.setTextAlignment(Qt.AlignCenter if col_idx > 0 else Qt.AlignLeft | Qt.AlignVCenter)
                self.table.setItem(row, col_idx, item)

        self.table.setSortingEnabled(True)
        layout.addWidget(self.table)

        footer = QLabel("★ = Als bestes Modell markiert  |  Klick auf Spaltenheader = Sortieren")
        footer.setStyleSheet("color: #8B949E; font-size: 10px; padding: 2px;")
        layout.addWidget(footer)

        btn = QDialogButtonBox(QDialogButtonBox.Close)
        btn.rejected.connect(self.reject)
        layout.addWidget(btn)
