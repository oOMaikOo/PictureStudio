"""
Inference panel: load model, classify images/folders, show results.
"""
import os
from typing import List, Dict, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QMessageBox, QSplitter, QAbstractItemView,
    QLineEdit,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPixmap

from core.inference import Inferencer
from utils.config import IMAGE_FORMATS


class InferenceThread(QThread):
    progress = Signal(int, int)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, inferencer: Inferencer, folder: str):
        super().__init__()
        self.inferencer = inferencer
        self.folder = folder

    def run(self) -> None:
        try:
            results = self.inferencer.predict_folder(
                self.folder,
                progress_callback=lambda cur, tot: self.progress.emit(cur, tot),
            )
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


class InferencePanel(QWidget):
    """Full inference panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.inferencer = Inferencer()
        self._thread: Optional[InferenceThread] = None
        self._results: List[Dict] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        splitter.addWidget(self._build_control_panel())
        splitter.addWidget(self._build_results_panel())
        splitter.setSizes([300, 700])

    def _build_control_panel(self) -> QGroupBox:
        box = QGroupBox("Inferenz-Steuerung")
        v = QVBoxLayout(box)

        # Model
        model_group = QGroupBox("Modell")
        mg = QVBoxLayout(model_group)
        self.model_path_label = QLabel("Kein Modell geladen")
        self.model_path_label.setWordWrap(True)
        mg.addWidget(self.model_path_label)
        load_model_btn = QPushButton("Modell laden (.pth)")
        load_model_btn.clicked.connect(self._load_model)
        mg.addWidget(load_model_btn)
        self.model_info_label = QLabel("")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setStyleSheet("color: #27AE60; font-size: 10px;")
        mg.addWidget(self.model_info_label)
        v.addWidget(model_group)

        # Input
        input_group = QGroupBox("Eingabe")
        ig = QVBoxLayout(input_group)

        single_btn = QPushButton("Einzelbild klassifizieren")
        single_btn.clicked.connect(self._classify_single)
        ig.addWidget(single_btn)

        folder_row = QHBoxLayout()
        self.folder_label = QLabel("Kein Ordner gewählt")
        self.folder_label.setWordWrap(True)
        folder_row.addWidget(self.folder_label)
        folder_btn = QPushButton("Ordner…")
        folder_btn.setFixedWidth(70)
        folder_btn.clicked.connect(self._select_folder)
        folder_row.addWidget(folder_btn)
        ig.addLayout(folder_row)

        self.classify_btn = QPushButton("Alle Bilder klassifizieren")
        self.classify_btn.setStyleSheet("background-color: #3498DB; color: white; font-weight: bold; padding: 6px;")
        self.classify_btn.clicked.connect(self._classify_folder)
        ig.addWidget(self.classify_btn)

        self.progress_bar = QProgressBar()
        ig.addWidget(self.progress_bar)
        v.addWidget(input_group)

        # Export
        export_group = QGroupBox("Export")
        eg = QVBoxLayout(export_group)
        export_btn = QPushButton("Ergebnisse als Excel exportieren")
        export_btn.clicked.connect(self._export_excel)
        eg.addWidget(export_btn)
        v.addWidget(export_group)

        v.addStretch()
        return box

    def _build_results_panel(self) -> QGroupBox:
        box = QGroupBox("Klassifikationsergebnisse")
        v = QVBoxLayout(box)

        self.result_count_label = QLabel("Keine Ergebnisse")
        v.addWidget(self.result_count_label)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Dateiname", "Vorhergesagtes Label", "Confidence", "Modell", "Fehler"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        v.addWidget(self.table)

        # Preview
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedHeight(180)
        self.preview_label.setStyleSheet("background: #2C3E50;")
        self.table.currentCellChanged.connect(self._on_table_select)
        v.addWidget(QLabel("Vorschau:"))
        v.addWidget(self.preview_label)

        return box

    # ------------------------------------------------------------------ slots

    def _load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Modell laden", "", "PyTorch Modell (*.pth);;Alle Dateien (*)"
        )
        if not path:
            return
        try:
            meta = self.inferencer.load_model(path)
            self.model_path_label.setText(os.path.basename(path))
            classes = meta.get("class_names", [])
            self.model_info_label.setText(
                f"Klassen: {', '.join(classes)}\n"
                f"Architektur: {meta.get('model_type', '?')}\n"
                f"Bildgröße: {meta.get('image_size', '?')}px"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Modellfehler", str(exc))

    def _classify_single(self) -> None:
        if not self.inferencer.is_ready():
            QMessageBox.warning(self, "Kein Modell", "Bitte zuerst ein Modell laden.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Bild wählen", "",
            f"Bilder ({' '.join('*' + ext for ext in IMAGE_FORMATS)});;Alle Dateien (*)"
        )
        if not path:
            return
        try:
            label, conf, probs = self.inferencer.predict_image(path)
            result = {
                "filename": os.path.basename(path),
                "path": path,
                "predicted_label": label,
                "confidence": conf,
                "all_probs": {cls: round(p, 4) for cls, p in zip(self.inferencer.class_names, probs)},
                "model_path": self.inferencer.model_path,
                "error": None,
            }
            self._results = [result]
            self._populate_table(self._results)
        except Exception as exc:
            QMessageBox.critical(self, "Inferenzfehler", str(exc))

    def _select_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Bildordner wählen")
        if folder:
            self.folder_label.setText(folder)

    def _classify_folder(self) -> None:
        if not self.inferencer.is_ready():
            QMessageBox.warning(self, "Kein Modell", "Bitte zuerst ein Modell laden.")
            return
        folder = self.folder_label.text()
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "Kein Ordner", "Bitte einen gültigen Ordner wählen.")
            return

        self.classify_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.table.setRowCount(0)

        self._thread = InferenceThread(self.inferencer, folder)
        self._thread.progress.connect(self._on_progress)
        self._thread.finished.connect(self._on_finished)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    @Slot(int, int)
    def _on_progress(self, cur: int, total: int) -> None:
        self.progress_bar.setValue(int(cur / total * 100))

    @Slot(list)
    def _on_finished(self, results: List[Dict]) -> None:
        self.classify_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self._results = results
        self._populate_table(results)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self.classify_btn.setEnabled(True)
        QMessageBox.critical(self, "Inferenzfehler", msg)

    def _populate_table(self, results: List[Dict]) -> None:
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(results))
        for row, r in enumerate(results):
            self.table.setItem(row, 0, QTableWidgetItem(r.get("filename", "")))
            label_item = QTableWidgetItem(r.get("predicted_label", ""))
            self.table.setItem(row, 1, label_item)
            conf = r.get("confidence", 0)
            conf_item = QTableWidgetItem(f"{conf*100:.1f}%")
            conf_item.setTextAlignment(Qt.AlignCenter)
            if conf >= 0.9:
                conf_item.setForeground(QColor("#2ECC71"))
            elif conf >= 0.7:
                conf_item.setForeground(QColor("#F39C12"))
            else:
                conf_item.setForeground(QColor("#E74C3C"))
            self.table.setItem(row, 2, conf_item)
            self.table.setItem(row, 3, QTableWidgetItem(os.path.basename(r.get("model_path", ""))))
            err_item = QTableWidgetItem(r.get("error") or "")
            if r.get("error"):
                err_item.setForeground(QColor("#E74C3C"))
            self.table.setItem(row, 4, err_item)

        self.table.setSortingEnabled(True)
        self.result_count_label.setText(
            f"{len(results)} Bilder klassifiziert | "
            f"Modell: {os.path.basename(self.inferencer.model_path)}"
        )

    def _on_table_select(self, row: int, col: int, *_) -> None:
        if row < 0 or row >= len(self._results):
            return
        r = self._results[row]
        path = r.get("path", "")
        if os.path.isfile(path):
            pix = QPixmap(path)
            if not pix.isNull():
                scaled = pix.scaled(
                    self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.preview_label.setPixmap(scaled)

    def _export_excel(self) -> None:
        if not self._results:
            QMessageBox.information(self, "Keine Daten", "Erst Bilder klassifizieren.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Excel speichern", "ergebnisse.xlsx", "Excel (*.xlsx)"
        )
        if not path:
            return
        try:
            from core.export import export_results_to_excel
            export_results_to_excel(
                self._results, path,
                model_name=os.path.basename(self.inferencer.model_path)
            )
            QMessageBox.information(self, "Exportiert", f"Gespeichert: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Exportfehler", str(exc))
