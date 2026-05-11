"""
Dataset management page: analysis, validation, COCO/YOLO/CSV export.
"""
import os
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QPushButton, QLabel, QTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QListWidget, QListWidgetItem, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont


class AnalysisThread(QThread):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, project):
        super().__init__()
        self.project = project

    def run(self):
        try:
            from core.dataset import analyze_dataset
            result = analyze_dataset(self.project)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class DataPage(QWidget):
    """Dataset analysis and export page."""

    images_loaded = Signal(int)   # emitted after loading; carries count of added images

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._analysis: dict = {}
        self._thread: Optional[AnalysisThread] = None
        self._build_ui()

        from gui.widgets.drop_mixin import ImageDropFilter
        self._drop_filter = ImageDropFilter(self)
        self._drop_filter.files_dropped.connect(self._on_files_dropped)

    def set_project(self, project) -> None:
        self.project = project
        self._clear()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left: controls
        ctrl = QGroupBox("Aktionen")
        cv = QVBoxLayout(ctrl)

        load_btn = QPushButton("Bilder laden…")
        load_btn.setStyleSheet("background:#2ECC71;color:white;padding:8px;font-weight:bold;")
        load_btn.clicked.connect(self._load_images)
        cv.addWidget(load_btn)

        cam_btn = QPushButton("Kamera aufnehmen…")
        cam_btn.setStyleSheet("background:#9B59B6;color:white;padding:8px;font-weight:bold;")
        cam_btn.clicked.connect(self._open_camera_dialog)
        cv.addWidget(cam_btn)

        analyze_btn = QPushButton("Dataset analysieren")
        analyze_btn.setStyleSheet("background:#3498DB;color:white;padding:8px;font-weight:bold;")
        analyze_btn.clicked.connect(self._run_analysis)
        cv.addWidget(analyze_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        cv.addWidget(self.progress)

        export_group = QGroupBox("Dataset exportieren")
        eg = QVBoxLayout(export_group)
        for label, slot in [
            ("Als COCO JSON", self._export_coco),
            ("Als YOLO TXT",  self._export_yolo),
            ("Als CSV",       self._export_csv),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            eg.addWidget(btn)
        cv.addWidget(export_group)

        valid_group = QGroupBox("Bilddateien prüfen")
        vg = QVBoxLayout(valid_group)
        check_btn = QPushButton("Fehlende Dateien prüfen")
        check_btn.clicked.connect(self._check_files)
        vg.addWidget(check_btn)
        fix_btn = QPushButton("Bildpfade korrigieren…")
        fix_btn.clicked.connect(self._fix_paths)
        vg.addWidget(fix_btn)
        cv.addWidget(valid_group)
        cv.addStretch()
        splitter.addWidget(ctrl)

        # Right: analysis results
        self._tabs = QTabWidget()

        self._summary_text = QTextEdit()
        self._summary_text.setReadOnly(True)
        self._summary_text.setFont(QFont("Courier New", 9))
        self._tabs.addTab(self._summary_text, "Zusammenfassung")

        self._missing_list = QListWidget()
        self._tabs.addTab(self._missing_list, "Fehlende Dateien")

        self._dup_list = QListWidget()
        self._tabs.addTab(self._dup_list, "Duplikate")

        self._warn_text = QTextEdit()
        self._warn_text.setReadOnly(True)
        self._tabs.addTab(self._warn_text, "Warnungen")

        splitter.addWidget(self._tabs)
        splitter.setSizes([280, 720])

    def _clear(self) -> None:
        self._summary_text.clear()
        self._missing_list.clear()
        self._dup_list.clear()
        self._warn_text.clear()

    # ------------------------------------------------------------------ image loading

    def _open_camera_dialog(self) -> None:
        if not self._check_project():
            return
        save_dir = os.path.join(
            os.path.dirname(self.project.project_path or ""),
            "camera_captures"
        ) if self.project.project_path else None

        from gui.camera_capture_dialog import CameraCaptureDialog
        dlg = CameraCaptureDialog(save_dir=save_dir, parent=self)
        if dlg.exec() and dlg.captured_paths:
            added = 0
            for path in dlg.captured_paths:
                if self.project.add_image(path):
                    added += 1
            if added:
                self.project.config.image_dir = os.path.dirname(dlg.captured_paths[0])
            QMessageBox.information(
                self, "Aufnahmen hinzugefügt",
                f"{added} Kamerabild(er) zum Projekt hinzugefügt."
                + (f"\n{len(dlg.captured_paths) - added} bereits vorhanden." if added < len(dlg.captured_paths) else "")
            )
            self.images_loaded.emit(added)

    def _on_files_dropped(self, paths: list) -> None:
        if not self._check_project():
            return
        added = 0
        for path in paths:
            if self.project.add_image(path):
                added += 1
        if added and paths:
            first_dir = os.path.dirname(paths[0])
            self.project.config.image_dir = first_dir
        QMessageBox.information(
            self, "Bilder per Drag & Drop geladen",
            f"{added} neue Bilder hinzugefügt."
            + (f"\n{len(paths) - added} bereits im Projekt." if added < len(paths) else "")
        )
        self.images_loaded.emit(added)

    def _load_images(self) -> None:
        if not self._check_project():
            return
        folder = QFileDialog.getExistingDirectory(self, "Bildordner wählen")
        if not folder:
            return
        from utils.config import IMAGE_FORMATS
        added = 0
        for fname in sorted(os.listdir(folder)):
            if os.path.splitext(fname)[1].lower() in IMAGE_FORMATS:
                path = os.path.join(folder, fname)
                if self.project.add_image(path):
                    added += 1
        if added:
            self.project.config.image_dir = folder
        QMessageBox.information(self, "Bilder geladen",
                                f"{added} neue Bilder hinzugefügt aus:\n{folder}")
        self.images_loaded.emit(added)

    # ------------------------------------------------------------------ analysis

    def _run_analysis(self) -> None:
        if not self.project:
            QMessageBox.warning(self, "Kein Projekt", "Bitte zuerst ein Projekt öffnen.")
            return
        self.progress.setVisible(True)
        self._thread = AnalysisThread(self.project)
        self._thread.finished.connect(self._on_analysis_done)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    @Slot(dict)
    def _on_analysis_done(self, result: dict) -> None:
        self.progress.setVisible(False)
        self._analysis = result

        # Summary
        from collections import Counter
        fmts = result.get("formats", Counter())
        sizes = result.get("sizes", [])
        size_stats = result.get("size_stats", {})
        lines = [
            f"Bilder gesamt:       {result['total']}",
            f"Gelabelt:            {result['labeled']}",
            f"Ungelabelt:          {result['unlabeled']}",
            f"Fehlende Dateien:    {len(result.get('missing_files', []))}",
            f"Defekte Bilder:      {len(result.get('corrupt_files', []))}",
            f"Duplikate:           {len(result.get('duplicates', []))}",
            "",
            "Dateiformate:",
        ]
        for ext, cnt in fmts.items():
            lines.append(f"  {ext}: {cnt}")
        if size_stats:
            lines += [
                "",
                "Bildgrößen:",
                f"  Breite:  {size_stats['min_w']} – {size_stats['max_w']} px",
                f"  Höhe:    {size_stats['min_h']} – {size_stats['max_h']} px",
                f"  Versch. Größen: {size_stats['unique_sizes']}",
            ]
        lines += ["", "Bilder pro Klasse:"]
        for lbl, cnt in result.get("label_counts", {}).items():
            lines.append(f"  {lbl}: {cnt}")
        self._summary_text.setPlainText("\n".join(lines))

        # Missing
        self._missing_list.clear()
        for p in result.get("missing_files", []):
            item = QListWidgetItem(p)
            item.setForeground(QColor("#E74C3C"))
            self._missing_list.addItem(item)

        # Duplicates
        self._dup_list.clear()
        for p1, p2 in result.get("duplicates", []):
            self._dup_list.addItem(f"{os.path.basename(p1)}  ↔  {os.path.basename(p2)}")

        # Warnings
        warns = result.get("warnings", [])
        self._warn_text.setPlainText("\n".join(warns) if warns else "✓ Keine Warnungen.")
        if warns:
            self._warn_text.setStyleSheet("color: #F39C12;")
        else:
            self._warn_text.setStyleSheet("color: #2ECC71;")

        self._tabs.setCurrentIndex(0)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Analysefehler", msg)

    # ------------------------------------------------------------------ export

    def _export_coco(self) -> None:
        if not self._check_project():
            return
        path, _ = QFileDialog.getSaveFileName(self, "COCO JSON", "annotations.json", "JSON (*.json)")
        if not path:
            return
        try:
            from core.dataset import export_coco
            export_coco(self.project, path)
            QMessageBox.information(self, "Exportiert", f"COCO JSON gespeichert:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Exportfehler", str(exc))

    def _export_yolo(self) -> None:
        if not self._check_project():
            return
        folder = QFileDialog.getExistingDirectory(self, "YOLO-Ausgabeordner wählen")
        if not folder:
            return
        try:
            from core.dataset import export_yolo
            export_yolo(self.project, folder)
            QMessageBox.information(self, "Exportiert", f"YOLO TXT gespeichert in:\n{folder}")
        except Exception as exc:
            QMessageBox.critical(self, "Exportfehler", str(exc))

    def _export_csv(self) -> None:
        if not self._check_project():
            return
        path, _ = QFileDialog.getSaveFileName(self, "CSV speichern", "dataset.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            from core.dataset import export_csv
            export_csv(self.project, path)
            QMessageBox.information(self, "Exportiert", f"CSV gespeichert:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Exportfehler", str(exc))

    def _check_files(self) -> None:
        if not self._check_project():
            return
        result = self.project.validate_image_files()
        missing = result.get("missing", [])
        msg = (
            f"OK: {len(result['ok'])} Dateien\n"
            f"Fehlend: {len(missing)}\n"
            f"Unlesbar: {len(result.get('unreadable', []))}"
        )
        if missing:
            msg += "\n\nFehlende Dateien:\n" + "\n".join(os.path.basename(p) for p in missing[:10])
            if len(missing) > 10:
                msg += f"\n... und {len(missing)-10} weitere"
        QMessageBox.information(self, "Datei-Prüfung", msg)

    def _fix_paths(self) -> None:
        if not self._check_project():
            return
        from PySide6.QtWidgets import QInputDialog
        old, ok1 = QInputDialog.getText(self, "Pfadkorrektur", "Altes Präfix (zu ersetzen):")
        if not ok1:
            return
        new, ok2 = QInputDialog.getText(self, "Pfadkorrektur", "Neues Präfix:")
        if not ok2:
            return
        count = self.project.relocate_images(old.strip(), new.strip())
        QMessageBox.information(self, "Erledigt", f"{count} Bildpfade aktualisiert.")

    def _check_project(self) -> bool:
        if not self.project:
            QMessageBox.warning(self, "Kein Projekt", "Bitte zuerst ein Projekt öffnen.")
            return False
        return True
