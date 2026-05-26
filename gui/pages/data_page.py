"""
Dataset management page: analysis, validation, COCO/YOLO/CSV export.
"""
import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QPushButton, QLabel, QTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QListWidget, QListWidgetItem, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont


class AnalysisThread(QThread):
    """Background thread that runs ``core.dataset.analyze_dataset`` without blocking the UI."""

    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, project):
        """
        Parameters
        ----------
        project : The ``Project`` instance to analyse.
        """
        super().__init__()
        self.project = project

    def run(self):
        """Execute the dataset analysis and emit ``finished`` or ``error``."""
        try:
            from core.dataset import analyze_dataset
            result = analyze_dataset(self.project)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class DataPage(QWidget):
    """
    Dataset management page (stack index 1).

    Provides:
    - Image loading from a folder or via drag-and-drop.
    - Camera-capture dialog and video-frame extraction.
    - Dataset analysis (class counts, missing files, MD5 duplicates, image sizes)
      run in a background ``AnalysisThread``.
    - Export to COCO JSON, YOLO TXT, and CSV formats.
    - File validation and path-relocation helpers.

    Signals
    -------
    images_loaded : Emitted with the count of newly added images after any load operation.
    """

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

    def set_project(self, project, audit=None) -> None:
        """Accept a new project and clear the previous analysis results."""
        self.project = project
        has_project = project is not None
        self._analyze_btn.setEnabled(has_project)
        for btn in self._export_btns:
            btn.setEnabled(has_project)
        self._clear()

    def _build_ui(self) -> None:
        from utils.i18n import tr
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left: controls
        ctrl = QGroupBox(tr("data.actions_group"))
        cv = QVBoxLayout(ctrl)

        load_btn = QPushButton(tr("data.load_images_btn"))
        load_btn.setStyleSheet("background:#2ECC71;color:white;padding:8px;font-weight:bold;")
        load_btn.setToolTip(
            "Ordner mit Bildern auswählen.\n"
            "Unterstützte Formate: JPG, PNG, BMP, TIFF, WebP.\n"
            "Alle Bilder im Ordner UND in Unterordnern werden geladen.\n"
            "Tipp: Bilder oder Ordner per Drag & Drop ins Fenster ziehen geht auch."
        )
        load_btn.clicked.connect(self._load_images)
        cv.addWidget(load_btn)

        cam_btn = QPushButton(tr("data.camera_btn"))
        cam_btn.setStyleSheet("background:#9B59B6;color:white;padding:8px;font-weight:bold;")
        cam_btn.setToolTip(
            "Live-Kamera öffnen und Bilder direkt ins Projekt aufnehmen.\n"
            "Unterstützt USB-Kameras (Index 0, 1, …) und IP-Kameras (RTSP/HTTP).\n"
            "Einzelbild, Burst-Modus und automatische Anomalieerkennung verfügbar."
        )
        cam_btn.clicked.connect(self._open_camera_dialog)
        cv.addWidget(cam_btn)

        video_btn = QPushButton(tr("data.video_import_btn"))
        video_btn.setStyleSheet("background:#D35400;color:white;padding:8px;font-weight:bold;")
        video_btn.setToolTip("Frames aus MP4/AVI/MOV extrahieren und zum Projekt hinzufügen")
        video_btn.clicked.connect(self._import_video)
        cv.addWidget(video_btn)

        self._analyze_btn = QPushButton(tr("data.analyze_btn"))
        self._analyze_btn.setStyleSheet("background:#3498DB;color:white;padding:8px;font-weight:bold;")
        self._analyze_btn.setToolTip(
            "Analysiert den Datensatz und zeigt:\n"
            "• Klassenverteilung und Ungleichgewicht\n"
            "• Fehlende oder unlesbare Dateien\n"
            "• MD5-Duplikate (identische Bilder)\n"
            "• Bildgrößen und Formatstatistiken\n"
            "Empfehlung: vor dem Training ausführen."
        )
        self._analyze_btn.clicked.connect(self._run_analysis)
        self._analyze_btn.setEnabled(False)
        cv.addWidget(self._analyze_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        cv.addWidget(self.progress)

        export_group = QGroupBox(tr("data.export_group"))
        eg = QVBoxLayout(export_group)
        _export_tips = {
            tr("data.export_coco_btn"): (
                "Exportiert Annotationen im COCO-Format (JSON).\n"
                "Kompatibel mit: Detectron2, MMDetection, Ultralytics YOLO v8+,\n"
                "CVAT, LabelStudio und vielen anderen Frameworks."
            ),
            tr("data.export_yolo_btn"): (
                "Exportiert Annotationen im YOLO-Format (eine .txt-Datei pro Bild).\n"
                "Kompatibel mit: Ultralytics YOLOv5/v8, Darknet."
            ),
            tr("data.export_csv_btn"): (
                "Exportiert Labels und ROIs als CSV-Tabelle.\n"
                "Gut für eigene Tools, Excel oder pandas-Analysen."
            ),
        }
        self._export_btns = []
        for label, slot in [
            (tr("data.export_coco_btn"), self._export_coco),
            (tr("data.export_yolo_btn"), self._export_yolo),
            (tr("data.export_csv_btn"),  self._export_csv),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            if label in _export_tips:
                btn.setToolTip(_export_tips[label])
            btn.setEnabled(False)
            eg.addWidget(btn)
            self._export_btns.append(btn)
        cv.addWidget(export_group)

        valid_group = QGroupBox(tr("data.validation_group"))
        vg = QVBoxLayout(valid_group)
        check_btn = QPushButton(tr("data.check_files_btn"))
        check_btn.setToolTip(
            "Prüft ob alle Bilddateien noch an ihrem gespeicherten Pfad vorhanden sind.\n"
            "Fehlende Dateien werden rot markiert."
        )
        check_btn.clicked.connect(self._check_files)
        vg.addWidget(check_btn)
        fix_btn = QPushButton(tr("data.fix_paths_btn"))
        fix_btn.setToolTip(
            "Bilder die verschoben oder umbenannt wurden neu verknüpfen.\n"
            "Nützlich wenn das Projekt auf einen anderen Rechner kopiert wurde."
        )
        fix_btn.clicked.connect(self._fix_paths)
        vg.addWidget(fix_btn)

        remove_missing_btn = QPushButton(tr("data.remove_missing_btn"))
        remove_missing_btn.setToolTip(tr("data.remove_missing_tip"))
        remove_missing_btn.clicked.connect(self._remove_missing_files)
        vg.addWidget(remove_missing_btn)
        cv.addWidget(valid_group)
        cv.addStretch()
        splitter.addWidget(ctrl)

        # Right: analysis results
        self._tabs = QTabWidget()

        self._summary_text = QTextEdit()
        self._summary_text.setReadOnly(True)
        self._summary_text.setFont(QFont("Courier New", 9))
        self._summary_text.setPlaceholderText(tr("data.no_project_hint"))
        self._tabs.addTab(self._summary_text, tr("data.tab.summary"))

        self._missing_list = QListWidget()
        self._tabs.addTab(self._missing_list, tr("data.tab.missing"))

        self._dup_list = QListWidget()
        self._tabs.addTab(self._dup_list, tr("data.tab.duplicates"))

        self._warn_text = QTextEdit()
        self._warn_text.setReadOnly(True)
        self._tabs.addTab(self._warn_text, tr("data.tab.warnings"))

        splitter.addWidget(self._tabs)
        splitter.setSizes([280, 720])

    def _clear(self) -> None:
        """Reset all analysis result tabs to an empty state."""
        self._summary_text.clear()
        self._missing_list.clear()
        self._dup_list.clear()
        self._warn_text.clear()

    # ------------------------------------------------------------------ image loading

    def _import_video(self) -> None:
        from utils.i18n import tr
        if not self._check_project():
            return
        out_dir = os.path.join(
            os.path.dirname(self.project.project_path or ""),
            "video_frames"
        ) if self.project.project_path else os.path.expanduser("~")

        from gui.video_import_dialog import VideoImportDialog
        dlg = VideoImportDialog(default_out_dir=out_dir, parent=self)
        if dlg.exec() and dlg.extracted_paths:
            added = 0
            for path in dlg.extracted_paths:
                if self.project.add_image(path):
                    added += 1
            if added and dlg.extracted_paths:
                self.project.config.image_dir = os.path.dirname(dlg.extracted_paths[0])
            QMessageBox.information(
                self, tr("data.video_imported_title"),
                tr("data.msg.video_imported", added=added)
                + (f"\n{len(dlg.extracted_paths) - added} bereits vorhanden."
                   if added < len(dlg.extracted_paths) else "")
            )
            self.images_loaded.emit(added)

    def _open_camera_dialog(self) -> None:
        from utils.i18n import tr
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
                self, tr("data.camera_captured_title"),
                tr("data.msg.camera_captured", added=added)
                + (f"\n{len(dlg.captured_paths) - added} bereits vorhanden." if added < len(dlg.captured_paths) else "")
            )
            self.images_loaded.emit(added)

    def _on_files_dropped(self, paths: list) -> None:
        """Handle files/folders dropped onto the widget; scan subfolders recursively."""
        if not self._check_project():
            return
        from utils.config import IMAGE_FORMATS
        added = 0
        total = 0
        first_dir = None
        for path in paths:
            if os.path.isdir(path):
                # Dropped a folder → scan recursively
                if first_dir is None:
                    first_dir = path
                for root, _dirs, files in os.walk(path):
                    for fname in sorted(files):
                        if os.path.splitext(fname)[1].lower() in IMAGE_FORMATS:
                            total += 1
                            if self.project.add_image(os.path.join(root, fname)):
                                added += 1
            else:
                if os.path.splitext(path)[1].lower() in IMAGE_FORMATS:
                    total += 1
                    if first_dir is None:
                        first_dir = os.path.dirname(path)
                    if self.project.add_image(path):
                        added += 1
        if added and first_dir:
            self.project.config.image_dir = first_dir
        already = total - added
        from utils.i18n import tr
        msg = tr("data.msg.images_loaded", added=added, folder=first_dir or "")
        if already:
            msg += f"\n{already} bereits im Projekt."
        QMessageBox.information(self, tr("data.images_loaded_title"), msg)
        self.images_loaded.emit(added)

    def _load_images(self) -> None:
        """Open a folder chooser and add all supported image files recursively."""
        from utils.i18n import tr
        if not self._check_project():
            return
        folder = QFileDialog.getExistingDirectory(self, tr("data.dlg.folder_select"))
        if not folder:
            return
        from utils.config import IMAGE_FORMATS
        added = 0
        total = 0
        for root, _dirs, files in os.walk(folder):
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() in IMAGE_FORMATS:
                    total += 1
                    path = os.path.join(root, fname)
                    if self.project.add_image(path):
                        added += 1
        if added:
            self.project.config.image_dir = folder
        already = total - added
        msg = tr("data.msg.images_loaded", added=added, folder=folder)
        if already:
            msg += f"\n{already} bereits im Projekt."
        QMessageBox.information(self, tr("data.images_loaded_title"), msg)
        self.images_loaded.emit(added)

    # ------------------------------------------------------------------ analysis

    def _run_analysis(self) -> None:
        """Start the background ``AnalysisThread`` and show the progress bar."""
        from utils.i18n import tr
        if not self.project:
            QMessageBox.warning(self, tr("common.no_project"), tr("common.no_project_msg"))
            return
        self.progress.setVisible(True)
        self._summary_text.setPlainText("Analysiere Datensatz …")
        self._thread = AnalysisThread(self.project)
        self._thread.finished.connect(self._on_analysis_done)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    @Slot(dict)
    def _on_analysis_done(self, result: dict) -> None:
        """Populate all result tabs with the finished analysis data."""
        from utils.i18n import tr
        self._thread = None
        self.progress.setVisible(False)
        self._analysis = result

        # Summary
        from collections import Counter
        fmts = result.get("formats", Counter())
        sizes = result.get("sizes", [])
        size_stats = result.get("size_stats", {})
        lines = [
            f"{tr('data.stats.total'):<20} {result['total']}",
            f"{tr('data.stats.labeled'):<20} {result['labeled']}",
            f"{tr('data.stats.unlabeled'):<20} {result['unlabeled']}",
            f"{tr('data.stats.missing'):<20} {len(result.get('missing_files', []))}",
            f"{tr('data.stats.corrupt'):<20} {len(result.get('corrupt_files', []))}",
            f"{tr('data.stats.duplicates'):<20} {len(result.get('duplicates', []))}",
            "",
            tr("data.stats.formats"),
        ]
        for ext, cnt in fmts.items():
            lines.append(f"  {ext}: {cnt}")
        if size_stats:
            lines += [
                "",
                tr("data.stats.sizes"),
                f"  Breite:  {size_stats['min_w']} – {size_stats['max_w']} px",
                f"  Höhe:    {size_stats['min_h']} – {size_stats['max_h']} px",
                f"  Versch. Größen: {size_stats['unique_sizes']}",
            ]
        lines += ["", tr("data.stats.per_class")]
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
        """Show a critical dialog when the analysis thread reports an error."""
        from utils.i18n import tr
        self._thread = None
        self.progress.setVisible(False)
        QMessageBox.critical(self, tr("common.error"), msg)

    # ------------------------------------------------------------------ export

    def _export_coco(self) -> None:
        from utils.i18n import tr
        if not self._check_project():
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("data.dlg.coco_title"), "annotations.json", "JSON (*.json)")
        if not path:
            return
        try:
            from core.dataset import export_coco
            export_coco(self.project, path)
            QMessageBox.information(self, tr("common.saved"), tr("data.msg.coco_saved", path=path))
        except Exception as exc:
            QMessageBox.critical(self, tr("common.error"), str(exc))

    def _export_yolo(self) -> None:
        from utils.i18n import tr
        if not self._check_project():
            return
        folder = QFileDialog.getExistingDirectory(self, tr("data.dlg.yolo_title"))
        if not folder:
            return
        try:
            from core.dataset import export_yolo
            export_yolo(self.project, folder)
            QMessageBox.information(self, tr("common.saved"), tr("data.msg.yolo_saved", folder=folder))
        except Exception as exc:
            QMessageBox.critical(self, tr("common.error"), str(exc))

    def _export_csv(self) -> None:
        from utils.i18n import tr
        if not self._check_project():
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("data.dlg.csv_title"), "dataset.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            from core.dataset import export_csv
            export_csv(self.project, path)
            QMessageBox.information(self, tr("common.saved"), tr("data.msg.csv_saved", path=path))
        except Exception as exc:
            QMessageBox.critical(self, tr("common.error"), str(exc))

    def _check_files(self) -> None:
        """Validate all image file paths and display a summary message box."""
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

    def _remove_missing_files(self) -> None:
        """Remove all image paths that no longer exist on disk after user confirmation."""
        from utils.i18n import tr
        if not self._check_project():
            return
        missing = [p for p in self.project.images if not os.path.isfile(p)]
        if not missing:
            QMessageBox.information(self, tr("common.info"), "Keine fehlenden Dateien.")
            return
        reply = QMessageBox.question(
            self, tr("data.confirm_remove_title"),
            tr("data.confirm_remove_msg", n=len(missing)),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        for p in missing:
            if p in self.project.images:
                self.project.images.remove(p)
        self.project.save()
        QMessageBox.information(
            self, tr("data.confirm_remove_title"),
            f"{len(missing)} Einträge entfernt.",
        )

    def _fix_paths(self) -> None:
        """Prompt for an old/new path prefix pair and relocate all matching image paths."""
        from utils.i18n import tr
        if not self._check_project():
            return
        from PySide6.QtWidgets import QInputDialog
        old, ok1 = QInputDialog.getText(self, tr("data.paths_updated_title"), tr("data.dlg.relocation_old"))
        if not ok1:
            return
        new, ok2 = QInputDialog.getText(self, tr("data.paths_updated_title"), tr("data.dlg.relocation_new"))
        if not ok2:
            return
        count = self.project.relocate_images(old.strip(), new.strip())
        QMessageBox.information(self, tr("data.paths_updated_title"), tr("data.msg.paths_updated", count=count))

    def _check_project(self) -> bool:
        """Return True if a project is loaded; otherwise show a warning and return False."""
        from utils.i18n import tr
        if not self.project:
            QMessageBox.warning(self, tr("common.no_project"), tr("common.no_project_msg"))
            return False
        return True
