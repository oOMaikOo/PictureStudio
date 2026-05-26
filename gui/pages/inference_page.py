"""
Inference page: batch classification, top-3, confidence filter, low-confidence marking.
"""
import logging
import os
from typing import List, Dict, Optional

log = logging.getLogger(__name__)

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QMessageBox, QAbstractItemView,
    QDoubleSpinBox, QComboBox, QCheckBox, QSpinBox, QTabWidget,
    QTextEdit, QScrollArea, QFrame,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QObject
from PySide6.QtGui import QColor, QFont, QKeySequence, QPixmap, QShortcut

from utils.i18n import tr


class InferenceThread(QThread):
    """Background thread that calls ``Inferencer.predict_folder`` without blocking the UI."""

    progress = Signal(int, int)
    finished = Signal(list)
    error    = Signal(str)

    def __init__(self, inferencer, folder: str, top_k: int, roi_templates: list,
                 tta_passes: int = 1, recursive: bool = False):
        """
        Parameters
        ----------
        inferencer    : Ready ``Inferencer`` instance with a model already loaded.
        folder        : Directory of images to classify.
        top_k         : Number of top predictions to include per image.
        roi_templates : Optional ROI crop templates applied to every image.
        tta_passes    : Test-time augmentation passes (1 = disabled).
        recursive     : When True, scan all subfolders recursively.
        """
        super().__init__()
        self.inferencer = inferencer
        self.folder = folder
        self.top_k = top_k
        self.roi_templates = roi_templates
        self.tta_passes = tta_passes
        self.recursive = recursive

    def run(self) -> None:
        """Run folder inference and emit ``finished`` with the result list, or ``error``."""
        try:
            results = self.inferencer.predict_folder(
                self.folder,
                roi_templates=self.roi_templates,
                top_k=self.top_k,
                progress_callback=lambda c, t: self.progress.emit(c, t),
                tta_passes=self.tta_passes,
                recursive=self.recursive,
            )
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


class InferencePage(QWidget):
    """
    Single-image and batch classification page (stack index 5).

    Features:
    - Load a primary model (``Inferencer``) and optional ensemble models.
    - Classify a single image or an entire folder via ``InferenceThread``.
    - Top-K predictions, confidence colour coding, and a low-confidence tab.
    - Filter results by label and minimum confidence threshold.
    - Active-Learning integration: push low-confidence images to the labeling queue.
    - Semi-automatic labeling: apply high-confidence predictions as project labels.
    - Grad-CAM visualisation for any selected result row.
    - Export results to Excel or rename images with their predicted label appended.

    Signals
    -------
    al_queue_updated : Emitted after adding images to the AL queue.
    labels_applied   : Emitted with the count of labels written to the project.
    """

    al_queue_updated = Signal()      # emitted when images are added to AL queue
    labels_applied   = Signal(int)   # emitted after semi-auto labeling; carries count

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._audit = None
        self._thread: Optional[InferenceThread] = None
        self._all_results: List[Dict] = []
        self._filtered: List[Dict] = []

        from core.inference import Inferencer
        self.inferencer = Inferencer()
        self._ensemble_inferencers: List = []   # additional Inferencer instances
        self._build_ui()
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self._classify_folder)

    def set_project(self, project, audit=None) -> None:
        """Accept the active project and optional audit trail."""
        self.project = project
        self._audit = audit

    def load_model_path(self, path: str) -> None:
        """Load a model checkpoint from *path* and update the model-info label."""
        try:
            meta = self.inferencer.load_model(path)
            self.model_path_label.setText(os.path.basename(path))
            self.model_info_label.setText(
                f"Klassen: {', '.join(meta.get('class_names', []))}\n"
                f"Architektur: {meta.get('model_type', '?')}\n"
                f"Bildgröße: {meta.get('image_size', '?')}px"
            )
        except Exception as exc:
            QMessageBox.critical(self, tr("common.error"), str(exc))

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Wrap control panel in a scroll area so it never clips on small windows
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self._build_control_panel())
        splitter.addWidget(scroll)

        splitter.addWidget(self._build_results_panel())
        splitter.setSizes([320, 680])

    def _build_control_panel(self) -> QGroupBox:
        box = QGroupBox("Inferenz-Steuerung")
        v = QVBoxLayout(box)

        # Model
        mg = QGroupBox(tr("inference.model_group"))
        mv = QVBoxLayout(mg)
        self.model_path_label = QLabel(tr("inference.no_model"))
        self.model_path_label.setWordWrap(True)
        mv.addWidget(self.model_path_label)
        load_model_btn = QPushButton(tr("inference.load_primary_btn"))
        load_model_btn.clicked.connect(self._load_model)
        mv.addWidget(load_model_btn)
        self.model_info_label = QLabel("")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setStyleSheet("color:#27AE60;font-size:10px;")
        mv.addWidget(self.model_info_label)

        ens_info = QLabel("Ensemble: weitere Modelle hinzufügen (average softmax):")
        ens_info.setStyleSheet("color:#aaa;font-size:10px;")
        ens_info.setWordWrap(True)
        mv.addWidget(ens_info)
        self._ens_list = QLabel("(kein Ensemble)")
        self._ens_list.setStyleSheet("color:#888;font-size:10px;")
        self._ens_list.setWordWrap(True)
        mv.addWidget(self._ens_list)
        ens_btn_row = QHBoxLayout()
        add_ens_btn = QPushButton(tr("inference.add_ensemble_btn"))
        add_ens_btn.setStyleSheet("font-size:10px;padding:3px;")
        add_ens_btn.clicked.connect(self._add_ensemble_model)
        ens_btn_row.addWidget(add_ens_btn)
        clr_ens_btn = QPushButton(tr("inference.clear_ensemble_btn"))
        clr_ens_btn.setStyleSheet("font-size:10px;padding:3px;")
        clr_ens_btn.clicked.connect(self._clear_ensemble)
        ens_btn_row.addWidget(clr_ens_btn)
        mv.addLayout(ens_btn_row)
        v.addWidget(mg)

        # Input
        ig = QGroupBox(tr("inference.input_group"))
        iv = QVBoxLayout(ig)
        single_btn = QPushButton(tr("inference.single_btn"))
        single_btn.clicked.connect(self._classify_single)
        iv.addWidget(single_btn)
        folder_row = QHBoxLayout()
        self.folder_label = QLabel(tr("inference.no_folder"))
        self.folder_label.setWordWrap(True)
        folder_row.addWidget(self.folder_label)
        fb = QPushButton(tr("inference.folder_btn"))
        fb.setFixedWidth(70)
        fb.clicked.connect(self._select_folder)
        folder_row.addWidget(fb)
        iv.addLayout(folder_row)
        self._recursive_cb = QCheckBox(tr("inference.recursive_cb"))
        self._recursive_cb.setToolTip(
            "Scannt den gewählten Ordner und alle Unterordner rekursiv.\n"
            "Der Dateiname in der Tabelle zeigt 'Unterordner/Dateiname'."
        )
        iv.addWidget(self._recursive_cb)
        self.classify_btn = QPushButton(tr("inference.classify_folder_btn"))
        self.classify_btn.setStyleSheet("background:#3498DB;color:white;font-weight:bold;padding:6px;")
        self.classify_btn.clicked.connect(self._classify_folder)
        iv.addWidget(self.classify_btn)
        self.progress_bar = QProgressBar()
        iv.addWidget(self.progress_bar)
        v.addWidget(ig)

        # Options
        og = QGroupBox(tr("inference.options_group"))
        ov = QVBoxLayout(og)
        topk_row = QHBoxLayout()
        topk_row.addWidget(QLabel(tr("inference.topk_label")))
        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 5)
        self.topk_spin.setValue(3)
        topk_row.addWidget(self.topk_spin)
        ov.addLayout(topk_row)
        tta_row = QHBoxLayout()
        tta_row.addWidget(QLabel(tr("inference.tta_label")))
        self.tta_spin = QSpinBox()
        self.tta_spin.setRange(1, 20)
        self.tta_spin.setValue(1)
        self.tta_spin.setToolTip(
            "Test-Time Augmentation: 1 = deaktiviert.\n"
            "Höher = robustere Vorhersagen, aber langsamer.\n"
            "Empfehlung: 5–10 für unsichere Bilder."
        )
        tta_row.addWidget(self.tta_spin)
        ov.addLayout(tta_row)
        self.use_roi_template_cb = QCheckBox(tr("inference.use_roi_cb"))
        ov.addWidget(self.use_roi_template_cb)
        v.addWidget(og)

        # Filter
        fg = QGroupBox(tr("inference.filter_group"))
        fv = QVBoxLayout(fg)
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel(tr("inference.min_conf_label")))
        self.min_conf_spin = QDoubleSpinBox()
        self.min_conf_spin.setRange(0.0, 1.0)
        self.min_conf_spin.setValue(0.0)
        self.min_conf_spin.setSingleStep(0.05)
        conf_row.addWidget(self.min_conf_spin)
        fv.addLayout(conf_row)
        label_row = QHBoxLayout()
        label_row.addWidget(QLabel(tr("inference.label_filter_label")))
        self.label_filter_combo = QComboBox()
        self.label_filter_combo.addItem("Alle")
        label_row.addWidget(self.label_filter_combo)
        fv.addLayout(label_row)
        self.low_conf_only_cb = QCheckBox(tr("inference.low_conf_only_cb"))
        fv.addWidget(self.low_conf_only_cb)
        apply_filter_btn = QPushButton(tr("inference.apply_filter_btn"))
        apply_filter_btn.clicked.connect(self._apply_filter)
        fv.addWidget(apply_filter_btn)
        v.addWidget(fg)

        # Active Learning
        al_box = QGroupBox(tr("inference.al_group"))
        av = QVBoxLayout(al_box)
        al_info = QLabel(
            "Unsichere Vorhersagen (Confidence < 70 %) zur\n"
            "Labeling-Queue hinzufügen und dann neu trainieren."
        )
        al_info.setWordWrap(True)
        al_info.setStyleSheet("color:#aaa;font-size:10px;")
        av.addWidget(al_info)
        self._al_btn = QPushButton(tr("inference.al_btn"))
        self._al_btn.setStyleSheet(
            "background:#E67E22;color:white;font-weight:bold;padding:5px;"
        )
        self._al_btn.setToolTip(
            "Fügt alle Bilder mit Confidence < 70 % der AL-Queue im Labeling-Reiter hinzu."
        )
        self._al_btn.clicked.connect(self._add_to_al_queue)
        av.addWidget(self._al_btn)
        self._al_status = QLabel("")
        self._al_status.setStyleSheet("color:#E67E22;font-size:10px;")
        self._al_status.setWordWrap(True)
        av.addWidget(self._al_status)
        v.addWidget(al_box)

        # Semi-automatic labeling
        sl_box = QGroupBox(tr("inference.semi_auto_group"))
        sv = QVBoxLayout(sl_box)
        sl_info = QLabel(
            "Überträgt Vorhersagen mit hoher Konfidenz als Labels\n"
            "direkt ins Projekt (nur noch nicht gelabelte Bilder)."
        )
        sl_info.setWordWrap(True)
        sl_info.setStyleSheet("color:#aaa;font-size:10px;")
        sv.addWidget(sl_info)
        sl_conf_row = QHBoxLayout()
        sl_conf_row.addWidget(QLabel(tr("inference.semi_auto_conf_label")))
        self._sl_conf_spin = QDoubleSpinBox()
        self._sl_conf_spin.setRange(0.5, 1.0)
        self._sl_conf_spin.setValue(0.90)
        self._sl_conf_spin.setSingleStep(0.05)
        self._sl_conf_spin.setDecimals(2)
        sl_conf_row.addWidget(self._sl_conf_spin)
        sv.addLayout(sl_conf_row)
        self._sl_overwrite_cb = QCheckBox(tr("inference.semi_auto_overwrite_cb"))
        sv.addWidget(self._sl_overwrite_cb)
        self._sl_btn = QPushButton(tr("inference.semi_auto_apply_btn"))
        self._sl_btn.setStyleSheet(
            "background:#1565C0;color:white;font-weight:bold;padding:5px;"
        )
        self._sl_btn.clicked.connect(self._apply_label_suggestions)
        sv.addWidget(self._sl_btn)
        self._sl_status = QLabel("")
        self._sl_status.setStyleSheet("color:#5DADE2;font-size:10px;")
        self._sl_status.setWordWrap(True)
        sv.addWidget(self._sl_status)
        v.addWidget(sl_box)

        # Export
        eg = QGroupBox(tr("inference.export_group"))
        ev = QVBoxLayout(eg)
        exp_btn = QPushButton(tr("inference.export_excel_btn"))
        exp_btn.clicked.connect(self._export_excel)
        ev.addWidget(exp_btn)
        rename_btn = QPushButton(tr("inference.rename_btn"))
        rename_btn.setToolTip(
            "Hängt das vorhergesagte Label an den Dateinamen:\n"
            "foto.jpg  →  foto_gut.jpg"
        )
        rename_btn.clicked.connect(self._rename_images)
        ev.addWidget(rename_btn)
        v.addWidget(eg)

        v.addStretch()
        return box

    def _build_results_panel(self) -> QGroupBox:
        box = QGroupBox(tr("inference.results_group"))
        v = QVBoxLayout(box)
        self.result_count_label = QLabel(tr("inference.no_results"))
        v.addWidget(self.result_count_label)

        self.tabs = QTabWidget()

        # Main results table
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "Dateiname", "Vorhergesagtes Label", "Confidence", "Top-2", "Top-3", "⚠"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.itemSelectionChanged.connect(
            lambda: self._on_table_select(self.table.currentRow())
        )
        self.tabs.addTab(self.table, tr("inference.tab.results"))

        # Low-confidence list
        self.low_conf_text = QTextEdit()
        self.low_conf_text.setReadOnly(True)
        self.tabs.addTab(self.low_conf_text, tr("inference.tab.low_conf"))
        v.addWidget(self.tabs)

        preview_row = QHBoxLayout()
        preview_row.addWidget(QLabel(tr("inference.preview_label")))
        self._gradcam_btn = QPushButton(tr("inference.gradcam_btn"))
        self._gradcam_btn.setEnabled(False)
        self._gradcam_btn.setToolTip(
            "Zeigt eine Aktivierungskarte (Grad-CAM), die erklärt,\n"
            "welche Bildregionen die Vorhersage beeinflusst haben."
        )
        self._gradcam_btn.setStyleSheet(
            "background:#8E44AD;color:white;font-weight:bold;padding:4px 10px;"
        )
        self._gradcam_btn.clicked.connect(self._show_gradcam)
        preview_row.addStretch()
        preview_row.addWidget(self._gradcam_btn)
        v.addLayout(preview_row)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedHeight(160)
        self.preview_label.setStyleSheet("background:#2C3E50;")
        v.addWidget(self.preview_label)
        return box

    # ------------------------------------------------------------------ actions

    def _load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Modell laden", "", "PyTorch (*.pth);;Alle (*)")
        if path:
            self.load_model_path(path)

    def _add_ensemble_model(self) -> None:
        """Load an additional model and append it to the ensemble list."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Ensemble-Modell hinzufügen", "", "PyTorch (*.pth)"
        )
        if not path:
            return
        from core.inference import Inferencer
        inf = Inferencer()
        try:
            inf.load_model(path)
            self._ensemble_inferencers.append(inf)
            names = [os.path.basename(i.model_path) for i in self._ensemble_inferencers]
            self._ens_list.setText("Ensemble:\n" + "\n".join(f"  • {n}" for n in names))
        except Exception as exc:
            QMessageBox.critical(self, tr("common.error"), str(exc))

    def _clear_ensemble(self) -> None:
        """Remove all ensemble models, reverting to single-model inference."""
        self._ensemble_inferencers.clear()
        self._ens_list.setText("(kein Ensemble)")

    def _ensemble_predict(self, img_path: str, top_k: int, tta_passes: int) -> Dict:
        """Average softmax over primary + ensemble models."""
        import torch
        import torch.nn.functional as F

        all_inferencers = [self.inferencer] + self._ensemble_inferencers
        avg_probs = None
        for inf in all_inferencers:
            pred = inf.predict_image(img_path, top_k=top_k, tta_passes=tta_passes)
            probs = list(pred["all_probs"].values())
            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs = [a + b for a, b in zip(avg_probs, probs)]

        n = len(all_inferencers)
        avg_probs = [p / n for p in avg_probs]
        indexed = sorted(enumerate(avg_probs), key=lambda x: x[1], reverse=True)
        class_names = self.inferencer.class_names
        top = [{"label": class_names[i], "prob": round(p, 4)} for i, p in indexed[:top_k]]
        return {
            "predicted_label": top[0]["label"],
            "confidence": round(top[0]["prob"], 4),
            "top_k": top,
            "all_probs": {cls: round(p, 4) for cls, p in zip(class_names, avg_probs)},
            "ensemble_size": n,
        }

    def _classify_single(self) -> None:
        """Open a file chooser, classify the selected image, and display the result."""
        if not self.inferencer.is_ready():
            QMessageBox.warning(self, tr("common.no_model"), "Bitte erst ein Modell laden.")
            return
        from utils.config import IMAGE_FORMATS
        path, _ = QFileDialog.getOpenFileName(
            self, "Bild wählen", "",
            f"Bilder ({' '.join('*' + e for e in IMAGE_FORMATS)});;Alle (*)"
        )
        if not path:
            return
        try:
            top_k = self.topk_spin.value()
            tta = self.tta_spin.value()
            # If the image is a project image with labeled ROIs, classify the ROI crop
            # (the model was trained on crops, not full frames)
            roi = None
            if self.project and path in self.project.images:
                rois = self.project.get_rois(path)
                if rois:
                    roi = next((r for r in rois if r.get("label")), rois[0])
            if self._ensemble_inferencers:
                pred = self._ensemble_predict(path, top_k, tta)
            else:
                pred = self.inferencer.predict_image(path, top_k=top_k, tta_passes=tta, roi=roi)
            result = {
                "filename": os.path.basename(path), "path": path,
                "predicted_label": pred["predicted_label"],
                "confidence": pred["confidence"],
                "top_k": pred["top_k"], "all_probs": pred["all_probs"],
                "model_path": self.inferencer.model_path,
                "model_type": self.inferencer.model_type,
                "low_confidence": pred["confidence"] < 0.70,
                "error": None,
            }
            self._all_results = [result]
            self._populate_table(self._all_results)
        except Exception as exc:
            QMessageBox.critical(self, tr("common.error"), str(exc))

    def _select_folder(self) -> None:
        """Open a folder chooser and update the folder label."""
        folder = QFileDialog.getExistingDirectory(self, "Bildordner wählen")
        if folder:
            self.folder_label.setText(folder)

    def _classify_folder(self) -> None:
        """Start ``InferenceThread`` to classify all images in the selected folder."""
        if not self.inferencer.is_ready():
            QMessageBox.warning(self, tr("common.no_model"), "Bitte erst ein Modell laden.")
            return
        folder = self.folder_label.text()
        if not os.path.isdir(folder):
            QMessageBox.warning(self, tr("common.warning"), tr("inference.no_folder_msg"))
            return

        roi_templates = []
        if self.use_roi_template_cb.isChecked() and self.project:
            roi_templates = self.project.get_roi_templates()

        # Fallback: if no template is active but the project has per-image ROIs,
        # use the first project ROI so new images are cropped the same way as
        # the training data was.
        if not roi_templates and self.project:
            fallback_roi = None
            for img_path in self.project.images:
                rois = self.project.get_rois(img_path)
                if rois:
                    fallback_roi = next((r for r in rois if r.get("label")), rois[0])
                    break
            if fallback_roi:
                roi_templates = [{"roi": fallback_roi}]
                w = int(fallback_roi.get("w", 0))
                h = int(fallback_roi.get("h", 0))
                x = int(fallback_roi.get("x", 0))
                y = int(fallback_roi.get("y", 0))
                QMessageBox.information(
                    self, "ROI-Fallback aktiv",
                    f"Das Projekt enthält ROIs. Das Training wurde wahrscheinlich auf "
                    f"ROI-Ausschnitten durchgeführt.\n\n"
                    f"Der erste Projekt-ROI wird automatisch auf alle Bilder angewendet:\n"
                    f"  Position: x={x}, y={y}  |  Größe: {w} × {h} px\n\n"
                    f"Um einen anderen ROI zu verwenden, aktiviere 'ROI-Vorlagen anwenden'\n"
                    f"und konfiguriere ein Template in den Projekteinstellungen."
                )

        self.classify_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.table.setRowCount(0)

        self._thread = InferenceThread(
            self.inferencer, folder, self.topk_spin.value(), roi_templates,
            tta_passes=self.tta_spin.value(),
            recursive=self._recursive_cb.isChecked(),
        )
        self._thread.progress.connect(lambda c, t: self.progress_bar.setValue(int(c / t * 100)))
        self._thread.finished.connect(self._on_finished)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def hideEvent(self, event) -> None:
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)
        super().hideEvent(event)

    @Slot(list)
    def _on_finished(self, results: List[Dict]) -> None:
        """Store results, update label-filter combo, and apply the current filter."""
        self._thread = None
        self.classify_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self._all_results = results
        if self._audit and self.project:
            self._audit.log_inference(
                self.folder_label.text(),
                os.path.basename(self.inferencer.model_path),
                len(results)
            )
        if self.project:
            self.project.inference_results = results
        # Update label filter
        labels = set(r.get("predicted_label", "") for r in results if not r.get("error"))
        self.label_filter_combo.blockSignals(True)
        self.label_filter_combo.clear()
        self.label_filter_combo.addItem("Alle")
        self.label_filter_combo.addItems(sorted(labels))
        self.label_filter_combo.blockSignals(False)
        self._apply_filter()

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        """Re-enable the classify button and show the error in a critical dialog."""
        self._thread = None
        self.classify_btn.setEnabled(True)
        QMessageBox.critical(self, tr("common.error"), msg)

    def _apply_filter(self) -> None:
        """Filter ``_all_results`` with the current UI filter settings and refresh the table."""
        lbl = self.label_filter_combo.currentText()
        lbl = "" if lbl == "Alle" else lbl
        self._filtered = self.inferencer.filter_results(
            self._all_results,
            min_confidence=self.min_conf_spin.value(),
            label_filter=lbl,
            only_low_confidence=self.low_conf_only_cb.isChecked(),
        )
        self._populate_table(self._filtered)

    def _populate_table(self, results: List[Dict]) -> None:
        """Fill the results table from *results* and update the low-confidence tab."""
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(results))
        low_conf_lines = []

        for row, r in enumerate(results):
            self.table.setItem(row, 0, QTableWidgetItem(r.get("filename", "")))

            lbl_item = QTableWidgetItem(r.get("predicted_label", ""))
            if r.get("error"):
                lbl_item.setForeground(QColor("#E74C3C"))
            self.table.setItem(row, 1, lbl_item)

            conf = r.get("confidence", 0)
            conf_item = QTableWidgetItem(f"{conf*100:.1f}%")
            conf_item.setTextAlignment(Qt.AlignCenter)
            conf_item.setForeground(
                QColor("#2ECC71") if conf >= 0.9 else
                (QColor("#F39C12") if conf >= 0.7 else QColor("#E74C3C"))
            )
            self.table.setItem(row, 2, conf_item)

            top_k = r.get("top_k", [])
            for col_offset, k_idx in enumerate([1, 2], start=3):
                if k_idx < len(top_k):
                    t = top_k[k_idx]
                    self.table.setItem(row, col_offset, QTableWidgetItem(
                        f"{t['label']} ({t['prob']*100:.0f}%)"
                    ))

            warn_item = QTableWidgetItem("⚠" if r.get("low_confidence") else "")
            if r.get("low_confidence"):
                warn_item.setForeground(QColor("#F39C12"))
            self.table.setItem(row, 5, warn_item)

            if r.get("low_confidence") and not r.get("error"):
                low_conf_lines.append(
                    f"{r['filename']}  →  {r['predicted_label']} ({conf*100:.1f}%)"
                )

        self.table.setSortingEnabled(True)
        n_low = sum(1 for r in results if r.get("low_confidence"))
        self.result_count_label.setText(
            f"{len(results)} Bilder  |  {n_low} unsichere Vorhersagen  |  "
            f"Modell: {os.path.basename(self.inferencer.model_path)}"
        )
        self.low_conf_text.setPlainText(
            "\n".join(low_conf_lines) if low_conf_lines else "Keine unsicheren Vorhersagen."
        )

    def _on_table_select(self, row: int) -> None:
        """Update the image preview and Grad-CAM button when the selected row changes."""
        if row < 0 or row >= len(self._filtered):
            self._gradcam_btn.setEnabled(False)
            return
        r = self._filtered[row]
        path = r.get("path", "")
        model_ready = self.inferencer.is_ready() and self.inferencer.model is not None
        self._gradcam_btn.setEnabled(bool(path and os.path.isfile(path) and model_ready))
        if os.path.isfile(path):
            pix = QPixmap(path)
            if not pix.isNull():
                self.preview_label.setPixmap(
                    pix.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )

    def _add_to_al_queue(self) -> None:
        """Add all low-confidence results to the project's Active Learning queue."""
        if not self.project:
            QMessageBox.warning(self, tr("common.no_project"), tr("inference.no_project_msg"))
            return
        candidates = [
            r for r in self._all_results
            if r.get("low_confidence") and not r.get("error") and os.path.isfile(r.get("path", ""))
        ]
        if not candidates:
            self._al_status.setText("Keine unsicheren Vorhersagen vorhanden.")
            return
        added = 0
        skipped = 0
        for r in candidates:
            ok = self.project.add_to_al_queue(
                r["path"], r.get("predicted_label", ""), r.get("confidence", 0.0)
            )
            if ok:
                added += 1
            else:
                skipped += 1
        parts = [f"{added} Bilder hinzugefügt"]
        if skipped:
            parts.append(f"{skipped} bereits in Queue")
        self._al_status.setText("  ".join(parts))
        if added > 0:
            self.al_queue_updated.emit()

    def _apply_label_suggestions(self) -> None:
        """Write high-confidence predictions as project labels for unlabeled images."""
        if not self.project:
            QMessageBox.warning(self, tr("common.no_project"), tr("inference.no_project_msg"))
            return
        if not self._all_results:
            self._sl_status.setText("Bitte zuerst Bilder klassifizieren.")
            return

        min_conf = self._sl_conf_spin.value()
        overwrite = self._sl_overwrite_cb.isChecked()
        project_labels = set(self.project.labels.keys())

        applied = skipped_label = skipped_conf = skipped_exists = 0
        for r in self._all_results:
            if r.get("error"):
                continue
            conf = r.get("confidence", 0.0)
            if conf < min_conf:
                skipped_conf += 1
                continue
            pred = r.get("predicted_label", "")
            if pred not in project_labels:
                skipped_label += 1
                continue
            path = r.get("path", "")
            if not os.path.isfile(path):
                continue
            already_labeled = bool(self.project.image_labels.get(path))
            if already_labeled and not overwrite:
                skipped_exists += 1
                continue
            self.project.image_labels[path] = pred
            applied += 1

        if applied > 0:
            self.project.save()

        parts = [f"{applied} Labels übernommen"]
        if skipped_conf:
            parts.append(f"{skipped_conf} unter Schwelle")
        if skipped_exists:
            parts.append(f"{skipped_exists} bereits gelabelt")
        if skipped_label:
            parts.append(f"{skipped_label} unbekannte Labels")
        self._sl_status.setText("  |  ".join(parts))
        if applied > 0:
            self.labels_applied.emit(applied)

    def _show_gradcam(self) -> None:
        """Open the Grad-CAM dialog for the currently selected result row."""
        row = self.table.currentRow()
        if row < 0 or row >= len(self._filtered):
            return
        r = self._filtered[row]
        path = r.get("path", "")
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Datei nicht gefunden", f"Bilddatei nicht gefunden:\n{path}")
            return

        model = getattr(self.inferencer, "model", None)
        model_type = getattr(self.inferencer, "model_type", "")
        class_names = getattr(self.inferencer, "class_names", [])
        image_size  = getattr(self.inferencer, "image_size", 224)

        if model is None:
            QMessageBox.warning(self, tr("common.no_model"), "Bitte erst ein Modell laden.")
            return

        # Map predicted label to class index
        pred_label = r.get("predicted_label", "")
        class_idx: Optional[int] = None
        if pred_label and class_names:
            try:
                class_idx = class_names.index(pred_label)
            except ValueError:
                class_idx = None

        # Look up the project ROI for this image — the model was trained on crops,
        # so Grad-CAM must see the same crop to produce meaningful activations.
        roi = None
        if self.project:
            rois = self.project.get_rois(path)
            if rois:
                # Prefer the ROI whose label matches the predicted class
                roi = next(
                    (r for r in rois if r.get("label") == pred_label),
                    rois[0],
                )

        from gui.gradcam_dialog import GradCAMDialog
        dlg = GradCAMDialog(
            model=model,
            model_type=model_type,
            image_path=path,
            class_names=class_names,
            class_idx=class_idx,
            image_size=image_size,
            roi=roi,
            parent=self,
        )
        dlg.exec()

    def _export_excel(self) -> None:
        """Save all inference results to an Excel workbook."""
        if not self._all_results:
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
                self._all_results, path,
                model_name=os.path.basename(self.inferencer.model_path)
            )
            QMessageBox.information(self, tr("common.done"), f"Gespeichert:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, tr("common.error"), str(exc))

    def _rename_images(self) -> None:
        """Rename each classified image to append its predicted label before the extension."""
        results = self._all_results
        renameable = [r for r in results if not r.get("error") and os.path.isfile(r.get("path", ""))]
        if not renameable:
            QMessageBox.information(self, tr("inference.rename_btn"),
                                    "Keine gültigen Bilddateien zum Umbenennen gefunden.")
            return

        # Show preview of first 5 renames
        preview_lines = []
        for r in renameable[:5]:
            old = os.path.basename(r["path"])
            stem, ext = os.path.splitext(old)
            new = f"{stem}_{r['predicted_label']}{ext}"
            preview_lines.append(f"  {old}  →  {new}")
        if len(renameable) > 5:
            preview_lines.append(f"  … und {len(renameable) - 5} weitere")

        reply = QMessageBox.question(
            self, tr("inference.rename_btn"),
            f"{len(renameable)} Dateien werden umbenannt:\n\n"
            + "\n".join(preview_lines)
            + "\n\nFortfahren?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        renamed = 0
        skipped = 0
        errors = []
        path_map: dict = {}   # old_path → new_path  (for project update)

        for r in renameable:
            old_path = r["path"]
            stem, ext = os.path.splitext(old_path)
            label_tag = r["predicted_label"].replace(" ", "_")
            new_path = f"{stem}_{label_tag}{ext}"

            # Skip if already renamed (has the tag) or target exists
            if os.path.exists(new_path):
                skipped += 1
                continue
            try:
                os.rename(old_path, new_path)
                path_map[old_path] = new_path
                r["path"] = new_path
                r["filename"] = os.path.basename(new_path)
                renamed += 1
            except OSError as exc:
                errors.append(f"{os.path.basename(old_path)}: {exc}")

        # Update project image list and labels if a project is loaded
        if self.project and path_map:
            self.project.relocate_images.__func__  # existence check (noop)
            for old, new in path_map.items():
                if old in self.project.images:
                    idx = self.project.images.index(old)
                    self.project.images[idx] = new
                if old in self.project.image_labels:
                    self.project.image_labels[new] = self.project.image_labels.pop(old)
                if old in self.project.rois:
                    self.project.rois[new] = self.project.rois.pop(old)

        # Refresh table with updated filenames
        self._populate_table(self._filtered if self._filtered else self._all_results)

        msg = f"{renamed} Dateien umbenannt."
        if skipped:
            msg += f"\n{skipped} übersprungen (Zieldatei existiert bereits)."
        if errors:
            msg += f"\n{len(errors)} Fehler:\n" + "\n".join(errors[:5])
        QMessageBox.information(self, "Umbenennen abgeschlossen", msg)
