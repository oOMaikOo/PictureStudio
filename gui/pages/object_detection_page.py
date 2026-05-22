"""
Object Detection Page (stack index 15) — YOLOv8-based detection training and inference.

Workflow:
  1. User annotates images with labeled ROIs in the Labeling page.
  2. Here: prepare dataset → train YOLOv8 → run detection on new images.
"""
import os
import tempfile
from typing import List, Dict, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QProgressBar, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QMessageBox, QCheckBox, QScrollArea,
    QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt, Slot, QThread, Signal
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QBrush

from core.object_detection import ObjectDetector, has_ultralytics


class ObjectDetectionPage(QWidget):
    """YOLOv8 object detection: dataset prep, training, and folder inference (stack index 15)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._detector = ObjectDetector()
        self._dataset_dir: Optional[str] = None
        self._train_thread = None
        self._infer_thread = None
        self._all_results: List[Dict] = []
        self._build_ui()
        self._check_ultralytics()

    # ------------------------------------------------------------------ UI build

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_center())
        splitter.addWidget(self._build_right())
        splitter.setSizes([260, 600, 320])

    # ---- Left: config ----

    def _build_left(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)

        # Dependency warning
        self._dep_label = QLabel()
        self._dep_label.setWordWrap(True)
        self._dep_label.setStyleSheet(
            "background:#7f1d1d;color:#fca5a5;padding:6px;border-radius:4px;font-size:10px;"
        )
        self._dep_label.hide()
        v.addWidget(self._dep_label)

        # Dataset info
        ds_box = QGroupBox("Dataset (ROI-Annotationen)")
        ds_v = QVBoxLayout(ds_box)
        self._ds_info_label = QLabel("Noch kein Projekt geladen.")
        self._ds_info_label.setWordWrap(True)
        self._ds_info_label.setStyleSheet("color:#aaa;font-size:10px;")
        ds_v.addWidget(self._ds_info_label)
        self._prepare_btn = QPushButton("Dataset vorbereiten")
        self._prepare_btn.setToolTip(
            "Konvertiert die ROI-Annotationen des Projekts in das YOLO-Format\n"
            "und legt den Trainings-/Validierungsordner an."
        )
        self._prepare_btn.clicked.connect(self._prepare_dataset)
        ds_v.addWidget(self._prepare_btn)
        v.addWidget(ds_box)

        # Training config
        train_box = QGroupBox("Training")
        tf = QVBoxLayout(train_box)

        tf.addWidget(QLabel("Modellgröße:"))
        self._model_combo = QComboBox()
        for key, desc in ObjectDetector.MODEL_SIZES.items():
            self._model_combo.addItem(f"{key}  ({desc})", key)
        self._model_combo.setCurrentIndex(0)
        tf.addWidget(self._model_combo)

        row_epochs = QHBoxLayout()
        row_epochs.addWidget(QLabel("Epochen:"))
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 500)
        self._epochs_spin.setValue(50)
        row_epochs.addWidget(self._epochs_spin)
        tf.addLayout(row_epochs)

        row_imgsz = QHBoxLayout()
        row_imgsz.addWidget(QLabel("Bildgröße:"))
        self._imgsz_spin = QSpinBox()
        self._imgsz_spin.setRange(320, 1280)
        self._imgsz_spin.setSingleStep(32)
        self._imgsz_spin.setValue(640)
        self._imgsz_spin.setToolTip("Standard: 640. Höher = genauer, langsamer.")
        row_imgsz.addWidget(self._imgsz_spin)
        tf.addLayout(row_imgsz)

        row_batch = QHBoxLayout()
        row_batch.addWidget(QLabel("Batch-Größe:"))
        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 128)
        self._batch_spin.setValue(16)
        row_batch.addWidget(self._batch_spin)
        tf.addLayout(row_batch)

        self._device_combo = QComboBox()
        self._device_combo.addItems(["auto", "cpu", "cuda", "mps"])
        tf.addWidget(QLabel("Gerät:"))
        tf.addWidget(self._device_combo)

        self._train_btn = QPushButton("⚡ Training starten")
        self._train_btn.setStyleSheet(
            "background:#1565C0;color:white;padding:6px;border-radius:4px;font-weight:bold;"
        )
        self._train_btn.clicked.connect(self._start_training)
        tf.addWidget(self._train_btn)

        self._stop_btn = QPushButton("■ Stopp")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_training)
        tf.addWidget(self._stop_btn)

        self._train_progress = QProgressBar()
        self._train_progress.setRange(0, 100)
        self._train_progress.setTextVisible(True)
        tf.addWidget(self._train_progress)

        self._train_status = QLabel("")
        self._train_status.setWordWrap(True)
        self._train_status.setStyleSheet("font-size:9px;color:#aaa;")
        tf.addWidget(self._train_status)
        v.addWidget(train_box)

        # Model load
        model_box = QGroupBox("Modell laden")
        mv = QVBoxLayout(model_box)
        self._load_model_btn = QPushButton("Modell laden (.pt)…")
        self._load_model_btn.clicked.connect(self._load_model)
        mv.addWidget(self._load_model_btn)
        self._model_info_label = QLabel("Kein Modell geladen.")
        self._model_info_label.setWordWrap(True)
        self._model_info_label.setStyleSheet("color:#aaa;font-size:9px;")
        mv.addWidget(self._model_info_label)
        v.addWidget(model_box)

        v.addStretch()
        return w

    # ---- Center: image + boxes ----

    def _build_center(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)

        # Single image inference
        img_row = QHBoxLayout()
        self._pick_img_btn = QPushButton("Bild wählen…")
        self._pick_img_btn.clicked.connect(self._pick_single_image)
        img_row.addWidget(self._pick_img_btn)

        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.05, 0.95)
        self._conf_spin.setValue(0.25)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setToolTip("Mindest-Konfidenz für Erkennungen (0.25 = Standard)")
        img_row.addWidget(QLabel("Konf.:"))
        img_row.addWidget(self._conf_spin)
        img_row.addStretch()
        v.addLayout(img_row)

        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setMinimumHeight(300)
        self._img_label.setStyleSheet("background:#1a1a1a;border-radius:4px;")
        self._img_label.setText("← Bild wählen oder Ordner klassifizieren")
        self._img_label.setWordWrap(True)
        v.addWidget(self._img_label, 1)

        # Training log
        log_box = QGroupBox("Training-Log")
        lv = QVBoxLayout(log_box)
        self._log_edit = QTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setMaximumHeight(160)
        self._log_edit.setFont(QFont("Courier", 9))
        lv.addWidget(self._log_edit)
        v.addWidget(log_box)

        return w

    # ---- Right: folder inference + results ----

    def _build_right(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)

        infer_box = QGroupBox("Ordner-Erkennung")
        iv = QVBoxLayout(infer_box)

        folder_row = QHBoxLayout()
        self._folder_label = QLabel("(kein Ordner)")
        self._folder_label.setWordWrap(True)
        folder_row.addWidget(self._folder_label, 1)
        pick_folder_btn = QPushButton("Ordner…")
        pick_folder_btn.clicked.connect(self._pick_folder)
        folder_row.addWidget(pick_folder_btn)
        iv.addLayout(folder_row)

        self._recursive_cb = QCheckBox("Unterordner einschließen")
        iv.addWidget(self._recursive_cb)

        self._infer_btn = QPushButton("Erkennung starten")
        self._infer_btn.setStyleSheet(
            "background:#00695C;color:white;padding:5px;border-radius:4px;"
        )
        self._infer_btn.clicked.connect(self._start_folder_inference)
        iv.addWidget(self._infer_btn)

        self._infer_progress = QProgressBar()
        self._infer_progress.setRange(0, 100)
        iv.addWidget(self._infer_progress)
        v.addWidget(infer_box)

        # Results table
        res_box = QGroupBox("Ergebnisse")
        rv = QVBoxLayout(res_box)
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Datei", "Objekte", "Labels", "Fehler"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.itemSelectionChanged.connect(self._on_result_selected)
        rv.addWidget(self._table)

        export_btn = QPushButton("CSV exportieren")
        export_btn.clicked.connect(self._export_csv)
        rv.addWidget(export_btn)
        v.addWidget(res_box, 1)

        return w

    # ------------------------------------------------------------------ project

    def set_project(self, project) -> None:
        self.project = project
        self._dataset_dir = None
        self._update_ds_info()

    def _update_ds_info(self):
        if not self.project:
            self._ds_info_label.setText("Noch kein Projekt geladen.")
            return
        total = len(self.project.images)
        annotated = sum(
            1 for p in self.project.images
            if any(r.get("label") for r in self.project.get_rois(p))
        )
        classes = list(self.project.labels.keys())
        self._ds_info_label.setText(
            f"{annotated} / {total} Bilder mit ROI-Labels\n"
            f"Klassen: {', '.join(classes) if classes else '(keine)'}"
        )

    # ------------------------------------------------------------------ dependency check

    def _check_ultralytics(self):
        if not has_ultralytics():
            self._dep_label.setText(
                "⚠ ultralytics nicht installiert.\n"
                "Bitte ausführen:\n  pip install ultralytics"
            )
            self._dep_label.show()
            self._train_btn.setEnabled(False)
            self._prepare_btn.setEnabled(False)
            self._infer_btn.setEnabled(False)

    # ------------------------------------------------------------------ dataset

    def _prepare_dataset(self):
        if not self.project:
            QMessageBox.warning(self, "Kein Projekt", "Bitte zuerst ein Projekt öffnen.")
            return
        from core.detection_dataset import prepare_yolo_dataset

        proj_dir = os.path.dirname(self.project.project_path or "") if self.project.project_path else tempfile.gettempdir()
        self._dataset_dir = os.path.join(proj_dir, "yolo_dataset")
        os.makedirs(self._dataset_dir, exist_ok=True)

        try:
            yaml_path, stats = prepare_yolo_dataset(self.project, self._dataset_dir)
            self._ds_info_label.setText(
                f"✓ Dataset bereit\n"
                f"{stats['n_train']} Train / {stats['n_val']} Val\n"
                f"{stats['n_classes']} Klassen | {stats['n_annotations']} Annotationen\n"
                f"Pfad: {self._dataset_dir}"
            )
            self._ds_info_label.setStyleSheet("color:#4caf50;font-size:10px;")
            self._log("Dataset vorbereitet: " + yaml_path)
        except Exception as exc:
            QMessageBox.critical(self, "Fehler", str(exc))

    # ------------------------------------------------------------------ training

    def _start_training(self):
        if not has_ultralytics():
            return
        if not self._dataset_dir or not os.path.exists(
            os.path.join(self._dataset_dir, "data.yaml")
        ):
            reply = QMessageBox.question(
                self, "Dataset fehlt",
                "Dataset noch nicht vorbereitet. Jetzt vorbereiten?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._prepare_dataset()
            if not self._dataset_dir:
                return

        yaml_path = os.path.join(self._dataset_dir, "data.yaml")
        if not os.path.exists(yaml_path):
            QMessageBox.warning(self, "Fehler", "data.yaml nicht gefunden.")
            return

        proj_dir = (
            os.path.dirname(self.project.project_path)
            if self.project and self.project.project_path else ""
        )

        from core.object_detection import DetectionTrainingThread
        self._train_thread = DetectionTrainingThread(
            data_yaml=yaml_path,
            model_size=self._model_combo.currentData(),
            epochs=self._epochs_spin.value(),
            imgsz=self._imgsz_spin.value(),
            batch=self._batch_spin.value(),
            device=self._device_combo.currentText(),
            project_dir=proj_dir,
        )
        self._train_thread.progress.connect(self._on_train_progress)
        self._train_thread.log_line.connect(self._log)
        self._train_thread.finished.connect(self._on_train_finished)
        self._train_thread.error.connect(self._on_train_error)

        self._train_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._train_progress.setValue(0)
        self._train_status.setText("Training läuft…")
        self._log_edit.clear()
        self._log(f"Starte Training: {self._model_combo.currentData()}, "
                  f"{self._epochs_spin.value()} Epochen")
        self._train_thread.start()

    def _stop_training(self):
        if self._train_thread:
            self._train_thread.stop()
        self._stop_btn.setEnabled(False)
        self._train_status.setText("Stopp angefordert…")

    @Slot(int, int, float, float)
    def _on_train_progress(self, epoch: int, total: int, box_loss: float, cls_loss: float):
        pct = int(epoch / total * 100) if total else 0
        self._train_progress.setValue(pct)
        self._train_status.setText(
            f"Epoche {epoch}/{total} | box={box_loss:.4f} cls={cls_loss:.4f}"
        )

    @Slot(str)
    def _on_train_finished(self, best_path: str):
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._train_progress.setValue(100)
        self._train_status.setText("Training abgeschlossen ✓")
        self._log(f"Bestes Modell: {best_path}")
        if best_path and os.path.exists(best_path):
            reply = QMessageBox.question(
                self, "Training abgeschlossen",
                f"Training erfolgreich.\nModell laden?\n{best_path}",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._do_load_model(best_path)

    @Slot(str)
    def _on_train_error(self, msg: str):
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._train_status.setText("Fehler beim Training.")
        self._log(f"FEHLER: {msg}")
        QMessageBox.critical(self, "Trainingsfehler", msg)

    # ------------------------------------------------------------------ model

    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Detektionsmodell laden", "", "YOLO Modell (*.pt)"
        )
        if path:
            self._do_load_model(path)

    def _do_load_model(self, path: str):
        try:
            self._detector.load(path)
            classes = ", ".join(self._detector.class_names)
            self._model_info_label.setText(
                f"✓ {os.path.basename(path)}\nKlassen: {classes}"
            )
            self._model_info_label.setStyleSheet("color:#4caf50;font-size:9px;")
        except Exception as exc:
            QMessageBox.critical(self, "Fehler", str(exc))

    # ------------------------------------------------------------------ single image

    def _pick_single_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Bild wählen", "",
            "Bilder (*.jpg *.jpeg *.png *.bmp *.tiff *.webp)"
        )
        if not path:
            return
        if not self._detector.is_ready():
            QMessageBox.warning(self, "Kein Modell", "Bitte zuerst ein Modell laden.")
            return
        try:
            dets = self._detector.predict_image(path, conf=self._conf_spin.value())
            self._show_image_with_boxes(path, dets)
        except Exception as exc:
            QMessageBox.critical(self, "Fehler", str(exc))

    def _show_image_with_boxes(self, image_path: str, detections: List[Dict]):
        pix = QPixmap(image_path)
        if pix.isNull():
            return

        # Scale to fit the label while keeping aspect ratio
        label_size = self._img_label.size()
        scaled = pix.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        sx = scaled.width()  / pix.width()
        sy = scaled.height() / pix.height()

        result = QPixmap(scaled)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing)

        COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
                  "#1ABC9C", "#E67E22", "#E91E63", "#00BCD4", "#8BC34A"]

        label_map = {n: COLORS[i % len(COLORS)]
                     for i, n in enumerate(self._detector.class_names)}

        font = QFont()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)

        for det in detections:
            color = QColor(label_map.get(det["label"], "#E74C3C"))
            pen = QPen(color, 2)
            painter.setPen(pen)
            x1 = int(det["x1"] * sx)
            y1 = int(det["y1"] * sy)
            x2 = int(det["x2"] * sx)
            y2 = int(det["y2"] * sy)
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

            lbl_text = f"{det['label']} {det['confidence']*100:.0f}%"
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(lbl_text) + 6
            th = fm.height() + 2
            bg = QColor(color)
            bg.setAlpha(200)
            painter.fillRect(x1, max(0, y1 - th), tw, th, QBrush(bg))
            painter.setPen(QPen(Qt.white))
            painter.drawText(x1 + 3, max(th, y1) - 2, lbl_text)

        painter.end()
        self._img_label.setPixmap(result)

    # ------------------------------------------------------------------ folder inference

    def _pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Ordner wählen")
        if folder:
            self._folder_label.setText(folder)

    def _start_folder_inference(self):
        if not self._detector.is_ready():
            QMessageBox.warning(self, "Kein Modell", "Bitte zuerst ein Modell laden.")
            return
        folder = self._folder_label.text()
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "Kein Ordner", "Bitte einen gültigen Ordner wählen.")
            return

        from core.object_detection import DetectionInferenceThread
        self._infer_thread = DetectionInferenceThread(
            self._detector, folder,
            conf=self._conf_spin.value(),
            recursive=self._recursive_cb.isChecked(),
        )
        self._infer_thread.progress.connect(
            lambda c, t: self._infer_progress.setValue(int(c / t * 100) if t else 0)
        )
        self._infer_thread.finished.connect(self._on_infer_finished)
        self._infer_thread.error.connect(lambda e: QMessageBox.critical(self, "Fehler", e))
        self._infer_btn.setEnabled(False)
        self._infer_progress.setValue(0)
        self._infer_thread.start()

    @Slot(list)
    def _on_infer_finished(self, results: List[Dict]):
        self._all_results = results
        self._infer_btn.setEnabled(True)
        self._infer_progress.setValue(100)
        self._populate_table(results)

    def _populate_table(self, results: List[Dict]):
        self._table.setRowCount(0)
        for r in results:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(r["filename"]))
            self._table.setItem(row, 1, QTableWidgetItem(str(r["n_objects"])))
            labels_str = ", ".join(
                f"{d['label']}({d['confidence']:.0%})"
                for d in r["detections"]
            ) if r["detections"] else "–"
            self._table.setItem(row, 2, QTableWidgetItem(labels_str))
            err_item = QTableWidgetItem(r.get("error") or "")
            if r.get("error"):
                err_item.setForeground(QColor("#ef5350"))
            self._table.setItem(row, 3, err_item)
            if r.get("error"):
                for col in range(4):
                    item = self._table.item(row, col)
                    if item:
                        item.setBackground(QColor(40, 10, 10))

    def _on_result_selected(self):
        rows = {i.row() for i in self._table.selectedItems()}
        if not rows:
            return
        row = min(rows)
        if row >= len(self._all_results):
            return
        r = self._all_results[row]
        if r.get("detections") and self._detector.is_ready():
            self._show_image_with_boxes(r["path"], r["detections"])
        else:
            pix = QPixmap(r["path"])
            if not pix.isNull():
                self._img_label.setPixmap(
                    pix.scaled(self._img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )

    # ------------------------------------------------------------------ CSV export

    def _export_csv(self):
        if not self._all_results:
            QMessageBox.information(self, "Keine Daten", "Bitte zuerst eine Erkennung starten.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "CSV speichern", "detection_results.csv", "CSV (*.csv)"
        )
        if not path:
            return
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Datei", "Pfad", "n_Objekte", "Label", "Konfidenz",
                             "x1", "y1", "x2", "y2", "Fehler"])
            for r in self._all_results:
                if r["detections"]:
                    for d in r["detections"]:
                        writer.writerow([
                            r["filename"], r["path"], r["n_objects"],
                            d["label"], d["confidence"],
                            round(d["x1"]), round(d["y1"]),
                            round(d["x2"]), round(d["y2"]),
                            r.get("error") or "",
                        ])
                else:
                    writer.writerow([
                        r["filename"], r["path"], 0,
                        "", "", "", "", "", "",
                        r.get("error") or "",
                    ])
        QMessageBox.information(self, "Exportiert", f"CSV gespeichert: {path}")

    # ------------------------------------------------------------------ log helper

    def _log(self, msg: str):
        self._log_edit.append(msg)
        self._log_edit.verticalScrollBar().setValue(
            self._log_edit.verticalScrollBar().maximum()
        )
