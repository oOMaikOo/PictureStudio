"""
Training configuration panel + live progress display.
Runs training in a QThread to keep the GUI responsive.
"""
import os
from typing import Dict, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QProgressBar, QTextEdit, QSplitter,
    QTabWidget, QMessageBox, QFileDialog, QSlider,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont

from models.classifier import get_available_models
from utils.config import DEFAULT_TRAIN_CONFIG
from core.metrics import format_metrics_text


class TrainingThread(QThread):
    progress = Signal(int, int, float, float, float, float)  # epoch, total, tl, vl, ta, va
    log_msg = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, project, training_config: Dict, save_dir: str):
        super().__init__()
        self.project = project
        self.training_config = training_config
        self.save_dir = save_dir
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            from core.training import TrainingWorker
            worker = TrainingWorker(
                project=self.project,
                training_config=self.training_config,
                save_dir=self.save_dir,
                progress_callback=lambda *a: self.progress.emit(*a),
                log_callback=lambda m: self.log_msg.emit(m),
                stop_flag=lambda: self._stop_requested,
            )
            result = worker.run()
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class TrainingPanel(QWidget):
    """Full training panel: config + progress + results."""

    training_finished = Signal(dict)  # emitted so MainWindow can update project

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._thread: Optional[TrainingThread] = None
        self._history: Dict = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        self._build_ui()

    def set_project(self, project) -> None:
        self.project = project
        save_dir = os.path.join(project.get_project_dir(), "models") if project.get_project_dir() else "models"
        self.save_dir_label.setText(save_dir)

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        splitter.addWidget(self._build_config_panel())
        splitter.addWidget(self._build_progress_panel())
        splitter.setSizes([350, 650])

    def _build_config_panel(self) -> QGroupBox:
        box = QGroupBox("Trainingsparameter")
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.model_combo = QComboBox()
        self.model_combo.addItems(get_available_models())
        form.addRow("Modellarchitektur:", self.model_combo)

        self.pretrained_cb = QCheckBox("Vortrainierte Gewichte (ImageNet)")
        self.pretrained_cb.setChecked(True)
        form.addRow("", self.pretrained_cb)

        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(32, 1024)
        self.img_size_spin.setValue(DEFAULT_TRAIN_CONFIG["image_size"])
        self.img_size_spin.setSingleStep(32)
        form.addRow("Bildgröße (px):", self.img_size_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(DEFAULT_TRAIN_CONFIG["batch_size"])
        form.addRow("Batch-Größe:", self.batch_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(DEFAULT_TRAIN_CONFIG["epochs"])
        form.addRow("Epochen:", self.epochs_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setValue(DEFAULT_TRAIN_CONFIG["learning_rate"])
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(0.0001)
        form.addRow("Learning Rate:", self.lr_spin)

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["adam", "sgd"])
        form.addRow("Optimizer:", self.optimizer_combo)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(DEFAULT_TRAIN_CONFIG["seed"])
        form.addRow("Seed:", self.seed_spin)

        # Splits
        split_box = QGroupBox("Daten-Split")
        split_layout = QFormLayout(split_box)

        self.train_split_spin = QDoubleSpinBox()
        self.train_split_spin.setRange(0.1, 0.9)
        self.train_split_spin.setSingleStep(0.05)
        self.train_split_spin.setValue(DEFAULT_TRAIN_CONFIG["train_split"])
        split_layout.addRow("Train:", self.train_split_spin)

        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.05, 0.5)
        self.val_split_spin.setSingleStep(0.05)
        self.val_split_spin.setValue(DEFAULT_TRAIN_CONFIG["val_split"])
        split_layout.addRow("Validation:", self.val_split_spin)
        form.addRow(split_box)

        # Augmentation
        aug_box = QGroupBox("Datenaugmentation")
        aug_layout = QVBoxLayout(aug_box)
        self.aug_rotation_cb = QCheckBox("Rotation (±15°)")
        self.aug_rotation_cb.setChecked(True)
        self.aug_flip_cb = QCheckBox("Horizontaler Flip")
        self.aug_flip_cb.setChecked(True)
        self.aug_brightness_cb = QCheckBox("Helligkeit/Kontrast")
        self.aug_brightness_cb.setChecked(True)
        self.aug_scale_cb = QCheckBox("Skalierung (Crop)")
        for cb in [self.aug_rotation_cb, self.aug_flip_cb, self.aug_brightness_cb, self.aug_scale_cb]:
            aug_layout.addWidget(cb)
        form.addRow(aug_box)

        self.use_rois_cb = QCheckBox("ROI-Bereiche verwenden (falls definiert)")
        self.use_rois_cb.setChecked(True)
        form.addRow("", self.use_rois_cb)

        # Save dir
        self.save_dir_label = QLabel("(Projekt öffnen)")
        self.save_dir_label.setWordWrap(True)
        form.addRow("Speicherort:", self.save_dir_label)

        # Buttons
        btn_layout = QVBoxLayout()
        self.start_btn = QPushButton("Training starten")
        self.start_btn.setStyleSheet("background-color: #2ECC71; color: white; font-weight: bold; padding: 8px;")
        self.start_btn.clicked.connect(self._start_training)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Training stoppen")
        self.stop_btn.setStyleSheet("background-color: #E74C3C; color: white; padding: 8px;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)
        btn_layout.addWidget(self.stop_btn)

        form.addRow(btn_layout)
        return box

    def _build_progress_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Progress bar
        prog_row = QHBoxLayout()
        self.epoch_label = QLabel("Epoche: 0 / 0")
        prog_row.addWidget(self.epoch_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        prog_row.addWidget(self.progress_bar)
        layout.addLayout(prog_row)

        # Live metrics row
        metrics_row = QHBoxLayout()
        self.train_loss_label = QLabel("Train-Loss: -")
        self.val_loss_label = QLabel("Val-Loss: -")
        self.train_acc_label = QLabel("Train-Acc: -")
        self.val_acc_label = QLabel("Val-Acc: -")
        for lbl in [self.train_loss_label, self.val_loss_label, self.train_acc_label, self.val_acc_label]:
            lbl.setStyleSheet("font-weight: bold; padding: 4px 8px; background: #34495E; color: white; border-radius: 4px;")
            metrics_row.addWidget(lbl)
        layout.addLayout(metrics_row)

        # Tab: Log / Results / Confusion Matrix
        self.result_tabs = QTabWidget()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier New", 9))
        self.result_tabs.addTab(self.log_text, "Log")

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setFont(QFont("Courier New", 10))
        self.result_tabs.addTab(self.metrics_text, "Metriken")

        self.confusion_widget = ConfusionMatrixWidget()
        self.result_tabs.addTab(self.confusion_widget, "Konfusionsmatrix")

        self.curves_widget = TrainingCurvesWidget()
        self.result_tabs.addTab(self.curves_widget, "Trainingskurven")

        layout.addWidget(self.result_tabs)
        return widget

    # ------------------------------------------------------------------ training control

    def _get_config(self) -> Dict:
        return {
            "model_type": self.model_combo.currentText(),
            "use_pretrained": self.pretrained_cb.isChecked(),
            "image_size": self.img_size_spin.value(),
            "batch_size": self.batch_spin.value(),
            "epochs": self.epochs_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "optimizer": self.optimizer_combo.currentText(),
            "seed": self.seed_spin.value(),
            "train_split": self.train_split_spin.value(),
            "val_split": self.val_split_spin.value(),
            "test_split": max(0.0, 1.0 - self.train_split_spin.value() - self.val_split_spin.value()),
            "use_rois": self.use_rois_cb.isChecked(),
            "augmentation": {
                "rotation": self.aug_rotation_cb.isChecked(),
                "flip": self.aug_flip_cb.isChecked(),
                "brightness": self.aug_brightness_cb.isChecked(),
                "contrast": self.aug_brightness_cb.isChecked(),
                "scale": self.aug_scale_cb.isChecked(),
            },
        }

    def _start_training(self) -> None:
        if self.project is None:
            QMessageBox.warning(self, "Kein Projekt", "Bitte zuerst ein Projekt öffnen.")
            return
        if not self.project.labels:
            QMessageBox.warning(self, "Keine Labels", "Bitte mindestens 2 Labels definieren.")
            return

        cfg = self._get_config()
        save_dir = os.path.join(self.project.get_project_dir() or ".", "models")

        self.log_text.clear()
        self.metrics_text.clear()
        self._history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        self.progress_bar.setValue(0)

        self._thread = TrainingThread(self.project, cfg, save_dir)
        self._thread.progress.connect(self._on_progress)
        self._thread.log_msg.connect(self._on_log)
        self._thread.finished.connect(self._on_finished)
        self._thread.error.connect(self._on_error)
        self._thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _stop_training(self) -> None:
        if self._thread:
            self._thread.request_stop()
        self.stop_btn.setEnabled(False)

    # ------------------------------------------------------------------ slots

    @Slot(int, int, float, float, float, float)
    def _on_progress(self, epoch: int, total: int, tl: float, vl: float, ta: float, va: float) -> None:
        pct = int(epoch / total * 100)
        self.progress_bar.setValue(pct)
        self.epoch_label.setText(f"Epoche: {epoch} / {total}")
        self.train_loss_label.setText(f"Train-Loss: {tl:.4f}")
        self.val_loss_label.setText(f"Val-Loss: {vl:.4f}")
        self.train_acc_label.setText(f"Train-Acc: {ta*100:.1f}%")
        self.val_acc_label.setText(f"Val-Acc: {va*100:.1f}%")

        self._history["train_loss"].append(tl)
        self._history["val_loss"].append(vl)
        self._history["train_acc"].append(ta)
        self._history["val_acc"].append(va)
        self.curves_widget.update_curves(self._history)

    @Slot(str)
    def _on_log(self, msg: str) -> None:
        self.log_text.append(msg)

    @Slot(dict)
    def _on_finished(self, result: Dict) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)

        metrics = result.get("metrics", {})
        self.metrics_text.setPlainText(format_metrics_text(metrics))

        cm = metrics.get("confusion_matrix", [])
        class_names = result.get("class_names", [])
        self.confusion_widget.set_matrix(cm, class_names)

        self.result_tabs.setCurrentIndex(1)
        self.training_finished.emit(result)
        QMessageBox.information(self, "Training abgeschlossen",
                                f"Bestes Modell gespeichert:\n{result.get('best_model_path', '')}")

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_text.append(f"FEHLER: {msg}")
        QMessageBox.critical(self, "Trainingsfehler", msg)

    def apply_project_config(self) -> None:
        """Fill UI from project.training_config."""
        if not self.project:
            return
        cfg = self.project.training_config
        idx = self.model_combo.findText(cfg.get("model_type", "resnet18"))
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.pretrained_cb.setChecked(cfg.get("use_pretrained", True))
        self.img_size_spin.setValue(cfg.get("image_size", 224))
        self.batch_spin.setValue(cfg.get("batch_size", 16))
        self.epochs_spin.setValue(cfg.get("epochs", 20))
        self.lr_spin.setValue(cfg.get("learning_rate", 0.001))
        self.seed_spin.setValue(cfg.get("seed", 42))


# ------------------------------------------------------------------ sub-widgets

class ConfusionMatrixWidget(QWidget):
    """Renders a confusion matrix as a simple HTML table."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text)

    def set_matrix(self, cm, class_names) -> None:
        if not cm or not class_names:
            self.text.setPlainText("Keine Konfusionsmatrix verfügbar.")
            return
        html = "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse;'>"
        html += "<tr><th>T\\P</th>"
        for cn in class_names:
            html += f"<th style='background:#2196F3;color:white;'>{cn}</th>"
        html += "</tr>"
        for i, row in enumerate(cm):
            html += f"<tr><td style='background:#2196F3;color:white;font-weight:bold;'>{class_names[i]}</td>"
            for j, val in enumerate(row):
                color = "#2ECC71" if i == j else ("#E74C3C" if val > 0 else "white")
                html += f"<td style='background:{color};text-align:center;'>{val}</td>"
            html += "</tr>"
        html += "</table>"
        self.text.setHtml(html)


class TrainingCurvesWidget(QWidget):
    """Simple ASCII-art training curves (no matplotlib dependency required)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFont("Courier New", 9))
        layout.addWidget(self.text)

    def update_curves(self, history: Dict) -> None:
        lines = ["=== Trainingskurven ===", ""]
        for key in ["train_loss", "val_loss", "train_acc", "val_acc"]:
            values = history.get(key, [])
            if not values:
                continue
            label = {"train_loss": "Train-Loss", "val_loss": "Val-Loss",
                     "train_acc": "Train-Acc", "val_acc": "Val-Acc"}.get(key, key)
            lines.append(f"{label}:")
            lines.append(self._spark(values))
            lines.append(f"  Zuletzt: {values[-1]:.4f}  Min: {min(values):.4f}  Max: {max(values):.4f}")
            lines.append("")
        self.text.setPlainText("\n".join(lines))

    @staticmethod
    def _spark(values, width=50) -> str:
        if not values:
            return ""
        mn, mx = min(values), max(values)
        rng = mx - mn if mx != mn else 1
        chars = "▁▂▃▄▅▆▇█"
        out = ""
        for v in values[-width:]:
            idx = int((v - mn) / rng * (len(chars) - 1))
            out += chars[idx]
        return "  " + out
