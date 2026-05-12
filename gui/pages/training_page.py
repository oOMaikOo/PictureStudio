"""
Training page: config, live progress, metrics, curves, confusion matrix.
"""
import os
from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QProgressBar, QTextEdit, QSplitter, QTabWidget,
    QMessageBox, QFileDialog, QLineEdit,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont

from models.classifier import get_available_models
from utils.config import DEFAULT_TRAIN_CONFIG
from core.metrics import format_metrics_text


class TrainingThread(QThread):
    progress = Signal(int, int, float, float, float, float)
    log_msg  = Signal(str)
    finished = Signal(dict)
    error    = Signal(str)

    def __init__(self, project, cfg: Dict, save_dir: str):
        super().__init__()
        self.project = project
        self.cfg = cfg
        self.save_dir = save_dir
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            from core.training import TrainingWorker
            worker = TrainingWorker(
                project=self.project,
                training_config=self.cfg,
                save_dir=self.save_dir,
                progress_callback=lambda *a: self.progress.emit(*a),
                log_callback=lambda m: self.log_msg.emit(m),
                stop_flag=lambda: self._stop,
            )
            self.finished.emit(worker.run())
        except Exception as exc:
            self.error.emit(str(exc))


class TrainingPage(QWidget):
    training_finished = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._thread: Optional[QThread] = None
        self._history: Dict = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}
        self._audit = None
        self._settings = None
        self._ssh_profiles: List[Dict] = []
        self._test_predictions: List[Dict] = []
        self._last_class_names: List[str] = []
        self._build_ui()

    def set_project(self, project, audit=None) -> None:
        self.project = project
        self._audit = audit
        save_dir = os.path.join(project.get_project_dir(), "models") if project.get_project_dir() else "models"
        self.save_dir_label.setText(save_dir)
        self._load_config()

    def set_settings(self, settings) -> None:
        self._settings = settings
        self._refresh_ssh_profiles()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        splitter.addWidget(self._build_config_panel())
        splitter.addWidget(self._build_progress_panel())
        splitter.setSizes([370, 630])

    def _build_config_panel(self) -> QGroupBox:
        box = QGroupBox("Trainingsparameter")
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.model_combo = QComboBox()
        self.model_combo.addItems(get_available_models())
        form.addRow("Architektur:", self.model_combo)

        self.pretrained_cb = QCheckBox("Vortrainierte Gewichte (ImageNet)")
        self.pretrained_cb.setChecked(True)
        form.addRow("", self.pretrained_cb)

        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(32, 1024)
        self.img_size_spin.setValue(224)
        self.img_size_spin.setSingleStep(32)
        form.addRow("Bildgröße (px):", self.img_size_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(16)
        form.addRow("Batch-Größe:", self.batch_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(20)
        form.addRow("Epochen:", self.epochs_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-7, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(7)
        self.lr_spin.setSingleStep(0.0001)
        form.addRow("Learning Rate:", self.lr_spin)

        self.opt_combo = QComboBox()
        self.opt_combo.addItems(["adam", "adamw", "sgd"])
        form.addRow("Optimizer:", self.opt_combo)

        self.sched_combo = QComboBox()
        self.sched_combo.addItems(["reduce_on_plateau", "cosine", "step"])
        form.addRow("LR-Scheduler:", self.sched_combo)

        self.early_stop_spin = QSpinBox()
        self.early_stop_spin.setRange(0, 100)
        self.early_stop_spin.setValue(0)
        self.early_stop_spin.setToolTip("0 = deaktiviert")
        form.addRow("Early Stopping (Geduld):", self.early_stop_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        form.addRow("Seed:", self.seed_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda", "mps"])
        form.addRow("Gerät:", self.device_combo)

        self.amp_cb = QCheckBox("Mixed Precision (AMP, nur CUDA)")
        form.addRow("", self.amp_cb)

        # Split
        split_box = QGroupBox("Daten-Split")
        sf = QFormLayout(split_box)
        self.train_split = QDoubleSpinBox()
        self.train_split.setRange(0.1, 0.9)
        self.train_split.setValue(0.7)
        self.train_split.setSingleStep(0.05)
        sf.addRow("Train:", self.train_split)
        self.val_split = QDoubleSpinBox()
        self.val_split.setRange(0.05, 0.5)
        self.val_split.setValue(0.2)
        self.val_split.setSingleStep(0.05)
        sf.addRow("Validation:", self.val_split)
        form.addRow(split_box)

        # Augmentation
        aug_box = QGroupBox("Augmentation")
        ab = QVBoxLayout(aug_box)
        self.aug_flip = QCheckBox("Flip (horizontal + vertikal)")
        self.aug_flip.setChecked(True)
        self.aug_rotation = QCheckBox("Rotation (±15°)")
        self.aug_rotation.setChecked(True)
        self.aug_brightness = QCheckBox("Helligkeit / Kontrast")
        self.aug_brightness.setChecked(True)
        self.aug_scale = QCheckBox("Skalierung (Random Crop)")
        for cb in [self.aug_flip, self.aug_rotation, self.aug_brightness, self.aug_scale]:
            ab.addWidget(cb)
        aug_preview_btn = QPushButton("🔍 Vorschau anzeigen…")
        aug_preview_btn.setToolTip(
            "Zeigt, wie die gewählten Augmentierungen auf Bilder aus dem Projekt wirken."
        )
        aug_preview_btn.setStyleSheet(
            "background:#6C3483; color:white; padding:4px; border-radius:3px;"
        )
        aug_preview_btn.clicked.connect(self._show_aug_preview)
        ab.addWidget(aug_preview_btn)
        form.addRow(aug_box)

        self.use_rois_cb = QCheckBox("ROI-Bereiche verwenden")
        self.use_rois_cb.setChecked(True)
        form.addRow("", self.use_rois_cb)

        self.class_balance_cb = QCheckBox("Klassenausgleich (WeightedSampler)")
        self.class_balance_cb.setToolTip(
            "Gleicht ungleichmäßige Klassenverteilungen aus, indem unterrepräsentierte "
            "Klassen häufiger gesampelt werden."
        )
        form.addRow("", self.class_balance_cb)

        self.save_dir_label = QLabel("(Projekt öffnen)")
        self.save_dir_label.setWordWrap(True)
        form.addRow("Speicherort:", self.save_dir_label)

        # Resume
        self.resume_cb = QCheckBox("Training fortsetzen (Resume)")
        form.addRow("", self.resume_cb)
        resume_btn = QPushButton("Checkpoint wählen…")
        resume_btn.clicked.connect(self._pick_checkpoint)
        form.addRow(resume_btn)
        self.resume_path_label = QLabel("")
        self.resume_path_label.setWordWrap(True)
        form.addRow(self.resume_path_label)

        # SSH remote training
        ssh_box = QGroupBox("Ferntraining per SSH")
        ssh_f = QFormLayout(ssh_box)

        self.ssh_enabled_cb = QCheckBox("SSH-Ferntraining aktivieren")
        self.ssh_enabled_cb.stateChanged.connect(self._on_ssh_toggled)
        ssh_f.addRow("", self.ssh_enabled_cb)

        self.ssh_profile_combo = QComboBox()
        self.ssh_profile_combo.setEnabled(False)
        ssh_f.addRow("Profil:", self.ssh_profile_combo)

        self.ssh_python_edit = QLineEdit("python3")
        self.ssh_python_edit.setEnabled(False)
        self.ssh_python_edit.setToolTip("Python-Interpreter auf dem Server (z.B. python3 oder /opt/venv/bin/python)")
        ssh_f.addRow("Python:", self.ssh_python_edit)

        self.ssh_remote_path_edit = QLineEdit("/tmp/ils_project")
        self.ssh_remote_path_edit.setEnabled(False)
        self.ssh_remote_path_edit.setToolTip("Basis-Arbeitsverzeichnis auf dem Remote-Server")
        ssh_f.addRow("Remote-Pfad:", self.ssh_remote_path_edit)

        self.ssh_test_btn = QPushButton("Verbindung testen")
        self.ssh_test_btn.setEnabled(False)
        self.ssh_test_btn.clicked.connect(self._test_ssh)
        ssh_f.addRow(self.ssh_test_btn)

        self.ssh_status_lbl = QLabel("")
        self.ssh_status_lbl.setWordWrap(True)
        ssh_f.addRow(self.ssh_status_lbl)

        form.addRow(ssh_box)

        # Buttons
        self.start_btn = QPushButton("Training starten")
        self.start_btn.setStyleSheet("background:#2ECC71;color:white;font-weight:bold;padding:8px;")
        self.start_btn.clicked.connect(self._start)
        form.addRow(self.start_btn)

        self.stop_btn = QPushButton("Training stoppen")
        self.stop_btn.setStyleSheet("background:#E74C3C;color:white;padding:8px;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)
        form.addRow(self.stop_btn)

        return box

    def _build_progress_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        # Progress row
        pr = QHBoxLayout()
        self.epoch_label = QLabel("Epoche: – / –")
        pr.addWidget(self.epoch_label)
        self.progress_bar = QProgressBar()
        pr.addWidget(self.progress_bar)
        v.addLayout(pr)

        # Live metrics
        mr = QHBoxLayout()
        self.train_loss_lbl = QLabel("Train-Loss: –")
        self.val_loss_lbl   = QLabel("Val-Loss: –")
        self.train_acc_lbl  = QLabel("Train-Acc: –")
        self.val_acc_lbl    = QLabel("Val-Acc: –")
        for lbl in [self.train_loss_lbl, self.val_loss_lbl, self.train_acc_lbl, self.val_acc_lbl]:
            lbl.setStyleSheet(
                "font-weight:bold;padding:4px 8px;"
                "background:#1565C0;color:white;border-radius:4px;"
            )
            mr.addWidget(lbl)
        v.addLayout(mr)

        # Tabs
        self.tabs = QTabWidget()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier New", 9))
        self.tabs.addTab(self.log_text, "Log")

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.tabs.addTab(self.metrics_text, "Val-Metriken")

        from gui.widgets.charts import ConfusionMatrixWidget, TrainingCurvesWidget
        self.cm_widget = ConfusionMatrixWidget()
        self.cm_widget.cell_clicked.connect(self._on_cm_cell_clicked)
        self.tabs.addTab(self.cm_widget, "Val-Konfusionsmatrix")

        self.curves_widget = TrainingCurvesWidget()
        self.tabs.addTab(self.curves_widget, "Trainingskurven")

        # ── Test-Ergebnisse tab ──────────────────────────────────────────────
        self.tabs.addTab(self._build_test_tab(), "🔬 Test-Ergebnisse")

        v.addWidget(self.tabs)

        # Export button
        export_btn = QPushButton("HTML-Bericht erstellen…")
        export_btn.clicked.connect(self._export_report)
        v.addWidget(export_btn)
        excel_btn = QPushButton("Excel-Bericht erstellen…")
        excel_btn.clicked.connect(self._export_excel)
        v.addWidget(excel_btn)
        return w

    def _build_test_tab(self) -> QWidget:
        """Dedicated widget for held-out test-set evaluation results."""
        from PySide6.QtWidgets import QSplitter
        from gui.widgets.charts import ConfusionMatrixWidget

        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)

        # Info banner
        self._test_banner = QLabel(
            "Noch kein Training abgeschlossen.\n"
            "Nach dem Training wird hier die Evaluation auf dem "
            "gehaltenen Test-Set angezeigt (Bilder, die das Modell nie gesehen hat)."
        )
        self._test_banner.setWordWrap(True)
        self._test_banner.setStyleSheet(
            "background:#1A3A5C; color:#AED6F1; padding:10px; "
            "border-radius:5px; font-size:11px;"
        )
        v.addWidget(self._test_banner)

        splitter = QSplitter(Qt.Vertical)

        self._test_metrics_text = QTextEdit()
        self._test_metrics_text.setReadOnly(True)
        self._test_metrics_text.setFont(QFont("Courier New", 9))
        splitter.addWidget(self._test_metrics_text)

        self._test_cm_widget = ConfusionMatrixWidget()
        self._test_cm_widget.cell_clicked.connect(self._on_cm_cell_clicked)
        splitter.addWidget(self._test_cm_widget)

        splitter.setSizes([280, 300])
        v.addWidget(splitter)
        return w

    # ------------------------------------------------------------------ augmentation preview

    def _show_aug_preview(self) -> None:
        aug_cfg = {
            "flip":       self.aug_flip.isChecked(),
            "rotation":   self.aug_rotation.isChecked(),
            "brightness": self.aug_brightness.isChecked(),
            "contrast":   self.aug_brightness.isChecked(),
            "scale":      self.aug_scale.isChecked(),
        }
        from gui.augmentation_preview_dialog import AugmentationPreviewDialog
        dlg = AugmentationPreviewDialog(
            project=self.project,
            aug_cfg=aug_cfg,
            image_size=self.img_size_spin.value(),
            parent=self,
        )
        dlg.exec()

    # ------------------------------------------------------------------ config

    def _get_config(self) -> Dict:
        return {
            "model_type": self.model_combo.currentText(),
            "use_pretrained": self.pretrained_cb.isChecked(),
            "image_size": self.img_size_spin.value(),
            "batch_size": self.batch_spin.value(),
            "epochs": self.epochs_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "optimizer": self.opt_combo.currentText(),
            "scheduler": self.sched_combo.currentText(),
            "early_stopping_patience": self.early_stop_spin.value(),
            "seed": self.seed_spin.value(),
            "device": self.device_combo.currentText(),
            "mixed_precision": self.amp_cb.isChecked(),
            "train_split": self.train_split.value(),
            "val_split": self.val_split.value(),
            "test_split": max(0.0, 1.0 - self.train_split.value() - self.val_split.value()),
            "use_rois": self.use_rois_cb.isChecked(),
            "class_balance": self.class_balance_cb.isChecked(),
            "augmentation": {
                "flip": self.aug_flip.isChecked(),
                "rotation": self.aug_rotation.isChecked(),
                "brightness": self.aug_brightness.isChecked(),
                "contrast": self.aug_brightness.isChecked(),
                "scale": self.aug_scale.isChecked(),
            },
            "resume_checkpoint": self.resume_path_label.text() if self.resume_cb.isChecked() else "",
            "ssh_enabled": self.ssh_enabled_cb.isChecked(),
        }

    def _load_config(self) -> None:
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

    def _pick_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Checkpoint wählen", "", "PyTorch (*.pth)")
        if path:
            self.resume_path_label.setText(path)
            self.resume_cb.setChecked(True)

    # ------------------------------------------------------------------ training

    def _start(self) -> None:
        if not self.project:
            QMessageBox.warning(self, "Kein Projekt", "Bitte zuerst ein Projekt öffnen.")
            return
        if len(self.project.labels) < 2:
            QMessageBox.warning(self, "Zu wenig Labels", "Mindestens 2 Labels erforderlich.")
            return
        cfg = self._get_config()
        save_dir = os.path.join(self.project.get_project_dir() or ".", "models")
        self.project.training_config = cfg
        self.log_text.clear()
        self.metrics_text.clear()
        self._history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}
        self.progress_bar.setValue(0)
        if self._audit:
            self._audit.log_training_started(cfg.get("seed", 42), cfg)

        if self.ssh_enabled_cb.isChecked():
            ssh_cfg = self._current_ssh_cfg()
            if ssh_cfg is None:
                QMessageBox.warning(self, "SSH-Fehler", "Kein SSH-Profil ausgewählt.")
                return
            from core.remote_training import RemoteTrainingThread
            self._thread = RemoteTrainingThread(self.project, cfg, save_dir, ssh_cfg)
        else:
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

    @Slot(int, int, float, float, float, float)
    def _on_progress(self, epoch, total, tl, vl, ta, va) -> None:
        self.progress_bar.setValue(int(epoch / total * 100))
        self.epoch_label.setText(f"Epoche: {epoch} / {total}")
        self.train_loss_lbl.setText(f"Train-Loss: {tl:.4f}")
        self.val_loss_lbl.setText(f"Val-Loss: {vl:.4f}")
        self.train_acc_lbl.setText(f"Train-Acc: {ta*100:.1f}%")
        self.val_acc_lbl.setText(f"Val-Acc: {va*100:.1f}%")
        for k, v in [("train_loss", tl), ("val_loss", vl), ("train_acc", ta), ("val_acc", va)]:
            self._history[k].append(v)
        self.curves_widget.update_curves(self._history)

    @Slot(str)
    def _on_log(self, msg: str) -> None:
        self.log_text.append(msg)

    @Slot(dict)
    def _on_finished(self, result: Dict) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)

        class_names   = result.get("class_names", [])
        self._last_class_names = class_names
        test_metrics  = result.get("test_metrics", result.get("metrics", {}))
        bvm           = result.get("best_val_metrics", {})

        # ── Val-Metriken tab: show best-epoch val summary + val metrics ──────
        val_header = []
        if bvm:
            val_header = [
                f"Bestes Modell bei Epoche {bvm.get('epoch','?')}:",
                f"  Val-Accuracy:  {bvm.get('val_acc', 0)*100:.2f}%",
                f"  Val-Loss:      {bvm.get('val_loss', 0):.4f}",
                f"  Train-Accuracy:{bvm.get('train_acc', 0)*100:.2f}%",
                "",
                "─" * 46,
                "Test-Set Metriken (bestes Modell, nie gesehen):",
                "",
            ]
        from core.metrics import format_metrics_text
        val_body = "\n".join(val_header) + format_metrics_text(test_metrics)
        self.metrics_text.setPlainText(val_body)

        # ── Val-Konfusionsmatrix: validation confusion matrix (history-based) ─
        self.cm_widget.set_matrix(
            test_metrics.get("confusion_matrix", []), class_names
        )

        # ── Test-Ergebnisse tab ───────────────────────────────────────────────
        train_n = result.get("train_size", "?")
        val_n   = result.get("val_size",   "?")
        test_n  = result.get("test_size",  "?")
        total   = (train_n + val_n + test_n) if all(
            isinstance(x, int) for x in [train_n, val_n, test_n]) else "?"

        self._test_banner.setText(
            f"Test-Set: {test_n} Bilder  |  "
            f"Train: {train_n}  Val: {val_n}  "
            f"(Gesamt: {total})\n"
            "Evaluation auf dem besten Checkpoint — diese Bilder wurden "
            "während des Trainings nie verwendet."
        )
        self._test_banner.setStyleSheet(
            "background:#1A3A2A; color:#58D68D; padding:10px; "
            "border-radius:5px; font-size:11px; font-weight:bold;"
        )

        test_lines = [
            "╔══════════════════════════════════════════════╗",
            f"║  Test-Accuracy:  {test_metrics.get('accuracy',0)*100:6.2f}%                     ║",
            f"║  F1 (Macro):     {test_metrics.get('macro_f1',0)*100:6.2f}%                     ║",
            f"║  F1 (Weighted):  {test_metrics.get('weighted_f1',0)*100:6.2f}%                     ║",
            f"║  Precision:      {test_metrics.get('macro_precision',0)*100:6.2f}%                     ║",
            f"║  Recall:         {test_metrics.get('macro_recall',0)*100:6.2f}%                     ║",
            "╚══════════════════════════════════════════════╝",
            "",
        ]
        if bvm:
            test_lines += [
                f"Vergleich  →  Val-Acc (best): {bvm.get('val_acc',0)*100:.2f}%  "
                f"vs.  Test-Acc: {test_metrics.get('accuracy',0)*100:.2f}%",
                "",
            ]
        test_lines += ["Klassen-Detail:", ""]
        for cls, vals in test_metrics.get("per_class", {}).items():
            test_lines.append(
                f"  {cls:<20}  P={vals['precision']*100:.1f}%  "
                f"R={vals['recall']*100:.1f}%  F1={vals['f1']*100:.1f}%  "
                f"n={vals['support']}"
            )
        if "top3_accuracy" in test_metrics:
            test_lines += ["", f"Top-3-Accuracy: {test_metrics['top3_accuracy']*100:.2f}%"]
        if "roc_auc" in test_metrics:
            test_lines += [f"ROC-AUC:        {test_metrics['roc_auc']:.4f}"]

        self._test_metrics_text.setPlainText("\n".join(test_lines))
        self._test_cm_widget.set_matrix(
            test_metrics.get("confusion_matrix", []), class_names
        )

        self._last_result = result
        self._test_predictions = result.get("test_predictions", [])
        if self._audit:
            self._audit.log_training_finished(result.get("run_id", ""), test_metrics)

        # Jump straight to Test-Ergebnisse tab
        self.tabs.setCurrentIndex(4)
        self.training_finished.emit(result)
        QMessageBox.information(
            self, "Training abgeschlossen",
            f"Test-Accuracy:  {test_metrics.get('accuracy',0)*100:.2f}%\n"
            f"Test-F1 (macro):{test_metrics.get('macro_f1',0)*100:.2f}%\n\n"
            f"Bestes Modell: {result.get('best_model_path', '')}"
        )

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_text.append(f"FEHLER: {msg}")
        QMessageBox.critical(self, "Trainingsfehler", msg)

    # ------------------------------------------------------------------ SSH helpers

    def _refresh_ssh_profiles(self) -> None:
        if not self._settings:
            return
        self._ssh_profiles = self._settings.get_ssh_profiles()
        self.ssh_profile_combo.clear()
        for p in self._ssh_profiles:
            name = p.get("name", "?")
            host = p.get("host", "?")
            self.ssh_profile_combo.addItem(f"{name}  —  {host}")

    def _on_ssh_toggled(self, state: int) -> None:
        enabled = bool(state)
        self.ssh_profile_combo.setEnabled(enabled)
        self.ssh_python_edit.setEnabled(enabled)
        self.ssh_remote_path_edit.setEnabled(enabled)
        self.ssh_test_btn.setEnabled(enabled)
        if enabled:
            self._refresh_ssh_profiles()

    def _current_ssh_cfg(self) -> Optional[Dict]:
        idx = self.ssh_profile_combo.currentIndex()
        if idx < 0 or idx >= len(self._ssh_profiles):
            return None
        profile = dict(self._ssh_profiles[idx])
        profile["python_env"] = self.ssh_python_edit.text().strip() or "python3"
        profile["remote_path"] = self.ssh_remote_path_edit.text().strip() or "/tmp/ils_project"
        return profile

    def _test_ssh(self) -> None:
        cfg = self._current_ssh_cfg()
        if cfg is None:
            QMessageBox.warning(self, "SSH", "Kein Profil ausgewählt.")
            return
        self.ssh_status_lbl.setText("Verbinde …")
        self.ssh_test_btn.setEnabled(False)

        class _TestThread(QThread):
            done = Signal(bool, str)
            def __init__(self, cfg):
                super().__init__()
                self._cfg = cfg
            def run(self):
                from core.remote_ssh import SSHManager
                ok, msg = SSHManager().test_connection(self._cfg)
                self.done.emit(ok, msg)

        t = _TestThread(cfg)
        t.done.connect(self._on_ssh_test_done)
        t.start()
        self._ssh_test_thread = t  # keep reference

    @Slot(bool, str)
    def _on_ssh_test_done(self, ok: bool, msg: str) -> None:
        self.ssh_test_btn.setEnabled(True)
        if ok:
            self.ssh_status_lbl.setStyleSheet("color: #2ECC71;")
            self.ssh_status_lbl.setText(f"✓ {msg}")
        else:
            self.ssh_status_lbl.setStyleSheet("color: #E74C3C;")
            self.ssh_status_lbl.setText(f"✗ {msg}")

    def _on_cm_cell_clicked(self, true_idx: int, pred_idx: int) -> None:
        """Open misclassified-images dialog for the clicked confusion matrix cell."""
        if not self._test_predictions or not self._last_class_names:
            QMessageBox.information(
                self, "Keine Daten",
                "Bitte erst ein Training mit Test-Set abschließen."
            )
            return

        true_label = self._last_class_names[true_idx]
        pred_label = self._last_class_names[pred_idx]

        samples = [
            p for p in self._test_predictions
            if p["true_label"] == true_label and p["pred_label"] == pred_label
        ]

        if not samples:
            QMessageBox.information(
                self, "Keine Bilder",
                f"Keine Bilder für Zelle ({true_label} → {pred_label}) gefunden."
            )
            return

        from gui.misclassified_dialog import MisclassifiedDialog
        dlg = MisclassifiedDialog(samples, true_label, pred_label, parent=self)
        dlg.exec()

    def _export_report(self) -> None:
        if not hasattr(self, "_last_result"):
            QMessageBox.information(self, "Kein Ergebnis", "Erst Training durchführen.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "HTML-Bericht speichern", "training_report.html", "HTML (*.html)"
        )
        if not path:
            return
        try:
            from core.report import generate_html_report
            generate_html_report(
                self._last_result, path,
                project_name=self.project.config.name if self.project else ""
            )
            QMessageBox.information(self, "Exportiert", f"Bericht gespeichert:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Fehler", str(exc))

    def _export_excel(self) -> None:
        if not hasattr(self, "_last_result"):
            QMessageBox.information(self, "Kein Ergebnis", "Erst Training durchführen.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Excel-Bericht speichern", "training_report.xlsx", "Excel (*.xlsx)"
        )
        if not path:
            return
        try:
            from core.export import export_training_report
            export_training_report(self._last_result, path)
            QMessageBox.information(self, "Exportiert", f"Bericht gespeichert:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Fehler", str(exc))
