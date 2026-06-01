"""
Training page: config, live progress, metrics, curves, confusion matrix.
"""
import json
import logging
import os
import time
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QProgressBar, QTextEdit, QSplitter, QTabWidget,
    QMessageBox, QFileDialog, QLineEdit,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont, QKeySequence, QShortcut

from models.classifier import get_available_models
from utils.config import DEFAULT_TRAIN_CONFIG
from core.metrics import format_metrics_text
from utils.i18n import tr


class TrainingThread(QThread):
    """
    QThread wrapper around ``core.training.TrainingWorker``.

    Bridges the worker's callback-based interface to Qt signals so that
    ``TrainingPage`` can update the UI from the main thread.

    Signals
    -------
    progress : (epoch, total, train_loss, val_loss, train_acc, val_acc)
    log_msg  : Informational log line from the worker.
    finished : Full result dict returned by ``TrainingWorker.run()``.
    error    : Exception message when the worker raises.
    """

    progress = Signal(int, int, float, float, float, float)
    log_msg  = Signal(str)
    finished = Signal(dict)
    error    = Signal(str)

    def __init__(self, project, cfg: Dict, save_dir: str):
        """
        Parameters
        ----------
        project  : Loaded ``Project`` instance.
        cfg      : Training configuration dict (see ``TrainingPage._get_config()``).
        save_dir : Directory where model checkpoints will be written.
        """
        super().__init__()
        self.project = project
        self.cfg = cfg
        self.save_dir = save_dir
        self._stop = False

    def request_stop(self) -> None:
        """Signal the underlying worker to stop after the current epoch."""
        self._stop = True

    def run(self) -> None:
        """Instantiate ``TrainingWorker`` and run it; emit ``finished`` or ``error``."""
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
    """
    Training configuration and live-monitoring page (stack index 3).

    Left panel: all hyperparameters (architecture, image size, batch size, epochs,
    learning rate, optimizer, scheduler, augmentation, SSH remote training).
    Right panel: live progress bar, loss/accuracy badges, training-curves chart,
    validation metrics and confusion matrix, and a dedicated test-results tab.

    Emits ``training_finished(dict)`` when the worker completes; ``MainWindow``
    persists the result and refreshes the Models and Dashboard pages.
    """

    training_finished = Signal(dict)
    al_queue_updated  = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._thread: Optional[QThread] = None
        self._al_thread = None
        self._hpt_thread = None
        self._history: Dict = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}
        self._audit = None
        self._settings = None
        self._ssh_profiles: List[Dict] = []
        self._test_predictions: List[Dict] = []
        self._last_class_names: List[str] = []
        self._last_model_path: str = ""
        self._train_start_time = None
        self._build_ui()
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self._start)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._stop_training)

    def set_project(self, project, audit=None) -> None:
        """Accept a new project, update the model save directory, and reload saved config."""
        self.project = project
        self._audit = audit
        save_dir = os.path.join(project.get_project_dir(), "models") if project.get_project_dir() else "models"
        self.save_dir_label.setText(save_dir)
        self._save_cfg_btn.setEnabled(True)
        self._load_cfg_btn.setEnabled(True)
        self._load_config()
        self._load_config_file(silent=True)
        # Re-enable AL scan if the project already has a model
        model_path = getattr(project, "current_model_path", "")
        if model_path and os.path.isfile(model_path):
            self._last_model_path = model_path
            self._al_scan_btn.setEnabled(True)
            self._al_status.setText(
                f"Modell geladen. Ungelabelte Bilder: {len(project.get_unlabeled_images())}"
            )
        else:
            self._al_scan_btn.setEnabled(False)
            self._al_status.setText("")

    def set_settings(self, settings) -> None:
        """Inject the application settings and populate the SSH profile combo."""
        self._settings = settings
        self._refresh_ssh_profiles()

    def closeEvent(self, event) -> None:
        if self._thread and self._thread.isRunning():
            self._thread.request_stop()
            self._thread.quit()
            self._thread.wait(5000)
        if self._al_thread and self._al_thread.isRunning():
            self._al_thread.request_stop()
            self._al_thread.quit()
            self._al_thread.wait(3000)
        if self._hpt_thread and self._hpt_thread.isRunning():
            self._hpt_thread.quit()
            self._hpt_thread.wait(3000)
        super().closeEvent(event)

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        splitter.addWidget(self._build_config_panel())
        splitter.addWidget(self._build_progress_panel())
        splitter.setSizes([370, 630])

    def _build_config_panel(self) -> QGroupBox:
        box = QGroupBox(tr("training.config_group"))
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.model_combo = QComboBox()
        self.model_combo.addItems(get_available_models())
        self.model_combo.setToolTip(tr("training.tip_model"))
        form.addRow(tr("training.arch_label"), self.model_combo)

        self.pretrained_cb = QCheckBox(tr("training.pretrained_cb"))
        self.pretrained_cb.setChecked(True)
        self.pretrained_cb.setToolTip(tr("training.tip_pretrained"))
        form.addRow("", self.pretrained_cb)

        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(32, 1024)
        self.img_size_spin.setValue(224)
        self.img_size_spin.setSingleStep(32)
        self.img_size_spin.setToolTip(tr("training.tip_img_size"))
        form.addRow(tr("training.image_size_label"), self.img_size_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(16)
        self.batch_spin.setToolTip(tr("training.tip_batch"))
        form.addRow(tr("training.batch_size_label"), self.batch_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(20)
        self.epochs_spin.setToolTip(tr("training.tip_epochs"))
        form.addRow(tr("training.epochs_label"), self.epochs_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-7, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(7)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setToolTip(tr("training.tip_lr"))
        form.addRow(tr("training.lr_label"), self.lr_spin)

        self.opt_combo = QComboBox()
        self.opt_combo.addItems(["adam", "adamw", "sgd"])
        self.opt_combo.setToolTip(tr("training.tip_optimizer"))
        form.addRow(tr("training.optimizer_label"), self.opt_combo)

        self.sched_combo = QComboBox()
        self.sched_combo.addItems(["reduce_on_plateau", "cosine", "step"])
        self.sched_combo.setToolTip(tr("training.tip_scheduler"))
        form.addRow(tr("training.scheduler_label"), self.sched_combo)

        self.early_stop_spin = QSpinBox()
        self.early_stop_spin.setRange(0, 100)
        self.early_stop_spin.setValue(0)
        self.early_stop_spin.setToolTip(tr("training.tip_early_stop"))
        form.addRow(tr("training.early_stop_label"), self.early_stop_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        self.seed_spin.setToolTip(tr("training.tip_seed"))
        form.addRow(tr("training.seed_label"), self.seed_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda", "mps"])
        self.device_combo.setToolTip(tr("training.tip_device"))
        form.addRow(tr("training.device_label"), self.device_combo)

        self.amp_cb = QCheckBox(tr("training.amp_cb"))
        self.amp_cb.setToolTip(tr("training.tip_amp"))
        form.addRow("", self.amp_cb)

        # Split
        split_box = QGroupBox(tr("training.split_group"))
        sf = QFormLayout(split_box)
        self.train_split = QDoubleSpinBox()
        self.train_split.setRange(0.1, 0.9)
        self.train_split.setValue(0.7)
        self.train_split.setSingleStep(0.05)
        self.train_split.setToolTip(tr("training.tip_train_split"))
        sf.addRow(tr("training.train_split_label"), self.train_split)
        self.val_split = QDoubleSpinBox()
        self.val_split.setRange(0.05, 0.5)
        self.val_split.setValue(0.2)
        self.val_split.setSingleStep(0.05)
        self.val_split.setToolTip(tr("training.tip_val_split"))
        sf.addRow(tr("training.val_split_label"), self.val_split)
        form.addRow(split_box)

        # Augmentation
        aug_box = QGroupBox(tr("training.aug_group"))
        ab = QVBoxLayout(aug_box)
        self.aug_flip = QCheckBox(tr("training.aug_flip_cb"))
        self.aug_flip.setChecked(True)
        self.aug_flip.setToolTip(tr("training.tip_aug_flip"))
        self.aug_rotation = QCheckBox(tr("training.aug_rotation_cb"))
        self.aug_rotation.setChecked(True)
        self.aug_rotation.setToolTip(tr("training.tip_aug_rotation"))
        self.aug_brightness = QCheckBox(tr("training.aug_brightness_cb"))
        self.aug_brightness.setChecked(True)
        self.aug_brightness.setToolTip(tr("training.tip_aug_brightness"))
        self.aug_scale = QCheckBox(tr("training.aug_scale_cb"))
        self.aug_scale.setToolTip(tr("training.tip_aug_scale"))
        self.aug_blur = QCheckBox(tr("training.aug_blur_cb"))
        self.aug_blur.setToolTip(tr("training.tip_aug_blur"))
        for cb in [self.aug_flip, self.aug_rotation, self.aug_brightness,
                   self.aug_scale, self.aug_blur]:
            ab.addWidget(cb)
        aug_preview_btn = QPushButton(tr("training.aug_preview_btn"))
        aug_preview_btn.setToolTip(tr("training.tip_aug_preview"))
        aug_preview_btn.setStyleSheet(
            "background:#6C3483; color:white; padding:4px; border-radius:3px;"
        )
        aug_preview_btn.clicked.connect(self._show_aug_preview)
        ab.addWidget(aug_preview_btn)
        form.addRow(aug_box)

        self.use_rois_cb = QCheckBox(tr("training.use_rois_cb"))
        self.use_rois_cb.setChecked(True)
        self.use_rois_cb.setToolTip(tr("training.tip_use_rois"))
        form.addRow("", self.use_rois_cb)

        self.class_balance_cb = QCheckBox(tr("training.class_balance_cb"))
        self.class_balance_cb.setToolTip(tr("training.tip_class_balance"))
        form.addRow("", self.class_balance_cb)

        self.focal_loss_cb = QCheckBox(tr("training.focal_loss_cb"))
        self.focal_loss_cb.setToolTip(tr("training.tip_focal_loss"))
        self.focal_loss_cb.toggled.connect(self._on_focal_toggled)
        form.addRow("", self.focal_loss_cb)

        from PySide6.QtWidgets import QDoubleSpinBox as _DSB, QHBoxLayout as _HBox
        focal_row = _HBox()
        self._focal_gamma_label = QLabel(tr("training.focal_gamma_label"))
        self._focal_gamma_label.setEnabled(False)
        focal_row.addWidget(self._focal_gamma_label)
        self.focal_gamma_spin = QDoubleSpinBox()
        self.focal_gamma_spin.setRange(0.5, 5.0)
        self.focal_gamma_spin.setValue(2.0)
        self.focal_gamma_spin.setSingleStep(0.5)
        self.focal_gamma_spin.setDecimals(1)
        self.focal_gamma_spin.setEnabled(False)
        self.focal_gamma_spin.setToolTip(tr("training.tip_focal_gamma"))
        focal_row.addWidget(self.focal_gamma_spin)
        focal_row.addStretch()
        form.addRow("", focal_row)

        self.save_dir_label = QLabel("(Projekt öffnen)")
        self.save_dir_label.setWordWrap(True)
        form.addRow("Speicherort:", self.save_dir_label)

        # Resume
        self.resume_cb = QCheckBox(tr("training.resume_cb"))
        self.resume_cb.setToolTip(tr("training.tip_resume"))
        form.addRow("", self.resume_cb)
        resume_btn = QPushButton(tr("training.checkpoint_btn"))
        resume_btn.setToolTip(tr("training.resume_btn_tip"))
        resume_btn.clicked.connect(self._pick_checkpoint)
        form.addRow(resume_btn)
        self.resume_path_label = QLabel("")
        self.resume_path_label.setWordWrap(True)
        form.addRow(self.resume_path_label)

        # SSH remote training
        ssh_box = QGroupBox(tr("training.ssh_group"))
        ssh_f = QFormLayout(ssh_box)

        self.ssh_enabled_cb = QCheckBox(tr("training.ssh_enabled_cb"))
        self.ssh_enabled_cb.stateChanged.connect(self._on_ssh_toggled)
        ssh_f.addRow("", self.ssh_enabled_cb)

        self.ssh_profile_combo = QComboBox()
        self.ssh_profile_combo.setEnabled(False)
        ssh_f.addRow(tr("training.ssh_profile_label"), self.ssh_profile_combo)

        self.ssh_python_edit = QLineEdit("python3")
        self.ssh_python_edit.setEnabled(False)
        self.ssh_python_edit.setToolTip(tr("training.ssh_python_tip"))
        ssh_f.addRow(tr("training.ssh_python_label"), self.ssh_python_edit)

        self.ssh_remote_path_edit = QLineEdit("/tmp/ils_project")
        self.ssh_remote_path_edit.setEnabled(False)
        self.ssh_remote_path_edit.setToolTip(tr("training.ssh_path_tip"))
        ssh_f.addRow(tr("training.ssh_remote_path_label"), self.ssh_remote_path_edit)

        self.ssh_test_btn = QPushButton(tr("training.ssh_test_btn"))
        self.ssh_test_btn.setEnabled(False)
        self.ssh_test_btn.clicked.connect(self._test_ssh)
        ssh_f.addRow(self.ssh_test_btn)

        self.ssh_status_lbl = QLabel("")
        self.ssh_status_lbl.setWordWrap(True)
        ssh_f.addRow(self.ssh_status_lbl)

        form.addRow(ssh_box)

        # Buttons — Reihenfolge: HPT → Start → Stop
        self.hpt_btn = QPushButton(tr("training.hpt_btn"))
        self.hpt_btn.setStyleSheet(
            "background:#6C3483;color:white;padding:6px;border-radius:3px;"
        )
        self.hpt_btn.setToolTip(tr("training.tip_hpt_btn"))
        self.hpt_btn.clicked.connect(self._start_hpt)
        form.addRow(self.hpt_btn)

        self.start_btn = QPushButton(tr("training.start_btn"))
        self.start_btn.setStyleSheet("background:#2ECC71;color:white;font-weight:bold;padding:8px;")
        self.start_btn.setToolTip(tr("training.start_btn_tip"))
        self.start_btn.clicked.connect(self._start)
        form.addRow(self.start_btn)

        self.stop_btn = QPushButton(tr("training.stop_btn"))
        self.stop_btn.setStyleSheet("background:#E74C3C;color:white;padding:8px;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip(tr("training.stop_btn_tip"))
        self.stop_btn.clicked.connect(self._stop_training)
        form.addRow(self.stop_btn)

        # Config save / load
        cfg_row = QHBoxLayout()
        self._save_cfg_btn = QPushButton(tr("training.save_config_btn"))
        self._save_cfg_btn.setEnabled(False)
        self._save_cfg_btn.clicked.connect(self._save_config)
        self._load_cfg_btn = QPushButton(tr("training.load_config_btn"))
        self._load_cfg_btn.setEnabled(False)
        self._load_cfg_btn.clicked.connect(self._load_config_file)
        cfg_row.addWidget(self._save_cfg_btn)
        cfg_row.addWidget(self._load_cfg_btn)
        form.addRow(cfg_row)

        return box

    def _build_progress_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        # Progress row
        pr = QHBoxLayout()
        self.epoch_label = QLabel(tr("training.epoch_init"))
        pr.addWidget(self.epoch_label)
        self._eta_lbl = QLabel(tr("training.eta_init"))
        pr.addWidget(self._eta_lbl)
        self.progress_bar = QProgressBar()
        pr.addWidget(self.progress_bar)
        v.addLayout(pr)

        # Live metrics
        mr = QHBoxLayout()
        self.train_loss_lbl = QLabel(tr("training.metric_train_loss_init"))
        self.val_loss_lbl   = QLabel(tr("training.metric_val_loss_init"))
        self.train_acc_lbl  = QLabel(tr("training.metric_train_acc_init"))
        self.val_acc_lbl    = QLabel(tr("training.metric_val_acc_init"))
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
        self.tabs.addTab(self.log_text, tr("training.tab.log"))

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.tabs.addTab(self.metrics_text, tr("training.tab.metrics"))

        from gui.widgets.charts import ConfusionMatrixWidget, TrainingCurvesWidget
        self.cm_widget = ConfusionMatrixWidget()
        self.cm_widget.cell_clicked.connect(self._on_cm_cell_clicked)
        self.tabs.addTab(self.cm_widget, tr("training.tab.confusion"))

        self.curves_widget = TrainingCurvesWidget()
        self.tabs.addTab(self.curves_widget, tr("training.tab.curves"))

        # ── Test-Ergebnisse tab ──────────────────────────────────────────────
        self.tabs.addTab(self._build_test_tab(), tr("training.tab.test"))

        # ── Active Learning tab ──────────────────────────────────────────────
        self.tabs.addTab(self._build_al_tab(), tr("training.tab.al"))

        v.addWidget(self.tabs)

        # Export button
        export_btn = QPushButton(tr("training.export_html_btn"))
        export_btn.clicked.connect(self._export_report)
        v.addWidget(export_btn)
        excel_btn = QPushButton(tr("training.export_excel_btn"))
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

    def _build_al_tab(self) -> QWidget:
        """Active Learning scan: find unlabeled images where the model is most uncertain."""
        from PySide6.QtWidgets import QFormLayout
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        info = QLabel(
            "Nach dem Training die ungelabelten Bilder des Projekts scannen und die\n"
            "unsichersten Vorhersagen in die AL-Queue (Labeling-Seite) eintragen."
        )
        info.setWordWrap(True)
        info.setStyleSheet(
            "background:#1A3A5C; color:#AED6F1; padding:10px; "
            "border-radius:5px; font-size:11px;"
        )
        v.addWidget(info)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self._al_thr_spin = QDoubleSpinBox()
        self._al_thr_spin.setRange(0.30, 0.99)
        self._al_thr_spin.setSingleStep(0.05)
        self._al_thr_spin.setValue(0.70)
        self._al_thr_spin.setToolTip(tr("training.tip_al_thr"))
        form.addRow(tr("training.al_threshold_label"), self._al_thr_spin)

        self._al_n_spin = QSpinBox()
        self._al_n_spin.setRange(5, 500)
        self._al_n_spin.setValue(50)
        self._al_n_spin.setToolTip(tr("training.tip_al_n"))
        form.addRow(tr("training.al_max_label"), self._al_n_spin)

        v.addLayout(form)

        self._al_scan_btn = QPushButton(tr("training.al_scan_btn"))
        self._al_scan_btn.setEnabled(False)
        self._al_scan_btn.setStyleSheet(
            "background:#1565C0; color:white; font-weight:bold; "
            "border-radius:4px; padding:5px 10px;"
        )
        self._al_scan_btn.clicked.connect(self._start_al_scan)
        v.addWidget(self._al_scan_btn)

        self._al_progress = QProgressBar()
        self._al_progress.setValue(0)
        self._al_progress.hide()
        v.addWidget(self._al_progress)

        self._al_status = QLabel("")
        self._al_status.setWordWrap(True)
        self._al_status.setStyleSheet("font-size:11px; color:#B0BEC5;")
        v.addWidget(self._al_status)

        v.addStretch()
        return w

    def _start_al_scan(self) -> None:
        """Run ActiveLearningThread on all unlabeled project images."""
        if not self.project or not self._last_model_path:
            return

        unlabeled = self.project.get_unlabeled_images()
        if not unlabeled:
            self._al_status.setText(tr("training.al_no_unlabeled"))
            return

        from core.active_learning import ActiveLearningThread

        self._al_scan_btn.setEnabled(False)
        self._al_progress.setValue(0)
        self._al_progress.show()
        self._al_status.setText(tr("training.al_scanning", n=len(unlabeled)))

        roi_template = None
        if self.project.rois:
            first_path = next(iter(self.project.rois))
            rois = self.project.rois[first_path]
            if rois:
                roi_template = rois[0]

        self._al_thread = ActiveLearningThread(
            model_path=self._last_model_path,
            image_paths=unlabeled,
            confidence_threshold=self._al_thr_spin.value(),
            n_samples=self._al_n_spin.value(),
            roi_template=roi_template,
            parent=self,
        )
        self._al_thread.progress.connect(
            lambda c, t: self._al_progress.setValue(int(c / t * 100))
        )
        self._al_thread.finished.connect(self._on_al_finished)
        self._al_thread.error.connect(self._on_al_error)
        self._al_thread.start()

    def _on_al_finished(self, candidates: list) -> None:
        self._al_thread = None
        self._al_progress.hide()
        self._al_scan_btn.setEnabled(True)

        if not candidates:
            self._al_status.setText(
                tr("training.al_no_uncertain") +
                f" (Schwellwert {self._al_thr_spin.value():.0%})."
            )
            return

        added = skipped = 0
        for r in candidates:
            ok = self.project.add_to_al_queue(
                r["path"], r["predicted_label"], r["confidence"]
            )
            if ok:
                added += 1
            else:
                skipped += 1

        parts = [tr("training.al_finished", added=added)]
        if skipped:
            parts.append(f"({skipped} bereits vorhanden)")
        parts.append(tr("training.al_open_labeling"))
        self._al_status.setText("  ".join(parts))

        if added > 0:
            self.al_queue_updated.emit()

    def _on_al_error(self, msg: str) -> None:
        self._al_thread = None
        self._al_progress.hide()
        self._al_scan_btn.setEnabled(True)
        self._al_status.setText(f"Fehler: {msg}")

    # ------------------------------------------------------------------ augmentation preview

    def _show_aug_preview(self) -> None:
        aug_cfg = self._get_config().get("augmentation", {})
        from gui.augmentation_preview_dialog import AugmentationPreviewDialog
        dlg = AugmentationPreviewDialog(
            project=self.project,
            aug_cfg=aug_cfg,
            image_size=self.img_size_spin.value(),
            parent=self,
        )
        dlg.config_accepted.connect(self._apply_aug_cfg)
        dlg.exec()

    def _apply_aug_cfg(self, cfg: dict) -> None:
        """Apply augmentation settings returned from the editor dialog and store extra params."""
        self.aug_flip.setChecked(cfg.get("flip", True))
        self.aug_rotation.setChecked(cfg.get("rotation", True))
        self.aug_brightness.setChecked(cfg.get("brightness", True))
        self.aug_scale.setChecked(cfg.get("scale", False))
        self.aug_blur.setChecked(cfg.get("blur", False))
        # Store intensity params for use in _get_config
        self._aug_extra = {
            "rotation_degrees":    cfg.get("rotation_degrees", 15),
            "brightness_strength": cfg.get("brightness_strength", 0.3),
            "scale_min":           cfg.get("scale_min", 0.8),
            "blur_radius":         cfg.get("blur_radius", 3),
        }

    # ------------------------------------------------------------------ config

    def _get_config(self) -> Dict:
        """Collect all form values into a training configuration dict."""
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
            "focal_loss": self.focal_loss_cb.isChecked(),
            "focal_gamma": self.focal_gamma_spin.value(),
            "augmentation": {
                "flip":                self.aug_flip.isChecked(),
                "rotation":            self.aug_rotation.isChecked(),
                "brightness":          self.aug_brightness.isChecked(),
                "contrast":            self.aug_brightness.isChecked(),
                "scale":               self.aug_scale.isChecked(),
                "blur":                self.aug_blur.isChecked(),
                **getattr(self, "_aug_extra", {}),
            },
            "resume_checkpoint": self.resume_path_label.text() if self.resume_cb.isChecked() else "",
            "ssh_enabled": self.ssh_enabled_cb.isChecked(),
        }

    def _load_config(self) -> None:
        """Restore form values from the project's previously saved training config."""
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

    def _config_file_path(self) -> str:
        """Return the path to the per-project training_config.json file."""
        base = self.project.get_project_dir() if self.project else None
        return os.path.join(base or ".", "training_config.json")

    def _save_config(self) -> None:
        """Save current hyperparameter form values to training_config.json."""
        if not self.project:
            return
        cfg = self._get_config()
        try:
            with open(self._config_file_path(), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            # Show status via the status bar if available, otherwise log
            parent = self.window()
            if hasattr(parent, "statusBar"):
                parent.statusBar().showMessage(tr("training.config_saved"), 3000)
        except Exception as exc:
            log.warning("Could not save training config: %s", exc)

    def _load_config_file(self, silent: bool = False) -> None:
        """Load hyperparameters from training_config.json into the form."""
        if not self.project:
            return
        path = self._config_file_path()
        if not os.path.isfile(path):
            if not silent:
                QMessageBox.information(
                    self, tr("training.load_config_btn"),
                    tr("training.config_not_found")
                )
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self._set_config(cfg)
            if not silent:
                parent = self.window()
                if hasattr(parent, "statusBar"):
                    parent.statusBar().showMessage(tr("training.config_loaded"), 3000)
        except Exception as exc:
            if not silent:
                QMessageBox.warning(self, tr("common.error"), str(exc))
            log.warning("Could not load training config: %s", exc)

    def _set_config(self, cfg: Dict) -> None:
        """Apply a config dict to all hyperparameter form controls."""
        if "model_type" in cfg:
            idx = self.model_combo.findText(cfg["model_type"])
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        if "use_pretrained" in cfg:
            self.pretrained_cb.setChecked(bool(cfg["use_pretrained"]))
        if "image_size" in cfg:
            self.img_size_spin.setValue(int(cfg["image_size"]))
        if "batch_size" in cfg:
            self.batch_spin.setValue(int(cfg["batch_size"]))
        if "epochs" in cfg:
            self.epochs_spin.setValue(int(cfg["epochs"]))
        if "learning_rate" in cfg:
            self.lr_spin.setValue(float(cfg["learning_rate"]))
        if "optimizer" in cfg:
            idx = self.opt_combo.findText(cfg["optimizer"])
            if idx >= 0:
                self.opt_combo.setCurrentIndex(idx)
        if "scheduler" in cfg:
            idx = self.sched_combo.findText(cfg["scheduler"])
            if idx >= 0:
                self.sched_combo.setCurrentIndex(idx)
        if "early_stopping_patience" in cfg:
            self.early_stop_spin.setValue(int(cfg["early_stopping_patience"]))
        if "seed" in cfg:
            self.seed_spin.setValue(int(cfg["seed"]))
        if "device" in cfg:
            idx = self.device_combo.findText(cfg["device"])
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)
        if "mixed_precision" in cfg:
            self.amp_cb.setChecked(bool(cfg["mixed_precision"]))
        if "train_split" in cfg:
            self.train_split.setValue(float(cfg["train_split"]))
        if "val_split" in cfg:
            self.val_split.setValue(float(cfg["val_split"]))
        aug = cfg.get("augmentation", {})
        if "flip" in aug:
            self.aug_flip.setChecked(bool(aug["flip"]))
        if "rotation" in aug:
            self.aug_rotation.setChecked(bool(aug["rotation"]))
        if "brightness" in aug:
            self.aug_brightness.setChecked(bool(aug["brightness"]))
        if "scale" in aug:
            self.aug_scale.setChecked(bool(aug["scale"]))
        if "blur" in aug:
            self.aug_blur.setChecked(bool(aug["blur"]))

    def _pick_checkpoint(self) -> None:
        """Open a file chooser to select a .pth checkpoint for resume training."""
        path, _ = QFileDialog.getOpenFileName(self, "Checkpoint wählen", "", "PyTorch (*.pth)")
        if path:
            self.resume_path_label.setText(path)
            self.resume_cb.setChecked(True)

    # ------------------------------------------------------------------ training

    def _start(self) -> None:
        """Validate prerequisites, build the config, and start the training thread."""
        if not self.project:
            QMessageBox.warning(self, tr("common.no_project"), tr("common.no_project_msg"))
            return
        if len(self.project.labels) < 2:
            QMessageBox.warning(self, tr("training.no_labels_title"), tr("training.no_labels_msg"))
            return
        cfg = self._get_config()
        save_dir = os.path.join(self.project.get_project_dir() or ".", "models")
        self.project.training_config = cfg
        self.log_text.clear()
        self.metrics_text.clear()
        self._history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}
        self.progress_bar.setValue(0)
        self._train_start_time = time.time()
        self._eta_lbl.setText(tr("training.eta_init"))
        if self._audit:
            self._audit.log_training_started(cfg.get("seed", 42), cfg)

        if self.ssh_enabled_cb.isChecked():
            ssh_cfg = self._current_ssh_cfg()
            if ssh_cfg is None:
                QMessageBox.warning(self, tr("training.ssh_error"), tr("training.ssh_no_profile"))
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
        """Request a graceful stop of the running training thread."""
        if self._thread:
            self._thread.request_stop()
        self.stop_btn.setEnabled(False)

    def _start_hpt(self) -> None:
        """Launch Optuna hyperparameter search dialog, then run HPTThread."""
        if not self.project:
            QMessageBox.warning(self, tr("common.no_project"), tr("common.no_project_msg"))
            return
        try:
            from core.hyperparameter_tuning import HPTThread
        except ImportError:
            QMessageBox.warning(self, tr("training.hpt_optuna_missing"),
                                tr("training.hpt_optuna_install"))
            return

        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QFormLayout, QSpinBox,
            QComboBox, QDialogButtonBox,
        )
        from gui.widgets.hpt_progress_dialog import HptProgressDialog

        dlg = QDialog(self)
        dlg.setWindowTitle(tr("training.hpt_dialog_title"))
        v = QVBoxLayout(dlg)
        form = QFormLayout()
        n_spin = QSpinBox()
        n_spin.setRange(5, 200)
        n_spin.setValue(20)
        form.addRow(tr("training.hpt_trials_label"), n_spin)
        t_spin = QSpinBox()
        t_spin.setRange(60, 7200)
        t_spin.setValue(300)
        t_spin.setSuffix(" s")
        form.addRow(tr("training.hpt_timeout_label"), t_spin)
        dev_combo = QComboBox()
        dev_combo.addItems(["cpu", "cuda", "mps"])
        form.addRow(tr("training.hpt_device_label"), dev_combo)
        v.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        if dlg.exec() != QDialog.Accepted:
            return

        n_trials = n_spin.value()
        prog = HptProgressDialog(n_trials, parent=self)
        prog.setModal(True)
        prog.show()

        hpt = HPTThread(
            project=self.project,
            n_trials=n_trials,
            timeout=float(t_spin.value()),
            device=dev_combo.currentText(),
            parent=self,
        )
        # Keep a strong Python reference so the thread is not GC'd while running.
        self._hpt_thread = hpt
        self.hpt_btn.setEnabled(False)

        hpt.progress.connect(
            lambda cur, tot, val: prog.update_progress(cur, tot, f"Beste Val-Acc: {val*100:.2f}%")
        )
        hpt.log.connect(prog.append_log)

        def _hpt_cleanup() -> None:
            self._hpt_thread = None
            self.hpt_btn.setEnabled(True)

        def _on_hpt_done(result: dict) -> None:
            _hpt_cleanup()
            prog.set_done()
            params = result.get("best_params", {})
            best = result.get("best_value", 0.0)
            lines = [f"Beste Val-Accuracy: {best*100:.2f}%\n", tr("training.hpt_best_params")]
            for k, val in params.items():
                lines.append(f"  {k}: {val}")
            lines.append("\n" + tr("training.hpt_apply_question"))
            reply = QMessageBox.question(self, tr("training.hpt_completed"), "\n".join(lines),
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self._apply_hpt_params(params)

        def _on_hpt_error(msg: str) -> None:
            _hpt_cleanup()
            prog.close()
            QMessageBox.critical(self, tr("common.error"), msg)

        hpt.finished.connect(_on_hpt_done)
        hpt.error.connect(_on_hpt_error)
        prog.rejected.connect(hpt.stop)
        hpt.start()

    def _apply_hpt_params(self, params: dict) -> None:
        """Copy best HPT parameters into the UI controls."""
        from models.classifier import get_available_models
        if "model_type" in params:
            models = get_available_models()
            if params["model_type"] in models:
                self.model_combo.setCurrentText(params["model_type"])
        if "batch_size" in params:
            self.batch_spin.setValue(int(params["batch_size"]))
        if "lr" in params:
            self.lr_spin.setValue(float(params["lr"]))
        if "optimizer" in params:
            idx = self.opt_combo.findText(params["optimizer"])
            if idx >= 0:
                self.opt_combo.setCurrentIndex(idx)

    @Slot(int, int, float, float, float, float)
    def _on_progress(self, epoch, total, tl, vl, ta, va) -> None:
        """Update progress bar, metric badges, ETA, and live training curves each epoch."""
        self.progress_bar.setValue(int(epoch / total * 100))
        self.epoch_label.setText(tr("training.epoch_progress", epoch=epoch, total=total))
        self.train_loss_lbl.setText(tr("training.metric_train_loss", val=f"{tl:.4f}"))
        self.val_loss_lbl.setText(tr("training.metric_val_loss", val=f"{vl:.4f}"))
        self.train_acc_lbl.setText(tr("training.metric_train_acc", val=f"{ta*100:.1f}%"))
        self.val_acc_lbl.setText(tr("training.metric_val_acc", val=f"{va*100:.1f}%"))
        # ETA
        if hasattr(self, "_train_start_time") and epoch > 0:
            elapsed = time.time() - self._train_start_time
            avg = elapsed / epoch
            remaining = (total - epoch) * avg
            if remaining >= 60:
                eta_str = f"~{int(remaining // 60)}m {int(remaining % 60)}s"
            else:
                eta_str = f"~{int(remaining)}s"
            self._eta_lbl.setText(tr("training.eta_label", eta=eta_str))
        for k, v in [("train_loss", tl), ("val_loss", vl), ("train_acc", ta), ("val_acc", va)]:
            self._history[k].append(v)
        self.curves_widget.update_curves(self._history)

    @Slot(str)
    def _on_log(self, msg: str) -> None:
        """Append a log message to the training log text area."""
        self.log_text.append(msg)

    @Slot(dict)
    def _on_finished(self, result: Dict) -> None:
        """Populate all result tabs and emit ``training_finished`` when training is done."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self._train_start_time = None
        self._eta_lbl.setText(tr("training.eta_init"))

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
            f"║  {tr('training.test_accuracy')}  {test_metrics.get('accuracy',0)*100:6.2f}%                     ║",
            f"║  {tr('training.test_f1')}  {test_metrics.get('macro_f1',0)*100:6.2f}%                     ║",
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

        # Enable AL scan now that we have a model
        self._last_model_path = result.get("best_model_path", "")
        if self._last_model_path and os.path.isfile(self._last_model_path):
            self._al_scan_btn.setEnabled(True)
            self._al_status.setText(
                f"Modell bereit. Ungelabelte Bilder: "
                f"{len(self.project.get_unlabeled_images()) if self.project else '?'}"
            )

        # Jump straight to Test-Ergebnisse tab
        self.tabs.setCurrentIndex(4)
        self.training_finished.emit(result)
        self._thread = None
        QMessageBox.information(
            self, tr("training.completed_title"),
            f"{tr('training.test_accuracy')}  {test_metrics.get('accuracy',0)*100:.2f}%\n"
            f"{tr('training.test_f1')} {test_metrics.get('macro_f1',0)*100:.2f}%\n\n"
            f"{tr('training.best_model')} {result.get('best_model_path', '')}"
        )

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        """Re-enable the start button and show a critical dialog on training error."""
        self._thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._train_start_time = None
        self._eta_lbl.setText(tr("training.eta_init"))
        self.log_text.append(f"FEHLER: {msg}")
        QMessageBox.critical(self, tr("training.error_title"), msg)

    # ------------------------------------------------------------------ SSH helpers

    def _refresh_ssh_profiles(self) -> None:
        """Reload SSH profile list from settings and repopulate the combo box."""
        if not self._settings:
            return
        self._ssh_profiles = self._settings.get_ssh_profiles()
        self.ssh_profile_combo.clear()
        for p in self._ssh_profiles:
            name = p.get("name", "?")
            host = p.get("host", "?")
            self.ssh_profile_combo.addItem(f"{name}  —  {host}")

    def _on_focal_toggled(self, checked: bool) -> None:
        """Enable or disable the gamma spinner when Focal Loss is toggled."""
        self._focal_gamma_label.setEnabled(checked)
        self.focal_gamma_spin.setEnabled(checked)

    def _on_ssh_toggled(self, state: int) -> None:
        """Enable or disable SSH-related controls when the checkbox is toggled."""
        enabled = bool(state)
        self.ssh_profile_combo.setEnabled(enabled)
        self.ssh_python_edit.setEnabled(enabled)
        self.ssh_remote_path_edit.setEnabled(enabled)
        self.ssh_test_btn.setEnabled(enabled)
        if enabled:
            self._refresh_ssh_profiles()

    def _current_ssh_cfg(self) -> Optional[Dict]:
        """Build an SSH config dict from the currently selected profile and form fields."""
        idx = self.ssh_profile_combo.currentIndex()
        if idx < 0 or idx >= len(self._ssh_profiles):
            return None
        profile = dict(self._ssh_profiles[idx])
        profile["python_env"] = self.ssh_python_edit.text().strip() or "python3"
        profile["remote_path"] = self.ssh_remote_path_edit.text().strip() or "/tmp/ils_project"
        return profile

    def _test_ssh(self) -> None:
        """Test the selected SSH connection in a background thread and update the status label."""
        cfg = self._current_ssh_cfg()
        if cfg is None:
            QMessageBox.warning(self, tr("training.ssh_error"), tr("training.ssh_no_profile"))
            return
        self.ssh_status_lbl.setText(tr("training.ssh_connecting"))
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
        """Display the SSH connection test result with green (success) or red (failure)."""
        self.ssh_test_btn.setEnabled(True)
        if ok:
            self.ssh_status_lbl.setStyleSheet("color: #2ECC71;")
            self.ssh_status_lbl.setText(tr("training.ssh_status_ok", msg=msg))
        else:
            self.ssh_status_lbl.setStyleSheet("color: #E74C3C;")
            self.ssh_status_lbl.setText(tr("training.ssh_status_err", msg=msg))

    def _on_cm_cell_clicked(self, true_idx: int, pred_idx: int) -> None:
        """Open misclassified-images dialog for the clicked confusion matrix cell."""
        if not self._test_predictions or not self._last_class_names:
            QMessageBox.information(
                self, tr("training.no_test_data_title"),
                tr("training.no_test_data_msg")
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
        """Save the last training result as a self-contained HTML report file."""
        if not hasattr(self, "_last_result"):
            QMessageBox.information(self, "Kein Ergebnis", "Erst Training durchführen.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, tr("training.html_report_dlg"), "training_report.html", "HTML (*.html)"
        )
        if not path:
            return
        try:
            from core.report import generate_html_report
            generate_html_report(
                self._last_result, path,
                project_name=self.project.config.name if self.project else ""
            )
            QMessageBox.information(self, tr("common.done"), tr("training.export_success_msg", path=path))
        except Exception as exc:
            QMessageBox.critical(self, tr("common.error"), str(exc))

    def _export_excel(self) -> None:
        """Save the last training result as an Excel (.xlsx) workbook."""
        if not hasattr(self, "_last_result"):
            QMessageBox.information(self, "Kein Ergebnis", "Erst Training durchführen.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, tr("training.excel_report_dlg"), "training_report.xlsx", "Excel (*.xlsx)"
        )
        if not path:
            return
        try:
            from core.export import export_training_report
            export_training_report(self._last_result, path)
            QMessageBox.information(self, tr("common.done"), tr("training.export_success_msg", path=path))
        except Exception as exc:
            QMessageBox.critical(self, tr("common.error"), str(exc))
