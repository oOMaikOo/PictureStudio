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
        self._history: Dict = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}
        self._audit = None
        self._settings = None
        self._ssh_profiles: List[Dict] = []
        self._test_predictions: List[Dict] = []
        self._last_class_names: List[str] = []
        self._last_model_path: str = ""
        self._build_ui()

    def set_project(self, project, audit=None) -> None:
        """Accept a new project, update the model save directory, and reload saved config."""
        self.project = project
        self._audit = audit
        save_dir = os.path.join(project.get_project_dir(), "models") if project.get_project_dir() else "models"
        self.save_dir_label.setText(save_dir)
        self._load_config()
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
        self.model_combo.setToolTip(
            "Architektur des neuronalen Netzes:\n"
            "• ResNet-18 — schnell, guter Startpunkt, ~11 M Parameter\n"
            "• ResNet-50 — höhere Kapazität, ~25 M Parameter\n"
            "• MobileNetV2 — sehr effizient, gut für CPU-Deployment\n"
            "• EfficientNet-B0 — kompakt, ~77% ImageNet-Acc\n"
            "• EfficientNet-B3 — deutlich besser als B0, ~82% ImageNet-Acc ★\n"
            "• ConvNeXt-Tiny — modernste CNN-Architektur, ~82% ImageNet-Acc ★\n"
            "  Empfehlung: EfficientNet-B3 oder ConvNeXt-Tiny für beste Ergebnisse\n"
            "• SimpleCNN — kein Transfer Learning, ideal für Tests"
        )
        form.addRow("Architektur:", self.model_combo)

        self.pretrained_cb = QCheckBox("Vortrainierte Gewichte (ImageNet)")
        self.pretrained_cb.setChecked(True)
        self.pretrained_cb.setToolTip(
            "Transfer Learning: Gewichte aus ImageNet-Vortraining laden.\n"
            "Empfohlen — braucht viel weniger Daten und Epochen.\n"
            "Deaktivieren nur wenn die Bilder sehr unähnlich zu Fotos sind\n"
            "(z. B. Röntgenbilder, Mikroskopie, Satellitenbilder)."
        )
        form.addRow("", self.pretrained_cb)

        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(32, 1024)
        self.img_size_spin.setValue(224)
        self.img_size_spin.setSingleStep(32)
        self.img_size_spin.setToolTip(
            "Eingabegröße in Pixel (quadratisch). Bilder werden auf dieses Format\n"
            "skaliert bevor sie ins Netz gehen.\n"
            "• 224 px — Standard für ImageNet-vortrainierte Modelle\n"
            "• 128 px — schneller, weniger Speicher, ausreichend für einfache Aufgaben\n"
            "• 320–512 px — für kleine Details oder feine Strukturen\n"
            "Tipp: immer Vielfaches von 32 wählen."
        )
        form.addRow("Bildgröße (px):", self.img_size_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(16)
        self.batch_spin.setToolTip(
            "Anzahl Bilder pro Trainingsschritt.\n"
            "• Größere Batches → stabilere Gradienten, brauchen mehr GPU-Speicher\n"
            "• Kleinere Batches → weniger Speicher, etwas rauschigeres Training\n"
            "Empfehlung: 32 (GPU) | 8–16 (CPU) | 4–8 (wenig Daten)"
        )
        form.addRow("Batch-Größe:", self.batch_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(20)
        self.epochs_spin.setToolTip(
            "Anzahl vollständiger Durchläufe durch den Trainingsdatensatz.\n"
            "• Zu wenig → Underfitting (Modell lernt zu wenig)\n"
            "• Zu viel → Overfitting (Modell lernt auswendig)\n"
            "Empfehlung: 20–50 mit Early Stopping; \n"
            "beobachte Val-Loss in den Live-Kurven."
        )
        form.addRow("Epochen:", self.epochs_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-7, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(7)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setToolTip(
            "Lernrate — wie groß die Gewichtsänderungen pro Schritt sind.\n"
            "• 0.001 (1e-3) — Standardwert, gut für Adam/AdamW\n"
            "• 0.0001 (1e-4) — konservativ, wenn Training instabil\n"
            "• 0.01 — aggressiv, funktioniert manchmal mit SGD\n"
            "Tipp: bei 'NaN Loss' Lernrate um Faktor 10 reduzieren."
        )
        form.addRow("Learning Rate:", self.lr_spin)

        self.opt_combo = QComboBox()
        self.opt_combo.addItems(["adam", "adamw", "sgd"])
        self.opt_combo.setToolTip(
            "Optimierungsalgorithmus:\n"
            "• adam — adaptiv, robust, guter Standard für die meisten Aufgaben\n"
            "• adamw — wie Adam + L2-Gewichtsregularisierung (gegen Overfitting)\n"
            "• sgd — klassisch, oft mit Momentum; braucht sorgfältige LR-Wahl"
        )
        form.addRow("Optimizer:", self.opt_combo)

        self.sched_combo = QComboBox()
        self.sched_combo.addItems(["reduce_on_plateau", "cosine", "step"])
        self.sched_combo.setToolTip(
            "Lernraten-Scheduler — passt die LR während des Trainings an:\n"
            "• reduce_on_plateau — halbiert LR wenn Val-Loss stagniert (empfohlen)\n"
            "• cosine — sanfte Cosinuskurve von LR bis fast 0 über alle Epochen\n"
            "• step — reduziert LR alle N Epochen um festen Faktor"
        )
        form.addRow("LR-Scheduler:", self.sched_combo)

        self.early_stop_spin = QSpinBox()
        self.early_stop_spin.setRange(0, 100)
        self.early_stop_spin.setValue(0)
        self.early_stop_spin.setToolTip(
            "Training automatisch stoppen wenn Val-Loss sich N Epochen nicht verbessert.\n"
            "0 = deaktiviert\n"
            "Empfehlung: 5–10 — schützt vor Overfitting und spart Zeit.\n"
            "Das beste Modell (niedrigster Val-Loss) wird gespeichert."
        )
        form.addRow("Early Stopping (Geduld):", self.early_stop_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        self.seed_spin.setToolTip(
            "Zufalls-Seed für reproduzierbare Ergebnisse.\n"
            "Gleicher Seed → gleicher Train/Val-Split und gleiche Augmentation.\n"
            "Wert ändern um zu prüfen ob Ergebnisse stabil sind."
        )
        form.addRow("Seed:", self.seed_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda", "mps"])
        self.device_combo.setToolTip(
            "Rechengerät für das Training:\n"
            "• auto — wählt automatisch GPU (cuda) > Apple MPS > CPU\n"
            "• cuda — NVIDIA GPU (10–50× schneller als CPU)\n"
            "• mps — Apple Silicon GPU (Mac M1/M2/M3, ~5–15× schneller)\n"
            "• cpu — immer verfügbar, aber langsam"
        )
        form.addRow("Gerät:", self.device_combo)

        self.amp_cb = QCheckBox("Mixed Precision (AMP, nur CUDA)")
        self.amp_cb.setToolTip(
            "Automatic Mixed Precision: berechnet in float16 wo möglich.\n"
            "Nur auf NVIDIA-GPUs mit Tensor Cores (RTX/Ampere+) sinnvoll.\n"
            "Vorteil: ~1,5–2× schneller, weniger GPU-Speicher.\n"
            "Nachteil: marginale Präzisionsverluste (normalerweise unkritisch)."
        )
        form.addRow("", self.amp_cb)

        # Split
        split_box = QGroupBox("Daten-Split")
        sf = QFormLayout(split_box)
        self.train_split = QDoubleSpinBox()
        self.train_split.setRange(0.1, 0.9)
        self.train_split.setValue(0.7)
        self.train_split.setSingleStep(0.05)
        self.train_split.setToolTip(
            "Anteil der Bilder für das Training (0.7 = 70%).\n"
            "Rest wird auf Validation + Test aufgeteilt.\n"
            "Bei wenig Daten (<200 Bilder): 0.8 empfohlen."
        )
        sf.addRow("Train:", self.train_split)
        self.val_split = QDoubleSpinBox()
        self.val_split.setRange(0.05, 0.5)
        self.val_split.setValue(0.2)
        self.val_split.setSingleStep(0.05)
        self.val_split.setToolTip(
            "Anteil der Bilder für die Validation (0.2 = 20%).\n"
            "Wird nach jeder Epoche ausgewertet — steuert Early Stopping\n"
            "und LR-Scheduler. Nicht für das Training verwendet."
        )
        sf.addRow("Validation:", self.val_split)
        form.addRow(split_box)

        # Augmentation
        aug_box = QGroupBox("Augmentation")
        ab = QVBoxLayout(aug_box)
        self.aug_flip = QCheckBox("Flip (horizontal + vertikal)")
        self.aug_flip.setChecked(True)
        self.aug_flip.setToolTip(
            "Spiegelt Bilder zufällig horizontal und/oder vertikal.\n"
            "Günstig wenn Ausrichtung keine Rolle spielt (z. B. Qualitätskontrolle).\n"
            "Deaktivieren wenn Orientierung wichtig ist (z. B. Schriften, Pfeile)."
        )
        self.aug_rotation = QCheckBox("Rotation (±15°)")
        self.aug_rotation.setChecked(True)
        self.aug_rotation.setToolTip(
            "Dreht Bilder zufällig um bis zu ±15° (per Editor anpassbar).\n"
            "Hilft gegen Rotation der Kamera / des Objekts.\n"
            "Intensität im Augmentierungs-Editor einstellen."
        )
        self.aug_brightness = QCheckBox("Helligkeit / Kontrast")
        self.aug_brightness.setChecked(True)
        self.aug_brightness.setToolTip(
            "Variiert Helligkeit und Kontrast zufällig.\n"
            "Macht das Modell robuster gegen unterschiedliche Beleuchtung.\n"
            "Intensität im Augmentierungs-Editor einstellen."
        )
        self.aug_scale = QCheckBox("Skalierung (Random Crop)")
        self.aug_scale.setToolTip(
            "Schneidet einen zufälligen Bildausschnitt aus und skaliert ihn.\n"
            "Hilft gegen leichte Positionsänderungen des Objekts im Bild.\n"
            "Intensität (min. Crop-Anteil) im Augmentierungs-Editor einstellen."
        )
        self.aug_blur = QCheckBox("Blur (Gaussian)")
        self.aug_blur.setToolTip(
            "Unscharfe Bilder durch Gaussian-Blur simulieren.\n"
            "Gut gegen Unschärfe durch Kamerabewegung oder Defokus.\n"
            "Radius im Augmentierungs-Editor einstellen."
        )
        for cb in [self.aug_flip, self.aug_rotation, self.aug_brightness,
                   self.aug_scale, self.aug_blur]:
            ab.addWidget(cb)
        aug_preview_btn = QPushButton("🔍 Editor / Vorschau…")
        aug_preview_btn.setToolTip(
            "Augmentierungs-Editor öffnen: Intensitäten einstellen und Live-Vorschau sehen."
        )
        aug_preview_btn.setStyleSheet(
            "background:#6C3483; color:white; padding:4px; border-radius:3px;"
        )
        aug_preview_btn.clicked.connect(self._show_aug_preview)
        ab.addWidget(aug_preview_btn)
        form.addRow(aug_box)

        self.use_rois_cb = QCheckBox("ROI-Bereiche verwenden")
        self.use_rois_cb.setChecked(True)
        self.use_rois_cb.setToolTip(
            "Wenn Bilder ROIs (Regions of Interest) haben, wird nur der\n"
            "ROI-Bereich für das Training ausgeschnitten.\n"
            "Nützlich um irrelevante Bildbereiche auszublenden."
        )
        form.addRow("", self.use_rois_cb)

        self.class_balance_cb = QCheckBox("Klassenausgleich (WeightedSampler)")
        self.class_balance_cb.setToolTip(
            "Gleicht ungleichmäßige Klassenverteilungen aus, indem unterrepräsentierte "
            "Klassen häufiger gesampelt werden."
        )
        form.addRow("", self.class_balance_cb)

        self.focal_loss_cb = QCheckBox("Focal Loss")
        self.focal_loss_cb.setToolTip(
            "Focal Loss fokussiert das Training auf schwierige Beispiele.\n"
            "Empfohlen bei stark ungleichen Klassen (z.B. 10:1 Normal/Defekt).\n"
            "Nicht aktiv bei Multi-Label-Klassifikation."
        )
        self.focal_loss_cb.toggled.connect(self._on_focal_toggled)
        form.addRow("", self.focal_loss_cb)

        from PySide6.QtWidgets import QDoubleSpinBox as _DSB, QHBoxLayout as _HBox
        focal_row = _HBox()
        self._focal_gamma_label = QLabel("γ (Gamma):")
        self._focal_gamma_label.setEnabled(False)
        focal_row.addWidget(self._focal_gamma_label)
        self.focal_gamma_spin = QDoubleSpinBox()
        self.focal_gamma_spin.setRange(0.5, 5.0)
        self.focal_gamma_spin.setValue(2.0)
        self.focal_gamma_spin.setSingleStep(0.5)
        self.focal_gamma_spin.setDecimals(1)
        self.focal_gamma_spin.setEnabled(False)
        self.focal_gamma_spin.setToolTip(
            "Focal-Loss-Exponent γ (gamma).\n"
            "γ=0 → identisch mit CrossEntropy\n"
            "γ=2 → Standardwert (empfohlen)\n"
            "γ=5 → sehr starker Fokus auf schwierige Bilder"
        )
        focal_row.addWidget(self.focal_gamma_spin)
        focal_row.addStretch()
        form.addRow("", focal_row)

        self.save_dir_label = QLabel("(Projekt öffnen)")
        self.save_dir_label.setWordWrap(True)
        form.addRow("Speicherort:", self.save_dir_label)

        # Resume
        self.resume_cb = QCheckBox("Training fortsetzen (Resume)")
        self.resume_cb.setToolTip(
            "Setzt ein unterbrochenes Training ab einem gespeicherten\n"
            "Checkpoint fort. Checkpoint unten auswählen.\n"
            "Architektur und Bildgröße müssen mit dem Checkpoint übereinstimmen."
        )
        form.addRow("", self.resume_cb)
        resume_btn = QPushButton("Checkpoint wählen…")
        resume_btn.setToolTip("PyTorch-Checkpoint (.pth) für Resume-Training laden")
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

        # Buttons — Reihenfolge: HPT → Start → Stop
        self.hpt_btn = QPushButton("⚙ Hyperparameter-Suche…")
        self.hpt_btn.setStyleSheet(
            "background:#6C3483;color:white;padding:6px;border-radius:3px;"
        )
        self.hpt_btn.setToolTip(
            "Automatische Hyperparameter-Suche mit Optuna starten.\n"
            "Testet verschiedene Lernraten, Batch-Größen und Architekturen\n"
            "und gibt die besten Parameter zurück.\n\n"
            "Benötigt: pip install optuna"
        )
        self.hpt_btn.clicked.connect(self._start_hpt)
        form.addRow(self.hpt_btn)

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

        # ── Active Learning tab ──────────────────────────────────────────────
        self.tabs.addTab(self._build_al_tab(), "🔄 Active Learning")

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
        self._al_thr_spin.setToolTip(
            "Bilder mit Confidence unterhalb dieses Schwellwerts gelten als unsicher."
        )
        form.addRow("Unsicherheits-Schwellwert:", self._al_thr_spin)

        self._al_n_spin = QSpinBox()
        self._al_n_spin.setRange(5, 500)
        self._al_n_spin.setValue(50)
        self._al_n_spin.setToolTip("Maximale Anzahl Bilder, die in die AL-Queue eingetragen werden.")
        form.addRow("Max. Kandidaten:", self._al_n_spin)

        v.addLayout(form)

        self._al_scan_btn = QPushButton("🔍 AL-Scan starten")
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
            self._al_status.setText("Keine ungelabelten Bilder im Projekt.")
            return

        from core.active_learning import ActiveLearningThread

        self._al_scan_btn.setEnabled(False)
        self._al_progress.setValue(0)
        self._al_progress.show()
        self._al_status.setText(f"Scanne {len(unlabeled)} ungelabelte Bilder…")

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
        self._al_progress.hide()
        self._al_scan_btn.setEnabled(True)

        if not candidates:
            self._al_status.setText(
                "Keine unsicheren Vorhersagen gefunden "
                f"(Schwellwert {self._al_thr_spin.value():.0%})."
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

        parts = [f"✅ {added} Bilder in AL-Queue eingetragen"]
        if skipped:
            parts.append(f"({skipped} bereits vorhanden)")
        parts.append("→ Labeling-Seite öffnen, um zu reviewen.")
        self._al_status.setText("  ".join(parts))

        if added > 0:
            self.al_queue_updated.emit()

    def _on_al_error(self, msg: str) -> None:
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
        """Request a graceful stop of the running training thread."""
        if self._thread:
            self._thread.request_stop()
        self.stop_btn.setEnabled(False)

    def _start_hpt(self) -> None:
        """Launch Optuna hyperparameter search dialog, then run HPTThread."""
        if not self.project:
            QMessageBox.warning(self, "Kein Projekt", "Bitte zuerst ein Projekt öffnen.")
            return
        try:
            from core.hyperparameter_tuning import HPTThread
        except ImportError:
            QMessageBox.warning(self, "Optuna fehlt",
                                "Hyperparameter-Suche benötigt Optuna:\npip install optuna")
            return

        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QFormLayout, QSpinBox,
            QComboBox, QDialogButtonBox,
        )
        from gui.widgets.hpt_progress_dialog import HptProgressDialog

        dlg = QDialog(self)
        dlg.setWindowTitle("Hyperparameter-Suche konfigurieren")
        v = QVBoxLayout(dlg)
        form = QFormLayout()
        n_spin = QSpinBox()
        n_spin.setRange(5, 200)
        n_spin.setValue(20)
        form.addRow("Anzahl Versuche:", n_spin)
        t_spin = QSpinBox()
        t_spin.setRange(60, 7200)
        t_spin.setValue(300)
        t_spin.setSuffix(" s")
        form.addRow("Zeitlimit:", t_spin)
        dev_combo = QComboBox()
        dev_combo.addItems(["cpu", "cuda", "mps"])
        form.addRow("Gerät:", dev_combo)
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
            lines = [f"Beste Val-Accuracy: {best*100:.2f}%\n", "Beste Parameter:"]
            for k, val in params.items():
                lines.append(f"  {k}: {val}")
            lines.append("\nParameter in Konfiguration übernehmen?")
            reply = QMessageBox.question(self, "HPT abgeschlossen", "\n".join(lines),
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self._apply_hpt_params(params)

        def _on_hpt_error(msg: str) -> None:
            _hpt_cleanup()
            prog.close()
            QMessageBox.critical(self, "HPT-Fehler", msg)

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
        """Update progress bar, metric badges, and live training curves each epoch."""
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
        """Append a log message to the training log text area."""
        self.log_text.append(msg)

    @Slot(dict)
    def _on_finished(self, result: Dict) -> None:
        """Populate all result tabs and emit ``training_finished`` when training is done."""
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
        QMessageBox.information(
            self, "Training abgeschlossen",
            f"Test-Accuracy:  {test_metrics.get('accuracy',0)*100:.2f}%\n"
            f"Test-F1 (macro):{test_metrics.get('macro_f1',0)*100:.2f}%\n\n"
            f"Bestes Modell: {result.get('best_model_path', '')}"
        )

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        """Re-enable the start button and show a critical dialog on training error."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_text.append(f"FEHLER: {msg}")
        QMessageBox.critical(self, "Trainingsfehler", msg)

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
        """Display the SSH connection test result with green (success) or red (failure)."""
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
        """Save the last training result as a self-contained HTML report file."""
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
        """Save the last training result as an Excel (.xlsx) workbook."""
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
