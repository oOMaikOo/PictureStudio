# Image Labeling Studio

A production-ready desktop application for image annotation, ROI definition, CNN model training, and batch inference — built with **PySide6** and **PyTorch**.

---

## Feature Overview

| Category | Features |
|---|---|
| **Project management** | Versioned JSON projects, atomic saves, automatic backups, project dashboard, image validation & relocation |
| **Camera capture** | USB & IP/RTSP camera live preview, single & burst capture, optional timestamp overlay (preview + burned into saved PNG) |
| **Anomaly detection** | Unsupervised Conv-Autoencoder trained on normal frames; live reconstruction-error scoring, configurable threshold, alarm banner, auto-save flagged frames |
| **ROI editor** | Rectangle, ellipse, polygon drawing; copy/paste; keyboard shortcuts; label quick-assign (1–9); bounds validation; ROI templates |
| **Labeling** | Label hierarchies (multi-label), statistics, label filter, review mode, change history via audit trail |
| **Dataset analysis** | Format/size statistics, missing-file detection, MD5 duplicate detection, class imbalance warnings; COCO / YOLO / CSV export |
| **Training** | ResNet18/50, MobileNetV2, EfficientNet-B0, SimpleCNN; early stopping, LR schedulers, mixed precision, GPU/CPU/MPS selection, resume from checkpoint |
| **SSH remote training** | Connection profiles, live log streaming, conda/venv environment support |
| **Model library** | Versioned model registry, ONNX export, accuracy/F1 comparison, archive/delete |
| **Metrics & reports** | Accuracy, F1, weighted F1, ROC/AUC (binary), top-K accuracy, HTML training report, Excel training report |
| **Inference** | Batch inference with top-3 display, confidence color coding, low-confidence tab, label/confidence filters |
| **Excel export** | Custom column mapping (enable/disable + rename), append/overwrite mode, styled headers, red highlight for uncertain predictions |
| **UX** | 8-page sidebar navigation, dark/light theme, QSettings persistence, lazy thumbnail loading, crash reports |
| **Tests** | Unit tests (project, dataset, metrics, ROI, export, anomaly detector) + integration tests (train → infer pipeline) |

---

## Installation

### Prerequisites

- Python 3.10 or later
- (Optional) CUDA-capable GPU for faster training

### Steps

```bash
git clone <repo-url>
cd Picture

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the application
python main.py
```

> **macOS note:** PyTorch may use the MPS backend automatically on Apple Silicon. Select *auto* or *mps* in the Training page device dropdown.

---

## Project Structure

```
Picture/
├── main.py                        # Entry point
├── requirements.txt
│
├── core/
│   ├── project.py                 # Central data model (labels, images, ROIs)
│   ├── dataset.py                 # Analysis, split, COCO/YOLO/CSV export
│   ├── training.py                # TrainingWorker (QThread) + EarlyStopping
│   ├── inference.py               # Inferencer: single image & folder batch
│   ├── metrics.py                 # Accuracy, F1, ROC/AUC, top-K
│   ├── export.py                  # Excel export (results + training report)
│   ├── model_manager.py           # Model registry + ONNX export
│   ├── camera.py                  # USB/IP camera thread + frame utilities
│   ├── anomaly_detector.py        # Conv-Autoencoder: train on normal frames, score live frames
│   ├── audit.py                   # JSONL audit trail
│   └── report.py                  # HTML training report generator
│
├── models/
│   └── classifier.py              # Model factory + SimpleCNN + checkpoint I/O
│
├── gui/
│   ├── main_window.py             # MainWindow with sidebar + QStackedWidget
│   ├── sidebar.py                 # Navigation sidebar (8 pages)
│   ├── camera_capture_dialog.py   # Camera live-preview + capture dialog
│   ├── help_dialog.py             # Integrated help browser
│   ├── guide_tour.py              # Step-by-step guided tour overlay
│   ├── pages/
│   │   ├── dashboard_page.py      # Project stats overview
│   │   ├── data_page.py           # Dataset analysis + export
│   │   ├── labeling_page.py       # Thumbnail list + ROI editor
│   │   ├── training_page.py       # Training config + progress curves
│   │   ├── models_page.py         # Model library table
│   │   ├── inference_page.py      # Batch inference + low-confidence tab
│   │   ├── export_page.py         # Custom Excel export
│   │   └── settings_page.py       # Theme, autosave, SSH profiles
│   └── widgets/
│       ├── roi_editor.py          # QGraphicsView ROI editor (rect/ellipse/polygon)
│       ├── thumbnail_list.py      # Lazy-loading QListWidget
│       └── charts.py              # Training curves + confusion matrix
│
├── utils/
│   ├── config.py                  # App constants, defaults
│   ├── logging_utils.py           # File + console logging setup
│   ├── reproducibility.py         # Seed setting, software version capture
│   └── settings.py                # QSettings wrapper (AppSettings)
│
└── tests/
    ├── conftest.py                # pytest fixtures (sample_project, sample_images)
    ├── test_project.py            # Unit: labels, images, ROIs, save/load, backup
    ├── test_dataset.py            # Unit: analysis, splits, duplicates, exports
    ├── test_metrics.py            # Unit: accuracy, F1, ROC/AUC, top-K
    ├── test_roi.py                # Unit: ROI CRUD, serialization, templates
    ├── test_export.py             # Unit: Excel column mapping, append mode
    └── test_integration.py        # Integration: train → checkpoint → infer
```

---

## Quickstart Workflow

1. **New project** — `Datei → Neues Projekt`, give it a name.
2. **Load images** — Go to **Daten** page → *Bilder laden*, pick a folder — **or** go to the **Labeling** page → *Ordner laden…* — **or** capture directly from camera with `Datei → Kamera aufnehmen…` (`Ctrl+K`).
3. **Define labels** — In the **Labeling** page, add labels (name + colour).
4. **Label images** — Click an image in the thumbnail list, press 1–9 to quick-assign, or use the label dropdown.
5. **Draw ROIs** — Use the ROI toolbar (R = rectangle, E = ellipse, G = polygon). Delete with **Del**, copy/paste with **Ctrl+C / Ctrl+V**.
6. **Analyse dataset** — **Daten** page → *Analyse starten* to check for missing files, duplicates, class imbalance.
7. **Train** — **Training** page → configure architecture, epochs, LR → *Training starten*.
8. **Review metrics** — Training curves and confusion matrix update live. Export HTML or Excel report after training.
9. **Classify new images** — **Klassifikation** page → pick a model → *Klassifizieren*.
10. **Export results** — **Export** page → map columns → *Excel exportieren*.

---

## Camera Capture

Open via `Datei → Kamera aufnehmen…` (`Ctrl+K`).

| Feature | Description |
|---|---|
| **USB camera** | Auto-detected; select from dropdown, click *Verbinden* |
| **IP / RTSP camera** | Enter stream URL (rtsp://, http://); supports MJPEG and RTSP |
| **Single capture** | Click *Bild aufnehmen* or press `Space` |
| **Burst capture** | Set count + interval → *Burst starten* |
| **Timestamp overlay** | Show date/time in live preview (toggle, does not affect saved file) |
| **Timestamp burn-in** | Permanently render date/time into the saved PNG (`YYYY-MM-DD HH:MM:SS`, bottom-left) |
| **Anomaly detection** | Unsupervised Conv-Autoencoder; train on normal frames, live-score every frame, alarm on reconstruction-error spike |

Captured images are listed in the dialog; click *In Projekt übernehmen* to add them to the project.

> **Requirements:** `pip install opencv-python` (camera) — PyTorch is already required for training.

### Anomaly detection workflow

```
1. Connect camera → run normal process
2. "Normalframes aufnehmen" → collect 100–300 frames
3. "Training starten" → autoencoder trains on normal frames only
   Threshold = mean + 2.5 × std of training reconstruction errors (auto-set)
4. Checkbox "Aktiv" → live scoring begins (every 3rd frame, CPU-friendly)
   Green score label + normal border  →  normal
   Red banner + red border            →  anomaly detected
5. Optional: "Anomalie-Frames automatisch speichern" to collect evidence
6. Save/load the trained .pth model for reuse
```

---

## ROI Editor Keyboard Shortcuts

| Key | Action |
|---|---|
| `R` | Rectangle mode |
| `E` | Ellipse mode |
| `G` | Polygon mode |
| `Esc` | Cancel current drawing |
| `Del` | Delete selected ROI |
| `Ctrl+C` | Copy selected ROI |
| `Ctrl+V` | Paste copied ROI |
| `Arrow keys` | Nudge selected ROI by 2 px |
| `1`–`9` | Quick-assign label to selected ROI |
| `N` / `P` | Next / Previous image (labeling page) |

---

## Supported Architectures

| ID | Model | Notes |
|---|---|---|
| `resnet18` | ResNet-18 | Fast, good baseline |
| `resnet50` | ResNet-50 | Higher capacity |
| `mobilenet_v2` | MobileNetV2 | Efficient, mobile-friendly |
| `efficientnet_b0` | EfficientNet-B0 | Strong accuracy/size trade-off |
| `simple_cnn` | Custom 4-block CNN | No pretrained weights; fast for CPU testing |

All transfer-learning models use ImageNet pretrained weights by default (set *pretrained* to off for custom datasets with very different content).

---

## Training Options

| Option | Description |
|---|---|
| **Device** | `auto` / `cpu` / `cuda` / `mps` |
| **Scheduler** | `none`, `reduce_on_plateau`, `cosine`, `step` |
| **Early stopping patience** | Stop after N epochs without val improvement (`0` = disabled) |
| **Mixed precision** | AMP via `torch.cuda.amp.GradScaler` (CUDA only) |
| **Resume checkpoint** | Continue training from a saved `.pth` file |
| **Augmentation** | Random horizontal flip + colour jitter |

---

## Dataset Export Formats

| Format | File(s) | Use case |
|---|---|---|
| **COCO JSON** | `annotations.json` | Object detection frameworks |
| **YOLO TXT** | `<image>.txt` per image + `classes.txt` | Ultralytics / Darknet |
| **CSV** | `annotations.csv` | Spreadsheet / custom tooling |

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/test_project.py tests/test_dataset.py tests/test_metrics.py tests/test_roi.py tests/test_export.py -v

# Integration tests (require torch + Pillow)
pytest tests/test_integration.py -v

# With coverage
pip install pytest-cov
pytest tests/ --cov=core --cov=models --cov-report=term-missing
```

> Integration tests train a small model on 12 synthetic images — they run on CPU in ~10–30 seconds.

---

## Settings

Persistent preferences are stored via `QSettings` (platform native: `~/Library/Preferences` on macOS, registry on Windows).

| Setting | Default | Description |
|---|---|---|
| Theme | `dark` | `dark` or `light` |
| Font size | `9` | 7–16 pt |
| Autosave interval | `300 s` | 30–3600 s |
| Backup before save | `on` | Creates timestamped `.json` backup |
| Thumbnail size | `100 px` | 60–240 px |
| Low-confidence threshold | `0.70` | Predictions below this are flagged |
| Top-K display | `3` | 1–5 top predictions shown |
| SSH profiles | — | Host, user, key path per profile |

---

## Project File Format

Projects are saved as UTF-8 JSON (`*.json`). Atomic writes use a temporary file + `os.replace()` to prevent corruption on crash.

```json
{
  "config": { "name": "...", "version": "2.0", "created_at": "..." },
  "labels": [{ "name": "gut", "color": "#2ECC71" }],
  "images": ["path/to/img.jpg"],
  "image_labels": { "path/to/img.jpg": "gut" },
  "rois": {
    "path/to/img.jpg": [
      { "id": "r1", "type": "rect", "x": 10, "y": 10, "w": 50, "h": 50, "label": "gut", "color": "#2ECC71" }
    ]
  },
  "training_config": { ... },
  "inference_results": [ ... ]
}
```

---

## Audit Trail

Every label change, ROI addition/deletion, and training run is appended to `<project_name>_audit.jsonl` in the project directory. Each line is a JSON object:

```json
{"timestamp": "2025-01-01T12:00:00", "action": "image_labeled", "entity": "img.jpg", "details": {"label": "gut"}}
```

---

## Troubleshooting

**Application does not start**
- Ensure PySide6 is installed: `pip install PySide6`
- On Linux, install Qt platform plugins: `apt install libxcb-cursor0`

**Training is very slow**
- Select `cuda` or `mps` in the device dropdown.
- Reduce image size to 128 px and batch size to 16 for CPU testing.

**`ImportError: No module named 'openpyxl'`**
- `pip install openpyxl` — required for Excel export.

**`ImportError: No module named 'paramiko'`**
- `pip install paramiko` — only needed for SSH remote training.

**Charts do not appear**
- `pip install matplotlib` — the application falls back to ASCII sparklines if absent.

**Thumbnails load slowly**
- Increase thumbnail thread count or reduce thumbnail size in Settings.

**Project file is corrupt**
- The most recent `.bak` backup is saved alongside the project file. Rename it to `.json` to restore.

---

## License

MIT — see `LICENSE` for details.
