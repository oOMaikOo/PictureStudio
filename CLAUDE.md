# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the application
source .venv/bin/activate
python main.py

# Run all tests (696 pass, 3 skipped; integration tests take ~30 s each)
.venv/bin/python -m pytest tests/ -v

# Skip slow ML integration tests
.venv/bin/python -m pytest tests/ -q --ignore=tests/test_integration.py

# Run a single test class or function
.venv/bin/python -m pytest tests/test_project.py::TestQAFlags -v
.venv/bin/python -m pytest tests/test_project.py::TestSaveLoad::test_load_roundtrip -v

# Quick syntax check on a file
python3 -c "import ast; ast.parse(open('gui/pages/camera_page.py').read()); print('OK')"

# Run the standalone monitor daemon
python monitor.py                      # interactive camera scan + browser setup wizard
python monitor.py --model anomalie.pth
python monitor.py --setup             # multi-channel setup wizard (web UI on :8765)
python monitor.py --channels cfg.json # load pre-configured channels
```

Integration tests (`test_integration.py`) train a small model on 12 synthetic images on CPU and take ~10–30 seconds each. All other tests run in milliseconds.

## Architecture

### Project types and sidebar pages

The app has two project types — **Image** (classification) and **Video** (anomaly/stream). The sidebar (`gui/sidebar.py`) switches between `_IMAGE_PAGES` and `_VIDEO_PAGES` lists. `MainWindow` holds a single `QStackedWidget`; sidebar entries are `(label, icon, stack_index)` tuples.

Stack indices (add new pages here):
| Index | Page class | Visible in |
|-------|-----------|------------|
| 0 | `DashboardPage` | both |
| 1 | `DataPage` | both |
| 2 | `LabelingPage` | image |
| 3 | `TrainingPage` | image |
| 4 | `ModelsPage` | both |
| 5 | `InferencePage` | image |
| 6 | `ExportPage` | both |
| 7 | `SettingsPage` | both |
| 8 | `CameraPage` | video |
| 9 | `BatchInferencePage` | image |
| 10 | `MultiCameraPage` | video |
| 11 | `AnomalyClusteringPage` | both |
| 12 | `DatasetStatsPage` | image |
| 13 | `VideoAnnotationPage` | video |
| 14 | `FleetPage` | video |

### Central data model: `core/project.py`

`Project` is the single source of truth. All pages receive it via `set_project(project, audit=None)`. Key attributes:

- `images`, `labels`, `image_labels`, `image_multi_labels`, `rois` — labeling state
- `image_label_flags` — QA uncertain flags `{uncertain, comment}`
- `active_learning_queue` — AL review queue
- `config.multi_label` — single vs multi-label mode
- `config.project_type` — `"image"` or `"video"`

Project JSON uses atomic writes (temp file + `os.replace()`). `Project.load(path)` is a classmethod.

### GUI: `gui/main_window.py`

`_load_project()` calls `set_project()` on every page in sequence and switches the sidebar to the correct page list. Label changes go through a `QDialog` that emits `labels_changed` → `labeling_page.on_labels_changed()`.

### Labeling mutations — undo/redo

All label/ROI changes go through `QUndoStack` via command objects in `gui/labeling_commands.py`:
1. A public method pushes a `QUndoCommand` subclass.
2. The command's `redo()`/`undo()` call a `_do_*` method on the page.
3. The `_do_*` method mutates `Project` and updates the UI.

Commands: `SetImageLabelCommand`, `BulkSetImageLabelCommand`, `SetMultiLabelsCommand`, `SetLabelFlagCommand`, `AddROICommand`, `DeleteROICommand`, `AssignROILabelCommand`, `MoveROICommand`.

### Data loading: `gui/pages/data_page.py`

`_load_images()` uses `os.walk()` to scan the selected folder **and all subfolders** recursively. Drag & drop also handles dropped folders recursively — no need to flatten image directories before import. Train/test split is applied automatically at training time; users do not pre-separate images.

### Classification training pipeline

`TrainingPage` wraps `TrainingWorker` (plain class) inside `TrainingThread(QThread)`. Multi-label mode detected in two places: `create_datasets()` in `core/dataset.py` and `TrainingWorker.run()` (switches loss to `BCEWithLogitsLoss`).

Button order on `TrainingPage`: ① Hyperparameter-Suche → ② Training starten → ③ Training stoppen.

SSH remote training: `RemoteTrainingThread` (`core/remote_training.py`) zips images via `core/remote_ssh.py`, uploads, runs `scripts/remote_train.py` (self-contained, no local imports), streams logs, downloads checkpoint.

Hyperparameter search: `HPTWorker` + `HPTThread` in `core/hyperparameter_tuning.py` — Optuna study over `lr`, `batch_size`, `architecture`, `optimizer`. Requires `pip install optuna`.

### Anomaly detection pipeline

`AnomalyDetector` (`core/anomaly_detector.py`) wraps a `_ConvAutoencoder` (configurable via `base_ch=16`). Key API:
- `collect_frame(frame)` — accumulate training frames
- `n_collected()` — frame count
- `train(epochs, batch_size, lr)` — returns threshold float
- `save(path)` / `load(path)` — checkpoint includes `base_ch` in metadata
- `score_detailed(frame)` → `(score, reconstruction, heatmap_overlay, bbox)`

The detector is used by `CameraPage`, `CameraCaptureDialog`, and `MultiCameraPage`.

**HPT for anomaly**: `AnomalyHPTWorker` + `AnomalyHPTThread` in `core/hyperparameter_tuning.py` — Optuna study over `base_ch` (8/16/32), `lr`, `batch_size`. Triggered from `CameraCaptureDialog`.

**Grad-CAM**: `core/gradcam.py` — `compute_gradcam_anomaly(detector, frame_bgr)` targets `model.encoder[4]`, returns BGR overlay. Available as checkbox in `CameraPage` and `CameraCaptureDialog`.

### Camera and preprocessing

`core/camera.py` provides:
- `CameraFrameThread(QThread)` — emits `frame_ready(np.ndarray)`, supports `set_cam_props(dict)` for live property updates
- `apply_cam_props(cap, props)` — maps names (`brightness`, `contrast`, `saturation`, `sharpness`, `exposure`, `gain`) to `cv2.CAP_PROP_*`
- `apply_frame_filter(frame, name)` — returns BGR frame after `"grayscale"` / `"canny"` / `"sobel"` / `"laplacian"` (or original for `"none"`)
- `list_usb_cameras()` — enumerates USB cameras; on macOS uses a Swift subprocess for reliable names

`CameraPage` (stack 8) has two GroupBoxes in the left panel:
- **Kamera-Einstellungen** — sliders forwarded live via `set_cam_props()`
- **Vorverarbeitung** — filter dropdown; settings are passed to `CameraCaptureDialog` as start values on open

`CameraCaptureDialog` (`gui/camera_capture_dialog.py`) contains its own camera settings sliders and filter dropdown — these are available regardless of which page opens the dialog (DataPage or CameraPage). Button order inside the dialog: ① Hyperparameter-Suche → ② Training starten. After training, model is auto-loaded back into `CameraPage` without a confirmation dialog.

### Multi-camera

`MultiCameraPage` (stack 10) shows a dynamic grid (1–9 channels, 2×2 per page). Each channel has its own `CameraFrameThread` + `AnomalyDetector`. Alarm events are forwarded to `IndustrialNotifier`. REST endpoints: `GET /api/mc/channels`, `/api/mc/scores?channel=N`, `/api/mc/latest_alarm?channel=N`.

### Fleet & Edge deployment

`FleetPage` (stack 14) polls remote `monitor.py` daemons via `GET /api/status`. Devices are persisted in `QSettings`. Two per-device actions:
- **Einrichten** — opens `monitor.py --setup` web UI in browser
- **Training** — opens `_RemoteTrainDialog` (3 tabs: collect frames via JPEG polling → train locally → deploy model via multipart POST)

Edge export (`core/edge_export.py`): `EdgeExporter.export_quantized_onnx()` (ONNX INT8) and `export_coreml()` (macOS only, requires `coremltools`).

Docker deployment (`core/docker_generator.py`): generates `Dockerfile`, `docker-compose.yml`, `requirements_monitor.txt`, `run_monitor.sh`, `README_deploy.md`.

### Standalone monitor daemon: `monitor.py`

Runs without the GUI. Designed for headless Windows/Linux deployment. Key modes:

- **No arguments**: scans USB cameras via `_discover_cameras()`, shows interactive terminal selection (`_terminal_camera_select()`), auto-opens browser to setup wizard
- **Normal** (`--model path`): single-camera anomaly detection
- **Multi-channel** (`--channels cfg.json`): `run_monitor_multi()` with N parallel channels
- **Setup wizard** (`--setup`): web UI on `--setup-port` (default 8765) — camera preview (JPEG polling at `/setup/channels/{id}/frame.jpg`), ROI drawing, model deploy via multipart POST; no training on the daemon

The setup wizard web UI exposes `/setup/cameras` (GET) which returns the pre-scanned camera list for one-click channel creation buttons.

On macOS, `cv2.VideoCapture` for the built-in camera needs up to 60 read() calls before the first frame arrives — the `_CameraThread` warmup limit is set accordingly. Camera permission must be granted to Terminal in System Settings → Privacy & Security → Camera.

Embedded REST API (`--api-port`): `GET /api/status`, `/api/scores`, `/api/latest_alarm`, `/api/frame/<file>`, `/dashboard`. Auth via `--api-key`. MQTT publishing via `--mqtt-host`.

Minimal dependencies for monitor-only deployment: `requirements_monitor.txt` (no PySide6, no GUI).

### REST API: `api/rest_server.py`

`RestApiServer` runs in a background daemon thread (stdlib `http.server`, no extra deps). Call `set_project(project)` after load. Optional API key auth (`X-Api-Key` header); `/api/status` and `/dashboard` are always public.

### Thumbnail list: `gui/widgets/thumbnail_list.py`

`LazyThumbnailList` loads via `QThreadPool`. `filter()` accepts `label_set`, `roi_paths`, `uncertain_paths` (all `None` = show all). `update_flag(path, uncertain)` sets orange background for QA flags.

## Test fixtures

`conftest.py` provides:
- `sample_project` — 15 fake image paths, 3 labels, 6 ROIs, saved to a temp dir.
- `sample_images` — 12 real tiny PNG files (requires Pillow; skips otherwise).

Key facts that have caused test failures:
- `compute_metrics` in `core/metrics.py` takes **integer** class indices, not label strings.
- ROI removal is `project.remove_roi(path, roi_id)` — not `delete_roi`.
- `QTableWidget` row-change signal is `itemSelectionChanged`, not `currentRowChanged`.
- `QWidget.isVisible()` returns `False` when parent is not shown — use `not widget.isHidden()` instead.
- `CameraCaptureDialog` calls `list_usb_cameras()` (subprocess) in `__init__` — patch it in tests to avoid macOS subprocess crashes.
- `AnomalyDetector.load()` recreates `_ConvAutoencoder(base_ch)` if `base_ch` in checkpoint differs from current model.

## Optional dependencies

| Package | Feature |
|---------|---------|
| `optuna` | Hyperparameter search (classification + anomaly HPT) |
| `imagehash` | Perceptual duplicate detection in DatasetStatsPage |
| `scipy` | Temperature scaling calibration |
| `coremltools` | CoreML export (macOS only) |
| `paho-mqtt` | MQTT alarm publishing |
| `onnxruntime` | ONNX inference in monitor.py |
