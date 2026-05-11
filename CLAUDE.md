# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the application
source .venv/bin/activate
python main.py

# Run all tests
.venv/bin/python -m pytest tests/ -v

# Run a single test class or function
.venv/bin/python -m pytest tests/test_project.py::TestQAFlags -v
.venv/bin/python -m pytest tests/test_project.py::TestSaveLoad::test_load_roundtrip -v

# Quick syntax check on a file
python3 -c "import ast; ast.parse(open('gui/pages/labeling_page.py').read()); print('OK')"
```

Integration tests (`test_integration.py`) train a small model on 12 synthetic images on CPU and take ~10–30 seconds. All others run in milliseconds.

## Architecture

### Central data model: `core/project.py`

`Project` is the single source of truth. All pages receive it via `set_project(project, audit=None)`. Key data attributes:

- `images: List[str]` — ordered list of absolute paths
- `labels: Dict[str, Dict]` — `{name: {color, description, parent}}`
- `image_labels: Dict[str, str]` — primary label per image
- `image_multi_labels: Dict[str, List[str]]` — multi-label mode
- `rois: Dict[str, List[Dict]]` — ROIs keyed by image path
- `image_label_flags: Dict[str, Dict]` — QA uncertain flags `{uncertain, comment}`
- `active_learning_queue: List[Dict]` — AL review queue
- `config.multi_label: bool` — single vs multi-label mode

Project JSON uses atomic writes (temp file + `os.replace()`). `Project.load(path)` is a classmethod.

### GUI: `gui/main_window.py`

`MainWindow` holds a `QStackedWidget` with 8 pages. On project load, `_load_project()` calls `set_project()` on every page in sequence. Label changes go through a `QDialog` (launched from the menu) that emits `labels_changed` → `labeling_page.on_labels_changed()`.

### Labeling mutations — undo/redo pattern

All label/ROI changes in `LabelingPage` go through `QUndoStack` via command objects in `gui/labeling_commands.py`. The pattern is always:

1. A public method (e.g. `_assign_label_direct`) pushes a `QUndoCommand` subclass.
2. The command's `redo()`/`undo()` call a `_do_*` method on the page (e.g. `_do_set_image_label`).
3. The `_do_*` method mutates `Project` and updates the UI (thumbnail, combo, stats).

Available commands: `SetImageLabelCommand`, `BulkSetImageLabelCommand`, `SetMultiLabelsCommand`, `SetLabelFlagCommand`, `AddROICommand`, `DeleteROICommand`, `AssignROILabelCommand`, `MoveROICommand`.

### Training pipeline

`TrainingPage` wraps `TrainingWorker` (a plain class, not a QThread) inside `TrainingThread(QThread)`. Signals: `progress(epoch, total, train_loss, val_loss, train_acc, val_acc)`, `log_msg(str)`, `finished(dict)`, `error(str)`.

Multi-label mode is detected at two places: `create_datasets()` in `core/dataset.py` dispatches to `create_multi_label_datasets()`, and `TrainingWorker.run()` checks `project.config.multi_label` to switch between `CrossEntropyLoss` and `BCEWithLogitsLoss`.

SSH remote training uses `RemoteTrainingThread` (`core/remote_training.py`), which zips images via `build_training_bundle()` (`core/remote_ssh.py`), uploads them, runs `scripts/remote_train.py` on the server, streams structured log lines, then downloads the checkpoint. `scripts/remote_train.py` is self-contained with no local imports.

### Thumbnail list: `gui/widgets/thumbnail_list.py`

`LazyThumbnailList` loads thumbnails via `QThreadPool`. `filter()` accepts `label_set`, `roi_paths`, and `uncertain_paths` — all `None` means "show all". `update_flag(path, uncertain)` sets an orange background for QA-flagged images.

### REST API: `api/rest_server.py`

`RestApiServer` runs in a background daemon thread (stdlib `http.server`, no extra deps). Call `set_project(project)` to update its project reference after load. All responses include CORS headers.

## Test fixtures

`conftest.py` provides:
- `sample_project` — 15 fake image paths, 3 labels, 6 ROIs, all already saved to a temp dir.
- `sample_images` — 12 real tiny PNG files (requires Pillow; skips otherwise).

Key facts that have caused test failures before:
- `compute_metrics` in `core/metrics.py` takes **integer** class indices, not label strings.
- ROI removal is `project.remove_roi(path, roi_id)` — not `delete_roi`.
- `QTableWidget` row-change signal is `itemSelectionChanged`, not `currentRowChanged`.
