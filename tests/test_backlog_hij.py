"""Backlog H/I/J: ground-truth CSV, dataset-stats worker, pattern labeling."""
import csv
import os


def test_export_ground_truth_csv(sample_project, tmp_path):
    from core.export import export_ground_truth_csv

    path = sample_project.images[0]
    sample_project.set_label_flag(path, True, "maybe")
    out = tmp_path / "gt.csv"
    export_ground_truth_csv(sample_project, str(out))

    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    assert set(rows[0].keys()) == {"image_path", "label", "uncertain"}
    by_path = {r["image_path"]: r for r in rows}
    assert by_path[path]["uncertain"] == "1"
    assert by_path[path]["label"] == sample_project.get_image_label(path)


def test_data_page_has_gt_export_button(qtbot):
    from utils.i18n import init_i18n, tr
    from gui.pages.data_page import DataPage
    from PySide6.QtWidgets import QPushButton

    init_i18n("de")
    page = DataPage()
    qtbot.addWidget(page)
    labels = [b.text() for b in page.findChildren(QPushButton)]
    assert tr("data.export_gt_btn") in labels


def test_dataset_stats_worker_sizes(sample_images, qtbot):
    from gui.pages.dataset_stats_page import DatasetStatsWorker

    # sample_images → ([(path, label), ...], tmp_dir)
    paths = [p for p, _lbl in sample_images[0]]
    worker = DatasetStatsWorker(paths, "sizes")
    results = []

    def _done(kind, payload):
        results.append((kind, payload))

    worker.finished.connect(_done)
    worker.start()
    qtbot.waitUntil(lambda: bool(results), timeout=5000)
    kind, payload = results[0]
    assert kind == "sizes"
    assert payload.get("ok") is True
    assert payload["n"] >= 1


def test_dataset_stats_page_uses_worker(qtbot, sample_project, monkeypatch, tmp_path):
    from gui.pages.dataset_stats_page import DatasetStatsPage, DatasetStatsWorker

    # Worker only starts when at least one path exists on disk.
    real = tmp_path / "real.jpg"
    real.write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xd9"
    )
    sample_project.images[0] = str(real)

    started = []

    class FakeWorker(DatasetStatsWorker):
        def start(self):
            started.append(self._kind)
            # Don't actually run — just record
            self.finished.emit("sizes", {"ok": True, "avg_w": 10, "avg_h": 10, "n": 1})

    monkeypatch.setattr(
        "gui.pages.dataset_stats_page.DatasetStatsWorker", FakeWorker
    )
    page = DatasetStatsPage()
    qtbot.addWidget(page)
    page.set_project(sample_project)
    assert "sizes" in started


def test_apply_label_by_pattern(qtbot, sample_project, tmp_path):
    from utils.i18n import init_i18n
    from gui.pages.labeling_page import LabelingPage
    from PySide6.QtWidgets import QMessageBox

    init_i18n("de")
    # Give images names that match a pattern
    # sample_project uses img_000.jpg etc. in tmp_dir
    page = LabelingPage()
    qtbot.addWidget(page)
    page.set_project(sample_project)

    page._pattern_edit.setText("img_00[0-2].jpg")
    # pick first real label
    labels = list(sample_project.labels.keys())
    page._pattern_label_combo.setCurrentText(labels[0])

    # Auto-accept confirmation dialogs
    monkey_answers = {"n": 0}

    def _auto_yes(*args, **kwargs):
        monkey_answers["n"] += 1
        return QMessageBox.Yes

    # Patch question + information
    import gui.pages.labeling_page as lp
    original_question = QMessageBox.question
    original_info = QMessageBox.information
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
    try:
        page._apply_label_by_pattern()
    finally:
        QMessageBox.question = original_question
        QMessageBox.information = original_info

    matched = [
        p for p in sample_project.images
        if os.path.basename(p) in ("img_000.jpg", "img_001.jpg", "img_002.jpg")
    ]
    assert matched
    for p in matched:
        assert sample_project.get_image_label(p) == labels[0]


def test_locale_keys_hij():
    from locales.de import STRINGS as de
    from locales.en import STRINGS as en
    keys = [
        "data.export_gt_btn", "data.msg.gt_saved",
        "dataset_stats.size_sample", "dataset_stats.dup_group_row",
        "labeling.pattern_group", "labeling.pattern_confirm",
    ]
    assert set(de) == set(en)
    for k in keys:
        assert k in de and k in en
