"""Backlog C/D/E: models_page, data_page, fleet RemoteTrainDialog via tr()."""
from locales.de import STRINGS as de
from locales.en import STRINGS as en

_C_KEYS = [
    "models.refresh_tip", "models.compare_label", "models.edge_section",
    "models.history_header", "models.hist.col.acc", "models.file_missing",
    "models.cal_existing", "models.coreml_success", "models.docker_created",
]
_D_KEYS = [
    "data.load_tip", "data.camera_tip", "data.video_tip", "data.analyze_tip",
    "data.export_coco_tip", "data.check_files_tip", "data.analyzing",
    "data.no_warnings", "data.check_title", "data.no_missing",
]
_E_KEYS = [
    "fleet.dl_group", "fleet.count_label", "fleet.ready", "fleet.train_group",
    "fleet.epochs_label", "fleet.connecting", "fleet.frames_loaded",
    "fleet.train_done", "fleet.download_failed", "fleet.zip_empty",
]


def test_locales_in_sync():
    assert set(de) == set(en)


def test_cde_keys_present():
    for key in _C_KEYS + _D_KEYS + _E_KEYS:
        assert key in de and key in en, key
        assert de[key].strip() and en[key].strip(), key


def test_models_page_labels(qtbot):
    from utils.i18n import init_i18n, tr
    from gui.pages.models_page import ModelsPage
    from PySide6.QtWidgets import QLabel

    init_i18n("de")
    page = ModelsPage()
    qtbot.addWidget(page)
    texts = [w.text() for w in page.findChildren(QLabel)]
    assert tr("models.compare_label") in texts
    assert tr("models.edge_section") in texts
    assert tr("models.history_header") in texts


def test_data_page_tooltips(qtbot):
    from utils.i18n import init_i18n, tr
    from gui.pages.data_page import DataPage
    from PySide6.QtWidgets import QPushButton

    init_i18n("de")
    page = DataPage()
    qtbot.addWidget(page)
    tips = {b.toolTip() for b in page.findChildren(QPushButton)}
    assert tr("data.load_tip") in tips
    assert tr("data.camera_tip") in tips
    assert tr("data.export_coco_tip") in tips


def test_fleet_remote_train_dialog_labels(qtbot):
    from utils.i18n import init_i18n, tr
    from gui.pages.fleet_page import _RemoteTrainDialog
    from PySide6.QtWidgets import QGroupBox, QLabel

    init_i18n("en")
    try:
        dlg = _RemoteTrainDialog({"name": "Cam1", "url": "http://127.0.0.1:8766", "api_key": ""})
        qtbot.addWidget(dlg)
        titles = [g.title() for g in dlg.findChildren(QGroupBox)]
        assert tr("fleet.dl_group") in titles
        assert tr("fleet.train_group") in titles
        labels = [l.text() for l in dlg.findChildren(QLabel)]
        assert tr("fleet.ready") in labels
        assert tr("fleet.model_not_trained") in labels
    finally:
        init_i18n("de")


def test_banned_literals_gone():
    import pathlib
    checks = {
        "gui/pages/models_page.py": [
            "Modell vergleichen:",
            "Kalibrierung & Edge-Deployment:",
            "Alle Trainingsläufe im Vergleich:",
            "CoreML exportiert",
        ],
        "gui/pages/data_page.py": [
            "Ordner mit Bildern auswählen.",
            "Analysiere Datensatz …",
            "✓ Keine Warnungen.",
            "Keine fehlenden Dateien.",
        ],
        "gui/pages/fleet_page.py": [
            "Frames herunterladen",
            "Modell trainieren",
            "Lade Frames herunter…",
            "ZIP enthält keine Frames.",
        ],
    }
    for path, banned in checks.items():
        src = pathlib.Path(path).read_text(encoding="utf-8")
        for s in banned:
            assert s not in src, f"{path}: {s}"
