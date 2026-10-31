"""Backlog B: inference_page UI strings go through tr()."""
from locales.de import STRINGS as de
from locales.en import STRINGS as en

_B_KEYS = [
    "inference.control_group",
    "inference.ensemble_info",
    "inference.ensemble_empty",
    "inference.recursive_tip",
    "inference.tta_tip",
    "inference.filter_all",
    "inference.al_info",
    "inference.al_btn_tip",
    "inference.semi_auto_info",
    "inference.rename_tip",
    "inference.col_filename",
    "inference.col_pred_label",
    "inference.col_confidence",
    "inference.gradcam_tip",
    "inference.roi_fallback_title",
    "inference.result_count",
    "inference.rename_confirm",
]


def test_locale_files_stay_in_sync():
    assert set(de) == set(en)


def test_backlog_b_keys_present():
    for key in _B_KEYS:
        assert key in de, key
        assert key in en, key


def test_control_group_and_headers_translated(qtbot):
    from utils.i18n import init_i18n, tr
    from gui.pages.inference_page import InferencePage

    init_i18n("de")
    page = InferencePage()
    qtbot.addWidget(page)
    from PySide6.QtWidgets import QGroupBox
    titles = [g.title() for g in page.findChildren(QGroupBox)]
    assert tr("inference.control_group") in titles
    headers = [
        page.table.horizontalHeaderItem(i).text()
        for i in range(page.table.columnCount())
    ]
    assert headers[0] == tr("inference.col_filename")
    assert headers[1] == tr("inference.col_pred_label")
    assert headers[2] == tr("inference.col_confidence")
    assert page.label_filter_combo.itemText(0) == tr("inference.filter_all")


def test_filter_all_uses_index_not_german_literal(qtbot):
    """Filter must treat index 0 as 'all' so EN 'All' still works."""
    from utils.i18n import init_i18n, tr
    from gui.pages.inference_page import InferencePage

    init_i18n("en")
    try:
        page = InferencePage()
        qtbot.addWidget(page)
        assert page.label_filter_combo.itemText(0) == tr("inference.filter_all")
        assert page.label_filter_combo.itemText(0) != "Alle"
        page._all_results = [
            {
                "filename": "a.jpg", "path": "/tmp/a.jpg",
                "predicted_label": "gut", "confidence": 0.95,
                "top_k": [{"label": "gut", "prob": 0.95}],
                "all_probs": {}, "low_confidence": False, "error": None,
            }
        ]
        page.label_filter_combo.blockSignals(True)
        page.label_filter_combo.clear()
        page.label_filter_combo.addItem(tr("inference.filter_all"))
        page.label_filter_combo.addItem("gut")
        page.label_filter_combo.setCurrentIndex(0)
        page.label_filter_combo.blockSignals(False)
        page._apply_filter()
        assert len(page._filtered) == 1
    finally:
        init_i18n("de")


def test_hardcoded_german_gone():
    import pathlib
    src = pathlib.Path("gui/pages/inference_page.py").read_text(encoding="utf-8")
    for s in (
        "Inferenz-Steuerung",
        "Vorhergesagtes Label",
        "(kein Ensemble)",
        "Bitte erst ein Modell laden.",
        "ROI-Fallback aktiv",
        "Umbenennen abgeschlossen",
    ):
        assert s not in src, s
