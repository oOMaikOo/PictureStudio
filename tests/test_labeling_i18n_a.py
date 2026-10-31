"""Backlog A: labeling ROI/AL/sort/prelabel strings go through tr()."""
from locales.de import STRINGS as de
from locales.en import STRINGS as en

_A_KEYS = [
    "labeling.select_image_first",
    "labeling.no_rois_copy",
    "labeling.no_rois",
    "labeling.roi_apply_title",
    "labeling.roi_apply_confirm",
    "labeling.done_title",
    "labeling.roi_apply_done",
    "labeling.roi_size_title",
    "labeling.roi_size_confirm",
    "labeling.roi_size_done",
    "labeling.sort_name_asc",
    "labeling.sort_name_desc",
    "labeling.sort_unlabeled_first",
    "labeling.sort_labeled_first",
    "labeling.reset_filter_tip",
    "labeling.al_count",
    "labeling.al_suggestion",
    "labeling.al_hint_next",
    "labeling.al_all_labeled",
    "labeling.al_done_title",
    "labeling.al_done_retrain",
    "labeling.al_clear_title",
    "labeling.al_clear_confirm",
    "labeling.al_no_candidates",
    "labeling.al_bulk_confirm",
    "labeling.al_bulk_done",
    "labeling.al_next_btn",
    "labeling.al_done_btn",
    "labeling.al_accept_btn",
    "labeling.al_bulk_btn",
    "labeling.pre_analyzing",
    "labeling.pre_result_status",
    "labeling.pre_applied",
]


def test_locale_files_stay_in_sync():
    assert set(de) == set(en)


def test_backlog_a_keys_present_in_both_locales():
    for key in _A_KEYS:
        assert key in de, key
        assert key in en, key
        assert de[key].strip(), key
        assert en[key].strip(), key


def test_sort_combo_uses_translated_items(qtbot):
    from utils.i18n import init_i18n, tr
    from gui.pages.labeling_page import LabelingPage

    init_i18n("de")
    page = LabelingPage()
    qtbot.addWidget(page)
    items = [page.sort_combo.itemText(i) for i in range(page.sort_combo.count())]
    assert items == [
        tr("labeling.sort_name_asc"),
        tr("labeling.sort_name_desc"),
        tr("labeling.sort_label_asc"),
        tr("labeling.sort_unlabeled_first"),
        tr("labeling.sort_labeled_first"),
    ]


def test_al_panel_buttons_translated(qtbot):
    from utils.i18n import init_i18n, tr
    from gui.pages.labeling_page import LabelingPage

    init_i18n("en")
    try:
        page = LabelingPage()
        qtbot.addWidget(page)
        assert page._al_next_btn.text() == tr("labeling.al_next_btn")
        assert page._al_done_btn.toolTip() == tr("labeling.al_done_tip")
        assert page._al_bulk_btn.text() == tr("labeling.al_bulk_btn")
    finally:
        # conftest and many UI tests assume German strings
        init_i18n("de")


def test_roi_mixin_hardcoded_german_gone():
    """Guard: the ROI/AL dialog strings named in BACKLOG A must not remain literal."""
    import pathlib
    src = pathlib.Path("gui/pages/labeling_page.py").read_text(encoding="utf-8")
    banned = [
        "Bitte zuerst ein Bild auswählen.",
        "Das aktuelle Bild hat keine ROIs zum Kopieren.",
        "Das aktuelle Bild hat keine ROIs.",
        "Alle Queue-Bilder wurden gelabelt.",
        "Dateiname A→Z",
        "Erst ungelabelt",
    ]
    for s in banned:
        assert s not in src, s
