"""Tests that keep the stack numbering and its four consumers in sync.

The page index used to be hard-coded as a bare integer in MainWindow, the
sidebar, the guided tour and the help dialog. These tests fail as soon as one
of them drifts from ``gui.page_index.Page`` again.
"""
import pytest


def test_page_values_are_a_gapless_range():
    from gui.page_index import Page
    assert [p.value for p in sorted(Page)] == list(range(len(Page)))


def test_stack_order_matches_page_enum(qtbot):
    """Each Page member must address the widget it names in MainWindow.stack."""
    from gui.page_index import Page
    from gui.main_window import MainWindow

    expected = {
        Page.DASHBOARD:        "DashboardPage",
        Page.DATA:             "DataPage",
        Page.LABELING:         "LabelingPage",
        Page.TRAINING:         "TrainingPage",
        Page.MODELS:           "ModelsPage",
        Page.INFERENCE:        "InferencePage",
        Page.EXPORT:           "ExportPage",
        Page.SETTINGS:         "SettingsPage",
        Page.CAMERA:           "CameraPage",
        Page.BATCH:            "BatchInferencePage",
        Page.MULTI_CAMERA:     "MultiCameraPage",
        Page.DATASET_STATS:    "DatasetStatsPage",
        Page.VIDEO_ANNOTATION: "VideoAnnotationPage",
        Page.FLEET:            "FleetPage",
        Page.DATA_DRIFT:       "DataDriftPage",
        Page.ANOMALY_TRAINING: "AnomalyTrainingPage",
        Page.LIVE_CLASSIFY:    "LiveClassificationPage",
    }
    assert set(expected) == set(Page), "expected-map and Page have drifted apart"

    win = MainWindow()
    qtbot.addWidget(win)
    assert win.stack.count() == len(Page)
    for page, cls_name in expected.items():
        assert type(win.stack.widget(page)).__name__ == cls_name


def test_sidebar_lists_use_page_members():
    from gui.page_index import Page
    from gui.sidebar import _IMAGE_PAGES, _VIDEO_PAGES, _BEGINNER_PAGES

    for nav in (_IMAGE_PAGES, _VIDEO_PAGES, _BEGINNER_PAGES):
        for _key, _icon, idx in nav:
            # None marks a section header
            assert idx is None or isinstance(idx, Page)


def test_every_page_has_a_tour_and_a_help_section():
    from gui.page_index import Page
    from gui.guide_tour import TOUR_STEPS
    from gui.help_dialog import PAGE_TO_SECTION, SECTIONS, CONTENT

    assert set(TOUR_STEPS) == set(Page)
    assert set(PAGE_TO_SECTION) == set(Page)

    section_ids = {sid for sid, _icon, _label in SECTIONS}
    for page, section_id in PAGE_TO_SECTION.items():
        assert section_id in section_ids, f"{page.name} points at a missing section"
        assert section_id in CONTENT, f"{page.name} points at a section without content"


def test_beginner_allowed_is_a_subset_of_beginner_sidebar():
    """Every page the beginner sidebar offers must survive the navigation lock."""
    from gui.page_index import Page
    from gui.main_window import MainWindow
    from gui.sidebar import _BEGINNER_PAGES

    offered = {idx for _k, _i, idx in _BEGINNER_PAGES if idx is not None}
    assert offered <= set(MainWindow._BEGINNER_ALLOWED)


def test_wizard_navigates_to_real_pages():
    from gui.page_index import Page
    from gui.quick_start_wizard import _IMAGE_STEPS, _VIDEO_STEPS

    for steps in (_IMAGE_STEPS, _VIDEO_STEPS):
        for step in steps:
            idx = step.get("stack_idx")
            assert idx is None or isinstance(idx, Page)
