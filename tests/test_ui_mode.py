"""Tests for the Beginner/Expert UI mode (sidebar filter, settings, dialog)."""
import pytest


def test_ui_mode_setting_roundtrip():
    from utils.settings import AppSettings
    s = AppSettings()
    try:
        s.set_ui_mode("beginner")
        assert s.get_ui_mode() == "beginner"
        s.set_ui_mode("expert")
        assert s.get_ui_mode() == "expert"
        # unknown values normalise to expert
        s.set_ui_mode("nonsense")
        assert s.get_ui_mode() == "expert"
    finally:
        s.set_ui_mode("expert")


def test_beginner_mode_shows_only_label_train(qtbot):
    from gui.sidebar import Sidebar
    sb = Sidebar()
    qtbot.addWidget(sb)
    sb.set_ui_mode("beginner")
    stacks = sorted(i for _, i in sb._buttons)
    # dashboard, data, labeling, training, classification, settings
    assert stacks == [0, 1, 2, 3, 5, 7]


def test_expert_mode_shows_full_image_set(qtbot):
    from gui.sidebar import Sidebar
    sb = Sidebar()
    qtbot.addWidget(sb)
    sb.set_ui_mode("expert")
    sb.set_project_type("image")
    stacks = [i for _, i in sb._buttons]
    assert 16 in stacks   # live classification
    assert 5 in stacks    # inference


def test_beginner_persists_across_project_type(qtbot):
    from gui.sidebar import Sidebar
    sb = Sidebar()
    qtbot.addWidget(sb)
    sb.set_ui_mode("beginner")
    sb.set_project_type("video")   # must not re-expand the sidebar
    assert sorted(i for _, i in sb._buttons) == [0, 1, 2, 3, 5, 7]


def test_mode_dialog_default_and_choice(qtbot):
    from gui.mode_select_dialog import ModeSelectDialog
    dlg = ModeSelectDialog("beginner")
    qtbot.addWidget(dlg)
    assert dlg.selected_mode == "beginner"
    dlg._choose("expert")
    assert dlg.selected_mode == "expert"


# Stacks reachable in beginner mode — must stay in sync with sidebar/main_window.
_BEGINNER_STACKS = {0, 1, 2, 3, 5, 7}


def test_beginner_tour_filters_expert_steps():
    from gui.guide_tour import TOUR_STEPS, _EXPERT_ONLY_TITLES

    for idx in _BEGINNER_STACKS:
        raw = TOUR_STEPS[idx]
        filtered = [s for s in raw if s[0] not in _EXPERT_ONLY_TITLES]
        # never leave a beginner page with an empty tour
        assert filtered, f"page {idx} has no beginner steps"
        # filtered steps must not point to pages hidden in beginner mode
        for _title, desc, _btn in filtered:
            assert "Modelle-Seite" not in desc


def test_expert_only_titles_exist():
    # guards against typos / stale titles in the filter set
    from gui.guide_tour import TOUR_STEPS, _EXPERT_ONLY_TITLES
    all_titles = {s[0] for steps in TOUR_STEPS.values() for s in steps}
    assert _EXPERT_ONLY_TITLES <= all_titles
