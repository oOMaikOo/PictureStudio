"""Tests for Round 6 improvements:
- project.notes field persistence
- dashboard notes debounce timer
- ModelSearch filter in ModelsPage
- Pause-all scoring in MultiCameraPage
- Keyboard shortcuts wired up
- Shortcut tooltips on buttons
"""
from __future__ import annotations
import json
import os
import pytest


# ---------------------------------------------------------------------------
# project.notes — field exists, saves, loads
# ---------------------------------------------------------------------------

class TestProjectNotes:
    def test_notes_field_default_empty(self, sample_project):
        assert sample_project.notes == ""

    def test_notes_field_assignment(self, sample_project):
        sample_project.notes = "hello world"
        assert sample_project.notes == "hello world"

    def test_notes_persisted_in_save(self, sample_project, tmp_path):
        from core.project import Project
        sample_project.notes = "test note"
        path = str(tmp_path / "proj.json")
        sample_project.save(path)
        with open(path) as f:
            data = json.load(f)
        assert data["notes"] == "test note"

    def test_notes_loaded_from_json(self, sample_project, tmp_path):
        from core.project import Project
        sample_project.notes = "loaded note"
        path = str(tmp_path / "proj.json")
        sample_project.save(path)
        loaded = Project.load(path)
        assert loaded.notes == "loaded note"

    def test_notes_defaults_to_empty_on_missing_key(self, tmp_path):
        from core.project import Project
        path = str(tmp_path / "proj.json")
        with open(path, "w") as f:
            json.dump({"format_version": 1, "config": {}, "images": []}, f)
        proj = Project.load(path)
        assert proj.notes == ""


# ---------------------------------------------------------------------------
# DashboardPage — notes widget + debounce timer
# ---------------------------------------------------------------------------

class TestDashboardNotes:
    def test_notes_edit_exists(self, qtbot, sample_project):
        from gui.pages.dashboard_page import DashboardPage
        page = DashboardPage()
        qtbot.addWidget(page)
        page.set_project(sample_project)
        assert hasattr(page, "_notes_edit")

    def test_notes_save_timer_exists(self, qtbot):
        from gui.pages.dashboard_page import DashboardPage
        page = DashboardPage()
        qtbot.addWidget(page)
        assert hasattr(page, "_notes_save_timer")
        assert page._notes_save_timer.isSingleShot()

    def test_notes_text_synced_on_set_project(self, qtbot, sample_project):
        from gui.pages.dashboard_page import DashboardPage
        sample_project.notes = "synced note"
        page = DashboardPage()
        qtbot.addWidget(page)
        page.set_project(sample_project)
        page.refresh()
        assert page._notes_edit.toPlainText() == "synced note"

    def test_notes_written_to_project_on_change(self, qtbot, sample_project):
        from gui.pages.dashboard_page import DashboardPage
        page = DashboardPage()
        qtbot.addWidget(page)
        page.set_project(sample_project)
        page._notes_edit.setPlainText("new text")
        assert sample_project.notes == "new text"

    def test_flush_notes_saves_to_disk(self, qtbot, sample_project, tmp_path):
        from core.project import Project
        from gui.pages.dashboard_page import DashboardPage
        path = str(tmp_path / "proj.json")
        sample_project.save(path)
        sample_project.project_path = path
        page = DashboardPage()
        qtbot.addWidget(page)
        page.set_project(sample_project)
        sample_project.notes = "disk note"
        page._flush_notes_to_disk()
        loaded = Project.load(path)
        assert loaded.notes == "disk note"


# ---------------------------------------------------------------------------
# ModelsPage — search field filters table rows
# ---------------------------------------------------------------------------

class TestModelsPageSearch:
    def test_search_edit_exists(self, qtbot):
        from gui.pages.models_page import ModelsPage
        page = ModelsPage()
        qtbot.addWidget(page)
        assert hasattr(page, "_search_edit")

    def test_filter_shows_all_on_empty(self, qtbot, sample_project):
        from gui.pages.models_page import ModelsPage
        page = ModelsPage()
        qtbot.addWidget(page)
        page.set_project(sample_project)
        page._search_edit.setText("")
        visible = sum(
            not page.table.isRowHidden(r)
            for r in range(page.table.rowCount())
        )
        assert visible == page.table.rowCount()

    def test_filter_hides_non_matching_rows(self, qtbot, sample_project):
        from gui.pages.models_page import ModelsPage
        page = ModelsPage()
        qtbot.addWidget(page)
        page.set_project(sample_project)
        # inject a dummy row
        page.table.insertRow(0)
        from PySide6.QtWidgets import QTableWidgetItem
        page.table.setItem(0, 0, QTableWidgetItem("resnet50_v1"))
        page._search_edit.setText("xxxxxxxxxxxxxxxx_nomatch")
        hidden = sum(
            page.table.isRowHidden(r)
            for r in range(page.table.rowCount())
        )
        assert hidden > 0


# ---------------------------------------------------------------------------
# MultiCameraPage — pause-all scoring flag
# ---------------------------------------------------------------------------

class TestMultiCameraPause:
    def test_pause_all_btn_exists(self, qtbot):
        from gui.pages.multi_camera_page import MultiCameraPage
        page = MultiCameraPage()
        qtbot.addWidget(page)
        assert hasattr(page, "_pause_all_btn")

    def test_scoring_paused_default_false(self, qtbot):
        from gui.pages.multi_camera_page import MultiCameraPage
        page = MultiCameraPage()
        qtbot.addWidget(page)
        assert page._scoring_paused is False

    def test_pause_toggle_sets_flag(self, qtbot):
        from gui.pages.multi_camera_page import MultiCameraPage
        page = MultiCameraPage()
        qtbot.addWidget(page)
        page._on_pause_all_toggled(True)
        assert page._scoring_paused is True
        page._on_pause_all_toggled(False)
        assert page._scoring_paused is False

    def test_pause_button_text_changes(self, qtbot):
        from gui.pages.multi_camera_page import MultiCameraPage
        from utils.i18n import tr
        page = MultiCameraPage()
        qtbot.addWidget(page)
        page._on_pause_all_toggled(True)
        assert page._pause_all_btn.text() == tr("multicam.resume_scoring_btn")
        page._on_pause_all_toggled(False)
        assert page._pause_all_btn.text() == tr("multicam.pause_all_btn")

    def test_resume_btn_label_differs_from_start_all(self, qtbot):
        from gui.pages.multi_camera_page import MultiCameraPage
        from utils.i18n import tr
        page = MultiCameraPage()
        qtbot.addWidget(page)
        # The resume label must not equal the start-all label
        assert tr("multicam.resume_scoring_btn") != tr("multicam.start_all_btn")


# ---------------------------------------------------------------------------
# Keyboard shortcuts — buttons have tooltips mentioning shortcuts
# ---------------------------------------------------------------------------

class TestShortcutTooltips:
    def test_training_start_btn_has_tooltip(self, qtbot):
        from gui.pages.training_page import TrainingPage
        page = TrainingPage()
        qtbot.addWidget(page)
        tip = page.start_btn.toolTip()
        assert "Ctrl" in tip or "ctrl" in tip.lower()

    def test_training_stop_btn_has_tooltip(self, qtbot):
        from gui.pages.training_page import TrainingPage
        page = TrainingPage()
        qtbot.addWidget(page)
        tip = page.stop_btn.toolTip()
        assert "Ctrl" in tip or "ctrl" in tip.lower()

    def test_inference_classify_btn_has_tooltip(self, qtbot):
        from gui.pages.inference_page import InferencePage
        page = InferencePage()
        qtbot.addWidget(page)
        tip = page.classify_btn.toolTip()
        assert "Ctrl" in tip or "ctrl" in tip.lower()

    def test_dataset_stats_refresh_btn_has_tooltip(self, qtbot):
        from gui.pages.dataset_stats_page import DatasetStatsPage
        page = DatasetStatsPage()
        qtbot.addWidget(page)
        # Find the refresh button
        from PySide6.QtWidgets import QPushButton
        btns = page.findChildren(QPushButton)
        tips = [b.toolTip() for b in btns]
        assert any("F5" in t for t in tips)

    def test_camera_scoring_btn_has_shortcut_tooltip(self, qtbot):
        from gui.pages.camera_page import CameraPage
        from unittest.mock import patch
        with patch("gui.pages.camera_page.list_usb_cameras", return_value=[]):
            page = CameraPage()
        qtbot.addWidget(page)
        tip = page._scoring_btn.toolTip()
        assert "Leertaste" in tip or "Space" in tip


# ---------------------------------------------------------------------------
# i18n — new keys resolve to real strings, not raw keys
# ---------------------------------------------------------------------------

class TestNewI18nKeys:
    @pytest.mark.parametrize("key", [
        "inference.no_uncertain_text",
        "inference.no_uncertain_al",
        "inference.classify_btn_tip",
        "labeling.pre_no_images",
        "training.start_btn_tip",
        "training.stop_btn_tip",
        "models.delete_btn_tip",
        "camera.scoring_btn_tip",
        "multicam.resume_scoring_btn",
        "dataset_stats.refresh_tip",
        "objdetect.yaml_not_found",
    ])
    def test_key_not_returned_raw(self, key):
        from utils.i18n import tr
        result = tr(key)
        assert result != key, f"Key {key!r} was returned as-is (not translated)"
        assert result != "", f"Key {key!r} returned empty string"
