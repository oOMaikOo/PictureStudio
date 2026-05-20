"""Tests für gui/pages/fleet_page.py"""
from __future__ import annotations
import pytest
from gui.pages.fleet_page import FleetPage


def test_fleet_page_instantiates(qtbot):
    page = FleetPage()
    qtbot.addWidget(page)


def test_fleet_page_has_table(qtbot):
    from PySide6.QtWidgets import QTableWidget
    page = FleetPage()
    qtbot.addWidget(page)
    tables = page.findChildren(QTableWidget)
    assert tables, "Kein QTableWidget gefunden"


def test_fleet_page_set_project_does_not_crash(qtbot, sample_project):
    page = FleetPage()
    qtbot.addWidget(page)
    page.set_project(sample_project)


def test_fleet_page_set_project_none_does_not_crash(qtbot):
    page = FleetPage()
    qtbot.addWidget(page)
    page.set_project(None)


def test_fleet_page_has_add_device_button(qtbot):
    from PySide6.QtWidgets import QAbstractButton
    page = FleetPage()
    qtbot.addWidget(page)
    btns = [b for b in page.findChildren(QAbstractButton) if "hinzufügen" in b.text().lower()]
    assert btns


def test_fleet_page_devices_initially_loaded(qtbot):
    page = FleetPage()
    qtbot.addWidget(page)
    assert isinstance(page._devices, list)
