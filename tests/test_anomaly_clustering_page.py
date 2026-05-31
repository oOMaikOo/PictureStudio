"""Smoke tests for AnomalyClusteringPage."""
import pytest


@pytest.fixture
def page(qtbot):
    from gui.pages.anomaly_clustering_page import AnomalyClusteringPage
    p = AnomalyClusteringPage()
    qtbot.addWidget(p)
    return p


def test_instantiates_without_args(page):
    assert page is not None


def test_set_project_with_sample_project(page, sample_project):
    page.set_project(sample_project)
    assert page.project is sample_project


def test_set_project_none_does_not_crash(page):
    page.set_project(None)
    assert page.project is None


def test_key_attributes_exist(page):
    assert hasattr(page, "btn_start")
    assert hasattr(page, "btn_export")
    assert hasattr(page, "spin_clusters")


def test_export_disabled_after_set_project(page, sample_project):
    page.set_project(sample_project)
    assert not page.btn_export.isEnabled()


def test_spin_clusters_range(page):
    assert page.spin_clusters.minimum() == 2
    assert page.spin_clusters.maximum() == 20
