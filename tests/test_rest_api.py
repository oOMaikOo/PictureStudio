"""
Unit tests for api/rest_server.py → RestApiServer

Uses stdlib urllib.request to hit real HTTP endpoints; no third-party
HTTP client libraries required. The server is started on an OS-assigned
free port.

NOTE: All endpoints except /api/status require a project to be set —
      the handler returns 503 "No project loaded" otherwise.
      Fixtures that need those endpoints pre-load a Project instance.
"""
import json
import socket
import time
import urllib.error
import urllib.request

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    """Ask the OS for an unused TCP port."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _get(url: str, timeout: float = 5.0):
    """GET *url* and return (status_code, headers, body_bytes)."""
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def _get_json(url: str, timeout: float = 5.0) -> dict:
    _, _, body = _get(url, timeout)
    return json.loads(body)


def _reset_handler_state():
    """Reset all shared _ProjectHandler class-level state between tests."""
    from api.rest_server import _ProjectHandler
    _ProjectHandler.project = None
    _ProjectHandler.score_buffer = []
    _ProjectHandler.latest_alarm = {}
    _ProjectHandler.event_log_path = ""
    _ProjectHandler.alarm_frame_dir = ""
    _ProjectHandler.request_count = 0


def _make_project(tmp_path):
    """Create a minimal Project with labels, images and ROIs."""
    from core.project import Project
    p = Project()
    p.config.name = "TestProjekt"
    p.add_label("gut",      "#2ECC71")
    p.add_label("schlecht", "#E74C3C")
    p.add_label("neutral",  "#3498DB")
    for i in range(15):
        lbl = ["gut", "schlecht", "neutral"][i % 3]
        path = str(tmp_path / f"img_{i:03d}.jpg")
        p.add_image(path)
        p.set_image_label(path, lbl)
    return p


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bare_server():
    """
    Start a RestApiServer with NO project; stop it after the test.
    Only /api/status works without a project.
    """
    _reset_handler_state()
    port = _free_port()
    from api.rest_server import RestApiServer
    srv = RestApiServer()
    srv.start(port=port)
    time.sleep(0.15)
    yield srv
    srv.stop()


@pytest.fixture
def server(tmp_path):
    """Start a RestApiServer with a minimal project pre-loaded."""
    _reset_handler_state()
    port = _free_port()
    from api.rest_server import RestApiServer
    srv = RestApiServer()
    srv.start(port=port)
    srv.set_project(_make_project(tmp_path))
    time.sleep(0.15)
    yield srv
    srv.stop()


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

class TestServerLifecycle:
    def test_server_starts_and_is_running(self, bare_server):
        assert bare_server.is_running is True

    def test_port_property_is_integer(self, bare_server):
        assert isinstance(bare_server.port, int)
        assert bare_server.port > 0

    def test_stop_sets_is_running_false(self):
        _reset_handler_state()
        from api.rest_server import RestApiServer
        port = _free_port()
        srv = RestApiServer()
        srv.start(port=port)
        time.sleep(0.1)
        assert srv.is_running is True
        srv.stop()
        assert srv.is_running is False


# ---------------------------------------------------------------------------
# GET /api/status  (works without project)
# ---------------------------------------------------------------------------

class TestStatusEndpoint:
    def test_status_returns_200(self, bare_server):
        status, _, _ = _get(f"http://localhost:{bare_server.port}/api/status")
        assert status == 200

    def test_status_body_is_valid_json(self, bare_server):
        data = _get_json(f"http://localhost:{bare_server.port}/api/status")
        assert isinstance(data, dict)

    def test_status_contains_running_status(self, bare_server):
        data = _get_json(f"http://localhost:{bare_server.port}/api/status")
        assert data.get("status") == "running"

    def test_status_contains_version_key(self, bare_server):
        data = _get_json(f"http://localhost:{bare_server.port}/api/status")
        assert "version" in data

    def test_status_version_matches_app_version(self, bare_server):
        from utils.config import APP_VERSION
        data = _get_json(f"http://localhost:{bare_server.port}/api/status")
        assert data["version"] == APP_VERSION

    def test_status_cors_header_present(self, bare_server):
        _, headers, _ = _get(f"http://localhost:{bare_server.port}/api/status")
        assert headers.get("Access-Control-Allow-Origin") == "*"

    def test_status_shows_project_name_when_project_set(self, server):
        data = _get_json(f"http://localhost:{server.port}/api/status")
        assert data.get("project") == "TestProjekt"


# ---------------------------------------------------------------------------
# CORS headers
# ---------------------------------------------------------------------------

class TestCorsHeaders:
    def test_status_endpoint_has_cors_header(self, bare_server):
        _, headers, _ = _get(f"http://localhost:{bare_server.port}/api/status")
        assert headers.get("Access-Control-Allow-Origin") == "*"

    def test_scores_endpoint_has_cors_header(self, server):
        _, headers, _ = _get(f"http://localhost:{server.port}/api/scores")
        assert headers.get("Access-Control-Allow-Origin") == "*"

    def test_labels_endpoint_has_cors_header(self, server):
        _, headers, _ = _get(f"http://localhost:{server.port}/api/labels")
        assert headers.get("Access-Control-Allow-Origin") == "*"

    def test_latest_alarm_endpoint_has_cors_header(self, server):
        _, headers, _ = _get(f"http://localhost:{server.port}/api/latest_alarm")
        assert headers.get("Access-Control-Allow-Origin") == "*"


# ---------------------------------------------------------------------------
# GET /api/scores
# ---------------------------------------------------------------------------

class TestScoresEndpoint:
    def test_scores_returns_empty_list_initially(self, server):
        from api.rest_server import _ProjectHandler
        _ProjectHandler.score_buffer = []
        data = _get_json(f"http://localhost:{server.port}/api/scores")
        assert data["scores"] == []
        assert data["count"] == 0

    def test_push_score_appears_in_scores_endpoint(self, server):
        from api.rest_server import _ProjectHandler
        _ProjectHandler.score_buffer = []
        server.push_score(0.123, 0.5)
        data = _get_json(f"http://localhost:{server.port}/api/scores")
        assert data["count"] >= 1
        last = data["scores"][-1]
        assert abs(last["score"] - 0.123) < 1e-4
        assert abs(last["threshold"] - 0.5) < 1e-4

    def test_push_score_alarm_flag_true_when_score_exceeds_threshold(self, server):
        from api.rest_server import _ProjectHandler
        _ProjectHandler.score_buffer = []
        server.push_score(0.9, 0.5)
        data = _get_json(f"http://localhost:{server.port}/api/scores")
        last = data["scores"][-1]
        assert last["alarm"] is True

    def test_push_score_alarm_flag_false_when_score_below_threshold(self, server):
        from api.rest_server import _ProjectHandler
        _ProjectHandler.score_buffer = []
        server.push_score(0.1, 0.5)
        data = _get_json(f"http://localhost:{server.port}/api/scores")
        last = data["scores"][-1]
        assert last["alarm"] is False

    def test_score_buffer_capped_at_500(self, server):
        from api.rest_server import _ProjectHandler
        _ProjectHandler.score_buffer = []
        for i in range(600):
            server.push_score(float(i) / 600, 0.5)
        data = _get_json(f"http://localhost:{server.port}/api/scores?limit=1000")
        assert data["count"] <= 500


# ---------------------------------------------------------------------------
# GET /api/latest_alarm
# ---------------------------------------------------------------------------

class TestLatestAlarmEndpoint:
    def test_latest_alarm_returns_empty_dict_before_any_alarm(self, server):
        from api.rest_server import _ProjectHandler
        _ProjectHandler.latest_alarm = {}
        data = _get_json(f"http://localhost:{server.port}/api/latest_alarm")
        assert data == {}

    def test_push_latest_alarm_updates_endpoint(self, server):
        server.push_latest_alarm("/tmp/alarm_001.jpg", 0.95, 0.5)
        data = _get_json(f"http://localhost:{server.port}/api/latest_alarm")
        assert abs(data["score"] - 0.95) < 1e-4
        assert abs(data["threshold"] - 0.5) < 1e-4
        assert "ts" in data

    def test_push_latest_alarm_stores_frame_filename(self, server):
        server.push_latest_alarm("/tmp/alarm_frame_42.jpg", 0.8, 0.5)
        data = _get_json(f"http://localhost:{server.port}/api/latest_alarm")
        assert data.get("frame_filename") == "alarm_frame_42.jpg"


# ---------------------------------------------------------------------------
# GET /dashboard
# ---------------------------------------------------------------------------

class TestDashboardEndpoint:
    def test_dashboard_returns_200(self, server):
        status, _, _ = _get(f"http://localhost:{server.port}/dashboard")
        assert status == 200

    def test_dashboard_content_type_is_html(self, server):
        _, headers, _ = _get(f"http://localhost:{server.port}/dashboard")
        content_type = headers.get("Content-Type", "")
        assert "text/html" in content_type

    def test_dashboard_body_is_non_empty_html(self, server):
        _, _, body = _get(f"http://localhost:{server.port}/dashboard")
        assert len(body) > 100
        assert b"<html" in body.lower()


# ---------------------------------------------------------------------------
# GET /api/project
# ---------------------------------------------------------------------------

class TestProjectEndpoint:
    def test_project_returns_200(self, server):
        status, _, _ = _get(f"http://localhost:{server.port}/api/project")
        assert status == 200

    def test_project_contains_name(self, server):
        data = _get_json(f"http://localhost:{server.port}/api/project")
        assert "name" in data

    def test_project_contains_label_list(self, server):
        data = _get_json(f"http://localhost:{server.port}/api/project")
        assert "labels" in data
        assert isinstance(data["labels"], list)

    def test_project_contains_image_counts(self, server):
        data = _get_json(f"http://localhost:{server.port}/api/project")
        assert "total_images" in data
        assert "labeled_images" in data

    def test_project_total_images_matches_added_images(self, server):
        data = _get_json(f"http://localhost:{server.port}/api/project")
        assert data["total_images"] == 15


# ---------------------------------------------------------------------------
# GET /api/labels
# ---------------------------------------------------------------------------

class TestLabelsEndpoint:
    def test_labels_returns_list_of_label_objects(self, server):
        data = _get_json(f"http://localhost:{server.port}/api/labels")
        assert "labels" in data
        assert isinstance(data["labels"], list)
        assert data["count"] == len(data["labels"])

    def test_labels_include_name_and_color(self, server):
        data = _get_json(f"http://localhost:{server.port}/api/labels")
        for label in data["labels"]:
            assert "name" in label
            assert "color" in label

    def test_three_labels_present_in_fixture_project(self, server):
        data = _get_json(f"http://localhost:{server.port}/api/labels")
        assert data["count"] == 3


# ---------------------------------------------------------------------------
# 404 for unknown endpoints (requires project; the 404 path is after project check)
# ---------------------------------------------------------------------------

class TestUnknownEndpoints:
    def test_unknown_endpoint_returns_404(self, server):
        status, _, _ = _get(f"http://localhost:{server.port}/api/does_not_exist")
        assert status == 404
