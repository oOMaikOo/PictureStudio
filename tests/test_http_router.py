"""Unit tests for core/http_router.py — the shared stdlib HTTP router."""
import http.client
import json
import socket
import time

import pytest

from core.http_router import Router, Response, RouterServer, _compile, _match


# ── pattern matching ────────────────────────────────────────────────────────

class TestMatching:
    def test_literal_exact(self):
        assert _match(_compile("/api/status"), "/api/status") == {}

    def test_literal_mismatch(self):
        assert _match(_compile("/api/status"), "/api/other") is None

    def test_param_capture(self):
        assert _match(_compile("/api/frame/{name}"), "/api/frame/x.jpg") == {"name": "x.jpg"}

    def test_param_multi_segment_does_not_match(self):
        # a single-segment param must not swallow a slash
        assert _match(_compile("/api/frame/{name}"), "/api/frame/a/b") is None

    def test_nested_params(self):
        segs = _compile("/setup/channels/{id}/roi")
        assert _match(segs, "/setup/channels/3/roi") == {"id": "3"}
        assert _match(segs, "/setup/channels/3/deploy") is None

    def test_root(self):
        assert _match(_compile("/"), "/") == {}
        assert _match(_compile("/"), "/x") is None

    def test_length_mismatch(self):
        assert _match(_compile("/a/b"), "/a") is None


# ── live server ─────────────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _build_router(api_key=None):
    r = Router(api_key=api_key, public_paths={"/api/status"}, cors=True)
    r.get("/api/status", lambda req: Response.json({"ok": True}))
    r.get("/api/echo/{name}", lambda req: Response.json({"name": req.params["name"]}))
    r.get("/page", lambda req: Response.html("<h1>hi</h1>"))
    r.get("/blob", lambda req: Response.data(b"\x00\x01", "application/octet-stream"))
    r.post("/api/sum", lambda req: Response.json({"s": sum(req.json().get("vals", []))}))
    r.delete("/api/item/{id}", lambda req: Response.json({"deleted": req.params["id"]}))
    return r


@pytest.fixture
def server():
    port = _free_port()
    srv = RouterServer(port, _build_router())
    srv.start()
    time.sleep(0.1)
    yield port
    srv.stop()
    srv.server_close()


def _req(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, dict(resp.getheaders()), data


def test_json_route(server):
    status, headers, data = _req(server, "GET", "/api/status")
    assert status == 200
    assert json.loads(data) == {"ok": True}
    assert headers.get("Access-Control-Allow-Origin") == "*"


def test_path_param(server):
    status, _, data = _req(server, "GET", "/api/echo/hello")
    assert status == 200
    assert json.loads(data)["name"] == "hello"


def test_html_route(server):
    status, headers, data = _req(server, "GET", "/page")
    assert status == 200
    assert "text/html" in headers["Content-Type"]
    assert b"<h1>hi</h1>" in data


def test_binary_route(server):
    status, headers, data = _req(server, "GET", "/blob")
    assert status == 200
    assert headers["Content-Type"] == "application/octet-stream"
    assert data == b"\x00\x01"


def test_post_json_body(server):
    body = json.dumps({"vals": [1, 2, 3]})
    status, _, data = _req(server, "POST", "/api/sum", body=body,
                           headers={"Content-Type": "application/json"})
    assert status == 200
    assert json.loads(data)["s"] == 6


def test_delete_route(server):
    status, _, data = _req(server, "DELETE", "/api/item/7")
    assert status == 200
    assert json.loads(data)["deleted"] == "7"


def test_unknown_route_404(server):
    status, _, _ = _req(server, "GET", "/nope")
    assert status == 404


def test_options_preflight(server):
    status, headers, _ = _req(server, "OPTIONS", "/api/status")
    assert status == 204
    assert headers["Access-Control-Allow-Origin"] == "*"
    assert "GET" in headers["Access-Control-Allow-Methods"]


# ── auth ────────────────────────────────────────────────────────────────────

@pytest.fixture
def auth_server():
    port = _free_port()
    srv = RouterServer(port, _build_router(api_key="secret"))
    srv.start()
    time.sleep(0.1)
    yield port
    srv.stop()
    srv.server_close()


def test_public_path_skips_auth(auth_server):
    status, _, _ = _req(auth_server, "GET", "/api/status")
    assert status == 200


def test_protected_path_requires_key(auth_server):
    status, _, _ = _req(auth_server, "GET", "/api/echo/x")
    assert status == 401


def test_valid_key_accepted(auth_server):
    status, _, _ = _req(auth_server, "GET", "/api/echo/x",
                        headers={"X-Api-Key": "secret"})
    assert status == 200


def test_bearer_token_accepted(auth_server):
    status, _, _ = _req(auth_server, "GET", "/api/echo/x",
                        headers={"Authorization": "Bearer secret"})
    assert status == 200


def test_callable_api_key():
    """api_key as a callable is evaluated per request (can change at runtime)."""
    box = {"key": ""}
    r = Router(api_key=lambda: box["key"], public_paths=set())
    r.get("/x", lambda req: Response.json({"ok": True}))
    port = _free_port()
    srv = RouterServer(port, r)
    srv.start()
    time.sleep(0.1)
    try:
        # no key configured yet → open
        assert _req(port, "GET", "/x")[0] == 200
        box["key"] = "abc"
        # now protected
        assert _req(port, "GET", "/x")[0] == 401
        assert _req(port, "GET", "/x", headers={"X-Api-Key": "abc"})[0] == 200
    finally:
        srv.stop()
        srv.server_close()
