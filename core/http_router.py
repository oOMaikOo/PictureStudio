"""
Minimal, dependency-free HTTP routing layer (stdlib ``http.server`` only).

Shared by the GUI REST server (``api/rest_server.py``) and the standalone
monitor daemon (``monitor/``) so routing, JSON, auth, CORS and OPTIONS are
implemented **once** instead of three times by hand.

Usage
-----
    router = Router(api_key=lambda: state.api_key, public_paths={"/api/status"})
    router.get("/api/status", lambda req: Response.json(state.snapshot()))
    router.get("/api/frame/{name}", serve_frame)        # req.params["name"]
    server = RouterServer(8765, router)
    server.start()

A route handler takes a :class:`Request` and returns a :class:`Response`
(or ``None`` → 404). Path patterns use ``{name}`` for a single path segment.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional, Union
from urllib.parse import parse_qs, urlparse


# ── Request / Response ──────────────────────────────────────────────────────

class Request:
    """Parsed view of one HTTP request, passed to route handlers."""

    def __init__(self, handler: BaseHTTPRequestHandler, params: dict) -> None:
        self._h = handler
        self.method: str = handler.command
        parsed = urlparse(handler.path)
        self.path: str = parsed.path.rstrip("/") or "/"
        self.query: dict = parse_qs(parsed.query)
        self.headers = handler.headers
        self.params: dict = params          # path params captured from the pattern
        self.client_address = handler.client_address
        self._body: Optional[bytes] = None

    @property
    def body(self) -> bytes:
        """Raw request body (read lazily, cached)."""
        if self._body is None:
            try:
                length = int(self.headers.get("Content-Length", 0))
            except (TypeError, ValueError):
                length = 0
            self._body = self._h.rfile.read(length) if length > 0 else b""
        return self._body

    def json(self) -> dict:
        """Parsed JSON body, or ``{}`` if empty/invalid."""
        if not self.body:
            return {}
        try:
            return json.loads(self.body)
        except Exception:
            return {}

    def content_length(self) -> int:
        try:
            return int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            return 0

    def query_int(self, name: str, default: int) -> int:
        try:
            return int(self.query.get(name, [str(default)])[0])
        except (TypeError, ValueError):
            return default

    def query_str(self, name: str, default: str = "") -> str:
        return self.query.get(name, [default])[0]


class Response:
    """An HTTP response. Use the classmethod constructors for common cases."""

    def __init__(self, status: int = 200, body: bytes = b"",
                 content_type: str = "application/json; charset=utf-8",
                 headers: Optional[dict] = None) -> None:
        self.status = status
        self.body = body
        self.content_type = content_type
        self.headers = headers or {}

    @classmethod
    def json(cls, data, status: int = 200) -> "Response":
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        return cls(status, body, "application/json; charset=utf-8")

    @classmethod
    def error(cls, message: str, status: int = 400) -> "Response":
        return cls.json({"error": message}, status)

    @classmethod
    def html(cls, text: str, status: int = 200) -> "Response":
        return cls(status, text.encode("utf-8"), "text/html; charset=utf-8")

    @classmethod
    def data(cls, body: bytes, content_type: str, status: int = 200,
             headers: Optional[dict] = None) -> "Response":
        return cls(status, body, content_type, headers)

    @classmethod
    def redirect(cls, location: str, status: int = 301) -> "Response":
        return cls(status, b"", "text/plain", {"Location": location})


# ── Pattern matching ────────────────────────────────────────────────────────

def _compile(pattern: str) -> list:
    """Split a pattern into segments; ``{name}`` segments capture one path part."""
    pattern = pattern.rstrip("/") or "/"
    segs = []
    for part in pattern.strip("/").split("/"):
        if part.startswith("{") and part.endswith("}"):
            segs.append(("param", part[1:-1]))
        else:
            segs.append(("lit", part))
    return segs


def _match(segs: list, path: str) -> Optional[dict]:
    """Return captured params if *path* matches *segs*, else None."""
    parts = [p for p in path.strip("/").split("/") if p != ""]
    # Root: pattern [("lit","")] vs path [] both represent "/"
    if segs == [("lit", "")]:
        return {} if not parts else None
    if len(parts) != len(segs):
        return None
    params = {}
    for (kind, name), value in zip(segs, parts):
        if kind == "lit":
            if name != value:
                return None
        else:
            params[name] = value
    return params


# ── Router ──────────────────────────────────────────────────────────────────

ApiKey = Union[str, Callable[[], Optional[str]], None]


class Router:
    """Maps (method, path-pattern) to handler functions.

    Parameters
    ----------
    api_key:
        Shared secret, or a callable returning it (evaluated per request so it
        can change at runtime). Falsy → no authentication.
    public_paths:
        Exact paths that never require authentication (e.g. status/dashboard).
    cors:
        When True, every response carries permissive CORS headers and OPTIONS
        is answered automatically.
    """

    def __init__(self, *, api_key: ApiKey = None,
                 public_paths=(), cors: bool = True) -> None:
        self._routes: list = []   # (method, segments, handler)
        self._api_key = api_key
        self._public = set(public_paths)
        self.cors = cors

    def add(self, method: str, pattern: str, handler: Callable[[Request], Optional[Response]]) -> None:
        self._routes.append((method.upper(), _compile(pattern), handler))

    def get(self, pattern: str, handler) -> None:
        self.add("GET", pattern, handler)

    def post(self, pattern: str, handler) -> None:
        self.add("POST", pattern, handler)

    def delete(self, pattern: str, handler) -> None:
        self.add("DELETE", pattern, handler)

    def resolve_key(self) -> Optional[str]:
        k = self._api_key
        return k() if callable(k) else k

    def is_public(self, path: str) -> bool:
        return path in self._public

    def match(self, method: str, path: str):
        """Return (handler, params) for the first matching route, else (None, None)."""
        norm = path.rstrip("/") or "/"
        for m, segs, handler in self._routes:
            if m != method:
                continue
            params = _match(segs, norm)
            if params is not None:
                return handler, params
        return None, None

    def allowed_methods(self) -> list:
        return sorted({m for m, _, _ in self._routes} | {"OPTIONS"})


def _check_key(headers, key: str) -> bool:
    provided = (
        headers.get("X-Api-Key", "")
        or headers.get("Authorization", "").removeprefix("Bearer ").strip()
    )
    return provided == key


def make_handler(router: Router):
    """Build a ``BaseHTTPRequestHandler`` subclass bound to *router*."""

    class _RouterHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args) -> None:  # silence access logs
            pass

        def _dispatch(self) -> None:
            req = Request(self, {})
            key = router.resolve_key()
            if key and not router.is_public(req.path) and not _check_key(self.headers, key):
                self._write(Response.error("Unauthorized — provide X-Api-Key header", 401))
                return
            handler, params = router.match(self.command, req.path)
            if handler is None:
                self._write(Response.error("Not found", 404))
                return
            req.params = params
            try:
                resp = handler(req)
            except Exception as exc:  # never leak a stack trace to the client
                resp = Response.error(str(exc) or type(exc).__name__, 500)
            self._write(resp if resp is not None else Response.error("Not found", 404))

        do_GET = _dispatch
        do_POST = _dispatch
        do_DELETE = _dispatch

        def do_OPTIONS(self) -> None:
            self._write(Response(204, b"", ""))

        def _write(self, resp: Response) -> None:
            self.send_response(resp.status)
            if resp.body and resp.content_type:
                self.send_header("Content-Type", resp.content_type)
            self.send_header("Content-Length", str(len(resp.body)))
            for name, value in resp.headers.items():
                self.send_header(name, value)
            if router.cors:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods",
                                 ", ".join(router.allowed_methods()))
                self.send_header("Access-Control-Allow-Headers",
                                 "Content-Type, X-Api-Key, Authorization")
            self.end_headers()
            if resp.body:
                self.wfile.write(resp.body)

    return _RouterHandler


class RouterServer:
    """Runs a :class:`Router` on a ``ThreadingHTTPServer`` daemon thread."""

    def __init__(self, port: int, router: Router, *, bind: str = "",
                 thread_name: str = "http-router") -> None:
        self._server = ThreadingHTTPServer((bind, port), make_handler(router))
        self._server.allow_reuse_address = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name=thread_name, daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()

    def server_close(self) -> None:
        self._server.server_close()
