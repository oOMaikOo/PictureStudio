"""
Unit tests for core/alarm_notifier.py → AlarmNotifier

Tests cover: no-op with empty config, cooldown enforcement, email mock,
webhook with a real local HTTP server, error paths, and config updates.
"""
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_notifier(config=None):
    from core.alarm_notifier import AlarmNotifier
    return AlarmNotifier(config)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class _WebhookCapture:
    """Tiny HTTP server that captures exactly one POST request."""

    def __init__(self):
        self.received_body: bytes = b""
        self.received_headers: dict = {}
        self.request_count: int = 0
        self._ready = threading.Event()
        self._done = threading.Event()
        self._port = _free_port()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._port}/webhook"

    def start(self):
        capture = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self_h):
                length = int(self_h.headers.get("Content-Length", 0))
                capture.received_body = self_h.rfile.read(length)
                capture.received_headers = dict(self_h.headers)
                capture.request_count += 1
                self_h.send_response(200)
                self_h.end_headers()
                capture._done.set()

            def log_message(self_h, *args):
                pass  # silence request logs in test output

        self._server = HTTPServer(("127.0.0.1", self._port), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()

    def wait_for_request(self, timeout: float = 3.0) -> bool:
        return self._done.wait(timeout)

    def stop(self):
        if self._server:
            self._server.shutdown()


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_empty_config_does_not_crash(self):
        n = _make_notifier()
        assert n is not None

    def test_none_config_does_not_crash(self):
        n = _make_notifier(None)
        assert n is not None

    def test_empty_dict_config_does_not_crash(self):
        n = _make_notifier({})
        assert n is not None


# ---------------------------------------------------------------------------
# notify() no-op when disabled
# ---------------------------------------------------------------------------

class TestNotifyNoOp:
    def test_notify_with_email_and_webhook_disabled_is_noop(self):
        """notify() should return quickly without spawning threads when both are off."""
        n = _make_notifier({"email_enabled": False, "webhook_enabled": False})
        # No exception, no hang
        n.notify(0.005, 0.001)

    def test_notify_with_empty_config_is_noop(self):
        n = _make_notifier({})
        n.notify(0.005, 0.001)

    def test_notify_no_config_is_noop(self):
        n = _make_notifier()
        n.notify(0.005, 0.001)


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_second_notify_within_cooldown_does_not_fire(self):
        """Both calls are within 60 s cooldown; only the first should trigger."""
        call_count = [0]

        n = _make_notifier({"webhook_enabled": True,
                             "webhook_url": "http://127.0.0.1:1/nonexistent",
                             "cooldown_s": 60})

        def _mock_bg(*args, **kwargs):
            call_count[0] += 1

        n._notify_bg = _mock_bg  # type: ignore[method-assign]

        n.notify(0.005, 0.001)
        n.notify(0.005, 0.001)  # should be suppressed by cooldown
        time.sleep(0.1)
        assert call_count[0] == 1

    def test_notify_after_cooldown_fires_again(self):
        """After cooldown expires the next notify() must fire."""
        call_count = [0]

        n = _make_notifier({"webhook_enabled": True,
                             "webhook_url": "http://127.0.0.1:1/nonexistent",
                             "cooldown_s": 0.1})

        def _mock_bg(*args, **kwargs):
            call_count[0] += 1

        n._notify_bg = _mock_bg  # type: ignore[method-assign]

        n.notify(0.005, 0.001)
        time.sleep(0.2)          # wait past cooldown
        n.notify(0.005, 0.001)
        time.sleep(0.1)
        assert call_count[0] == 2

    def test_notify_zero_cooldown_always_fires(self):
        call_count = [0]
        n = _make_notifier({"webhook_enabled": True,
                             "webhook_url": "http://127.0.0.1:1/nonexistent",
                             "cooldown_s": 0})

        def _mock_bg(*args, **kwargs):
            call_count[0] += 1

        n._notify_bg = _mock_bg  # type: ignore[method-assign]

        n.notify(0.005, 0.001)
        n.notify(0.005, 0.001)
        time.sleep(0.1)
        assert call_count[0] == 2


# ---------------------------------------------------------------------------
# update_config
# ---------------------------------------------------------------------------

class TestUpdateConfig:
    def test_update_config_changes_active_config(self):
        n = _make_notifier({"cooldown_s": 100})
        n.update_config({"cooldown_s": 5})
        cfg = n._get_config()
        assert cfg["cooldown_s"] == 5

    def test_update_config_replaces_old_keys(self):
        n = _make_notifier({"email_enabled": True, "cooldown_s": 30})
        n.update_config({"cooldown_s": 10})
        cfg = n._get_config()
        assert "email_enabled" not in cfg

    def test_update_config_with_empty_dict(self):
        n = _make_notifier({"cooldown_s": 30})
        n.update_config({})
        cfg = n._get_config()
        assert cfg == {}


# ---------------------------------------------------------------------------
# test_email() error path
# ---------------------------------------------------------------------------

class TestTestEmailErrorPath:
    def test_test_email_returns_false_on_wrong_host(self):
        n = _make_notifier({
            "email_enabled": True,
            "email_smtp_host": "nonexistent.invalid.host.example",
            "email_smtp_port": 587,
            "email_to": "test@example.com",
            "email_from": "sender@example.com",
        })
        ok, msg = n.test_email()
        assert ok is False
        assert len(msg) > 0

    def test_test_email_returns_false_with_no_host(self):
        n = _make_notifier({"email_enabled": True})
        ok, msg = n.test_email()
        assert ok is False
        assert msg != ""

    def test_test_email_returns_false_with_no_recipients(self):
        n = _make_notifier({
            "email_smtp_host": "smtp.example.com",
            "email_to": "",
        })
        ok, msg = n.test_email()
        assert ok is False
        assert msg != ""


# ---------------------------------------------------------------------------
# test_email() success with mock SMTP
# ---------------------------------------------------------------------------

class TestTestEmailMockSMTP:
    def test_test_email_success_with_mock_smtp(self):
        n = _make_notifier({
            "email_enabled": True,
            "email_smtp_host": "smtp.example.com",
            "email_smtp_port": 587,
            "email_to": "user@example.com",
            "email_from": "sender@example.com",
            "email_smtp_user": "user",
            "email_smtp_password": "secret",
            "email_use_tls": True,
        })
        with patch("smtplib.SMTP") as mock_smtp_cls:
            mock_server = MagicMock()
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            ok, err = n.test_email()
        assert ok is True
        assert err == ""

    def test_send_email_called_with_email_enabled_and_mock_smtp(self):
        """_send_email is invoked when notify fires with email_enabled=True."""
        n = _make_notifier({
            "email_enabled": True,
            "email_smtp_host": "smtp.example.com",
            "email_smtp_port": 587,
            "email_to": "user@example.com",
            "email_from": "sender@example.com",
            "email_use_tls": True,
            "cooldown_s": 0,
        })
        called = []

        original_send = n._send_email

        def _capturing_send(*args, **kwargs):
            called.append(True)
            original_send(*args, **kwargs)

        with patch("smtplib.SMTP") as mock_smtp_cls:
            mock_server = MagicMock()
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            n._send_email = _capturing_send  # type: ignore[method-assign]
            n.notify(0.005, 0.002)
            time.sleep(0.3)

        assert len(called) >= 1

    def test_email_subject_contains_score(self):
        """The MIME message subject must embed the anomaly score."""
        import email as _email_mod
        n = _make_notifier({
            "email_enabled": True,
            "email_smtp_host": "smtp.example.com",
            "email_smtp_port": 587,
            "email_to": "user@example.com",
            "email_from": "sender@example.com",
            "email_use_tls": False,
        })
        captured = []
        with patch("smtplib.SMTP") as mock_smtp_cls:
            mock_server = MagicMock()

            def _fake_sendmail(from_addr, to_addrs, msg_str):
                captured.append(msg_str)

            mock_server.sendmail = _fake_sendmail
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

            score = 0.00251
            n._send_email(score=score, threshold=0.001,
                          frame_path="", model_name="test",
                          cfg=n._get_config())

        assert len(captured) == 1
        # Decode the MIME message to get the plain-text subject
        parsed = _email_mod.message_from_string(captured[0])
        subject_parts = _email_mod.header.decode_header(parsed["Subject"])
        subject_decoded = "".join(
            (part.decode(enc or "utf-8") if isinstance(part, bytes) else part)
            for part, enc in subject_parts
        )
        assert "0.00251" in subject_decoded

    def test_email_body_contains_threshold(self):
        """Email body must contain the threshold value."""
        import email as _email_mod
        import base64
        n = _make_notifier({
            "email_smtp_host": "smtp.example.com",
            "email_smtp_port": 587,
            "email_to": "user@example.com",
            "email_use_tls": False,
        })
        captured = []
        with patch("smtplib.SMTP") as mock_smtp_cls:
            mock_server = MagicMock()

            def _fake_sendmail(from_addr, to_addrs, msg_str):
                captured.append(msg_str)

            mock_server.sendmail = _fake_sendmail
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

            threshold = 0.00150
            n._send_email(score=0.003, threshold=threshold,
                          frame_path="", model_name="test",
                          cfg=n._get_config())

        assert len(captured) == 1
        # Decode the MIME message body (may be base64-encoded)
        parsed = _email_mod.message_from_string(captured[0])
        if parsed.is_multipart():
            body = parsed.get_payload(0).get_payload(decode=True).decode("utf-8")
        else:
            body = parsed.get_payload(decode=True).decode("utf-8")
        # Threshold formatted with 5 decimal places → "0.00150"
        assert "0.00150" in body


# ---------------------------------------------------------------------------
# Webhook — real local HTTP server
# ---------------------------------------------------------------------------

class TestWebhookRealServer:
    def test_test_webhook_sends_correct_json_payload(self):
        capture = _WebhookCapture()
        capture.start()
        try:
            n = _make_notifier({
                "webhook_enabled": True,
                "webhook_url": capture.url,
                "webhook_method": "POST",
            })
            ok, err = n.test_webhook()
            assert capture.wait_for_request(timeout=3.0), "Webhook request not received"
            assert ok is True
            payload = json.loads(capture.received_body)
            assert "event" in payload
            assert "timestamp" in payload
            assert "score" in payload
            assert "threshold" in payload
            assert "score_pct" in payload
            assert "model" in payload
        finally:
            capture.stop()

    def test_webhook_payload_event_field_is_anomaly_alarm(self):
        capture = _WebhookCapture()
        capture.start()
        try:
            n = _make_notifier({
                "webhook_enabled": True,
                "webhook_url": capture.url,
            })
            n._send_webhook(score=0.005, threshold=0.002,
                             frame_path="", model_name="mymodel.pt",
                             cfg=n._get_config())
            assert capture.wait_for_request(timeout=3.0)
            payload = json.loads(capture.received_body)
            assert payload["event"] == "anomaly_alarm"
        finally:
            capture.stop()

    def test_webhook_payload_score_and_threshold(self):
        capture = _WebhookCapture()
        capture.start()
        try:
            n = _make_notifier({
                "webhook_enabled": True,
                "webhook_url": capture.url,
            })
            n._send_webhook(score=0.0025, threshold=0.0015,
                             frame_path="", model_name="m.pt",
                             cfg=n._get_config())
            assert capture.wait_for_request(timeout=3.0)
            payload = json.loads(capture.received_body)
            assert abs(payload["score"] - 0.0025) < 1e-6
            assert abs(payload["threshold"] - 0.0015) < 1e-6
        finally:
            capture.stop()

    def test_webhook_payload_score_pct(self):
        capture = _WebhookCapture()
        capture.start()
        try:
            n = _make_notifier({
                "webhook_enabled": True,
                "webhook_url": capture.url,
            })
            score, threshold = 0.0030, 0.0015
            expected_pct = int(score / threshold * 100)  # 200
            n._send_webhook(score=score, threshold=threshold,
                             frame_path="", model_name="m.pt",
                             cfg=n._get_config())
            assert capture.wait_for_request(timeout=3.0)
            payload = json.loads(capture.received_body)
            assert payload["score_pct"] == expected_pct
        finally:
            capture.stop()

    def test_webhook_payload_model_name(self):
        capture = _WebhookCapture()
        capture.start()
        try:
            n = _make_notifier({
                "webhook_enabled": True,
                "webhook_url": capture.url,
            })
            n._send_webhook(score=0.001, threshold=0.001,
                             frame_path="", model_name="mein_modell.pt",
                             cfg=n._get_config())
            assert capture.wait_for_request(timeout=3.0)
            payload = json.loads(capture.received_body)
            assert payload["model"] == "mein_modell.pt"
        finally:
            capture.stop()

    def test_webhook_content_type_is_json(self):
        capture = _WebhookCapture()
        capture.start()
        try:
            n = _make_notifier({
                "webhook_enabled": True,
                "webhook_url": capture.url,
            })
            n._send_webhook(score=0.001, threshold=0.001,
                             frame_path="", model_name="m.pt",
                             cfg=n._get_config())
            assert capture.wait_for_request(timeout=3.0)
            ct = capture.received_headers.get("Content-Type", "")
            assert "application/json" in ct
        finally:
            capture.stop()


# ---------------------------------------------------------------------------
# test_webhook() error paths
# ---------------------------------------------------------------------------

class TestTestWebhookErrors:
    def test_test_webhook_returns_false_on_connection_failure(self):
        # Use a port that is not listening
        n = _make_notifier({
            "webhook_enabled": True,
            "webhook_url": f"http://127.0.0.1:{_free_port()}/webhook",
        })
        ok, msg = n.test_webhook()
        assert ok is False
        assert len(msg) > 0

    def test_test_webhook_returns_false_on_unreachable_url(self):
        n = _make_notifier({
            "webhook_enabled": True,
            "webhook_url": "http://nonexistent.invalid.host.example/hook",
        })
        ok, msg = n.test_webhook()
        assert ok is False
        assert msg != ""

    def test_test_webhook_returns_false_with_no_url(self):
        n = _make_notifier({"webhook_enabled": True})
        ok, msg = n.test_webhook()
        assert ok is False
        assert msg != ""


# ---------------------------------------------------------------------------
# notify() integration with real local webhook server
# ---------------------------------------------------------------------------

class TestNotifyIntegration:
    def test_notify_with_webhook_enabled_delivers_payload(self):
        capture = _WebhookCapture()
        capture.start()
        try:
            n = _make_notifier({
                "webhook_enabled": True,
                "webhook_url": capture.url,
                "cooldown_s": 0,
            })
            n.notify(score=0.009, threshold=0.003, model_name="live.pt")
            received = capture.wait_for_request(timeout=4.0)
            assert received is True
            assert capture.request_count == 1
        finally:
            capture.stop()

    def test_notify_with_email_and_webhook_both_enabled_does_not_raise(self):
        """When email fails (bad host) and webhook also fails, notify() must not crash."""
        n = _make_notifier({
            "email_enabled": True,
            "email_smtp_host": "127.0.0.1",
            "email_smtp_port": 1,
            "email_to": "x@x.com",
            "webhook_enabled": True,
            "webhook_url": "http://127.0.0.1:1/hook",
            "cooldown_s": 0,
        })
        # Should complete without raising
        n.notify(0.005, 0.001)
        time.sleep(0.5)  # let background thread finish
