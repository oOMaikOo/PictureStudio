"""
Alarm notification: send email (SMTP) and/or HTTP webhook when anomaly is detected.
Uses only stdlib — no extra dependencies required.
"""
import json
import logging
import os
import smtplib
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

log = logging.getLogger("ImageLabelingStudio.alarm_notifier")

_MAX_ATTACHMENT_BYTES = 2 * 1024 * 1024  # 2 MB


class AlarmNotifier:
    """
    Sends anomaly alarm notifications via SMTP email and/or HTTP webhook.

    Config dict keys (all optional with safe defaults):
        email_enabled      : bool   — send email on alarm
        email_to           : str    — comma-separated recipient addresses
        email_from         : str    — sender address
        email_smtp_host    : str    — SMTP server hostname
        email_smtp_port    : int    — SMTP port (default 587)
        email_smtp_user    : str    — SMTP login username
        email_smtp_password: str    — SMTP login password
        email_use_tls      : bool   — use STARTTLS (default True)
        webhook_enabled    : bool   — send HTTP POST on alarm
        webhook_url        : str    — full URL for webhook
        webhook_method     : str    — "POST" or "GET" (default "POST")
        cooldown_s         : float  — minimum seconds between notifications (default 60)
    """

    def __init__(self, config: dict | None = None):
        self._lock = threading.Lock()
        self._last_sent: dict[str, float] = {}
        self._config: dict = {}
        if config:
            self.update_config(config)

    # ------------------------------------------------------------------ public

    def update_config(self, config: dict) -> None:
        """Replace the active configuration with *config*."""
        with self._lock:
            self._config = dict(config)

    def notify(self, score: float, threshold: float,
               frame_path: str = "", model_name: str = "",
               source: str = "") -> None:
        """Fire email + webhook in a background thread. Respects cooldown.

        ``source`` is an optional human-readable label for the alarm origin
        (e.g. ``"Kanal 2"``). The cooldown is tracked **per source**, so an
        alarm on one camera channel never suppresses another channel's alarm.
        """
        cfg = self._get_config()
        email_enabled = cfg.get("email_enabled", False)
        webhook_enabled = cfg.get("webhook_enabled", False)

        if not email_enabled and not webhook_enabled:
            return

        cooldown = float(cfg.get("cooldown_s", 60))
        now = time.monotonic()
        with self._lock:
            if now - self._last_sent.get(source, 0.0) < cooldown:
                return
            self._last_sent[source] = now

        t = threading.Thread(
            target=self._notify_bg,
            args=(score, threshold, frame_path, model_name, cfg, source),
            daemon=True,
        )
        t.start()

    def test_email(self) -> tuple[bool, str]:
        """Send a test email. Returns (success, error_message)."""
        cfg = self._get_config()
        try:
            self._send_email(
                score=0.0,
                threshold=0.0,
                frame_path="",
                model_name="test",
                cfg=cfg,
                subject_override="[PictureStudio] Test-E-Mail",
                body_override="Dies ist eine Test-E-Mail von PictureStudio.",
            )
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def test_webhook(self) -> tuple[bool, str]:
        """Send a test webhook POST. Returns (success, error_message)."""
        cfg = self._get_config()
        try:
            self._send_webhook(
                score=0.0,
                threshold=0.0,
                frame_path="",
                model_name="test",
                cfg=cfg,
            )
            return True, ""
        except Exception as exc:
            return False, str(exc)

    # ------------------------------------------------------------------ internal

    def _get_config(self) -> dict:
        with self._lock:
            return dict(self._config)

    def _notify_bg(self, score, threshold, frame_path, model_name, cfg, source="") -> None:
        """Background thread: send email and/or webhook."""
        if cfg.get("email_enabled", False):
            try:
                self._send_email(score, threshold, frame_path, model_name, cfg, source=source)
            except Exception as exc:
                log.warning("E-Mail-Alarm fehlgeschlagen: %s", exc)

        if cfg.get("webhook_enabled", False):
            try:
                self._send_webhook(score, threshold, frame_path, model_name, cfg, source=source)
            except Exception as exc:
                log.warning("Webhook-Alarm fehlgeschlagen: %s", exc)

    def _send_email(self, score, threshold, frame_path, model_name, cfg,
                    subject_override: str = "", body_override: str = "",
                    source: str = "") -> None:
        """Internal: builds MIME email with optional JPEG attachment and sends via SMTP."""
        smtp_host = cfg.get("email_smtp_host", "")
        smtp_port = int(cfg.get("email_smtp_port", 587))
        smtp_user = cfg.get("email_smtp_user", "")
        smtp_password = cfg.get("email_smtp_password", "")
        use_tls = cfg.get("email_use_tls", True)
        from_addr = cfg.get("email_from", smtp_user or "picturestudio@localhost")
        to_raw = cfg.get("email_to", "")
        to_addrs = [a.strip() for a in to_raw.split(",") if a.strip()]

        if not smtp_host:
            raise ValueError("email_smtp_host ist nicht konfiguriert")
        if not to_addrs:
            raise ValueError("email_to ist nicht konfiguriert")

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        frame_file = os.path.basename(frame_path) if frame_path else ""
        pct = int(score / threshold * 100) if threshold > 0 else 0

        subject = subject_override or (
            f"[PictureStudio] Anomalie erkannt"
            + (f" – {source}" if source else "")
            + f" – Score {score:.5f}"
        )
        body = body_override or (
            f"PictureStudio Alarm\n"
            f"===================\n"
            + (f"Quelle      : {source}\n" if source else "")
            + f"Zeitstempel : {ts}\n"
            f"Score       : {score:.5f}\n"
            f"Schwellwert : {threshold:.5f}\n"
            f"Score %     : {pct}%\n"
            f"Modell      : {model_name}\n"
            f"Frame-Datei : {frame_file}\n"
        )

        # Build MIME message
        has_attachment = (
            bool(frame_path)
            and os.path.isfile(frame_path)
            and os.path.getsize(frame_path) < _MAX_ATTACHMENT_BYTES
        )
        if has_attachment:
            msg = MIMEMultipart()
            msg.attach(MIMEText(body, "plain", "utf-8"))
            with open(frame_path, "rb") as fh:
                data = fh.read()
            part = MIMEBase("image", "jpeg")
            part.set_payload(data)
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                "attachment",
                filename=frame_file or "alarm.jpg",
            )
            msg.attach(part)
        else:
            msg = MIMEText(body, "plain", "utf-8")

        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = ", ".join(to_addrs)

        import socket
        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
                if use_tls:
                    server.starttls()
                if smtp_user and smtp_password:
                    server.login(smtp_user, smtp_password)
                server.sendmail(from_addr, to_addrs, msg.as_string())
        except socket.timeout as exc:
            raise RuntimeError(
                f"SMTP-Timeout beim Verbinden mit {smtp_host}:{smtp_port}"
            ) from exc

    @staticmethod
    def _is_teams_url(url: str) -> bool:
        """True for Microsoft Teams *Incoming Webhook* connector URLs."""
        u = url.lower()
        return "webhook.office.com" in u or "office365.com" in u or "outlook.office.com" in u

    @staticmethod
    def _build_teams_card(score, threshold, pct, model_name, frame_file, ts,
                          source: str = "") -> dict:
        """
        Build a legacy *MessageCard* payload accepted by the classic Teams
        Incoming Webhook connector. A plain JSON body is rejected with HTTP 400
        ("Summary or Text is required"), so ``summary`` must always be present.
        """
        facts = []
        if source:
            facts.append({"name": "Quelle", "value": source})
        facts += [
            {"name": "Zeitstempel", "value": ts},
            {"name": "Score", "value": f"{score:.5f}"},
            {"name": "Schwellwert", "value": f"{threshold:.5f}"},
            {"name": "Score %", "value": f"{pct}%"},
        ]
        if model_name:
            facts.append({"name": "Modell", "value": model_name})
        if frame_file:
            facts.append({"name": "Frame", "value": frame_file})
        title = "⚠ Anomalie erkannt" + (f" – {source}" if source else "")
        return {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": "E74C3C",
            "summary": "PictureStudio Anomalie-Alarm",
            "title": title,
            "sections": [{"facts": facts, "markdown": False}],
        }

    def _send_webhook(self, score, threshold, frame_path, model_name, cfg,
                      source: str = "") -> None:
        """Internal: sends JSON payload via HTTP POST/GET.

        Teams Incoming Webhook URLs get a MessageCard payload (required by the
        connector); all other URLs receive the generic event JSON.
        """
        url = cfg.get("webhook_url", "")
        method = cfg.get("webhook_method", "POST").upper()

        if not url:
            raise ValueError("webhook_url ist nicht konfiguriert")

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        frame_file = os.path.basename(frame_path) if frame_path else ""
        pct = int(score / threshold * 100) if threshold > 0 else 0

        if self._is_teams_url(url):
            payload = self._build_teams_card(
                score, threshold, pct, model_name, frame_file, ts, source
            )
        else:
            payload = {
                "event": "anomaly_alarm",
                "timestamp": ts,
                "source": source,
                "score": round(score, 7),
                "threshold": round(threshold, 7),
                "score_pct": pct,
                "model": model_name,
                "frame_file": frame_file,
            }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data if method == "POST" else None,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
