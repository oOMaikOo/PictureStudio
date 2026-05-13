"""
MQTT client for publishing anomaly alarm events to an MQTT broker.
Uses paho-mqtt if available; silently disabled otherwise.
Designed to be called from any thread (publish is thread-safe in paho).
"""
import json
import threading
from datetime import datetime
from typing import Optional

try:
    import paho.mqtt.client as mqtt
    HAS_MQTT = True
except ImportError:
    HAS_MQTT = False


class MQTTAlarmClient:
    """
    Connects to an MQTT broker and publishes JSON payloads on anomaly events.

    JSON payload:
        {
          "timestamp": "2026-05-12T14:23:01.123",
          "score": 0.045123,
          "threshold": 0.02,
          "score_pct": 225,
          "alarm": true,
          "frame_path": "/path/to/frame.png"
        }
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 1883,
        topic: str = "picture_studio/anomaly",
        client_id: str = "picture_studio",
        username: str = "",
        password: str = "",
        qos: int = 0,
    ):
        self._host = host
        self._port = port
        self._topic = topic
        self._qos = qos
        self._connected = False
        self._client: Optional[object] = None
        self._lock = threading.Lock()
        self._last_error: str = ""

        if not HAS_MQTT:
            self._last_error = "paho-mqtt nicht installiert (pip install paho-mqtt)"
            return

        self._client = mqtt.Client(client_id=client_id)
        if username:
            self._client.username_pw_set(username, password)
        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect

    # ------------------------------------------------------------------ public

    @property
    def available(self) -> bool:
        return HAS_MQTT

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_error(self) -> str:
        return self._last_error

    def connect(self) -> bool:
        """Start connection (non-blocking). Returns False if paho not available."""
        if not HAS_MQTT or self._client is None:
            return False
        try:
            self._client.connect_async(self._host, self._port, keepalive=60)
            self._client.loop_start()
            return True
        except Exception as exc:
            self._last_error = str(exc)
            return False

    def disconnect(self) -> None:
        if self._client and HAS_MQTT:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
        self._connected = False

    def publish_alarm(
        self,
        score: float,
        threshold: float,
        frame_path: str = "",
        alarm: bool = True,
    ) -> bool:
        """Publish one alarm event. Thread-safe. Returns True on success."""
        if not self._connected or self._client is None:
            return False
        pct = int(score / threshold * 100) if threshold > 0 else 0
        payload = json.dumps({
            "timestamp":  datetime.now().isoformat(timespec="milliseconds"),
            "score":      round(score, 6),
            "threshold":  round(threshold, 6),
            "score_pct":  pct,
            "alarm":      alarm,
            "frame_path": frame_path,
        })
        with self._lock:
            try:
                self._client.publish(self._topic, payload, qos=self._qos)
                return True
            except Exception as exc:
                self._last_error = str(exc)
                return False

    # ------------------------------------------------------------------ callbacks

    def _on_connect(self, client, userdata, flags, rc) -> None:
        self._connected = rc == 0
        if rc != 0:
            self._last_error = f"Verbindungsfehler (rc={rc})"

    def _on_disconnect(self, client, userdata, rc) -> None:
        self._connected = False
