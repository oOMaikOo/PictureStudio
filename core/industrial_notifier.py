"""
Industrial protocol integration: OPC-UA and Modbus TCP.
Writes a boolean coil/node when an anomaly alarm fires.
Falls back gracefully when libraries are not installed.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

log = logging.getLogger(__name__)


class OpcUaNotifier:
    """
    Sets an OPC-UA Boolean node to True on alarm, False when score drops below threshold.
    Uses asyncua (async) in a background thread with its own event loop.

    Config keys:
        url        : str  — OPC-UA server URL, e.g. "opc.tcp://192.168.1.10:4840"
        node_id    : str  — NodeId string, e.g. "ns=2;i=1001"
        namespace  : int  — namespace index (default 2)
        enabled    : bool — whether to send (default False)
    """

    def __init__(self, config: dict | None = None) -> None:
        self._url: str = ""
        self._node_id: str = ""
        self._namespace: int = 2
        self._enabled: bool = False

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._client = None  # asyncua.Client instance
        self._lock = threading.Lock()
        self._connected = False

        if config:
            self.update_config(config)

    def update_config(self, config: dict) -> None:
        """Apply new configuration values."""
        self._url = config.get("url", self._url)
        self._node_id = config.get("node_id", self._node_id)
        self._namespace = int(config.get("namespace", self._namespace))
        new_enabled = bool(config.get("enabled", self._enabled))
        if new_enabled and not self._enabled:
            self._start_loop()
        elif not new_enabled and self._enabled:
            self._stop_loop()
        self._enabled = new_enabled

    # ── background event loop ─────────────────────────────────────────────────

    def _start_loop(self) -> None:
        """Start a background thread running its own asyncio event loop."""
        if self._thread and self._thread.is_alive():
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="opcua-loop"
        )
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _stop_loop(self) -> None:
        """Stop the background event loop and disconnect the client."""
        if self._loop and self._loop.is_running():
            if self._connected and self._client:
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self._disconnect_client(), self._loop
                    )
                    future.result(timeout=5)
                except Exception:
                    pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread = None
        self._connected = False
        self._client = None

    # ── async helpers ─────────────────────────────────────────────────────────

    async def _ensure_connected(self) -> bool:
        """Connect to the OPC-UA server if not already connected."""
        try:
            from asyncua import Client
        except ImportError:
            log.warning("asyncua ist nicht installiert – OPC-UA deaktiviert.")
            return False

        if self._connected and self._client:
            return True
        try:
            client = Client(url=self._url)
            await client.connect()
            self._client = client
            self._connected = True
            return True
        except Exception as exc:
            log.error("OPC-UA Verbindungsfehler: %s", exc)
            self._connected = False
            self._client = None
            return False

    async def _disconnect_client(self) -> None:
        if self._client:
            try:
                await self._client.disconnect()
            except Exception:
                pass
        self._client = None
        self._connected = False

    async def _write_node(self, value: bool) -> None:
        try:
            from asyncua import ua
        except ImportError:
            return

        if not await self._ensure_connected():
            return
        try:
            node = self._client.get_node(self._node_id)
            await node.write_value(ua.DataValue(ua.Variant(value, ua.VariantType.Boolean)))
        except Exception as exc:
            log.error("OPC-UA Schreibfehler: %s", exc)
            self._connected = False
            self._client = None

    # ── public API ────────────────────────────────────────────────────────────

    def on_alarm(self, is_anomaly: bool, score: float, threshold: float) -> None:
        """Call on every scored frame. Sets node True on alarm, False when normal."""
        if not self._enabled:
            return
        if not self._loop or not self._loop.is_running():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._write_node(is_anomaly), self._loop)
        except Exception as exc:
            log.error("OPC-UA on_alarm Fehler: %s", exc)

    def test_connection(self) -> tuple[bool, str]:
        """Synchronous connection test. Returns (True, "") or (False, error_msg)."""
        try:
            from asyncua import Client
        except ImportError:
            return False, "asyncua ist nicht installiert (pip install asyncua)"

        async def _test() -> tuple[bool, str]:
            try:
                client = Client(url=self._url)
                await client.connect()
                await client.disconnect()
                return True, ""
            except Exception as exc:
                return False, str(exc) or type(exc).__name__

        try:
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(_test())
            loop.close()
            return result
        except Exception as exc:
            return False, str(exc)

    def disconnect(self) -> None:
        """Disconnect and stop the background thread."""
        self._stop_loop()


class ModbusNotifier:
    """
    Writes a Modbus TCP coil when an anomaly alarm fires.
    Uses pymodbus synchronous client in a background thread.

    Config keys:
        host       : str  — Modbus server IP
        port       : int  — Modbus TCP port (default 502)
        coil_addr  : int  — coil address to write (default 0)
        unit_id    : int  — Modbus unit/slave ID (default 1)
        enabled    : bool — whether to send (default False)
    """

    def __init__(self, config: dict | None = None) -> None:
        self._host: str = ""
        self._port: int = 502
        self._coil_addr: int = 0
        self._unit_id: int = 1
        self._enabled: bool = False

        if config:
            self.update_config(config)

    def update_config(self, config: dict) -> None:
        """Apply new configuration values."""
        self._host = config.get("host", self._host)
        self._port = int(config.get("port", self._port))
        self._coil_addr = int(config.get("coil_addr", self._coil_addr))
        self._unit_id = int(config.get("unit_id", self._unit_id))
        self._enabled = bool(config.get("enabled", self._enabled))

    def _write_coil_in_thread(self, value: bool) -> None:
        """Connect, write coil, disconnect — executed in a daemon thread."""
        try:
            from pymodbus.client import ModbusTcpClient
        except ImportError:
            log.warning("pymodbus ist nicht installiert – Modbus deaktiviert.")
            return

        try:
            client = ModbusTcpClient(host=self._host, port=self._port)
            connected = client.connect()
            if not connected:
                log.error("Modbus TCP: Verbindung zu %s:%d fehlgeschlagen", self._host, self._port)
                return
            client.write_coil(self._coil_addr, value, slave=self._unit_id)
            client.close()
        except Exception as exc:
            log.error("Modbus Schreibfehler: %s", exc)

    def on_alarm(self, is_anomaly: bool, score: float, threshold: float) -> None:
        """Writes coil True on alarm, False when normal. Non-blocking background thread."""
        if not self._enabled:
            return
        t = threading.Thread(
            target=self._write_coil_in_thread,
            args=(is_anomaly,),
            daemon=True,
        )
        t.start()

    def test_connection(self) -> tuple[bool, str]:
        """Try connect + read coil status. Returns (True, "") or (False, error_msg)."""
        try:
            from pymodbus.client import ModbusTcpClient
        except ImportError:
            return False, "pymodbus ist nicht installiert (pip install pymodbus)"

        try:
            client = ModbusTcpClient(host=self._host, port=self._port)
            connected = client.connect()
            if not connected:
                return False, f"Verbindung zu {self._host}:{self._port} fehlgeschlagen"
            client.read_coils(self._coil_addr, 1, slave=self._unit_id)
            client.close()
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def disconnect(self) -> None:
        """No persistent connection to close; no-op."""
        pass


class IndustrialNotifier:
    """
    Facade combining OpcUaNotifier and ModbusNotifier.
    Call on_alarm() on every scored frame; the facade forwards to active sub-notifiers.

    Config dict structure:
        opcua: { url, node_id, namespace, enabled }
        modbus: { host, port, coil_addr, unit_id, enabled }
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self._opcua = OpcUaNotifier(cfg.get("opcua"))
        self._modbus = ModbusNotifier(cfg.get("modbus"))

    def update_config(self, config: dict) -> None:
        """Push updated sub-configs to both notifiers."""
        cfg = config or {}
        if "opcua" in cfg:
            self._opcua.update_config(cfg["opcua"])
        if "modbus" in cfg:
            self._modbus.update_config(cfg["modbus"])

    def on_alarm(self, is_anomaly: bool, score: float, threshold: float) -> None:
        """Forward alarm to both sub-notifiers."""
        self._opcua.on_alarm(is_anomaly, score, threshold)
        self._modbus.on_alarm(is_anomaly, score, threshold)

    def test_opcua(self) -> tuple[bool, str]:
        """Test the OPC-UA connection."""
        return self._opcua.test_connection()

    def test_modbus(self) -> tuple[bool, str]:
        """Test the Modbus TCP connection."""
        return self._modbus.test_connection()

    def disconnect(self) -> None:
        """Disconnect both sub-notifiers."""
        self._opcua.disconnect()
        self._modbus.disconnect()
