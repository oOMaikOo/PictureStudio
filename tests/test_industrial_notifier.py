"""
Tests for core/industrial_notifier.py
"""
import time
import pytest

from core.industrial_notifier import (
    IndustrialNotifier,
    OpcUaNotifier,
    ModbusNotifier,
)


# ── IndustrialNotifier basic construction ─────────────────────────────────────

def test_industrial_notifier_empty_config_no_crash():
    """IndustrialNotifier() with empty config must not raise."""
    n = IndustrialNotifier()
    assert n is not None


def test_industrial_notifier_none_config_no_crash():
    """IndustrialNotifier(None) must not raise."""
    n = IndustrialNotifier(None)
    assert n is not None


# ── on_alarm with disabled sub-notifiers ──────────────────────────────────────

def test_opcua_disabled_on_alarm_is_noop():
    """OpcUaNotifier.on_alarm() with enabled=False must be a no-op."""
    n = OpcUaNotifier({"enabled": False, "url": "opc.tcp://127.0.0.1:4840", "node_id": "ns=2;i=1"})
    # Should not raise or block
    n.on_alarm(True, 0.9, 0.5)
    n.on_alarm(False, 0.1, 0.5)


def test_modbus_disabled_on_alarm_is_noop():
    """ModbusNotifier.on_alarm() with enabled=False must be a no-op."""
    n = ModbusNotifier({"enabled": False, "host": "127.0.0.1", "port": 502})
    n.on_alarm(True, 0.9, 0.5)
    n.on_alarm(False, 0.1, 0.5)


def test_industrial_notifier_opcua_disabled_on_alarm_is_noop():
    """IndustrialNotifier.on_alarm() with opcua.enabled=False must be a no-op."""
    cfg = {"opcua": {"enabled": False}, "modbus": {"enabled": False}}
    n = IndustrialNotifier(cfg)
    n.on_alarm(True, 0.9, 0.5)


def test_industrial_notifier_modbus_disabled_on_alarm_is_noop():
    """IndustrialNotifier.on_alarm() with modbus.enabled=False must be a no-op."""
    cfg = {"opcua": {"enabled": False}, "modbus": {"enabled": False}}
    n = IndustrialNotifier(cfg)
    n.on_alarm(False, 0.1, 0.5)


# ── test_connection for unreachable hosts ──────────────────────────────────────

def test_modbus_test_connection_unreachable():
    """ModbusNotifier.test_connection() must return (False, msg) for unreachable host."""
    n = ModbusNotifier()
    n.update_config({"host": "192.0.2.1", "port": 502, "enabled": False})
    ok, msg = n.test_connection()
    assert ok is False
    assert isinstance(msg, str)
    assert len(msg) > 0


def test_opcua_test_connection_unreachable():
    """OpcUaNotifier.test_connection() must return (False, msg) for unreachable URL."""
    n = OpcUaNotifier()
    n.update_config({"url": "opc.tcp://192.0.2.1:4840", "enabled": False})
    ok, msg = n.test_connection()
    assert ok is False
    assert isinstance(msg, str)
    assert len(msg) > 0


def test_industrial_notifier_test_modbus_unreachable():
    """IndustrialNotifier.test_modbus() must return (False, msg) for wrong host."""
    cfg = {"modbus": {"host": "192.0.2.1", "port": 502, "enabled": False}}
    n = IndustrialNotifier(cfg)
    ok, msg = n.test_modbus()
    assert ok is False
    assert isinstance(msg, str)


def test_industrial_notifier_test_opcua_unreachable():
    """IndustrialNotifier.test_opcua() must return (False, msg) for wrong URL."""
    cfg = {"opcua": {"url": "opc.tcp://192.0.2.1:4840", "enabled": False}}
    n = IndustrialNotifier(cfg)
    ok, msg = n.test_opcua()
    assert ok is False
    assert isinstance(msg, str)


# ── update_config ──────────────────────────────────────────────────────────────

def test_update_config_empty_dict_no_crash():
    """update_config() with empty dict must not crash."""
    n = IndustrialNotifier()
    n.update_config({})


def test_opcua_update_config_empty_dict_no_crash():
    """OpcUaNotifier.update_config({}) must not crash."""
    n = OpcUaNotifier()
    n.update_config({})


def test_modbus_update_config_empty_dict_no_crash():
    """ModbusNotifier.update_config({}) must not crash."""
    n = ModbusNotifier()
    n.update_config({})


# ── disconnect ────────────────────────────────────────────────────────────────

def test_industrial_notifier_disconnect_not_connected():
    """IndustrialNotifier.disconnect() must not crash when not connected."""
    n = IndustrialNotifier()
    n.disconnect()  # should not raise


def test_opcua_disconnect_not_connected():
    """OpcUaNotifier.disconnect() must not crash when never connected."""
    n = OpcUaNotifier()
    n.disconnect()


def test_modbus_disconnect_not_connected():
    """ModbusNotifier.disconnect() must not crash (it's a no-op)."""
    n = ModbusNotifier()
    n.disconnect()


# ── on_alarm(is_anomaly=False) with disabled notifiers ────────────────────────

def test_on_alarm_false_disabled_is_noop():
    """on_alarm(is_anomaly=False, ...) with disabled notifiers must be a no-op."""
    n = IndustrialNotifier({
        "opcua": {"enabled": False, "url": "opc.tcp://192.0.2.1:4840"},
        "modbus": {"enabled": False, "host": "192.0.2.1"},
    })
    n.on_alarm(False, 0.1, 0.5)  # Must not raise or block


# ── library-not-installed path ────────────────────────────────────────────────

def test_modbus_test_connection_returns_tuple():
    """test_connection() must always return a (bool, str) tuple."""
    n = ModbusNotifier({"host": "192.0.2.255", "port": 1, "enabled": False})
    result = n.test_connection()
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], bool)
    assert isinstance(result[1], str)


def test_opcua_test_connection_returns_tuple():
    """test_connection() must always return a (bool, str) tuple."""
    n = OpcUaNotifier({"url": "opc.tcp://192.0.2.255:1", "enabled": False})
    result = n.test_connection()
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], bool)
    assert isinstance(result[1], str)
