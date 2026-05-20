# 🌐 Fleet-Management

> **PictureStudio v2.3.0** — Mehrere Monitor-Instanzen zentral überwachen

---

# 🌐 Fleet-Management

Mehrere remote `monitor.py`-Instanzen von einer zentralen Stelle überwachen.

## Gerät hinzufügen

Klicke **+ Gerät hinzufügen**.**
Name, Basis-URL (z. B. `http://192.168.1.100:8765`) und optional API-Key eingeben.

## Status prüfen

Alle aktualisieren** oder **Auto-Refresh (30 s)** aktivieren.**
Tabelle zeigt: Online/Offline, letzter Score, letzter Alarm-Zeitstempel.

## monitor.py auf Edge-Geräten starten

```
python monitor.py --model modell.onnx --api-port 8765 --api-key MEIN_SCHLÜSSEL
```

> 💡 Docker-Deployment:** Modelle-Seite → **Docker-Deployment generieren…** erzeugt Dockerfile + docker-compose.yml.
