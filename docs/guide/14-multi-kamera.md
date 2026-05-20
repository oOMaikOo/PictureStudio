# 📹 Multi-Kamera

> **PictureStudio v2.3.0** — Bis zu 9 Kamerakanäle gleichzeitig überwachen

---

# 📹 Multi-Kamera-Monitoring

Überwache **1–9 Kamera-Quellen gleichzeitig**, jede mit eigenem Modell und ROI.

## Anzahl Kanäle festlegen

**Kanäle-Selector (Toolbar oben)****
Drehfeld *Kanäle: [2]* – Bereich 1–9, Standard 2.

Bei Änderung werden alle laufenden Kanäle gestoppt.

## Grid und Paginierung

Bis zu 4 Kanäle im 2×2-Raster pro Seite.

Bei >4 Kanälen erscheinen ◀ Vorherige** und **Nächste ▶** Buttons.

## Kanal einrichten und starten

**1 – Kanal konfigurieren****
*⚙ Konfigurieren* → Kamera (USB-Index) und Modell (.pth oder .onnx) wählen.

2 – Kanal starten****
*▶ Starten* oder *Alle starten*.

## REST-API – Per-Kanal-Endpunkte

| Methode | Endpunkt | Beschreibung |
|---|---|---|
| GET | /api/mc/channels | Zusammenfassung aller Kanäle |
| GET | /api/mc/scores?channel=N | Score-Puffer für Kanal N |
| GET | /api/mc/latest_alarm?channel=N | Letztes Alarm-Event für Kanal N |

> 💡 Alarm-JPEG-Pfad:**
 `monitor_logs/multi_cam/mc_chN_YYYYMMDDTHHMMSSZ.jpg`

> ⚠️ Kanalzahl ändern stoppt alle Kanäle — konfigurierte Einstellungen bleiben erhalten.
