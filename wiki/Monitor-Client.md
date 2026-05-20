# 💻 Monitor-Client

> **PictureStudio v2.3.0** — Eigenständiges CLI-Tool für den Produktionseinsatz

---

# 💻 Monitor-Client

Eigenständiges Kommandozeilen-Werkzeug für den Produktionseinsatz.
Lädt ein trainiertes Anomalie-Modell und verbindet sich automatisch mit der Kamera.

**Schnellstart****
`python monitor.py --model pfad/zum/modell.pt`

Kamera, ROI und Schwellwert werden automatisch aus den Modell-Metadaten geladen.

## Alle Optionen

| Option | Standard | Beschreibung |
|---|---|---|
| --model PFAD | – | Pfad zur .pt-Modelldatei (Pflicht) |
| --model PFAD.onnx | – | ONNX-Modell laden (nur onnxruntime nötig) |
| --camera INDEX | auto | Kamera-Index manuell überschreiben |
| --threshold WERT | aus Modell | Anomalie-Schwellwert überschreiben |
| --output VERZ | monitor_logs | Ausgabeverzeichnis für Logs und Alarm-Bilder |
| --fps FPS | 15 | Bilder pro Sekunde |
| --cooldown SEK | 10 | Mindestabstand zwischen Alarm-Saves |
| --headless | aus | Kein Fenster — nur Terminal + CSV |

## Typische Szenarien

Produktionslinie überwachen****
`python monitor.py --model modelle/linie1.pt --headless --output /var/log/anomalien`

Schwellwert anpassen**

`python monitor.py --model modelle/linie1.pt --threshold 0.0015`

> 💡 Beenden: Im Fenster-Modus Q oder ESC. Im Headless-Modus Strg+C.
