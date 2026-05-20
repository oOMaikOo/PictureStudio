# 🔧 Fehlerbehebung

> **PictureStudio v2.3.0** — Häufige Probleme und ihre Lösungen

---

# 🔧 Fehlerbehebung

## Anwendung startet nicht

`pip install PySide6` installieren.

Linux: `apt install libxcb-cursor0`

## Training sehr langsam

Gerät auf `cuda` oder `mps` stellen.

CPU-Test: Bildgröße 128 px, Batch-Größe 8, SimpleCNN.

## ImportError: openpyxl

`pip install openpyxl`

## ImportError: paramiko

`pip install paramiko` – nur für SSH-Ferntraining

## MQTT funktioniert nicht

`pip install paho-mqtt`

Broker-Verbindung testen: `mosquitto_pub -h localhost -t test -m hello`

## Kamera wird nicht gefunden

`pip install opencv-python`

macOS: Kamera-Zugriff in Systemeinstellungen → Datenschutz → Kamera prüfen.

## SSH-Verbindung schlägt fehl

• Host, Benutzername und Key-Pfad prüfen

• `ssh-add `

• `chmod 600 ~/.ssh/id_rsa`

## Anomalie-Score immer 0

Autoencoder muss erst trainiert sein. *Scoring aktiv* muss aktiviert (grün) sein.

## Viele Fehlalarme

• Schwellwert erhöhen oder kalibrieren

• Glättung auf 5–10 Frames erhöhen

• Mehr Normalframes sammeln und neu trainieren

• ROI setzen und Bewegungsfilter aktivieren

## Projektdatei beschädigt

`projekt.bak` → `projekt.json` umbenennen

## Video-Datei öffnet nicht

Mit ffmpeg in H.264 konvertieren:

`ffmpeg -i input.avi -c:v libx264 output.mp4`
