# 📷 Kamera & Videoanalyse

> **PictureStudio v2.3.0** — Live-Aufnahme, Anomalieerkennung und Batch-Analyse mit Kamera oder Video

---

# 📷 Kamera & Videoanalyse

Live-Aufnahme von USB-/IP-Kameras, Video-Datei-Analyse, automatische Anomalieerkennung.

Öffnen über: *Datei → Kamera aufnehmen…* (`Strg+K`)

## Kameraquellen

### USB-Kamera

Kamera im Dropdown wählen → *Verbinden* klicken.**
Kamera nicht sichtbar? → *Kameras neu suchen* klicken

### IP-Kamera / Netzwerkkamera

URL eingeben und *Verbinden* klicken.

`rtsp://user:pass@192.168.1.100:554/stream` – RTSP

`http://192.168.1.100:8080/video` – HTTP-MJPEG

### Video-Datei

*Datei wählen…* → MP4, AVI, MOV, MKV, WebM.

Wiedergabe fps auf 0 = originale Geschwindigkeit.

## Aufnahme-Funktionen

Einzelbild aufnehmen** – `Leertaste`

**Burst-Aufnahme** – Anzahl + Intervall einstellen → *Burst starten*

**Live-Aufzeichnung (MP4)** – *Aufnahme starten* → laufendes MP4 wird gespeichert

## Anomalie-Erkennung (Autoencoder)

> 💡 **Funktionsprinzip:** Conv-Autoencoder lernt auf normalen Frames. Bei Anomalie steigt der Rekonstruktionsfehler (MSE) über den Schwellwert → Alarm.

### Schritt 1 – Normalframes aufnehmen

150–500 Frames empfohlen. *Aufnehmen starten* → Frames werden gesammelt.

### Schritt 2 – Autoencoder trainieren

Epochen einstellen (Standard: 40) → *Training starten*.**
Schwellwert = Mittelwert + 2,5 × Standardabweichung der Trainings-Rekonstruktionsfehler.

### Schritt 3 – Live-Erkennung

Scoring aktiv** Button aktivieren.**
Grün = Normal | Rot = Anomalie. Heatmap zeigt abweichende Bereiche.

## Batch-Analyse

Tab „📁 Batch" → Ordner oder Dateien wählen → *Batch starten* → CSV exportieren.

## Kamera-Einstellungen

Sliders für **Helligkeit, Kontrast, Sättigung, Schärfe und Belichtung** stehen an zwei Stellen zur Verfügung:

- **CameraPage (Anomalie-Erkennung):** Im linken Panel unter „Kamera-Einstellungen" (eingeklappt). Änderungen wirken live auf den laufenden Stream.
- **Aufnahme-Dialog (Bildklassifikation):** Direkt im *CameraCaptureDialog* — auch beim Kamera-Button auf der Daten-Seite verfügbar. Von der CameraPage übergebene Werte werden als Startwerte übernommen.

**Zurücksetzen** setzt alle Slider auf Neutral-Werte zurück.

## Vorverarbeitungsfilter

Der Filter-Dropdown ist ebenfalls in beiden Dialogen verfügbar (Anomalie-Erkennung *und* Bildklassifikations-Aufnahme):

- **Kein Filter** — Original-Frame
- **Graustufen** — als BGR zurückgegeben
- **Canny-Kanten** — Kantenlinien (Schwellwerte 50/150)
- **Sobel-Gradient** — Gradientenstärke in X und Y
- **Laplacian** — zweite Ableitung (feine Details)

## Hyperparameter-Suche (Anomalie-Autoencoder)

Schaltfläche **⚙ Hyperparameter-Suche…** startet eine Optuna-Studie.

Suchraum: base_ch (8/16/32), lr (1e-4 bis 1e-2), batch_size (8/16/32).

**Button-Reihenfolge im Aufnahme-Dialog:** ① Hyperparameter-Suche → ② Training starten → ③ Training stoppen

> ⚠️ `pip install optuna`
