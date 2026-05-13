# Picture Studio

Eine produktionsreife Desktop-Anwendung zur Bildannotation, Videoanalyse, CNN-Modelltraining, Anomalieerkennung und Batch-Inferenz — entwickelt mit **PySide6** und **PyTorch**.

---

## Funktionsübersicht

| Bereich | Funktionen |
|---|---|
| **Projektverwaltung** | Zwei Projekttypen (Bild / Video), versionierte JSON-Projekte, atomares Speichern, automatische Backups, Projekt-Dashboard, Bildvalidierung & Pfadkorrektur |
| **Datenverwaltung** | Ordner-Import, Videoimport mit Frame-Extraktion, Drag & Drop, Kameraaufnahme (USB/IP/RTSP), MD5-Duplikaterkennung |
| **Labeling** | Schnellzuweisung 1–9, Multi-Label-Modus, Label-Hierarchien, Undo/Redo, Audit-Trail, Pixel-Segmentierungsmasken (5 Klassen) |
| **ROI-Editor** | Rechteck, Ellipse, Polygon; Kopieren/Einfügen; Tastenkürzel; ROI-Vorlagen; Batch-Übertragung auf alle Bilder |
| **Training** | ResNet-18/50, MobileNetV2, EfficientNet-B0, SimpleCNN; Early Stopping, LR-Scheduler, Mixed Precision, Klassenausgleich (WeightedSampler), SSH-Ferntraining |
| **Anomalie-Erkennung** | Unüberwachter Conv-Autoencoder auf Normalframes; Live-Scoring, Heatmap, Bounding Box, konfigurierbarer Schwellwert, Schwellwert-Kalibrierungsdialog, Event-Log (CSV), MQTT-Alarm |
| **Modellbibliothek** | Versioniertes Registry, ONNX-Export (Opset 17), TorchScript-Export, Accuracy/F1-Vergleich, Run-History, Archivieren/Löschen |
| **Inferenz** | Batch-Inferenz, Top-K-Anzeige, Test-Time Augmentation (TTA), Ensemble-Inferenz, Semi-automatisches Labeling, Konfidenz-Farbkodierung |
| **Metriken & Berichte** | Accuracy, F1, gewichteter F1, ROC/AUC, Top-K, HTML- und Excel-Trainingsbericht, Konfusionsmatrix |
| **REST-API** | `POST /api/classify` (Pfad oder Base64-Bild), `GET /api/status`, `GET /api/labels`, `GET /api/scores`, `GET /api/events`, `GET /dashboard` — für externe Integration und Web-Monitoring |
| **24/7-Daemon** | `scripts/monitor_daemon.py` — headless, kein GUI, ONNX/PyTorch, CSV-Log, MQTT; Autostart via launchd (macOS) oder systemd (Linux) |
| **Export** | COCO JSON, YOLO TXT, CSV-Annotationen; Excel-Inferenzergebnisse (konfigurierbare Spalten) |
| **UX** | Modernes Dark-Theme (GitHub-Dark Palette), Sidebar-Navigation (gesperrt bis Projekt geladen), geführte Tour, F1-Hilfe, QSettings-Persistenz |

---

## Installation

### Voraussetzungen

- Python 3.10 oder neuer
- (Optional) CUDA-fähige GPU für schnelleres Training

### Schritte

```bash
git clone <repo-url>
cd Picture

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt

python main.py
```

> **macOS Apple Silicon:** PyTorch nutzt automatisch das MPS-Backend. Im Gerät-Dropdown `auto` oder `mps` wählen.

---

## Schritt-für-Schritt-Anleitung

---

### Anleitung A — Bildklassifikation

> Ziel: Bilder in Klassen einteilen, ein CNN trainieren und neue Bilder damit bewerten.

---

#### Schritt 1 — Bildprojekt anlegen

1. Menü **Datei → Neues Projekt** (`Strg+N`)
2. Im Dialog: Projektname eingeben, Typ **📸 Bildklassifikation** wählen
3. Speicherort wählen → Projekt wird als `.json` angelegt
4. Die Sidebar schaltet sich frei, das Dashboard öffnet sich

---

#### Schritt 2 — Bilder importieren

**Option A: Ordner importieren**
- Seite **Daten** → `Bilder laden…` → Ordner wählen
- Alle `.jpg`, `.png`, `.bmp`, `.tiff` im Ordner werden hinzugefügt

**Option B: Kamera aufnehmen**
- Menü **Datei → Kamera aufnehmen…** (`Strg+K`)
- Kamera verbinden → Einzelbilder oder Burst aufnehmen → `In Projekt übernehmen`

**Option C: Drag & Drop**
- Bilder oder Ordner direkt ins Anwendungsfenster ziehen

---

#### Schritt 3 — Klassen (Labels) definieren

1. Menü **Projekt → Labels verwalten…** (`Strg+L`)
2. `+` klicken → Name eingeben (z. B. `gut`, `defekt`, `unsicher`)
3. Farbe zuweisen
4. Wiederholen für alle Klassen → Dialog schließen

> **Empfehlung:** Mindestens 50 Bilder pro Klasse für sinnvolle Ergebnisse. Ideal: 200+.

---

#### Schritt 4 — Bilder labeln

1. Seite **Labeling** öffnen
2. Bild in der Thumbnail-Liste anklicken
3. **Schnellzuweisung:** Taste `1`–`9` drückt das entsprechende Label direkt zu
4. **Alternativ:** Label-Dropdown oben rechts verwenden
5. Navigation: `N` = nächstes Bild, `P` = vorheriges Bild

**Optional — ROI zeichnen** (wenn nur bestimmte Bildbereiche relevant sind):
- Toolbar: `R` = Rechteck, `E` = Ellipse, `G` = Polygon
- Bereich im Bild ziehen → ROI erscheint
- ROI in der Liste auswählen → Label zuweisen → `ROI-Label zuweisen`
- `ROIs dieses Bildes → alle Bilder` überträgt ROI-Positionen auf alle anderen Bilder

**Optional — Segmentierungsmaske** (für Pixelgenaue Annotation):
- Im mittleren Bereich auf Tab **🎨 Segmentierungsmaske** wechseln
- Linksklick = malen, Rechtsklick = löschen, Scroll = Zoom
- Klasse und Pinselgröße über die Toolbar wählen → `Maske speichern`

---

#### Schritt 5 — Datensatz prüfen

1. Seite **Daten** → `Dataset analysieren`
2. Auf Warnungen achten:
   - **Klassenungleichgewicht** → auf der Trainingsseite `Klassenausgleich (WeightedSampler)` aktivieren
   - **Fehlende Dateien** → Dateien wiederherstellen oder `Bildpfade korrigieren`
   - **Duplikate** → MD5-Duplikate manuell entfernen

---

#### Schritt 6 — Modell trainieren

1. Seite **Training** öffnen
2. **Architektur wählen:**
   | Modell | Empfehlung |
   |---|---|
   | ResNet-18 | Guter Ausgangspunkt, schnell |
   | MobileNetV2 | Für CPU-Deployment |
   | EfficientNet-B0 | Beste Genauigkeit |
   | SimpleCNN | Sehr schnell, kein Pretrained |
3. **Hyperparameter:**
   - Epochen: `20–50`
   - Lernrate: `0.001`
   - Batch-Größe: `32` (GPU) / `8–16` (CPU)
   - Gerät: `auto` (wählt GPU > MPS > CPU)
   - Early Stopping: `5` (stoppt automatisch bei Plateaus)
4. `Training starten` klicken
5. Kurven (Loss, Accuracy) und Konfusionsmatrix aktualisieren sich live
6. Das **beste Checkpoint** wird automatisch gespeichert

**Optional — SSH-Ferntraining auf GPU-Server:**
- Einstellungen → SSH-Profil anlegen (Host, User, Key-Pfad)
- Trainingsseite → SSH-Ferntraining aktivieren → Profil wählen → `Verbindung testen`
- Grünes Signal → `Training starten` läuft auf dem Server, Log streamt live

---

#### Schritt 7 — Trainingsergebnis prüfen

1. Nach dem Training öffnet sich automatisch die Ergebnisanzeige
2. **Wichtige Metriken:**
   - **Accuracy** — Anteil korrekt klassifizierter Bilder
   - **F1 (Macro)** — aussagekräftiger bei ungleichen Klassen
   - **Konfusionsmatrix** — zeigt wo das Modell verwechselt
3. Seite **Modelle** → Tab **📊 Run-History** zeigt alle Trainingsläufe im Vergleich
4. `Als Best markieren` setzt das Modell als Standard für Inferenz
5. **Berichte erstellen:** `HTML-Bericht erstellen…` oder `Excel-Bericht erstellen…`

---

#### Schritt 8 — Neue Bilder bewerten (Inferenz)

1. Seite **Klassifikation** öffnen
2. `Modell laden (.pth)` → Modelldatei wählen **oder** auf Seite **Modelle** → `In Inferenz laden`
3. `Ordner…` → Ordner mit neuen (unbekannten) Bildern wählen
4. **Optionen:**
   - **TTA (Test-Time Augmentation):** Spinner auf `3–5` erhöhen für stabilere Vorhersagen bei schwierigen Bildern
   - **Ensemble:** Mehrere Modelle laden (`+ Modell hinzufügen`), Vorhersagen werden gemittelt
5. `Alle Bilder klassifizieren` klicken
6. **Farbkodierung:** Grün >90% | Gelb 70–90% | Rot <70% Konfidenz
7. Tab **Niedrige Konfidenz** zeigt alle unsicheren Vorhersagen
8. **Automatisch labeln:** Mindest-Konfidenz einstellen → `Auf Projekt anwenden` übernimmt Hochkonfidenz-Ergebnisse als Labels

---

#### Schritt 9 — Ergebnisse exportieren

**Als Excel:**
1. Seite **Export** → `Ergebnisse aus letzter Inferenz laden`
2. Zieldatei wählen (neu erstellen oder anhängen)
3. Spalten konfigurieren (ein-/ausschalten, umbenennen)
4. `Excel exportieren`

**Als ONNX (für Deployment in anderen Systemen):**
1. Seite **Modelle** → Modell auswählen
2. `Als ONNX exportieren` → `.onnx`-Datei wird gespeichert (Opset 17)
3. Alternativ: `Als TorchScript exportieren` für reine PyTorch-Umgebungen

---

### Anleitung B — Videoanalyse & Anomalieerkennung

> Ziel: Einen Prozess per Kamera oder Video überwachen und grobe Abweichungen vom Normalzustand automatisch erkennen.

---

#### Schritt 1 — Videoprojekt anlegen

1. Menü **Datei → Neues Projekt** (`Strg+N`)
2. Im Dialog: Projektname eingeben, Typ **🎬 Videoanalyse & Anomalie** wählen
3. Speicherort wählen
4. Die Sidebar zeigt die Videoseiten: Dashboard, Daten, Live & Anomalie, Modelle, Export, Einstellungen

---

#### Schritt 2 — Normalzustand aufnehmen

**Option A: Video importieren (vorhandenes Material)**
1. Seite **Daten** → `Video importieren…`
2. Videodatei wählen (MP4, AVI, MOV, MKV, WebM, M4V)
3. Frame-Intervall einstellen (z. B. alle 5 Frames = ca. 6 Bilder/Sekunde bei 30 fps)
4. `Frames extrahieren` → Frames werden ins Projekt-Verzeichnis gespeichert
5. Die extrahierten Frames erscheinen automatisch im Projekt

**Option B: Live-Kamera aufnehmen**
1. Seite **Live & Anomalie** öffnen
2. Kamera-Index wählen (0 = erste USB-Kamera), `▶ Kamera starten`
3. `⚙ Aufnahme & Anomalie-Erkennung…` öffnen
4. Normalprozess vor die Kamera bringen
5. `Normalframes aufnehmen` → mindestens **100 Frames** sammeln (200–300 für stabilere Ergebnisse)
6. Kamera dabei nicht bewegen, Beleuchtung konstant halten

> **Wichtig:** Der Autoencoder lernt ausschließlich den Normalzustand. Es werden **keine Beispiele von Anomalien** benötigt.

---

#### Schritt 3 — Anomaliemodell trainieren

1. Im Kamera/Aufnahme-Dialog: `→ Trainieren` klicken
2. Standard: 40 Epochen — für grobe Abweichungen reicht das
3. Nach dem Training wird der **Schwellwert automatisch berechnet** (µ + 2,5 × σ der Trainings-Rekonstruktionsfehler)
4. Warten bis Training abgeschlossen (Fortschrittsbalken)

---

#### Schritt 4 — Schwellwert kalibrieren

1. `📊 Schwellwert kalibrieren…` klicken
2. Das Histogramm zeigt die Score-Verteilung der gesammelten Frames
3. **Vorschläge:** µ+1σ (sensitiv) bis µ+3σ (nur grobe Abweichungen)
4. Für die Überwachung grober Abweichungen: **µ+2σ oder µ+2,5σ** als Startwert
5. `Anwenden` setzt den gewählten Schwellwert

> **Faustregel:** Ist der Fehlalarm-Anteil zu hoch → Schwellwert erhöhen. Werden echte Abweichungen übersehen → Schwellwert senken.

---

#### Schritt 5 — Live-Überwachung aktivieren

1. Im Aufnahme-Dialog: `→ Live-Scoring` aktivieren
2. **Anzeige im Live-Bild:**
   - **Grüner Rahmen + normaler Score** = Prozess im Normalbereich
   - **Roter Rahmen + Alarmband** = Abweichung erkannt
   - **Heatmap** zeigt welche Bildregion auffällig ist
   - **Bounding Box** markiert den Bereich mit der größten Abweichung
3. Optional: `Anomalie-Frames automatisch speichern` zur Dokumentation aktivieren

---

#### Schritt 6 — Modell speichern & wiederverwenden

1. Im Aufnahme-Dialog: `Modell speichern` → `.pth`-Datei wählen
2. Beim nächsten Start: `Modell laden` → sofort einsatzbereit ohne Neutraining
3. **Für Deployment:** `ONNX exportieren` oder `TorchScript exportieren`
   - ONNX kann in ONNX Runtime, OpenCV DNN oder TensorRT geladen werden

---

#### Schritt 7 — Typische Einsatzszenarien

| Anwendung | Empfohlene Konfiguration |
|---|---|
| Montagelinie (Teileinspektion) | 200 Normalframes, µ+2σ, Kamera fest montiert |
| Füllstandskontrolle | 100 Normalframes, µ+1,5σ, Kontraststarke Beleuchtung |
| Oberflächenprüfung | 300+ Normalframes, µ+2,5σ, diffuses Licht |
| Positionskontrolle | 150 Normalframes, µ+2σ, ROI auf relevante Zone setzen |

---

### Anleitung C — 24/7-Monitoring ohne GUI (Daemon)

> Ziel: Einen Prozess dauerhaft überwachen, ohne die komplette Desktop-App laufen zu lassen. Der Daemon läuft headless im Hintergrund, schreibt Events in ein CSV-Log, publiziert Alarme per MQTT und stellt ein Web-Dashboard bereit.

---

#### Schritt 1 — Monitoring-Profil erstellen (in der GUI)

1. Anomaliemodell wie in Anleitung B (Schritte 1–6) trainieren und speichern
2. Im Kamera-Dialog: **"📋 Profil exportieren…"** klicken (Gruppe *Modell speichern / laden / exportieren*)
3. Speicherort wählen (z. B. `~/monitor/monitor_profile.json`)
4. Das JSON-Profil enthält alle Parameter:

```json
{
  "version": 1,
  "model_path": "/pfad/zum/autoencoder.pth",
  "model_format": "pytorch",
  "threshold": 0.023,
  "camera_source": 0,
  "save_dir": "/pfad/zu/anomalie-frames/",
  "smooth_n": 5,
  "roi": [0.1, 0.2, 0.9, 0.8],
  "mqtt": { "enabled": false, "host": "localhost", "port": 1883, "topic": "picture_studio/anomaly" },
  "scoring_interval": 3,
  "save_anomalies": true
}
```

> Profil kann auch manuell erstellt oder per Text-Editor angepasst werden.

---

#### Schritt 2 — Daemon starten

```bash
# Virtualenv aktivieren
source .venv/bin/activate

# Daemon starten
python scripts/monitor_daemon.py --profile /pfad/zum/monitor_profile.json

# Optional: Frames nicht speichern (nur Logging + MQTT)
python scripts/monitor_daemon.py --profile monitor_profile.json --no-save
```

Ausgabe im Terminal:
```
[daemon] PyTorch model loaded: /pfad/autoencoder.pth
[daemon] Camera opened: 0
[daemon] Threshold: 0.02300  |  Smooth: 5 frames
[daemon] Event log: /pfad/anomalies/anomaly_events.csv
[daemon] Running — press Ctrl+C to stop

[status] frames=150  score=0.01823 (79%)  alarms=0  streak=0
[ALARM]  2026-05-13T14:23:01 score=0.04512 (196%)  frame=anomaly_20260513_142301_0001.png
```

> Beenden mit `Ctrl+C` oder `kill -TERM <PID>` — der Daemon schließt Kamera und Log sauber.

---

#### Schritt 3 — Web-Dashboard (optional)

Der integrierte REST-Server der Desktop-App liefert unter `/dashboard` ein Live-Dashboard.

1. In der App: **Einstellungen → REST-API Server → API starten**
2. **📊 Dashboard**-Button klicken (oder Browser öffnen: `http://localhost:8765/dashboard`)
3. Das Dashboard aktualisiert sich alle 3 Sekunden automatisch:
   - Score-Verlaufsgraph (120 Frames, grün/rot je nach Alarm)
   - Aktuelle Score-Werte und Schwellwert
   - Tabelle der letzten Anomalie-Events

**API-Endpunkte für das Dashboard:**
| Endpunkt | Beschreibung |
|---|---|
| `GET /dashboard` | HTML-Dashboard (selbstenthaltend) |
| `GET /api/scores?limit=120` | Live-Score-Puffer (JSON) |
| `GET /api/events?limit=50` | Letzte Events aus dem CSV-Log (JSON) |

---

#### Schritt 4 — Autostart konfigurieren

**macOS (launchd):**
```bash
# 1. Plist anpassen (Pfade editieren)
nano scripts/autostart/de.picturestudio.monitor.plist

# 2. In LaunchDaemons installieren (systemweit, startet beim Boot)
sudo cp scripts/autostart/de.picturestudio.monitor.plist /Library/LaunchDaemons/
sudo launchctl load /Library/LaunchDaemons/de.picturestudio.monitor.plist

# Oder als Benutzer-Agent (nur bei Anmeldung)
cp scripts/autostart/de.picturestudio.monitor.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/de.picturestudio.monitor.plist

# Status prüfen
launchctl list | grep picturestudio
# Log anzeigen
tail -f /var/log/picture-monitor.log
```

**Linux (systemd):**
```bash
# 1. Service-Datei anpassen (User, WorkingDirectory, Pfade)
nano scripts/autostart/picture-monitor.service

# 2. Installieren und aktivieren
sudo cp scripts/autostart/picture-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable picture-monitor
sudo systemctl start picture-monitor

# Status und Logs
sudo systemctl status picture-monitor
sudo journalctl -u picture-monitor -f
```

---

#### Schritt 5 — ONNX-Modell für schnellere Inferenz (optional)

ONNX Runtime ist deutlich schneller als PyTorch auf CPU und benötigt keine PyTorch-Installation.

```bash
pip install onnxruntime

# Modell in der App exportieren: Kamera-Dialog → "→ ONNX"
# Profil anpassen:
```
```json
{
  "model_path": "/pfad/autoencoder.onnx",
  "model_format": "onnx"
}
```

Der Daemon wählt automatisch ONNX Runtime wenn verfügbar, sonst PyTorch als Fallback.

| Backend | Geschwindigkeit | Abhängigkeit |
|---|---|---|
| ONNX Runtime | ~3–5× schneller (CPU) | `pip install onnxruntime` |
| PyTorch | Standard-Fallback | Bereits in requirements.txt |

---

#### Übersicht: GUI vs. Daemon

| Feature | Desktop-App | Daemon |
|---|---|---|
| Live-Vorschau | ✅ | ❌ |
| Anomalie-Scoring | ✅ | ✅ |
| CSV-Event-Log | ✅ | ✅ |
| MQTT-Alarm | ✅ | ✅ |
| Web-Dashboard | ✅ (via REST-API) | ❌ (eigene Integration möglich) |
| ONNX-Inferenz | ❌ | ✅ |
| Autostart | ❌ | ✅ |
| Ressourcenverbrauch | ~500 MB RAM | ~150 MB RAM |

---

## Tastenkürzel

### Global
| Taste | Aktion |
|---|---|
| `Strg+N` | Neues Projekt |
| `Strg+O` | Projekt öffnen |
| `Strg+S` | Projekt speichern |
| `Strg+K` | Kamera-Dialog öffnen |
| `Strg+L` | Labels verwalten |
| `F1` | Hilfe zur aktuellen Seite |

### Labeling-Seite
| Taste | Aktion |
|---|---|
| `1`–`9` | Label schnell zuweisen |
| `N` / `P` | Nächstes / Vorheriges Bild |
| `R` | Rechteck-ROI |
| `E` | Ellipse-ROI |
| `G` | Polygon-ROI |
| `Esc` | Zeichnen abbrechen |
| `Entf` | Ausgewählte ROI löschen |
| `Strg+C / V` | ROI kopieren / einfügen |
| `Pfeiltasten` | ROI um 2 px verschieben |

---

## Unterstützte Architekturen

| ID | Modell | Empfehlung |
|---|---|---|
| `resnet18` | ResNet-18 | Schnell, guter Ausgangspunkt |
| `resnet50` | ResNet-50 | Höhere Kapazität |
| `mobilenet_v2` | MobileNetV2 | Effizient, gut für CPU-Deployment |
| `efficientnet_b0` | EfficientNet-B0 | Bestes Genauigkeits-/Größe-Verhältnis |
| `simple_cnn` | Eigenes 4-Block-CNN | Keine vortrainierten Gewichte; schnell für Tests |

---

## Trainingsoptionen

| Option | Standard | Beschreibung |
|---|---|---|
| Gerät | `auto` | Wählt automatisch GPU (CUDA) > MPS > CPU |
| Scheduler | `reduce_on_plateau` | Passt Lernrate bei Plateau an |
| Early Stopping | `5` | Stopp nach N Epochen ohne Verbesserung |
| Mixed Precision | aus | AMP via `torch.cuda.amp` (nur CUDA) |
| Klassenausgleich | aus | WeightedRandomSampler bei ungleichen Klassen |
| TTA-Passes | `1` | Test-Time Augmentation (Inferenz) |

---

## REST-API

Der integrierte REST-Server läuft auf `http://localhost:8765` (konfigurierbar in Einstellungen).

| Endpunkt | Methode | Beschreibung |
|---|---|---|
| `/api/status` | GET | Projektstatus und geladenes Modell |
| `/api/project` | GET | Vollständige Projektstatistik |
| `/api/labels` | GET | Liste aller definierten Klassen |
| `/api/images` | GET | Alle Bilder mit Labels (`?labeled=true/false`) |
| `/api/images/<name>` | GET | Einzelbild mit ROIs |
| `/api/images/label` | POST | Label zuweisen (`{path, label}`) |
| `/api/classify` | POST | Bild klassifizieren (`{path}` oder `{image_b64}`) |
| `/api/scores` | GET | Live-Score-Puffer für Anomalie-Dashboard |
| `/api/events` | GET | Letzte Anomalie-Events aus CSV-Log |
| `/dashboard` | GET | HTML Live-Monitoring-Dashboard |

**Beispiel-Request:**
```bash
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"path": "/pfad/zum/bild.jpg", "top_k": 3}'
```

**Beispiel-Response:**
```json
{
  "predicted_label": "gut",
  "confidence": 0.9412,
  "top_k": [
    {"label": "gut",    "prob": 0.9412},
    {"label": "unsicher","prob": 0.0421},
    {"label": "defekt", "prob": 0.0167}
  ],
  "low_confidence": false
}
```

---

## Projektstruktur

```
Picture/
├── main.py
├── requirements.txt
│
├── core/
│   ├── project.py          # Zentrales Datenmodell (ProjectConfig, Labels, Bilder, ROIs)
│   ├── dataset.py          # Analyse, Split, COCO/YOLO/CSV-Export
│   ├── training.py         # TrainingWorker + WeightedSampler + Dataset-Snapshot
│   ├── inference.py        # Inferencer: TTA, Batch, Ordner
│   ├── metrics.py          # Accuracy, F1, ROC/AUC, Top-K
│   ├── export.py           # Excel-Export (Ergebnisse + Trainingsbericht)
│   ├── model_manager.py    # Registry, ONNX- & TorchScript-Export
│   ├── anomaly_detector.py # Conv-Autoencoder: Training, Scoring, Bounding Box, ONNX/TS-Export
│   ├── remote_ssh.py       # SSHManager, build_training_bundle()
│   ├── remote_training.py  # RemoteTrainingThread (SSH-Ferntraining)
│   ├── camera.py           # USB/IP-Kamera-Thread
│   ├── audit.py            # JSONL-Audit-Trail
│   ├── anomaly_logger.py   # CSV-Event-Logger für Anomalie-Alarme
│   ├── mqtt_client.py      # MQTT-Publisher (paho-mqtt, optional)
│   ├── monitoring_profile.py # Profil-Format für Headless-Daemon
│   └── report.py           # HTML-Trainingsbericht
│
├── models/
│   └── classifier.py       # Modell-Factory, SimpleCNN, Checkpoint-I/O
│
├── api/
│   └── rest_server.py      # REST-API (stdlib http.server, kein Framework)
│
├── gui/
│   ├── main_window.py
│   ├── sidebar.py          # Navigationssidebar (gesperrt bis Projekt geladen, Bild/Video)
│   ├── theme.py            # Globales Dark-Theme (Palette + QSS)
│   ├── new_project_dialog.py  # Projekttyp-Auswahl bei Projekterstellung
│   ├── camera_capture_dialog.py
│   ├── video_import_dialog.py
│   ├── calibration_dialog.py
│   ├── pages/
│   │   ├── dashboard_page.py
│   │   ├── data_page.py
│   │   ├── labeling_page.py  # ROI-Editor + Segmentierungsmasken-Tab
│   │   ├── training_page.py
│   │   ├── models_page.py    # Modellbibliothek + Run-History
│   │   ├── inference_page.py # TTA, Ensemble, Semi-Auto-Labeling
│   │   ├── export_page.py
│   │   ├── camera_page.py
│   │   └── settings_page.py
│   └── widgets/
│       ├── roi_editor.py
│       ├── mask_editor.py    # Pixel-Segmentierungsmasken-Editor
│       └── thumbnail_list.py
│
├── scripts/
│   ├── remote_train.py     # Standalone-Skript für SSH-Ferntraining
│   ├── monitor_daemon.py   # Headless Anomalie-Monitor-Daemon (kein GUI)
│   └── autostart/
│       ├── start_monitor.sh                    # Shell-Helfer
│       ├── picture-monitor.service             # systemd (Linux)
│       └── de.picturestudio.monitor.plist      # launchd (macOS)
│
└── tests/
    ├── conftest.py
    ├── test_project.py
    ├── test_dataset.py
    ├── test_metrics.py
    ├── test_roi.py
    ├── test_export.py
    ├── test_anomaly_detector.py
    └── test_integration.py
```

---

## Fehlerbehebung

| Problem | Lösung |
|---|---|
| Anwendung startet nicht | `pip install PySide6`; Linux: `apt install libxcb-cursor0` |
| Training sehr langsam | Gerät `cuda` oder `mps` wählen; Bildgröße auf 128 px, Batch auf 16 reduzieren |
| `ImportError: openpyxl` | `pip install openpyxl` |
| `ImportError: paramiko` | `pip install paramiko` (nur SSH-Ferntraining) |
| Diagramme fehlen | `pip install matplotlib` |
| Kamera nicht erkannt | `pip install opencv-python`; andere Apps schließen die Kamera blockieren |
| Viele Fehlalarme (Anomalie) | Schwellwert erhöhen (`📊 Schwellwert kalibrieren`), Beleuchtung stabilisieren |
| Projektdatei beschädigt | `.bak`-Backup im Projektverzeichnis in `.json` umbenennen |
| Thumbnails laden langsam | Thumbnail-Größe in Einstellungen reduzieren (z. B. 60 px) |
| Daemon startet nicht | Virtualenv prüfen; `model_path` in Profil muss absoluter Pfad sein |
| Dashboard leer | REST-API in Einstellungen starten; Kamera-Dialog öffnen → Scoring aktivieren |
| ONNX-Daemon: `ImportError` | `pip install onnxruntime`; Fallback auf PyTorch ist automatisch aktiv |
| MQTT verbindet nicht | Broker-Adresse und Port prüfen; `paho-mqtt` installiert? (`pip install paho-mqtt`) |

---

## Tests ausführen

```bash
# Alle Tests
.venv/bin/python -m pytest tests/ -v

# Nur Unit-Tests (schnell, keine GPU nötig)
.venv/bin/python -m pytest tests/ -v --ignore=tests/test_integration.py

# Integrationstests (~10–30 s, trainiert ein kleines Modell)
.venv/bin/python -m pytest tests/test_integration.py -v

# Mit Coverage
pip install pytest-cov
.venv/bin/python -m pytest tests/ --cov=core --cov=models --cov-report=term-missing
```

---

## Lizenz

MIT — siehe `LICENSE` für Details.
