# Image Labeling Studio

Eine produktionsreife Desktop-Anwendung zur Bildannotation, ROI-Definition, CNN-Modelltraining und Batch-Inferenz — entwickelt mit **PySide6** und **PyTorch**.

---

## Funktionsübersicht

| Bereich | Funktionen |
|---|---|
| **Projektverwaltung** | Versionierte JSON-Projekte, atomares Speichern, automatische Backups, Projekt-Dashboard, Bildvalidierung & Pfadkorrektur |
| **Kameraaufnahme** | USB- & IP/RTSP-Kamera Live-Vorschau, Einzel- & Burst-Aufnahme, optionaler Zeitstempel (Vorschau + dauerhaft in PNG eingebrannt) |
| **Anomalie-Erkennung** | Unüberwachter Conv-Autoencoder, trainiert auf Normalframes; Live-Rekonstruktionsfehler-Scoring, konfigurierbarer Schwellwert, Alarm-Banner, automatisches Speichern von Anomalie-Frames |
| **ROI-Editor** | Rechteck, Ellipse, Polygon; Kopieren/Einfügen; Tastenkürzel; Label-Schnellzuweisung (1–9); Begrenzungsprüfung; ROI-Vorlagen |
| **Labeling** | Label-Hierarchien (Multi-Label), Statistiken, Label-Filter, Review-Modus, Änderungsprotokoll via Audit-Trail |
| **Datensatzanalyse** | Format-/Größenstatistiken, Erkennung fehlender Dateien, MD5-Duplikaterkennung, Klassenungleichgewichts-Warnungen; COCO / YOLO / CSV-Export |
| **Training** | ResNet18/50, MobileNetV2, EfficientNet-B0, SimpleCNN; Early Stopping, LR-Scheduler, Mixed Precision, GPU/CPU/MPS-Auswahl, Training von Checkpoint fortsetzen |
| **SSH-Ferntraining** | Verbindungsprofile, Live-Log-Streaming, conda/venv-Unterstützung |
| **Modellbibliothek** | Versioniertes Modell-Registry, ONNX-Export, Accuracy/F1-Vergleich, Archivieren/Löschen |
| **Metriken & Berichte** | Accuracy, F1, gewichteter F1, ROC/AUC (binär), Top-K-Accuracy, HTML-Trainingsbericht, Excel-Trainingsbericht |
| **Inferenz** | Batch-Inferenz mit Top-3-Anzeige, Konfidenz-Farbkodierung, Niedrig-Konfidenz-Tab, Label-/Konfidenz-Filter |
| **Excel-Export** | Benutzerdefiniertes Spalten-Mapping (aktivieren/deaktivieren + umbenennen), Anhängen/Überschreiben-Modus, formatierte Kopfzeilen, rote Markierung unsicherer Vorhersagen |
| **UX** | 8-seitige Sidebar-Navigation, Dunkel-/Hell-Theme, QSettings-Persistenz, Lazy-Thumbnail-Laden, Absturzberichte |
| **Tests** | Unit-Tests (Projekt, Datensatz, Metriken, ROI, Export, Anomalie-Erkennung) + Integrationstests (Train → Infer Pipeline) |

---

## Installation

### Voraussetzungen

- Python 3.10 oder neuer
- (Optional) CUDA-fähige GPU für schnelleres Training

### Schritte

```bash
git clone <repo-url>
cd Picture

# Virtuelle Umgebung erstellen
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Abhängigkeiten installieren
pip install -r requirements.txt

# Anwendung starten
python main.py
```

> **macOS-Hinweis:** PyTorch nutzt auf Apple Silicon automatisch das MPS-Backend. Im Gerät-Dropdown auf der Trainingsseite *auto* oder *mps* wählen.

---

## Projektstruktur

```
Picture/
├── main.py                        # Einstiegspunkt
├── requirements.txt
│
├── core/
│   ├── project.py                 # Zentrales Datenmodell (Labels, Bilder, ROIs)
│   ├── dataset.py                 # Analyse, Split, COCO/YOLO/CSV-Export
│   ├── training.py                # TrainingWorker (QThread) + EarlyStopping
│   ├── inference.py               # Inferencer: Einzelbild & Ordner-Batch
│   ├── metrics.py                 # Accuracy, F1, ROC/AUC, Top-K
│   ├── export.py                  # Excel-Export (Ergebnisse + Trainingsbericht)
│   ├── model_manager.py           # Modell-Registry + ONNX-Export
│   ├── camera.py                  # USB/IP-Kamera-Thread + Frame-Hilfsfunktionen
│   ├── anomaly_detector.py        # Conv-Autoencoder: auf Normalframes trainieren, Live-Frames bewerten
│   ├── audit.py                   # JSONL-Audit-Trail
│   └── report.py                  # HTML-Trainingsbericht-Generator
│
├── models/
│   └── classifier.py              # Modell-Factory + SimpleCNN + Checkpoint-I/O
│
├── gui/
│   ├── main_window.py             # MainWindow mit Sidebar + QStackedWidget
│   ├── sidebar.py                 # Navigations-Sidebar (8 Seiten)
│   ├── camera_capture_dialog.py   # Kamera Live-Vorschau + Aufnahme-Dialog
│   ├── help_dialog.py             # Integrierter Hilfe-Browser
│   ├── guide_tour.py              # Schrittweise geführte Tour als Overlay
│   ├── pages/
│   │   ├── dashboard_page.py      # Projektstatistik-Übersicht
│   │   ├── data_page.py           # Datensatzanalyse + Export
│   │   ├── labeling_page.py       # Thumbnail-Liste + ROI-Editor
│   │   ├── training_page.py       # Trainingskonfiguration + Fortschrittskurven
│   │   ├── models_page.py         # Modellbibliothek-Tabelle
│   │   ├── inference_page.py      # Batch-Inferenz + Niedrig-Konfidenz-Tab
│   │   ├── export_page.py         # Benutzerdefinierter Excel-Export
│   │   └── settings_page.py       # Theme, Autosave, SSH-Profile
│   └── widgets/
│       ├── roi_editor.py          # QGraphicsView ROI-Editor (Rect/Ellipse/Polygon)
│       ├── thumbnail_list.py      # Lazy-Loading QListWidget
│       └── charts.py              # Trainingskurven + Konfusionsmatrix
│
├── utils/
│   ├── config.py                  # App-Konstanten, Standardwerte
│   ├── logging_utils.py           # Datei- und Konsolen-Logging
│   ├── reproducibility.py         # Seed-Setzung, Software-Versionierung
│   └── settings.py                # QSettings-Wrapper (AppSettings)
│
└── tests/
    ├── conftest.py                # pytest-Fixtures (sample_project, sample_images)
    ├── test_project.py            # Unit: Labels, Bilder, ROIs, Speichern/Laden, Backup
    ├── test_dataset.py            # Unit: Analyse, Splits, Duplikate, Exporte
    ├── test_metrics.py            # Unit: Accuracy, F1, ROC/AUC, Top-K
    ├── test_roi.py                # Unit: ROI CRUD, Serialisierung, Vorlagen
    ├── test_export.py             # Unit: Excel-Spalten-Mapping, Anhängen-Modus
    ├── test_anomaly_detector.py   # Unit: Sammlung, Training, Scoring, Persistenz
    └── test_integration.py        # Integration: Train → Checkpoint → Infer
```

---

## Schnellstart

1. **Neues Projekt** — `Datei → Neues Projekt` (`Strg+N`), Namen vergeben.
2. **Bilder laden** — Seite **Daten** → *Bilder laden*, Ordner wählen — **oder** Seite **Labeling** → *Ordner laden…* — **oder** direkt von der Kamera aufnehmen mit `Datei → Kamera aufnehmen…` (`Strg+K`).
3. **Labels definieren** — Auf der Seite **Labeling** Labels anlegen (Name + Farbe).
4. **Bilder labeln** — Bild in der Thumbnail-Liste anklicken, `1–9` für Schnellzuweisung drücken oder das Label-Dropdown nutzen.
5. **ROIs zeichnen** — ROI-Toolbar verwenden (`R` = Rechteck, `E` = Ellipse, `G` = Polygon). Löschen mit **Entf**, Kopieren/Einfügen mit **Strg+C / Strg+V**.
6. **Datensatz analysieren** — Seite **Daten** → *Analyse starten*: fehlende Dateien, Duplikate, Klassenungleichgewicht prüfen.
7. **Training** — Seite **Training** → Architektur, Epochen, Lernrate konfigurieren → *Training starten*.
8. **Metriken prüfen** — Trainingskurven und Konfusionsmatrix aktualisieren sich live. Nach dem Training HTML- oder Excel-Bericht exportieren.
9. **Neue Bilder klassifizieren** — Seite **Klassifikation** → Modell wählen → *Klassifizieren*.
10. **Ergebnisse exportieren** — Seite **Export** → Spalten zuordnen → *Excel exportieren*.

---

## Kameraaufnahme

Öffnen über `Datei → Kamera aufnehmen…` (`Strg+K`).

| Funktion | Beschreibung |
|---|---|
| **USB-Kamera** | Automatisch erkannt; aus Dropdown wählen, *Verbinden* klicken |
| **IP- / RTSP-Kamera** | Stream-URL eingeben (rtsp://, http://); unterstützt MJPEG und RTSP |
| **Einzelaufnahme** | *Bild aufnehmen* klicken oder `Leertaste` drücken |
| **Burst-Aufnahme** | Anzahl + Intervall einstellen → *Burst starten* |
| **Zeitstempel-Overlay** | Datum/Uhrzeit im Live-Bild anzeigen (Toggle, beeinflusst gespeicherte Datei nicht) |
| **Zeitstempel einbrennen** | Datum/Uhrzeit dauerhaft in die gespeicherte PNG rendern (`JJJJ-MM-TT HH:MM:SS`, unten links) |
| **Anomalie-Erkennung** | Unüberwachter Conv-Autoencoder; auf Normalframes trainieren, jeden Frame live bewerten, Alarm bei Rekonstruktionsfehler-Spitze |

Aufgenommene Bilder erscheinen in der Liste im Dialog; *In Projekt übernehmen* klicken um sie hinzuzufügen.

> **Voraussetzung:** `pip install opencv-python` — PyTorch ist für das Training bereits erforderlich.

### Ablauf Anomalie-Erkennung

```
1. Kamera verbinden → normalen Prozess ablaufen lassen
2. "Normalframes aufnehmen" → 100–300 Frames sammeln
3. "Training starten" → Autoencoder trainiert ausschließlich auf Normalframes
   Schwellwert = Mittelwert + 2,5 × Std-Abw. der Trainings-Rekonstruktionsfehler (automatisch gesetzt)
4. Checkbox "Aktiv" → Live-Scoring beginnt (jeder 3. Frame, CPU-schonend)
   Grüne Score-Anzeige + normaler Rahmen  →  normaler Frame
   Roter Banner + roter Rahmen            →  Anomalie erkannt
5. Optional: "Anomalie-Frames automatisch speichern" zur Dokumentation
6. Trainiertes Modell als .pth speichern/laden für Wiederverwendung
```

---

## ROI-Editor Tastenkürzel

| Taste | Aktion |
|---|---|
| `R` | Rechteck-Modus |
| `E` | Ellipse-Modus |
| `G` | Polygon-Modus |
| `Esc` | Zeichnen abbrechen |
| `Entf` | Ausgewählte ROI löschen |
| `Strg+C` | ROI kopieren |
| `Strg+V` | ROI einfügen |
| `Pfeiltasten` | ROI um 2 px verschieben |
| `1`–`9` | Label schnell zuweisen |
| `N` / `P` | Nächstes / Vorheriges Bild (Labeling-Seite) |

---

## Unterstützte Architekturen

| ID | Modell | Hinweise |
|---|---|---|
| `resnet18` | ResNet-18 | Schnell, guter Ausgangspunkt |
| `resnet50` | ResNet-50 | Höhere Kapazität |
| `mobilenet_v2` | MobileNetV2 | Effizient, gut für CPU |
| `efficientnet_b0` | EfficientNet-B0 | Starkes Genauigkeits-/Größe-Verhältnis |
| `simple_cnn` | Eigenes 4-Block-CNN | Keine vortrainierten Gewichte; schnell für CPU-Tests |

Alle Transfer-Learning-Modelle verwenden standardmäßig ImageNet-Vortrainingsgewichte (bei sehr spezifischen Datensätzen *Pretrained* deaktivieren).

---

## Trainingsoptionen

| Option | Beschreibung |
|---|---|
| **Gerät** | `auto` / `cpu` / `cuda` / `mps` |
| **Scheduler** | `none`, `reduce_on_plateau`, `cosine`, `step` |
| **Early-Stopping-Geduld** | Stopp nach N Epochen ohne Verbesserung der Validation (`0` = deaktiviert) |
| **Mixed Precision** | AMP über `torch.cuda.amp.GradScaler` (nur CUDA) |
| **Checkpoint fortsetzen** | Training von einer gespeicherten `.pth`-Datei fortführen |
| **Augmentierung** | Zufälliges horizontales Spiegeln + Farb-Jitter |

---

## Datensatz-Exportformate

| Format | Datei(en) | Verwendung |
|---|---|---|
| **COCO JSON** | `annotations.json` | Object-Detection-Frameworks |
| **YOLO TXT** | `<bild>.txt` pro Bild + `classes.txt` | Ultralytics / Darknet |
| **CSV** | `annotations.csv` | Tabellenkalkulation / eigene Tools |

---

## Tests ausführen

```bash
# Alle Tests
pytest tests/ -v

# Nur Unit-Tests
pytest tests/test_project.py tests/test_dataset.py tests/test_metrics.py tests/test_roi.py tests/test_export.py tests/test_anomaly_detector.py -v

# Integrationstests (erfordern torch + Pillow)
pytest tests/test_integration.py -v

# Mit Coverage
pip install pytest-cov
pytest tests/ --cov=core --cov=models --cov-report=term-missing
```

> Integrationstests trainieren ein kleines Modell auf 12 synthetischen Bildern — Laufzeit auf CPU ca. 10–30 Sekunden.

---

## Einstellungen

Dauerhafte Einstellungen werden über `QSettings` gespeichert (plattformnativ: `~/Library/Preferences` unter macOS, Registry unter Windows).

| Einstellung | Standard | Beschreibung |
|---|---|---|
| Theme | `dark` | `dark` oder `light` |
| Schriftgröße | `9` | 7–16 pt |
| Autosave-Intervall | `300 s` | 30–3600 s |
| Backup vor Speichern | `ein` | Erstellt timestamped `.json`-Backup |
| Thumbnail-Größe | `100 px` | 60–240 px |
| Niedrig-Konfidenz-Schwellwert | `0,70` | Vorhersagen darunter werden markiert |
| Top-K-Anzeige | `3` | 1–5 angezeigte Top-Vorhersagen |
| SSH-Profile | — | Host, Benutzer, Key-Pfad pro Profil |

---

## Projektdatei-Format

Projekte werden als UTF-8-JSON (`*.json`) gespeichert. Atomares Schreiben via temporäre Datei + `os.replace()` verhindert Dateikorruption bei Absturz.

```json
{
  "config": { "name": "...", "version": "2.0", "created_at": "..." },
  "labels": [{ "name": "gut", "color": "#2ECC71" }],
  "images": ["pfad/zum/bild.jpg"],
  "image_labels": { "pfad/zum/bild.jpg": "gut" },
  "rois": {
    "pfad/zum/bild.jpg": [
      { "id": "r1", "type": "rect", "x": 10, "y": 10, "w": 50, "h": 50, "label": "gut", "color": "#2ECC71" }
    ]
  },
  "training_config": { ... },
  "inference_results": [ ... ]
}
```

---

## Audit-Trail

Jede Label-Änderung, ROI-Hinzufügung/-Löschung und jeder Trainingsstart wird in `<projektname>_audit.jsonl` im Projektverzeichnis angehängt. Jede Zeile ist ein JSON-Objekt:

```json
{"timestamp": "2025-01-01T12:00:00", "action": "image_labeled", "entity": "bild.jpg", "details": {"label": "gut"}}
```

---

## Fehlerbehebung

**Anwendung startet nicht**
- PySide6 installieren: `pip install PySide6`
- Unter Linux Qt-Plattform-Plugins installieren: `apt install libxcb-cursor0`

**Training sehr langsam**
- Im Gerät-Dropdown `cuda` oder `mps` wählen.
- Für CPU-Tests: Bildgröße auf 128 px und Batch-Größe auf 16 reduzieren.

**`ImportError: No module named 'openpyxl'`**
- `pip install openpyxl` — für Excel-Export erforderlich.

**`ImportError: No module named 'paramiko'`**
- `pip install paramiko` — nur für SSH-Ferntraining benötigt.

**Diagramme erscheinen nicht**
- `pip install matplotlib` — die Anwendung fällt sonst auf ASCII-Sparklines zurück.

**Thumbnails laden langsam**
- Thumbnail-Größe in den Einstellungen reduzieren (z. B. 60 px).

**Projektdatei beschädigt**
- Das aktuellste `.bak`-Backup liegt neben der Projektdatei. In `.json` umbenennen um es wiederherzustellen.

**Kamera wird nicht erkannt**
- `pip install opencv-python` installieren.
- Andere Anwendungen schließen, die die Kamera blockieren könnten.
- Bei IP-Kameras: Stream-URL im Browser prüfen.

---

## Lizenz

MIT — siehe `LICENSE` für Details.
