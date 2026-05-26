# Picture Studio v2.5.0-beta

> ⚠ **Beta-Version** — Funktionsumfang vollständig, noch nicht für den produktiven Einsatz freigegeben.

Eine Desktop-Anwendung zur Bildannotation, Videoanalyse, CNN-Modelltraining, Anomalieerkennung, Objekterkennung, Data-Drift-Überwachung und Active Learning — entwickelt mit **PySide6** und **PyTorch**.

---

## Funktionsübersicht

| Bereich | Funktionen |
|---|---|
| **Projektverwaltung** | Zwei Projekttypen (Bild / Video), versionierte JSON-Projekte, atomares Speichern, automatische Backups, Projekt-Dashboard, Bildvalidierung & Pfadkorrektur |
| **Datenverwaltung** | Ordner-Import (inkl. Unterordner), Videoimport mit Frame-Extraktion, Drag & Drop, Kameraaufnahme (USB/IP/RTSP), MD5-Duplikaterkennung |
| **Datensatz-Statistiken** | Klassenverteilung, Format-/Größenstatistiken, Label-Rate, perceptual-hash Duplikaterkennung (imagehash, optional) |
| **Video-Annotation** | Frame-für-Frame-Annotation aus Videos (cv2 Slider-Navigation), direktes Hinzufügen zum Projekt |
| **Labeling** | Schnellzuweisung 1–9, Multi-Label-Modus, Label-Hierarchien, Undo/Redo, Audit-Trail, Pixel-Segmentierungsmasken (5 Klassen), Rechtsklick-Menü (Bilder entfernen) |
| **Pre-Labeling** | Trainiertes Modell schlägt Labels für ungelabelte Bilder vor — konfigurierbarer Konfidenz-Schwellwert, vollständiger Undo/Redo-Support |
| **Active Learning** | Automatischer Unsicherheits-Scan nach dem Training: findet ungelabelte Bilder mit niedrigster Modell-Confidence, befüllt die AL-Review-Queue — iterativer Label-Kreislauf mit minimalem Aufwand |
| **ROI-Editor** | Rechteck, Ellipse, Polygon; Kopieren/Einfügen; Tastenkürzel; ROI-Vorlagen; Batch-Übertragung auf alle Bilder; Drag-to-Move |
| **Training** | ResNet-18/50, MobileNetV2, EfficientNet-B0/B3, ConvNeXt-Tiny, **DINOv2 ViT-S/14** (Foundation Model, frozen backbone + linear probe), SimpleCNN; Early Stopping, LR-Scheduler, Mixed Precision, Klassenausgleich (WeightedSampler), Focal Loss, SSH-Ferntraining |
| **Hyperparameter-Suche** | Optuna-basiert (lr, batch_size, architecture, optimizer), beste Parameter direkt in die UI übernommen, Live-Log-Dialog |
| **Augmentation-Pipeline** | Rotation, Flip H/V, Helligkeit, Kontrast, Blur, Rauschen; konfigurierbare Kopien pro Bild |
| **Anomalie-Erkennung** | Unüberwachter Conv-Autoencoder; Live-Scoring, Heatmap, Bounding Box, Grad-CAM-Overlay, konfigurierbarer Schwellwert, Kalibrierungsdialog, Event-Log (CSV), MQTT-Alarm, **Auto-Retraining-Banner** (nach N Alarmen), **Shadow Mode / A/B-Vergleich** (zwei Modelle parallel, Divergenz-CSV) |
| **Objekterkennung (YOLOv8)** | Bounding-Box-Training auf Projekt-ROIs; Modellgrößen n/s/m/l; Ordner-Inferenz mit konfigurierbarem Konfidenz-Schwellwert; CSV-Export; optional (`pip install ultralytics`) |
| **Data Drift Detection** | Z-Score-basierter Vergleich von Produktionsbildern zur Trainingsdistribution; Merkmale: Farbe, Schärfe (Laplacian), Kantendichte (Canny), Histogramm; Baseline speicher-/ladbar; farbkodierte Ergebnistabelle |
| **Kamera-Einstellungen** | Helligkeit, Kontrast, Sättigung, Schärfe, Belichtung live anpassen (USB/UVC); Zurücksetzen auf Neutral |
| **Vorverarbeitungsfilter** | Graustufen, Canny-Kanten, Sobel-Gradient, Laplacian vor Anzeige und optional vor Scoring |
| **Modellbibliothek** | Versioniertes Registry, ONNX-Export (Opset 17/INT8), TorchScript-Export, CoreML-Export (macOS), sortierbare Vergleichs-Tabelle, Run-History, Archivieren/Löschen |
| **Modell-Kalibrierung** | Temperature Scaling (scipy) für korrekte Konfidenzwerte post-hoc |
| **Inferenz** | Batch-Inferenz, Top-K-Anzeige, Test-Time Augmentation (TTA), Ensemble-Inferenz, Semi-automatisches Labeling, Konfidenz-Farbkodierung, ROI-Fallback |
| **Anomalie-Clustering** | K-Means/DBSCAN auf Autoencoder-Latentspace; Cluster-Übersicht mit Galerie |
| **Fleet-Management** | Zentrale Überwachung mehrerer remote `monitor.py`-Instanzen; Remote-Training und -Deploy direkt aus der GUI |
| **Docker-Deployment** | Einzeilen-Generator für `Dockerfile`, `docker-compose.yml`, Startskript und README |
| **Metriken & Berichte** | Accuracy, F1, gewichteter F1, ROC/AUC, Top-K, HTML- und Excel-Trainingsbericht, Konfusionsmatrix |
| **REST-API** | `POST /api/classify`, `GET /api/status`, `GET /api/labels`, `GET /api/scores`, `GET /api/events`, `GET /dashboard`, per-Kanal Multi-Kamera-Endpunkte |
| **Standalone Monitor** | `monitor.py` — headless, kein GUI, ONNX/PyTorch, RTSP/HTTP/Video-Datei, MQTT, REST-API + Dashboard, Auto-Reconnect |
| **Export** | COCO JSON, YOLO TXT, CSV-Annotationen; Excel-Inferenzergebnisse (konfigurierbare Spalten) |
| **UX** | Modernes Dark-Theme (GitHub-Dark Palette), Sidebar-Navigation, geführte Tour, F1-Hilfe, QSettings-Persistenz |

---

## Wettbewerbs-Vergleich

| Feature | PictureStudio | Roboflow | LandingLens | HALCON |
|---|---|---|---|---|
| Klassifikation | ✅ | ✅ | ✅ | ✅ |
| Anomalie-Erkennung | ✅ | ❌ | ✅ | ✅ |
| Object Detection | ✅ | ✅ | ✅ | ✅ |
| OPC-UA / Modbus | ✅ | ❌ | ❌ | ✅ |
| Fleet / Edge-Deployment | ✅ | teilweise | ✅ | ✅ |
| Pre-Labeling | ✅ | ✅ | ✅ | — |
| Data Drift Detection | ✅ | ✅ | ✅ | — |
| Active Learning | ✅ | ✅ | ✅ | — |
| Lokal / kein Abo | ✅ | ❌ | ❌ | ✅ |
| Open Source | ✅ | ❌ | ❌ | ❌ |
| **Preis** | **kostenlos** | ab 249 $/Mo | Enterprise | ~10 k€ |

---

## Neue Features in v2.5.x

### DINOv2 Foundation Model Backbone (v2.5.0)

`DINOv2 ViT-S/14` ist jetzt als Architektur im Training-Dropdown verfügbar:

- **Backbone eingefroren** — die 21 M ViT-Parameter werden nicht trainiert; nur ein linearer Kopf (384 → N Klassen) lernt
- **Ideal bei wenig Daten** — < 100 Bilder pro Klasse reichen für gute Ergebnisse (klassisches Transfer Learning braucht 200+)
- **Erster Start** lädt ~85 MB via `torch.hub` (Internet nötig, danach gecacht)
- Der Optimizer übergibt automatisch nur trainierbare Parameter — kein Extra-Aufwand

```python
# Training-Seite → Architektur-Dropdown → "dinov2_vits14" wählen → Training starten
```

---

### Auto-Retraining Loop (v2.5.0)

Die **Live & Anomalie**-Seite zeigt nach 20 geloggten Alarmen automatisch einen Banner:

```
⚠  20 Alarme in dieser Sitzung  —  Retraining empfohlen
                                [Jetzt trainieren]  [✕]
```

- **Jetzt trainieren** navigiert direkt zur Training-Seite
- **✕** schließt den Banner und setzt den Zähler zurück
- Schliesst den Lernzyklus: Alarm → neue Daten → Nachtraining → besseres Modell

---

### Shadow Mode / A/B Modellvergleich (v2.5.0)

Ein zweites Anomalie-Modell kann parallel betrieben werden:

1. **Shadow-Modell laden…** (lila Button) → `.pth` wählen
2. Beide Modelle bewerten jeden Frame gleichzeitig (gleicher ROI-Crop)
3. **Oranger Balken** zeigt Shadow-Score; **Δ-Anzeige** zeigt Differenz
4. Bei Meinungsverschiedenheit: **⚡ Divergenz** + Logeintrag in `shadow_divergences.csv`

Anwendungsfall: altes Modell vs. neu trainiertes Modell sicher im Produktivbetrieb vergleichen, bevor umgestellt wird.

---

## Neue Features in v2.4.x

### Active Learning — Automatischer Unsicherheits-Scan (v2.4.0)

Schließt den Kreislauf zwischen Training und Labeling:

1. **Training abschließen** — beliebiges Klassifikationsmodell
2. **Tab „🔄 Active Learning" öffnen** (Training-Seite, rechter Bereich)
3. **Schwellwert einstellen** (Standard: 70 %) und **Max. Kandidaten** (Standard: 50)
4. **🔍 AL-Scan starten** — das Modell klassifiziert alle ungelabelten Projektbilder im Hintergrund
5. Die unsichersten Bilder (niedrigste Confidence) werden in die **AL-Queue** eingetragen
6. **Labeling-Seite**: das orange AL-Panel erscheint automatisch — Bilder reviewen, Vorschläge akzeptieren oder manuell labeln
7. **Neu trainieren** — das Modell verbessert sich mit den informativsten Bildern

**Warum Active Learning?**  
Statt zufälliger Stichproben labelst du gezielt die Grenzfälle, bei denen das Modell unsicher ist. Gleiche Modellqualität mit deutlich weniger Labeling-Aufwand — typisch reichen 3–5 Iterationen.

| Parameter | Beschreibung | Standard |
|---|---|---|
| Unsicherheits-Schwellwert | Confidence < Schwellwert → in Queue | 0.70 |
| Max. Kandidaten | Maximale Queue-Größe pro Scan | 50 |

---

## Neue Features in v2.3.x

### Pre-Labeling — Modell-Vorschläge im Labeling-Editor (v2.3.9)

Das neue **Pre-Labeling**-Panel (Labeling-Seite, rechte Spalte) beschleunigt die Annotation großer Datensätze erheblich:

1. **📂 Modell laden** — trainiertes `.pth`-Klassifikationsmodell wählen
2. **Konfidenz-Schwellwert einstellen** (Standard: 75 %)
3. **▶ Vorschläge generieren** — läuft im Hintergrund, zeigt Fortschritt
4. **✅ Vorschläge übernehmen** — weist Labels zu (vollständiger Undo/Redo-Support)

Vorschläge unter dem Schwellwert oder mit unbekannten Labels werden automatisch übersprungen. Bilder mit vorhandenem Label können optional eingeschlossen werden.

---

### Data Drift Detection — Produktionsüberwachung (v2.3.8)

Erkennt automatisch, wenn sich Produktionsbilder von der Trainingsdistribution unterscheiden (Seite **Data Drift**, Stack 16):

- **Baseline erstellen** aus Projektbildern oder beliebigem Ordner — analysiert Farbe, Schärfe (Laplacian-Varianz), Kantendichte (Canny), Graustufenhistogramm
- **Z-Score-Vergleich** neuer Bilder zur Baseline — kein Data Scientist benötigt
- **Farbkodierung**: Grün (kein Drift) · Orange (leichter Drift) · Rot (starker Drift)
- Baseline als JSON speicher-/ladbar; optionaler KS-Test mit `scipy`
- CSV-Export für Weiterverarbeitung

Keine zusätzlichen Abhängigkeiten — nur numpy + Pillow (bereits enthalten).

---

### Objekterkennung (YOLOv8) — Stack 15 (v2.3.7)

Neue Seite **Objekterkennung** für Bild-Projekte:

1. **Bilder mit ROIs annotieren** (Labeling-Seite — jeder ROI + Label = eine Bounding-Box-Annotation)
2. **Dataset vorbereiten** — automatische Konvertierung ins YOLO-Format (normalisierte Koordinaten, Train-/Val-Split)
3. **Modell wählen & trainieren** — yolov8n / s / m / l, Epochen/Batch/Gerät konfigurierbar
4. **Erkennung auf neuen Bildern** — Einzelbild mit Bounding-Box-Preview oder Ordner-Inferenz mit Ergebnistabelle

| Modell | Parameter | Empfehlung |
|---|---|---|
| yolov8n | ~3 M | CPU, sehr schnell |
| yolov8s | ~11 M | Gutes Gleichgewicht |
| yolov8m | ~26 M | Empfohlen für Produktion |
| yolov8l | ~44 M | Maximale Genauigkeit (GPU) |

```bash
pip install ultralytics   # optional
```

---

### EfficientNet-B3 + ConvNeXt-Tiny (v2.3.5)

Zwei neue Architekturen direkt im Training-Dropdown:

| Modell | ImageNet-Acc | Empfehlung |
|---|---|---|
| EfficientNet-B3 | ~82 % | ★ Beste Genauigkeit/Größe |
| ConvNeXt-Tiny | ~82 % | Modernes CNN, gut bei wenig Daten |

Beide werden automatisch in den Optuna-Suchraum einbezogen.

---

### Focal Loss für unbalancierte Datensätze (v2.3.6)

Checkbox **Focal Loss** in der Trainingsseite — dämpft den Verlust einfacher Beispiele und fokussiert das Training auf schwierige/seltene Klassen. Parameter `γ` (Gamma) einstellbar von 0,5–5,0 (Standard: 2,0; γ = 0 entspricht CrossEntropy). Kombinierbar mit WeightedSampler.

---

### Ältere Features (v2.0–v2.3.4)

- **ROI per Maus verschieben** — Smart-Click in Zeichenmodi: Klick auf vorhandenen ROI = Drag; Klick auf leerem Bereich = neu zeichnen
- **Bilder aus Datensatz entfernen** — Rechtsklick auf Thumbnail → Bild(er) entfernen (Dateien bleiben erhalten)
- **ROI-Fallback bei Ordner-Klassifikation** — automatische Nutzung des ersten Projekt-ROI wenn kein Template aktiv ist
- **Rekursive Unterordner-Klassifikation** — Checkbox "Unterordner einschließen" in Inferenz & Batch-Inferenz
- **HPT Live-Log-Dialog** — Pro-Trial-Log mit ★-Markierung für neue Bestmarken
- **Anomalie-HPT** — Optuna-Suche über `base_ch`, Lernrate, Batch-Größe
- **Grad-CAM** — Heatmap-Overlay für Anomalie-Erkennung (Checkbox in CameraPage)
- **Multi-Kamera** — dynamisches Grid (1–9 Kanäle), eigener Detektor je Kanal
- **Fleet-Management** — Zentrale Überwachung remote laufender `monitor.py`-Daemons; Remote-Training (Frames per `GET /api/frames` herunterladen, lokal trainieren) und Hot-Swap-Deploy (Modell per `POST /api/deploy` übertragen — kein Neustart)

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

### Optionale Abhängigkeiten

| Paket | Feature |
|---|---|
| `ultralytics` | YOLOv8 Objekterkennung |
| `optuna` | Hyperparameter-Suche (Klassifikation + Anomalie) |
| `imagehash` | Perceptual-Duplikaterkennung in Datensatz-Statistiken |
| `scipy` | Temperature Scaling + KS-Test für Data Drift |
| `coremltools` | CoreML-Export (nur macOS) |
| `paho-mqtt` | MQTT-Alarm-Publishing |
| `onnxruntime` | ONNX-Inferenz in monitor.py |

```bash
pip install ultralytics optuna imagehash scipy paho-mqtt onnxruntime
```

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
- Alle `.jpg`, `.png`, `.bmp`, `.tiff` im Ordner **und Unterordnern** werden hinzugefügt

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
5. Navigation: Pfeiltasten oder `Leertaste` = nächstes Bild

**Optional — Pre-Labeling** (wenn bereits ein Modell vorhanden ist):
- Rechtes Panel → **🤖 Pre-Labeling** → `📂` → `.pth`-Modell laden
- Konfidenz-Schwellwert einstellen → `▶ Vorschläge generieren` → `✅ Vorschläge übernehmen`
- Nur Bilder über dem Schwellwert werden gelabelt — Undo jederzeit möglich

**Optional — ROI zeichnen** (wenn nur bestimmte Bildbereiche relevant sind):
- Toolbar: `R` = Rechteck, `E` = Ellipse, `G` = Polygon
- Bereich im Bild ziehen → ROI erscheint
- Bestehenden ROI verschieben: direkt mit der Maus auf den ROI klicken und ziehen
- `ROIs dieses Bildes → alle Bilder` überträgt ROI-Positionen auf alle anderen Bilder
- `ROI-Größe → alle Bilder` überträgt nur Breite und Höhe (behält je Position)

**Optional — Segmentierungsmaske** (für pixelgenaue Annotation):
- Im mittleren Bereich auf Tab **🎨 Segmentierungsmaske** wechseln
- Linksklick = malen, Rechtsklick = löschen, Scroll = Zoom
- Klasse und Pinselgröße über die Toolbar wählen → `Maske speichern`

---

#### Schritt 5 — Datensatz prüfen

1. Seite **Daten** → `Dataset analysieren`
2. Auf Warnungen achten:
   - **Klassenungleichgewicht** → Klassenausgleich (WeightedSampler) oder **Focal Loss** aktivieren
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
   | EfficientNet-B0 | Gutes Genauigkeits-/Größe-Verhältnis |
   | EfficientNet-B3 ★ | Beste Genauigkeit |
   | ConvNeXt-Tiny ★ | Modernes CNN, gut bei kleinen Datensätzen |
   | SimpleCNN | Sehr schnell, kein Pretrained |
3. **Hyperparameter:**
   - Epochen: `20–50`
   - Lernrate: `0.001`
   - Batch-Größe: `32` (GPU) / `8–16` (CPU)
   - Gerät: `auto` (wählt GPU > MPS > CPU)
   - Early Stopping: `5` (stoppt automatisch bei Plateaus)
   - **Focal Loss** aktivieren bei stark unbalancierten Klassen (γ = 2,0 als Startwert)
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
   - **TTA (Test-Time Augmentation):** Spinner auf `3–5` erhöhen für stabilere Vorhersagen
   - **Ensemble:** Mehrere Modelle laden (`+ Modell hinzufügen`), Vorhersagen werden gemittelt
   - **Unterordner einschließen:** Checkbox für rekursive Ordner-Klassifikation
5. `Alle Bilder klassifizieren` klicken
6. **Farbkodierung:** Grün >90% | Gelb 70–90% | Rot <70% Konfidenz
7. Tab **Niedrige Konfidenz** zeigt alle unsicheren Vorhersagen

---

#### Schritt 9 — Active Learning (iterative Verbesserung)

1. Nach dem Training: Training-Seite → Tab **„🔄 Active Learning"**
2. **Schwellwert** (Standard 0,70) und **Max. Kandidaten** (Standard 50) einstellen
3. **`🔍 AL-Scan starten`** — scannt alle ungelabelten Projektbilder
4. Labeling-Seite öffnen → das **orange AL-Panel** erscheint automatisch
5. Bilder reviewen: `⚡ Übernehmen` = Vorschlag akzeptieren · `✓ Gelabelt` = manuell gesetzt · `⚡ Alle ≥ 80% übernehmen` für Bulk-Accept
6. Neu trainieren → Scan wiederholen (3–5 Iterationen genügen)

---

#### Schritt 10 — Data Drift überwachen (Produktion)

1. Seite **Data Drift** öffnen
2. **Baseline erstellen:** `📊 Baseline aus Projektbildern erstellen`
3. **Schwellwert einstellen:** Max. Z-Score (Standard: 3,0)
4. **Produktionsordner wählen** → `🔍 Drift analysieren`
5. Farbkodierung zeigt gedriftete Bilder — CSV-Export für Weiterverarbeitung
6. Gedriftete Bilder identifizieren → in Projekt aufnehmen → neu labeln → neu trainieren

---

#### Schritt 11 — Ergebnisse exportieren

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

### Anleitung B — Objekterkennung (YOLOv8)

> Ziel: Mehrere Objekte gleichzeitig in einem Bild lokalisieren und klassifizieren.

---

#### Schritt 1 — Bilder annotieren (ROIs mit Labels)

1. Bildprojekt öffnen, Bilder importieren
2. Labels definieren (jedes Label = eine Objektklasse)
3. Seite **Labeling** → Für jedes Bild ROIs zeichnen + Label zuweisen
   - Mehrere ROIs pro Bild möglich (jeder ROI = eine Bounding Box)

---

#### Schritt 2 — Dataset vorbereiten + Training

1. Seite **Objekterkennung** öffnen
2. `Dataset vorbereiten` — konvertiert ROIs ins YOLO-Format (80 %/20 % Split)
3. Modellgröße wählen (yolov8n für schnellen Test, yolov8m für Produktion)
4. `⚡ Training starten` — Fortschritt und Losses werden live angezeigt
5. Bestes Modell (`best.pt`) wird automatisch angeboten

---

#### Schritt 3 — Neue Bilder analysieren

- **Einzelbild:** `Bild wählen` → Bounding Boxes werden eingezeichnet
- **Ordner:** Ordner wählen → `Erkennung starten` → Tabelle mit Ergebnissen
- Konfidenz-Schwellwert anpassen (0,25 = Standard; 0,5+ für weniger Fehlerkennungen)
- `CSV exportieren` für Weiterverarbeitung

```bash
pip install ultralytics   # einmalig installieren
```

---

### Anleitung C — Videoanalyse & Anomalieerkennung

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
3. Frame-Intervall einstellen (z. B. alle 5 Frames)
4. `Frames extrahieren` → Frames werden ins Projekt-Verzeichnis gespeichert

**Option B: Live-Kamera aufnehmen**
1. Seite **Live & Anomalie** öffnen
2. Kamera-Index wählen (0 = erste USB-Kamera), `▶ Kamera starten`
3. `⚙ Aufnahme & Anomalie-Erkennung…` öffnen
4. Normalprozess vor die Kamera bringen
5. `Normalframes aufnehmen` → mindestens **100 Frames** sammeln (200–300 empfohlen)

> **Wichtig:** Der Autoencoder lernt ausschließlich den Normalzustand. Es werden **keine Anomalie-Beispiele** benötigt.

---

#### Schritt 3 — Anomaliemodell trainieren & kalibrieren

1. Im Kamera/Aufnahme-Dialog: `→ Trainieren` klicken (Standard: 40 Epochen)
2. `📊 Schwellwert kalibrieren…` — Histogramm zeigt Score-Verteilung
3. Empfehlung: µ+2σ für normale Produktion, µ+2,5σ für weniger Fehlalarme

---

#### Schritt 4 — Live-Überwachung aktivieren

1. Im Aufnahme-Dialog: `→ Live-Scoring` aktivieren
2. **Anzeige im Live-Bild:**
   - **Grüner Rahmen** = Normalbereich
   - **Roter Rahmen + Alarmband** = Abweichung erkannt
   - **Heatmap** zeigt welche Region auffällig ist (Grad-CAM optional)
3. Optional: Anomalie-Frames automatisch speichern

---

### Anleitung D — 24/7-Monitoring ohne GUI (Daemon)

> Ziel: Prozess dauerhaft headless überwachen — kein GUI, MQTT-Alarm, REST-Dashboard.

---

#### Schritt 1 — Daemon starten

```bash
source .venv/bin/activate

# Einzelkamera mit vorhandenem Modell (Port 8766)
python monitor.py --model autoencoder.pth

# Collection-only (kein Modell — Frames sammeln, später deployen)
python monitor.py --camera 0 --api-port 8766

# Interaktiver Setup-Wizard (Web-UI auf :8765)
python monitor.py --setup

# Multi-Kamera (vorher konfigurierte Kanäle)
python monitor.py --channels kanäle.json
```

---

#### Schritt 2 — Web-Dashboard und REST-API

1. Browser: `http://<daemon-ip>:8766/dashboard`
2. Automatische Aktualisierung alle 3 Sekunden

**Daemon-REST-API (Port 8766):**

| Endpunkt | Beschreibung |
|---|---|
| `GET /dashboard` | HTML Live-Dashboard |
| `GET /api/status` | Status, `frame_count`, `model_name` |
| `GET /api/scores?limit=120` | Score-Puffer (JSON) |
| `GET /api/latest_alarm` | Letztes Alarm-Event |
| `GET /api/frames?n=150` | ZIP-Archiv mit bis zu N gepufferten JPEG-Frames |
| `POST /api/deploy` | Modell hochladen (multipart, Feld `model`) → Hot-Swap ohne Neustart |

**PictureStudio-Integration (Fleet-Seite):**

1. Gerät mit URL `http://<ip>:8766` hinzufügen
2. **Training** → Frames herunterladen → lokal trainieren → Modell deployen (Hot-Swap)

---

## Trainings-Tipps & Best Practices

### Architektur-Wahl

| Modell | Stärke | Wählen wenn… |
|---|---|---|
| ResNet-18 | Schnell, stabil | Erster Test, viele Bilder |
| MobileNetV2 | CPU-effizient | Deployment ohne GPU |
| EfficientNet-B0 | Kompakt + genau | Standard-Empfehlung |
| **EfficientNet-B3** ★ | Beste Genauigkeit | Schwierige Aufgaben, GPU vorhanden |
| **ConvNeXt-Tiny** ★ | Starkes Vortraining | Kleine Datensätze (<200 Bilder) |
| **DINOv2 ViT-S/14** ★★ | Foundation Model | Sehr wenig Daten (<100/Klasse), Internet beim ersten Start |
| SimpleCNN | Kein Pretrained | Reine Machbarkeitstests |

### Hyperparameter-Empfehlungen

| Parameter | Empfehlung | Hinweis |
|---|---|---|
| **Epochen** | 20–50 + Early Stopping | Geduld 5 stoppt bei Plateau |
| **Lernrate** | 0.001 | Bei NaN/Explosion → 0.0001 |
| **Batch-Größe** | 32 (GPU) / 8–16 (CPU) | Größer = stabiler, mehr RAM |
| **Bildgröße** | 224 px | 128 px für einfache Aufgaben |
| **Optimizer** | AdamW | Bessere L2-Regularisierung |
| **Klassenausgleich** | WeightedSampler | Bei >2:1 Verhältnis aktivieren |
| **Focal Loss** | γ = 2.0 | Zusätzlich bei starkem Ungleichgewicht |

### Wann was tun?

| Problem | Lösung |
|---|---|
| Val-Accuracy stagniert früh | LR erhöhen oder größere Architektur wählen |
| Overfitting | Mehr Augmentation, AdamW, weniger Epochen, mehr Daten |
| Klasse wird kaum erkannt | Focal Loss aktivieren; mehr Bilder sammeln |
| Loss wird NaN | LR um Faktor 10 reduzieren |
| Training sehr langsam | `cuda` oder `mps`; Bildgröße auf 128 reduzieren |
| Instabile Produktions-Vorhersagen | Data Drift Detection prüfen; Pre-Labeling für Nachkorrektur |
| Zu wenig Labeling-Budget | Active Learning: nur die unsichersten Bilder labeln (Training → AL-Scan → Label → neu trainieren) |

### Anomalie-Erkennung Tipps

| Situation | Empfehlung |
|---|---|
| Zu viele Fehlalarme | Schwellwert erhöhen → µ+2,5σ |
| Echte Anomalien werden übersehen | Schwellwert senken → µ+1,5σ; mehr Normalframes |
| Modell schlecht nach Beleuchtungswechsel | Data Drift Detection → neue Frames aufnehmen |
| Kameraschwingungen lösen Alarm aus | Glättung (Smooth-N) auf 7–10 Frames erhöhen |

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
| `Leertaste` | Nächstes Bild |
| `←` / `→` | Voriges / Nächstes Bild |
| `R` | Rechteck-ROI |
| `E` | Ellipse-ROI |
| `G` | Polygon-ROI |
| `Esc` | Zeichnen abbrechen |
| `Entf` | Ausgewählte ROI löschen |
| `Strg+C / V` | ROI kopieren / einfügen |
| `Pfeiltasten` | ROI um 2 px verschieben |
| `Strg+Z / Y` | Rückgängig / Wiederholen |

---

## Unterstützte Architekturen (Klassifikation)

| ID | Modell | ImageNet-Acc | Empfehlung |
|---|---|---|---|
| `resnet18` | ResNet-18 | ~70 % | Schnell, guter Ausgangspunkt |
| `resnet50` | ResNet-50 | ~76 % | Höhere Kapazität |
| `mobilenet_v2` | MobileNetV2 | ~72 % | CPU-Deployment |
| `efficientnet_b0` | EfficientNet-B0 | ~77 % | Gutes Verhältnis |
| `efficientnet_b3` | EfficientNet-B3 ★ | ~82 % | Beste Genauigkeit |
| `convnext_tiny` | ConvNeXt-Tiny ★ | ~82 % | Starkes Vortraining |
| `dinov2_vits14` | DINOv2 ViT-S/14 ★★ | DINO-pretrained | Foundation Model, Linear Probe, ideal < 100 Bilder/Klasse |
| `simple_cnn` | Eigenes 4-Block-CNN | — | Tests ohne GPU |

---

## Sidebar-Seiten (Stack-Indices)

| Index | Seite | Projekte |
|---|---|---|
| 0 | Dashboard | beide |
| 1 | Daten | beide |
| 2 | Labeling | Bild |
| 3 | Training | Bild |
| 4 | Modelle | Bild |
| 5 | Klassifikation | Bild |
| 6 | Export | beide |
| 7 | Einstellungen | beide |
| 8 | Live & Anomalie | Video |
| 9 | Batch-Inferenz | Bild |
| 10 | Multi-Kamera | Video |
| 11 | Anomalie-Clustering | beide |
| 12 | Datensatz-Statistiken | Bild |
| 13 | Video-Annotation | Video |
| 14 | Fleet | Video |
| 15 | Objekterkennung | Bild |
| 16 | Data Drift | Bild |
| 17 | Anomalie-Training | Video |

> **Active Learning** ist kein eigener Stack-Index — der AL-Scan-Tab befindet sich auf der Training-Seite (Index 3); das AL-Review-Panel ist in die Labeling-Seite (Index 2) integriert.

---

## Projektstruktur

```
Picture/
├── main.py
├── monitor.py              # Standalone headless Monitor-Daemon
├── requirements.txt
│
├── core/
│   ├── project.py          # Zentrales Datenmodell
│   ├── dataset.py          # Analyse, Split, COCO/YOLO/CSV-Export
│   ├── training.py         # TrainingWorker, FocalLoss, WeightedSampler
│   ├── inference.py        # Inferencer: TTA, Batch, Ordner
│   ├── pre_labeling.py     # PreLabeler: Modell-Vorschläge für ungelabelte Bilder
│   ├── active_learning.py  # ActiveLearningSampler + ActiveLearningThread
│   ├── data_drift.py       # DriftDetector: Z-Score-basierter Drift-Vergleich
│   ├── object_detection.py # ObjectDetector: YOLOv8-Wrapper + QThread
│   ├── detection_dataset.py# YOLO-Dataset-Konvertierung aus Projekt-ROIs
│   ├── metrics.py          # Accuracy, F1, ROC/AUC, Top-K
│   ├── export.py           # Excel-Export
│   ├── model_manager.py    # Registry, ONNX/TorchScript/CoreML-Export
│   ├── anomaly_detector.py # Conv-Autoencoder: Training, Scoring, Bounding Box
│   ├── anomaly_clustering.py
│   ├── hyperparameter_tuning.py  # HPTWorker + AnomalyHPTWorker (Optuna)
│   ├── remote_ssh.py       # SSHManager, Training-Bundle
│   ├── remote_training.py  # RemoteTrainingThread
│   ├── camera.py           # USB/IP-Kamera-Thread
│   ├── audit.py            # JSONL-Audit-Trail
│   ├── anomaly_logger.py   # CSV-Event-Logger
│   ├── mqtt_client.py      # MQTT-Publisher (optional)
│   ├── calibration.py      # Temperature Scaling
│   ├── gradcam.py          # Grad-CAM für Anomalie-Heatmap
│   ├── edge_export.py      # ONNX INT8 + CoreML Export
│   ├── docker_generator.py # Dockerfile + docker-compose Generator
│   └── report.py           # HTML-Trainingsbericht
│
├── models/
│   └── classifier.py       # Modell-Factory (8 Architekturen inkl. DINOv2), Checkpoint-I/O
│
├── api/
│   └── rest_server.py      # REST-API (stdlib http.server)
│
├── gui/
│   ├── main_window.py
│   ├── sidebar.py
│   ├── pages/
│   │   ├── dashboard_page.py
│   │   ├── data_page.py
│   │   ├── labeling_page.py      # ROI-Editor + Segmentierungsmaske + Pre-Labeling
│   │   ├── training_page.py      # Focal Loss, HPT, Active Learning Tab
│   │   ├── models_page.py
│   │   ├── inference_page.py     # TTA, Ensemble, ROI-Fallback
│   │   ├── batch_inference_page.py
│   │   ├── object_detection_page.py  # Stack 15
│   │   ├── data_drift_page.py        # Stack 16
│   │   ├── export_page.py
│   │   ├── camera_page.py
│   │   ├── multi_camera_page.py
│   │   ├── anomaly_clustering_page.py
│   │   ├── dataset_stats_page.py
│   │   ├── video_annotation_page.py
│   │   ├── fleet_page.py
│   │   └── settings_page.py
│   └── widgets/
│       ├── roi_editor.py         # Drag-to-Move, Undo/Redo
│       ├── mask_editor.py
│       └── thumbnail_list.py
│
├── scripts/
│   └── remote_train.py     # Standalone-Skript für SSH-Ferntraining
│
└── tests/                  # 756 Tests (Unit + Integration)
    ├── conftest.py
    ├── test_project.py
    ├── test_dataset.py
    ├── test_metrics.py
    ├── test_object_detection.py
    ├── test_data_drift.py
    ├── test_pre_labeling.py
    └── test_integration.py
```

---

## Fehlerbehebung

| Problem | Lösung |
|---|---|
| Anwendung startet nicht | `pip install PySide6`; Linux: `apt install libxcb-cursor0` |
| Training sehr langsam | Gerät `cuda` oder `mps` wählen; Bildgröße auf 128 px reduzieren |
| `ImportError: ultralytics` | `pip install ultralytics` (nur für Objekterkennung nötig) |
| `ImportError: openpyxl` | `pip install openpyxl` |
| `ImportError: paramiko` | `pip install paramiko` (nur SSH-Ferntraining) |
| Diagramme fehlen | `pip install matplotlib` |
| Kamera nicht erkannt | Andere Apps schließen; macOS: Kamerazugriff in Systemeinstellungen erlauben |
| Viele Fehlalarme (Anomalie) | Schwellwert erhöhen (`📊 Schwellwert kalibrieren`); Beleuchtung stabilisieren |
| Data Drift zeigt alles rot | Schwellwert erhöhen; sicherstellen dass Baseline repräsentative Bilder enthält |
| Pre-Labeling keine Vorschläge | Konfidenz-Schwellwert senken; Modell auf gleiche Klassen wie Projekt prüfen |
| YOLO-Training startet nicht | `pip install ultralytics`; mindestens 1 annotiertes Bild mit Label nötig |
| Projektdatei beschädigt | `.bak`-Backup im Projektverzeichnis in `.json` umbenennen |
| MQTT verbindet nicht | Broker-Adresse und Port prüfen; `pip install paho-mqtt` |

---

## Tests ausführen

```bash
# Alle Tests (942 Tests, ~210 s)
.venv/bin/python -m pytest tests/ -v

# Nur Unit-Tests (schnell, < 5 s)
.venv/bin/python -m pytest tests/ -q --ignore=tests/test_integration.py

# Einzelne Test-Datei
.venv/bin/python -m pytest tests/test_data_drift.py -v
.venv/bin/python -m pytest tests/test_object_detection.py -v
.venv/bin/python -m pytest tests/test_pre_labeling.py -v

# Mit Coverage
pip install pytest-cov
.venv/bin/python -m pytest tests/ --cov=core --cov=models --cov-report=term-missing
```

---

## Lizenz

MIT — siehe `LICENSE` für Details.
