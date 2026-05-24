# Changelog

All notable changes to PictureStudio are documented here.

---

## [2.5.0] – 2026-05-24

### Added

- **DINOv2 Foundation Model Backbone** — `models/classifier.py`: `DINOv2Classifier`-Klasse lädt `dinov2_vits14` via `torch.hub`. Backbone (ViT-S/14, 21 M Parameter) wird eingefroren (`requires_grad=False`); nur der lineare Kopf (384 → N Klassen) trainiert. Ideal für < 100 Bilder pro Klasse — kein vollständiges Retraining nötig. Erste Nutzung lädt ~85 MB (gecacht). Robuste Fehlermeldung bei fehlendem Internet.
- **`get_available_models()` erweitert** — `"dinov2_vits14"` erscheint automatisch im Training-Dropdown der UI.
- **Optimizer: nur trainierbare Parameter** — `core/training.py`: Der Optimizer erhält jetzt `[p for p in model.parameters() if p.requires_grad]` statt `model.parameters()`. Für DINOv2 werden damit nur die 197k Kopf-Parameter an Adam übergeben; der Log zeigt "N von M Parametern trainierbar (Backbone eingefroren)". Alle anderen Architekturen verhalten sich unverändert.
- **Auto-Retraining Loop** — `gui/pages/camera_page.py`: Nach `_RETRAIN_THRESHOLD` (Standard: 20) geloggten Alarm-Events erscheint ein blauer Banner: `⚠ N Alarme — Retraining empfohlen`. Schaltfläche **Jetzt trainieren** navigiert direkt zur Training-Seite (Stack-Index 3); **✕** schließt den Banner und setzt den Alarm-Zähler zurück. Schließt den Lernzyklus ohne manuelle Zählung.
- **Shadow Mode / A/B Modellvergleich** — `gui/pages/camera_page.py`: Zweites Anomalie-Modell parallel betreiben. Neuer **Shadow-Modell laden…** Button (lila) unter der Haupt-Modell-Zeile. Im Score-Panel: orangefarbener Sekundär-Balken + Shadow-Score-Label + Δ-Divergenz-Anzeige. Bei Meinungsverschiedenheit (Alarm vs. Normal) wird `⚡ Divergenz` angezeigt und der Event in `anomaly_events/shadow_divergences.csv` geloggt. Beiden Detektoren wird derselbe ROI-gecropte Frame übergeben (fairer Vergleich).
- **Architektur-Tooltip** — `gui/pages/training_page.py`: DINOv2-Eintrag mit Beschreibung und Internet-Hinweis ergänzt.

### Changed

- **Help-Dialog** (`gui/help_dialog.py`): Architektur-Tabelle um `efficientnet_b3`, `convnext_tiny` und `dinov2_vits14` ergänzt. Neuer Abschnitt **Auto-Retraining Loop** und **Shadow Mode / A/B Modellvergleich** in der Kamera-Sektion (Section 10).
- **Geführte Tour** (`gui/guide_tour.py`): Training-Schritt "Architektur wählen" zeigt DINOv2 und EfficientNet-B3/ConvNeXt-Tiny mit Empfehlungsstern. Neue Tour-Schritte **Auto-Retraining** und **Shadow Mode** in der Kamera-Tour (Section 8).
- **Quick-Start Wizard** (`gui/quick_start_wizard.py`): Schritt 4 (Modell trainieren) erwähnt DINOv2 als Empfehlung für kleine Datensätze.
- **README.md**: Neue Feature-Tabellen für v2.5.x, Architektur-Tabellen um DINOv2 ergänzt, Anomalie-Zeile um Auto-Retraining + Shadow Mode erweitert, Test-Anzahl auf 942 korrigiert.
- `APP_VERSION` → `2.5.0`

### Tests

- 942 Tests grün (vormals 9 pre-existing UI-Failures durch `init_i18n("de")` in `conftest.py` behoben).

---

## [2.3.9] – 2026-05-22

### Added
- **Pre-Labeling** — `core/pre_labeling.py`: `PreLabeler`-Klasse führt ein trainiertes Klassifikationsmodell auf ungelabelten Projektbildern aus und schlägt Labels mit Konfidenzwert vor. Konfigurierbar: Konfidenz-Schwellwert (Standard: 75 %), optionaler Projekt-ROI als Crop-Template. Labels unter dem Schwellwert oder außerhalb der Projektklassen werden übersprungen.
- **Pre-Labeling-Panel in der Labeling-Seite** — Neue Sektion "🤖 Pre-Labeling" in der rechten Spalte: `.pth`-Modell laden → Schwellwert einstellen → Vorschläge generieren (Hintergrund-Thread) → Vorschläge übernehmen. Vollständiger Undo/Redo-Support über `BulkSetImageLabelCommand` mit Per-Bild-Label-Map.
- **`BulkSetImageLabelCommand` erweitert** — `gui/labeling_commands.py`: Neuer optionaler Parameter `label_map: dict` für Per-Bild-Zuweisung unterschiedlicher Labels. Rückwärtskompatibel — ohne `label_map` Verhalten unverändert.
- **13 Unit-Tests** — `tests/test_pre_labeling.py`: Suggest-Logik (Schwellwert, unbekannte Labels, Fehlerbehandlung, leere Eingabe), Fortschritts-Callback, BulkCommand mit label_map (redo/undo).
- **Hilfe + Tour** — Pre-Labeling-Abschnitt in Hilfe (Sektion 4, Labeling & ROIs) und geführte Tour (neuer Schritt in Labeling-Tour).

---

## [2.3.8] – 2026-05-22

### Added
- **Data Drift Detection** — `core/data_drift.py`: `DriftDetector`-Klasse erkennt automatisch, wenn Produktionsbilder von der Trainingsdistribution abweichen. Merkmalsvektor je Bild: RGB-Mittelwert/-Streuung, Laplacian-Schärfe, Canny-Kantendichte, Graustufenhistogramm (16 Bins). Baseline aus Trainingsdaten → Z-Score-basierter Vergleich mit neuen Bildern. Keine zusätzlichen Abhängigkeiten nötig (nur numpy + Pillow). Baseline speicher-/ladbar als JSON.
- **Data-Drift-Seite** — `gui/pages/data_drift_page.py` (Stack-Index 16): Zweispaltiges Layout. Links: Baseline erstellen (aus Projektbildern oder beliebigem Ordner), Schwellwert-Einstellung (Max. Z-Score, Standard 3.0), Produktionsordner wählen und analysieren, Zusammenfassung (Anteil gedrifteter Bilder). Rechts: Tabelle mit Z-Scores je Bild, farbcodiert (Grün/Orange/Rot). CSV-Export.
- **Farbcodierung** — Grün (kein Drift ≤ Schwellwert) · Orange (leichter Drift ≤ 2×) · Rot (starker Drift > 2×).
- **KS-Test** — Wenn `scipy` installiert ist, steht zusätzlich ein Kolmogorov-Smirnov-Test für p-Wert-basierte Auswertung je Merkmalsgruppe zur Verfügung.
- **Sidebar + Hilfe + Tour** — Data Drift in `_IMAGE_PAGES` (Sidebar), `help_dialog.py` (Sektion 22 mit Workflow + Merkmalstabelle), `guide_tour.py` (5-Schritt-Tour für Stack-Index 16).
- **18 Unit-Tests** — `tests/test_data_drift.py`: Feature-Extraktion, Baseline-Aufbau, Scoring (in-dist < OOD), rekursiver Ordner-Scan, Fortschritts-Callback, JSON-Roundtrip, KS-Test-Fallback.

---

## [2.3.7] – 2026-05-22

### Added
- **Objekterkennung (YOLOv8)** — `core/object_detection.py`: `ObjectDetector`-Klasse als schlanker Wrapper um ultralytics. Unterstützt `load(path)`, `predict_image()` (gibt Liste mit `label`, `confidence`, `x1/y1/x2/y2`, `w/h` zurück), `predict_folder()` (inkl. rekursivem Modus) und `export_onnx()`. Optionale Abhängigkeit: `pip install ultralytics`.
- **YOLO-Dataset-Vorbereitung** — `core/detection_dataset.py`: `prepare_yolo_dataset()` konvertiert Projekt-ROIs ins YOLO-Format (normalisierte Koordinaten 0–1), erstellt `images/train/`, `images/val/`, `labels/train/`, `labels/val/` und `data.yaml`. Nur annotierte Bilder (mit beschriftetem ROI) werden einbezogen.
- **Objekterkennung-Seite** — `gui/pages/object_detection_page.py` (Stack-Index 15): Dreispaltiges Layout: Links Trainings-Konfiguration (Modellgröße yolov8n/s/m/l, Epochen, Bildgröße, Batch, Gerät), Mitte Bild-Preview mit eingezeichneten Bounding-Boxes + Trainingslog, Rechts Ordner-Inferenz mit Ergebnistabelle und CSV-Export. Startet Training im Hintergrund-Thread mit Fortschrittsbalken und Epoch-Log.
- **Modellgrößen** — yolov8n (~3 M Parameter, sehr schnell), yolov8s (~11 M), yolov8m (~26 M), yolov8l (~44 M, sehr genau). Alle Größen direkt in der UI auswählbar.
- **Sidebar + Hilfe + Tour** — Objekterkennung in `_IMAGE_PAGES` (Sidebar), `help_dialog.py` (Sektion 21 mit Workflow-Beschreibung und Tipps), `guide_tour.py` (5-Schritt-Tour für Stack-Index 15).
- **14 Unit-Tests** — `tests/test_object_detection.py`: `TestObjectDetector` (7 Tests inkl. Mock-YOLO-Ausgabe-Parsing) und `TestDetectionDataset` (7 Tests inkl. YOLO-Label-Format, data.yaml-Inhalt, Multi-Klassen, Label-Filterung).

---

## [2.3.6] – 2026-05-22

### Added
- **Focal Loss** — `core/training.py`: `FocalLoss`-Klasse (Lin et al., 2017). Dämpft den Verlust einfacher Beispiele automatisch und fokussiert das Training auf schwierige/seltene Klassen. Besonders effektiv bei unbalancierten Datensätzen (z.B. wenige Defekt-Bilder vs. viele Normal-Bilder). Parameter `γ` (Gamma) einstellbar von 0.5–5.0 (Standard: 2.0; γ=0 entspricht CrossEntropy). Kombinierbar mit Klassenausgleich (WeightedSampler). Nur für Single-Label-Klassifikation aktiv.
- **Focal-Loss-UI** — `gui/pages/training_page.py`: Checkbox "Focal Loss" + Gamma-Spinner in den Trainingsparametern. Gamma-Spinner wird nur aktiviert wenn Focal Loss eingeschaltet ist.

---

## [2.3.5] – 2026-05-22

### Added
- **EfficientNet-B3** — `models/classifier.py`: Neue Architektur (~82% ImageNet-Acc, +5% vs B0). Direkt auswählbar in der Training-Seite und im HPT-Suchraum.
- **ConvNeXt-Tiny** — `models/classifier.py`: Modernste CNN-Architektur (~82% ImageNet-Acc). Besonders gut bei kleinen Datensätzen durch stärkeres Vortraining. Direkt auswählbar in der Training-Seite und im HPT-Suchraum.
- **HPT-Suchraum erweitert** — `core/hyperparameter_tuning.py`: EfficientNet-B3 und ConvNeXt-Tiny werden bei der Hyperparameter-Suche automatisch mit evaluiert.

### Changed
- Architektur-Tooltip in der Training-Seite zeigt Empfehlung (★) für EfficientNet-B3 und ConvNeXt-Tiny.

---

## [2.3.4] – 2026-05-22

### Added
- **ROI per Maus verschieben** — `gui/widgets/roi_editor.py`: Im Rechteck- und Ellipse-Zeichenmodus erkennt ein Klick auf einen bestehenden ROI automatisch die Drag-Absicht. Der ROI lässt sich mit gedrückter linker Maustaste verschieben, ohne den Zeichenmodus zu verlassen. Cursor wechselt zu ✋ beim Hover über einem ROI und zu ✚ auf leerem Bereich. Nach dem Loslassen wird `roi_moved` emittiert → vollständiger Undo/Redo-Support.
- **ROI-Größe → alle Bilder** — `gui/pages/labeling_page.py`: Neuer Button (neben "ROIs dieses Bildes → alle Bilder") überträgt **nur Breite und Höhe** des ausgewählten ROI auf alle Projektbilder. Bilder mit bestehendem ROI behalten ihre x/y-Position; Bilder ohne ROI erhalten einen neuen ROI an der Quellposition.
- **Bilder aus Datensatz entfernen** — `gui/pages/labeling_page.py`: Rechtsklick auf ein Thumbnail (oder Mehrfachauswahl) → "🗑 Bild(er) aus Datensatz entfernen". Entfernt Bild, Label, ROIs und Flags aus dem Projekt nach Bestätigung; Dateien auf der Festplatte bleiben erhalten. Navigiert automatisch zum nächsten verfügbaren Bild.
- **ROI-Fallback bei Ordner-Klassifikation** — `gui/pages/inference_page.py`: Wenn das Projekt per-Bild-ROIs enthält aber kein explizites ROI-Template aktiv ist, wird der erste Projekt-ROI automatisch als Fallback auf alle zu klassifizierenden Bilder angewendet. Ein Info-Dialog informiert über Position/Größe des verwendeten ROI und erklärt, wie ein eigenes Template konfiguriert werden kann.

### Fixed
- **ROI-Drag in Zeichenmodi blockiert** — Bisher wurden Mausklicks im Rect-/Ellipse-Modus immer als "neues ROI zeichnen" interpretiert, selbst wenn der Klick auf einem bestehenden ROI landete. Behoben durch Smart-Click-Detection in `mousePressEvent`.
- **Ordner-Klassifikation ignoriert Trainings-ROIs** — Modelle, die auf ROI-Ausschnitten trainiert wurden, bewerteten neue unbekannte Bilder stillschweigend auf dem vollen Bild. Nun wird automatisch der erste Projekt-ROI als Zuschnitt-Vorlage genutzt.

---

## [2.3.3] – 2026-05-21

### Added
- **HPT Live-Log-Dialog** — `gui/widgets/hpt_progress_dialog.py`: `HptProgressDialog` ersetzt den einfachen `QProgressDialog` bei der Hyperparameter-Suche. Zeigt Fortschrittsbalken, Status-Zeile (beste Val-Acc / Threshold) und ein scrollendes Monospace-Log mit einer Zeile pro Optuna-Trial. Jede Zeile enthält alle ausprobierter Parameter und das Ergebnis; ein **★** markiert neue Bestmarken. Button wechselt von *Abbrechen* zu *Schließen* nach Abschluss.
- **Rekursive Unterordner-Klassifikation** — `core/inference.py` `predict_folder()` hat neuen Parameter `recursive=False`. Bei `recursive=True` wird `os.walk()` verwendet; der Dateiname in der Ergebnistabelle zeigt `Unterordner/Dateiname`. `InferencePage` erhält Checkbox **"Unterordner einschließen"** im Eingabe-Panel.

### Fixed
- **HPT Stop-Fehler** (`RuntimeError: Study.stop is supposed to be invoked inside an objective function`) — `HPTWorker` und `AnomalyHPTWorker` nutzen jetzt `threading.Event`; `stop()` setzt das Event, `objective()` prüft es und ruft `study.stop()` aus dem gültigen Kontext auf.
- **HPT GC-Crash** (`QThread: Destroyed while thread is still running`) — `HPTThread` und `AnomalyHPTThread` werden jetzt als `self._hpt_thread` / `self._ae_hpt_thread` gehalten um vorzeitige Python-GC zu verhindern.
- **HPT `study.best_value` ValueError** — Aufruf von `study.best_value` im `finally`-Block während der Trial noch läuft warf `ValueError('No trials are completed yet.')`. Ersetzt durch `self._best_val_seen` welcher bereits im selben Block aktualisiert wird.
- **Rekursives Bilder-Laden** — `DataPage._load_images()` und `_on_files_dropped()` nutzen `os.walk()` statt `os.listdir()`. Unterordner werden automatisch eingeschlossen.
- **Unterordner-Anzeige im Labeling** — `LazyThumbnailList` zeigt `Unterordner/Dateiname` wenn Bilder aus Unterordnern geladen wurden.

### Changed
- `AnomalyHPTThread` erhält `stop()`-Methode (fehlte bisher).
- `HPTThread` und `AnomalyHPTThread` erhalten `log = Signal(str)` für per-Trial-Nachrichten.
- Dokumentation: Training, Klassifikation, Daten, Kamera-Videoanalyse aktualisiert.

---

## [2.0.0] – 2026-05-20

### Added

**Phase A – Training & Model Intelligence**
- **Hyperparameter-Suche (Optuna)** — `core/hyperparameter_tuning.py`: `HPTWorker` + `HPTThread` mit Optuna-Studie (lr, batch_size, architecture, optimizer). Training-Seite: Schaltfläche "⚙ Hyperparameter-Suche…" öffnet Konfigurations-Dialog, startet Suche und übernimmt beste Parameter in die UI.
- **Temperature Scaling (Kalibrierung)** — `core/calibration.py`: `TemperatureScaler` passt Konfidenzwerte post-hoc an (scipy). Modelle-Seite: "Kalibrieren (Temperature Scaling)…"-Schaltfläche.
- **Modell-Vergleichs-Dialog** — `gui/dialogs/model_comparison_dialog.py`: `ModelComparisonDialog` zeigt sortierbare Tabelle (Accuracy, F1, Architektur, ★ Bestes Modell in Gold). Ersetzt den einfachen `QMessageBox`-Vergleich.

**Phase B – Datensatz & Annotation**
- **Datensatz-Statistiken** — `gui/pages/dataset_stats_page.py`: Klassenverteilung (QProgressBars), Format-/Größenstatistiken (200-Bilder-Sample), perceptual-hash Duplikaterkennung (imagehash, optional), Label-Rate. Sidebar-Eintrag "Datensatz" (Stack-Index 12).
- **Augmentation-Pipeline** — `core/augmentation_pipeline.py`: `AugmentationPipeline` (PIL: Rotation ±15°, Flip H/V, Helligkeit, Kontrast, Blur, Rauschen, `copies_per_image=3`), `AugmentationWorker`, `AugmentationThread`.
- **Video-Annotation** — `gui/pages/video_annotation_page.py`: Frame-Navigation per Schieberegler (cv2), Label-Auswahl, direktes Hinzufügen von Frames zum Projekt. Sidebar-Eintrag "Video-Annotation" (Stack-Index 13) für Video-Projekte.

**Phase C – Fleet & Edge-Deployment**
- **Fleet-Management** — `gui/pages/fleet_page.py`: `FleetPage` überwacht mehrere remote `monitor.py`-Instanzen. QTableWidget (Name/URL/Status/Score/Letzter Alarm/Aktionen), `_PollThread` (urllib GET /api/status), `_AddDeviceDialog` (URL-Validierung), QSettings-Persistenz, Auto-Refresh-Timer (30 s). Sidebar-Eintrag "Fleet" (Stack-Index 14) für Video-Projekte.
- **Docker-Deployment-Generator** — `core/docker_generator.py`: `DockerGenerator.generate()` erstellt 5 Deployment-Dateien: `Dockerfile` (python:3.11-slim, EXPOSE, CMD monitor.py), `docker-compose.yml` (ports, volumes, restart: unless-stopped), `requirements_monitor.txt`, `run_monitor.sh`, `README_deploy.md`. Modelle-Seite: "Docker-Deployment generieren…"-Schaltfläche.
- **Edge-Exporter** — `core/edge_export.py`: `EdgeExporter.export_quantized_onnx()` (torch.onnx.export + optionaler INT8 `quantize_dynamic`), `export_coreml()` (coremltools, nur macOS). `has_coreml()` / `has_quantization()` Statik-Methoden. Modelle-Seite: "ONNX INT8 exportieren…" und "CoreML exportieren…".

### Changed
- **Modelle-Seite** — `_compare_models()` verwendet jetzt `ModelComparisonDialog` statt `QMessageBox`. Neue Schaltflächen: Kalibrieren, ONNX INT8, CoreML, Docker-Deployment.
- **Sidebar** — `_IMAGE_PAGES` um "Datensatz" (Index 12) erweitert. `_VIDEO_PAGES` um "Video-Annotation" (Index 13) und "Fleet" (Index 14) erweitert.
- **Help-Dialog** — 4 neue Abschnitte (16–19): Datensatz-Statistiken, Video-Annotation, Fleet-Management, Modelle Erweitert. Feature-Übersichtstabelle aktualisiert.
- **Guided Tour** — Neue Tour-Schritte für Stack-Index 12 (DatasetStats), 13 (VideoAnnotation), 14 (Fleet).
- `APP_VERSION` → `2.0.0`
- `requirements.txt` — neue optionale Abhängigkeiten dokumentiert (optuna, imagehash, scipy, coremltools, onnxscript).

### Tests
- 54 neue Tests (Phasen A, B, C): 17 + 18 + 19 = 54 grüne Tests; Gesamt 664 → 664+ bestanden.

---

## [1.3.0] – 2026-05-19

### Added
- **OPC-UA / Modbus TCP: Multi-Kamera-Integration** — `MultiCameraPage` forwards
  alarm events to `IndustrialNotifier` (`on_alarm(True, score, threshold)`) for
  every channel that fires an alarm. Previously only the single-channel
  `CameraPage` was wired to the industrial notifier; multi-camera was missing.
  `MainWindow` now calls `multi_camera_page.set_industrial_notifier()` on startup.
- **Anomalie-Clustering: vollständig** — `AnomalyClusteringPage` (Stack-Index 11),
  `AnomalyClustering` core, `ClusteringThread` QThread-Wrapper, CSV-Export und
  Cluster-Browser-UI sind vollständig implementiert und getestet.

### Changed
- Help (Section 9 – Einstellungen): OPC-UA/Modbus section already covers
  general alarm forwarding; no content change needed.
- Tour (Step 7 – Einstellungen): existing OPC-UA step unchanged.

---

## [1.2.0] – 2026-05-19

### Added
- **Multi-camera: channel count selector** — new `QSpinBox` in the toolbar lets you
  choose 1–9 simultaneous monitoring channels (default: 2). The grid rebuilds
  dynamically; existing channel configs (model, camera index) are preserved when
  the count grows.
- **Multi-camera: pagination** — when more than 4 channels are active a navigation
  row (◀ Vorherige / Seite N / Gesamt / Nächste ▶) appears automatically. Each
  page shows up to 4 channels in a 2 × 2 grid; channels on hidden pages keep
  running.
- **Multi-camera: alarm JPEG saving** — every alarm frame is automatically saved as
  `monitor_logs/multi_cam/mc_ch<N>_<YYYYMMDDTHHMMSSZ>.jpg`.
- **REST API: per-channel multi-camera endpoints**
  - `GET /api/mc/channels` — summary of all channels (score, threshold, is_alarm,
    event_count, cam_status).
  - `GET /api/mc/scores?channel=N` — rolling score buffer (up to 500 entries) for
    channel N.
  - `GET /api/mc/latest_alarm?channel=N` — most recent alarm event for channel N.
  - `RestApiServer` gains `set_mc_channel_count()`, `push_mc_score()`,
    `push_mc_alarm()`, `set_mc_cam_status()` for thread-safe state updates.
- **Dashboard: multi-camera section** — the web dashboard (`/dashboard`) shows a live
  per-channel grid (score, alarm state, cam status) that appears automatically when
  channels are registered and hides when none are active.
- **monitor.py: IP/RTSP-URL + video file support** — `--url URL` accepts RTSP/HTTP
  streams and local video files (mp4, avi, mov, mkv, …). `--camera INDEX` and
  `--url` are mutually exclusive.
- **monitor.py: auto-reconnect** — live streams (USB + RTSP/HTTP) reconnect
  automatically after `--reconnect-delay` seconds (default: 5). Video files play
  once without reconnect.
- **monitor.py: MQTT publishing** — `--mqtt-host HOST` publishes alarm events as JSON
  to a configurable broker and topic (`--mqtt-topic`, default:
  `picture_studio/monitor`). Optional auth via `--mqtt-user` / `--mqtt-pass`.
  Graceful no-op when paho-mqtt is not installed.
- **monitor.py: embedded REST API + dashboard** — `--api-port PORT` starts a
  lightweight HTTP server with `/api/status`, `/api/scores`, `/api/latest_alarm`,
  `/api/frame/<file>`, and `/dashboard`. Auth via `--api-key KEY`; status and
  dashboard are always public.

### Changed
- Help dialog section 14 (Multi-Kamera) fully rewritten: documents channel selector,
  pagination, alarm JPEG path, per-channel REST endpoints with curl examples.
- Guided tour step index 10 expanded from 4 → 7 steps covering all new features.
- Feature-overview table in help updated: Multi-Kamera row added, REST-API and
  Kamera/Video rows updated.

---

## [1.1.0] – 2026-05-19

### Added
- **REST API authentication** — optional shared-secret API key (`X-Api-Key` header).
  Generate / show / clear key in Settings → REST-API. Public endpoints (`/api/status`,
  `/dashboard`) never require a key. All other endpoints return HTTP 401 without it.
  The dashboard JS injects the key automatically so it keeps working.
- **Camera auto-reconnect** — when a live camera stream drops, the Live-Monitoring page
  waits 5 seconds and reconnects automatically. Status label turns yellow during reconnect
  and green again on the first successful frame. Manual disconnect stops the cycle.
- **Video file inference in Live-Monitor** — the camera dropdown now includes
  "Videodatei (MP4, AVI, …)". A file dialog opens on selection; native FPS is read from
  the file automatically (fallback: 25 fps). Video playback does not trigger auto-reconnect.

### Fixed
- `show_roi_labels` setting was rendered in the UI but never loaded or saved via QSettings.
  The value is now correctly persisted on save and restored on next launch.

### Changed
- Help dialog and Tour updated for all new features: API key auth workflow,
  auto-reconnect behaviour, video-from-combo usage.
- REST API endpoint table in help now marks public vs. protected endpoints.
- Feature overview table updated.

---

## [1.0.0] – 2026-05-15

### Added
- Initial stable release.
- Image labeling with single- and multi-label mode, ROI editor (rectangle, ellipse, polygon).
- Training pipeline: ResNet18/50, MobileNetV2, EfficientNet-B0; GPU/CPU/MPS; mixed precision.
- SSH remote training with live log streaming.
- Batch inference with auto-labeling and confidence colour coding.
- Anomaly detection: Conv-Autoencoder, heatmap overlay, ROI, alarm deduplication, CSV log.
- Live-Monitoring page with scoring, score chart, alarm banner, and JPEG snapshots.
- Multi-camera monitoring (up to 4 simultaneous feeds).
- Anomaly clustering (DBSCAN/K-Means) for grouping alarm frames.
- ONNX and TorchScript export for edge deployment.
- E-Mail and webhook alarm notifications.
- OPC-UA and Modbus TCP integration for industrial SPS connectivity.
- MQTT alarm publishing (paho-mqtt).
- Standalone monitor client (`monitor.py`) — runs without the GUI.
- REST API with live dashboard (HTML, auto-refresh every 3 s).
- LRU thumbnail cache (max 500 entries) to prevent memory growth.
- CSV and JSON export for inference results.
- Camera stream retry logic: 5 consecutive failures required before error signal.
- Project load error handling with user-friendly messages.
- Thread cleanup on application close (camera, training, clustering, industrial notifier).
