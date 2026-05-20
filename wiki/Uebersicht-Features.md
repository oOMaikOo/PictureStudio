# 📋 Übersicht & Features

> **PictureStudio v2.3.0** — Vollständige Feature-Liste aller Bereiche und unterstützten Architekturen

---

# 📋 Feature-Übersicht

| Bereich | Features |
|---|---|
| Projektverwaltung | Versionierte JSON-Projekte, atomares Speichern, automatische Backups, Projekttypen (Klassifikation / Videoanalyse) |
| ROI-Editor | Rechteck, Ellipse, Polygon; Kopieren/Einfügen; Tastenkürzel; Label-Schnellzuweisung 1–9; Segmentierungsmaske |
| Labeling | Label-Hierarchien, Statistiken, Label-Filter, Review-Modus, Multi-Label, Audit-Trail |
| Datensatz-Analyse | MD5-Duplikaterkennung, Klassenungleichgewicht, COCO/YOLO/CSV-Export, Video-Frame-Import |
| Training | ResNet18/50, MobileNetV2, EfficientNet-B0, SimpleCNN; Early Stopping; Mixed Precision; GPU/CPU/MPS; Klassenausgleich |
| SSH-Ferntraining | Verbindungsprofile, Live-Log-Streaming, conda/venv-Unterstützung, automatischer Download |
| Modellbibliothek | Versioniertes Registry, ONNX/TorchScript-Export, Accuracy/F1-Vergleich, Run-History |
| Inferenz | Batch-Inferenz, Top-K-Anzeige, TTA, Ensemble, Konfidenz-Farbkodierung, Auto-Labeling |
| Excel-Export | Konfigurierbare Spalten, Anhängen/Überschreiben, rote Markierung unter Schwellwert |
| REST-API | HTTP-Server (Port konfigurierbar), optionaler API-Key-Schutz, Label zuweisen per POST, Live-Dashboard im Browser, Per-Kanal-Endpunkte für Multi-Kamera |
| MQTT-Alarm | JSON-Events bei Anomalie-Alarm an beliebigen Broker (paho-mqtt), auth-fähig |
| Kamera / Video | USB-Kamera, IP-Kamera (RTSP/HTTP), Video-Datei direkt im Live-Monitor, Auto-Reconnect, Live-Aufzeichnung (MP4), Burst-Modus |
| Multi-Kamera | 1–9 Kanäle gleichzeitig (Selector im Toolbar), dynamisches 2×2-Grid, Seitenblättern bei >4 Kanälen, Alarm-JPEG-Saving pro Kanal, per-Kanal REST-API |
| Anomalie-Erkennung | Conv-Autoencoder, ROI-Bereich, Bewegungsfilter, Schwellwert-Kalibrierung, Heatmap, Bounding-Box, Alarm-Pause, Audit-Log, False-Positive-Markierung |
| Batch-Analyse | Ordner oder Dateien auf Anomalien prüfen, CSV-Export der Ergebnisse |
| Datensatz-Statistiken | Klassenverteilung, Format-/Größenstatistiken, Label-Rate, perceptual-hash Duplikaterkennung |
| Video-Annotation | Frame-für-Frame-Annotation aus Videodateien, Slider-Navigation, direktes Hinzufügen zum Projekt |
| Fleet-Management | Zentrale Überwachung mehrerer monitor.py-Instanzen, Status-Polling, Auto-Refresh, QSettings-Persistenz |
| Hyperparameter-Suche | Optuna-basierte Suche (lr, batch_size, architecture, optimizer), bestes Ergebnis direkt in UI übernehmen |
| Modell-Kalibrierung | Temperature Scaling (scipy) für korrektere Konfidenzwerte |
| Edge-Export | ONNX INT8 (onnxruntime.quantization), Apple CoreML (.mlpackage via coremltools) |
| Docker-Deployment | Einzeilen-Generator für Dockerfile, docker-compose.yml, Startskript und README |
| Augmentation-Pipeline | Rotation, Flip, Helligkeit, Kontrast, Blur, Rauschen; konfigurierbare Kopien pro Bild |

## Unterstützte Architekturen

| ID | Modell | Empfehlung |
|---|---|---|
| resnet18 | ResNet-18 | Schnell, guter Ausgangspunkt (~11 M Parameter) |
| resnet50 | ResNet-50 | Höhere Kapazität, braucht mehr Daten (~25 M) |
| mobilenet_v2 | MobileNetV2 | Effizient, gut für CPU-Deployment |
| efficientnet_b0 | EfficientNet-B0 | Bestes Genauigkeits-/Größe-Verhältnis |
| simple_cnn | SimpleCNN | Kein Pretrained, schnell für erste Tests |

> 💡 Alle Transfer-Learning-Modelle nutzen ImageNet Pretrained Weights. Deaktiviere Pretrained nur bei sehr spezifischen Datensätzen (z. B. Röntgenbilder, Mikroskopie).
