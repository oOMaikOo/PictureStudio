# Backlog

Verbesserungen, die noch nicht umgesetzt wurden. Sortiert nach Aufwand.

---

## Sofort sinnvoll

| ID | Beschreibung | Datei(en) |
|----|-------------|-----------|
| A | **i18n labeling_page** — ~30 hardcodierte Strings in ROI-Mixin und AL-Panel-Methoden → tr(): `"Das aktuelle Bild hat keine ROIs"`, `"Fertig"`, `"Queue leeren"`, `"Alle Queue-Bilder wurden gelabelt"`, Sortier-Optionen, Button-Tooltips | `gui/pages/labeling_page.py`, locales |
| B | **i18n inference_page** — ~24 Strings: GroupBox `"Inferenz-Steuerung"`, Spaltenheader (`"Dateiname"`, `"Vorhergesagtes Label"`, `"Confidence"`), Ensemble-Label, alle Tooltips, `"Alle"` im Filter-Combo | `gui/pages/inference_page.py`, locales |
| C | **i18n models_page** — ~27 Strings: ONNX/TorchScript/CoreML/Docker-Export-Tooltips, `"Modell vergleichen:"`, `"Kalibrierung & Edge-Deployment:"`, `"Accuracy"` im Chart | `gui/pages/models_page.py`, locales |
| D | **i18n data_page** — ~23 Strings: Lade-Tooltips (Ordner, Kamera, Video), COCO/YOLO/CSV-Export-Beschreibungen | `gui/pages/data_page.py`, locales |
| E | **i18n fleet_page `_RemoteTrainDialog`** — ~17 Strings: `"Frames herunterladen"`, `"Anzahl:"`, `"Bereit"`, `"Modell trainieren"`, `"Epochen:"`, `"Fehler"`, QMessageBox-Texte | `gui/pages/fleet_page.py`, locales |
| F | **Stille Exceptions in camera_page** — 7 `except Exception:` ohne Logging. Jede Bare-Exception sollte mindestens `log.debug()` / `log.warning()` bekommen, damit Fehler nicht lautlos verschwinden | `gui/pages/camera_page.py` |

---

## Mittlerer Aufwand

| ID | Beschreibung | Datei(en) |
|----|-------------|-----------|
| G | **Undo/Redo im Edit-Menü** — `LabelingPage._undo_stack` ist mit Ctrl+Z/Y verknüpft, aber kein Edit-Menü existiert in MainWindow. Edit-Menü mit „Rückgängig (Ctrl+Z)" und „Wiederholen (Ctrl+Y)" hinzufügen; Actions an `labeling_page._undo_stack` binden | `gui/main_window.py`, `gui/pages/labeling_page.py` |
| H | **Ground-Truth-Label-Export als CSV** — Kein direkter Export der Wahrheits-Labels (Bild → Label) als CSV. Button in DataPage oder ExportPage: Spalten `image_path, label, uncertain` | `gui/pages/data_page.py` oder `gui/pages/export_page.py`, `core/export.py` |
| I | **dataset_stats_page: PIL auf QThread** — Zeilen 196–197 und 228–240 öffnen PIL-Images synchron im Main-Thread; bei großen Datensätzen friert die UI ein. `DatasetStatsWorker(QThread)` analog zu anderen Workern einführen | `gui/pages/dataset_stats_page.py` |
| J | **Bulk-Label nach Dateinamen-Muster** — In LabelingPage: alle Bilder, deren Dateiname ein Muster enthält (z.B. `*defect*` → Label „defect"), automatisch labeln. Nützlich nach Import von bereits sortiertem Material | `gui/pages/labeling_page.py` |

---

## Großer Aufwand

| ID | Beschreibung | Datei(en) |
|----|-------------|-----------|
| K | **Batch-Inferenz: Konfidenz-Histogramm** — BatchInferencePage zeigt nur Tabelle; ein Balkendiagramm der Confidence-Verteilung (Anteil Bilder pro 10%-Bucket) würde Modellqualität auf einen Blick zeigen | `gui/pages/batch_inference_page.py` |
| L | **Multi-Kamera: Alarm-Protokoll-Export** — MultiCameraPage loggt Alarme in CSV, aber kein UI-Button zum Herunterladen. „Protokoll exportieren"-Button mit QFileDialog | `gui/pages/multi_camera_page.py` |
