# Backlog

Verbesserungen, die noch nicht umgesetzt wurden. Sortiert nach Aufwand.

---

## Sofort sinnvoll

A–E sind Teilmengen eines größeren Befunds: projektweit stehen **386 hartkodierte UI-Strings** in `gui/` (Stand 2026-09-25). Der größte Einzelposten fehlt in A–E und steht als **R**.

| ID | Beschreibung | Datei(en) |
|----|-------------|-----------|
| A | **i18n labeling_page** — ~30 hardcodierte Strings in ROI-Mixin und AL-Panel-Methoden → tr(): `"Das aktuelle Bild hat keine ROIs"`, `"Fertig"`, `"Queue leeren"`, `"Alle Queue-Bilder wurden gelabelt"`, Sortier-Optionen, Button-Tooltips | `gui/pages/labeling_page.py`, locales |
| B | **i18n inference_page** — ~24 Strings: GroupBox `"Inferenz-Steuerung"`, Spaltenheader (`"Dateiname"`, `"Vorhergesagtes Label"`, `"Confidence"`), Ensemble-Label, alle Tooltips, `"Alle"` im Filter-Combo | `gui/pages/inference_page.py`, locales |
| C | **i18n models_page** — ~27 Strings: ONNX/TorchScript/CoreML/Docker-Export-Tooltips, `"Modell vergleichen:"`, `"Kalibrierung & Edge-Deployment:"`, `"Accuracy"` im Chart | `gui/pages/models_page.py`, locales |
| D | **i18n data_page** — ~23 Strings: Lade-Tooltips (Ordner, Kamera, Video), COCO/YOLO/CSV-Export-Beschreibungen | `gui/pages/data_page.py`, locales |
| E | **i18n fleet_page `_RemoteTrainDialog`** — ~17 Strings: `"Frames herunterladen"`, `"Anzahl:"`, `"Bereit"`, `"Modell trainieren"`, `"Epochen:"`, `"Fehler"`, QMessageBox-Texte | `gui/pages/fleet_page.py`, locales |
| F | **Stille Exceptions in camera_page** — 7 `except Exception:` ohne Logging. Jede Bare-Exception sollte mindestens `log.debug()` / `log.warning()` bekommen, damit Fehler nicht lautlos verschwinden. Teilmenge von **S** (projektweit 173 Stück) | `gui/pages/camera_page.py` |

---

## Aufräumen / Technische Schulden

Ergebnisse des Code-Audits vom 2026-09-25 (Stand v2.5.1). Innerhalb der Sektion nach Aufwand sortiert.
IDs M (tote Module) und N (tote Abhängigkeiten) sind erledigt — siehe CHANGELOG [Unreleased].
ID O (angeblich unerreichbare Seiten 11/14) war ein Fehlbefund und wurde zurückgezogen: beide Seiten sind über **Ansicht → Datensatz-Stats / Data Drift** erreichbar (`main_window.py:301-302`); sie stehen seit `30099de` bewusst nur im Menü statt in der Sidebar.

| ID | Beschreibung | Datei(en) |
|----|-------------|-----------|
| P | **Stack-Index als `IntEnum` statt Magic Number** — derselbe Index wird an vier Stellen von Hand gepflegt: Listenposition in `main_window._build_ui()`, Tupel in `sidebar.py`, Dict-Keys in `guide_tour.py`, `PAGE_TO_SECTION` in `help_dialog.py`. Der Folgeschaden am Hilfe-Mapping ist behoben (eigene Batch-Sektion 15, `PAGE_TO_SECTION[9]`, siehe CHANGELOG [Unreleased]) — die Ursache bleibt. Offen: Stack 16 mappt behelfsweise auf Sektion 7 statt auf eine eigene Live-Klassifikations-Sektion; `SECTIONS` hat noch ein Loch bei 21 (harmlos, IDs sind freie Schlüssel — nur schließen, wenn ohnehin renumbert wird). Ein `IntEnum Page` in einem gemeinsamen Modul entschärft künftiges Renumbering | `gui/main_window.py`, `gui/sidebar.py`, `gui/guide_tour.py`, `gui/help_dialog.py` |
| Q | **`CameraSettingsGroup` extrahieren** — `camera_capture_dialog.py` (2458 Z.) und `pages/camera_page.py` (1340 Z.) teilen ~533 Zeilen gleichnamiger Methodenkörper. Die Slider-Spezifikationen sind wörtlich doppelt gepflegt (`camera_capture_dialog.py:392` vs. `camera_page.py:493`), ebenso `_reset_cam_settings` und das Filter-Dropdown. Gemeinsames Widget (Slider + Filter + Reset) spart ~500 Zeilen | `gui/widgets/` (neu), `gui/camera_capture_dialog.py`, `gui/pages/camera_page.py` |
| R | **i18n camera_capture_dialog + nie übersetzte Dialoge** — größter i18n-Posten im Projekt und in A–E nicht enthalten: `camera_capture_dialog.py` 114 hartkodierte Strings gegen 38 `tr()`. Dazu zwei Dialoge komplett ohne `tr()`: `qa_review_dialog.py` (14 Strings, 0 `tr()`) und `dialogs/model_comparison_dialog.py` (6 Strings, 0 `tr()`) | `gui/camera_capture_dialog.py`, `gui/qa_review_dialog.py`, `gui/dialogs/model_comparison_dialog.py`, locales |
| S | **Breite `except`-Blöcke projektweit** — 173 Handler, die weder loggen noch weiterreichen, davon 37 reines `except: pass`. Schwerpunkte: `api/rest_server.py` (11), `monitor/runner.py` (9), `core/industrial_notifier.py` (8), `core/data_drift.py` (8), `gui/pages/fleet_page.py` (8), `gui/camera_capture_dialog.py` (7). Legitime Fälle ausnehmen (`utils/reproducibility.py` probiert optionale Imports), Rest mindestens auf `log.debug()` heben. **F** ist die camera_page-Teilmenge | projektweit |
| T | **Exporte in data_page auf QThread** — `_export_coco` (`:427`) und `_export_yolo` (`:441`) laufen synchron; `core/dataset.py:467` und `:517` öffnen darin jedes Bild einzeln mit PIL. Die Analyse daneben läuft korrekt im `AnalysisThread` — die Exporte wurden nicht mitgezogen | `gui/pages/data_page.py`, `core/dataset.py` |

Beim Audit **nicht** beanstandet: Locale-Dateien exakt synchron (891/891 Keys, keine Lücke in beide Richtungen); 0 TODO/FIXME/HACK-Marker; keine Laufzeitartefakte im Git getrackt; für die neuen Features existieren dedizierte Tests (`test_live_classification.py`, `test_http_router.py`, `test_ui_mode.py`); COCO/YOLO-Reste sind bewusst behalten (Export), Clustering ist rückstandsfrei entfernt.

---

## Mittlerer Aufwand

| ID | Beschreibung | Datei(en) |
|----|-------------|-----------|
| G | **Undo/Redo im Edit-Menü** — `LabelingPage._undo_stack` ist mit Ctrl+Z/Y verknüpft, aber kein Edit-Menü existiert in MainWindow. Edit-Menü mit „Rückgängig (Ctrl+Z)" und „Wiederholen (Ctrl+Y)" hinzufügen; Actions an `labeling_page._undo_stack` binden | `gui/main_window.py`, `gui/pages/labeling_page.py` |
| H | **Ground-Truth-Label-Export als CSV** — Kein direkter Export der Wahrheits-Labels (Bild → Label) als CSV. Button in DataPage oder ExportPage: Spalten `image_path, label, uncertain` | `gui/pages/data_page.py` oder `gui/pages/export_page.py`, `core/export.py` |
| I | **dataset_stats_page: PIL auf QThread** — Die Seite hat **keinen einzigen Thread**. Die Größenanalyse (`:196`) ist auf 200 Samples gedeckelt, die Duplikatsuche (`:228`) läuft aber mit `imagehash` ungedeckelt über *alle* Projektbilder im Main-Thread → UI friert minutenlang ein. `DatasetStatsWorker(QThread)` analog zu `AnalysisThread` in `data_page.py` einführen. | `gui/pages/dataset_stats_page.py` |
| J | **Bulk-Label nach Dateinamen-Muster** — In LabelingPage: alle Bilder, deren Dateiname ein Muster enthält (z.B. `*defect*` → Label „defect"), automatisch labeln. Nützlich nach Import von bereits sortiertem Material | `gui/pages/labeling_page.py` |

---

## Großer Aufwand

| ID | Beschreibung | Datei(en) |
|----|-------------|-----------|
| K | **Batch-Inferenz: Konfidenz-Histogramm** — BatchInferencePage zeigt nur Tabelle; ein Balkendiagramm der Confidence-Verteilung (Anteil Bilder pro 10%-Bucket) würde Modellqualität auf einen Blick zeigen | `gui/pages/batch_inference_page.py` |
| L | **Multi-Kamera: Alarm-Protokoll-Export** — MultiCameraPage loggt Alarme in CSV, aber kein UI-Button zum Herunterladen. „Protokoll exportieren"-Button mit QFileDialog | `gui/pages/multi_camera_page.py` |

---

## Vertrieb / Release

| ID | Beschreibung | Datei(en) |
|----|-------------|-----------|
| V1 | **Release-Installer vervollständigen** — macOS `.app`/DMG ist vorhanden; Windows `.exe` + Installer (NSIS/Inno Setup) und signierte Release-Artefakte fehlen noch | `Picture.spec`, `build_macos.sh`, `build_windows.bat` |
| V3 | **THIRD_PARTY_LICENSES.md** — PySide6 (LGPL) erfordert Attribution bei Weitergabe. Lizenzen aller direkten Abhängigkeiten (PySide6, PyTorch, OpenCV, Pillow, etc.) zusammenstellen | `THIRD_PARTY_LICENSES.md` |
| V6 | **Support-Kanal definieren** — Kein Hinweis in der App oder Doku, wohin Nutzer bei Problemen gehen sollen. GitHub Issues-Link im About-Dialog und in `Fehlerbehebung.md` ergänzen | `gui/main_window.py` (About-Dialog), `wiki/Fehlerbehebung.md` |
| V7 | **Datenschutzerklärung (DSGVO)** — Beim Einsatz in deutschen Unternehmen ggf. erforderlich. Dokument erstellen, das erklärt, welche Daten lokal gespeichert werden (keine Cloud-Übertragung, keine Telemetrie) | `PRIVACY.md` |
| V8 | **Update-Check** — Keine automatische Prüfung auf neue Versionen. Beim Start einmal GitHub Releases API abfragen (`github.com/oOMaikOo/Picture/releases/latest`) und bei neuerer Version einen nicht-blockierenden Banner zeigen | `gui/main_window.py`, `utils/updater.py` |
| V10 | **Windows-Kompatibilitätstest** — App nur auf macOS entwickelt und getestet. Auf einer Windows-VM prüfen: Schriftarten (Segoe UI), Pfadtrennzeichen, `list_usb_cameras()` Swift-Subprocess, Icon-Format `.ico` vs `.icns` | manuell / CI matrix |
