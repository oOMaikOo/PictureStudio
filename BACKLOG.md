# Backlog

Verbesserungen, die noch nicht umgesetzt wurden. Sortiert nach Aufwand.

---

## Sofort sinnvoll

A–E (hartkodierte UI-Strings aus dem Audit 2026-09-25) sind erledigt — siehe CHANGELOG Unreleased.

---

## Aufräumen / Technische Schulden

Ergebnisse des Code-Audits vom 2026-09-25 (Stand v2.5.1). Innerhalb der Sektion nach Aufwand sortiert.
IDs M (tote Module), N (tote Abhängigkeiten), P (Stack-Index als `IntEnum`), Q (`CameraSettingsGroup`), R (i18n Kamera-Dialog + unübersetzte Dialoge), S (stille Exception-Handler, inkl. **F**), T (Exporte auf QThread) und U (Hilfesektion Live-Klassifikation) sind erledigt — siehe CHANGELOG 2.5.2.
ID **G** (Edit-Menü Undo/Redo), **A–E** (i18n), **H–J** (Ground-Truth-CSV, DatasetStatsWorker, Pattern-Bulk-Label) sowie **K–L** (Batch-Konfidenz-Histogramm, Multi-Cam Alarm-Export) sind erledigt — siehe CHANGELOG Unreleased.
Zu S: die ursprünglich genannten 173 Handler waren eine Überzählung — der Detektor wertete nur `log.`/`raise`/`QMessageBox`/`print` als Fehlerbehandlung und übersah `Response.error(...)`, `self._notify(...)`, `sys.exit(...)` und Rückgaben vom Typ `(False, msg)`. Tatsächlich waren es 80, davon 34 behoben; die verbleibenden 46 sind bewusst still (Eingabevalidierung mit dokumentiertem Default, Teardown, `KeyboardInterrupt`, optionale Abhängigkeiten, Fehler die bereits zurückgegeben werden).
ID O (angeblich unerreichbare Seiten 11/14) war ein Fehlbefund und wurde zurückgezogen: beide Seiten sind über **Ansicht → Datensatz-Stats / Data Drift** erreichbar (`main_window.py:301-302`); sie stehen seit `30099de` bewusst nur im Menü statt in der Sidebar.

**Alle Punkte dieser Sektion sind abgearbeitet.**

Beim Audit **nicht** beanstandet: Locale-Dateien exakt synchron (891/891 Keys, keine Lücke in beide Richtungen); 0 TODO/FIXME/HACK-Marker; keine Laufzeitartefakte im Git getrackt; für die neuen Features existieren dedizierte Tests (`test_live_classification.py`, `test_http_router.py`, `test_ui_mode.py`); COCO/YOLO-Reste sind bewusst behalten (Export), Clustering ist rückstandsfrei entfernt.

---

## Mittlerer Aufwand

H–J sind erledigt — siehe CHANGELOG Unreleased.

---

## Großer Aufwand

K–L sind erledigt — siehe CHANGELOG Unreleased.

---

## Vertrieb / Release

| ID | Beschreibung | Datei(en) |
|----|-------------|-----------|
| V1 | **Release-Installer vervollständigen** — macOS `.app`/DMG ist vorhanden; Windows-PyInstaller+Inno-Skript (`build_windows.bat`, `installer/windows/PictureStudio.iss`) sind vorbereitet — signierte Artefakte und CI-Build fehlen noch | `Picture.spec`, `build_macos.sh`, `build_windows.bat` |
| V3 | **THIRD_PARTY_LICENSES.md** — PySide6 (LGPL) erfordert Attribution bei Weitergabe. Lizenzen aller direkten Abhängigkeiten (PySide6, PyTorch, OpenCV, Pillow, etc.) zusammenstellen | `THIRD_PARTY_LICENSES.md` |
| V6 | **Support-Kanal definieren** — Kein Hinweis in der App oder Doku, wohin Nutzer bei Problemen gehen sollen. GitHub Issues-Link im About-Dialog und in `Fehlerbehebung.md` ergänzen | `gui/main_window.py` (About-Dialog), `wiki/Fehlerbehebung.md` |
| V7 | **Datenschutzerklärung (DSGVO)** — Beim Einsatz in deutschen Unternehmen ggf. erforderlich. Dokument erstellen, das erklärt, welche Daten lokal gespeichert werden (keine Cloud-Übertragung, keine Telemetrie) | `PRIVACY.md` |
| V8 | ~~Update-Check~~ — erledigt (Banner + What’s new); optionale Feinheit: Signatur-Prüfung der Release-Assets | `gui/main_window.py`, `utils/updater.py` |
| V10 | **Windows-Kompatibilitätstest** — App nur auf macOS entwickelt und getestet. Auf einer Windows-VM prüfen: Schriftarten (Segoe UI), Pfadtrennzeichen, `list_usb_cameras()` Swift-Subprocess, Icon-Format `.ico` vs `.icns` | manuell / CI matrix |
