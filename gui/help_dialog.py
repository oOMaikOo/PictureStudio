"""
Integrated help dialog: complete workflow guide + context-sensitive help per page.
"""
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem,
    QTextBrowser, QPushButton, QFrame, QLabel,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor

# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

SECTIONS = [
    ("🚀", "Erste Schritte"),
    ("📋", "Übersicht & Features"),
    ("🏠", "Dashboard"),
    ("📁", "Daten"),
    ("🏷", "Labeling & ROIs"),
    ("🧠", "Training"),
    ("📊", "Modellbibliothek"),
    ("🔍", "Klassifikation"),
    ("📤", "Excel-Export"),
    ("⚙", "Einstellungen"),
    ("📷", "Kamera & Videoanalyse"),
    ("⌨", "Tastenkürzel"),
    ("🔧", "Fehlerbehebung"),
    ("💻", "Monitor-Client"),   # 13
]

# Map sidebar page index → section index
PAGE_TO_SECTION = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9}

# ---------------------------------------------------------------------------
# Shared CSS
# ---------------------------------------------------------------------------

_CSS = """
<style>
body  { font-family: -apple-system, 'Segoe UI', sans-serif; font-size: 13px;
        line-height: 1.65; color: #E0E0E0; margin: 0; padding: 0; }
h1    { color: #5DADE2; font-size: 19px; margin: 0 0 4px 0;
        border-bottom: 2px solid #1A5276; padding-bottom: 8px; }
h2    { color: #85C1E9; font-size: 14px; margin: 20px 0 6px 0; }
h3    { color: #AED6F1; font-size: 13px; margin: 12px 0 4px 0; }
p     { margin: 6px 0; }
ul    { margin: 4px 0; padding-left: 20px; }
li    { margin: 3px 0; }
.step { background: #1B3A5C; border-left: 4px solid #2E86C1;
        border-radius: 5px; padding: 9px 13px; margin: 7px 0; }
.step b { color: #85C1E9; }
.tip  { background: #1A3A2A; border-left: 4px solid #27AE60;
        border-radius: 5px; padding: 8px 13px; margin: 7px 0; }
.tip b { color: #58D68D; }
.warn { background: #3A2A1A; border-left: 4px solid #E67E22;
        border-radius: 5px; padding: 8px 13px; margin: 7px 0; }
.warn b { color: #F0B27A; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 12px; }
th    { background: #154360; color: #AED6F1; padding: 7px 10px; text-align: left; }
td    { padding: 5px 10px; border-bottom: 1px solid #2C3E50; vertical-align: top; }
tr:nth-child(even) td { background: #1A252F; }
code  { background: #17202A; color: #F8C471; padding: 1px 6px;
        border-radius: 3px; font-size: 12px; font-family: 'Courier New', monospace; }
kbd   { background: #2C3E50; color: #ECF0F1; padding: 2px 7px;
        border-radius: 4px; border: 1px solid #566573;
        font-size: 11px; font-family: 'Courier New', monospace; }
.num  { display: inline-block; background: #2980B9; color: white;
        border-radius: 50%; width: 22px; height: 22px; text-align: center;
        line-height: 22px; font-weight: bold; font-size: 12px; margin-right: 6px; }
hr    { border: none; border-top: 1px solid #2C3E50; margin: 14px 0; }
</style>
"""


def page(body: str) -> str:
    return _CSS + f'<div style="padding:16px 20px 20px 20px">{body}</div>'


# ---------------------------------------------------------------------------
# Help content per section index
# ---------------------------------------------------------------------------

CONTENT = {

# ── 0  Erste Schritte ────────────────────────────────────────────────────────
0: page("""
<h1>🚀 Erste Schritte – Kompletter Workflow</h1>
<p>Dieser Guide führt dich Schritt für Schritt vom ersten Bild bis zum trainierten Modell.</p>
<hr>

<h2>Phase 1 – Projekt anlegen</h2>
<div class="step"><b>Schritt 1 – Neues Projekt</b><br>
Menü <i>Datei → Neues Projekt</i> (<kbd>Strg+N</kbd>). Vergib einen aussagekräftigen Namen
und wähle den Projekttyp:<br>
• <b>📸 Bildklassifikation</b> – klassifiziert Einzelbilder in Klassen<br>
• <b>🎬 Videoanalyse &amp; Anomalie</b> – Kamera-Livestream mit Autoencoder-Erkennung<br>
Das Projekt wird als <code>.json</code>-Datei gespeichert.</div>

<hr>
<h2>Phase 2 – Bilder laden</h2>
<div class="step"><b>Schritt 2 – Bildordner hinzufügen</b><br>
Gehe zur <b>Daten-Seite</b> → klicke <i>Bilder laden…</i> und wähle den Ordner.<br>
Alle <code>.jpg</code>, <code>.png</code>, <code>.bmp</code> und <code>.tiff</code> Dateien werden
automatisch hinzugefügt. Drag &amp; Drop ins Fenster funktioniert ebenfalls.</div>

<div class="step"><b>Schritt 3 – Datensatz analysieren</b><br>
Klicke <i>Dataset analysieren</i> um zu prüfen:<br>
• <b>Fehlende Dateien</b> – Bilder die nicht mehr auffindbar sind<br>
• <b>Duplikate</b> – identische Bilder (MD5-Hash)<br>
• <b>Klassenungleichgewicht</b> – schlechtes Training wenn eine Klasse viel mehr Bilder hat</div>

<hr>
<h2>Phase 3 – Labels definieren &amp; Bilder annotieren</h2>
<div class="step"><b>Schritt 4 – Labels anlegen</b><br>
Gehe zur <b>Labeling-Seite</b> → <i>Projekt → Labels verwalten…</i> (<kbd>Strg+L</kbd>).<br>
Füge für jede Klasse ein Label hinzu (z. B. "gut", "defekt", "unklar").</div>

<div class="step"><b>Schritt 5 – Bilder labeln</b><br>
1. Bild in der Thumbnail-Liste anklicken<br>
2. <kbd>1</kbd>–<kbd>9</kbd> drücken für schnelle Label-Zuweisung<br>
3. Mit <kbd>N</kbd> zum nächsten Bild, <kbd>P</kbd> zum vorherigen<br><br>
<b>Ziel:</b> Mindestens 50–100 Bilder pro Klasse. Für gute Modelle: 200+ pro Klasse.</div>

<div class="step"><b>Schritt 6 – ROIs zeichnen (optional)</b><br>
Falls du nur einen bestimmten Bildbereich analysieren willst:<br>
• <kbd>R</kbd> = Rechteck &nbsp;• <kbd>E</kbd> = Ellipse &nbsp;• <kbd>G</kbd> = Polygon<br>
ROI in der rechten Liste auswählen → Label zuweisen.</div>

<hr>
<h2>Phase 4 – Training konfigurieren &amp; starten</h2>
<div class="step"><b>Schritt 7 – Architektur &amp; Hyperparameter</b><br>
Gehe zur <b>Training-Seite</b>:<br>
• <b>Architektur:</b> Starte mit <code>ResNet-18</code><br>
• <b>Epochen:</b> 20–30 für erste Tests, 50+ für finale Modelle<br>
• <b>Gerät:</b> <code>auto</code> nutzt automatisch GPU/MPS/CPU<br>
• <b>Early Stopping:</b> 5–7 Epochen verhindert Overfitting</div>

<div class="step"><b>Schritt 8 – Training starten</b><br>
Klicke <i>Training starten</i>. Du siehst in Echtzeit Train-Loss, Val-Loss, Accuracy.<br>
Das beste Modell wird automatisch bei höchster Val-Accuracy gespeichert.</div>

<div class="tip"><b>Woran erkenne ich gutes Training?</b><br>
✓ Val-Loss sinkt parallel zu Train-Loss<br>
✓ Val-Accuracy steigt kontinuierlich<br>
✗ Val-Loss steigt während Train-Loss sinkt = Overfitting</div>

<hr>
<h2>Phase 5 – Modell einsetzen</h2>
<div class="step"><b>Schritt 9 – Neue Bilder klassifizieren</b><br>
Gehe zur <b>Klassifikations-Seite</b>:<br>
1. <i>Modell laden (.pth)</i> → Modelldatei wählen<br>
2. <i>Ordner…</i> → Ordner mit neuen Bildern wählen<br>
3. <i>Alle Bilder klassifizieren</i> → Ergebnis mit Top-3 Vorhersagen</div>

<div class="step"><b>Schritt 10 – Ergebnisse exportieren</b><br>
Gehe zur <b>Export-Seite</b>:<br>
1. <i>Ergebnisse aus letzter Inferenz laden</i><br>
2. Spalten konfigurieren (aktivieren, umbenennen)<br>
3. <i>Excel exportieren</i></div>

<div class="tip"><b>Fertig!</b> Für bessere Ergebnisse: mehr Daten → neu labeln → Training wiederholen.</div>
"""),

# ── 1  Übersicht ────────────────────────────────────────────────────────────
1: page("""
<h1>📋 Feature-Übersicht</h1>
<table>
  <tr><th>Bereich</th><th>Features</th></tr>
  <tr><td><b>Projektverwaltung</b></td><td>Versionierte JSON-Projekte, atomares Speichern, automatische Backups, Projekttypen (Klassifikation / Videoanalyse)</td></tr>
  <tr><td><b>ROI-Editor</b></td><td>Rechteck, Ellipse, Polygon; Kopieren/Einfügen; Tastenkürzel; Label-Schnellzuweisung 1–9; Segmentierungsmaske</td></tr>
  <tr><td><b>Labeling</b></td><td>Label-Hierarchien, Statistiken, Label-Filter, Review-Modus, Multi-Label, Audit-Trail</td></tr>
  <tr><td><b>Datensatz-Analyse</b></td><td>MD5-Duplikaterkennung, Klassenungleichgewicht, COCO/YOLO/CSV-Export, Video-Frame-Import</td></tr>
  <tr><td><b>Training</b></td><td>ResNet18/50, MobileNetV2, EfficientNet-B0, SimpleCNN; Early Stopping; Mixed Precision; GPU/CPU/MPS; Klassenausgleich</td></tr>
  <tr><td><b>SSH-Ferntraining</b></td><td>Verbindungsprofile, Live-Log-Streaming, conda/venv-Unterstützung, automatischer Download</td></tr>
  <tr><td><b>Modellbibliothek</b></td><td>Versioniertes Registry, ONNX/TorchScript-Export, Accuracy/F1-Vergleich, Run-History</td></tr>
  <tr><td><b>Inferenz</b></td><td>Batch-Inferenz, Top-K-Anzeige, TTA, Ensemble, Konfidenz-Farbkodierung, Auto-Labeling</td></tr>
  <tr><td><b>Excel-Export</b></td><td>Konfigurierbare Spalten, Anhängen/Überschreiben, rote Markierung unter Schwellwert</td></tr>
  <tr><td><b>REST-API</b></td><td>HTTP-Server (Port konfigurierbar), Label zuweisen per POST, Live-Dashboard im Browser</td></tr>
  <tr><td><b>MQTT-Alarm</b></td><td>JSON-Events bei Anomalie-Alarm an beliebigen Broker (paho-mqtt), auth-fähig</td></tr>
  <tr><td><b>Kamera / Video</b></td><td>USB-Kamera, IP-Kamera (RTSP/HTTP), Video-Datei, Live-Aufzeichnung (MP4), Burst-Modus</td></tr>
  <tr><td><b>Anomalie-Erkennung</b></td><td>Conv-Autoencoder, ROI-Bereich, Bewegungsfilter, Schwellwert-Kalibrierung, Heatmap, Bounding-Box, Alarm-Pause, Audit-Log, False-Positive-Markierung</td></tr>
  <tr><td><b>Batch-Analyse</b></td><td>Ordner oder Dateien auf Anomalien prüfen, CSV-Export der Ergebnisse</td></tr>
</table>

<h2>Unterstützte Architekturen</h2>
<table>
  <tr><th>ID</th><th>Modell</th><th>Empfehlung</th></tr>
  <tr><td><code>resnet18</code></td><td>ResNet-18</td><td>Schnell, guter Ausgangspunkt (~11 M Parameter)</td></tr>
  <tr><td><code>resnet50</code></td><td>ResNet-50</td><td>Höhere Kapazität, braucht mehr Daten (~25 M)</td></tr>
  <tr><td><code>mobilenet_v2</code></td><td>MobileNetV2</td><td>Effizient, gut für CPU-Deployment</td></tr>
  <tr><td><code>efficientnet_b0</code></td><td>EfficientNet-B0</td><td>Bestes Genauigkeits-/Größe-Verhältnis</td></tr>
  <tr><td><code>simple_cnn</code></td><td>SimpleCNN</td><td>Kein Pretrained, schnell für erste Tests</td></tr>
</table>
<div class="tip">Alle Transfer-Learning-Modelle nutzen ImageNet Pretrained Weights.
Deaktiviere Pretrained nur bei sehr spezifischen Datensätzen (z. B. Röntgenbilder, Mikroskopie).</div>
"""),

# ── 2  Dashboard ─────────────────────────────────────────────────────────────
2: page("""
<h1>🏠 Dashboard</h1>
<p>Der Startpunkt der Anwendung mit Projektübersicht auf einen Blick.</p>

<h2>Was wird angezeigt?</h2>
<ul>
  <li>Gesamtanzahl Bilder und Anteil gelabelter Bilder</li>
  <li>Definierte Klassen mit Farbkodierung</li>
  <li>Anzahl ROIs und Trainingsläufe im Projekt</li>
  <li>Metriken des letzten Trainings (Accuracy, F1)</li>
  <li>Klassenverteilung als Balkendiagramm</li>
  <li>Warnungen bei Klassenungleichgewicht oder zu wenig Daten</li>
</ul>

<h2>Schritt-für-Schritt</h2>
<div class="step"><b>1 – Neues Projekt</b><br>
<i>Datei → Neues Projekt</i> oder <kbd>Strg+N</kbd><br>
Projektname und -typ wählen (📸 Klassifikation oder 🎬 Videoanalyse).<br>
Das Projekt wird als <code>.json</code>-Datei angelegt.</div>

<div class="step"><b>2 – Bestehendes Projekt öffnen</b><br>
<i>Datei → Projekt öffnen</i> oder <kbd>Strg+O</kbd><br>
Zuletzt geöffnete Projekte: <i>Datei → Zuletzt geöffnet</i></div>

<div class="step"><b>3 – Projektinfo ansehen</b><br>
<i>Projekt → Projektinfo…</i> zeigt Pfad, Erstelldatum, Trainingshistorie.</div>

<div class="tip"><b>Tastenkürzel:</b>
<kbd>Strg+S</kbd> Speichern &nbsp;|&nbsp; <kbd>Strg+N</kbd> Neu &nbsp;|&nbsp; <kbd>Strg+O</kbd> Öffnen</div>
"""),

# ── 3  Daten ─────────────────────────────────────────────────────────────────
3: page("""
<h1>📁 Daten</h1>
<p>Bilder laden, Datensatz analysieren, Annotationen exportieren.</p>

<h2>Bilder laden</h2>
<div class="step"><b>Bilder laden…</b><br>
Wähle einen Ordner – alle <code>.jpg .png .bmp .tiff .webp</code> Dateien werden hinzugefügt.<br>
Alternativ: Bilder direkt ins Fenster ziehen (Drag &amp; Drop).<br>
Bilder werden nicht kopiert, der Pfad wird gespeichert.</div>

<div class="step"><b>Video importieren…</b><br>
Video-Datei (MP4, AVI, MOV, MKV, WebM) wählen.<br>
Frame-Intervall einstellen: z. B. alle 5 Frames = ca. 6 Bilder/s bei 30 fps.<br>
Die extrahierten Frames werden als PNG ins Projektverzeichnis gespeichert.</div>

<h2>Dataset analysieren</h2>
<div class="step"><b>Dataset analysieren</b><br>
Prüft deinen Datensatz auf:<br>
• <b>Fehlende Dateien</b> – Bilder die nicht mehr vorhanden sind<br>
• <b>MD5-Duplikate</b> – identische Bilder die Training verzerren<br>
• <b>Klassenungleichgewicht</b> – ungleiche Bildanzahl pro Klasse<br>
• <b>Bildstatistiken</b> – Formate, Größen, Farbmodi</div>

<div class="step"><b>Bildpfade korrigieren…</b><br>
Nach dem Verschieben von Bildern aktualisiert diese Funktion alle Pfade automatisch.</div>

<h2>Annotationen exportieren</h2>
<table>
  <tr><th>Format</th><th>Datei</th><th>Verwendung</th></tr>
  <tr><td>COCO JSON</td><td><code>annotations.json</code></td><td>Object-Detection-Frameworks (YOLO v5+, Detectron2)</td></tr>
  <tr><td>YOLO TXT</td><td><code>bild.txt</code> + <code>classes.txt</code></td><td>Ultralytics / Darknet</td></tr>
  <tr><td>CSV</td><td><code>annotations.csv</code></td><td>Tabellenkalkulation / eigene Tools</td></tr>
</table>
"""),

# ── 4  Labeling ───────────────────────────────────────────────────────────────
4: page("""
<h1>🏷 Labeling &amp; ROIs</h1>
<p>Bilder annotieren, Labels zuweisen, Regionen einzeichnen.</p>

<h2>Labels verwalten</h2>
<div class="step"><b>Labels hinzufügen</b><br>
<i>Projekt → Labels verwalten…</i> (<kbd>Strg+L</kbd>)<br>
Name + Farbe definieren. Labels können jederzeit umbenannt oder neu eingefärbt werden.<br>
Mindestens 2 Labels für das Training erforderlich.</div>

<h2>Bilder labeln</h2>
<div class="step"><b>Schnelles Labeln</b><br>
1. Bild in der Liste anklicken<br>
2. <kbd>1</kbd>–<kbd>9</kbd> drücken = Label 1–9 zuweisen<br>
3. <kbd>N</kbd> = nächstes Bild, <kbd>P</kbd> = vorheriges Bild<br>
Label-Filter: Nur Bilder einer bestimmten Klasse anzeigen.</div>

<h2>ROIs zeichnen</h2>
<div class="step"><b>Rechteck</b> <kbd>R</kbd> – Im Bild ziehen</div>
<div class="step"><b>Ellipse</b> <kbd>E</kbd> – Im Bild ziehen</div>
<div class="step"><b>Polygon</b> <kbd>G</kbd> – Punkte klicken, Doppelklick zum Abschließen</div>

<h2>ROI bearbeiten</h2>
<table>
  <tr><th>Taste</th><th>Aktion</th></tr>
  <tr><td><kbd>Entf</kbd></td><td>Ausgewählte ROI löschen</td></tr>
  <tr><td><kbd>Strg+C</kbd></td><td>ROI kopieren</td></tr>
  <tr><td><kbd>Strg+V</kbd></td><td>ROI einfügen</td></tr>
  <tr><td><kbd>Pfeiltasten</kbd></td><td>ROI um 2 px verschieben</td></tr>
  <tr><td><kbd>Esc</kbd></td><td>Zeichnen abbrechen</td></tr>
</table>

<div class="tip"><b>ROIs auf alle Bilder übertragen:</b>
Wenn alle Bilder dieselbe Aufnahmeposition haben, zeichne ROIs einmal und
nutze <i>ROIs dieses Bildes → alle Bilder</i>.</div>

<h2>Segmentierungsmaske</h2>
<div class="step"><b>Tab 🎨 Segmentierungsmaske</b><br>
Für pixelgenaue Annotation:<br>
Linksklick = malen | Rechtsklick = löschen | Scroll = Zoom<br>
Klasse und Pinselgröße über die Toolbar wählen.<br>
<i>Maske speichern</i> speichert als PNG neben der Bilddatei.</div>
"""),

# ── 5  Training ───────────────────────────────────────────────────────────────
5: page("""
<h1>🧠 Training</h1>
<p>CNN-Modell trainieren – lokal oder auf GPU-Server via SSH.</p>

<h2>Konfiguration</h2>
<table>
  <tr><th>Parameter</th><th>Empfehlung</th><th>Beschreibung</th></tr>
  <tr><td>Architektur</td><td>ResNet-18 zum Start</td><td>Modellarchitektur</td></tr>
  <tr><td>Epochen</td><td>20–30</td><td>Anzahl Trainingsdurchläufe</td></tr>
  <tr><td>Lernrate</td><td>0.001</td><td>Schrittgröße der Gewichts-Updates</td></tr>
  <tr><td>Batch-Größe</td><td>32 (GPU) / 8–16 (CPU)</td><td>Bilder pro Trainingsschritt</td></tr>
  <tr><td>Gerät</td><td>auto</td><td>Automatisch: GPU &gt; MPS (Apple) &gt; CPU</td></tr>
  <tr><td>Early Stopping</td><td>5–7</td><td>Stopp nach N Epochen ohne Verbesserung</td></tr>
  <tr><td>LR-Scheduler</td><td>cosine</td><td>Lernrate automatisch anpassen</td></tr>
  <tr><td>Klassenausgleich</td><td>bei Ungleichgewicht</td><td>WeightedRandomSampler ausgleichen</td></tr>
</table>

<h2>Training starten &amp; überwachen</h2>
<div class="step"><b>Training starten</b><br>
Klicke <i>Training starten</i>. Live-Anzeige:<br>
• <b>Train-Loss</b> und <b>Val-Loss</b> – sinken idealerweise gemeinsam<br>
• <b>Train-Acc</b> und <b>Val-Acc</b> – steigen idealerweise gemeinsam<br>
Das beste Modell (höchste Val-Acc) wird automatisch gespeichert.</div>

<div class="warn"><b>Overfitting erkennen:</b>
Train-Loss sinkt, Val-Loss steigt → Modell lernt Trainingsdaten auswendig.
Lösung: mehr Daten, Early Stopping aktivieren, größere Augmentierung.</div>

<h2>SSH-Ferntraining</h2>
<div class="step"><b>Einrichten</b><br>
1. <i>Einstellungen → SSH-Profile</i> – Profil anlegen (Host, User, Key-Pfad)<br>
2. Training-Seite: SSH-Checkbox aktivieren → Profil wählen → Verbindung testen<br>
3. Training starten → Anwendung zippt Daten, lädt hoch, streamt Logs,
   lädt das beste Modell automatisch herunter.</div>

<h2>Nach dem Training</h2>
<div class="step"><b>Berichte erstellen</b><br>
• <i>HTML-Bericht</i> – vollständiger Report mit Kurven und Konfusionsmatrix<br>
• <i>Excel-Bericht</i> – Metriken als Tabelle für Dokumentation</div>
"""),

# ── 6  Modelle ────────────────────────────────────────────────────────────────
6: page("""
<h1>📊 Modellbibliothek</h1>
<p>Alle trainierten Modelle verwalten, vergleichen und einsetzen.</p>

<h2>Modell auswählen</h2>
<div class="step"><b>In Inferenz laden</b><br>
Modell in der Tabelle anwählen → <i>In Inferenz laden</i>.<br>
Die Anwendung wechselt automatisch zur Klassifikations-Seite.</div>

<div class="tip"><b>Welches Modell ist das beste?</b><br>
Bei gleichmäßigen Klassen: Accuracy-Spalte vergleichen.<br>
Bei ungleichen Klassen: F1-Score ist aussagekräftiger.</div>

<h2>Exportieren</h2>
<div class="step"><b>Als ONNX exportieren (.onnx)</b><br>
Einsatz ohne PyTorch in: ONNX Runtime, OpenCV DNN, TensorRT, C++, C#, Edge-Geräten.</div>

<div class="step"><b>Als TorchScript exportieren (.pt)</b><br>
Einsatz in der PyTorch C++ API oder mobilen Apps (Android/iOS).</div>

<h2>Modelle vergleichen</h2>
<div class="step"><b>Ausgewählte vergleichen</b><br>
Mehrere Modelle mit <kbd>Strg+Klick</kbd> auswählen → <i>Ausgewählte vergleichen</i><br>
Zeigt Accuracy, F1, Architektur und Best-Markierung nebeneinander.<br>
<b>Run-History-Tab:</b> alle Läufe nach Datum sortiert mit Gerät, Epochen, Train-Acc.</div>
"""),

# ── 7  Klassifikation ─────────────────────────────────────────────────────────
7: page("""
<h1>🔍 Klassifikation</h1>
<p>Neue Bilder mit dem trainierten Modell klassifizieren.</p>

<h2>Schritt-für-Schritt</h2>
<div class="step"><b>1 – Modell laden</b><br>
<i>Modell laden (.pth)</i> → Modelldatei wählen.<br>
Schneller: Modelle-Seite → <i>In Inferenz laden</i><br>
<b>Ensemble:</b> mehrere Modelle per <i>+ Modell hinzufügen</i> kombinieren für stabilere Vorhersagen.</div>

<div class="step"><b>2 – Bildordner wählen</b><br>
<i>Ordner…</i> → Ordner mit neuen Bildern wählen.<br>
<b>TTA (Test-Time Augmentation):</b> Spinner auf 3–5 → mehrere augmentierte Versionen
je Bild, Durchschnitt = genauere Ergebnisse bei Grenzfällen.</div>

<div class="step"><b>3 – Klassifizieren</b><br>
<i>Alle Bilder klassifizieren</i> → Ergebnis mit Top-K Vorhersagen.<br>
Farbkodierung: Grün &gt;90% | Gelb 70–90% | Rot &lt;70%</div>

<div class="step"><b>4 – Unsichere prüfen</b><br>
<b>Niedrig-Konfidenz-Tab</b>: alle Bilder unter dem eingestellten Schwellwert (Standard: 70%).<br>
Diese manuell prüfen und ggf. ins Training aufnehmen.</div>

<div class="step"><b>5 – Automatisch labeln</b><br>
Hochkonfidente Ergebnisse direkt als Projekt-Labels übernehmen:<br>
Mindest-Konfidenz einstellen → <i>Auf Projekt anwenden</i><br>
Danach Labeling-Seite zur Kontrolle öffnen.</div>

<h2>Konfidenz-Schwellwert</h2>
<div class="tip">Einstellungen → Schwelle 'unsicher' (Standard: 0.70).<br>
Niedrigerer Wert = mehr Bilder gelten als sicher.<br>
Höherer Wert = strengere Qualitätskontrolle, mehr im Niedrig-Konfidenz-Tab.</div>
"""),

# ── 8  Export ─────────────────────────────────────────────────────────────────
8: page("""
<h1>📤 Excel-Export</h1>
<p>Klassifikationsergebnisse als formatierte Excel-Datei exportieren.</p>

<h2>Schritt-für-Schritt</h2>
<div class="step"><b>1 – Ergebnisse laden</b><br>
<i>Ergebnisse aus letzter Inferenz laden</i> – übernimmt die aktuellen
Klassifikationsergebnisse aus dem Projekt.</div>

<div class="step"><b>2 – Zieldatei wählen</b><br>
<i>Datei wählen…</i> → vorhandene Excel-Datei (für Anhängen-Modus)<br>
<i>Neue Datei erstellen</i> → neue Excel-Datei anlegen</div>

<div class="step"><b>3 – Spalten konfigurieren</b><br>
In der Tabelle:<br>
• Checkbox = Spalte ein/ausschalten<br>
• Doppelklick auf Name = Spalte umbenennen<br>
Reihenfolge der Tabelle = Reihenfolge in der Excel-Datei</div>

<div class="step"><b>4 – Modus &amp; Exportieren</b><br>
<b>Überschreiben:</b> Neue Datei / bestehende Datei komplett ersetzen<br>
<b>Anhängen:</b> Zeilen an bestehende Datei hinzufügen (gleiche Spalten!)<br>
Dann <i>Excel exportieren</i> klicken.</div>

<div class="warn"><b>Voraussetzung:</b> <code>pip install openpyxl</code> muss installiert sein.</div>

<h2>Besonderheiten</h2>
<ul>
  <li>Zeilen unter Konfidenz-Schwellwert werden <b>rot</b> markiert</li>
  <li>Kopfzeilen werden fett und farbig formatiert</li>
  <li>Top-2 und Top-3 Vorhersagen können optional hinzugefügt werden</li>
</ul>
"""),

# ── 9  Einstellungen ──────────────────────────────────────────────────────────
9: page("""
<h1>⚙ Einstellungen</h1>
<p>Alle Einstellungen werden automatisch gespeichert (QSettings) und beim nächsten Start wiederhergestellt.
Nach Änderungen <b>„Einstellungen speichern"</b> klicken.</p>

<h2>Erscheinungsbild</h2>
<table>
  <tr><th>Einstellung</th><th>Standard</th><th>Beschreibung</th></tr>
  <tr><td>Design</td><td>dark</td><td><b>dark</b> = dunkles Theme (Empfehlung für Industrie) | <b>light</b> = helles Theme – sofort wirksam</td></tr>
  <tr><td>Schriftgröße</td><td>9 pt</td><td>7–16 pt – wirkt nach Neustart vollständig</td></tr>
</table>

<h2>Projekt &amp; Autosave</h2>
<table>
  <tr><th>Einstellung</th><th>Standard</th><th>Beschreibung</th></tr>
  <tr><td>Autosave aktiviert</td><td>Ja</td><td>Projekt automatisch im eingestellten Intervall speichern</td></tr>
  <tr><td>Autosave-Intervall</td><td>300 s</td><td>30–3600 Sekunden. Alternativ manuell <kbd>Strg+S</kbd> nutzen</td></tr>
  <tr><td>Backup vor Speichern</td><td>Ja</td><td>Erstellt bei jedem Speichern eine <code>.bak</code>-Sicherungskopie im Projektordner</td></tr>
</table>

<h2>Labeling</h2>
<table>
  <tr><th>Einstellung</th><th>Standard</th><th>Beschreibung</th></tr>
  <tr><td>Thumbnail-Größe</td><td>100 px</td><td>60–240 px – kleinere Werte beschleunigen das Laden bei vielen Bildern</td></tr>
  <tr><td>ROI-Labels im Editor anzeigen</td><td>Ja</td><td>Blendet Label-Texte direkt auf die ROI-Rahmen im Bildeditor ein</td></tr>
</table>

<h2>Inferenz</h2>
<table>
  <tr><th>Einstellung</th><th>Standard</th><th>Beschreibung</th></tr>
  <tr><td>Schwelle 'unsicher'</td><td>0.70</td><td>Bilder unter diesem Konfidenzwert erscheinen im Niedrig-Konfidenz-Tab</td></tr>
  <tr><td>Standard Top-K</td><td>3</td><td>Anzahl angezeigte Top-Vorhersagen (1–5) in der Klassifikations-Ansicht</td></tr>
</table>

<hr>

<h2>REST-API Server</h2>
<p>Integrierter HTTP-Server für externe Steuerung und Monitoring.</p>
<div class="step"><b>API starten</b><br>
Port einstellen (Standard: <code>8765</code>) → <i>API starten</i> klicken.<br>
Status wechselt auf grün. Mit <i>URL kopieren</i> die Basis-URL in die Zwischenablage kopieren.</div>

<div class="step"><b>📊 Dashboard</b><br>
Öffnet ein Live-Monitoring-Dashboard im Browser (aktualisiert alle 3 Sekunden).<br>
Zeigt Projektstatus, Labels, aktuelle Scores und Anomalie-Events.</div>

<h3>API-Endpunkte (alle GET außer label/multilabel)</h3>
<table>
  <tr><th>Methode</th><th>Endpunkt</th><th>Beschreibung</th></tr>
  <tr><td><code>GET</code></td><td><code>/api/status</code></td><td>Server-Status und Versionsinformation</td></tr>
  <tr><td><code>GET</code></td><td><code>/api/project</code></td><td>Projektübersicht (Name, Bilder, Labels)</td></tr>
  <tr><td><code>GET</code></td><td><code>/api/labels</code></td><td>Alle Label-Definitionen (Name, Farbe)</td></tr>
  <tr><td><code>GET</code></td><td><code>/api/images</code></td><td>Alle Bilder mit zugewiesenen Labels</td></tr>
  <tr><td><code>GET</code></td><td><code>/api/images/&lt;name&gt;</code></td><td>Einzelbild-Details inkl. ROIs</td></tr>
  <tr><td><code>POST</code></td><td><code>/api/images/label</code></td><td>Label zuweisen: <code>{"path":"...","label":"..."}</code></td></tr>
  <tr><td><code>POST</code></td><td><code>/api/images/multilabel</code></td><td>Multi-Label: <code>{"path":"...","labels":[...]}</code></td></tr>
  <tr><td><code>GET</code></td><td><code>/api/scores</code></td><td>Live-Score-Puffer (Anomalie-Erkennung)</td></tr>
  <tr><td><code>GET</code></td><td><code>/api/events</code></td><td>Anomalie-Event-Liste</td></tr>
  <tr><td><code>GET</code></td><td><code>/dashboard</code></td><td>HTML Live-Dashboard</td></tr>
</table>

<div class="tip"><b>Beispiel-Aufruf:</b><br>
<code>curl http://localhost:8765/api/status</code><br>
<code>curl -X POST http://localhost:8765/api/images/label -H "Content-Type: application/json" -d '{"path":"/pfad/bild.jpg","label":"gut"}'</code></div>

<hr>

<h2>MQTT-Alarm (Anomalie-Erkennung)</h2>
<p>Bei jedem Anomalie-Alarm wird ein JSON-Event an einen MQTT-Broker gesendet.
Ideal für Industrie-4.0-Anbindung, Home-Automation (Node-RED, SCADA, etc.).</p>

<div class="step"><b>MQTT einrichten</b><br>
1. <b>MQTT-Publishing aktiviert</b> – Checkbox aktivieren<br>
2. <b>Broker-Host</b> – Hostname oder IP des MQTT-Brokers (z. B. <code>localhost</code>, <code>192.168.1.10</code>)<br>
3. <b>Port</b> – Standard MQTT-Port ist <code>1883</code> (SSL: 8883)<br>
4. <b>Topic</b> – MQTT-Pfad für die Nachrichten (Standard: <code>picture_studio/anomaly</code>)<br>
5. <b>Benutzername / Passwort</b> – optional, leer lassen wenn keine Authentifizierung nötig<br>
6. <b>Einstellungen speichern</b> klicken</div>

<h3>Gesendetes JSON-Payload (Beispiel)</h3>
<table>
  <tr><th>Feld</th><th>Beschreibung</th></tr>
  <tr><td><code>event</code></td><td><code>"anomaly"</code></td></tr>
  <tr><td><code>timestamp_utc</code></td><td>ISO-8601 Zeitstempel des Alarms</td></tr>
  <tr><td><code>score</code></td><td>Rekonstruktionsfehler (MSE-Wert)</td></tr>
  <tr><td><code>threshold</code></td><td>Aktueller Schwellwert</td></tr>
  <tr><td><code>saved_frame</code></td><td>Pfad zum gespeicherten Anomalie-Frame (falls aktiviert)</td></tr>
  <tr><td><code>camera_source</code></td><td>Index oder URL der Kamera</td></tr>
</table>

<div class="warn"><b>Voraussetzung:</b> <code>pip install paho-mqtt</code> muss installiert sein.<br>
Status in den Einstellungen zeigt <span style="color:#F85149">„paho-mqtt nicht installiert"</span>
wenn die Bibliothek fehlt.</div>

<div class="tip"><b>Lokaler Test-Broker:</b><br>
Mosquitto (open source) lokal starten: <code>brew install mosquitto &amp;&amp; mosquitto</code><br>
Empfangen: <code>mosquitto_sub -t "picture_studio/#"</code></div>

<hr>

<h2>SSH-Profile</h2>
<p>Profile für SSH-Ferntraining auf externen GPU-Servern.</p>
<div class="step"><b>Profil hinzufügen</b><br>
• <b>Profilname:</b> Bezeichnung (z. B. "GPU-Server Halle 1")<br>
• <b>Host:</b> Hostname oder IP-Adresse des Servers<br>
• <b>Benutzername:</b> SSH-Login-Name<br>
• <b>SSH-Key-Pfad:</b> Pfad zum privaten SSH-Schlüssel (empfohlen statt Passwort)<br>
  Beispiel: <code>~/.ssh/id_rsa</code> oder <code>~/.ssh/gpu_server_key</code></div>

<div class="tip"><b>SSH-Key erstellen:</b><br>
<code>ssh-keygen -t ed25519 -f ~/.ssh/gpu_key</code><br>
Public Key auf Server übertragen: <code>ssh-copy-id -i ~/.ssh/gpu_key.pub user@server</code></div>

<div class="tip"><b>Einstellungen gespeichert in:</b><br>
macOS: <code>~/Library/Preferences</code> (QSettings)<br>
Windows: Registry <code>HKEY_CURRENT_USER\\Software\\ImageLabelingStudio</code></div>
"""),

# ── 10  Kamera & Videoanalyse ─────────────────────────────────────────────────
10: page("""
<h1>📷 Kamera &amp; Videoanalyse</h1>
<p>Live-Aufnahme von USB-/IP-Kameras, Video-Datei-Analyse, automatische Anomalieerkennung.</p>
<p>Öffnen über: <i>Datei → Kamera aufnehmen…</i> (<kbd>Strg+K</kbd>)</p>

<hr>
<h2>Kameraquellen</h2>

<h3>USB-Kamera</h3>
<div class="step"><b>Tab „USB Kamera"</b><br>
1. Kamera im Dropdown wählen (werden automatisch mit Systemname erkannt)<br>
2. <i>Verbinden</i> klicken → Live-Vorschau erscheint<br>
3. Kamera nicht sichtbar? → <i>Kameras neu suchen</i> klicken</div>

<h3>IP-Kamera / Netzwerkkamera</h3>
<div class="step"><b>Tab „IP Kamera"</b><br>
URL eingeben und <i>Verbinden</i> klicken.<br>
Unterstützte Protokolle:
<ul>
  <li><code>rtsp://user:pass@192.168.1.100:554/stream</code> – RTSP (IP-Kameras, NVR)</li>
  <li><code>http://192.168.1.100:8080/video</code> – HTTP-MJPEG-Stream</li>
</ul></div>

<h3>Video-Datei</h3>
<div class="step"><b>Tab „Video-Datei"</b><br>
<i>Datei wählen…</i> → MP4, AVI, MOV, MKV, WebM wählen.<br>
<b>Wiedergabe fps:</b> Auf 0 lassen = originale Videogeschwindigkeit.<br>
Der Fortschrittsbalken zeigt Frame-Nummer / Gesamt während der Wiedergabe.</div>

<hr>
<h2>Aufnahme-Funktionen</h2>

<div class="step"><b>Einzelbild aufnehmen</b><br>
<i>Bild aufnehmen</i> oder <kbd>Leertaste</kbd>.<br>
Bilder werden als PNG im gewählten Speicherordner gesichert.</div>

<div class="step"><b>Burst-Aufnahme</b><br>
Anzahl Bilder + Intervall einstellen → <i>Burst starten</i>.<br>
Beispiel: 20 Bilder im 0,5-Sekunden-Abstand für schnelle Prozesse.</div>

<div class="step"><b>⏺ Live-Aufzeichnung (MP4)</b><br>
<i>Aufnahme starten</i> klicken → laufendes MP4 wird gespeichert.<br>
FPS-Wert rechts daneben einstellen (Standard: 15 fps).<br>
<i>Aufnahme stoppen</i> → Datei wird finalisiert.</div>

<h2>Zeitstempel</h2>
<div class="step"><b>Zeitstempel-Optionen</b><br>
• <b>Im Vorschaubild anzeigen</b> – blendet Datum/Uhrzeit live ein (ohne Einfluss auf Dateien)<br>
• <b>In gespeichertes Bild einbrennen</b> – Zeitstempel dauerhaft in PNG gerendert<br>
Format: <code>YYYY-MM-DD  HH:MM:SS</code>, weiße Schrift unten links</div>

<hr>
<h2>Anomalie-Erkennung (Autoencoder)</h2>
<p>Erkennt Abweichungen vom Normalablauf – ohne Anomalie-Beispiele zu benötigen.</p>

<div class="tip"><b>Funktionsprinzip:</b>
Ein Conv-Autoencoder lernt ausschließlich auf <b>normalen Frames</b> wie der Prozess aussieht.
Bei einem unbekannten Frame (Fehler, Störung) kann das Netz ihn nicht gut rekonstruieren →
der <b>Rekonstruktionsfehler (MSE)</b> überschreitet den Schwellwert → Alarm.
</div>

<h3>Schritt 0 – ROI (Analysebereich) – optional</h3>
<div class="step">Im Vorschaubild einen Bereich aufziehen (Klick auf <i>ROI aufziehen</i>, dann im Bild ziehen).<br>
<b>Nur dieser Bereich</b> fließt in Training und Scoring ein – Hintergrundbewegungen werden ignoriert.<br>
Empfehlung: immer einen ROI setzen wenn der Prozess auf eine Stelle begrenzt ist.<br>
<i>ROI löschen</i> → das gesamte Bild wird wieder analysiert.</div>

<h3>Schritt 1 – Normalframes aufnehmen</h3>
<div class="step">Prozess normal ablaufen lassen.<br>
<b>Anzahl</b> einstellen (empfohlen: 150–500 Frames, bis zu 25.000 möglich).<br>
<i>Aufnehmen starten</i> → Frames werden automatisch gesammelt.<br>
Alle Varianten des Normalzustands abdecken (z. B. verschiedene Werkstücke, Helligkeitsschwankungen).<br>
<i>Löschen</i> → gesammelte Frames verwerfen und neu starten.</div>

<h3>Schritt 2 – Autoencoder trainieren</h3>
<div class="step"><b>Epochen</b> einstellen (Standard: 40, mehr = besser aber langsamer).<br>
<i>Training starten</i> → Loss und Fortschritt werden live angezeigt.<br>
Nach dem Training wird der Schwellwert automatisch berechnet:<br>
<code>Schwellwert = Mittelwert + 2,5 × Standardabweichung</code> der Trainings-Rekonstruktionsfehler.</div>

<h3>Schritt 3 – Live-Erkennung</h3>
<div class="step"><b>Scoring aktiv</b> Button aktivieren.<br>
<b>Score-Anzeige:</b> Grün = Normal | Rot = Anomalie<br>
<b>Roter Banner</b> oben im Vorschaubild bei Alarm.<br>
<b>Heatmap-Overlay:</b> zeigt welcher Bereich abweicht (rot = hoher Fehler).<br>
<b>Rekonstruktions-Tab</b> (🔮 Modell): zeigt das rekonstruierte Bild des Autoencoders.</div>

<h3>Schwellwert &amp; Feintuning</h3>
<table>
  <tr><th>Einstellung</th><th>Beschreibung</th></tr>
  <tr><td>Schwellwert</td><td>Automatisch gesetzt. Manuell erhöhen = weniger Fehlalarme, senken = sensitiver</td></tr>
  <tr><td>Glättung</td><td>Alarm erst nach N aufeinanderfolgenden Frames über Schwellwert (Standard: 5). Verhindert Fehlalarme durch kurze Störungen</td></tr>
  <tr><td>Alarm-Pause</td><td>Mindestabstand zwischen zwei gespeicherten Events (Sekunden). Verhindert hunderte Duplikate bei anhaltender Anomalie</td></tr>
  <tr><td>📊 Kalibrieren…</td><td>Histogramm der Score-Verteilung anzeigen mit automatischen Vorschlägen (µ+1σ, µ+2σ, µ+3σ)</td></tr>
</table>

<h3>Bewegungsfilter</h3>
<div class="step"><b>„Nur bei Bewegung prüfen"</b> – Checkbox aktivieren.<br>
Frames ohne Bewegung werden übersprungen – der Autoencoder läuft nur bei Veränderung im Bild.<br>
<b>Sensitivität</b> (1–100 %): minimaler Pixelunterschied der als Bewegung gilt.<br>
Spart CPU und verhindert Fehlalarme bei statischer Kamera ohne Aktivität.</div>

<h3>Anomalie-Frames automatisch speichern</h3>
<div class="step">Checkbox <i>Anomalie-Frames automatisch speichern</i> aktivieren.<br>
Jeder Alarm-Frame wird als PNG mit rotem Bounding-Box-Rahmen um die anomale Region gespeichert.<br>
Eine JSON-Sidecar-Datei mit Score, Schwellwert und Zeitstempel wird angelegt.<br>
<b>False Positive markieren:</b> Rechtsklick auf Event in der Liste → Sidecar-Datei wird aktualisiert.<br>
Events erscheinen in der <b>Ereignis-Liste</b> rechts im Dialog.</div>

<h3>Modell verwalten</h3>
<div class="step"><b>Speichern…</b> → Autoencoder als <code>.pth</code>-Datei sichern (inkl. SHA256-Prüfsumme).<br>
<b>Laden…</b> → gespeichertes Modell laden – Prüfsumme wird automatisch verifiziert.<br>
<b>ℹ Info</b> → zeigt alle Metadaten: Trainingszeit, Frames, Epochen, Gerät, Schwellwert, SHA256.<br>
<b>ONNX exportieren</b> → <code>.onnx</code>-Datei (Opset 17) für andere Systeme.<br>
<b>TorchScript exportieren</b> → <code>.pt</code>-Datei für PyTorch C++ API.</div>

<h3>Audit-Log</h3>
<div class="step">Jede Modell-Aktion (TRAINED / SAVED / LOADED / UNLOADED) wird automatisch in<br>
<code>audit/model_audit.jsonl</code> im Projektordner protokolliert.<br>
<i>Log öffnen</i> → CSV-Ereignislog der Alarme öffnen.</div>

<hr>
<h2>Batch-Analyse (Ordner)</h2>
<div class="step"><b>Tab „📁 Batch"</b> (rechte Seite des Dialogs)<br>
<i>Ordner wählen…</i> oder <i>Dateien wählen…</i> → Bilder oder Frames auswählen.<br>
<i>Batch starten</i> → alle Bilder werden mit dem aktuellen Autoencoder bewertet.<br>
Ergebnis: Score, Anomalie ja/nein, farbige Markierung.<br>
<i>CSV exportieren</i> → Ergebnisse als Tabelle speichern.</div>

<hr>
<h2>Bilder ins Projekt übernehmen</h2>
<div class="step">Aufgenommene Bilder erscheinen in der Liste unten links.<br>
<i>In Projekt übernehmen</i> → Bilder werden dem Projekt hinzugefügt und sind sofort in der Labeling-Seite verfügbar.</div>

<div class="tip"><b>Tipps für gute Erkennung:</b>
<ul>
  <li>ROI setzen – nur den relevanten Prozessbereich analysieren</li>
  <li>Gleichmäßige Beleuchtung beim Aufnehmen der Normalframes</li>
  <li>Alle Varianten des Normalablaufs abdecken</li>
  <li>Kamera-Position nach dem Training nicht verändern</li>
  <li>Bewegungsfilter nutzen bei statischer Kamera</li>
  <li>Alarm-Pause auf 30–60 s setzen um Duplikate zu vermeiden</li>
</ul></div>
"""),

# ── 11  Tastenkürzel ──────────────────────────────────────────────────────────
11: page("""
<h1>⌨ Tastenkürzel</h1>

<h2>Anwendung</h2>
<table>
  <tr><th>Kürzel</th><th>Aktion</th></tr>
  <tr><td><kbd>Strg+N</kbd></td><td>Neues Projekt</td></tr>
  <tr><td><kbd>Strg+O</kbd></td><td>Projekt öffnen</td></tr>
  <tr><td><kbd>Strg+S</kbd></td><td>Projekt speichern</td></tr>
  <tr><td><kbd>Strg+Umschalt+S</kbd></td><td>Speichern unter…</td></tr>
  <tr><td><kbd>Strg+L</kbd></td><td>Labels verwalten</td></tr>
  <tr><td><kbd>Strg+K</kbd></td><td>Kamera aufnehmen</td></tr>
  <tr><td><kbd>Strg+Q</kbd></td><td>Beenden</td></tr>
  <tr><td><kbd>F1</kbd></td><td>Hilfe für aktuelle Seite</td></tr>
</table>

<h2>ROI-Editor (Labeling-Seite)</h2>
<table>
  <tr><th>Taste</th><th>Aktion</th></tr>
  <tr><td><kbd>R</kbd></td><td>Rechteck-Modus</td></tr>
  <tr><td><kbd>E</kbd></td><td>Ellipse-Modus</td></tr>
  <tr><td><kbd>G</kbd></td><td>Polygon-Modus</td></tr>
  <tr><td><kbd>Esc</kbd></td><td>Zeichnen abbrechen</td></tr>
  <tr><td><kbd>Entf</kbd></td><td>Ausgewählte ROI löschen</td></tr>
  <tr><td><kbd>Strg+C</kbd></td><td>ROI kopieren</td></tr>
  <tr><td><kbd>Strg+V</kbd></td><td>ROI einfügen</td></tr>
  <tr><td><kbd>↑ ↓ ← →</kbd></td><td>ROI um 2 px verschieben</td></tr>
  <tr><td><kbd>1</kbd>–<kbd>9</kbd></td><td>Label schnell zuweisen</td></tr>
  <tr><td><kbd>N</kbd></td><td>Nächstes Bild</td></tr>
  <tr><td><kbd>P</kbd></td><td>Vorheriges Bild</td></tr>
</table>

<h2>Kamera-Dialog</h2>
<table>
  <tr><th>Taste</th><th>Aktion</th></tr>
  <tr><td><kbd>Leertaste</kbd></td><td>Einzelbild aufnehmen (wenn verbunden)</td></tr>
</table>
"""),

# ── 12  Fehlerbehebung ────────────────────────────────────────────────────────
12: page("""
<h1>🔧 Fehlerbehebung</h1>

<h2>Anwendung startet nicht</h2>
<div class="step"><code>pip install PySide6</code> installieren.<br>
Linux: Qt-Plugins: <code>apt install libxcb-cursor0</code></div>

<h2>Training sehr langsam</h2>
<div class="step">Gerät auf <code>cuda</code> oder <code>mps</code> stellen.<br>
CPU-Test: Bildgröße 128 px, Batch-Größe 8, SimpleCNN-Architektur.</div>

<h2>ImportError: openpyxl</h2>
<div class="step"><code>pip install openpyxl</code> – für Excel-Export erforderlich.</div>

<h2>ImportError: paramiko</h2>
<div class="step"><code>pip install paramiko</code> – nur für SSH-Ferntraining.</div>

<h2>MQTT funktioniert nicht</h2>
<div class="step"><code>pip install paho-mqtt</code> – für MQTT-Alarm erforderlich.<br>
Status in Einstellungen prüfen: zeigt <i>„paho-mqtt nicht installiert"</i> wenn fehlend.<br>
Broker-Verbindung testen: <code>mosquitto_pub -h localhost -t test -m hello</code><br>
Benutzername/Passwort: nur eintragen wenn der Broker Authentifizierung erfordert.</div>

<h2>Charts erscheinen nicht</h2>
<div class="step"><code>pip install matplotlib</code> – sonst ASCII-Sparklines.</div>

<h2>Thumbnails laden langsam</h2>
<div class="step">Thumbnail-Größe in Einstellungen reduzieren (z. B. 60 px).</div>

<h2>Projektdatei beschädigt</h2>
<div class="step">Die <code>.bak</code>-Datei neben der Projektdatei umbenennen:<br>
<code>projekt.bak</code> → <code>projekt.json</code></div>

<h2>SSH-Verbindung schlägt fehl</h2>
<div class="step">
• Host, Benutzername und Key-Pfad in Einstellungen prüfen<br>
• SSH-Agent: <code>ssh-add &lt;key-pfad&gt;</code><br>
• Manuell testen: <code>ssh user@host</code> im Terminal<br>
• Key-Berechtigungen: <code>chmod 600 ~/.ssh/id_rsa</code></div>

<h2>Kamera wird nicht gefunden</h2>
<div class="step"><code>pip install opencv-python</code> installieren.<br>
<i>Kameras neu suchen</i> klicken.<br>
Andere Anwendungen schließen, die die Kamera blockieren.<br>
macOS: Kamera-Zugriff in <i>Systemeinstellungen → Datenschutz → Kamera</i> prüfen.<br>
iPhone/iPad-Kameras werden absichtlich ausgeblendet (AVFoundation-Kompatibilität).<br>
IP-Kamera: URL im Browser prüfen ob der Stream erreichbar ist.</div>

<h2>Anomalie-Score immer 0</h2>
<div class="step">Autoencoder muss erst trainiert sein (Schritt 2).<br>
<i>Scoring aktiv</i> Button prüfen – muss aktiviert (grün) sein.</div>

<h2>Viele Fehlalarme (False Positives)</h2>
<div class="step">
• Schwellwert erhöhen oder <i>📊 Schwellwert kalibrieren…</i> verwenden<br>
• Glättung auf 5–10 Frames erhöhen<br>
• Mehr und vielfältigere Normalframes sammeln und neu trainieren<br>
• ROI setzen um Hintergrundbewegungen auszuschließen<br>
• Bewegungsfilter aktivieren bei statischer Szene</div>

<h2>SHA256-Prüfsummen-Fehler beim Modell laden</h2>
<div class="step">Die <code>.pth</code>-Datei wurde nach dem Speichern verändert oder ist beschädigt.<br>
Modell erneut speichern oder neu trainieren.<br>
Die <code>.pth.sha256</code>-Datei muss immer neben der <code>.pth</code>-Datei liegen.</div>

<h2>Zeitstempel erscheint nicht im gespeicherten Bild</h2>
<div class="step">Checkbox <i>In gespeichertes Bild einbrennen</i> aktivieren.<br>
Die Vorschau-Checkbox beeinflusst nur das Live-Bild, nicht die gespeicherte Datei.</div>

<h2>Video-Datei öffnet nicht</h2>
<div class="step">OpenCV muss das Format unterstützen (MP4, AVI, MOV, MKV, WebM).<br>
Bei Codec-Problemen: Video mit <code>ffmpeg</code> in H.264 MP4 konvertieren:<br>
<code>ffmpeg -i input.avi -c:v libx264 output.mp4</code></div>
"""),

# ── 13  Monitor-Client ────────────────────────────────────────────────────────
13: page("""
<h2>💻 Monitor-Client</h2>
<p>Der <b>Monitor-Client</b> ist ein eigenständiges Kommandozeilen-Werkzeug für den Produktionseinsatz.
Er lädt ein trainiertes Anomalie-Modell und verbindet sich automatisch mit der Kamera, die beim Training verwendet wurde.</p>

<div class="step">
<b>Schnellstart</b><br>
<code>python monitor.py --model pfad/zum/modell.pt</code><br>
Kamera, ROI und Schwellwert werden automatisch aus den Modell-Metadaten geladen.
</div>

<h3>Alle Optionen</h3>
<table>
<tr><th>Option</th><th>Standard</th><th>Beschreibung</th></tr>
<tr><td><code>--model PFAD</code></td><td>–</td><td>Pfad zur .pt-Modelldatei (Pflicht)</td></tr>
<tr><td><code>--camera INDEX</code></td><td>auto</td><td>Kamera-Index manuell überschreiben</td></tr>
<tr><td><code>--threshold WERT</code></td><td>aus Modell</td><td>Anomalie-Schwellwert überschreiben</td></tr>
<tr><td><code>--output VERZ</code></td><td>monitor_logs</td><td>Ausgabeverzeichnis für Logs und Alarm-Bilder</td></tr>
<tr><td><code>--fps FPS</code></td><td>15</td><td>Bilder pro Sekunde</td></tr>
<tr><td><code>--cooldown SEK</code></td><td>10</td><td>Mindestabstand zwischen Alarm-Saves</td></tr>
<tr><td><code>--headless</code></td><td>aus</td><td>Kein Fenster — nur Terminal + CSV</td></tr>
</table>

<h3>Ausgabe-Dateien</h3>
<div class="tip">
Im Ausgabeverzeichnis (Standard: <code>monitor_logs/</code>) werden gespeichert:<br>
• <b>monitor_events.csv</b> — Alle Alarm-Events mit Zeitstempel, Score, Schwellwert und Bildname<br>
• <b>alarm_YYYYMMDDTHHMMSSZ.jpg</b> — Kamera-Schnappschuss bei jedem Alarm
</div>

<h3>Typische Szenarien</h3>
<div class="step"><b>Produktionslinie überwachen</b><br>
<code>python monitor.py --model modelle/linie1.pt --headless --output /var/log/anomalien</code>
</div>
<div class="step"><b>Schwellwert anpassen</b><br>
<code>python monitor.py --model modelle/linie1.pt --threshold 0.0015</code>
</div>
<div class="step"><b>Andere Kamera verwenden</b><br>
<code>python monitor.py --model modelle/linie1.pt --camera 2</code>
</div>

<div class="tip">
<b>Tipp:</b> Der Monitor-Client benötigt kein vollständiges PictureStudio — nur Python mit den installierten Abhängigkeiten (PyTorch, OpenCV, NumPy).
</div>
<div class="warn">
<b>Beenden:</b> Im Fenster-Modus Q oder ESC drücken. Im Headless-Modus Strg+C.
</div>
"""),

}


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class HelpDialog(QDialog):
    """Help dialog with sidebar navigation and content browser."""

    def __init__(self, page_index: int = 0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hilfe – Picture Studio")
        self.setMinimumSize(880, 600)
        self.resize(1060, 720)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left nav ─────────────────────────────────────────────────────────
        nav_frame = QFrame()
        nav_frame.setFixedWidth(210)
        nav_frame.setStyleSheet("background:#161625; border-right:1px solid #2C3E50;")
        nav_vbox = QVBoxLayout(nav_frame)
        nav_vbox.setContentsMargins(0, 0, 0, 0)
        nav_vbox.setSpacing(0)

        title_lbl = QLabel("  Hilfe")
        title_lbl.setFixedHeight(48)
        title_lbl.setStyleSheet(
            "color:#5DADE2; font-size:15px; font-weight:bold;"
            "background:#0D1117; border-bottom:1px solid #2C3E50;"
        )
        nav_vbox.addWidget(title_lbl)

        self._nav = QListWidget()
        self._nav.setStyleSheet("""
            QListWidget {
                background: #161625;
                border: none;
                font-size: 12px;
                color: #BDC3C7;
                outline: none;
            }
            QListWidget::item {
                padding: 9px 14px;
                border-bottom: 1px solid #1C2A3A;
            }
            QListWidget::item:selected {
                background: #1A5276;
                color: white;
                border-left: 3px solid #3498DB;
            }
            QListWidget::item:hover:!selected {
                background: #1C3A52;
                color: white;
            }
        """)
        for icon, label in SECTIONS:
            self._nav.addItem(f"{icon}  {label}")
        nav_vbox.addWidget(self._nav)
        root.addWidget(nav_frame)

        # ── Right content ─────────────────────────────────────────────────────
        self._browser = QTextBrowser()
        self._browser.setOpenExternalLinks(False)
        self._browser.setStyleSheet(
            "QTextBrowser {"
            "  background: #0D1117;"
            "  border: none;"
            "  color: #E0E0E0;"
            "}"
        )
        root.addWidget(self._browser)

        self._nav.currentRowChanged.connect(self._show)

        # Jump to the section matching the current page
        self._nav.setCurrentRow(PAGE_TO_SECTION.get(page_index, 0))

    def _show(self, row: int) -> None:
        html = CONTENT.get(row, "<p style='color:#aaa;padding:20px'>Kein Inhalt.</p>")
        self._browser.setHtml(html)
        self._browser.verticalScrollBar().setValue(0)
