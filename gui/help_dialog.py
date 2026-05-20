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
    ("💻", "Monitor-Client"),         # 13
    ("📹", "Multi-Kamera"),           # 14
    ("🔬", "Anomalie-Clustering"),    # 15
    ("📈", "Datensatz-Statistiken"),  # 16
    ("🎬", "Video-Annotation"),       # 17
    ("🌐", "Fleet-Management"),       # 18
    ("⚡", "Modelle Erweitert"),      # 19
    ("🔗", "Kontakt & Repository"),   # 20
]

# Map sidebar page index → section index
PAGE_TO_SECTION = {
    0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9,
    10: 14, 11: 15,
    12: 16, 13: 17, 14: 18,
}

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
  <tr><td><b>REST-API</b></td><td>HTTP-Server (Port konfigurierbar), optionaler API-Key-Schutz, Label zuweisen per POST, Live-Dashboard im Browser, Per-Kanal-Endpunkte für Multi-Kamera</td></tr>
  <tr><td><b>MQTT-Alarm</b></td><td>JSON-Events bei Anomalie-Alarm an beliebigen Broker (paho-mqtt), auth-fähig</td></tr>
  <tr><td><b>Kamera / Video</b></td><td>USB-Kamera, IP-Kamera (RTSP/HTTP), Video-Datei direkt im Live-Monitor, Auto-Reconnect, Live-Aufzeichnung (MP4), Burst-Modus</td></tr>
  <tr><td><b>Multi-Kamera</b></td><td>1–9 Kanäle gleichzeitig (Selector im Toolbar), dynamisches 2×2-Grid, Seitenblättern bei >4 Kanälen, Alarm-JPEG-Saving pro Kanal, per-Kanal REST-API</td></tr>
  <tr><td><b>Anomalie-Erkennung</b></td><td>Conv-Autoencoder, ROI-Bereich, Bewegungsfilter, Schwellwert-Kalibrierung, Heatmap, Bounding-Box, Alarm-Pause, Audit-Log, False-Positive-Markierung</td></tr>
  <tr><td><b>Batch-Analyse</b></td><td>Ordner oder Dateien auf Anomalien prüfen, CSV-Export der Ergebnisse</td></tr>
  <tr><td><b>Datensatz-Statistiken</b></td><td>Klassenverteilung, Format-/Größenstatistiken, Label-Rate, perceptual-hash Duplikaterkennung</td></tr>
  <tr><td><b>Video-Annotation</b></td><td>Frame-für-Frame-Annotation aus Videodateien, Slider-Navigation, direktes Hinzufügen zum Projekt</td></tr>
  <tr><td><b>Fleet-Management</b></td><td>Zentrale Überwachung mehrerer monitor.py-Instanzen, Status-Polling, Auto-Refresh, QSettings-Persistenz</td></tr>
  <tr><td><b>Hyperparameter-Suche</b></td><td>Optuna-basierte Suche (lr, batch_size, architecture, optimizer), bestes Ergebnis direkt in UI übernehmen</td></tr>
  <tr><td><b>Modell-Kalibrierung</b></td><td>Temperature Scaling (scipy) für korrektere Konfidenzwerte</td></tr>
  <tr><td><b>Edge-Export</b></td><td>ONNX INT8 (onnxruntime.quantization), Apple CoreML (.mlpackage via coremltools)</td></tr>
  <tr><td><b>Docker-Deployment</b></td><td>Einzeilen-Generator für Dockerfile, docker-compose.yml, Startskript und README</td></tr>
  <tr><td><b>Augmentation-Pipeline</b></td><td>Rotation, Flip, Helligkeit, Kontrast, Blur, Rauschen; konfigurierbare Kopien pro Bild</td></tr>
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

<h3>API-Key Authentifizierung</h3>
<div class="step"><b>Warum?</b> Ohne Key ist die API für jeden im Netzwerk erreichbar.<br>
<b>Generieren:</b> <i>Generieren</i>-Button → 64-stelliger Zufallsschlüssel wird erstellt und sofort aktiv.<br>
<b>Löschen:</b> <i>Löschen</i>-Button → Authentifizierung deaktiviert (alle Anfragen erlaubt).<br>
<b>Anzeigen:</b> <i>Anzeigen</i>-Toggle → zeigt den Key im Klartext (standardmäßig verborgen).</div>

<div class="tip"><b>Öffentliche Endpunkte</b> (brauchen keinen Key, auch im Browser erreichbar):<br>
<code>/api/status</code> und <code>/dashboard</code><br>
<b>Alle anderen</b> Endpunkte erfordern den Header:<br>
<code>X-Api-Key: &lt;dein-schlüssel&gt;</code><br>
Alternativ: <code>Authorization: Bearer &lt;dein-schlüssel&gt;</code></div>

<h3>API-Endpunkte (alle GET außer label/multilabel)</h3>
<table>
  <tr><th>Methode</th><th>Endpunkt</th><th>Beschreibung</th></tr>
  <tr><td><code>GET</code></td><td><code>/api/status</code></td><td>Server-Status und Versionsinformation <i>(öffentlich)</i></td></tr>
  <tr><td><code>GET</code></td><td><code>/dashboard</code></td><td>HTML Live-Dashboard <i>(öffentlich)</i></td></tr>
  <tr><td><code>GET</code></td><td><code>/api/project</code></td><td>Projektübersicht (Name, Bilder, Labels)</td></tr>
  <tr><td><code>GET</code></td><td><code>/api/labels</code></td><td>Alle Label-Definitionen (Name, Farbe)</td></tr>
  <tr><td><code>GET</code></td><td><code>/api/images</code></td><td>Alle Bilder mit zugewiesenen Labels</td></tr>
  <tr><td><code>GET</code></td><td><code>/api/images/&lt;name&gt;</code></td><td>Einzelbild-Details inkl. ROIs</td></tr>
  <tr><td><code>POST</code></td><td><code>/api/images/label</code></td><td>Label zuweisen: <code>{"path":"...","label":"..."}</code></td></tr>
  <tr><td><code>POST</code></td><td><code>/api/images/multilabel</code></td><td>Multi-Label: <code>{"path":"...","labels":[...]}</code></td></tr>
  <tr><td><code>GET</code></td><td><code>/api/scores</code></td><td>Live-Score-Puffer (Anomalie-Erkennung)</td></tr>
  <tr><td><code>GET</code></td><td><code>/api/events</code></td><td>Anomalie-Event-Liste</td></tr>
</table>

<div class="tip"><b>Beispiel-Aufruf ohne Key:</b><br>
<code>curl http://localhost:8765/api/status</code><br>
<b>Mit Key:</b><br>
<code>curl http://localhost:8765/api/labels -H "X-Api-Key: dein-schlüssel"</code><br>
<code>curl -X POST http://localhost:8765/api/images/label -H "Content-Type: application/json" -H "X-Api-Key: dein-schlüssel" -d '{"path":"/pfad/bild.jpg","label":"gut"}'</code></div>

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

<h3>Alarmierung (E-Mail & Webhook)</h3>
<p>Bei jeder Anomalie-Erkennung kann automatisch eine Benachrichtigung verschickt werden.</p>
<div class="step">
<b>E-Mail konfigurieren</b><br>
SMTP-Host, Port (587 für TLS), Benutzername und Passwort eintragen. Absender- und Empfängeradressen (kommagetrennt) angeben. Mit "Test-E-Mail senden" prüfen.
</div>
<div class="step">
<b>Webhook konfigurieren</b><br>
Vollständige URL eintragen (z.B. Teams, Slack, eigene API). Der Alarm-Payload wird als JSON-POST gesendet mit: event, timestamp, score, threshold, score_pct, model, frame_file.
</div>
<div class="tip">
<b>Mindestabstand:</b> Der Cooldown-Wert (Standard: 60 Sek.) verhindert Benachrichtigungs-Spam bei anhaltenden Anomalien.
</div>
<div class="tip">
<b>Anhang:</b> Alarm-JPEG wird automatisch an die E-Mail angehängt (max. 2 MB).
</div>

<hr>

<h3>Industrieanbindung (OPC-UA &amp; Modbus TCP)</h3>
<p>Sendet bei Anomalie-Erkennung direkt Signale an SPS-Systeme über OPC-UA oder Modbus TCP.</p>
<table>
<tr><th>Protokoll</th><th>Typische Anwendung</th><th>Standard-Port</th></tr>
<tr><td><b>OPC-UA</b></td><td>Siemens S7, Beckhoff, FANUC CNC</td><td>4840</td></tr>
<tr><td><b>Modbus TCP</b></td><td>Beckhoff, Wago, ältere SPS</td><td>502</td></tr>
</table>
<div class="step">
<b>OPC-UA konfigurieren</b><br>
Server-URL eingeben (z.B. <code>opc.tcp://192.168.1.10:4840</code>), Node-ID des Boolean-Ausgangs
angeben (z.B. <code>ns=2;i=1001</code>). Bei Anomalie wird der Node auf <code>True</code> gesetzt.
</div>
<div class="step">
<b>Modbus TCP konfigurieren</b><br>
Host-IP, Port (Standard 502), Coil-Adresse und Unit-ID eingeben. Bei Anomalie wird
die Coil auf <code>1</code> (True) geschrieben, bei normalem Betrieb auf <code>0</code>.
</div>
<div class="tip">
<b>Verbindungstest:</b> Mit "Verbindung testen" kann die SPS-Verbindung vor dem Live-Betrieb geprüft werden.
</div>
<div class="warn">
Stellt sicher, dass Firewall-Regeln die entsprechenden Ports freigeben (OPC-UA: 4840, Modbus: 502).
</div>

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
<h2>Live-Monitoring (Produktivbetrieb)</h2>
<p>Die <b>Live-Monitoring-Seite</b> ist für den Dauerbetrieb ausgelegt — Modell laden, Kamera verbinden, Scoring aktivieren.</p>

<h3>Kameraquellen im Live-Monitor</h3>
<div class="step">Dropdown enthält alle erkannten USB-Kameras sowie zwei Sondereinträge:<br>
• <b>IP-Kamera (URL eingeben…)</b> – RTSP/HTTP-URL eingeben (Format wird vorab geprüft)<br>
• <b>Videodatei (MP4, AVI, …)</b> – Dateidialog öffnet sich; FPS wird automatisch aus der Datei gelesen</div>

<h3>Auto-Reconnect</h3>
<div class="step">Bricht die Verbindung zu einer <b>Live-Kamera</b> ab, versucht der Monitor automatisch alle <b>5 Sekunden</b> eine Wiederverbindung.<br>
Die Statusanzeige wechselt auf gelb „Reconnect in 5 s…" und zählt die Versuche mit.<br>
Beim ersten erfolgreichen Frame wechselt der Status wieder auf grün „Verbunden".<br>
<b>Manuell stoppen:</b> <i>Trennen</i> klicken – bricht den Reconnect-Zyklus dauerhaft ab.<br>
<b>Hinweis:</b> Bei Video-Dateien gibt es kein Auto-Reconnect (Ende = „Video beendet").</div>

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

<h3>ONNX-Export für Edge-Deployment</h3>
<p>Ein trainiertes Anomalie-Modell kann als ONNX exportiert werden — damit läuft es auf jedem Gerät <b>ohne PyTorch</b> (z.B. Raspberry Pi, Produktions-PC mit onnxruntime).</p>
<div class="step">
<b>1. Modell exportieren</b><br>
Modell in "Live &amp; Anomalie" laden → Schaltfläche <b>"Als ONNX exportieren"</b> klicken → .onnx und .meta.json werden gespeichert.
</div>
<div class="step">
<b>2. ONNX-Modell mit Monitor-Client verwenden</b><br>
<code>python monitor.py --model mein_modell.onnx</code><br>
PyTorch wird nicht benötigt — nur <code>pip install onnxruntime opencv-python numpy</code>.
</div>
<div class="tip">
Die .meta.json Datei enthält Schwellwert und Metadaten und muss neben der .onnx Datei liegen.
</div>
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
<tr><td><code>--model PFAD.onnx</code></td><td>–</td><td>ONNX-Modell laden (kein PyTorch nötig, nur onnxruntime)</td></tr>
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

# ── 14  Multi-Kamera-Monitoring ───────────────────────────────────────────────
14: page("""
<h1>📹 Multi-Kamera-Monitoring</h1>
<p>Überwache <b>1–9 Kamera-Quellen gleichzeitig</b>, jede mit eigenem Modell und ROI.
Ideal für Produktionslinien mit mehreren Prüfstationen.</p>

<hr>
<h2>Anzahl Kanäle festlegen</h2>
<div class="step"><b>Kanäle-Selector (Toolbar oben)</b><br>
Das Drehfeld <i>Kanäle: [2]</i> legt fest, wie viele Kamera-Kanäle gleichzeitig aktiv sind.<br>
Bereich: <b>1–9</b>, Standard: <b>2</b>.<br>
Bei Änderung werden alle laufenden Kanäle gestoppt; bereits konfigurierte Kanäle behalten ihre Einstellungen.</div>

<h2>Grid und Paginierung</h2>
<div class="step"><b>2×2-Grid je Seite</b><br>
Bis zu <b>4 Kanäle</b> werden auf einer Seite im 2×2-Raster angezeigt.<br>
Bei mehr als 4 Kanälen erscheinen automatisch die Schaltflächen <b>◀ Vorherige</b> und <b>Nächste ▶</b>.<br>
Jede Seite zeigt den Ausschnitt z. B. Kanäle 1–4, 5–8, 9 — je nach gewählter Kanalzahl.<br>
Die aktuelle Seite und Gesamtzahl stehen zwischen den Buttons: <i>Seite 1 / 3</i>.</div>

<hr>
<h2>Kanal einrichten und starten</h2>
<div class="step"><b>1 – Kanal konfigurieren</b><br>
<i>⚙ Konfigurieren</i> im jeweiligen Kanal klicken.<br>
Kamera (USB-Index) und Modell (<code>.pth</code> oder <code>.onnx</code>) auswählen → OK.<br>
Der Kanal zeigt danach Kameraname und Modellname an; die Start-Schaltfläche wird freigeschaltet.</div>

<div class="step"><b>2 – Kanal starten</b><br>
<i>▶ Starten</i> für einzelne Kanäle oder <i>Alle starten</i> für alle konfigurierten Kanäle.<br>
Die Score-Leiste und der Status (<span style="color:#2ECC71">Normal</span> / <span style="color:#E74C3C">ANOMALIE</span>) aktualisieren sich live.</div>

<div class="step"><b>3 – Kanal stoppen</b><br>
<i>Stoppen</i> für einzelne Kanäle oder <i>Alle stoppen</i> für alle.</div>

<hr>
<h2>Alarm-Ereignisse und JPEG-Speicherung</h2>
<div class="step"><b>Alarm-Protokoll</b><br>
Jede erkannte Anomalie erscheint im Protokoll unten mit Zeitstempel, Kanal-Nummer und Score.<br>
E-Mail/Webhook-Benachrichtigungen aus den Einstellungen gelten automatisch für alle Kanäle.</div>

<div class="step"><b>Alarm-JPEG-Speicherung</b><br>
Bei jedem Alarm wird der aktuelle Frame automatisch als JPEG gespeichert:<br>
Pfad: <code>monitor_logs/multi_cam/mc_ch<i>N</i>_<i>YYYYMMDDTHHMMSSZ</i>.jpg</code><br>
Der Dateiname enthält Kanalnummer und UTC-Zeitstempel.</div>

<hr>
<h2>REST-API – Per-Kanal-Endpunkte</h2>
<p>Wenn der REST-Server läuft, stehen neben den Standard-Endpunkten auch drei Multi-Kamera-Endpunkte bereit:</p>

<table>
  <tr><th>Methode</th><th>Endpunkt</th><th>Beschreibung</th></tr>
  <tr><td><code>GET</code></td><td><code>/api/mc/channels</code></td><td>Zusammenfassung aller Kanäle (Score, Schwellwert, Alarm-Zähler, Kamerastatus)</td></tr>
  <tr><td><code>GET</code></td><td><code>/api/mc/scores?channel=N</code></td><td>Rollender Score-Puffer (bis 500 Einträge) für Kanal N</td></tr>
  <tr><td><code>GET</code></td><td><code>/api/mc/latest_alarm?channel=N</code></td><td>Letztes Alarm-Event für Kanal N (Zeitstempel, Score, Dateiname)</td></tr>
</table>

<div class="tip"><b>Beispiel:</b><br>
<code>curl http://localhost:8765/api/mc/channels</code><br>
<code>curl http://localhost:8765/api/mc/latest_alarm?channel=0 -H "X-Api-Key: dein-key"</code></div>

<p>Das Web-Dashboard (<code>/dashboard</code>) zeigt automatisch eine Multi-Kamera-Sektion, sobald Kanäle registriert sind.</p>

<hr>
<h2>Tipps</h2>
<div class="tip"><b>Performance:</b> Jeder Kanal bewertet jeden 3. Frame — bei 4 aktiven Kanälen bleibt die CPU-Last moderat.</div>
<div class="tip"><b>ONNX-Modelle:</b> <code>.onnx</code>-Modelle werden unterstützt und laufen ohne PyTorch (nur onnxruntime nötig).</div>
<div class="warn"><b>Kanalzahl ändern stoppt alle Kanäle</b> — laufende Streams werden beendet. Konfigurierte Kanäle (Modell, Kamera-Index) bleiben erhalten.</div>
"""),


# ── 15  Anomalie-Clustering ───────────────────────────────────────────────────
15: page("""
<h1>🔬 Anomalie-Clustering</h1>
<p>Alarm-Bilder automatisch nach visueller Ähnlichkeit gruppieren – ohne manuelle Annotation.</p>

<hr>
<h2>Was macht Anomalie-Clustering?</h2>
<p>Der k-Means-Algorithmus analysiert die visuellen Merkmale aller Bilder, die als Anomalie gelabelt
wurden, und teilt sie in Gruppen (<b>Cluster</b>) auf. Bilder in einem Cluster sehen sich ähnlich –
z. B. alle Kratzer an einer bestimmten Stelle, alle Farbausreißer oder alle Positionsfehler.</p>
<div class="tip"><b>Wozu ist das nützlich?</b><br>
Statt hunderte Alarm-Bilder manuell durchzusehen bekommst du auf einen Blick, welche
<b>Fehlerarten</b> im Datensatz vorkommen und wie häufig sie sind. Ideal als erster Schritt
zur Fehlerklassifikation.</div>

<hr>
<h2>Schritt-für-Schritt</h2>

<div class="step"><b><span class="num">1</span>Projekt mit Anomalie-Bildern laden</b><br>
Öffne ein Projekt, in dem Bilder mit dem Anomalie-Label (oder einem anderen Fehler-Label)
annotiert sind. Die Seite zeigt automatisch wie viele Bilder für das Clustering zur Verfügung stehen.</div>

<div class="step"><b><span class="num">2</span>Cluster-Anzahl wählen</b><br>
Stelle den Schieberegler oder das Eingabefeld auf die gewünschte Anzahl Cluster ein (2–20).<br>
<b>Empfehlung:</b> Mit <b>5 Clustern</b> starten. Erhöhe die Anzahl, wenn die Ergebnisse noch zu
gemischt wirken – d. h. ein Cluster enthält optisch sehr verschiedene Bilder.</div>

<div class="step"><b><span class="num">3</span>Clustering starten</b><br>
Klicke <i>Clustering starten</i>. Das Modell extrahiert Bildmerkmale und berechnet die Cluster.
Der Vorgang dauert je nach Bildanzahl wenige Sekunden bis ca. eine Minute.</div>

<div class="step"><b><span class="num">4</span>Ergebnisse im Cluster-Browser ansehen</b><br>
Nach der Berechnung erscheinen die Cluster als Karten:<br>
<ul>
  <li>Jede Karte zeigt das <b>repräsentative Bild</b> (den Cluster-Mittelpunkt) als Thumbnail</li>
  <li>Darunter steht die Anzahl der Bilder in diesem Cluster</li>
  <li>Klicke eine Karte an um alle Bilder des Clusters in der Thumbnail-Liste zu sehen</li>
</ul></div>

<div class="step"><b><span class="num">5</span>CSV exportieren</b><br>
Klicke <i>CSV exportieren</i> um die Clustering-Ergebnisse als Tabelle zu speichern.</div>

<hr>
<h2>CSV-Export – Spalten</h2>
<table>
  <tr><th>Spalte</th><th>Inhalt</th></tr>
  <tr><td><code>path</code></td><td>Absoluter Dateipfad des Bildes</td></tr>
  <tr><td><code>cluster_id</code></td><td>Cluster-Nummer (0-basiert) dem das Bild zugeordnet wurde</td></tr>
  <tr><td><code>is_representative</code></td><td><code>True</code> für das Bild das dem Cluster-Mittelpunkt am nächsten liegt, sonst <code>False</code></td></tr>
</table>

<hr>
<h2>Tipps &amp; Empfehlungen</h2>
<div class="tip"><b>Startpunkt: 5 Cluster</b><br>
Beginne mit 5 Clustern und erhöhe schrittweise. Sind Bilder eines Clusters optisch zu
verschieden, teile diesen Cluster durch Erhöhung der Gesamtanzahl weiter auf.</div>
<div class="tip"><b>Zu wenige Bilder?</b><br>
k-Means benötigt mindestens so viele Bilder wie Cluster. Bei weniger als 10 Anomalie-Bildern
2–3 Cluster verwenden.</div>
<div class="warn"><b>Cluster ≠ Fehlerklassen</b><br>
Clustering gruppiert nach visueller Ähnlichkeit, nicht nach technischer Fehlerursache.
Die inhaltliche Interpretation der Cluster (z. B. „Cluster 0 = Risse, Cluster 1 = Flecken")
muss manuell erfolgen.</div>
"""),

# ── 16  Datensatz-Statistiken ──────────────────────────────────────────────────
16: page("""
<h1>📈 Datensatz-Statistiken</h1>
<p>Die <b>Datensatz-Statistiken-Seite</b> (Sidebar: Datensatz) analysiert den aktuellen
Datensatz und zeigt detaillierte Qualitätsmetriken.</p>

<h2>Klassenverteilung</h2>
<p>Horizontale Balken zeigen für jede Klasse Anzahl und prozentualen Anteil.
Ein ausgeglichener Datensatz hat alle Balken annähernd gleich lang.
Starkes Ungleichgewicht verschlechtert die Modell-Performance für unterrepräsentierte Klassen.</p>
<div class="tip"><b>Tipp: Klassenausgleich</b><br>
Bei Ungleichgewicht: <b>WeightedRandomSampler</b> auf der Training-Seite aktivieren oder
mehr Bilder der unterrepräsentierten Klassen beschaffen.</div>

<h2>Format- & Größenstatistiken</h2>
<p>Zeigt Bildformate (JPEG, PNG …) und Auflösungsstatistiken (min, max, Median).
Sehr unterschiedliche Größen können die Training-Performance beeinflussen — ggf. Bilder vorverarbeiten.</p>

<h2>Label-Rate</h2>
<p>Zeigt wie viele Bilder bereits ein Label haben. Eine niedrige Label-Rate bedeutet viel noch zu erledigende
Annotation-Arbeit.</p>

<h2>Duplikaterkennung</h2>
<p>Verwendet <b>perceptual hashing (phash)</b> um visuell identische oder sehr ähnliche Bilder zu finden.
Duplikate können das Training verzerren (Overfitting auf wiederholte Bilder).</p>
<div class="warn"><b>Benötigt: imagehash</b><br>
<code>pip install imagehash</code> — ohne dieses Paket wird die Duplikaterkennung übersprungen.</div>

<h2>Analyse aktualisieren</h2>
<p>Klicke <b>Analyse aktualisieren</b> nach dem Hinzufügen neuer Bilder oder Labels.</p>
"""),

# ── 17  Video-Annotation ───────────────────────────────────────────────────────
17: page("""
<h1>🎬 Video-Annotation</h1>
<p>Die <b>Video-Annotation-Seite</b> ermöglicht das direkte Annotieren einzelner Video-Frames
ohne vorherigen Frame-Export.</p>

<h2>Video laden</h2>
<div class="step"><b>Schritt 1</b> – Klicke <b>Video laden…</b> und wähle eine Videodatei
(MP4, AVI, MOV, MKV, …).<br>
Das Video wird geladen und der erste Frame angezeigt.</div>

<h2>Frame-Navigation</h2>
<div class="step"><b>Schritt 2</b> – Verschiebe den <b>Schieberegler</b> um zum gewünschten Frame zu springen.
Die Frame-Nummer und der Zeitstempel werden oben angezeigt.</div>

<h2>Frame extrahieren & labeln</h2>
<div class="step"><b>Schritt 3</b> – Klicke <b>Frame extrahieren</b> um den aktuellen Frame als Bild zu speichern.<br>
Wähle das Label aus dem Dropdown.<br>
Klicke <b>Zum Projekt hinzufügen</b> um Frame + Label dem aktuellen Projekt hinzuzufügen.</div>

<div class="tip"><b>Tipp: Mehrere Frames</b><br>
Navigiere zu verschiedenen Frames und füge jeden relevanten Frame einzeln hinzu.
So bekommst du einen vielfältigen Trainingsdatensatz aus einem einzigen Video.</div>

<div class="warn"><b>Benötigt: OpenCV</b><br>
<code>pip install opencv-python</code> — ohne OpenCV ist Frame-Extraktion nicht möglich.</div>
"""),

# ── 18  Fleet-Management ───────────────────────────────────────────────────────
18: page("""
<h1>🌐 Fleet-Management</h1>
<p>Die <b>Fleet-Seite</b> überwacht mehrere remote <code>monitor.py</code>-Instanzen
(z. B. auf Edge-Geräten, Industrierechnern oder Cloud-VMs) von einer zentralen Stelle aus.</p>

<h2>Gerät hinzufügen</h2>
<div class="step"><b>Schritt 1</b> – Klicke <b>+ Gerät hinzufügen</b>.<br>
Gib einen Namen, die Basis-URL (z. B. <code>http://192.168.1.100:8765</code>) und
optional einen API-Key ein.<br>
Die URL muss auf einen laufenden <code>monitor.py --api-port</code> Prozess zeigen.</div>

<h2>Status prüfen</h2>
<div class="step"><b>Schritt 2</b> – Klicke <b>Alle aktualisieren</b> oder aktiviere
<b>Auto-Refresh (30 s)</b>.<br>
Die Tabelle zeigt für jedes Gerät: Online/Offline, letzten Score und letzten Alarm-Zeitstempel.</div>

<h2>Dashboard öffnen</h2>
<p>Klicke <b>Dashboard</b> in der Aktionen-Spalte um das Web-Dashboard des Geräts im Browser zu öffnen.</p>

<h2>Persistenz</h2>
<p>Die Gerätliste wird in QSettings gespeichert und beim nächsten Start automatisch geladen.</p>

<h2>monitor.py starten</h2>
<p>Auf jedem Edge-Gerät:</p>
<pre style="background:#0D1117;color:#F8C471;padding:8px;border-radius:4px;font-size:11px">
python monitor.py --model modell.onnx --api-port 8765 --api-key MEIN_SCHLÜSSEL
</pre>
<div class="tip"><b>Docker-Deployment</b><br>
Auf der Modelle-Seite → <b>Docker-Deployment generieren…</b> erzeugt ein fertiges
<code>Dockerfile</code> + <code>docker-compose.yml</code> für containerisierten Betrieb.</div>
"""),

# ── 19  Modelle Erweitert ──────────────────────────────────────────────────────
19: page("""
<h1>⚡ Modelle Erweitert</h1>
<p>Fortgeschrittene Werkzeuge auf der <b>Modellbibliothek-Seite</b> für Hyperparameter-Suche,
Kalibrierung und Edge-Deployment.</p>

<h2>Hyperparameter-Suche (Optuna)</h2>
<p>Auf der <b>Training-Seite</b> → Schaltfläche <b>⚙ Hyperparameter-Suche…</b></p>
<div class="step"><b>Wie es funktioniert</b><br>
Optuna testet automatisch verschiedene Kombinationen aus Lernrate, Batch-Größe,
Architektur und Optimizer. Jeder Versuch trainiert 5 Epochen. Am Ende werden die
besten Parameter direkt in die Trainings-Konfiguration übernommen.</div>
<div class="warn"><b>Benötigt: optuna</b><br><code>pip install optuna</code></div>

<h2>Modell-Vergleich (Dialog)</h2>
<p>Mehrere Modelle in der Tabelle auswählen (Strg+Klick) → <b>Ausgewählte vergleichen</b><br>
Zeigt eine sortierbare Tabelle mit Accuracy%, F1%, Architektur und Best-Markierung (★ Gold).</p>

<h2>Kalibrierung (Temperature Scaling)</h2>
<p>Modell auswählen → <b>Kalibrieren (Temperature Scaling)…</b><br>
Post-hoc Kalibrierung verbessert die Zuverlässigkeit von Konfidenzwerten.
Ein schlecht kalibriertes Modell sagt 95% vorher obwohl es nur 70% Trefferquote hat.</p>
<div class="warn"><b>Benötigt: scipy</b><br><code>pip install scipy</code></div>

<h2>ONNX INT8 Export</h2>
<p>Modell auswählen → <b>ONNX INT8 exportieren…</b><br>
Erstellt ein INT8-quantisiertes ONNX-Modell: typisch 2–4× kleiner und schneller als FP32,
ideal für CPU-Inferenz auf Edge-Geräten.</p>
<div class="warn"><b>Benötigt: onnxruntime</b><br><code>pip install onnxruntime</code></div>

<h2>CoreML Export (macOS)</h2>
<p>Modell auswählen → <b>CoreML exportieren…</b><br>
Erstellt ein <code>.mlpackage</code> für native Apple-Performance (Neural Engine auf M-Chips).</p>
<div class="warn"><b>Benötigt: coremltools (nur macOS)</b><br><code>pip install coremltools</code></div>

<h2>Docker-Deployment</h2>
<p>Modell auswählen → <b>Docker-Deployment generieren…</b><br>
Erstellt in einem gewählten Ordner:</p>
<ul>
  <li><code>Dockerfile</code> — Python 3.11-slim, EXPOSE Port, CMD monitor.py</li>
  <li><code>docker-compose.yml</code> — Ports, Volumes, restart: unless-stopped</li>
  <li><code>requirements_monitor.txt</code> — minimale Abhängigkeiten</li>
  <li><code>run_monitor.sh</code> — Startskript</li>
  <li><code>README_deploy.md</code> — Schritt-für-Schritt Anleitung</li>
</ul>
"""),

# ── 20  Kontakt & Repository ─────────────────────────────────────────────────
20: page("""
<h1>🔗 Kontakt &amp; Repository</h1>
<p>PictureStudio ist ein Open-Source-Projekt. Fragen, Fehlerberichte und
Pull-Requests sind herzlich willkommen.</p>
<hr>

<h2>GitHub-Repository</h2>
<p>
  <a href="https://github.com/oOMaikOo/PictureStudio">https://github.com/oOMaikOo/PictureStudio</a>
</p>
<ul>
  <li><b>Issues</b> — Fehler melden oder Feature-Wünsche eintragen</li>
  <li><b>Releases</b> — fertige Versionen als ZIP herunterladen</li>
  <li><b>Wiki</b> — weiterführende Dokumentation</li>
</ul>

<h2>Feedback &amp; Beiträge</h2>
<p>Pull Requests sind willkommen. Bitte einen Feature-Branch erstellen und
einen kurzen Issue anlegen, bevor größere Änderungen umgesetzt werden.</p>
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
        self._browser.setOpenExternalLinks(True)
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
