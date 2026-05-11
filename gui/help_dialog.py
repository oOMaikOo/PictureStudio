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
    ("📷", "Kamera"),
    ("⌨", "Tastenkürzel"),
    ("🔧", "Fehlerbehebung"),
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
(z. B. <code>Qualitätskontrolle_v1</code>) und wähle den Speicherort.
Das Projekt wird als <code>.json</code>-Datei gespeichert und enthält alle Labels, Bilder und Ergebnisse.</div>

<hr>
<h2>Phase 2 – Bilder laden</h2>
<div class="step"><b>Schritt 2 – Bildordner hinzufügen</b><br>
Gehe zur <b>Daten-Seite</b> → klicke <i>Bilder laden…</i> und wähle den Ordner mit deinen Bildern.
Alle <code>.jpg</code>, <code>.png</code>, <code>.bmp</code> und <code>.tiff</code> Dateien werden
automatisch hinzugefügt. <br><br>
<b>Tipp:</b> Strukturiere deine Bilder vorab in Unterordner nach Klasse – das erleichtert
die Annotation erheblich.</div>

<div class="step"><b>Schritt 3 – Datensatz analysieren</b><br>
Klicke <i>Dataset analysieren</i> um zu prüfen:<br>
• <b>Fehlende Dateien</b> – Bilder die nicht mehr auffindbar sind<br>
• <b>Duplikate</b> – identische Bilder (MD5-Hash) die das Training verzerren<br>
• <b>Klassenungleichgewicht</b> – z. B. 900× Klasse A, 50× Klasse B → schlechtes Training</div>

<hr>
<h2>Phase 3 – Labels definieren & Bilder annotieren</h2>
<div class="step"><b>Schritt 4 – Labels anlegen</b><br>
Gehe zur <b>Labeling-Seite</b> → <i>Projekt → Labels verwalten…</i> (<kbd>Strg+L</kbd>).<br>
Füge für jede Klasse ein Label hinzu (z. B. "gut", "defekt", "unklar").<br>
Wähle eindeutige Farben für jede Klasse.</div>

<div class="step"><b>Schritt 5 – Bilder labeln</b><br>
In der <b>Labeling-Seite</b>:<br>
1. Bild in der Thumbnail-Liste anklicken<br>
2. <kbd>1</kbd>–<kbd>9</kbd> drücken für schnelle Label-Zuweisung<br>
3. Mit <kbd>N</kbd> zum nächsten Bild, <kbd>P</kbd> zum vorherigen<br><br>
<b>Ziel:</b> Mindestens 50–100 Bilder pro Klasse für brauchbare Ergebnisse.
Für gute Modelle: 200+ Bilder pro Klasse.</div>

<div class="step"><b>Schritt 6 – ROIs zeichnen (optional)</b><br>
Falls du den interessanten Bereich eines Bildes markieren willst (z. B. nur den Defekt):<br>
• <kbd>R</kbd> = Rechteck zeichnen<br>
• <kbd>E</kbd> = Ellipse zeichnen<br>
• <kbd>G</kbd> = Polygon zeichnen (Doppelklick zum Abschließen)<br>
Dann ROI in der rechten Liste anwählen → Label zuweisen.</div>

<hr>
<h2>Phase 4 – Training konfigurieren & starten</h2>
<div class="step"><b>Schritt 7 – Architektur & Hyperparameter</b><br>
Gehe zur <b>Training-Seite</b>:<br>
• <b>Architektur:</b> Starte mit <code>ResNet-18</code> – schnell und gut als Baseline<br>
• <b>Epochen:</b> 20–30 für erste Tests, 50+ für finale Modelle<br>
• <b>Lernrate:</b> 0.001 (Standard); mit Scheduler 0.01<br>
• <b>Gerät:</b> <code>auto</code> nutzt automatisch GPU/MPS/CPU<br>
• <b>Early Stopping:</b> 5–7 Epochen verhindert Overfitting</div>

<div class="step"><b>Schritt 8 – Training starten</b><br>
Klicke <i>Training starten</i>. Du siehst in Echtzeit:<br>
• Train-Loss und Val-Loss (Val-Loss sollte sinken, sonst Overfitting)<br>
• Train-Accuracy und Val-Accuracy<br>
• Das beste Modell wird automatisch bei höchster Val-Accuracy gespeichert.</div>

<div class="tip"><b>Woran erkenne ich gutes Training?</b><br>
✓ Val-Loss sinkt parallel zu Train-Loss<br>
✓ Val-Accuracy steigt kontinuierlich<br>
✗ Val-Loss steigt während Train-Loss sinkt = Overfitting → Early Stopping oder mehr Daten</div>

<div class="step"><b>Schritt 9 – Ergebnisse auswerten</b><br>
Nach dem Training siehst du Accuracy und F1-Score.<br>
Erstelle mit <i>HTML-Bericht erstellen…</i> einen vollständigen Report mit Konfusionsmatrix.<br><br>
<b>Nicht zufrieden?</b><br>
• Mehr Daten beschaffen (wichtigste Maßnahme)<br>
• Andere Architektur probieren (EfficientNet-B0)<br>
• Lernrate anpassen (0.0001 oder 0.01)<br>
• Augmentierung ist bereits aktiv (Flip + Farbjitter)</div>

<hr>
<h2>Phase 5 – Modell einsetzen</h2>
<div class="step"><b>Schritt 10 – Neue Bilder klassifizieren</b><br>
Gehe zur <b>Klassifikations-Seite</b>:<br>
1. <i>Modell laden (.pth)</i> – oder von der Modelle-Seite direkt laden<br>
2. <i>Ordner…</i> – Ordner mit neuen Bildern wählen<br>
3. <i>Alle Bilder klassifizieren</i> – Ergebnis mit Top-3 Vorhersagen<br>
4. Niedrig-Konfidenz-Tab: Bilder unter dem Schwellwert manuell prüfen</div>

<div class="step"><b>Schritt 11 – Ergebnisse exportieren</b><br>
Gehe zur <b>Export-Seite</b>:<br>
1. <i>Ergebnisse aus letzter Inferenz laden</i><br>
2. Spalten konfigurieren (aktivieren, umbenennen)<br>
3. <i>Excel exportieren</i><br>
Rote Zeilen = Bilder unter Konfidenz-Schwellwert</div>

<div class="tip"><b>Fertig!</b> Du hast einen vollständigen ML-Pipeline-Durchlauf abgeschlossen.<br>
Für bessere Ergebnisse: mehr Daten hinzufügen → neu labeln → Training wiederholen.</div>
"""),

# ── 1  Übersicht ────────────────────────────────────────────────────────────
1: page("""
<h1>📋 Feature-Übersicht</h1>
<table>
  <tr><th>Bereich</th><th>Features</th></tr>
  <tr><td><b>Projektverwaltung</b></td><td>Versionierte JSON-Projekte, atomares Speichern, automatische Backups, Dashboard</td></tr>
  <tr><td><b>ROI-Editor</b></td><td>Rechteck, Ellipse, Polygon; Kopieren/Einfügen; Tastenkürzel; Label-Schnellzuweisung 1–9</td></tr>
  <tr><td><b>Labeling</b></td><td>Label-Hierarchien, Statistiken, Label-Filter, Review-Modus, Audit-Trail</td></tr>
  <tr><td><b>Datensatz-Analyse</b></td><td>MD5-Duplikaterkennung, Klassenungleichgewicht, COCO/YOLO/CSV-Export</td></tr>
  <tr><td><b>Training</b></td><td>ResNet18/50, MobileNetV2, EfficientNet-B0, SimpleCNN; Early Stopping; Mixed Precision; GPU/CPU/MPS</td></tr>
  <tr><td><b>SSH-Ferntraining</b></td><td>Verbindungsprofile, Live-Log-Streaming, conda/venv-Unterstützung</td></tr>
  <tr><td><b>Modellbibliothek</b></td><td>Versioniertes Registry, ONNX-Export, Accuracy/F1-Vergleich</td></tr>
  <tr><td><b>Inferenz</b></td><td>Batch-Inferenz, Top-3-Anzeige, Konfidenz-Farbkodierung, Niedrig-Konfidenz-Tab</td></tr>
  <tr><td><b>Excel-Export</b></td><td>Benutzerdefinierte Spalten, Anhängen/Überschreiben, rote Markierung</td></tr>
</table>

<h2>Unterstützte Architekturen</h2>
<table>
  <tr><th>ID</th><th>Modell</th><th>Empfehlung</th></tr>
  <tr><td><code>resnet18</code></td><td>ResNet-18</td><td>Schnell, guter Ausgangspunkt</td></tr>
  <tr><td><code>resnet50</code></td><td>ResNet-50</td><td>Höhere Kapazität, braucht mehr Daten</td></tr>
  <tr><td><code>mobilenet_v2</code></td><td>MobileNetV2</td><td>Effizient, gut für CPU</td></tr>
  <tr><td><code>efficientnet_b0</code></td><td>EfficientNet-B0</td><td>Bestes Genauigkeits-/Größe-Verhältnis</td></tr>
  <tr><td><code>simple_cnn</code></td><td>SimpleCNN</td><td>Kein Pretrained, schnell für erste Tests</td></tr>
</table>
<div class="tip">Alle Transfer-Learning-Modelle nutzen ImageNet Pretrained Weights.
Deaktiviere Pretrained nur bei sehr spezifischen Datensätzen die stark von Fotos abweichen.</div>
"""),

# ── 2  Dashboard ─────────────────────────────────────────────────────────────
2: page("""
<h1>🏠 Dashboard</h1>
<p>Der Startpunkt der Anwendung mit Projektübersicht.</p>

<h2>Was wird angezeigt?</h2>
<ul>
  <li>Gesamtanzahl Bilder und Anteil gelabelter Bilder</li>
  <li>Definierte Klassen mit Farbkodierung</li>
  <li>Anzahl ROIs im Projekt</li>
  <li>Metriken des letzten Trainings (Accuracy, F1)</li>
</ul>

<h2>Schritt-für-Schritt</h2>
<div class="step"><b>1 – Neues Projekt</b><br>
<i>Datei → Neues Projekt</i> oder <kbd>Strg+N</kbd><br>
Name vergeben → Speicherort wählen → Projekt wird als <code>.json</code> angelegt.</div>

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
Wähle einen Ordner – alle <code>.jpg .png .bmp .tiff</code> Dateien werden hinzugefügt.<br>
Bilder werden nicht kopiert, der Pfad wird gespeichert.</div>

<h2>Dataset analysieren</h2>
<div class="step"><b>Dataset analysieren</b><br>
Prüft deinen Datensatz auf:<br>
• <b>Fehlende Dateien</b> – Bilder die nicht mehr vorhanden sind<br>
• <b>MD5-Duplikate</b> – identische Bilder die Training verzerren<br>
• <b>Klassenungleichgewicht</b> – ungleiche Bildanzahl pro Klasse → schlechtes Modell<br>
• <b>Bildstatistiken</b> – Formate, Größen, Farbmodi</div>

<div class="step"><b>Fehlende Dateien prüfen / Pfade korrigieren</b><br>
Nach dem Verschieben von Bildern → <i>Bildpfade korrigieren…</i> aktualisiert Pfade.</div>

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
<h1>🏷 Labeling & ROIs</h1>
<p>Bilder annotieren, Labels zuweisen, Regionen einzeichnen.</p>

<h2>Labels verwalten</h2>
<div class="step"><b>Labels hinzufügen</b><br>
<i>Projekt → Labels verwalten…</i> (<kbd>Strg+L</kbd>)<br>
Name + Farbe definieren. Labels können jederzeit umbenannt oder neu eingefärbt werden.</div>

<h2>Bilder labeln</h2>
<div class="step"><b>Schnelles Labeln</b><br>
1. Bild in der Liste anklicken<br>
2. <kbd>1</kbd>–<kbd>9</kbd> drücken = Label 1–9 zuweisen<br>
3. <kbd>N</kbd> = nächstes Bild, <kbd>P</kbd> = vorheriges Bild<br>
Label-Filter: Nur Bilder einer bestimmten Klasse anzeigen</div>

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
nutze 'ROIs dieses Bildes → alle Bilder'.</div>
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
  <tr><td>Batch-Größe</td><td>32 (GPU) / 8 (CPU)</td><td>Bilder pro Trainingsschritt</td></tr>
  <tr><td>Gerät</td><td>auto</td><td>Automatisch: GPU > MPS > CPU</td></tr>
  <tr><td>Early Stopping</td><td>5–7</td><td>Stopp nach N Epochen ohne Verbesserung</td></tr>
  <tr><td>LR-Scheduler</td><td>cosine</td><td>Lernrate automatisch anpassen</td></tr>
</table>

<h2>Training starten & überwachen</h2>
<div class="step"><b>Training starten</b><br>
Klicke <i>Training starten</i>. Live-Anzeige:<br>
• <b>Train-Loss</b> und <b>Val-Loss</b> – sinken idealerweise gemeinsam<br>
• <b>Train-Acc</b> und <b>Val-Acc</b> – steigen idealerweise gemeinsam<br>
Das beste Modell (höchste Val-Acc) wird automatisch gespeichert.</div>

<div class="warn"><b>Overfitting erkennen:</b>
Train-Loss sinkt, Val-Loss steigt → Modell lernt die Trainingsdaten auswendig.
Lösung: mehr Daten, Early Stopping aktivieren, Dropout erhöhen.</div>

<h2>SSH-Ferntraining</h2>
<div class="step"><b>Einrichten</b><br>
1. <i>Einstellungen → SSH-Profile</i> – Profil anlegen (Host, User, Key)<br>
2. Training-Seite: SSH-Checkbox aktivieren → Profil wählen → Verbindung testen<br>
3. Training starten → Anwendung zippt Daten, lädt sie hoch, streamt Logs,
   lädt das beste Modell automatisch herunter</div>

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

<h2>ONNX exportieren</h2>
<div class="step"><b>Als ONNX exportieren</b><br>
ONNX ermöglicht den Einsatz ohne PyTorch:<br>
• TensorRT (NVIDIA GPU Inference)<br>
• OpenCV DNN<br>
• ONNX Runtime (cross-platform)<br>
• Edge-Deployment (Raspberry Pi, Jetson)</div>

<h2>Modelle vergleichen</h2>
<div class="step"><b>Ausgewählte vergleichen</b><br>
Mehrere Modelle mit <kbd>Strg+Klick</kbd> auswählen → <i>Ausgewählte vergleichen</i><br>
Zeigt alle Metriken nebeneinander.</div>
"""),

# ── 7  Klassifikation ─────────────────────────────────────────────────────────
7: page("""
<h1>🔍 Klassifikation</h1>
<p>Neue Bilder mit dem trainierten Modell klassifizieren.</p>

<h2>Schritt-für-Schritt</h2>
<div class="step"><b>1 – Modell laden</b><br>
<i>Modell laden (.pth)</i> → Modelldatei wählen.<br>
Schneller: Modelle-Seite → <i>In Inferenz laden</i></div>

<div class="step"><b>2 – Bildordner wählen</b><br>
<i>Ordner…</i> → Ordner mit neuen Bildern wählen.<br>
Einzelbild: <i>Einzelbild klassifizieren</i></div>

<div class="step"><b>3 – Klassifizieren</b><br>
<i>Alle Bilder klassifizieren</i> → Ergebnis mit Top-3 Vorhersagen.<br>
Farbkodierung: Grün &gt;90% | Gelb 70–90% | Rot &lt;70%</div>

<div class="step"><b>4 – Unsichere prüfen</b><br>
<b>Niedrig-Konfidenz-Tab</b>: alle Bilder unter dem Schwellwert (Standard: 70%).<br>
Diese manuell überprüfen und ggf. ins Training aufnehmen.</div>

<h2>Konfidenz-Schwellwert anpassen</h2>
<div class="tip">Einstellungen → Niedrig-Konfidenz-Schwellwert (Standard: 0.70).<br>
Niedrigerer Wert = mehr Bilder werden als sicher eingestuft.<br>
Höherer Wert = strengere Qualitätskontrolle.</div>
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

<div class="step"><b>4 – Modus & Exportieren</b><br>
<b>Überschreiben:</b> Neue Datei / bestehende Datei komplett ersetzen<br>
<b>Anhängen:</b> Zeilen an bestehende Datei hinzufügen (gleiche Spalten!)<br>
Dann <i>Excel exportieren</i> klicken.</div>

<div class="warn"><b>Anforderung:</b> <code>pip install openpyxl</code> muss installiert sein.</div>

<h2>Besonderheiten</h2>
<ul>
  <li>Zeilen unter Konfidenz-Schwellwert werden rot markiert</li>
  <li>Kopfzeilen werden fett und farbig formatiert</li>
  <li>Top-2 und Top-3 Vorhersagen können optional hinzugefügt werden</li>
</ul>
"""),

# ── 9  Einstellungen ──────────────────────────────────────────────────────────
# (shifted to index 11 due to camera section insertion)

# ── 10  Kamera ────────────────────────────────────────────────────────────────
10: page("""
<h1>📷 Kamera – Live-Aufnahme</h1>
<p>Bilder direkt aus USB-Kameras oder IP-Kamerastreams aufnehmen und ins Projekt übernehmen.</p>

<h2>Kamera öffnen</h2>
<div class="step"><b>Menü Datei → Kamera aufnehmen… <kbd>Strg+K</kbd></b><br>
Öffnet den Kamera-Dialog. Die Anwendung erkennt automatisch verfügbare USB-Kameras.</div>

<h2>USB-Kamera</h2>
<div class="step"><b>USB Kamera-Tab</b><br>
1. Kamera aus dem Dropdown wählen (werden automatisch erkannt)<br>
2. <i>Verbinden</i> klicken → Live-Vorschau erscheint<br>
3. Kamera nicht gefunden? → <i>Kameras neu suchen</i></div>

<h2>IP-Kamera / Netzwerkkamera</h2>
<div class="step"><b>IP Kamera-Tab</b><br>
Kamera-URL eingeben und <i>Verbinden</i> klicken.<br>
Unterstützte Protokolle:
<ul>
  <li><code>rtsp://user:pass@192.168.1.100:554/stream</code> – RTSP-Streams (IP-Kameras, NVR)</li>
  <li><code>http://192.168.1.100:8080/video</code> – HTTP-MJPEG-Streams</li>
  <li><code>http://192.168.1.100/cgi-bin/mjpeg</code> – Herstellerspezifische URLs</li>
</ul>
</div>

<h2>Aufnahme</h2>
<div class="step"><b>Einzelbild aufnehmen</b><br>
Klicke <i>Bild aufnehmen</i> oder drücke <kbd>Leertaste</kbd>.<br>
Bilder werden als PNG im gewählten Speicherordner gesichert.</div>

<div class="step"><b>Burst-Aufnahme</b><br>
Anzahl Bilder und Intervall einstellen → <i>Burst starten</i>.<br>
Ideal für schnelle Prozesse: z. B. 20 Bilder im 0,5-Sekunden-Abstand.</div>

<h2>Zeitstempel einblenden</h2>
<div class="step"><b>Zeitstempel-Optionen</b><br>
• <b>Im Vorschaubild anzeigen</b> – blendet Systemdatum und -uhrzeit live im Vorschaufenster ein.<br>
  Kann jederzeit ein- und ausgeschaltet werden ohne die gespeicherten Bilder zu beeinflussen.<br>
• <b>In gespeichertes Bild einbrennen</b> – der Zeitstempel wird dauerhaft in die PNG-Datei gerendert.<br>
  Format: <code>YYYY-MM-DD  HH:MM:SS</code>, weiße Schrift mit schwarzem Schatten unten links.</div>

<div class="tip"><b>Tipp für Qualitätskontrolle:</b>
Den Zeitstempel einzubrennen eignet sich für Dokumentationszwecke, wenn Aufnahmezeitpunkt
und Bildinhalt lückenlos nachvollziehbar sein müssen.</div>

<h2>Bilder ins Projekt übernehmen</h2>
<div class="step"><b>In Projekt übernehmen</b><br>
Aufgenommene Bilder erscheinen in der Liste unten links.<br>
Klicke <i>In Projekt übernehmen (N)</i> – die Bilder werden dem Projekt hinzugefügt
und sind sofort in der Labeling-Seite verfügbar.</div>

<div class="warn"><b>Anforderung:</b> <code>pip install opencv-python</code> muss installiert sein.<br>
Für RTSP-Streams ggf. <code>opencv-python-headless</code> oder eine vollständige OpenCV-Version.</div>

<h2>Anomalie-Erkennung (Autoencoder)</h2>
<p>Erkennt automatisch Ereignisse im Video-Stream, die vom Normalablauf abweichen –
ohne dass Anomalie-Beispiele bekannt sein müssen.</p>

<h3>Funktionsprinzip</h3>
<div class="tip">
Ein neuronales Netz (Conv-Autoencoder) lernt ausschließlich auf <b>normalen Prozessframes</b>
wie ein normaler Ablauf aussieht. Kommt ein unbekannter Frame – z. B. ein Fehler,
eine Abweichung vom Regelablauf – kann das Netz ihn nicht gut rekonstruieren.
Der <b>Rekonstruktionsfehler</b> (MSE) überschreitet den Schwellwert → Alarm.
</div>

<h3>Schritt-für-Schritt</h3>
<div class="step"><b>Schritt 1 – Normalframes aufnehmen</b><br>
Kamera verbinden, Prozess normal ablaufen lassen.<br>
Klicke <i>Aufnehmen starten</i> – die App sammelt automatisch N Frames.<br>
Empfehlung: <b>100–300 Frames</b>, die den typischen Ablauf abdecken.</div>

<div class="step"><b>Schritt 2 – Autoencoder trainieren</b><br>
Epochen einstellen (Standard: 20, mehr = besser aber langsamer).<br>
Klicke <i>Training starten</i>. Loss und Fortschritt werden live angezeigt.<br>
Nach dem Training wird der Schwellwert automatisch berechnet:
<code>Mittelwert + 2,5 × Standardabweichung</code> der Trainingsfehler.</div>

<div class="step"><b>Schritt 3 – Live-Erkennung aktivieren</b><br>
Checkbox <i>Aktiv</i> aktivieren. Jeder 3. Frame wird bewertet (CPU-schonend).<br>
<b>Score-Anzeige:</b> Grün = normaler Frame | Rot = Anomalie<br>
<b>Roter Banner</b> erscheint oben im Vorschaubild bei Alarm.<br>
<b>Schwellwert</b> kann manuell angepasst werden:
Höher = weniger Fehlalarme | Niedriger = sensitiver.</div>

<div class="step"><b>Anomalie-Frames automatisch speichern</b><br>
Checkbox aktivieren → alle Frames bei denen ein Alarm ausgelöst wird,
werden automatisch gespeichert und ins Projekt übernommen.<br>
Ideal für Dokumentation oder um Anomalie-Daten für späteres Training zu sammeln.</div>

<div class="step"><b>Modell speichern und laden</b><br>
<i>Speichern…</i> sichert den trainierten Autoencoder als <code>.pth</code>-Datei.<br>
<i>Laden…</i> lädt ein gespeichertes Modell – kein erneutes Training notwendig.<br>
Das Modell ist auf Prozess und Kamera abgestimmt und nicht portierbar.</div>

<h3>Tipps für gute Erkennung</h3>
<ul>
  <li>Gleichmäßige Beleuchtung beim Aufnehmen der Normalframes verwenden</li>
  <li>Alle Varianten des normalen Ablaufs aufnehmen (z. B. verschiedene Werkstücke)</li>
  <li>Kamera-Position nach dem Training nicht mehr verändern</li>
  <li>Schwellwert-Feintuning: im Normalbetrieb Score beobachten, Alarm-Grenze anpassen</li>
  <li>Bei vielen Fehlalarmen: mehr Normalframes sammeln und neu trainieren</li>
</ul>
"""),

9: page("""
<h1>⚙ Einstellungen</h1>
<p>Alle Einstellungen werden automatisch gespeichert und beim nächsten Start wiederhergestellt.</p>

<h2>Erscheinungsbild</h2>
<table>
  <tr><th>Einstellung</th><th>Standard</th><th>Beschreibung</th></tr>
  <tr><td>Theme</td><td>Dunkel</td><td>Dunkel oder Hell – sofort wirksam</td></tr>
  <tr><td>Schriftgröße</td><td>9 pt</td><td>7–16 pt</td></tr>
  <tr><td>Thumbnail-Größe</td><td>100 px</td><td>60–240 px – beeinflusst Ladezeit</td></tr>
</table>

<h2>Autosave</h2>
<table>
  <tr><th>Einstellung</th><th>Standard</th><th>Beschreibung</th></tr>
  <tr><td>Autosave-Intervall</td><td>300 s</td><td>30–3600 Sekunden</td></tr>
  <tr><td>Backup vor Speichern</td><td>Ein</td><td>Erstellt timestamped .bak Datei</td></tr>
</table>

<h2>Klassifikation</h2>
<table>
  <tr><th>Einstellung</th><th>Standard</th><th>Beschreibung</th></tr>
  <tr><td>Konfidenz-Schwellwert</td><td>0.70</td><td>Unterhalb = Niedrig-Konfidenz-Tab</td></tr>
  <tr><td>Top-K-Anzeige</td><td>3</td><td>Anzahl angezeigte Top-Vorhersagen (1–5)</td></tr>
</table>

<h2>SSH-Profile</h2>
<div class="step"><b>Profil hinzufügen</b><br>
• <b>Name:</b> Bezeichnung (z. B. "GPU-Server")<br>
• <b>Host:</b> Hostname oder IP-Adresse<br>
• <b>Benutzername:</b> SSH-Login<br>
• <b>Key-Pfad:</b> Pfad zum privaten SSH-Key (empfohlen statt Passwort)<br>
• <b>Port:</b> Standard 22</div>

<div class="tip"><b>Speicherort:</b><br>
macOS: <code>~/Library/Preferences</code><br>
Windows: Registry <code>HKCU\\Software</code></div>
"""),

# ── 10  Tastenkürzel ──────────────────────────────────────────────────────────
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
"""),

# ── 12  Fehlerbehebung ────────────────────────────────────────────────────────
12: page("""
<h1>🔧 Fehlerbehebung</h1>

<h2>Anwendung startet nicht</h2>
<div class="step">PySide6 installieren: <code>pip install PySide6</code><br>
Linux: Qt-Plugins: <code>apt install libxcb-cursor0</code></div>

<h2>Training sehr langsam</h2>
<div class="step">Gerät auf <code>cuda</code> oder <code>mps</code> stellen.<br>
CPU-Test: Bildgröße 128 px, Batch-Größe 8, SimpleCNN-Architektur.</div>

<h2>ImportError: openpyxl</h2>
<div class="step"><code>pip install openpyxl</code> – für Excel-Export erforderlich.</div>

<h2>ImportError: paramiko</h2>
<div class="step"><code>pip install paramiko</code> – nur für SSH-Ferntraining.</div>

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
• Manuell testen: <code>ssh user@host</code> im Terminal</div>

<h2>Kamera wird nicht gefunden</h2>
<div class="step"><code>pip install opencv-python</code> installieren.<br>
USB-Kamera: <i>Kameras neu suchen</i> klicken.<br>
Andere Anwendungen schließen, die die Kamera blockieren könnten.<br>
IP-Kamera: URL im Browser prüfen, ob der Stream erreichbar ist.</div>

<h2>Zeitstempel erscheint nicht im gespeicherten Bild</h2>
<div class="step">Checkbox <i>In gespeichertes Bild einbrennen</i> aktivieren.<br>
Die Vorschau-Checkbox allein beeinflusst nur das Live-Bild, nicht die Datei.</div>
"""),

}


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class HelpDialog(QDialog):
    """Help dialog with sidebar navigation and content browser."""

    def __init__(self, page_index: int = 0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hilfe – Image Labeling Studio")
        self.setMinimumSize(880, 600)
        self.resize(1020, 680)

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
