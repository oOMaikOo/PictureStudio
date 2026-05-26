"""
Interactive guided tour – floating panel that walks through each page step by step,
highlights the relevant button/widget with an overlay frame.
"""
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QAbstractButton, QWidget, QApplication,
)
from PySide6.QtCore import Qt, QTimer, QPoint, QRect
from PySide6.QtGui import QFont, QColor

# ---------------------------------------------------------------------------
# Tour steps per page index (0–9)
# 0=Dashboard, 1=Daten, 2=Labeling, 3=Training, 4=Modelle,
# 5=Klassifikation, 6=Export, 7=Einstellungen, 8=Kamera, 9=Batch,
# 10=Multi-Kamera, 11=Clustering, 12=Datensatz, 13=VideoAnnotation,
# 14=Fleet, 15=Objekterkennung, 16=DataDrift, 17=AnomalieTraining
# Each step: (title, description, button_text_to_highlight | None)
# ---------------------------------------------------------------------------
TOUR_STEPS = {
    0: [  # Dashboard
        ("Willkommen bei Picture Studio",
         "Das Dashboard zeigt den Projektstand auf einen Blick:\n"
         "Bilder, Labels, Klassen und letzte Trainingsmetriken.\n\n"
         "Die Navigationsleiste links ist gesperrt — erst ein\n"
         "Projekt öffnen oder neu anlegen um sie freizuschalten.",
         None),
        ("Neues Projekt anlegen",
         "Klicke '+ Neues Projekt' oder Strg+N.\n\n"
         "Im Dialog wählst du:\n"
         "• Projektname und Beschreibung\n"
         "• Projekttyp: 📸 Bildklassifikation\n"
         "  oder 🎬 Videoanalyse & Anomalie\n\n"
         "Der Typ bestimmt welche Seiten in der\n"
         "Sidebar angezeigt werden.",
         "Neues Projekt"),
        ("Vorhandenes Projekt öffnen",
         "Klicke 'Projekt öffnen…' oder Strg+O um eine\n"
         "bestehende Projektdatei (.json) zu laden.\n\n"
         "Zuletzt geöffnete Projekte:\n"
         "Menü → Datei → Zuletzt geöffnet\n\n"
         "Nach dem Laden ist die Sidebar freigeschaltet\n"
         "und zeigt die passenden Seiten für den Projekttyp.",
         "Projekt öffnen"),
        ("Dashboard-Statistiken",
         "Die Karten zeigen auf einen Blick:\n"
         "• Bilder gesamt / gelabelt / ungelabelt\n"
         "• ROIs und Klassen\n"
         "• Anzahl Trainingsläufe\n\n"
         "Darunter: Klassenverteilung mit Balkendiagramm\n"
         "und Warnungen bei Ungleichgewicht oder\n"
         "zu wenig Bildern pro Klasse.",
         None),
    ],
    1: [  # Daten
        ("Daten-Seite",
         "Hier importierst du Bilder oder Videos,\n"
         "analysierst den Datensatz und exportierst\n"
         "Annotationen in verschiedene Formate.",
         None),
        ("Bilder laden",
         "Klicke 'Bilder laden…' → Ordner wählen.\n"
         "Alle .jpg, .png, .bmp, .tiff werden\n"
         "rekursiv aus allen Unterordnern geladen.\n\n"
         "Alternativ: Ordner oder Bilder ins Fenster\n"
         "ziehen (Drag & Drop – Unterordner ebenfalls\n"
         "rekursiv eingeschlossen).\n\n"
         "Im Labeling-Reiter erscheint der Dateiname\n"
         "als 'Unterordner/Dateiname'.",
         "Bilder laden"),
        ("Video importieren",
         "Klicke 'Video importieren…' um ein Video\n"
         "(MP4, AVI, MOV, MKV, WebM) zu importieren.\n\n"
         "Frame-Intervall einstellen:\n"
         "• Alle 1 Frame = volle Framerate\n"
         "• Alle 5 Frames = ca. 6 Bilder/s bei 30 fps\n\n"
         "Die extrahierten Frames werden als PNG\n"
         "ins Projektverzeichnis gespeichert.",
         "Video importieren"),
        ("Datensatz analysieren",
         "Klicke 'Dataset analysieren' um zu prüfen:\n"
         "• Fehlende Dateien\n"
         "• MD5-Duplikate (identische Bilder)\n"
         "• Klassenungleichgewicht\n"
         "• Bildformat- und Größenstatistiken\n\n"
         "Bei starkem Ungleichgewicht: auf der\n"
         "Trainingsseite Klassenausgleich aktivieren.",
         "Dataset analysieren"),
        ("Annotationen exportieren",
         "Exportiere Annotationen (ROIs + Labels) als:\n"
         "• COCO JSON → Object-Detection-Frameworks\n"
         "• YOLO TXT → Ultralytics / Darknet\n"
         "• CSV → eigene Tools / Tabellenkalkulation",
         "COCO"),
    ],
    2: [  # Labeling (nur Bildprojekte)
        ("Labeling-Seite",
         "Weise Bildern Klassen zu und zeichne ROIs\n"
         "(Regions of Interest) für die Klassifikation.\n\n"
         "Links: Thumbnail-Liste  |  Mitte: Editor\n"
         "Rechts: ROI-Details und Label-Zuweisung",
         None),
        ("Bilder laden & Label definieren",
         "Bilder zuerst auf der Daten-Seite laden\n"
         "oder hier 'Ordner laden…' klicken.\n\n"
         "Labels definieren:\n"
         "Menü → Projekt → Labels verwalten… (Strg+L)\n"
         "• Namen und Farbe pro Klasse festlegen\n"
         "• Mindestens 2 Klassen für Training nötig",
         "Ordner laden"),
        ("Bild auswählen & schnell labeln",
         "Bild in der Thumbnail-Liste anklicken.\n\n"
         "Schnellzuweisung: Taste 1–9 drücken\n"
         "(Reihenfolge entspricht der Label-Liste)\n\n"
         "Alternativ: Label-Dropdown oben rechts\n\n"
         "Navigation:\n"
         "N = nächstes Bild  |  P = vorheriges Bild",
         None),
        ("ROI zeichnen",
         "Werkzeug in der Toolbar wählen:\n"
         "• R = Rechteck (häufigste Wahl)\n"
         "• E = Ellipse\n"
         "• G = Polygon (für unregelmäßige Formen)\n\n"
         "Im Bild ziehen um ROI zu zeichnen.\n"
         "Esc = Abbrechen  |  Entf = Löschen\n"
         "Strg+C / Strg+V = Kopieren / Einfügen",
         "Rechteck"),
        ("ROI-Label zuweisen",
         "ROI in der rechten Liste auswählen.\n"
         "Label im Dropdown darunter wählen.\n"
         "'ROI-Label zuweisen' klicken.\n\n"
         "Schneller: ROI auswählen + Taste 1–9\n\n"
         "'ROIs dieses Bildes → alle Bilder' überträgt\n"
         "die gleichen ROI-Positionen auf alle Bilder.",
         "ROI-Label zuweisen"),
        ("Segmentierungsmaske malen",
         "Für pixelgenaue Annotation:\n"
         "Tab '🎨 Segmentierungsmaske' im mittleren\n"
         "Bereich wählen.\n\n"
         "Linksklick = malen  |  Rechtsklick = löschen\n"
         "Scroll = Zoom  |  Klasse und Pinselgröße\n"
         "über die obere Toolbar wählen.\n\n"
         "'Maske speichern' speichert als PNG\n"
         "neben der Bilddatei.",
         "Maske speichern"),
        ("Pre-Labeling – Vorschläge vom Modell",
         "Hast du bereits ein trainiertes Modell?\n"
         "Lass es Label-Vorschläge machen:\n\n"
         "1. 📂 (oben rechts im Pre-Labeling-Panel)\n"
         "   → .pth-Modell auswählen\n"
         "2. Konfidenz-Schwellwert einstellen\n"
         "   (Standard: 75%)\n"
         "3. '▶ Vorschläge generieren'\n"
         "4. '✅ Vorschläge übernehmen'\n\n"
         "Nur Bilder über dem Schwellwert werden\n"
         "gelabelt — Undo jederzeit möglich.",
         "Vorschläge generieren"),
    ],
    3: [  # Training (nur Bildprojekte)
        ("Training-Seite",
         "Trainiere ein CNN-Modell auf deinen\n"
         "annotierten Bildern.\n\n"
         "Links: Konfiguration\n"
         "Rechts: Live-Kurven & Metriken\n\n"
         "Voraussetzung: Bilder müssen gelabelt sein.",
         None),
        ("Architektur wählen",
         "Im Architektur-Dropdown:\n"
         "• ResNet-18 — schnell, guter Startpunkt\n"
         "• EfficientNet-B3 ★ — beste Genauigkeit\n"
         "• ConvNeXt-Tiny ★ — stark bei wenig Daten\n"
         "• MobileNetV2 — effizient, gut für CPU\n"
         "• DINOv2 ViT-S/14 ★★ — Foundation Model\n"
         "  (Backbone eingefroren, nur Head trainiert;\n"
         "   ideal mit < 100 Bildern pro Klasse;\n"
         "   erfordert Internet beim ersten Laden)\n"
         "• SimpleCNN — kein GPU nötig, für Tests\n\n"
         "Alle außer SimpleCNN/DINOv2 nutzen\n"
         "ImageNet-Pretrained (Transfer Learning).",
         None),
        ("Hyperparameter einstellen",
         "Empfohlene Startwerte:\n"
         "• Epochen: 20–50\n"
         "• Lernrate: 0.001\n"
         "• Batch-Größe: 32 (GPU) / 8–16 (CPU)\n"
         "• Gerät: 'auto' wählt GPU > MPS > CPU\n"
         "• Early Stopping: 5 Epochen\n"
         "• Klassenausgleich: bei ungleichen Klassen\n"
         "  (WeightedRandomSampler) aktivieren",
         None),
        ("Hyperparameter-Suche & Training starten",
         "Button-Reihenfolge:\n"
         "① Hyperparameter-Suche → ② Training starten → ③ Stoppen\n\n"
         "⚙ Hyperparameter-Suche (optional):\n"
         "Optuna testet Lernrate, Batch-Größe,\n"
         "Architektur und Optimizer automatisch.\n"
         "Ein Live-Log zeigt jeden Versuch mit\n"
         "Parametern und Ergebnis. ★ = neuer Bestwert.\n\n"
         "Training starten → Kurven aktualisieren live.\n"
         "Bestes Checkpoint wird automatisch gespeichert.\n\n"
         "Nach dem Training:\n"
         "• HTML- oder Excel-Bericht erstellen\n"
         "• Modell auf der Modelle-Seite verwalten",
         "Training starten"),
        ("SSH-Ferntraining auf GPU-Server",
         "Training auf einem externen Server:\n"
         "1. Einstellungen → SSH-Profil anlegen\n"
         "   (Host, Benutzer, SSH-Key-Pfad)\n"
         "2. SSH-Ferntraining-Checkbox aktivieren\n"
         "3. Profil wählen → 'Verbindung testen'\n"
         "4. Grünes Signal → Training starten\n\n"
         "Der Log streamt live. Das Checkpoint\n"
         "wird automatisch heruntergeladen.",
         "Verbindung testen"),
        ("Active Learning nach dem Training",
         "Tab '🔄 Active Learning' erscheint nach\n"
         "dem Training automatisch mit dem Modell.\n\n"
         "Ablauf:\n"
         "① Schwellwert einstellen (Standard 0.70)\n"
         "② Max. Kandidaten festlegen (Standard 50)\n"
         "③ '🔍 AL-Scan starten'\n\n"
         "Das Modell klassifiziert alle ungelabelten\n"
         "Bilder und trägt die unsichersten in die\n"
         "AL-Queue (Labeling-Seite) ein.\n\n"
         "Dann: Labeling-Seite öffnen und\n"
         "Queue-Einträge reviewen → neu trainieren.",
         "🔍 AL-Scan starten"),
    ],
    4: [  # Modelle
        ("Modellbibliothek",
         "Alle trainierten Modelle des Projekts\n"
         "auf einen Blick.\n\n"
         "Tab 📦 Modellbibliothek — aktuelle Modelle\n"
         "Tab 📊 Run-History — alle Trainingsläufe\n"
         "im zeitlichen Verlauf vergleichen",
         None),
        ("Modell laden & als Best markieren",
         "Modell in der Tabelle auswählen.\n\n"
         "'In Inferenz laden' → Modell auf der\n"
         "Klassifikations-Seite verwenden.\n\n"
         "'Als Best markieren' → Modell als\n"
         "Standard für das Projekt setzen.\n\n"
         "Tipp: F1-Score bei ungleichen Klassen\n"
         "aussagekräftiger als Accuracy.",
         "In Inferenz laden"),
        ("ONNX & TorchScript exportieren",
         "Modell für andere Systeme exportieren:\n\n"
         "'Als ONNX exportieren' → .onnx (Opset 17)\n"
         "Einsatz in: ONNX Runtime, OpenCV DNN,\n"
         "TensorRT, Python, C++, C#\n\n"
         "'Als TorchScript exportieren' → .pt\n"
         "Einsatz in: PyTorch C++ API, mobile Apps",
         "Als ONNX exportieren"),
        ("Modelle vergleichen",
         "Mehrere Modelle auswählen (Strg+Klick)\n"
         "→ 'Ausgewählte vergleichen'\n\n"
         "Zeigt Accuracy, F1, Architektur\n"
         "und Best-Markierung im Überblick.\n\n"
         "Run-History-Tab: alle Läufe nach Datum\n"
         "sortiert mit Gerät, Epochen, Train-Acc.",
         "Ausgewählte vergleichen"),
    ],
    5: [  # Klassifikation (nur Bildprojekte)
        ("Klassifikations-Seite",
         "Neue (unbekannte) Bilder mit einem\n"
         "trainierten Modell bewerten.\n\n"
         "Ergebnis: Top-K Vorhersagen mit\n"
         "Konfidenz-Farbkodierung.\n\n"
         "Modell zuerst laden.",
         None),
        ("Modell laden",
         "'Modell laden (.pth)' → Datei wählen\n\n"
         "Oder direkt von der Modelle-Seite:\n"
         "Modell auswählen → 'In Inferenz laden'\n\n"
         "Ensemble — mehrere Modelle kombinieren:\n"
         "'+ Modell hinzufügen' → alle geladenen\n"
         "Modelle werden gemittelt (stabilere\n"
         "Vorhersagen bei schwierigen Bildern).",
         "Modell laden"),
        ("Ordner klassifizieren",
         "1. 'Ordner…' → Ordner mit neuen Bildern\n"
         "   ☑ Unterordner einschließen → scannt\n"
         "   alle Unterordner rekursiv. Dateiname\n"
         "   zeigt dann 'Unterordner/Dateiname'.\n"
         "2. TTA (Test-Time Augmentation):\n"
         "   Spinner auf 3–5 → mehrere augmentierte\n"
         "   Versionen je Bild → genauere Ergebnisse\n"
         "3. 'Alle Bilder klassifizieren'\n\n"
         "Farben: Grün >90% | Gelb 70–90% | Rot <70%",
         "Alle Bilder klassifizieren"),
        ("Unsichere Vorhersagen prüfen",
         "Tab 'Niedrige Konfidenz' zeigt alle Bilder\n"
         "unter dem eingestellten Schwellwert.\n\n"
         "Diese Bilder eignen sich für:\n"
         "• Manuelle Überprüfung\n"
         "• Nachtrainieren (Active Learning)\n"
         "• Hinzufügen zur Trainingsdatenbank",
         None),
        ("Automatisch labeln",
         "Hochkonfidente Ergebnisse direkt als\n"
         "Projekt-Labels übernehmen:\n\n"
         "1. Mindest-Konfidenz einstellen (z. B. 0.90)\n"
         "2. 'Auf Projekt anwenden' klicken\n"
         "3. Bilder mit Konfidenz ≥ Schwellwert\n"
         "   bekommen automatisch ein Label\n\n"
         "Danach: Labeling-Seite zur Kontrolle öffnen.",
         "Auf Projekt anwenden"),
    ],
    6: [  # Export
        ("Excel-Export",
         "Klassifikationsergebnisse in eine\n"
         "formatierte Excel-Datei exportieren.\n"
         "Spalten sind frei konfigurierbar.",
         None),
        ("Ergebnisse laden & Datei wählen",
         "'Ergebnisse aus letzter Inferenz laden'\n"
         "übernimmt die aktuellen Ergebnisse.\n\n"
         "Dann Zieldatei wählen:\n"
         "'Datei wählen…' → vorhandene Excel-Datei\n"
         "'Neue Datei erstellen' → neue Excel-Datei\n\n"
         "Modus:\n"
         "• Anhängen: fügt Zeilen unten an\n"
         "• Überschreiben: erstellt neue Datei",
         "Ergebnisse aus letzter"),
        ("Spalten konfigurieren & exportieren",
         "In der Spalten-Tabelle:\n"
         "• Checkbox: Spalte ein-/ausschalten\n"
         "• Doppelklick auf Name: umbenennen\n\n"
         "Verfügbare Spalten z. B.:\n"
         "Dateiname, Vorhersage, Konfidenz,\n"
         "Top-2/3, Zeitstempel, Modellpfad\n\n"
         "→ 'Excel exportieren' klicken.",
         "Excel exportieren"),
    ],
    7: [  # Einstellungen
        ("Einstellungen – Überblick",
         "Alle Einstellungen werden automatisch\n"
         "gespeichert (QSettings) und beim\n"
         "nächsten Start wiederhergestellt.\n\n"
         "Nach Änderungen immer\n"
         "'Einstellungen speichern' klicken.",
         None),
        ("Theme & Darstellung",
         "Design: 'dark' (Empfehlung) oder 'light'\n"
         "→ sofort wirksam.\n\n"
         "Schriftgröße: 7–16 pt\n\n"
         "Thumbnail-Größe: 60–240 px\n"
         "Kleinere Werte = schnelleres Laden\n"
         "bei vielen Bildern im Editor.\n\n"
         "'ROI-Labels im Editor anzeigen'\n"
         "blendet Label-Texte auf ROI-Rahmen ein.",
         None),
        ("Autosave & Backup",
         "Autosave aktivieren + Intervall setzen\n"
         "(Standard: 300 s = 5 Minuten).\n"
         "Deaktiviert → manuell Strg+S nutzen.\n\n"
         "'Backup vor Speichern' erstellt bei\n"
         "jedem Speichern eine .bak-Kopie\n"
         "im Projektverzeichnis.\n\n"
         "Im Fehlerfall: .bak → .json umbenennen.",
         None),
        ("REST-API & Web-Dashboard",
         "REST-Server starten:\n"
         "Port einstellen (Standard: 8765)\n"
         "→ 'API starten' klicken\n\n"
         "Öffentliche Endpunkte (kein Key nötig):\n"
         "GET  /api/status    → Projektstatus\n"
         "GET  /dashboard     → Browser-Dashboard\n\n"
         "Geschützte Endpunkte:\n"
         "GET  /api/labels    → Klassenliste\n"
         "GET  /api/images    → Bilder + Labels\n"
         "POST /api/images/label → Label setzen\n"
         "GET  /api/scores    → Score-Puffer\n"
         "GET  /api/events    → Alarm-Events\n\n"
         "API-Key (empfohlen):\n"
         "'Generieren' → Schlüssel erstellen.\n"
         "Anfragen brauchen dann den Header:\n"
         "X-Api-Key: <schlüssel>",
         "API starten"),
        ("MQTT-Alarm konfigurieren",
         "MQTT sendet bei jedem Anomalie-Alarm\n"
         "ein JSON-Event an einen Broker.\n\n"
         "Einrichten:\n"
         "1. 'MQTT-Publishing aktiviert' ✓\n"
         "2. Broker-Host eintragen\n"
         "   (z. B. 'localhost' oder IP)\n"
         "3. Port: Standard 1883\n"
         "4. Topic: 'picture_studio/anomaly'\n"
         "5. User/Passwort: nur wenn Broker\n"
         "   Authentifizierung erfordert\n"
         "6. 'Einstellungen speichern' klicken\n\n"
         "Voraussetzung: pip install paho-mqtt",
         None),
        ("SSH-Profile anlegen",
         "'Profil hinzufügen' für SSH-Ferntraining:\n"
         "• Profilname (frei wählbar)\n"
         "• Host / IP-Adresse des Servers\n"
         "• Benutzername (SSH-Login)\n"
         "• SSH-Key-Pfad (empfohlen)\n"
         "  z. B. ~/.ssh/id_rsa\n\n"
         "Profil auf der Training-Seite wählen.\n\n"
         "SSH-Key erstellen:\n"
         "ssh-keygen -t ed25519 -f ~/.ssh/gpu_key",
         "Profil hinzufügen"),
        ("Alarmierung",
         "Im Abschnitt <b>Alarmierung</b> konfigurierst du E-Mail- und Webhook-Benachrichtigungen. "
         "Bei jeder erkannten Anomalie wird automatisch eine Nachricht verschickt — "
         "mit Score, Schwellwert und dem Alarm-Bild als Anhang.",
         None),
        ("Industrieanbindung",
         "Im Abschnitt <b>Industrieanbindung</b> verbindest du PictureStudio direkt mit deiner SPS. "
         "Bei erkannten Anomalien wird ein <b>OPC-UA-Node</b> oder eine <b>Modbus-Coil</b> gesetzt — "
         "ohne zusätzliche Middleware.",
         None),
        ("Monitor-Client",
         "Das trainierte Modell kann mit <b>monitor.py</b> auch ohne die Studio-Oberfläche genutzt werden. "
         "Starte es im Terminal: <code>python monitor.py --model pfad/zum/modell.pt</code>. "
         "Kamera, ROI und Schwellwert werden automatisch aus den Modell-Metadaten geladen.",
         None),
    ],
    8: [  # Kamera & Videoanalyse (Dialog, kein eigener Stack-Index)
        ("Kamera & Videoanalyse – Überblick",
         "Datei → Kamera aufnehmen… (Strg+K)\n\n"
         "Drei Quelltypen stehen zur Wahl:\n"
         "• USB Kamera – direkt angeschlossene Kamera\n"
         "• IP Kamera – RTSP/HTTP-Netzwerkkamera\n"
         "• Video-Datei – MP4, AVI, MOV, MKV…\n\n"
         "Typischer Anomalie-Ablauf:\n"
         "1. Quelle wählen & verbinden\n"
         "2. ROI setzen (optional)\n"
         "3. Normalframes aufnehmen\n"
         "4. Autoencoder trainieren\n"
         "5. Live-Scoring aktivieren",
         None),
        ("Kameraquelle verbinden",
         "USB Kamera-Tab:\n"
         "Kamera im Dropdown wählen\n"
         "(Systemname wird automatisch erkannt)\n"
         "→ 'Verbinden' klicken.\n\n"
         "IP Kamera-Tab:\n"
         "URL eingeben, z. B.:\n"
         "rtsp://user:pass@192.168.1.100:554/stream\n"
         "→ 'Verbinden' klicken.\n\n"
         "Video-Datei-Tab:\n"
         "'Datei wählen…' → Video öffnen.\n"
         "FPS = 0 → originale Geschwindigkeit.\n\n"
         "Live-Monitor (Seite 'Live-Monitoring'):\n"
         "Dropdown enthält USB- und IP-Kameras\n"
         "sowie 'Videodatei (MP4, AVI, …)' direkt.\n"
         "FPS wird automatisch aus der Datei gelesen.\n"
         "Bricht eine Live-Verbindung ab → Auto-\n"
         "Reconnect alle 5 s (gelber Status).",
         "Verbinden"),
        ("ROI – Analysebereich setzen",
         "'ROI aufziehen' klicken, dann im\n"
         "Vorschaubild ein Rechteck ziehen.\n\n"
         "Nur dieser Bereich wird für Training\n"
         "und Scoring verwendet — Hintergrund-\n"
         "bewegungen werden ignoriert.\n\n"
         "Empfehlung: ROI immer setzen wenn\n"
         "der Prozess lokal begrenzt ist\n"
         "(z. B. Fließband, Bauteil, Maschine).\n\n"
         "'ROI löschen' → ganzes Bild analysieren.",
         "ROI aufziehen"),
        ("Normalframes aufnehmen",
         "Normalprozess vor die Kamera bringen.\n\n"
         "Anzahl einstellen:\n"
         "• Minimum: 50–100 Frames\n"
         "• Empfehlung: 150–300 Frames\n"
         "• Maximum: 25.000 Frames\n\n"
         "'Aufnehmen starten' → Frames werden\n"
         "automatisch gesammelt.\n\n"
         "Wichtig:\n"
         "• Alle Varianten des Normalzustands\n"
         "  abdecken (Werkstücke, Winkel, Licht)\n"
         "• Kamera NICHT bewegen\n"
         "• Beleuchtung konstant halten\n\n"
         "Kein Anomalie-Beispiel nötig!",
         "Aufnehmen starten"),
        ("Autoencoder trainieren",
         "Epochen einstellen (Standard: 40).\n"
         "Mehr Epochen = besser, aber langsamer.\n\n"
         "'Training starten' klicken.\n"
         "Loss und Fortschritt werden live angezeigt.\n\n"
         "Nach dem Training:\n"
         "Schwellwert wird automatisch berechnet:\n"
         "Mittelwert + 2,5 × Standardabweichung\n"
         "der Trainings-Rekonstruktionsfehler.\n\n"
         "→ Danach Schwellwert kalibrieren!",
         "Training starten"),
        ("Schwellwert kalibrieren",
         "'📊 Schwellwert kalibrieren…' klicken.\n\n"
         "Histogramm der Score-Verteilung zeigt\n"
         "wo normaler vs. anomaler Bereich liegt.\n\n"
         "Vorschläge:\n"
         "• µ+1σ → sehr sensitiv\n"
         "• µ+2σ → ausgewogen (Empfehlung)\n"
         "• µ+3σ → nur grobe Abweichungen\n\n"
         "Für Fertigungsüberwachung mit µ+2σ\n"
         "starten und dann beobachten:\n"
         "• Zu viele Alarme → Schwellwert erhöhen\n"
         "• Anomalien werden nicht erkannt\n"
         "  → Schwellwert senken",
         None),
        ("Live-Scoring & Bewegungsfilter",
         "'Scoring aktiv' Button aktivieren.\n\n"
         "Score-Anzeige:\n"
         "Grün = Normalzustand\n"
         "Rot = Anomalie erkannt\n\n"
         "Glättung (Standard: 5 Frames):\n"
         "Alarm erst nach N Frames über Schwellwert.\n"
         "Verhindert Fehlalarme durch Einzelstörer.\n\n"
         "Bewegungsfilter:\n"
         "'Nur bei Bewegung prüfen' aktivieren.\n"
         "Frames ohne Bewegung werden übersprungen\n"
         "→ spart CPU, weniger Fehlalarme bei\n"
         "statischer Kamera ohne Aktivität.\n"
         "Sensitivität: % der Pixel die sich\n"
         "ändern müssen.",
         None),
        ("Alarm-Pause & automatisch speichern",
         "Alarm-Pause (Standard: 30 s):\n"
         "Mindestabstand zwischen zwei Events.\n"
         "Verhindert hunderte Duplikate wenn\n"
         "eine Anomalie länger anhält.\n"
         "0 = kein Filter (alle Frames gespeichert).\n\n"
         "'Anomalie-Frames automatisch speichern':\n"
         "Jeder Alarm-Frame wird als PNG gespeichert\n"
         "mit rotem Bounding-Box-Rahmen um die\n"
         "anomale Region.\n\n"
         "Eine JSON-Sidecar-Datei enthält:\n"
         "Score, Schwellwert, Zeitstempel.\n\n"
         "False Positive: Rechtsklick auf Event\n"
         "in der Ereignis-Liste markieren.",
         None),
        ("Auto-Retraining — Lernzyklus schließen",
         "Nach einer konfigurierbaren Anzahl Alarme\n"
         "(Standard: 20) erscheint ein blauer Banner:\n\n"
         "  ⚠ N Alarme — Retraining empfohlen\n\n"
         "→ 'Jetzt trainieren': Wechselt direkt\n"
         "  zur Training-Seite (Bildprojekt) mit\n"
         "  allen Alarm-Frames für ein Nachtraining.\n"
         "→ '✕': Banner schließen, Zähler reset.\n\n"
         "Der Zyklus: Kamera läuft → Alarme sammeln\n"
         "→ Banner → Nachtraining → verbessertes\n"
         "Modell → weiter überwachen.\n\n"
         "headless (monitor.py): Flag-Datei wird\n"
         "angelegt sobald Schwellwert erreicht ist.",
         None),
        ("Shadow Mode — A/B Modellvergleich",
         "'Shadow-Modell laden…' (lila Button) lädt\n"
         "ein zweites Anomalie-Modell parallel.\n\n"
         "Beide Modelle bewerten jeden Frame:\n"
         "• Haupt-Modell: blauer Balken (wie gehabt)\n"
         "• Shadow-Modell: oranger Balken darunter\n\n"
         "Divergenz-Anzeige:\n"
         "• Δ0.00xxx — Differenz der Scores\n"
         "• ⚡ Divergenz — beide Modelle sind sich\n"
         "  uneinig (Alarm vs. Normal)\n\n"
         "Divergenz-Events werden separat geloggt:\n"
         "anomaly_events/shadow_divergences.csv\n\n"
         "Anwendungsfall: altes vs. neues Modell\n"
         "parallel testen bevor dem Rollout.",
         None),
        ("MQTT-Alarm & Event-Log",
         "Bei jedem Alarm:\n\n"
         "Event-Log (CSV): automatisch geloggt\n"
         "mit Zeitstempel, Score, Frame-Pfad.\n"
         "'Log öffnen' → CSV im Standardprogramm.\n\n"
         "MQTT: wenn in Einstellungen konfiguriert,\n"
         "wird ein JSON-Payload an den Broker\n"
         "publiziert:\n"
         "{\n"
         "  event: 'anomaly',\n"
         "  score: 0.042,\n"
         "  threshold: 0.025,\n"
         "  timestamp_utc: '...'\n"
         "}\n\n"
         "Voraussetzung: pip install paho-mqtt\n"
         "MQTT in Einstellungen konfigurieren.",
         "Log öffnen"),
        ("Modell sichern & exportieren",
         "'Speichern…' → .pth-Datei sichern.\n"
         "SHA256-Prüfsumme wird automatisch\n"
         "erstellt und beim Laden verifiziert.\n\n"
         "'Laden…' → gespeichertes Modell laden.\n"
         "Kein Neutraining nötig.\n\n"
         "'ℹ Info' → Metadaten anzeigen:\n"
         "Trainingszeit, Frames, Epochen,\n"
         "Gerät, Schwellwert, SHA256.\n\n"
         "Für andere Systeme exportieren:\n"
         "'ONNX exportieren' → .onnx (Opset 17)\n"
         "Einsatz in: ONNX Runtime, OpenCV DNN,\n"
         "TensorRT, C++, Edge-Geräten\n\n"
         "'TorchScript exportieren' → .pt\n"
         "Einsatz in: PyTorch C++ API",
         None),
        ("ONNX-Export",
         "Das geladene Anomalie-Modell kann mit <b>Als ONNX exportieren</b> für den Einsatz ohne PyTorch gespeichert werden. "
         "Das ONNX-Format läuft auf Raspberry Pi und anderen Edge-Geräten mit <code>pip install onnxruntime</code>.",
         "Als ONNX exportieren"),
        ("Kamera-Einstellungen",
         "Helligkeit, Kontrast, Sättigung, Schärfe und Belichtung direkt anpassen — "
         "Änderungen wirken sofort auf den laufenden Stream.\n\n"
         "Diese Einstellungen sind auch direkt im Aufnahme-Dialog\n"
         "für Bildklassifikation verfügbar (Daten-Seite → Kamera-Button).\n"
         "Von der CameraPage übergebene Werte werden als Startwerte übernommen.",
         "Zurücksetzen"),
        ("Vorverarbeitungsfilter",
         "Wähle einen Filter (Graustufen, Canny-Kanten, Sobel, Laplacian) der auf "
         "jeden Frame angewendet wird — für Anzeige und optional auch für das Scoring.\n\n"
         "Der Filter-Dropdown steht ebenfalls im Aufnahme-Dialog für\n"
         "Bildklassifikation zur Verfügung.\n\n"
         "Nützlich für industrielle Inspektion: das Modell lernt Kanten statt Texturen.",
         None),
        ("Hyperparameter-Suche (Anomalie)",
         "Nach dem Sammeln von Frames: '⚙ Hyperparameter-Suche…' startet eine "
         "Optuna-Studie die Architektur (base_ch), Lernrate und Batch-Größe optimiert.\n\n"
         "Ein Live-Log-Fenster zeigt jeden Versuch:\n"
         "base_ch · lr · batch → Threshold  ★ Neu bestes!\n\n"
         "Beste Parameter werden per Klick direkt auf den Autoencoder angewendet.\n\n"
         "Button-Reihenfolge:\n"
         "① Hyperparameter-Suche → ② Training starten → ③ Stoppen",
         "Hyperparameter-Suche"),
        ("Batch-Analyse & Live-Aufzeichnung",
         "Batch-Analyse (Tab 📁 Batch):\n"
         "'Ordner wählen…' oder 'Dateien wählen…'\n"
         "→ 'Batch starten'\n"
         "Alle Bilder werden mit dem Autoencoder\n"
         "bewertet. 'CSV exportieren' → Ergebnisse.\n\n"
         "Live-Aufzeichnung:\n"
         "'⏺ Aufnahme starten' → laufendes MP4\n"
         "wird gespeichert.\n"
         "FPS daneben einstellen (Standard: 15).\n"
         "'Aufnahme stoppen' → Datei finalisiert.\n\n"
         "Audit-Log:\n"
         "Alle Modell-Aktionen (TRAINED/SAVED/\n"
         "LOADED/UNLOADED) werden protokolliert\n"
         "in audit/model_audit.jsonl.",
         None),
    ],
    10: [  # Multi-Kamera-Monitoring
        ("Multi-Kamera-Monitoring – Überblick",
         "Überwache 1–9 Kameras gleichzeitig,\n"
         "jede mit eigenem Modell und eigenem ROI.\n\n"
         "Grid: bis zu 4 Kanäle pro Seite (2×2).\n"
         "Bei mehr als 4 Kanälen: blättere mit\n"
         "◀ Vorherige / Nächste ▶ zwischen Seiten.\n\n"
         "REST-API: per-Kanal-Endpunkte unter\n"
         "/api/mc/channels, /api/mc/scores?channel=N\n"
         "und /api/mc/latest_alarm?channel=N.",
         None),
        ("Kanalzahl festlegen",
         "Das Drehfeld 'Kanäle: [2]' oben links\n"
         "legt die Anzahl der Kanäle fest (1–9).\n\n"
         "Standard: 2 Kanäle.\n\n"
         "Beim Erhöhen werden neue leere Kanäle\n"
         "hinzugefügt. Beim Verringern werden\n"
         "die letzten Kanäle gestoppt und entfernt.\n\n"
         "Bereits konfigurierte Kanäle\n"
         "(Modell + Kamera) behalten ihre\n"
         "Einstellungen.",
         "Kanäle"),
        ("Kanal konfigurieren",
         "Klicke ⚙ Konfigurieren im jeweiligen\n"
         "Kanal-Widget.\n\n"
         "Wähle:\n"
         "• Kamera – USB-Index aus der Liste\n"
         "• Modell – .pth (PyTorch) oder\n"
         "  .onnx (onnxruntime, kein PyTorch)\n\n"
         "Nach OK:\n"
         "• Kanalanzeige zeigt Kamera und Modell\n"
         "• Starten-Button wird freigeschaltet\n"
         "• ROI und Schwellwert werden automatisch\n"
         "  aus den Modell-Metadaten geladen.",
         "Konfigurieren"),
        ("Kanäle starten und stoppen",
         "Einzelner Kanal:\n"
         "▶ Starten / ■ Stoppen im Kanal-Widget.\n\n"
         "Alle auf einmal:\n"
         "Alle starten → startet alle\n"
         "konfigurierten (aber noch nicht\n"
         "laufenden) Kanäle.\n\n"
         "Alle stoppen → stoppt alle\n"
         "aktiven Kanäle sofort.\n\n"
         "Status-Farben:\n"
         "Grün = Normal | Rot = ANOMALIE",
         "Alle starten"),
        ("Seitennavigation (> 4 Kanäle)",
         "Bei mehr als 4 Kanälen erscheint\n"
         "automatisch die Navigationszeile:\n\n"
         "◀ Vorherige  Seite N / Gesamt  Nächste ▶\n\n"
         "Jede Seite zeigt 4 Kanäle im 2×2-Raster.\n"
         "Beispiel bei 9 Kanälen:\n"
         "• Seite 1: Kanäle 1–4\n"
         "• Seite 2: Kanäle 5–8\n"
         "• Seite 3: Kanal 9\n\n"
         "Kanäle auf anderen Seiten laufen\n"
         "weiter — auch wenn sie nicht sichtbar sind.",
         "Nächste"),
        ("Alarm-Protokoll & JPEG-Speicherung",
         "Alarm-Protokoll (unten):\n"
         "Jede Anomalie erscheint mit:\n"
         "• Uhrzeit und Kanal-Nummer\n"
         "• Score und Schwellwert\n"
         "• Dateiname des gespeicherten Frames\n\n"
         "Alarm-JPEG:\n"
         "Bei jedem Alarm wird automatisch\n"
         "ein Schnappschuss gespeichert unter:\n"
         "monitor_logs/multi_cam/\n"
         "mc_chN_YYYYMMDDTHHMMSSZ.jpg\n\n"
         "E-Mail/Webhook aus den Einstellungen\n"
         "gelten für alle Kanäle.",
         None),
        ("REST-API – Per-Kanal-Endpunkte",
         "Wenn der REST-Server läuft\n"
         "(Einstellungen → REST-API → Starten),\n"
         "sind diese Endpunkte verfügbar:\n\n"
         "GET /api/mc/channels\n"
         "→ alle Kanäle: Score, is_alarm,\n"
         "  event_count, cam_status\n\n"
         "GET /api/mc/scores?channel=N\n"
         "→ Score-Verlauf für Kanal N\n\n"
         "GET /api/mc/latest_alarm?channel=N\n"
         "→ letzter Alarm für Kanal N\n\n"
         "Das /dashboard zeigt automatisch\n"
         "eine Multi-Kamera-Sektion.",
         None),
    ],
    11: [  # Anomalie-Clustering
        ("Anomalie-Clustering – Überblick",
         "Diese Seite gruppiert deine Alarm-Bilder automatisch\n"
         "nach visueller Ähnlichkeit – ohne manuelle Annotation.\n\n"
         "Der k-Means-Algorithmus erkennt, welche Fehlertypen\n"
         "im Datensatz vorkommen und wie häufig sie sind.\n\n"
         "Voraussetzung: Projekt mit Bildern, die als\n"
         "Anomalie (oder Fehler) gelabelt sind.",
         None),
        ("Cluster-Anzahl einstellen & starten",
         "Anzahl Cluster einstellen (2–20).\n\n"
         "Empfehlung: mit 5 Clustern starten.\n"
         "Sind Bilder innerhalb eines Clusters\n"
         "optisch zu verschieden → Anzahl erhöhen.\n\n"
         "Dann 'Clustering starten' klicken.\n"
         "Das Modell extrahiert Merkmale und\n"
         "berechnet die Cluster (wenige Sekunden).",
         "Clustering starten"),
        ("Ergebnisse im Cluster-Browser lesen",
         "Nach der Berechnung erscheinen Cluster-Karten:\n\n"
         "• Jede Karte zeigt das repräsentative Bild\n"
         "  (den Cluster-Mittelpunkt) als Thumbnail\n"
         "• Darunter steht die Bildanzahl des Clusters\n"
         "• Karte anklicken → alle Bilder des Clusters\n"
         "  werden in der Thumbnail-Liste angezeigt\n\n"
         "Cluster mit wenigen Bildern = seltene Anomalie.\n"
         "Cluster mit vielen Bildern = häufige Fehlerart.",
         None),
        ("CSV exportieren",
         "Klicke 'CSV exportieren' um die Ergebnisse\n"
         "als Tabelle zu speichern.\n\n"
         "Die CSV enthält drei Spalten:\n"
         "• path — absoluter Dateipfad des Bildes\n"
         "• cluster_id — Cluster-Nummer (0-basiert)\n"
         "• is_representative — True für das Bild\n"
         "  das dem Cluster-Mittelpunkt am nächsten liegt\n\n"
         "Die CSV kann direkt in Excel oder Python\n"
         "für weitere Analysen genutzt werden.",
         "CSV exportieren"),
    ],
    9: [  # Batch-Inferenz
        ("Batch-Inferenz",
         "Klassifiziere einen ganzen Ordner\n"
         "mit einem trainierten Modell in einem Durchlauf.\n\n"
         "Ergebnis: sortierbare Tabelle mit\n"
         "Dateiname, Klasse, Konfidenz, Fehler\n"
         "und CSV-Export.",
         None),
        ("Modell laden",
         "Modell aus dem Projekt wählen\n"
         "(Dropdown zeigt alle Trainingsläufe)\n"
         "oder externe .pth-Datei laden.\n\n"
         "'Ausgewähltes Modell laden' → bereit.",
         "Ausgewähltes Modell laden"),
        ("Bilder auswählen",
         "'Ordner wählen…' → Ordner mit Bildern\n"
         "oder 'Projektbilder verwenden' um alle\n"
         "Bilder des aktuellen Projekts zu nehmen.\n\n"
         "Konfidenz-Filter: nur Ergebnisse über\n"
         "dem Schwellwert werden angezeigt.",
         "Ordner wählen"),
        ("Batch starten & exportieren",
         "'Batch starten' → alle Bilder werden\n"
         "nacheinander klassifiziert.\n\n"
         "Ergebnisse in der Tabelle:\n"
         "• Klick auf Spaltenköpfe → sortieren\n"
         "• Rot = unter Min. Confidence\n\n"
         "'Als CSV exportieren' → Tabelle als\n"
         "CSV-Datei speichern.",
         "Batch starten"),
    ],
    12: [  # Datensatz-Statistiken
        ("Datensatz-Statistiken – Überblick",
         "Die Datensatz-Statistiken-Seite analysiert\n"
         "den aktuellen Datensatz auf Qualität und\n"
         "Ausgewogenheit.\n\n"
         "• Klassenverteilung mit Balken\n"
         "• Format- und Größenstatistiken\n"
         "• Label-Rate (% annotierter Bilder)\n"
         "• Perceptual-Hash-Duplikaterkennung",
         None),
        ("Klassenverteilung prüfen",
         "Die Balken zeigen Anzahl und Anteil\n"
         "jeder Klasse.\n\n"
         "Starkes Ungleichgewicht (z. B. 90%:10%)\n"
         "führt zu schlechtem Modell für die\n"
         "kleinere Klasse.\n\n"
         "Lösung: WeightedRandomSampler auf der\n"
         "Training-Seite aktivieren oder mehr\n"
         "Bilder der kleinen Klasse beschaffen.",
         None),
        ("Duplikaterkennung",
         "'Analyse aktualisieren' starten.\n\n"
         "Die Seite erkennt visuell ähnliche\n"
         "Bilder via perceptual hashing.\n\n"
         "Duplikate können das Training verzerren\n"
         "(Overfitting auf wiederholte Bilder).\n\n"
         "Benötigt: pip install imagehash",
         "Analyse aktualisieren"),
    ],
    13: [  # Video-Annotation
        ("Video-Annotation – Überblick",
         "Annotiere einzelne Frames direkt aus\n"
         "einer Videodatei ohne vorherigen Export.\n\n"
         "Ideal für:\n"
         "• Kurze Trainingsvideos\n"
         "• Auswahl interessanter Frames\n"
         "• Schnelle Datensatz-Erweiterung",
         None),
        ("Video laden & navigieren",
         "'Video laden…' → Datei wählen\n"
         "(MP4, AVI, MOV, MKV, …)\n\n"
         "Schieberegler = Frame-Navigation\n"
         "Frame-Nummer und Zeitstempel werden\n"
         "oben angezeigt.\n\n"
         "Benötigt: pip install opencv-python",
         "Video laden"),
        ("Frame extrahieren & labeln",
         "'Frame extrahieren' → Bild speichern\n\n"
         "Label aus Dropdown wählen.\n\n"
         "'Zum Projekt hinzufügen' → Frame + Label\n"
         "werden dem aktuellen Projekt hinzugefügt.\n\n"
         "Danach auf der Labeling-Seite prüfen.",
         "Frame extrahieren"),
    ],
    14: [  # Fleet-Management
        ("Fleet-Management – Überblick",
         "Überwache mehrere remote monitor.py-\n"
         "Instanzen (Edge-Geräte, Server, VMs)\n"
         "von einer zentralen Stelle.\n\n"
         "Jedes Gerät wird per HTTP-Poll abgefragt:\n"
         "• Online/Offline-Status\n"
         "• Letzter Score\n"
         "• Letzter Alarm-Zeitstempel",
         None),
        ("Gerät hinzufügen",
         "Klicke '+ Gerät hinzufügen'.\n\n"
         "Name: frei wählbar\n"
         "URL: Basis-URL des monitor.py-Servers\n"
         "z. B. http://192.168.1.100:8766\n"
         "API-Key: wenn --api-key gesetzt\n\n"
         "Auf dem Gerät muss laufen:\n"
         "python monitor.py --api-port 8766",
         "+ Gerät hinzufügen"),
        ("Status & Auto-Refresh",
         "'Alle aktualisieren' → einmalige Abfrage\n\n"
         "'Auto-Refresh (30 s)' → aktiviert\n"
         "automatische Abfrage alle 30 Sekunden.\n\n"
         "'Dashboard' in Aktionen → öffnet das\n"
         "Web-Dashboard des Geräts im Browser.\n\n"
         "Gerätliste wird in QSettings gespeichert.",
         "Alle aktualisieren"),
        ("Remote-Training & Hot-Swap Deploy",
         "Klicke 'Training' in der Aktionen-Spalte.\n\n"
         "Tab 1 — Frames & Training:\n"
         "• Gerätestatus prüfen (frame_count)\n"
         "• Frames per GET /api/frames herunterladen\n"
         "• Modell lokal trainieren\n\n"
         "Tab 2 — Deployen:\n"
         "• Schwellwert anpassen\n"
         "• Modell per POST /api/deploy hochladen\n"
         "• Daemon tauscht Modell ohne Neustart",
         "Training"),
    ],

    15: [  # Objekterkennung
        ("Objekterkennung – Überblick",
         "YOLOv8-basierte Objekterkennung:\n"
         "Erkennt und lokalisiert mehrere Objekte\n"
         "gleichzeitig mit Bounding Boxes.\n\n"
         "Voraussetzung: pip install ultralytics\n\n"
         "Unterschied zur Klassifikation:\n"
         "• Mehrere Objekte pro Bild möglich\n"
         "• Gibt Position + Klasse zurück\n"
         "• ROIs = Trainingsannotationen",
         None),
        ("Schritt 1 – Bilder annotieren",
         "Im Labeling-Editor ROIs zeichnen\n"
         "und jedem ROI ein Label zuweisen.\n\n"
         "Jeder ROI wird zur Bounding-Box-\n"
         "Annotation für das YOLO-Training.\n"
         "Mehrere ROIs pro Bild sind möglich.\n\n"
         "→ Dann hier zurückkommen.",
         None),
        ("Schritt 2 – Dataset vorbereiten",
         "Klicke 'Dataset vorbereiten'.\n\n"
         "Das System konvertiert automatisch:\n"
         "• ROI-Koordinaten → YOLO-Format\n"
         "• 80 % Training / 20 % Validation\n"
         "• Erstellt data.yaml für YOLOv8\n\n"
         "Status zeigt: Bilder, Klassen,\n"
         "Annotationen.",
         "Dataset vorbereiten"),
        ("Schritt 3 – Modell & Training",
         "Modellgröße wählen:\n"
         "• yolov8n — sehr schnell (CPU ok)\n"
         "• yolov8s — schnell, gute Qualität\n"
         "• yolov8m — empfohlen für Produktion\n"
         "• yolov8l — maximale Genauigkeit\n\n"
         "Epochen: 50 (Standard)\n"
         "Bildgröße: 640 px (Standard)\n\n"
         "Klicke '⚡ Training starten'.",
         "Training starten"),
        ("Schritt 4 – Erkennung",
         "'Bild wählen' → Einzelbild mit\n"
         "eingezeichneten Bounding Boxes.\n\n"
         "'Ordner…' + 'Erkennung starten' →\n"
         "alle Bilder im Ordner werden\n"
         "analysiert, Tabelle zeigt Ergebnisse.\n\n"
         "Konfidenz-Schwelle einstellen:\n"
         "0.25 = Standard\n"
         "0.5+ = weniger Fehlerkennungen\n\n"
         "CSV-Export für Weiterverarbeitung.",
         "Erkennung starten"),
    ],

    16: [  # Data Drift
        ("Data Drift – Überblick",
         "Erkennt automatisch, wenn sich\n"
         "Produktionsbilder statistisch von\n"
         "den Trainingsbildern unterscheiden.\n\n"
         "Typische Ursachen:\n"
         "• Kamera ausgetauscht/verstellt\n"
         "• Beleuchtung geändert\n"
         "• Bildqualität verschlechtert\n\n"
         "Kein Data Scientist nötig —\n"
         "Z-Score-basierte Erkennung.",
         None),
        ("Schritt 1 – Baseline erstellen",
         "Klicke '📊 Baseline aus\n"
         "Projektbildern erstellen'.\n\n"
         "Das System analysiert alle\n"
         "Trainingsbilder und speichert\n"
         "deren statistische Verteilung:\n"
         "• Farbmittelwert / -streuung\n"
         "• Schärfe (Laplacian)\n"
         "• Kantendichte (Canny)\n"
         "• Graustufenhistogramm",
         "Baseline aus Projektbildern erstellen"),
        ("Schritt 2 – Schwellwert",
         "Stelle den Max. Z-Score ein.\n\n"
         "Z-Score = wie viele Standard-\n"
         "abweichungen ein Bild von der\n"
         "Trainingsdistribution abweicht.\n\n"
         "• 3.0 = Standard (statistisch\n"
         "  unwahrscheinlich)\n"
         "• Niedriger = empfindlicher\n"
         "• Höher = nur starke Abweichungen",
         None),
        ("Schritt 3 – Analysieren",
         "Ordner mit Produktionsbildern\n"
         "wählen → '🔍 Drift analysieren'.\n\n"
         "Farbcodierung der Ergebnisse:\n"
         "🟢 Grün = kein Drift\n"
         "🟠 Orange = leichter Drift\n"
         "🔴 Rot = starker Drift\n\n"
         "Zusammenfassung zeigt Anteil\n"
         "gedrifteter Bilder.",
         "Drift analysieren"),
        ("Schritt 4 – Reagieren",
         "Gedriftete Bilder identifizieren\n"
         "und dem Training hinzufügen:\n\n"
         "1. Gedriftete Bilder in den\n"
         "   Projektordner kopieren\n"
         "2. Daten-Seite: Ordner neu laden\n"
         "3. Labeling-Seite: Labels vergeben\n"
         "4. Neu trainieren\n\n"
         "Baseline speichern ('💾') für\n"
         "spätere Vergleiche.",
         None),
    ],

    17: [  # Anomalie-Training (Video-Modus)
        ("🧠 Anomalie-Training – Überblick",
         "Diese Seite ist der Einstieg ins\n"
         "Anomalie-Tracking im Video-Modus.\n\n"
         "Der 3-Schritt-Workflow:\n"
         "1. Normale Frames sammeln\n"
         "2. Autoencoder trainieren\n"
         "3. Live-Monitoring starten\n\n"
         "Kein gelabelter Datensatz nötig —\n"
         "nur Bilder des Normalzustands.",
         None),
        ("Schritt 1 – Frames aufnehmen",
         "Klicke '🎬 Frames aufnehmen\n"
         "& Trainieren'.\n\n"
         "Im Dialog:\n"
         "• Kamera verbinden (USB / IP)\n"
         "• Optional: ROI-Bereich ziehen\n"
         "• 'Frames sammeln' starten\n"
         "• 50–200 Frames des Normal-\n"
         "  zustands aufnehmen\n\n"
         "Konstante Beleuchtung = bessere\n"
         "Erkennung.",
         "🎬  Frames aufnehmen & Trainieren"),
        ("Schritt 2 – Training starten",
         "Im selben Dialog:\n\n"
         "• Epochen: 30–50 für erste Tests\n"
         "• 'Training starten' klicken\n"
         "• Fortschrittsbalken zeigt\n"
         "  den Trainingsverlauf\n\n"
         "Nach dem Training:\n"
         "• Modell wird automatisch geladen\n"
         "• Schwellwert wird gesetzt\n"
         "• Status-Karte aktualisiert sich",
         None),
        ("Schritt 3 – Live-Monitoring",
         "Das Modell ist geladen —\n"
         "jetzt zur Kamera-Seite wechseln.\n\n"
         "• '🎥 Live & Anomalie' anklicken\n"
         "• Kamera verbinden\n"
         "• 'Scoring aktivieren'\n\n"
         "Normales Teil → niedriger Score\n"
         "Fehler / Fremdkörper → Alarm 🔴\n\n"
         "Schwellwert feinjustieren bis\n"
         "False-Positive-Rate akzeptabel.",
         None),
    ],
}


# ---------------------------------------------------------------------------
# Highlight overlay
# ---------------------------------------------------------------------------

class HighlightOverlay(QFrame):
    """Transparent orange-bordered frame that highlights a widget."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setStyleSheet(
            "background: rgba(243,156,18,15);"
            "border: 3px solid #F39C12;"
            "border-radius: 7px;"
        )
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.hide()

    def highlight(self, widget: QWidget | None) -> None:
        if widget is None or not widget.isVisible():
            self.hide()
            return
        pad = 5
        pos = widget.mapTo(self.parent(), QPoint(0, 0))
        self.setGeometry(
            pos.x() - pad, pos.y() - pad,
            widget.width() + 2 * pad, widget.height() + 2 * pad,
        )
        self.raise_()
        self.show()


# ---------------------------------------------------------------------------
# Guide tour panel
# ---------------------------------------------------------------------------

class GuideTour(QFrame):
    """Floating step-by-step guide panel anchored to the main window."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self._main = main_window
        self._steps: list = []
        self._index: int = 0
        self._page_widget: QWidget | None = None

        self._overlay = HighlightOverlay(main_window)

        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            GuideTour {
                background: #1C2A3A;
                border: 2px solid #2980B9;
                border-radius: 10px;
            }
        """)
        self.setFixedWidth(440)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)

        # Header row
        header_row = QHBoxLayout()
        self._header_lbl = QLabel("Geführte Tour")
        self._header_lbl.setStyleSheet("color:#F39C12; font-weight:bold; font-size:13px;")
        header_row.addWidget(self._header_lbl)
        header_row.addStretch()
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(26, 26)
        close_btn.setStyleSheet(
            "QPushButton{background:transparent;color:#aaa;border:none;font-size:16px;}"
            "QPushButton:hover{color:white;}"
        )
        close_btn.clicked.connect(self.stop)
        header_row.addWidget(close_btn)
        layout.addLayout(header_row)

        # Title
        self._title_lbl = QLabel()
        self._title_lbl.setStyleSheet("color:white; font-weight:bold; font-size:16px;")
        self._title_lbl.setWordWrap(True)
        layout.addWidget(self._title_lbl)

        # Description
        self._desc_lbl = QLabel()
        self._desc_lbl.setStyleSheet("color:#D6DBDF; font-size:13px; line-height:1.6;")
        self._desc_lbl.setWordWrap(True)
        self._desc_lbl.setMinimumHeight(140)
        self._desc_lbl.setAlignment(Qt.AlignTop)
        layout.addWidget(self._desc_lbl)

        # Step counter
        self._counter_lbl = QLabel()
        self._counter_lbl.setStyleSheet("color:#95A5A6; font-size:12px;")
        self._counter_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._counter_lbl)

        # Navigation buttons
        nav = QHBoxLayout()
        self._back_btn = QPushButton("← Zurück")
        self._back_btn.setStyleSheet(
            "QPushButton{background:#2C3E50;color:#BDC3C7;border:1px solid #34495E;"
            "border-radius:6px;padding:7px 14px;font-size:13px;}"
            "QPushButton:hover{background:#34495E;color:white;}"
            "QPushButton:disabled{color:#555;}"
        )
        self._back_btn.clicked.connect(self._prev_step)
        nav.addWidget(self._back_btn)

        self._next_btn = QPushButton("Weiter →")
        self._next_btn.setStyleSheet(
            "QPushButton{background:#2980B9;color:white;border:none;"
            "border-radius:6px;padding:7px 14px;font-size:13px;font-weight:bold;}"
            "QPushButton:hover{background:#3498DB;}"
        )
        self._next_btn.clicked.connect(self._next_step)
        nav.addWidget(self._next_btn)
        layout.addLayout(nav)

        # Timer to keep highlight position in sync when window resizes
        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(250)
        self._sync_timer.timeout.connect(self._refresh_highlight)

        self.hide()

    # ------------------------------------------------------------------ public

    def start(self, page_index: int, page_widget: QWidget) -> None:
        """Start tour for the given page."""
        self._steps = TOUR_STEPS.get(page_index, [])
        self._page_widget = page_widget
        self._index = 0
        if not self._steps:
            return
        self._update_ui()
        self._reposition()
        self.show()
        self.raise_()
        self._overlay.raise_()
        self._sync_timer.start()

    def stop(self) -> None:
        self._sync_timer.stop()
        self._overlay.hide()
        self.hide()

    # ------------------------------------------------------------------ steps

    def _next_step(self) -> None:
        if self._index < len(self._steps) - 1:
            self._index += 1
            self._update_ui()
        else:
            self.stop()

    def _prev_step(self) -> None:
        if self._index > 0:
            self._index -= 1
            self._update_ui()

    def _update_ui(self) -> None:
        title, desc, btn_text = self._steps[self._index]
        total = len(self._steps)
        self._header_lbl.setText(f"Geführte Tour  •  Schritt {self._index + 1} von {total}")
        self._title_lbl.setText(title)
        self._desc_lbl.setText(desc)
        self._counter_lbl.setText("─" * 36)
        self._back_btn.setEnabled(self._index > 0)
        last = self._index == total - 1
        self._next_btn.setText("Tour beenden" if last else "Weiter →")
        self._next_btn.setStyleSheet(
            "QPushButton{background:%s;color:white;border:none;"
            "border-radius:5px;padding:5px 10px;font-weight:bold;}"
            "QPushButton:hover{background:%s;}"
            % (("#27AE60", "#2ECC71") if last else ("#2980B9", "#3498DB"))
        )
        self._highlight_step(btn_text)
        self.adjustSize()
        self._reposition()

    def _highlight_step(self, btn_text: str | None) -> None:
        if not btn_text or not self._page_widget:
            self._overlay.hide()
            return
        target = self._find_widget(btn_text)
        self._overlay.highlight(target)

    def _find_widget(self, text: str) -> QWidget | None:
        if not self._page_widget:
            return None
        text_lower = text.lower()
        for btn in self._page_widget.findChildren(QAbstractButton):
            if text_lower in btn.text().lower() and btn.isVisible():
                return btn
        # Fallback: search all visible children with matching text
        for lbl in self._page_widget.findChildren(QLabel):
            if text_lower in lbl.text().lower() and lbl.isVisible():
                return lbl
        return None

    def _refresh_highlight(self) -> None:
        if not self.isVisible():
            return
        _, _, btn_text = self._steps[self._index]
        self._highlight_step(btn_text)

    # ------------------------------------------------------------------ positioning

    def _reposition(self) -> None:
        """Keep panel in bottom-right corner of main window."""
        parent = self._main
        margin = 16
        x = parent.width() - self.width() - margin
        y = parent.height() - self.height() - margin - 30  # 30 = statusbar approx
        self.move(max(0, x), max(0, y))
