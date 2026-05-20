# 🚀 Erste Schritte

> **PictureStudio v2.3.0** — Kompletter Workflow von den ersten Bildern bis zum trainierten Modell

---

# 🚀 Erste Schritte – Kompletter Workflow

Dieser Guide führt dich Schritt für Schritt vom ersten Bild bis zum trainierten Modell.

---

## Phase 1 – Projekt anlegen

**Schritt 1 – Neues Projekt****
Menü *Datei → Neues Projekt* (`Strg+N`). Vergib einen aussagekräftigen Namen
und wähle den Projekttyp:

• 📸 Bildklassifikation** – klassifiziert Einzelbilder in Klassen**
• 🎬 Videoanalyse & Anomalie** – Kamera-Livestream mit Autoencoder-Erkennung**
Das Projekt wird als `.json`-Datei gespeichert.

---

## Phase 2 – Bilder laden

Schritt 2 – Bildordner hinzufügen****
Gehe zur Daten-Seite** → klicke *Bilder laden…* und wähle den Ordner.**
Alle `.jpg`, `.png`, `.bmp` und `.tiff` Dateien werden
automatisch hinzugefügt. Drag & Drop ins Fenster funktioniert ebenfalls.

Schritt 3 – Datensatz analysieren****
Klicke *Dataset analysieren* um zu prüfen:

• Fehlende Dateien** – Bilder die nicht mehr auffindbar sind**
• Duplikate** – identische Bilder (MD5-Hash)**
• Klassenungleichgewicht** – schlechtes Training wenn eine Klasse viel mehr Bilder hat

---

## Phase 3 – Labels definieren & Bilder annotieren

**Schritt 4 – Labels anlegen****
Gehe zur Labeling-Seite** → *Projekt → Labels verwalten…* (`Strg+L`).**
Füge für jede Klasse ein Label hinzu (z. B. "gut", "defekt", "unklar").

Schritt 5 – Bilder labeln****
1. Bild in der Thumbnail-Liste anklicken

2. `1`–`9` drücken für schnelle Label-Zuweisung

3. Mit `N` zum nächsten Bild, `P` zum vorherigen

Ziel:** Mindestens 50–100 Bilder pro Klasse. Für gute Modelle: 200+ pro Klasse.

**Schritt 6 – ROIs zeichnen (optional)****
Falls du nur einen bestimmten Bildbereich analysieren willst:

• `R` = Rechteck  • `E` = Ellipse  • `G` = Polygon

ROI in der rechten Liste auswählen → Label zuweisen.

---

## Phase 4 – Training konfigurieren & starten

Schritt 7 – Architektur & Hyperparameter****
Gehe zur Training-Seite**:**
• Architektur:** Starte mit `ResNet-18`**
• Epochen:** 20–30 für erste Tests, 50+ für finale Modelle**
• Gerät:** `auto` nutzt automatisch GPU/MPS/CPU**
• Early Stopping:** 5–7 Epochen verhindert Overfitting

**Schritt 8 – Training starten****
Klicke *Training starten*. Du siehst in Echtzeit Train-Loss, Val-Loss, Accuracy.

Das beste Modell wird automatisch bei höchster Val-Accuracy gespeichert.

> 💡 Woran erkenne ich gutes Training?**** ✓ Val-Loss sinkt parallel zu Train-Loss
 ✓ Val-Accuracy steigt kontinuierlich
 ✗ Val-Loss steigt während Train-Loss sinkt = Overfitting

---

## Phase 5 – Modell einsetzen

Schritt 9 – Neue Bilder klassifizieren****
Gehe zur Klassifikations-Seite**:**
1. *Modell laden (.pth)* → Modelldatei wählen

2. *Ordner…* → Ordner mit neuen Bildern wählen

3. *Alle Bilder klassifizieren* → Ergebnis mit Top-3 Vorhersagen

Schritt 10 – Ergebnisse exportieren****
Gehe zur Export-Seite**:**
1. *Ergebnisse aus letzter Inferenz laden*

2. Spalten konfigurieren (aktivieren, umbenennen)

3. *Excel exportieren*

> 💡 Fertig!** Für bessere Ergebnisse: mehr Daten → neu labeln → Training wiederholen.
