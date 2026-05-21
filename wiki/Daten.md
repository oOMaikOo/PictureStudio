# 📁 Daten

> **PictureStudio v2.3.0** — Bilder laden, Datensatz analysieren und Annotationen exportieren

---

# 📁 Daten

Bilder laden, Datensatz analysieren, Annotationen exportieren.

## Bilder laden

**Bilder laden…****
Wähle einen Ordner – alle `.jpg .png .bmp .tiff .webp` Dateien werden **rekursiv aus allen Unterordnern** geladen.

Alternativ: Bilder oder Ordner direkt ins Fenster ziehen (Drag & Drop – Unterordner ebenfalls rekursiv).

Bilder werden nicht kopiert, der Pfad wird gespeichert.

> 💡 Im Labeling-Reiter wird der Dateiname als **`Unterordner/Dateiname`** angezeigt, damit die Herkunft jedes Bildes auf einen Blick erkennbar ist.

Video importieren…****
Video-Datei (MP4, AVI, MOV, MKV, WebM) wählen.

Frame-Intervall einstellen: z. B. alle 5 Frames = ca. 6 Bilder/s bei 30 fps.

Die extrahierten Frames werden als PNG ins Projektverzeichnis gespeichert.

## Dataset analysieren

Dataset analysieren****
Prüft deinen Datensatz auf:

• Fehlende Dateien** – Bilder die nicht mehr vorhanden sind**
• MD5-Duplikate** – identische Bilder die Training verzerren**
• Klassenungleichgewicht** – ungleiche Bildanzahl pro Klasse**
• Bildstatistiken** – Formate, Größen, Farbmodi

**Bildpfade korrigieren…**

Nach dem Verschieben von Bildern aktualisiert diese Funktion alle Pfade automatisch.

## Annotationen exportieren

| Format | Datei | Verwendung |
|---|---|---|
| COCO JSON | annotations.json | Object-Detection-Frameworks (YOLO v5+, Detectron2) |
| YOLO TXT | bild.txt + classes.txt | Ultralytics / Darknet |
| CSV | annotations.csv | Tabellenkalkulation / eigene Tools |
