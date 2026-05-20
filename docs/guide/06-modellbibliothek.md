# 📊 Modellbibliothek

> **PictureStudio v2.3.0** — Trainierte Modelle verwalten, vergleichen und exportieren

---

# 📊 Modellbibliothek

Alle trainierten Modelle verwalten, vergleichen und einsetzen.

## Modell auswählen

**In Inferenz laden****
Modell in der Tabelle anwählen → *In Inferenz laden*.

Die Anwendung wechselt automatisch zur Klassifikations-Seite.

> 💡 Welches Modell ist das beste?**** Bei gleichmäßigen Klassen: Accuracy-Spalte vergleichen.
 Bei ungleichen Klassen: F1-Score ist aussagekräftiger.

## Exportieren

Als ONNX exportieren (.onnx)****
Einsatz ohne PyTorch in: ONNX Runtime, OpenCV DNN, TensorRT, C++, C#, Edge-Geräten.

Als TorchScript exportieren (.pt)****
Einsatz in der PyTorch C++ API oder mobilen Apps (Android/iOS).

## Modelle vergleichen

Ausgewählte vergleichen****
Mehrere Modelle mit `Strg+Klick` auswählen → *Ausgewählte vergleichen*

Zeigt Accuracy, F1, Architektur und Best-Markierung nebeneinander.

Run-History-Tab:** alle Läufe nach Datum sortiert mit Gerät, Epochen, Train-Acc.
