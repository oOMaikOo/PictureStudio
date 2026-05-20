# ⚡ Modelle Erweitert

> **PictureStudio v2.3.0** — Hyperparameter-Suche, Kalibrierung, INT8, CoreML und Docker-Deployment

---

# ⚡ Modelle Erweitert

Fortgeschrittene Werkzeuge für Hyperparameter-Suche, Kalibrierung und Edge-Deployment.

## Hyperparameter-Suche (Optuna)

Training-Seite → **⚙ Hyperparameter-Suche…****
Optuna testet Lernrate, Batch-Größe, Architektur und Optimizer (je 5 Epochen).

> ⚠️ `pip install optuna`

## Modell-Vergleich

Mehrere Modelle (Strg+Klick) → Ausgewählte vergleichen****
Sortierbare Tabelle: Accuracy%, F1%, Architektur, Best-Markierung.

## Kalibrierung (Temperature Scaling)

Modell → Kalibrieren (Temperature Scaling)…****
Verbessert die Zuverlässigkeit von Konfidenzwerten.

> ⚠️ `pip install scipy`

## ONNX INT8 Export

Modell → ONNX INT8 exportieren…****
2–4× kleiner und schneller als FP32, ideal für CPU-Inferenz auf Edge-Geräten.

> ⚠️ `pip install onnxruntime`

## CoreML Export (macOS)

Modell → CoreML exportieren…** → `.mlpackage` für Apple Neural Engine.

> ⚠️ `pip install coremltools`

## Docker-Deployment

Modell → **Docker-Deployment generieren…**

Erzeugt: Dockerfile, docker-compose.yml, requirements_monitor.txt, run_monitor.sh, README_deploy.md
