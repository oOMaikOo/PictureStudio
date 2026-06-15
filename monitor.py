#!/usr/bin/env python3
"""
PictureStudio Monitor-Client — Standalone Anomalie-Erkennung (Entry-Point).

Die Implementierung liegt im ``monitor/``-Package (siehe ``monitor/__init__.py``).
Diese Datei ist nur der ausführbare Einstieg, damit ``python monitor.py …``
unverändert funktioniert.

Verwendung:
    python monitor.py                         # interaktive Kamera-Auswahl + Browser-Setup
    python monitor.py --model anomalie.pth    # direkt starten mit bestehendem Modell
    python monitor.py --setup                 # Multi-Channel Setup-Wizard
    python monitor.py --channels cfg.json      # vorkonfigurierte Kanäle
"""
import os
import sys

# Project root on path so ``monitor`` package + ``core`` imports work anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monitor.cli import main

if __name__ == "__main__":
    main()
