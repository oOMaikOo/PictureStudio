"""
Standalone-Test: Anomalie-Erkennung mit echter Kamera (ohne GUI).

Ablauf:
  Phase 1 – Normalframes sammeln (Kamera auf normalen Prozess richten)
  Phase 2 – Autoencoder trainieren
  Phase 3 – Live-Scoring (Kamera ggf. auf Anomalie richten)

Starten:
  python scripts/test_anomaly_camera.py [--camera 0] [--frames 80] [--epochs 40]
"""
import sys
import os
import time
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.anomaly_detector import AnomalyDetector


# ── ANSI Farben für Terminal-Output ──────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

BAR_WIDTH = 40


def progress_bar(value: float, maximum: float, width: int = BAR_WIDTH) -> str:
    filled = int(width * min(value, maximum) / maximum)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {value:.0f}/{maximum:.0f}"


def score_bar(score: float, threshold: float, width: int = BAR_WIDTH) -> str:
    ratio = min(score / threshold, 3.0) / 3.0  # clamp at 3× threshold
    filled = int(width * ratio)
    color = RED if score > threshold else GREEN
    bar = "█" * filled + "░" * (width - filled)
    pct = score / threshold * 100
    return f"{color}[{bar}]{RESET} {score:.5f}  ({pct:.0f}% von Schwellwert {threshold:.5f})"


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"{RED}Fehler: Kamera {index} konnte nicht geöffnet werden.{RESET}")
        sys.exit(1)
    try:
        # Warm-up: erste Frames verwerfen
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        if not ret:
            print(f"{RED}Fehler: Kein Frame von Kamera {index}.{RESET}")
            cap.release()
            sys.exit(1)
    except Exception as exc:
        cap.release()
        print(f"{RED}Fehler beim Kamera-Warm-up: {exc}{RESET}")
        sys.exit(1)
    h, w = frame.shape[:2]
    print(f"{GREEN}✓ Kamera {index} geöffnet: {w}×{h} px{RESET}")
    return cap


# ── Phase 1: Normalframes sammeln ─────────────────────────────────────────────

def collect_normal_frames(cap: cv2.VideoCapture, detector: AnomalyDetector,
                           n: int) -> None:
    print(f"\n{BOLD}{CYAN}── Phase 1: Normalframes sammeln ────────────────────────{RESET}")
    print(f"  Kamera auf den {BOLD}normalen Prozess{RESET} richten.")
    print(f"  {n} Frames werden automatisch aufgenommen …\n")
    time.sleep(2)  # Nutzer Zeit geben, Kamera auszurichten

    t_start = time.time()
    for i in range(n):
        ret, frame = cap.read()
        if not ret:
            print(f"{RED}Kein Frame – Kamera getrennt?{RESET}")
            break
        detector.collect_frame(frame)
        print(f"\r  {progress_bar(i + 1, n)}  ", end="", flush=True)
        time.sleep(0.05)  # ~20 fps Aufnahmerate

    elapsed = time.time() - t_start
    print(f"\n\n  {GREEN}✓ {detector.n_collected()} Frames gesammelt in {elapsed:.1f}s{RESET}")


# ── Phase 2: Training ─────────────────────────────────────────────────────────

def train(detector: AnomalyDetector, epochs: int) -> float:
    print(f"\n{BOLD}{CYAN}── Phase 2: Autoencoder trainieren ──────────────────────{RESET}")
    print(f"  {epochs} Epochen auf {detector.n_collected()} Normalframes …\n")

    epoch_times = []

    def cb(epoch, total, loss):
        t = time.time()
        epoch_times.append(t)
        eta = ""
        if len(epoch_times) >= 2:
            avg = (epoch_times[-1] - epoch_times[0]) / (len(epoch_times) - 1)
            remaining = avg * (total - epoch)
            eta = f"  ETA {remaining:.0f}s"
        print(f"\r  Epoche {epoch:3d}/{total}  Loss: {loss:.6f}{eta}   ", end="", flush=True)

    t0 = time.time()
    threshold = detector.train(epochs=epochs, progress_cb=cb)
    elapsed = time.time() - t0

    print(f"\n\n  {GREEN}✓ Training abgeschlossen in {elapsed:.1f}s{RESET}")
    print(f"  Auto-Schwellwert: {BOLD}{threshold:.5f}{RESET}  "
          f"(Mittelwert + 2,5 × Std-Abw. der Trainingsfehler)")
    return threshold


# ── Phase 3: Live-Scoring ─────────────────────────────────────────────────────

def live_score(cap: cv2.VideoCapture, detector: AnomalyDetector,
               duration: int = 30) -> None:
    print(f"\n{BOLD}{CYAN}── Phase 3: Live-Scoring ({duration}s) ──────────────────────{RESET}")
    print(f"  Jetzt {BOLD}Anomalie einführen{RESET} (fremdes Objekt, andere Farbe, Abdeckung …)")
    print(f"  Drücke {BOLD}Strg+C{RESET} zum vorzeitigen Beenden.\n")
    time.sleep(1)

    history: list[tuple[float, bool]] = []
    t_end = time.time() + duration
    frame_count = 0

    try:
        while time.time() < t_end:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 3 != 0:   # jeder 3. Frame = ~7 fps bei 20 fps Kamera
                continue

            score, is_anomaly = detector.is_anomaly(frame)
            history.append((score, is_anomaly))

            remaining = max(0, t_end - time.time())
            status = f"{RED}{BOLD}⚠ ANOMALIE{RESET}" if is_anomaly else f"{GREEN}✓ normal{RESET}"
            bar = score_bar(score, detector.threshold)
            print(f"\r  {status}  {bar}  [{remaining:.0f}s]  ", end="", flush=True)
            time.sleep(0.05)

    except KeyboardInterrupt:
        pass

    print("\n")
    _print_summary(history, detector.threshold)


def _print_summary(history: list[tuple[float, bool]], threshold: float) -> None:
    if not history:
        print("Keine Frames bewertet.")
        return

    scores = [s for s, _ in history]
    alarms = sum(1 for _, a in history if a)

    print(f"{BOLD}{CYAN}── Zusammenfassung ──────────────────────────────────────{RESET}")
    print(f"  Frames bewertet : {len(history)}")
    print(f"  Alarme          : {RED}{alarms}{RESET} ({alarms/len(history)*100:.1f}%)")
    print(f"  Score min/Ø/max : "
          f"{min(scores):.5f} / {np.mean(scores):.5f} / {max(scores):.5f}")
    print(f"  Schwellwert     : {threshold:.5f}")

    # Einfaches Histogramm
    buckets = 8
    hist, edges = np.histogram(scores, bins=buckets)
    peak = max(hist)
    print(f"\n  Score-Verteilung:")
    for count, edge in zip(hist, edges):
        bar = "█" * int(40 * count / peak) if peak else ""
        marker = " ← Schwellwert" if edges[0] < threshold <= edge else ""
        print(f"    {edge:.5f}  {bar}{marker}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Anomalie-Erkennung mit echter Kamera")
    parser.add_argument("--camera",  type=int, default=0,  help="Kamera-Index (Standard: 0)")
    parser.add_argument("--frames",  type=int, default=80, help="Normalframes sammeln (Standard: 80)")
    parser.add_argument("--epochs",  type=int, default=40, help="Trainingsepochen (Standard: 40)")
    parser.add_argument("--score-duration", type=int, default=30,
                        help="Live-Scoring Dauer in Sekunden (Standard: 30)")
    args = parser.parse_args()

    print(f"\n{BOLD}Picture Studio – Anomalie-Erkennungstest{RESET}")
    print(f"Kamera {args.camera}  |  {args.frames} Normalframes  |  "
          f"{args.epochs} Epochen  |  {args.score_duration}s Scoring\n")

    cap = open_camera(args.camera)
    detector = AnomalyDetector()

    try:
        collect_normal_frames(cap, detector, args.frames)
        train(detector, args.epochs)
        live_score(cap, detector, args.score_duration)
    finally:
        cap.release()
        print(f"{GREEN}Kamera freigegeben.{RESET}\n")


if __name__ == "__main__":
    main()
