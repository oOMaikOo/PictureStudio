"""Command-line interface: argument parser + main entry point."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Union

from monitor.camera import _discover_cameras, _terminal_camera_select
from monitor.runner import run_monitor, run_monitor_multi
from monitor.setup_server import run_setup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="monitor.py",
        description="PictureStudio Monitor-Client — Standalone Anomalie-Erkennung",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # USB-Kamera (Index aus Modell-Metadaten)
  python monitor.py --model models/anomalie.pth

  # IP-Kamera / RTSP-Stream
  python monitor.py --model models/anomalie.pth \\
      --url rtsp://admin:pass@192.168.1.100:554/stream

  # Video-Datei analysieren (einmalig)
  python monitor.py --model models/anomalie.pth --url /pfad/video.mp4 --headless

  # Headless + MQTT + REST-Dashboard
  python monitor.py --model models/anomalie.pth \\
      --url rtsp://kamera.local/stream \\
      --headless --mqtt-host 192.168.1.50 --api-port 8766

  # ONNX-Modell auf Raspberry Pi (kein PyTorch nötig)
  python monitor.py --model models/anomalie.onnx \\
      --url rtsp://kamera.local/stream \\
      --headless --output /var/log/anomalien

  # Multi-Channel Setup-Wizard (Training in Picture Studio)
  python monitor.py --setup --setup-port 8765

  # Multi-Channel aus gespeicherter Konfiguration
  python monitor.py --channels /pfad/channels.json --headless
        """,
    )

    # Setup-Wizard
    setup_grp = parser.add_argument_group("Einrichtungsmodus (Setup-Wizard, kein Training auf Edge-Gerät)")
    setup_grp.add_argument("--setup", action="store_true",
                           help="Multi-Channel Setup-Wizard starten (kein --model nötig)")
    setup_grp.add_argument("--setup-port", type=int, default=8765, metavar="PORT",
                           help="HTTP-Port für den Setup-Wizard (Standard: 8765)")
    setup_grp.add_argument("--channels", default=None, metavar="PFAD",
                           help="Pfad zu einer JSON-Datei mit gespeicherter Kanal-Konfiguration")

    # Modell
    parser.add_argument("--model", nargs="?", default=None, metavar="PFAD",
                        help="Pfad zur trainierten .pth- oder .onnx-Modelldatei (Pflicht außer mit --setup/--channels)")

    # Kamera
    cam_grp = parser.add_argument_group("Kameraquelle (Standard: aus Modell-Metadaten)")
    src = cam_grp.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, metavar="INDEX",
                     help="USB-Kamera-Index überschreiben (z. B. 0, 1, 2)")
    src.add_argument("--url", metavar="URL",
                     help="IP-Kamera-URL (rtsp://…, http://…) oder Video-Datei-Pfad (mp4, avi, …)")

    # Allgemein
    parser.add_argument("--threshold", type=float, metavar="WERT",
                        help="Anomalie-Schwellwert überschreiben (Standard: aus Modell)")
    parser.add_argument("--output", default="monitor_logs", metavar="VERZ",
                        help="Ausgabeverzeichnis für Logs und Alarm-Bilder (Standard: monitor_logs)")
    parser.add_argument("--fps", type=float, default=15.0, metavar="FPS",
                        help="Bilder pro Sekunde für live Streams (Standard: 15; Videos: nativ)")
    parser.add_argument("--cooldown", type=float, default=10.0, metavar="SEK",
                        help="Mindestabstand zwischen Alarm-Saves in Sekunden (Standard: 10)")
    parser.add_argument("--headless", action="store_true",
                        help="Kein OpenCV-Fenster — nur Terminal und CSV-Log")
    parser.add_argument("--no-notify", action="store_true",
                        help="E-Mail/Webhook-Benachrichtigungen deaktivieren")
    parser.add_argument("--reconnect-delay", type=float, default=5.0, metavar="SEK",
                        help="Sekunden bis zum Reconnect-Versuch bei Verbindungsverlust (0=deaktiviert, Standard: 5)")

    # MQTT
    mqtt_grp = parser.add_argument_group("MQTT (Alarm-Publishing an Broker)")
    mqtt_grp.add_argument("--mqtt-host", default="", metavar="HOST",
                          help="MQTT-Broker-Hostname oder IP")
    mqtt_grp.add_argument("--mqtt-port", type=int, default=1883, metavar="PORT",
                          help="MQTT-Broker-Port (Standard: 1883)")
    mqtt_grp.add_argument("--mqtt-topic", default="picture_studio/monitor", metavar="TOPIC",
                          help="MQTT-Topic (Standard: picture_studio/monitor)")
    mqtt_grp.add_argument("--mqtt-user", default="", metavar="USER",
                          help="MQTT-Benutzername (optional)")
    mqtt_grp.add_argument("--mqtt-pass", default="", metavar="PASS",
                          help="MQTT-Passwort (optional)")

    # REST API
    api_grp = parser.add_argument_group("REST-API + Dashboard (eingebaut)")
    api_grp.add_argument("--api-port", type=int, default=0, metavar="PORT",
                         help="Port für eingebauten REST-Server + HTML-Dashboard (0=deaktiviert)")
    api_grp.add_argument("--api-key", default="", metavar="KEY",
                         help="API-Key für REST-Authentifizierung (optional)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── Setup-Wizard (multi-channel, no training on edge device) ──────────────
    if args.setup:
        if args.url:
            camera_source: Union[int, str, None] = args.url
        elif args.camera is not None:
            camera_source = args.camera
        else:
            camera_source = None  # wizard starts with no channels pre-configured

        channels = run_setup(
            camera_source=camera_source,
            setup_port=args.setup_port,
            output_dir=args.output,
            discovered_cameras=_discover_cameras(),
        )
        if not channels:
            print("Keine Kanäle konfiguriert.")
            sys.exit(1)
        print(f"\nEinrichtung abgeschlossen. Starte Live-Monitoring mit {len(channels)} Kanal/Kanäle...")
        if len(channels) == 1:
            ch = channels[0]
            run_monitor(
                model_path=ch["model_path"],
                camera_source=ch.get("camera_source"),
                output_dir=args.output,
                fps=args.fps,
                cooldown=args.cooldown,
                headless=args.headless,
                api_port=args.api_port,
                api_key=args.api_key,
                reconnect_delay=args.reconnect_delay,
                mqtt_host=args.mqtt_host,
                mqtt_port=args.mqtt_port,
                mqtt_topic=args.mqtt_topic,
                mqtt_user=args.mqtt_user,
                mqtt_pass=args.mqtt_pass,
            )
        else:
            run_monitor_multi(
                channels=channels,
                output_dir=args.output,
                fps=args.fps,
                cooldown=args.cooldown,
                headless=args.headless,
                api_port=args.api_port,
                api_key=args.api_key,
                reconnect_delay=args.reconnect_delay,
                mqtt_host=args.mqtt_host,
                mqtt_port=args.mqtt_port,
                mqtt_topic=args.mqtt_topic,
                mqtt_user=args.mqtt_user,
                mqtt_pass=args.mqtt_pass,
            )
        return

    # ── Channels JSON (pre-configured multi-channel) ───────────────────────────
    if args.channels:
        import json as _json
        try:
            with open(args.channels, encoding="utf-8") as fh:
                channels = _json.load(fh)
        except Exception as exc:
            print(f"Fehler beim Lesen der Kanal-Konfiguration: {exc}")
            sys.exit(1)
        run_monitor_multi(
            channels=channels,
            output_dir=args.output,
            fps=args.fps,
            cooldown=args.cooldown,
            headless=args.headless,
            api_port=args.api_port,
            api_key=args.api_key,
            reconnect_delay=args.reconnect_delay,
            mqtt_host=args.mqtt_host,
            mqtt_port=args.mqtt_port,
            mqtt_topic=args.mqtt_topic,
            mqtt_user=args.mqtt_user,
            mqtt_pass=args.mqtt_pass,
        )
        return

    # ── api_port: default to 8766 for collection/deploy mode ──────────────────
    # If --api-port wasn't specified explicitly but we're in collect mode (no model),
    # default to 8766 so PictureStudio can reach /api/frames and /api/deploy.
    effective_api_port = args.api_port if args.api_port > 0 else (8766 if not args.model else 0)

    # ── No arguments: interactive camera selection → auto setup wizard ─────────
    if not args.model and not args.camera and args.url is None:
        import webbrowser
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║  PictureStudio Monitor — Kamera-Erkennung                   ║")
        print("╚══════════════════════════════════════════════════════════════╝\n")
        print("  Scanne verfügbare Kameras…")
        discovered = _discover_cameras()
        if discovered:
            print(f"  {len(discovered)} Kamera(s) gefunden.")
        else:
            print("  Keine USB-Kameras gefunden (IP-Kameras können im Browser eingegeben werden).")

        # B) Terminal-Auswahl
        selected = _terminal_camera_select(discovered)

        # A) Web-UI starten
        setup_port = args.setup_port
        print(f"\n  Starte Setup-Wizard auf http://localhost:{setup_port}/setup …")
        webbrowser.open(f"http://localhost:{setup_port}/setup")

        # Pre-populate selected cameras as channels
        first_source: Union[int, str, None] = selected[0] if selected else None
        channels = run_setup(
            camera_source=first_source,
            setup_port=setup_port,
            output_dir=args.output,
            discovered_cameras=discovered,
        )

        # Add remaining selected cameras (beyond the first) as extra channels
        # (they can also be added via the web UI, but pre-populate if terminal selection was used)

        if not channels:
            print("Keine Kanäle konfiguriert.")
            sys.exit(1)
        print(f"\nEinrichtung abgeschlossen. Starte Live-Monitoring mit {len(channels)} Kanal/Kanäle…")
        if len(channels) == 1:
            ch = channels[0]
            run_monitor(
                model_path=ch["model_path"],
                camera_source=ch.get("camera_source"),
                output_dir=args.output,
                fps=args.fps,
                cooldown=args.cooldown,
                headless=args.headless,
                api_port=args.api_port,
                api_key=args.api_key,
                reconnect_delay=args.reconnect_delay,
                mqtt_host=args.mqtt_host,
                mqtt_port=args.mqtt_port,
                mqtt_topic=args.mqtt_topic,
                mqtt_user=args.mqtt_user,
                mqtt_pass=args.mqtt_pass,
            )
        else:
            run_monitor_multi(
                channels=channels,
                output_dir=args.output,
                fps=args.fps,
                cooldown=args.cooldown,
                headless=args.headless,
                api_port=args.api_port,
                api_key=args.api_key,
                reconnect_delay=args.reconnect_delay,
                mqtt_host=args.mqtt_host,
                mqtt_port=args.mqtt_port,
                mqtt_topic=args.mqtt_topic,
                mqtt_user=args.mqtt_user,
                mqtt_pass=args.mqtt_pass,
            )
        return
    # ── Collection-only mode: camera specified but no model yet ───────────────
    if not args.model:
        if args.url:
            camera_source: Union[int, str, None] = args.url
        elif args.camera is not None:
            camera_source = args.camera
        else:
            camera_source = 0
        print("\n╔══════════════════════════════════════════════════════════════════╗")
        print("║  PictureStudio Monitor — Sammel-Modus (kein Modell)             ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        print(f"║  Frames API : http://localhost:{effective_api_port}/api/frames          ║")
        print(f"║  Deploy API : http://localhost:{effective_api_port}/api/deploy   (POST) ║")
        print("╚══════════════════════════════════════════════════════════════════╝\n")
        result = run_monitor(
            model_path="",
            camera_source=camera_source,
            output_dir=args.output,
            fps=args.fps,
            headless=args.headless,
            no_notify=True,
            reconnect_delay=args.reconnect_delay,
            api_port=effective_api_port,
            api_key=args.api_key,
        )
        sys.exit(0 if result >= 0 else 1)

    if not os.path.isfile(args.model):
        parser.error(f"Modell-Datei nicht gefunden: {args.model}")

    # Resolve camera source
    if args.url:
        camera_source = args.url
    elif args.camera is not None:
        camera_source = args.camera
    else:
        camera_source = None   # auto-detect from model metadata

    result = run_monitor(
        model_path=args.model,
        camera_source=camera_source,
        threshold_override=args.threshold,
        output_dir=args.output,
        fps=args.fps,
        cooldown=args.cooldown,
        headless=args.headless,
        no_notify=args.no_notify,
        reconnect_delay=args.reconnect_delay,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        mqtt_topic=args.mqtt_topic,
        mqtt_user=args.mqtt_user,
        mqtt_pass=args.mqtt_pass,
        api_port=args.api_port or effective_api_port,
        api_key=args.api_key,
    )
    sys.exit(0 if result >= 0 else 1)
