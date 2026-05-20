from __future__ import annotations
import os
from typing import Optional


class DockerGenerator:
    """Erstellt deployment-fertige Docker-Dateien für monitor.py."""

    _REQUIREMENTS = """\
opencv-python-headless>=4.8.0
numpy>=1.24.0
onnxruntime>=1.16.0
paho-mqtt>=1.6.0
"""

    def generate(
        self,
        output_dir: str,
        model_path: str = "",
        api_port: int = 8765,
        camera_idx: int = 0,
        threshold: float = 0.002,
        mqtt_host: str = "",
    ) -> list[str]:
        """
        Erstellt Deployment-Dateien in output_dir.
        Gibt Liste der erstellten Dateipfade zurück.
        """
        os.makedirs(output_dir, exist_ok=True)
        model_name = os.path.basename(model_path) if model_path else "model.onnx"

        created = []
        created.append(self._write_dockerfile(output_dir, model_name, api_port))
        created.append(self._write_compose(output_dir, api_port))
        created.append(self._write_requirements(output_dir))
        created.append(self._write_start_script(output_dir, model_name, api_port, camera_idx, threshold, mqtt_host))
        created.append(self._write_readme(output_dir, model_name, api_port))
        return created

    def _write_dockerfile(self, out: str, model_name: str, api_port: int) -> str:
        content = f"""\
FROM python:3.11-slim
LABEL maintainer="PictureStudio"

WORKDIR /app

COPY requirements_monitor.txt .
RUN pip install --no-cache-dir -r requirements_monitor.txt

COPY monitor.py .
COPY models/ models/

EXPOSE {api_port}

CMD ["python", "monitor.py", \\
     "--model", "models/{model_name}", \\
     "--camera", "0", \\
     "--api-port", "{api_port}", \\
     "--headless"]
"""
        path = os.path.join(out, "Dockerfile")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def _write_compose(self, out: str, api_port: int) -> str:
        content = f"""\
version: "3.8"

services:
  picture-monitor:
    build: .
    ports:
      - "{api_port}:{api_port}"
    volumes:
      - ./models:/app/models:ro
      - ./monitor_logs:/app/monitor_logs
    restart: unless-stopped
    environment:
      - PYTHONUNBUFFERED=1
"""
        path = os.path.join(out, "docker-compose.yml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def _write_requirements(self, out: str) -> str:
        path = os.path.join(out, "requirements_monitor.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._REQUIREMENTS)
        return path

    def _write_start_script(
        self, out: str, model_name: str, api_port: int,
        camera_idx: int, threshold: float, mqtt_host: str,
    ) -> str:
        mqtt_args = f" --mqtt-host {mqtt_host}" if mqtt_host else ""
        content = f"""\
#!/usr/bin/env bash
# PictureStudio Monitor – Startskript
set -euo pipefail

docker compose up --build -d
echo "Monitor gestartet auf Port {api_port}"
echo "Dashboard: http://localhost:{api_port}/dashboard"
"""
        path = os.path.join(out, "run_monitor.sh")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        try:
            os.chmod(path, 0o755)
        except Exception:
            pass
        return path

    def _write_readme(self, out: str, model_name: str, api_port: int) -> str:
        content = f"""\
# PictureStudio Monitor – Deployment

## Schnellstart

1. **Modell kopieren**
   ```bash
   mkdir -p models
   cp /pfad/zum/modell/{model_name} models/
   ```

2. **Container starten**
   ```bash
   bash run_monitor.sh
   ```
   Oder manuell:
   ```bash
   docker compose up --build
   ```

3. **Dashboard öffnen**
   http://localhost:{api_port}/dashboard

## REST API

| Endpoint | Beschreibung |
|----------|-------------|
| `GET /api/status` | Server-Status + letzter Score |
| `GET /api/scores` | Score-Verlauf (bis 500) |
| `GET /api/latest_alarm` | Letzter Alarm-Event |
| `GET /dashboard` | Live-Monitoring HTML-Dashboard |

## Anforderungen

- Docker + Docker Compose
- Port {api_port} muss erreichbar sein
"""
        path = os.path.join(out, "README_deploy.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
