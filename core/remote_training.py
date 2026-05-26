"""
QThread that drives the full SSH remote training pipeline:
  connect → zip/upload images → upload script+config → run → stream logs → download checkpoint
Emits the same signals as TrainingThread so TrainingPage can use it as a drop-in.
"""
import os
import re
import shutil
from datetime import datetime
from typing import Dict, Optional

from PySide6.QtCore import QThread, Signal

from utils.logging_utils import get_logger

log = get_logger()

_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "remote_train.py",
)


class RemoteTrainingThread(QThread):
    """
    QThread that runs the full SSH remote training pipeline.

    Mirrors the TrainingThread signal interface so TrainingPage can use either
    thread without changing its signal connections.

    Pipeline steps:
      1. Connect to SSH host.
      2. Build and upload the training bundle (images.zip + config.json).
      3. Upload remote_train.py script.
      4. Execute the script; stream and parse structured log lines.
      5. Download the best checkpoint.
      6. Emit finished(result_dict).
    """

    progress = Signal(int, int, float, float, float, float)   # epoch, total, tl, vl, ta, va
    log_msg  = Signal(str)
    finished = Signal(dict)
    error    = Signal(str)

    def __init__(self, project, cfg: Dict, save_dir: str, ssh_cfg: Dict):
        super().__init__()
        self.project  = project
        self.cfg      = cfg
        self.save_dir = save_dir
        self.ssh_cfg  = ssh_cfg
        self._stop    = False

    def request_stop(self) -> None:
        """Signal the thread to abort after the current remote operation."""
        self._stop = True

    # ------------------------------------------------------------------ run

    def run(self) -> None:
        """Thread entry point; delegates to _pipeline() and emits error on exception."""
        from core.remote_ssh import SSHManager
        ssh = SSHManager()
        try:
            self._pipeline(ssh)
        except Exception as exc:
            log.exception("RemoteTrainingThread failed")
            self.error.emit(str(exc))
        finally:
            ssh.disconnect()

    def _pipeline(self, ssh) -> None:
        """Execute the nine-step remote training pipeline with *ssh* already injected."""
        from core.remote_ssh import SSHManager, build_training_bundle

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        remote_base = self.ssh_cfg.get("remote_path", "/tmp/ils_project")
        remote_wd   = f"{remote_base}/{run_id}"
        python_env  = self.ssh_cfg.get("python_env", "python3")
        if not re.fullmatch(r"[a-zA-Z0-9/_.-]+", python_env):
            raise ValueError(f"Ungültige python_env-Konfiguration: {python_env!r}")
        tmp_dir: Optional[str] = None

        # 1. Connect
        self._log("Verbinde SSH …")
        ssh.connect(self.ssh_cfg)
        self._log(f"Verbunden mit {self.ssh_cfg.get('host')}")

        # 2. Build bundle
        self._log("Erstelle lokales Trainings-Bundle …")
        zip_path, cfg_path, tmp_dir = build_training_bundle(
            self.project, self.cfg, run_id, remote_wd
        )
        zip_mb = os.path.getsize(zip_path) / 1024 / 1024
        self._log(
            f"Bundle: {len(self.project.image_labels)} Bilder gepackt"
            f" ({zip_mb:.1f} MB)"
        )

        try:
            # 3. Create remote dirs
            self._log(f"Remote-Verzeichnis: {remote_wd}")
            ssh.mkdir_p(f"{remote_wd}/images")
            ssh.mkdir_p(f"{remote_wd}/output")

            # 4. Upload training script
            remote_script = f"{remote_wd}/remote_train.py"
            self._log("Lade Trainings-Skript hoch …")
            ssh.upload_file(_SCRIPT_PATH, remote_script)

            # 5. Upload config
            remote_cfg = f"{remote_wd}/config.json"
            self._log("Lade Konfiguration hoch …")
            ssh.upload_file(cfg_path, remote_cfg)

            # 6. Upload images zip
            remote_zip = f"{remote_wd}/images.zip"
            self._log(f"Lade Bilder hoch ({zip_mb:.1f} MB) …")
            ssh.upload_file(zip_path, remote_zip)
            self._log("Entpacke Bilder auf Remote …")
            ssh.exec_check(
                f"unzip -q -o '{remote_zip}' -d '{remote_wd}/images'"
            )
            self._log("Bilder hochgeladen und entpackt.")

            # 7. Run remote training
            cmd = f"{python_env} '{remote_script}' '{remote_cfg}'"
            self._log(f"Starte: {cmd}")
            _, stdout, stderr = ssh.exec(cmd)

            best_remote: Optional[str] = None
            done_meta: Dict = {}

            for raw in iter(stdout.readline, ""):
                if self._stop:
                    self._log("Training durch Benutzer gestoppt.")
                    break
                line = raw.rstrip("\n\r")
                if not line:
                    continue

                if line.startswith("LOG "):
                    self._log(line[4:])

                elif line.startswith("PROGRESS "):
                    kv = _kv(line[9:])
                    epoch = int(kv.get("epoch", 0))
                    total = int(kv.get("total", self.cfg.get("epochs", 1)))
                    tl    = float(kv.get("train_loss", 0))
                    vl    = float(kv.get("val_loss", 0))
                    ta    = float(kv.get("train_acc", 0))
                    va    = float(kv.get("val_acc", 0))
                    self.progress.emit(epoch, total, tl, vl, ta, va)
                    self._log(
                        f"Epoch {epoch}/{total}  "
                        f"loss={tl:.4f}/{vl:.4f}  "
                        f"acc={ta*100:.1f}%/{va*100:.1f}%"
                    )

                elif line.startswith("BEST "):
                    kv = _kv(line[5:])
                    best_remote = kv.get("best_model", "")
                    self._log(
                        f"  ✓ Bestes Modell: val_acc={float(kv.get('val_acc', 0)):.4f}"
                    )

                elif line.startswith("DONE "):
                    kv = _kv(line[5:])
                    done_meta = {k: kv[k] for k in kv}
                    best_remote = kv.get("best_model", best_remote)
                    self._log("Training abgeschlossen auf Remote-Server.")

                elif line.startswith("ERROR "):
                    raise RuntimeError(f"Remote-Fehler: {line[6:]}")

                else:
                    self._log(line)

            rc = stdout.channel.recv_exit_status()
            if rc != 0 and not self._stop:
                err_txt = stderr.read().decode().strip()
                raise RuntimeError(
                    f"Remote-Prozess endete mit Code {rc}.\n{err_txt}"
                )

            if self._stop:
                return

            if not best_remote:
                raise RuntimeError(
                    "Remote-Skript hat keinen Modellpfad zurückgegeben."
                )

            # 8. Download checkpoint
            os.makedirs(self.save_dir, exist_ok=True)
            local_ckpt = os.path.join(self.save_dir, f"remote_{run_id}_best.pth")
            self._log(f"Lade Checkpoint herunter …")
            ssh.download_file(best_remote, local_ckpt)
            self._log(f"Modell gespeichert: {local_ckpt}")

            # 9. Emit finished (compatible with local TrainingThread result)
            class_names = sorted(self.project.labels.keys())
            result: Dict = {
                "run_id": done_meta.get("run_id", run_id),
                "best_model_path": local_ckpt,
                "class_names": class_names,
                "metrics": {
                    "accuracy": float(done_meta.get("accuracy", done_meta.get("val_acc", 0))),
                    "macro_f1":  float(done_meta.get("f1", 0)),
                },
                "remote": True,
                "remote_host": self.ssh_cfg.get("host", ""),
            }
            self.finished.emit(result)

        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _log(self, msg: str) -> None:
        """Emit a log message both to the GUI (via signal) and the Python logger."""
        self.log_msg.emit(msg)
        log.info("[remote] %s", msg)


def _kv(text: str) -> Dict:
    """Parse 'k1=v1 k2=v2 …' → dict (values may contain path separators)."""
    result: Dict = {}
    for m in re.finditer(r"(\w+)=([^\s=]+)", text):
        result[m.group(1)] = m.group(2)
    return result
