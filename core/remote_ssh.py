"""
SSH/SFTP helpers for remote GPU training.
"""
import json
import os
import shutil
import tempfile
import zipfile
from typing import Callable, Dict, Optional, Tuple

from utils.logging_utils import get_logger

log = get_logger()


class SSHManager:
    """Thin wrapper around paramiko for SSH + SFTP operations."""

    def __init__(self):
        self._client = None
        self._sftp: Optional[object] = None

    # ---------------------------------------------------------------- connect

    def connect(self, cfg: Dict) -> None:
        try:
            import paramiko
        except ImportError:
            raise RuntimeError("paramiko nicht installiert — pip install paramiko")

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        kw: Dict = {
            "hostname": cfg["host"],
            "port": int(cfg.get("port", 22)),
            "username": cfg["username"],
            "timeout": 20,
        }
        key_path = cfg.get("key_path", "")
        if key_path and os.path.exists(key_path):
            kw["key_filename"] = key_path
        elif cfg.get("password"):
            kw["password"] = cfg["password"]

        client.connect(**kw)
        self._client = client
        self._sftp = client.open_sftp()
        log.info("SSH connected: %s@%s:%s", cfg["username"], cfg["host"], kw["port"])

    def disconnect(self) -> None:
        if self._sftp:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def test_connection(self, cfg: Dict) -> Tuple[bool, str]:
        try:
            self.connect(cfg)
            _, stdout, _ = self._client.exec_command("echo OK", timeout=10)
            result = stdout.read().decode().strip()
            self.disconnect()
            if result == "OK":
                return True, "Verbindung erfolgreich"
            return False, f"Unerwartete Antwort: {result!r}"
        except Exception as exc:
            self.disconnect()
            return False, str(exc)

    # ---------------------------------------------------------------- remote exec

    def exec(self, cmd: str):
        """Return (stdin, stdout, stderr) for streaming."""
        return self._client.exec_command(cmd, get_pty=False)

    def exec_check(self, cmd: str) -> str:
        """Run command, return stdout; raise RuntimeError on non-zero exit."""
        _, stdout, stderr = self._client.exec_command(cmd)
        out = stdout.read().decode()
        rc = stdout.channel.recv_exit_status()
        if rc != 0:
            err = stderr.read().decode().strip()
            raise RuntimeError(f"Remote command failed (rc={rc}): {err or cmd}")
        return out

    # ---------------------------------------------------------------- file ops

    def mkdir_p(self, remote_dir: str) -> None:
        self.exec_check(f"mkdir -p '{remote_dir}'")

    def upload_file(
        self,
        local_path: str,
        remote_path: str,
        progress_cb: Callable[[int, int], None] = None,
    ) -> None:
        self._sftp.put(local_path, remote_path, callback=progress_cb)

    def download_file(
        self,
        remote_path: str,
        local_path: str,
        progress_cb: Callable[[int, int], None] = None,
    ) -> None:
        self._sftp.get(remote_path, local_path, callback=progress_cb)

    def file_exists(self, remote_path: str) -> bool:
        try:
            self._sftp.stat(remote_path)
            return True
        except FileNotFoundError:
            return False


# ------------------------------------------------------------------ bundle builder

def build_training_bundle(
    project,
    train_cfg: Dict,
    run_id: str,
    remote_workdir: str,
) -> Tuple[str, str, str]:
    """
    Zip all labeled images and write a JSON config for remote_train.py.
    Returns (zip_path, config_path, tmp_dir).
    Caller must delete tmp_dir when done.
    """
    tmp = tempfile.mkdtemp(prefix="ils_remote_")
    zip_path = os.path.join(tmp, "images.zip")
    cfg_path = os.path.join(tmp, "config.json")

    labeled = {
        img: lbl
        for img, lbl in project.image_labels.items()
        if lbl and lbl in project.labels
    }
    class_names = sorted(project.labels.keys())

    # Deduplicate filenames: two images with the same basename get a suffix.
    seen: Dict[str, int] = {}
    name_map: Dict[str, str] = {}
    for img_path in labeled:
        base = os.path.basename(img_path)
        if base in seen:
            seen[base] += 1
            stem, ext = os.path.splitext(base)
            base = f"{stem}_{seen[base]}{ext}"
        else:
            seen[base] = 0
        name_map[img_path] = base

    samples = [[name_map[p], lbl] for p, lbl in labeled.items()]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_path, safe_name in name_map.items():
            if os.path.exists(img_path):
                zf.write(img_path, safe_name)

    config = {
        "run_id": run_id,
        "samples": samples,
        "class_names": class_names,
        "image_base_dir": f"{remote_workdir}/images",
        "output_dir": f"{remote_workdir}/output",
        "model_type": train_cfg.get("model_type", "resnet18"),
        "use_pretrained": train_cfg.get("use_pretrained", True),
        "image_size": train_cfg.get("image_size", 224),
        "batch_size": train_cfg.get("batch_size", 16),
        "epochs": train_cfg.get("epochs", 20),
        "learning_rate": train_cfg.get("learning_rate", 0.001),
        "optimizer": train_cfg.get("optimizer", "adam"),
        "scheduler": train_cfg.get("scheduler", "reduce_on_plateau"),
        "early_stopping_patience": train_cfg.get("early_stopping_patience", 0),
        "seed": train_cfg.get("seed", 42),
        "device": train_cfg.get("device", "auto"),
        "mixed_precision": train_cfg.get("mixed_precision", False),
        "train_split": train_cfg.get("train_split", 0.7),
        "val_split": train_cfg.get("val_split", 0.2),
        "augmentation": train_cfg.get("augmentation", {
            "flip": True, "rotation": True, "brightness": True, "scale": False,
        }),
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    log.info(
        "Training bundle: %d images, %d classes, zip=%.1f MB",
        len(samples), len(class_names),
        os.path.getsize(zip_path) / 1024 / 1024,
    )
    return zip_path, cfg_path, tmp
