"""
Convolutional autoencoder for unsupervised frame-level anomaly detection.

Train exclusively on normal-process frames; reconstruction error spikes when
the model sees something it has never learned to reconstruct → anomaly.
"""
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from datetime import datetime, timezone
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
import time as _time

from utils.logging_utils import get_logger

log = get_logger("ImageLabelingStudio.anomaly_detector")

_IMG = 128  # all frames resized to _IMG × _IMG before encode/decode


def _best_device() -> torch.device:
    """Return the fastest available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _ConvAutoencoder(nn.Module):
    """
    Small convolutional autoencoder: 3×128×128 → latent bottleneck → 3×128×128.

    The architecture uses three stride-2 Conv2d layers to compress the input
    and three ConvTranspose2d layers to reconstruct it. A sigmoid output
    constrains pixel values to [0, 1] for MSE loss computation.

    ``base_ch`` controls the width of all conv layers (default 16 matches the
    original hardcoded values).  Use 8 for a smaller/faster model or 32 for
    greater capacity.
    """

    def __init__(self, base_ch: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode the input tensor."""
        return self.decoder(self.encoder(x))


class AnomalyDetector:
    """
    Wraps the autoencoder with collection, training, scoring and persistence.

    Typical lifecycle:
        det = AnomalyDetector()
        # while camera is running:
        det.collect_frame(frame)   # collect N normal frames
        det.train()                # train (blocking; use AETrainThread in GUI)
        score = det.score(frame)   # per-frame MSE; compare with det.threshold
        det.save(path)
        det.load(path)
    """

    def __init__(self, base_ch: int = 16):
        self._base_ch = base_ch
        self._model = _ConvAutoencoder(base_ch)
        self._device = _best_device()
        self._trained = False
        self._threshold: float = 0.02
        self._train_frames: list[np.ndarray] = []  # stored as numpy float32 C×H×W
        self._metadata: dict = {}

    # ------------------------------------------------------------------ metadata

    @property
    def metadata(self) -> dict:
        """Read-only copy of the model metadata dict."""
        return dict(self._metadata)

    def set_meta(self, key: str, value) -> None:
        """Set an arbitrary metadata field (e.g. camera_source, roi, description)."""
        self._metadata[key] = value

    # ------------------------------------------------------------------ collection

    def collect_frame(self, frame: np.ndarray) -> None:
        """Append a BGR frame to the normal-frame training buffer."""
        self._train_frames.append(self._preprocess(frame).numpy())

    def add_preprocessed_frames(self, frames: list) -> None:
        """Bulk-append already-preprocessed (float32 C×H×W numpy arrays) to the buffer.

        Use this instead of directly accessing _train_frames — e.g. when copying
        frames from one detector to another during hyperparameter search.
        """
        self._train_frames.extend(frames)

    def clear_frames(self) -> None:
        """Discard all buffered normal frames (e.g. to start a new collection)."""
        self._train_frames.clear()

    def n_collected(self) -> int:
        """Return the number of frames currently in the training buffer."""
        return len(self._train_frames)

    # ------------------------------------------------------------------ preprocessing

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Convert BGR frame → normalised float32 tensor of shape (3, _IMG, _IMG)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (_IMG, _IMG), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)

    # ------------------------------------------------------------------ training

    def train(
        self,
        epochs: int = 20,
        batch_size: int = 16,
        lr: float = 1e-3,
        progress_cb=None,
        seed: int = 42,
    ) -> float:
        """
        Train on collected frames (blocking).

        progress_cb(epoch: int, total: int, loss: float) is called each epoch.
        Returns the auto-computed threshold (mean + 2.5 × std of training MSEs).
        Raises ValueError if no frames have been collected.
        """
        if not self._train_frames:
            raise ValueError("Keine Frames gesammelt – zuerst Normalframes aufnehmen.")

        n_frames = len(self._train_frames)
        trained_at = datetime.now(timezone.utc).isoformat()
        t0 = _time.perf_counter()

        # Fix random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        data = torch.tensor(np.stack(self._train_frames), dtype=torch.float32)
        loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True, drop_last=False)

        self._model.to(self._device)
        self._model.train()
        opt = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self._device)
                out = self._model(batch)
                loss = criterion(out, batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            if progress_cb:
                progress_cb(epoch, epochs, epoch_loss / len(loader))

        # Derive threshold from training-set reconstruction errors
        self._model.eval()
        errors: list[float] = []
        with torch.no_grad():
            for (batch,) in DataLoader(TensorDataset(data), batch_size=batch_size):
                batch = batch.to(self._device)
                out = self._model(batch)
                per_sample = ((out - batch) ** 2).mean(dim=(1, 2, 3))
                errors.extend(per_sample.cpu().tolist())

        arr = np.array(errors)
        self._threshold = float(arr.mean() + 2.5 * arr.std())
        self._trained = True

        # Record training provenance — preserves user-set fields (camera_source etc.)
        self._metadata.update({
            "trained_at":      trained_at,
            "n_frames":        n_frames,
            "train_epochs":    epochs,
            "train_duration_s": round(_time.perf_counter() - t0, 1),
            "threshold":       self._threshold,
            "base_ch":         self._base_ch,
            "opencv_version":  str(cv2.__version__),
            "torch_version":   str(torch.__version__),
            "model_img_size":  _IMG,
            "device":          str(self._device),
            "random_seed":     seed,
        })
        return self._threshold

    # ------------------------------------------------------------------ scoring

    def score(self, frame: np.ndarray) -> float:
        """Return per-frame reconstruction MSE (0 if not trained yet)."""
        if not self._trained:
            return 0.0
        t = self._preprocess(frame).unsqueeze(0).to(self._device)
        with torch.no_grad():
            out = self._model(t)
            return float(((out - t) ** 2).mean().item())

    def is_anomaly(self, frame: np.ndarray) -> tuple[float, bool]:
        """Return (score, is_above_threshold)."""
        s = self.score(frame)
        return s, s > self._threshold

    def score_detailed(
        self, frame: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray, Optional[tuple[int, int, int, int]]]:
        """
        Return (score, reconstruction_bgr, heatmap_overlay_bgr, anomaly_bbox).

        reconstruction_bgr : model's 128×128 reconstruction in BGR.
        heatmap_overlay_bgr: per-pixel error heatmap blended onto original frame.
        anomaly_bbox       : (x, y, w, h) of the largest anomalous region in the
                             original frame's coordinate space, or None if no
                             region was detected.
        Falls back to (0.0, frame copy, frame copy, None) when not trained.
        """
        if not self._trained:
            fc = frame.copy()
            return 0.0, fc, fc, None

        t = self._preprocess(frame).unsqueeze(0).to(self._device)
        with torch.no_grad():
            out = self._model(t)

        score = float(((out - t) ** 2).mean().item())

        # Reconstruction → BGR uint8
        rec_np = out.squeeze(0).cpu().permute(1, 2, 0).numpy()
        rec_np = (rec_np * 255).clip(0, 255).astype(np.uint8)
        rec_bgr = cv2.cvtColor(rec_np, cv2.COLOR_RGB2BGR)

        # Heatmap: channel-averaged MSE per pixel (128×128 → original size)
        diff = ((out - t) ** 2).mean(dim=1).squeeze(0).cpu().numpy()
        peak = diff.max()
        diff_u8 = ((diff / peak) * 255).clip(0, 255).astype(np.uint8) if peak > 0 \
                  else np.zeros_like(diff, dtype=np.uint8)
        heatmap = cv2.applyColorMap(diff_u8, cv2.COLORMAP_JET)
        h, w = frame.shape[:2]
        heatmap_full = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        overlay = cv2.addWeighted(frame, 0.55, heatmap_full, 0.45, 0)

        # Bounding box around the hottest anomaly region (top 15% of pixels)
        anomaly_bbox: Optional[tuple[int, int, int, int]] = None
        if peak > 0:
            thresh = float(np.percentile(diff, 85))
            mask = ((diff > thresh) * 255).astype(np.uint8)
            mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                bx, by, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
                anomaly_bbox = (bx, by, bw, bh)
                color = (0, 0, 255) if score > self._threshold else (0, 200, 255)
                cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), color, 2)
                label = "ANOMALIE" if score > self._threshold else "Hotspot"
                cv2.putText(overlay, label, (bx, max(by - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        return score, rec_bgr, overlay, anomaly_bbox

    # ------------------------------------------------------------------ threshold

    @property
    def threshold(self) -> float:
        """MSE value above which a frame is classified as an anomaly."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set a custom threshold; clipped to a minimum of 1e-6 to avoid division by zero."""
        self._threshold = max(1e-6, float(value))

    @property
    def trained(self) -> bool:
        """True after at least one successful call to train()."""
        return self._trained

    # ------------------------------------------------------------------ persistence

    def export_onnx(self, path: str) -> None:
        """Export autoencoder to ONNX (opset 17). Model must be trained."""
        if not self._trained:
            raise RuntimeError("Autoencoder ist noch nicht trainiert.")
        self._model.eval()
        dummy = torch.randn(1, 3, _IMG, _IMG)
        # Patch _add_onnxscript_fn to tolerate a missing `onnx` package.
        # The function only modifies bytes when custom opsets are present (none
        # are used here), so bypassing the onnx import is safe.
        try:
            import torch.onnx._internal.torchscript_exporter.onnx_proto_utils as _pu
            _orig = _pu._add_onnxscript_fn
            def _noop_add(model_bytes, custom_opsets):  # noqa: E306
                try:
                    return _orig(model_bytes, custom_opsets)
                except Exception as exc:
                    log.debug("onnxscript-Patch übersprungen: %s", exc)
                    return model_bytes  # return raw bytes if onnx package absent
            _pu._add_onnxscript_fn = _noop_add
            _patched = True
        except Exception as exc:
            log.debug("ONNX-Interna nicht patchbar: %s", exc)
            _patched = False
        try:
            torch.onnx.export(
                self._model.cpu(), dummy, path,
                dynamo=False,
                export_params=True,
                opset_version=17,
                input_names=["input"],
                output_names=["reconstruction"],
                dynamic_axes={"input": {0: "batch"}, "reconstruction": {0: "batch"}},
            )
        finally:
            if _patched:
                _pu._add_onnxscript_fn = _orig
            self._model.to(self._device)

    def export_onnx_with_meta(self, path: str) -> str:
        """Export ONNX + write .meta.json sidecar with threshold and metadata.
        Returns the onnx_path."""
        self.export_onnx(path)
        meta = {"threshold": self._threshold, "metadata": self._metadata}
        import json as _json
        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            _json.dump(meta, f, indent=2, default=str)
        return path

    def export_torchscript(self, path: str) -> None:
        """Export autoencoder as TorchScript (.pt). Model must be trained."""
        if not self._trained:
            raise RuntimeError("Autoencoder ist noch nicht trainiert.")
        self._model.eval()
        dummy = torch.randn(1, 3, _IMG, _IMG)
        cpu_model = self._model.cpu()
        scripted = torch.jit.trace(cpu_model, dummy)
        scripted.save(path)
        self._model.to(self._device)

    def save(self, path: str) -> None:
        """Persist model weights, threshold, and metadata; also writes a .sha256 sidecar."""
        import hashlib, json as _json
        torch.save({
            "model":     self._model.state_dict(),
            "threshold": self._threshold,
            "metadata":  self._metadata,
            "base_ch":   self._base_ch,
        }, path)
        # Write SHA256 checksum sidecar so integrity can be verified on load
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        checksum_path = path + ".sha256"
        with open(checksum_path, "w", encoding="utf-8") as f:
            _json.dump({"sha256": sha, "file": os.path.basename(path)}, f)

    def verify_checksum(self, path: str) -> tuple[bool, str]:
        """Return (ok, message). ok=True means file matches stored checksum."""
        import hashlib, json as _json
        checksum_path = path + ".sha256"
        if not os.path.exists(checksum_path):
            return True, "Keine Prüfsumme vorhanden (altes Modell)"
        try:
            stored = _json.load(open(checksum_path, encoding="utf-8"))["sha256"]
            actual = hashlib.sha256(open(path, "rb").read()).hexdigest()
            if actual == stored:
                return True, f"SHA256 OK: {actual[:16]}…"
            return False, f"PRÜFSUMME UNGÜLTIG!\nErwartet: {stored[:16]}…\nGefunden:  {actual[:16]}…"
        except Exception as e:
            return False, f"Prüfsummen-Fehler: {e}"

    def load(self, path: str) -> None:
        """
        Load a saved autoencoder checkpoint from *path*.

        Verifies the SHA256 checksum (if present) before loading.
        Raises RuntimeError when the checksum does not match.
        """
        # Verify checksum before loading
        ok, msg = self.verify_checksum(path)
        if not ok:
            raise RuntimeError(f"Integritätsprüfung fehlgeschlagen:\n{msg}")
        # Always load to CPU first to avoid cross-device errors.
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        base_ch = int(ckpt.get("base_ch", 16))
        if base_ch != self._base_ch:
            self._base_ch = base_ch
            self._model = _ConvAutoencoder(base_ch)
        self._model.load_state_dict(ckpt["model"])
        self._model.to(self._device)
        self._model.eval()
        self._threshold = float(ckpt["threshold"])
        self._metadata = dict(ckpt.get("metadata", {}))  # graceful for old files
        self._metadata["loaded_at"] = datetime.now(timezone.utc).isoformat()
        self._metadata["sha256_verified"] = msg
        self._trained = True
