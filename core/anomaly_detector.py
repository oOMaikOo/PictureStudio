"""
Convolutional autoencoder for unsupervised frame-level anomaly detection.

Train exclusively on normal-process frames; reconstruction error spikes when
the model sees something it has never learned to reconstruct → anomaly.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_IMG = 128  # all frames resized to _IMG × _IMG before encode/decode


def _best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _ConvAutoencoder(nn.Module):
    """Small encoder-decoder: 3×128×128 → bottleneck → 3×128×128."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),   # → 16×64×64
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # → 32×32×32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # → 64×16×16
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # → 32×32×32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # → 16×64×64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),   # → 3×128×128
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def __init__(self):
        self._model = _ConvAutoencoder()
        self._device = _best_device()
        self._trained = False
        self._threshold: float = 0.02
        self._train_frames: list[np.ndarray] = []  # stored as numpy float32 C×H×W

    # ------------------------------------------------------------------ collection

    def collect_frame(self, frame: np.ndarray) -> None:
        """Append a BGR frame to the normal-frame training buffer."""
        self._train_frames.append(self._preprocess(frame).numpy())

    def clear_frames(self) -> None:
        self._train_frames.clear()

    def n_collected(self) -> int:
        return len(self._train_frames)

    # ------------------------------------------------------------------ preprocessing

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (_IMG, _IMG), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)

    # ------------------------------------------------------------------ training

    def train(
        self,
        epochs: int = 20,
        batch_size: int = 16,
        progress_cb=None,
    ) -> float:
        """
        Train on collected frames (blocking).

        progress_cb(epoch: int, total: int, loss: float) is called each epoch.
        Returns the auto-computed threshold (mean + 2.5 × std of training MSEs).
        Raises ValueError if no frames have been collected.
        """
        if not self._train_frames:
            raise ValueError("Keine Frames gesammelt – zuerst Normalframes aufnehmen.")

        data = torch.tensor(np.stack(self._train_frames), dtype=torch.float32)
        loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True, drop_last=False)

        self._model.to(self._device)
        self._model.train()
        opt = torch.optim.Adam(self._model.parameters(), lr=1e-3)
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
        self._model.eval()   # einmalig setzen, nicht bei jedem score()-Aufruf
        self._trained = True
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
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Return (score, reconstruction_bgr, heatmap_overlay_bgr).

        reconstruction_bgr: model's 128×128 reconstruction in BGR.
        heatmap_overlay_bgr: per-pixel error heatmap blended onto original frame.
        Falls back to (0.0, frame copy, frame copy) when not trained.
        """
        if not self._trained:
            fc = frame.copy()
            return 0.0, fc, fc

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

        return score, rec_bgr, overlay

    # ------------------------------------------------------------------ threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = max(1e-6, float(value))

    @property
    def trained(self) -> bool:
        return self._trained

    # ------------------------------------------------------------------ persistence

    def save(self, path: str) -> None:
        torch.save({"model": self._model.state_dict(), "threshold": self._threshold}, path)

    def load(self, path: str) -> None:
        # Immer erst auf CPU laden, dann auf Zielgerät verschieben —
        # vermeidet Cross-Device-Fehler bei load_state_dict.
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(ckpt["model"])
        self._model.to(self._device)
        self._model.eval()
        self._threshold = float(ckpt["threshold"])
        self._trained = True
