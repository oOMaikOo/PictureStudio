from __future__ import annotations
import json
import numpy as np
from typing import Optional


class TemperatureScaler:
    """Post-hoc Konfidenz-Kalibrierung via Temperature Scaling (Guo et al. 2017)."""

    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Findet optimale Temperatur T via NLL-Minimierung.
        logits: shape (N, C) — rohe Modell-Ausgaben vor Softmax
        labels: shape (N,) — Integer-Klassen-Indizes
        Gibt optimale T zurück und setzt self.temperature.
        """
        from scipy.optimize import minimize_scalar

        def nll(T: float) -> float:
            T = max(T, 1e-6)
            scaled = logits / T
            shifted = scaled - scaled.max(axis=1, keepdims=True)
            log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
            return -float(log_probs[np.arange(len(labels)), labels].mean())

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(result.x)
        return self.temperature

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Wendet Temperatur auf Wahrscheinlichkeiten an via re-softmax der Log-Probs.
        probabilities: shape (N, C) — bereits normalisierte Wahrscheinlichkeiten
        """
        eps = 1e-10
        log_probs = np.log(np.clip(probabilities, eps, 1.0))
        scaled_log = log_probs / self.temperature
        shifted = scaled_log - scaled_log.max(axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    def save(self, path: str) -> None:
        """Speichert {"temperature": float} als JSON."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"temperature": self.temperature}, fh)

    def load(self, path: str) -> None:
        """Lädt temperature aus JSON."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.temperature = float(data["temperature"])

    @property
    def is_fitted(self) -> bool:
        return self.temperature != 1.0
