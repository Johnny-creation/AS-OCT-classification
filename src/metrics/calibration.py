"""Calibration metrics, reliability diagrams, and temperature scaling."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.io import normalise_probabilities

LOGGER = logging.getLogger(__name__)


def brier_score(y_true: np.ndarray, probas: np.ndarray) -> float:
    """Compute the multiclass Brier score."""

    y_onehot = np.eye(probas.shape[1])[y_true]
    return float(np.mean(np.sum((y_onehot - probas) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray,
    probas: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Calculate expected calibration error using confidence bins."""

    preds = np.argmax(probas, axis=1)
    confidences = probas[np.arange(len(probas)), preds]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1] if i < n_bins - 1 else confidences <= bins[i + 1])
        if not np.any(mask):
            continue
        accuracy = np.mean(preds[mask] == y_true[mask])
        confidence = np.mean(confidences[mask])
        ece += np.abs(accuracy - confidence) * np.sum(mask) / len(probas)
    return float(ece)


def adaptive_calibration_error(
    y_true: np.ndarray,
    probas: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute a simple adaptive calibration error across classes."""

    n_classes = probas.shape[1]
    ace = 0.0
    for cls in range(n_classes):
        confidences = probas[:, cls]
        truth = (y_true == cls).astype(int)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        class_error = 0.0
        for i in range(n_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1] if i < n_bins - 1 else confidences <= bins[i + 1])
            if not np.any(mask):
                continue
            acc = np.mean(truth[mask])
            conf = np.mean(confidences[mask])
            class_error += np.abs(acc - conf) * np.sum(mask) / len(probas)
        ace += class_error
    return float(ace / n_classes)


def reliability_diagram(
    y_true: np.ndarray,
    probas: np.ndarray,
    out_path: Path,
    n_bins: int = 15,
    title: str = "Reliability diagram",
) -> None:
    """Plot reliability diagram with approximate confidence bands."""

    preds = np.argmax(probas, axis=1)
    confidences = probas[np.arange(len(probas)), preds]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    accuracies = []
    mean_confidences = []
    counts = []
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (
            confidences < bins[i + 1] if i < n_bins - 1 else confidences <= bins[i + 1]
        )
        counts.append(np.sum(mask))
        if not np.any(mask):
            accuracies.append(np.nan)
            mean_confidences.append(np.nan)
            continue
        accuracies.append(np.mean(preds[mask] == y_true[mask]))
        mean_confidences.append(np.mean(confidences[mask]))
    accuracies_arr = np.asarray(accuracies)
    mean_confidences_arr = np.asarray(mean_confidences)
    counts_arr = np.asarray(counts)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    plt.plot(mean_confidences_arr, accuracies_arr, marker="o", label="Model")
    valid = (~np.isnan(accuracies_arr)) & (counts_arr > 0)
    if np.any(valid):
        acc_valid = accuracies_arr[valid]
        se = np.sqrt(np.clip(acc_valid * (1 - acc_valid), 0.0, None) / counts_arr[valid])
        lower = np.clip(acc_valid - 1.96 * se, 0.0, 1.0)
        upper = np.clip(acc_valid + 1.96 * se, 0.0, 1.0)
        plt.fill_between(
            mean_confidences_arr[valid],
            lower,
            upper,
            color="lightgray",
            alpha=0.4,
            label="Approx. 95% CI",
        )
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


@dataclass
class TemperatureScalingResult:
    temperature: float


class TemperatureScaler:
    """Single-parameter temperature scaling for multiclass logits."""

    def __init__(self, init_temp: float = 1.0, max_iter: int = 50, lr: float = 0.01):
        if init_temp <= 0:
            raise ValueError("Initial temperature must be > 0")
        self.temperature = init_temp
        self.max_iter = max_iter
        self.lr = lr

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> TemperatureScalingResult:
        """Fit the temperature parameter by minimising negative log likelihood."""

        if logits.ndim != 2:
            raise ValueError("Logits must be a 2D array")
        if len(logits) != len(labels):
            raise ValueError("Logits and labels must have the same length")
        device = torch.device("cpu")
        logits_tensor = torch.tensor(logits, dtype=torch.float32, device=device)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
        temperature = torch.nn.Parameter(torch.tensor([self.temperature], dtype=torch.float32, device=device))
        optimizer = torch.optim.LBFGS([temperature], lr=self.lr, max_iter=self.max_iter)
        nll = torch.nn.CrossEntropyLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = logits_tensor / temperature.clamp(min=1e-6)
            loss = nll(scaled, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(temperature.detach().cpu().clamp(min=1e-6).item())
        LOGGER.info("Fitted temperature: %.4f", self.temperature)
        return TemperatureScalingResult(temperature=self.temperature)

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply the learnt temperature to logits and return probabilities."""

        if self.temperature <= 0:
            raise ValueError("Temperature must be positive before transform")
        scaled = logits / self.temperature
        return normalise_probabilities(torch.softmax(torch.tensor(scaled, dtype=torch.float32), dim=1).numpy())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Convenience helper to scale logits using a fixed temperature."""

    scaler = TemperatureScaler(init_temp=temperature)
    scaler.temperature = temperature
    return scaler.transform(logits)
