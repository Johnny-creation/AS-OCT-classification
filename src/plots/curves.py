"""Plotting helpers for evaluation figures."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, average_precision_score


def _timestamp() -> str:
    """Return a filesystem-friendly timestamp string."""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> None:
    """Create directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def plot_precision_recall(
    y_true: np.ndarray,
    probas: np.ndarray,
    class_names: Iterable[str],
    out_dir: Path,
    prefix: str,
) -> Path:
    """Plot per-class precision-recall curves and save to disk."""

    _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(6, 6))
    y_true_int = y_true.astype(int)
    num_classes = probas.shape[1]
    ap_scores = []
    for cls, name in enumerate(class_names):
        y_binary = (y_true_int == cls).astype(int)
        PrecisionRecallDisplay.from_predictions(
            y_binary,
            probas[:, cls],
            ax=ax,
            name=name,
        )
        ap_scores.append(average_precision_score(y_binary, probas[:, cls]))
    avg_precision = float(np.nanmean(ap_scores)) if ap_scores else float("nan")
    ax.set_title(f"PR curves (macro AP={avg_precision:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig_path = out_dir / f"{prefix}_pr_{_timestamp()}.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_roc(
    y_true: np.ndarray,
    probas: np.ndarray,
    class_names: Iterable[str],
    out_dir: Path,
    prefix: str,
) -> Path:
    """Plot one-vs-rest ROC curves for each class and save to disk."""

    _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(6, 6))
    y_true_int = y_true.astype(int)
    for cls, name in enumerate(class_names):
        if np.all(y_true_int == cls) or np.all(y_true_int != cls):
            continue
        RocCurveDisplay.from_predictions(
            (y_true_int == cls).astype(int),
            probas[:, cls],
            ax=ax,
            name=name,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC curves (OvR)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()
    fig_path = out_dir / f"{prefix}_roc_{_timestamp()}.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Iterable[str],
    out_dir: Path,
    prefix: str,
) -> Path:
    """Plot and save a confusion matrix heatmap."""

    _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="viridis")
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > cm.max() / 2 else "black",
            )
    fig.tight_layout()
    fig_path = out_dir / f"{prefix}_cm_{_timestamp()}.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    return fig_path
