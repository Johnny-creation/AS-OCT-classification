"""Patient-level aggregation and evaluation metrics."""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from src.utils.io import normalise_probabilities

LOGGER = logging.getLogger(__name__)

Aggregation = Literal["mean", "median"]


@dataclass
class AggregatedPatientResult:
    """Container for patient-level aggregation outputs."""

    patients: pd.DataFrame
    conflicts: int
    conflict_patients: List[str]


def _majority_vote(labels: Iterable[int]) -> Tuple[int, bool]:
    """Return majority label and whether a tie occurred."""

    counter = Counter(labels)
    if not counter:
        raise ValueError("Cannot compute majority vote for empty labels")
    most_common = counter.most_common()
    best_label, best_count = most_common[0]
    ties = [label for label, count in most_common if count == best_count]
    conflict = len(ties) > 1
    chosen = min(ties) if conflict else best_label
    return chosen, conflict


def aggregate_patient_predictions(
    df: pd.DataFrame,
    id_col: str,
    true_col: str,
    proba_cols: Dict[str, List[str]],
    agg: Aggregation = "mean",
) -> Dict[str, AggregatedPatientResult]:
    """Aggregate image-level predictions into patient-level representations."""

    if agg not in {"mean", "median"}:
        raise ValueError(f"Unsupported aggregation: {agg}")
    group = df.groupby(id_col)
    results: Dict[str, AggregatedPatientResult] = {}
    conflicts_total = 0
    conflict_patients: List[str] = []

    patient_labels: Dict[str, int] = {}
    conflict_flags: Dict[str, bool] = {}
    for patient_id, labels in group[true_col]:
        majority, conflict = _majority_vote(labels)
        patient_labels[patient_id] = majority
        conflict_flags[patient_id] = conflict
        if conflict:
            conflicts_total += 1
            conflict_patients.append(str(patient_id))

    if conflicts_total:
        LOGGER.warning(
            "Found %d/%d patients (%.2f%%) with conflicting labels when aggregating",
            conflicts_total,
            len(patient_labels),
            100 * conflicts_total / max(len(patient_labels), 1),
        )

    for prefix, cols in proba_cols.items():
        agg_func = np.mean if agg == "mean" else np.median
        aggregated_rows: List[Dict[str, float]] = []
        for patient_id, sub_df in group:
            probs = sub_df[cols].to_numpy(dtype=float)
            aggregated = agg_func(probs, axis=0)
            aggregated = normalise_probabilities(aggregated.reshape(1, -1))[0]
            aggregated_rows.append({
                id_col: patient_id,
                true_col: patient_labels[patient_id],
                **{f"proba_{i}": aggregated[i] for i in range(probs.shape[1])},
            })
        patient_df = pd.DataFrame(aggregated_rows)
        patient_df = patient_df.sort_values(by=id_col).reset_index(drop=True)
        results[prefix] = AggregatedPatientResult(
            patients=patient_df,
            conflicts=conflicts_total,
            conflict_patients=conflict_patients,
        )
    return results


def compute_classwise_ap(y_true: np.ndarray, probas: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Compute one-vs-rest average precision per class."""

    scores: Dict[str, float] = {}
    for cls in range(num_classes):
        if np.all(y_true == cls) or np.all(y_true != cls):
            scores[f"AP_class_{cls}"] = float("nan")
            continue
        scores[f"AP_class_{cls}"] = average_precision_score((y_true == cls).astype(int), probas[:, cls])
    return scores


def compute_classwise_auc(y_true: np.ndarray, probas: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Compute one-vs-rest ROC-AUC per class."""

    aucs: Dict[str, float] = {}
    for cls in range(num_classes):
        if np.all(y_true == cls) or np.all(y_true != cls):
            aucs[f"ROC_AUC_class_{cls}"] = float("nan")
            continue
        aucs[f"ROC_AUC_class_{cls}"] = roc_auc_score((y_true == cls).astype(int), probas[:, cls])
    return aucs


def evaluate_patient_metrics(
    patient_df: pd.DataFrame,
    true_col: str,
    num_classes: int = 4,
) -> Dict[str, float]:
    """Evaluate patient-level metrics for a probability table."""

    probas = patient_df[[f"proba_{i}" for i in range(num_classes)]].to_numpy(dtype=float)
    preds = np.argmax(probas, axis=1)
    y_true = patient_df[true_col].to_numpy(dtype=int)
    metrics: Dict[str, float] = {}
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, preds)
    metrics["macro_f1"] = f1_score(y_true, preds, average="macro")
    metrics["micro_f1"] = f1_score(y_true, preds, average="micro")

    ap_scores = compute_classwise_ap(y_true, probas, num_classes)
    auc_scores = compute_classwise_auc(y_true, probas, num_classes)
    metrics.update(ap_scores)
    metrics.update(auc_scores)

    metrics["macro_ap"] = float(np.nanmean(np.array(list(ap_scores.values()), dtype=float)))
    metrics["macro_roc_auc"] = float(np.nanmean(np.array(list(auc_scores.values()), dtype=float)))

    cm = confusion_matrix(y_true, preds, labels=list(range(num_classes)))
    metrics["confusion_matrix"] = cm
    return metrics


def build_metrics_dataframe(metrics_by_model: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Convert nested metric dict into a tidy DataFrame."""

    rows: List[Dict[str, float]] = []
    for model, metrics in metrics_by_model.items():
        row = {"model": model}
        for key, value in metrics.items():
            if key == "confusion_matrix":
                continue
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)
