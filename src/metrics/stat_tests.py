"""Statistical testing helpers for evaluation workflows."""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)

MetricFunc = Callable[[np.ndarray, np.ndarray], float]


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    probas: np.ndarray,
    metrics: Dict[str, MetricFunc],
    n_bootstrap: int = 2000,
    random_state: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap confidence intervals for metrics."""

    rng = np.random.default_rng(random_state)
    n_samples = len(y_true)
    indices = np.arange(n_samples)
    history: Dict[str, list] = {name: [] for name in metrics}
    for _ in range(n_bootstrap):
        sample_idx = rng.choice(indices, size=n_samples, replace=True)
        y_sample = y_true[sample_idx]
        proba_sample = probas[sample_idx]
        for name, fn in metrics.items():
            history[name].append(fn(y_sample, proba_sample))
    ci: Dict[str, Dict[str, float]] = {}
    for name, values in history.items():
        arr = np.asarray(values)
        ci[name] = {
            "lower": float(np.nanpercentile(arr, 2.5)),
            "upper": float(np.nanpercentile(arr, 97.5)),
            "mean": float(np.nanmean(arr)),
        }
    return ci


def mcnemar_test(y_true: np.ndarray, probas_a: np.ndarray, probas_b: np.ndarray) -> Dict[str, float]:
    """Perform McNemar's test on patient-level predictions."""

    preds_a = np.argmax(probas_a, axis=1)
    preds_b = np.argmax(probas_b, axis=1)
    correct_a = preds_a == y_true
    correct_b = preds_b == y_true
    n01 = int(np.sum((~correct_a) & correct_b))
    n10 = int(np.sum(correct_a & (~correct_b)))
    total = n01 + n10
    if total == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False, "n01": n01, "n10": n10}
    statistic = (abs(n01 - n10) - 1) ** 2 / total if total > 0 else 0.0
    # Exact binomial test fallback (two-sided)
    k = min(n01, n10)
    from math import comb

    p = 0.0
    for i in range(k + 1):
        p += comb(total, i)
    p *= 2 ** (1 - total)
    p_value = min(1.0, p * 2)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "n01": n01,
        "n10": n10,
    }


def auc_difference(
    y_true: np.ndarray,
    probas_a: np.ndarray,
    probas_b: np.ndarray,
    n_bootstrap: int = 2000,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """Estimate AUC difference via DeLong or bootstrap."""

    from sklearn.metrics import roc_auc_score

    def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
        scores = []
        for cls in np.unique(y):
            if np.all(y == cls) or np.all(y != cls):
                continue
            scores.append(roc_auc_score((y == cls).astype(int), p[:, cls]))
        return float(np.nanmean(scores)) if scores else float("nan")

    try:
        from delong import delong_roc_test  # type: ignore

        # If available, run DeLong test for OvR macro AUC.
        auc_a = macro_auc(y_true, probas_a)
        auc_b = macro_auc(y_true, probas_b)
        delta = auc_a - auc_b
        stat, p_value = delong_roc_test(y_true, probas_a, probas_b)  # type: ignore[arg-type]
        return {
            "delta": float(delta),
            "method": "delong",
            "p_value": float(p_value),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }
    except Exception:  # pragma: no cover - optional dependency
        LOGGER.info("DeLong implementation unavailable; falling back to bootstrap for AUC difference")

    rng = np.random.default_rng(random_state)
    indices = np.arange(len(y_true))
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(indices, size=len(indices), replace=True)
        auc_a = macro_auc(y_true[idx], probas_a[idx])
        auc_b = macro_auc(y_true[idx], probas_b[idx])
        diffs.append(auc_a - auc_b)
    diffs_arr = np.asarray(diffs)
    return {
        "delta": float(macro_auc(y_true, probas_a) - macro_auc(y_true, probas_b)),
        "method": "bootstrap",
        "p_value": float(_two_sided_p_value(diffs_arr)),
        "ci_lower": float(np.nanpercentile(diffs_arr, 2.5)),
        "ci_upper": float(np.nanpercentile(diffs_arr, 97.5)),
    }


def _two_sided_p_value(diffs: np.ndarray) -> float:
    """Compute an empirical two-sided p-value for centred bootstrap diffs."""

    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        return 1.0
    mean_diff = np.mean(diffs)
    return float(np.mean(np.abs(diffs) >= abs(mean_diff)))
