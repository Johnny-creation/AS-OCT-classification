"""Patient-level evaluation command-line interface."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.metrics.calibration import adaptive_calibration_error, brier_score, expected_calibration_error, reliability_diagram
from src.metrics.patient_level import aggregate_patient_predictions, build_metrics_dataframe, evaluate_patient_metrics
from src.metrics.stat_tests import bootstrap_confidence_intervals
from src.plots.curves import plot_confusion_matrix, plot_precision_recall, plot_roc
from src.utils.io import (
    ensure_probability_columns,
    infer_prefixes,
    load_prediction_tables,
    save_dataframe,
)

LOGGER = logging.getLogger("eval_patient_level")

CLASS_NAMES = ["Normal", "Cataract", "PACG", "PACG+Cataract"]


def _timestamp() -> str:
    """Return a filesystem friendly timestamp."""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_name(prefix: str) -> str:
    """Normalise a prefix for filesystem usage."""

    name = prefix or "model"
    name = name.replace(os.sep, "_")
    return name.replace(" ", "_")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for patient-level evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate predictions at patient level")
    parser.add_argument("--input", required=True, nargs="+", help="Prediction file(s) or directory")
    parser.add_argument("--id-col", required=True, help="Column containing patient identifiers")
    parser.add_argument("--true-col", required=True, help="Column containing true labels")
    parser.add_argument(
        "--proba-prefixes",
        default=None,
        help="Comma separated model prefixes (if omitted try auto-detect)",
    )
    parser.add_argument("--agg", choices=["mean", "median"], default="mean", help="Aggregation strategy")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Number of bootstrap iterations")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for bootstrap")
    parser.add_argument("--outdir", required=True, help="Directory to store reports")
    parser.add_argument("--n-bins", type=int, default=15, help="Number of bins for calibration metrics")
    return parser.parse_args()


def metric_functions(num_classes: int) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    """Construct patient-level metric functions for bootstrap estimation."""

    from sklearn.metrics import balanced_accuracy_score, f1_score
    from src.metrics.patient_level import compute_classwise_ap, compute_classwise_auc

    def macro_ap(y, p):
        values = np.array(list(compute_classwise_ap(y, p, num_classes).values()), dtype=float)
        return float(np.nanmean(values)) if np.any(~np.isnan(values)) else float("nan")

    def macro_auc(y, p):
        values = np.array(list(compute_classwise_auc(y, p, num_classes).values()), dtype=float)
        return float(np.nanmean(values)) if np.any(~np.isnan(values)) else float("nan")

    return {
        "balanced_accuracy": lambda y, p: float("nan")
        if len(np.unique(y)) < 2
        else balanced_accuracy_score(y, np.argmax(p, axis=1)),
        "macro_f1": lambda y, p: float("nan")
        if len(np.unique(y)) < 2
        else f1_score(y, np.argmax(p, axis=1), average="macro"),
        "micro_f1": lambda y, p: float("nan")
        if len(np.unique(y)) < 2
        else f1_score(y, np.argmax(p, axis=1), average="micro"),
        "macro_ap": macro_ap,
        "macro_roc_auc": macro_auc,
    }


def main() -> None:
    """Entry-point for patient-level evaluation."""

    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    outdir = Path(args.outdir)
    figs_dir = outdir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    data = load_prediction_tables(args.input)
    df = data.frame
    prefixes = (
        [p.strip() for p in args.proba_prefixes.split(",") if p.strip()]
        if args.proba_prefixes
        else infer_prefixes(df)
    )
    if not prefixes:
        raise ValueError("No probability prefixes detected. Provide via --proba-prefixes")
    LOGGER.info("Evaluating prefixes: %s", prefixes)

    proba_cols = ensure_probability_columns(df, prefixes, num_classes=len(CLASS_NAMES))
    aggregated = aggregate_patient_predictions(df, args.id_col, args.true_col, proba_cols, agg=args.agg)

    metrics_by_model: Dict[str, Dict[str, object]] = {}
    ci_rows: List[Dict[str, object]] = []

    for prefix, result in aggregated.items():
        display_name = prefix or "model"
        safe_name = _safe_name(prefix)
        patient_df = result.patients
        metrics = evaluate_patient_metrics(patient_df, args.true_col, num_classes=len(CLASS_NAMES))
        probas = patient_df[[f"proba_{i}" for i in range(len(CLASS_NAMES))]].to_numpy()
        y_true = patient_df[args.true_col].to_numpy()

        metrics["brier_score"] = brier_score(y_true, probas)
        metrics["ece"] = expected_calibration_error(y_true, probas, n_bins=args.n_bins)
        metrics["ace"] = adaptive_calibration_error(y_true, probas, n_bins=args.n_bins)
        metrics_by_model[display_name] = metrics

        metric_funcs = metric_functions(len(CLASS_NAMES))
        ci = bootstrap_confidence_intervals(
            y_true,
            probas,
            metric_funcs,
            n_bootstrap=args.bootstrap,
            random_state=args.seed,
        )
        for metric_name, values in ci.items():
            ci_rows.append(
                {
                    "model": display_name,
                    "metric": metric_name,
                    "lower": values["lower"],
                    "upper": values["upper"],
                    "mean": values["mean"],
                }
            )

        eval_dir = outdir / safe_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        patient_out = eval_dir / f"{safe_name}_patient_level.csv"
        save_dataframe(patient_df, patient_out)

        plot_precision_recall(y_true, probas, CLASS_NAMES, figs_dir, safe_name)
        plot_roc(y_true, probas, CLASS_NAMES, figs_dir, safe_name)
        plot_confusion_matrix(metrics["confusion_matrix"], CLASS_NAMES, figs_dir, safe_name)
        reliability_diagram(
            y_true,
            probas,
            figs_dir / f"{safe_name}_reliability_{_timestamp()}.png",
            n_bins=args.n_bins,
            title=f"Reliability ({display_name})",
        )

    metrics_df = build_metrics_dataframe(metrics_by_model)
    metrics_path = outdir / "patient_metrics.csv"
    save_dataframe(metrics_df, metrics_path)

    if ci_rows:
        ci_df = pd.DataFrame(ci_rows)
        save_dataframe(ci_df, outdir / "patient_metrics_ci.csv")


if __name__ == "__main__":
    main()
