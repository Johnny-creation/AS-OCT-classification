"""CLI for comparing patient-level model outputs via statistical tests."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.metrics.patient_level import aggregate_patient_predictions
from src.metrics.stat_tests import auc_difference, mcnemar_test
from src.utils.io import ensure_probability_columns, infer_prefixes, load_prediction_tables, save_dataframe

LOGGER = logging.getLogger("compare_models")
CLASS_NAMES = ["Normal", "Cataract", "PACG", "PACG+Cataract"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model comparison."""

    parser = argparse.ArgumentParser(description="Compare two or more models using statistical tests")
    parser.add_argument("--inputs", nargs="+", required=True, help="Prediction file(s) or directories")
    parser.add_argument("--id-col", required=True, help="Patient identifier column")
    parser.add_argument("--true-col", required=True, help="Ground truth column")
    parser.add_argument(
        "--proba-prefixes",
        help="Comma-separated prefixes to compare (if omitted auto-detect)",
    )
    parser.add_argument("--outdir", required=True, help="Output directory for reports")
    return parser.parse_args()


def main() -> None:
    """Entry-point for pairwise model comparisons."""

    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_prediction_tables(args.inputs)
    df = data.frame
    prefixes = (
        [p.strip() for p in args.proba_prefixes.split(",") if p.strip()]
        if args.proba_prefixes
        else infer_prefixes(df)
    )
    if len(prefixes) < 2:
        raise ValueError("Need at least two prefixes to compare")
    proba_cols = ensure_probability_columns(df, prefixes, num_classes=len(CLASS_NAMES))
    aggregated = aggregate_patient_predictions(df, args.id_col, args.true_col, proba_cols)

    comparisons: List[Dict[str, Any]] = []
    prefix_names = sorted(aggregated.keys())
    for i in range(len(prefix_names)):
        for j in range(i + 1, len(prefix_names)):
            name_a = prefix_names[i] or "model"
            name_b = prefix_names[j] or "model"
            a = aggregated[prefix_names[i]].patients
            b = aggregated[prefix_names[j]].patients
            y_true = a[args.true_col].to_numpy()
            prob_a = a[[f"proba_{k}" for k in range(len(CLASS_NAMES))]].to_numpy()
            prob_b = b[[f"proba_{k}" for k in range(len(CLASS_NAMES))]].to_numpy()
            mcn = mcnemar_test(y_true, prob_a, prob_b)
            auc_diff = auc_difference(y_true, prob_a, prob_b)
            comparisons.append(
                {
                    "model_a": name_a,
                    "model_b": name_b,
                    "mcnemar_statistic": mcn["statistic"],
                    "mcnemar_p_value": mcn["p_value"],
                    "mcnemar_significant": mcn["significant"],
                    "auc_delta": auc_diff["delta"],
                    "auc_method": auc_diff["method"],
                    "auc_p_value": auc_diff["p_value"],
                    "auc_ci_lower": auc_diff.get("ci_lower", float("nan")),
                    "auc_ci_upper": auc_diff.get("ci_upper", float("nan")),
                }
            )

    df_out = pd.DataFrame(comparisons)
    save_dataframe(df_out, outdir / "model_comparisons.csv")
    LOGGER.info("Saved comparisons for %d pairs", len(comparisons))


if __name__ == "__main__":
    main()
