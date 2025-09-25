"""Evaluate trained backbones with patient-level metrics."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable
import torch

from generate_predictions import run_inference
from src.metrics.patient_level import (
    AggregatedPatientResult,
    aggregate_patient_predictions,
    build_metrics_dataframe,
    evaluate_patient_metrics,
)


LOGGER = logging.getLogger("test_multimodel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run patient-level evaluation for trained backbones")
    parser.add_argument(
        "--dataset",
        default="dataset/asoct.test.json",
        help="JSON manifest used for evaluation",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet50"],
        help="Model identifiers matching fine-tuned weight files",
    )
    parser.add_argument(
        "--weights-dir",
        default="weights",
        help="Directory containing best_<model>.pth weight files",
    )
    parser.add_argument(
        "--outdir",
        default="reports/evaluation",
        help="Directory where evaluation tables will be written",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--agg", choices=["mean", "median"], default="mean")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()


def _build_proba_mapping(model_names: Iterable[str], num_classes: int) -> Dict[str, list]:
    mapping: Dict[str, list] = {}
    for name in model_names:
        mapping[name] = [f"{name}_proba_{i}" for i in range(num_classes)]
    return mapping


def _save_patient_tables(results: Dict[str, AggregatedPatientResult], outdir: Path) -> None:
    for prefix, result in results.items():
        path = outdir / f"{prefix}_patient_level.csv"
        result.patients.to_csv(path, index=False)
        if result.conflicts:
            LOGGER.warning(
                "[%s] Found %d patients with conflicting labels: %s",
                prefix,
                result.conflicts,
                ", ".join(result.conflict_patients),
            )


def evaluate_models(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    predictions = run_inference(
        dataset_path=args.dataset,
        model_names=args.models,
        weights_dir=Path(args.weights_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        include_logits=True,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    predictions_path = outdir / "image_level_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    LOGGER.info("Image-level predictions saved to %s", predictions_path)

    # Determine number of classes from probability columns of first model.
    first_model = args.models[0]
    proba_cols = [col for col in predictions.columns if col.startswith(f"{first_model}_proba_")]
    if not proba_cols:
        raise ValueError("No probability columns found in prediction table")
    num_classes = len(proba_cols)

    aggregated = aggregate_patient_predictions(
        predictions,
        id_col="patient_id",
        true_col="y_true",
        proba_cols=_build_proba_mapping(args.models, num_classes),
        agg=args.agg,
    )
    _save_patient_tables(aggregated, outdir)

    metrics_by_model: Dict[str, Dict[str, float]] = {}
    for model_name, result in aggregated.items():
        metrics_by_model[model_name] = evaluate_patient_metrics(
            result.patients,
            true_col="y_true",
            num_classes=num_classes,
        )

    metrics_df = build_metrics_dataframe(metrics_by_model)
    metrics_df.to_csv(outdir / "patient_metrics.csv", index=False)
    LOGGER.info("Saved patient-level metrics to %s", outdir / "patient_metrics.csv")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    evaluate_models(args)


if __name__ == "__main__":
    main()
