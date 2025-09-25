from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics.patient_level import aggregate_patient_predictions, evaluate_patient_metrics
from src.metrics.stat_tests import bootstrap_confidence_intervals, mcnemar_test


def make_dataframe() -> pd.DataFrame:
    data = {
        "patient_id": ["p1", "p1", "p2", "p3", "p3", "p4"],
        "image_id": [1, 2, 1, 1, 2, 1],
        "y_true": [0, 0, 1, 2, 2, 3],
        "modelA_proba_0": [0.8, 0.7, 0.1, 0.2, 0.1, 0.05],
        "modelA_proba_1": [0.1, 0.2, 0.6, 0.1, 0.1, 0.15],
        "modelA_proba_2": [0.05, 0.05, 0.2, 0.6, 0.65, 0.2],
        "modelA_proba_3": [0.05, 0.05, 0.1, 0.1, 0.15, 0.6],
        "modelB_proba_0": [0.6, 0.65, 0.2, 0.1, 0.15, 0.2],
        "modelB_proba_1": [0.2, 0.2, 0.5, 0.1, 0.1, 0.2],
        "modelB_proba_2": [0.1, 0.1, 0.2, 0.6, 0.6, 0.3],
        "modelB_proba_3": [0.1, 0.05, 0.1, 0.2, 0.15, 0.3],
    }
    return pd.DataFrame(data)


def test_patient_aggregation_mean():
    df = make_dataframe()
    proba_cols = {"modelA": [f"modelA_proba_{i}" for i in range(4)], "modelB": [f"modelB_proba_{i}" for i in range(4)]}
    aggregated = aggregate_patient_predictions(df, "patient_id", "y_true", proba_cols, agg="mean")
    assert set(aggregated.keys()) == {"modelA", "modelB"}
    model_a = aggregated["modelA"].patients
    assert len(model_a) == 4
    assert np.isclose(model_a.loc[0, "proba_0"], 0.75, atol=1e-3)


def test_patient_metrics_and_bootstrap():
    df = make_dataframe()
    proba_cols = {"modelA": [f"modelA_proba_{i}" for i in range(4)]}
    aggregated = aggregate_patient_predictions(df, "patient_id", "y_true", proba_cols, agg="mean")
    patients = aggregated["modelA"].patients
    metrics = evaluate_patient_metrics(patients, "y_true")
    assert metrics["balanced_accuracy"] > 0.5
    metric_funcs = {
        "balanced_accuracy": lambda y, p: float(np.mean(np.argmax(p, axis=1) == y)),
    }
    ci = bootstrap_confidence_intervals(
        patients["y_true"].to_numpy(),
        patients[[f"proba_{i}" for i in range(4)]].to_numpy(),
        metric_funcs,
        n_bootstrap=50,
        random_state=0,
    )
    assert "balanced_accuracy" in ci
    assert ci["balanced_accuracy"]["upper"] <= 1.0


def test_mcnemar():
    df = make_dataframe()
    proba_cols = {"modelA": [f"modelA_proba_{i}" for i in range(4)], "modelB": [f"modelB_proba_{i}" for i in range(4)]}
    aggregated = aggregate_patient_predictions(df, "patient_id", "y_true", proba_cols, agg="mean")
    patients_a = aggregated["modelA"].patients
    patients_b = aggregated["modelB"].patients
    result = mcnemar_test(
        patients_a["y_true"].to_numpy(),
        patients_a[[f"proba_{i}" for i in range(4)]].to_numpy(),
        patients_b[[f"proba_{i}" for i in range(4)]].to_numpy(),
    )
    assert set(result.keys()) >= {"statistic", "p_value", "significant"}
