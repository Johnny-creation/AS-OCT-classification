from __future__ import annotations

import numpy as np
import pandas as pd

from src.stacking.pipeline import (
    hard_vote,
    load_model,
    predict_meta_model,
    save_model,
    train_meta_model,
    weighted_average,
)


def make_stacking_df() -> pd.DataFrame:
    data = {
        "y_true": [0, 1, 2, 3],
        "modelA_logit_0": [3.0, 0.1, 0.2, 0.3],
        "modelA_logit_1": [0.1, 3.0, 0.2, 0.2],
        "modelA_logit_2": [0.2, 0.2, 3.0, 0.2],
        "modelA_logit_3": [0.1, 0.2, 0.3, 3.5],
        "modelB_logit_0": [2.5, 0.2, 0.1, 0.1],
        "modelB_logit_1": [0.2, 2.5, 0.2, 0.1],
        "modelB_logit_2": [0.1, 0.2, 2.5, 0.1],
        "modelB_logit_3": [0.1, 0.1, 0.2, 2.5],
    }
    return pd.DataFrame(data)


def test_train_and_predict_meta_model():
    df = make_stacking_df()
    model, feature_cols = train_meta_model(df, labels_col="y_true", feature_type="logit", meta="lr")
    assert feature_cols
    preds = predict_meta_model(model, df, feature_cols)
    assert preds.shape == (4, 4)
    assert np.allclose(preds.sum(axis=1), 1.0)


def test_weighted_average_and_hard_vote():
    probas_a = np.array([[0.7, 0.2, 0.05, 0.05], [0.1, 0.7, 0.1, 0.1]])
    probas_b = np.array([[0.6, 0.3, 0.05, 0.05], [0.15, 0.65, 0.1, 0.1]])
    combined = weighted_average([probas_a, probas_b], weights=[0.6, 0.4])
    assert combined.shape == probas_a.shape
    votes = hard_vote([probas_a, probas_b])
    assert votes.shape == probas_a.shape
    assert np.allclose(votes.sum(axis=1), 1.0)


def test_save_and_load_meta_model(tmp_path):
    df = make_stacking_df()
    model, feature_cols = train_meta_model(df, labels_col="y_true", feature_type="logit", meta="lr")
    path = tmp_path / "stacker.pkl"
    save_model(model, feature_cols, "logit", path)
    loaded_model, loaded_cols, feature_type = load_model(path)
    assert feature_type == "logit"
    assert loaded_cols == feature_cols
    preds = predict_meta_model(loaded_model, df, loaded_cols)
    assert preds.shape == (4, 4)
