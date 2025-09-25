from __future__ import annotations

import numpy as np

from src.metrics.calibration import TemperatureScaler, adaptive_calibration_error, expected_calibration_error


def test_temperature_scaler_fit_and_transform():
    logits = np.array([[3.0, 1.0, 0.5, 0.2], [0.1, 3.0, 0.2, 0.1], [0.2, 0.3, 3.0, 0.2], [0.1, 0.2, 0.3, 3.5]])
    labels = np.array([0, 1, 2, 3])
    scaler = TemperatureScaler(init_temp=1.5)
    result = scaler.fit(logits, labels)
    assert result.temperature > 0
    transformed = scaler.transform(logits)
    assert transformed.shape == (4, 4)
    assert np.allclose(transformed.sum(axis=1), 1.0)


def test_calibration_errors():
    probas = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ]
    )
    y_true = np.array([0, 1, 2, 3])
    ece = expected_calibration_error(y_true, probas, n_bins=5)
    ace = adaptive_calibration_error(y_true, probas, n_bins=5)
    assert 0 <= ece <= 1
    assert 0 <= ace <= 1
