from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import evaluate_regression_metrics, mae, mse, r2, rmse


def test_metrics_match_known_values() -> None:
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    assert mse(y_true, y_pred) == pytest.approx(0.375)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(0.375))
    assert mae(y_true, y_pred) == pytest.approx(0.5)
    assert r2(y_true, y_pred) == pytest.approx(0.9486081370449679)


def test_evaluate_regression_metrics_supports_metric_selection() -> None:
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.5, 2.5, 2.0]

    selected_metrics = evaluate_regression_metrics(y_true, y_pred, metrics=["mae", "rmse"])
    assert list(selected_metrics.keys()) == ["mae", "rmse"]
    assert selected_metrics["mae"] == pytest.approx(2.0 / 3.0)


def test_metrics_raise_on_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        mse([1.0, 2.0], [1.0])
