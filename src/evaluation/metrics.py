from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

MetricFunction = Callable[[np.ndarray, np.ndarray], float]


def _as_1d_float_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.ravel(array)


def _validate_targets(
    y_true: Iterable[float] | np.ndarray,
    y_pred: Iterable[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true_array = _as_1d_float_array(y_true)
    y_pred_array = _as_1d_float_array(y_pred)

    if y_true_array.size == 0:
        msg = "y_true must not be empty."
        raise ValueError(msg)
    if y_true_array.shape != y_pred_array.shape:
        msg = "y_true and y_pred must have the same shape."
        raise ValueError(msg)
    return y_true_array, y_pred_array


def mse(y_true: Iterable[float] | np.ndarray, y_pred: Iterable[float] | np.ndarray) -> float:
    y_true_array, y_pred_array = _validate_targets(y_true, y_pred)
    return float(np.mean((y_true_array - y_pred_array) ** 2))


def rmse(y_true: Iterable[float] | np.ndarray, y_pred: Iterable[float] | np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: Iterable[float] | np.ndarray, y_pred: Iterable[float] | np.ndarray) -> float:
    y_true_array, y_pred_array = _validate_targets(y_true, y_pred)
    return float(np.mean(np.abs(y_true_array - y_pred_array)))


def r2(y_true: Iterable[float] | np.ndarray, y_pred: Iterable[float] | np.ndarray) -> float:
    y_true_array, y_pred_array = _validate_targets(y_true, y_pred)
    residual_sum_squares = float(np.sum((y_true_array - y_pred_array) ** 2))
    total_sum_squares = float(np.sum((y_true_array - np.mean(y_true_array)) ** 2))

    if np.isclose(total_sum_squares, 0.0):
        return 1.0 if np.isclose(residual_sum_squares, 0.0) else 0.0
    return float(1.0 - residual_sum_squares / total_sum_squares)


METRIC_FUNCTIONS: dict[str, MetricFunction] = {
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
}


def evaluate_regression_metrics(
    y_true: Iterable[float] | np.ndarray,
    y_pred: Iterable[float] | np.ndarray,
    metrics: Iterable[str] | None = None,
) -> dict[str, float]:
    metric_names = (
        [metric.lower() for metric in metrics]
        if metrics is not None
        else list(METRIC_FUNCTIONS)
    )
    values: dict[str, float] = {}

    for metric_name in metric_names:
        if metric_name not in METRIC_FUNCTIONS:
            supported = ", ".join(sorted(METRIC_FUNCTIONS))
            msg = f"Unsupported metric '{metric_name}'. Supported metrics: {supported}."
            raise KeyError(msg)
        values[metric_name] = METRIC_FUNCTIONS[metric_name](y_true, y_pred)
    return values
