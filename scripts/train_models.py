from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features import select_numeric_features, split_features_target, split_train_test
from src.models import build_model_registry
from src.utils import ensure_dir, get_logger, get_path, load_config, set_random_seed

LOGGER = get_logger("train_models")
EVALUATION_FUNCTION_NAMES = (
    "evaluate_regression_metrics",
    "evaluate_regression",
    "compute_regression_metrics",
    "regression_metrics",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train regression models for superconductivity prediction."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def resolve_dataset_path(config: Mapping[str, Any]) -> Path:
    processed_dir = get_path(config, "processed_data")
    dataset_path = processed_dir / "superconductivity_clean.csv"
    if not dataset_path.exists():
        msg = (
            f"Processed dataset not found at '{dataset_path}'. "
            "Run 'python scripts/run_eda.py' from the project root first."
        )
        raise FileNotFoundError(msg)
    return dataset_path


def fallback_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def resolve_external_evaluation_function() -> Callable[..., Any] | None:
    try:
        module = importlib.import_module("src.evaluation")
    except Exception:
        return None

    for function_name in EVALUATION_FUNCTION_NAMES:
        evaluator = getattr(module, function_name, None)
        if callable(evaluator):
            LOGGER.info("Using metric function from src.evaluation: %s", function_name)
            return evaluator
    return None


def _normalize_metric_output(
    maybe_metrics: Any,
    metric_names: list[str],
) -> dict[str, float] | None:
    if not isinstance(maybe_metrics, Mapping):
        return None

    if metric_names:
        if not all(metric_name in maybe_metrics for metric_name in metric_names):
            return None
        return {metric_name: float(maybe_metrics[metric_name]) for metric_name in metric_names}

    normalized = {
        str(metric_name): float(metric_value)
        for metric_name, metric_value in maybe_metrics.items()
        if isinstance(metric_value, int | float)
    }
    if normalized:
        return normalized
    return None


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    *,
    metric_names: list[str],
    external_evaluator: Callable[..., Any] | None,
) -> dict[str, float]:
    if external_evaluator is not None:
        call_variants = (
            lambda: external_evaluator(y_true, y_pred),
            lambda: external_evaluator(y_true=y_true, y_pred=y_pred),
            lambda: external_evaluator(y_true, y_pred, metrics=metric_names),
            lambda: external_evaluator(y_true=y_true, y_pred=y_pred, metrics=metric_names),
        )
        for call in call_variants:
            try:
                output = call()
            except TypeError:
                continue
            except Exception as exc:
                LOGGER.warning("External evaluator failed; falling back to local metrics: %s", exc)
                break

            normalized = _normalize_metric_output(output, metric_names)
            if normalized is not None:
                return normalized

    return fallback_regression_metrics(y_true, y_pred)


def extract_model_search_metadata(model: Any) -> dict[str, Any]:
    inner = model
    if hasattr(inner, "regressor_"):
        inner = inner.regressor_
    elif hasattr(inner, "regressor"):
        inner = inner.regressor

    if hasattr(inner, "named_steps") and "regressor" in inner.named_steps:
        inner = inner.named_steps["regressor"]

    metadata: dict[str, Any] = {}
    if hasattr(inner, "best_params_") and inner.best_params_ is not None:
        metadata["best_params"] = dict(inner.best_params_)

    if hasattr(inner, "best_score_") and inner.best_score_ is not None:
        metadata["best_score"] = float(inner.best_score_)

    if hasattr(inner, "alpha_") and inner.alpha_ is not None:
        metadata["selected_alpha"] = float(inner.alpha_)

    if hasattr(inner, "n_neighbors") and inner.n_neighbors is not None:
        metadata["selected_neighbors"] = int(inner.n_neighbors)

    return metadata


def make_predictions_frame(
    *,
    model_name: str,
    split_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": y_true.index,
            "model": model_name,
            "split": split_name,
            "y_true": y_true.to_numpy(),
            "y_pred": y_pred,
            "residual": y_true.to_numpy() - y_pred,
        }
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    random_seed = int(config.get("project", {}).get("random_seed", 42))
    set_random_seed(random_seed)

    target_name = str(config.get("dataset", {}).get("target", "critical_temp"))
    split_config = config.get("split", {})
    test_size = float(split_config.get("test_size", 0.2))
    split_random_state = int(split_config.get("random_state", random_seed))
    metric_names = [str(name) for name in config.get("evaluation", {}).get("metrics", [])]

    dataset_path = resolve_dataset_path(config)
    LOGGER.info("Loading processed dataset from %s", dataset_path)
    dataset = pd.read_csv(dataset_path)

    X, y = split_features_target(dataset, target_name)
    X_numeric = select_numeric_features(X)
    split = split_train_test(
        X_numeric,
        y,
        test_size=test_size,
        random_state=split_random_state,
    )

    LOGGER.info(
        "Data ready: %d rows, %d numeric features, train=%d rows, test=%d rows",
        len(dataset),
        X_numeric.shape[1],
        split.X_train.shape[0],
        split.X_test.shape[0],
    )

    model_registry = build_model_registry(config, apply_log_target=True)
    external_evaluator = resolve_external_evaluation_function()

    models_dir = ensure_dir(get_path(config, "models"))
    tables_dir = ensure_dir(get_path(config, "tables"))

    trained_models: dict[str, Any] = {}
    predictions_frames: list[pd.DataFrame] = []
    comparison_rows: list[dict[str, Any]] = []

    for model_name, estimator in model_registry.items():
        LOGGER.info("Training model: %s", model_name)
        start = perf_counter()
        estimator.fit(split.X_train, split.y_train)
        fit_seconds = perf_counter() - start

        train_pred = estimator.predict(split.X_train)
        test_pred = estimator.predict(split.X_test)

        train_metrics = compute_metrics(
            split.y_train,
            train_pred,
            metric_names=metric_names,
            external_evaluator=external_evaluator,
        )
        test_metrics = compute_metrics(
            split.y_test,
            test_pred,
            metric_names=metric_names,
            external_evaluator=external_evaluator,
        )

        predictions_frames.append(
            make_predictions_frame(
                model_name=model_name,
                split_name="train",
                y_true=split.y_train,
                y_pred=train_pred,
            )
        )
        predictions_frames.append(
            make_predictions_frame(
                model_name=model_name,
                split_name="test",
                y_true=split.y_test,
                y_pred=test_pred,
            )
        )

        metadata = extract_model_search_metadata(estimator)
        comparison_row: dict[str, Any] = {
            "model": model_name,
            "fit_time_seconds": fit_seconds,
        }
        comparison_row.update({f"train_{key}": value for key, value in train_metrics.items()})
        comparison_row.update({f"test_{key}": value for key, value in test_metrics.items()})
        comparison_row["model_metadata"] = json.dumps(metadata, sort_keys=True)
        comparison_rows.append(comparison_row)

        trained_models[model_name] = estimator
        model_path = models_dir / f"{model_name}.joblib"
        joblib.dump(estimator, model_path)
        LOGGER.info("Saved fitted model to %s", model_path)

    model_bundle = {
        "target": target_name,
        "feature_columns": X_numeric.columns.tolist(),
        "models": trained_models,
    }
    bundle_path = models_dir / "model_bundle.joblib"
    joblib.dump(model_bundle, bundle_path)
    LOGGER.info("Saved model bundle to %s", bundle_path)

    predictions_path = tables_dir / "model_predictions.csv"
    comparison_path = tables_dir / "model_comparison.csv"
    pd.concat(predictions_frames, ignore_index=True).to_csv(predictions_path, index=False)
    comparison_df = pd.DataFrame(comparison_rows)
    sort_column = "test_rmse" if "test_rmse" in comparison_df.columns else "test_mse"
    if sort_column in comparison_df.columns:
        comparison_df = comparison_df.sort_values(by=sort_column, ascending=True)
    comparison_df.to_csv(comparison_path, index=False)

    LOGGER.info("Saved predictions table to %s", predictions_path)
    LOGGER.info("Saved comparison table to %s", comparison_path)
    LOGGER.info("Training workflow completed successfully.")


if __name__ == "__main__":
    main()
