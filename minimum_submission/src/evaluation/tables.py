from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import evaluate_regression_metrics
from src.utils import ensure_dir

TRUE_COLUMN_CANDIDATES = ("y_true", "true", "actual", "target", "critical_temp")
PRED_COLUMN_CANDIDATES = ("y_pred", "pred", "prediction", "yhat", "y_hat")
MODEL_COLUMN_CANDIDATES = ("model", "model_name", "estimator")
NON_PREDICTION_COLUMNS = {"id", "row_id", "sample_id", "split", "set", "fold", "index"}
PREDICTION_NAME_HINTS = ("pred", "prediction", "yhat")


def _to_dataframe(predictions: pd.DataFrame | Sequence[Mapping[str, object]]) -> pd.DataFrame:
    if isinstance(predictions, pd.DataFrame):
        return predictions.copy()
    return pd.DataFrame(predictions)


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def _clean_model_name(name: str) -> str:
    cleaned_name = name.strip()
    for suffix in ("_prediction", "_pred", "_y_pred", ".prediction", ".pred"):
        if cleaned_name.endswith(suffix):
            return cleaned_name[: -len(suffix)]
    return cleaned_name


def _wide_to_records(predictions: pd.DataFrame, true_column: str) -> pd.DataFrame:
    prediction_columns: list[str] = []
    for column in predictions.columns:
        column_lower = column.lower()
        if column == true_column or column_lower in NON_PREDICTION_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(predictions[column]) or any(
            hint in column_lower for hint in PREDICTION_NAME_HINTS
        ):
            prediction_columns.append(column)

    if not prediction_columns:
        msg = "Could not infer prediction columns from the predictions table."
        raise ValueError(msg)

    melted = predictions[[true_column, *prediction_columns]].melt(
        id_vars=[true_column],
        var_name="model",
        value_name="y_pred",
    )
    records = melted.rename(columns={true_column: "y_true"})
    records["model"] = records["model"].map(_clean_model_name)
    return records


def normalize_prediction_records(
    predictions: pd.DataFrame | Sequence[Mapping[str, object]],
    *,
    true_column: str | None = None,
    pred_column: str | None = None,
    model_column: str | None = None,
) -> pd.DataFrame:
    """Return long-format prediction records with at least columns: model, y_true, y_pred."""
    predictions_df = _to_dataframe(predictions)
    if predictions_df.empty:
        msg = "Predictions data is empty."
        raise ValueError(msg)

    true_column_name = true_column or _find_column(predictions_df.columns, TRUE_COLUMN_CANDIDATES)
    if true_column_name is None:
        msg = f"Could not infer y_true column. Tried: {', '.join(TRUE_COLUMN_CANDIDATES)}."
        raise ValueError(msg)

    model_column_name = model_column or _find_column(
        predictions_df.columns, MODEL_COLUMN_CANDIDATES
    )
    pred_column_name = pred_column or _find_column(predictions_df.columns, PRED_COLUMN_CANDIDATES)

    if model_column_name is not None and pred_column_name is None:
        msg = (
            "Found model column but not prediction column; expected long format with "
            "both model and y_pred-like columns."
        )
        raise ValueError(msg)

    if model_column_name is not None and pred_column_name is not None:
        passthrough_columns = [
            column
            for column in ("split", "row_id")
            if column in predictions_df.columns
            and column not in {model_column_name, true_column_name, pred_column_name}
        ]
        records = predictions_df[
            [*passthrough_columns, model_column_name, true_column_name, pred_column_name]
        ].rename(
            columns={
                model_column_name: "model",
                true_column_name: "y_true",
                pred_column_name: "y_pred",
            }
        )
    elif pred_column_name is not None:
        records = predictions_df[[true_column_name, pred_column_name]].rename(
            columns={true_column_name: "y_true", pred_column_name: "y_pred"}
        )
        records.insert(0, "model", "model")
    else:
        records = _wide_to_records(predictions_df, true_column=true_column_name)

    records["model"] = records["model"].astype(str).str.strip()
    records["y_true"] = pd.to_numeric(records["y_true"], errors="raise")
    records["y_pred"] = pd.to_numeric(records["y_pred"], errors="raise")

    if records[["model", "y_true", "y_pred"]].isna().any().any():
        msg = "Prediction records contain missing values."
        raise ValueError(msg)
    if (records["model"] == "").any():
        msg = "Prediction records contain empty model names."
        raise ValueError(msg)
    return records.reset_index(drop=True)


def build_model_comparison(
    predictions: pd.DataFrame | Sequence[Mapping[str, object]],
    metrics: Iterable[str] | None = None,
    *,
    split: str | None = None,
) -> pd.DataFrame:
    records = normalize_prediction_records(predictions)
    if split is not None:
        if "split" not in records.columns:
            msg = "Cannot filter by split because prediction records have no split column."
            raise KeyError(msg)
        records = records[records["split"] == split].reset_index(drop=True)

    metric_names = [metric.lower() for metric in metrics] if metrics is not None else None

    rows: list[dict[str, float | str]] = []
    for model_name, model_data in records.groupby("model", sort=False):
        metric_values = evaluate_regression_metrics(
            model_data["y_true"].to_numpy(),
            model_data["y_pred"].to_numpy(),
            metrics=metric_names,
        )
        rows.append({"model": model_name, **metric_values})

    if not rows:
        msg = "No model rows available to compare."
        raise ValueError(msg)

    comparison = pd.DataFrame(rows)
    if "rmse" in comparison.columns:
        comparison = comparison.sort_values("rmse", kind="stable").reset_index(drop=True)
    return comparison


def identify_best_model(
    comparison_table: pd.DataFrame,
    metric: str = "rmse",
) -> dict[str, float | str]:
    metric_name = metric.lower()
    if comparison_table.empty:
        msg = "Comparison table is empty."
        raise ValueError(msg)
    if metric_name not in comparison_table.columns:
        msg = f"Metric '{metric_name}' not available in comparison table."
        raise KeyError(msg)

    best_index = pd.to_numeric(comparison_table[metric_name], errors="raise").idxmin()
    best_model = comparison_table.loc[best_index].to_dict()
    return best_model


def write_comparison_tables(
    comparison_table: pd.DataFrame,
    output_dir: str | Path,
    stem: str = "final_model_comparison",
) -> tuple[Path, Path]:
    output_path = ensure_dir(output_dir)
    csv_path = output_path / f"{stem}.csv"
    tex_path = output_path / f"{stem}.tex"

    comparison_table.to_csv(csv_path, index=False)
    latex_table = comparison_table.copy()
    if "model" in latex_table.columns:
        latex_table["model"] = latex_table["model"].astype(str).str.replace("_", r"\_", regex=False)

    latex_table = latex_table.rename(
        columns={
            column: "Model" if column == "model" else column.upper()
            for column in comparison_table.columns
        }
    )
    tex_path.write_text(
        latex_table.to_latex(index=False, float_format=lambda value: f"{value:.4f}"),
        encoding="utf-8",
    )
    return csv_path, tex_path
