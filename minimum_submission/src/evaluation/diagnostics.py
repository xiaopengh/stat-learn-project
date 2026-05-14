from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.tables import normalize_prediction_records
from src.utils import ensure_dir

TARGET_RANGE_BINS = [-np.inf, 10.0, 30.0, 77.0, 100.0, np.inf]
TARGET_RANGE_LABELS = [
    "<10 K",
    "10-30 K",
    "30-77 K",
    "77-100 K",
    ">=100 K",
]


def build_residual_records(
    predictions: pd.DataFrame | Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    records = normalize_prediction_records(predictions)
    residuals = records.copy()
    residuals["residual"] = residuals["y_true"] - residuals["y_pred"]
    residuals["abs_residual"] = residuals["residual"].abs()
    residuals["squared_residual"] = residuals["residual"] ** 2
    return residuals


def summarize_residual_diagnostics(residual_records: pd.DataFrame) -> pd.DataFrame:
    if residual_records.empty:
        msg = "Residual records are empty."
        raise ValueError(msg)

    summary = (
        residual_records.groupby("model", sort=False)
        .agg(
            n_obs=("residual", "size"),
            residual_mean=("residual", "mean"),
            residual_std=("residual", "std"),
            residual_median=("residual", "median"),
            residual_q05=("residual", lambda series: series.quantile(0.05)),
            residual_q95=("residual", lambda series: series.quantile(0.95)),
            mae=("abs_residual", "mean"),
            rmse=("squared_residual", lambda series: float(np.sqrt(series.mean()))),
        )
        .reset_index()
    )
    return summary.sort_values("rmse", kind="stable").reset_index(drop=True)


def write_residual_diagnostics(
    predictions: pd.DataFrame | Sequence[Mapping[str, object]],
    output_dir: str | Path,
    stem: str = "final_model_residuals",
    *,
    split: str | None = None,
) -> dict[str, Path]:
    output_path = ensure_dir(output_dir)
    residuals = build_residual_records(predictions)
    if split is not None:
        if "split" not in residuals.columns:
            msg = (
                "Cannot filter residual diagnostics by split because records have no "
                "split column."
            )
            raise KeyError(msg)
        residuals = residuals[residuals["split"] == split].reset_index(drop=True)
    summary = summarize_residual_diagnostics(residuals)

    residuals_csv_path = output_path / f"{stem}.csv"
    summary_csv_path = output_path / f"{stem}_summary.csv"
    summary_tex_path = output_path / f"{stem}_summary.tex"

    residuals.to_csv(residuals_csv_path, index=False)
    summary.to_csv(summary_csv_path, index=False)

    latex_summary = summary.copy()
    if "model" in latex_summary.columns:
        latex_summary["model"] = (
            latex_summary["model"].astype(str).str.replace("_", r"\_", regex=False)
        )

    latex_summary = latex_summary.rename(
        columns={
            "model": "Model",
            "n_obs": "N",
            "residual_mean": "Mean resid.",
            "residual_std": "Resid. SD",
            "residual_median": "Median resid.",
            "residual_q05": "Resid. q05",
            "residual_q95": "Resid. q95",
            "mae": "MAE",
            "rmse": "RMSE",
        }
    )
    summary_tex_path.write_text(
        latex_summary.to_latex(index=False, float_format=lambda value: f"{value:.4f}"),
        encoding="utf-8",
    )
    return {
        "residuals_csv": residuals_csv_path,
        "summary_csv": summary_csv_path,
        "summary_tex": summary_tex_path,
    }


def summarize_residuals_by_target_range(
    residual_records: pd.DataFrame,
    *,
    model_name: str | None = None,
) -> pd.DataFrame:
    if residual_records.empty:
        msg = "Residual records are empty."
        raise ValueError(msg)

    records = residual_records.copy()
    if model_name is not None:
        records = records[records["model"] == model_name].copy()
        if records.empty:
            msg = f"No residual records found for model '{model_name}'."
            raise ValueError(msg)

    records["target_range"] = pd.cut(
        records["y_true"],
        bins=TARGET_RANGE_BINS,
        labels=TARGET_RANGE_LABELS,
        right=False,
    )
    summary = (
        records.groupby("target_range", observed=True)
        .agg(
            n_obs=("residual", "size"),
            mean_true_temp=("y_true", "mean"),
            mean_prediction=("y_pred", "mean"),
            residual_mean=("residual", "mean"),
            mae=("abs_residual", "mean"),
            rmse=("squared_residual", lambda series: float(np.sqrt(series.mean()))),
        )
        .reset_index()
    )
    summary.insert(0, "model", model_name if model_name is not None else "all_models")
    return summary


def write_target_range_diagnostics(
    predictions: pd.DataFrame | Sequence[Mapping[str, object]],
    output_dir: str | Path,
    *,
    model_name: str,
    stem: str = "best_model_error_by_target_range",
    split: str | None = None,
) -> dict[str, Path]:
    output_path = ensure_dir(output_dir)
    residuals = build_residual_records(predictions)
    if split is not None:
        if "split" not in residuals.columns:
            msg = (
                "Cannot filter target-range diagnostics by split because records have no "
                "split column."
            )
            raise KeyError(msg)
        residuals = residuals[residuals["split"] == split].reset_index(drop=True)

    summary = summarize_residuals_by_target_range(residuals, model_name=model_name)
    csv_path = output_path / f"{stem}.csv"
    tex_path = output_path / f"{stem}.tex"

    summary.to_csv(csv_path, index=False)
    latex_summary = summary.copy()
    for column in latex_summary.columns:
        dtype_name = str(latex_summary[column].dtype)
        if not (dtype_name in {"object", "str"} or dtype_name.startswith("string")):
            continue
        latex_summary[column] = latex_summary[column].astype(str).str.replace(
            "_",
            r"\_",
            regex=False,
        )
    latex_summary = latex_summary.rename(
        columns={
            "model": "Model",
            "target_range": "Target range",
            "n_obs": "N",
            "mean_true_temp": "Mean true K",
            "mean_prediction": "Mean pred. K",
            "residual_mean": "Mean resid.",
            "mae": "MAE",
            "rmse": "RMSE",
        }
    )
    tex_path.write_text(
        latex_summary.to_latex(index=False, float_format=lambda value: f"{value:.3f}"),
        encoding="utf-8",
    )
    return {"summary_csv": csv_path, "summary_tex": tex_path}
