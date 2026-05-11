from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.tables import normalize_prediction_records
from src.utils import ensure_dir


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
            column: "Model" if column == "model" else column.upper()
            for column in summary.columns
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
