from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation import (
    build_model_comparison,
    identify_best_model,
    normalize_prediction_records,
    write_comparison_tables,
    write_residual_diagnostics,
)
from src.utils import ensure_dir, get_logger, get_path, load_config
from src.visualization import plot_prediction_diagnostics

LOGGER = get_logger(__name__)


def evaluate_models(config_path: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path) if config_path is not None else load_config()
    tables_dir = ensure_dir(get_path(config, "tables"))
    try:
        figures_dir = ensure_dir(get_path(config, "figures"))
    except KeyError:
        figures_dir = ensure_dir(tables_dir.parent / "figures")
    predictions_path = tables_dir / "model_predictions.csv"

    if not predictions_path.exists():
        msg = f"Predictions file not found: {predictions_path}"
        raise FileNotFoundError(msg)

    LOGGER.info("Loading predictions from %s", predictions_path)
    predictions_df = pd.read_csv(predictions_path)
    prediction_records = normalize_prediction_records(predictions_df)

    metric_names = config.get("evaluation", {}).get("metrics")
    evaluation_split = "test" if "split" in prediction_records.columns else None
    comparison_table = build_model_comparison(
        prediction_records,
        metrics=metric_names,
        split=evaluation_split,
    )
    comparison_csv_path, comparison_tex_path = write_comparison_tables(
        comparison_table,
        output_dir=tables_dir,
        stem="final_model_comparison",
    )

    residual_paths = write_residual_diagnostics(
        prediction_records,
        output_dir=tables_dir,
        stem="final_model_residuals",
        split=evaluation_split,
    )
    best_model = identify_best_model(comparison_table, metric="rmse")
    diagnostic_figure_paths = plot_prediction_diagnostics(
        prediction_records,
        figures_dir,
        model_name=str(best_model["model"]),
    )

    LOGGER.info(
        "Best model by RMSE: %s (RMSE=%.6f)",
        best_model["model"],
        float(best_model["rmse"]),
    )
    LOGGER.info("Wrote comparison tables: %s and %s", comparison_csv_path, comparison_tex_path)
    LOGGER.info(
        "Wrote residual diagnostics: %s, %s, %s",
        residual_paths["residuals_csv"],
        residual_paths["summary_csv"],
        residual_paths["summary_tex"],
    )
    LOGGER.info(
        "Wrote diagnostic figures: %s and %s",
        diagnostic_figure_paths["predicted_vs_observed"],
        diagnostic_figure_paths["residual_diagnostics"],
    )

    return {
        "best_model": best_model,
        "comparison_csv": comparison_csv_path,
        "comparison_tex": comparison_tex_path,
        "residual_diagnostics": residual_paths,
        "diagnostic_figures": diagnostic_figure_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions and export final tables."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to a YAML config file (defaults to configs/default.yaml).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluate_models(config_path=args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
