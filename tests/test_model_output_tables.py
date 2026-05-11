from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from scripts.evaluate_models import evaluate_models
from src.evaluation import (
    build_model_comparison,
    identify_best_model,
    normalize_prediction_records,
    write_comparison_tables,
)


def test_comparison_helpers_write_csv_and_latex_tables(tmp_path: Path) -> None:
    records = [
        {"model": "ridge", "y_true": 1.0, "y_pred": 0.95},
        {"model": "ridge", "y_true": 2.0, "y_pred": 2.05},
        {"model": "lasso", "y_true": 1.0, "y_pred": 0.50},
        {"model": "lasso", "y_true": 2.0, "y_pred": 1.30},
    ]
    comparison = build_model_comparison(records, metrics=["mse", "rmse", "mae", "r2"])
    best_model = identify_best_model(comparison, metric="rmse")

    assert best_model["model"] == "ridge"

    csv_path, tex_path = write_comparison_tables(
        comparison,
        output_dir=tmp_path,
        stem="comparison_test",
    )
    assert csv_path.exists()
    assert tex_path.exists()

    saved = pd.read_csv(csv_path)
    assert list(saved.columns) == ["model", "mse", "rmse", "mae", "r2"]
    assert saved.iloc[0]["model"] == "ridge"


def test_evaluate_models_script_with_wide_predictions_input(tmp_path: Path) -> None:
    reports_tables = tmp_path / "reports" / "tables"
    reports_tables.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame(
        {
            "y_true": [10.0, 12.0, 15.0, 17.0],
            "ridge_pred": [10.2, 11.8, 14.9, 17.1],
            "lasso_pred": [8.0, 10.0, 13.0, 20.0],
        }
    )
    predictions.to_csv(reports_tables / "model_predictions.csv", index=False)

    config = {
        "paths": {"tables": str(reports_tables)},
        "evaluation": {"metrics": ["mse", "rmse", "mae", "r2"]},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = evaluate_models(config_path=config_path)
    assert result["best_model"]["model"] == "ridge"

    assert (reports_tables / "final_model_comparison.csv").exists()
    assert (reports_tables / "final_model_comparison.tex").exists()
    assert (reports_tables / "final_model_residuals.csv").exists()
    assert (reports_tables / "final_model_residuals_summary.csv").exists()
    assert (reports_tables / "final_model_residuals_summary.tex").exists()

    comparison = pd.read_csv(reports_tables / "final_model_comparison.csv")
    assert comparison.iloc[0]["model"] == "ridge"


def test_normalize_prediction_records_handles_wide_input() -> None:
    wide_predictions = pd.DataFrame(
        {
            "y_true": [1.0, 2.0],
            "ridge_pred": [1.1, 1.9],
            "lasso_pred": [0.7, 1.3],
        }
    )
    records = normalize_prediction_records(wide_predictions)
    assert set(records["model"]) == {"ridge", "lasso"}
    assert len(records) == 4
