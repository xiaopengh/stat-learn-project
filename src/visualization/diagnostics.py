from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import ensure_dir


def plot_prediction_diagnostics(
    predictions: pd.DataFrame,
    figures_dir: Path,
    *,
    model_name: str,
) -> dict[str, Path]:
    ensure_dir(figures_dir)
    model_predictions = predictions[
        (predictions["model"] == model_name) & (predictions.get("split", "test") == "test")
    ].copy()
    if model_predictions.empty:
        model_predictions = predictions[predictions["model"] == model_name].copy()
    if model_predictions.empty:
        msg = f"No predictions found for model '{model_name}'."
        raise ValueError(msg)

    pred_path = figures_dir / "best_model_predicted_vs_observed.png"
    residual_path = figures_dir / "best_model_residual_diagnostics.png"

    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=model_predictions, x="y_true", y="y_pred", s=18, alpha=0.55)
    min_value = min(model_predictions["y_true"].min(), model_predictions["y_pred"].min())
    max_value = max(model_predictions["y_true"].max(), model_predictions["y_pred"].max())
    plt.plot([min_value, max_value], [min_value, max_value], color="black", linewidth=1)
    plt.title(f"Predicted vs Observed Critical Temperature: {model_name}")
    plt.xlabel("Observed critical temperature (K)")
    plt.ylabel("Predicted critical temperature (K)")
    plt.tight_layout()
    plt.savefig(pred_path, dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    residuals = model_predictions["y_true"] - model_predictions["y_pred"]
    sns.scatterplot(x=model_predictions["y_pred"], y=residuals, s=18, alpha=0.55)
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Residuals vs Predictions: {model_name}")
    plt.xlabel("Predicted critical temperature (K)")
    plt.ylabel("Residual (observed - predicted, K)")
    plt.tight_layout()
    plt.savefig(residual_path, dpi=180, bbox_inches="tight")
    plt.close()

    return {
        "predicted_vs_observed": pred_path,
        "residual_diagnostics": residual_path,
    }
