from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data import feature_target_correlations
from src.utils import ensure_dir


def _save_current_figure(output_path: Path) -> Path:
    ensure_dir(output_path.parent)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    return output_path


def plot_target_distribution(dataset: pd.DataFrame, target_name: str, figures_dir: Path) -> Path:
    output_path = figures_dir / "target_distribution.png"
    target = dataset[target_name]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(target, bins=50, kde=True, ax=axes[0], color="#2f6f9f")
    axes[0].set_title("Distribution of Critical Temperature")
    axes[0].set_xlabel("Critical temperature (K)")
    axes[0].set_ylabel("Count")

    sns.boxplot(x=target, ax=axes[1], color="#df8f44")
    axes[1].set_title("Boxplot of Critical Temperature")
    axes[1].set_xlabel("Critical temperature (K)")
    fig.suptitle("Target Variable: Critical Temperature", y=1.03)
    return _save_current_figure(output_path)


def plot_log_target_distribution(
    dataset: pd.DataFrame,
    target_name: str,
    figures_dir: Path,
) -> Path:
    output_path = figures_dir / "log_target_distribution.png"
    log_target = np.log1p(dataset[target_name])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(log_target, bins=50, kde=True, ax=axes[0], color="#4f7f52")
    axes[0].set_title("Distribution of log(critical temperature + 1)")
    axes[0].set_xlabel("log(critical temperature + 1)")
    axes[0].set_ylabel("Count")

    sns.boxplot(x=log_target, ax=axes[1], color="#b46a6a")
    axes[1].set_title("Boxplot of log(critical temperature + 1)")
    axes[1].set_xlabel("log(critical temperature + 1)")
    fig.suptitle("Log-Transformed Target Variable", y=1.03)
    return _save_current_figure(output_path)


def plot_feature_target_correlations(
    dataset: pd.DataFrame,
    target_name: str,
    figures_dir: Path,
    *,
    top_n: int = 20,
) -> Path:
    output_path = figures_dir / "feature_target_correlations.png"
    correlations = feature_target_correlations(dataset, target_name).head(top_n)
    plot_data = correlations.sort_values("correlation")

    plt.figure(figsize=(9, 7))
    colors = ["#b46a6a" if value < 0 else "#2f6f9f" for value in plot_data["correlation"]]
    plt.barh(plot_data["feature"], plot_data["correlation"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(f"Top {top_n} Feature Correlations with Critical Temperature")
    plt.xlabel("Pearson correlation")
    plt.ylabel("Feature")
    return _save_current_figure(output_path)


def plot_selected_feature_correlation_matrix(
    dataset: pd.DataFrame,
    target_name: str,
    figures_dir: Path,
    *,
    top_n: int = 12,
) -> Path:
    output_path = figures_dir / "selected_feature_correlation_matrix.png"
    selected_features = (
        feature_target_correlations(dataset, target_name)
        .head(top_n)["feature"]
        .tolist()
    )
    corr_data = dataset[[*selected_features, target_name]].corr(numeric_only=True)

    plt.figure(figsize=(11, 9))
    sns.heatmap(corr_data, cmap="vlag", center=0, square=False, linewidths=0.3)
    plt.title("Correlation Matrix for Most Target-Correlated Features")
    return _save_current_figure(output_path)
