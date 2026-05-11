from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def dataset_overview(dataset: pd.DataFrame, target_name: str) -> dict[str, Any]:
    if target_name not in dataset.columns:
        msg = f"Target column '{target_name}' is missing."
        raise KeyError(msg)

    features = dataset.drop(columns=[target_name])
    numeric_features = features.select_dtypes(include="number")
    return {
        "n_rows": int(dataset.shape[0]),
        "n_columns": int(dataset.shape[1]),
        "n_features": int(features.shape[1]),
        "n_numeric_features": int(numeric_features.shape[1]),
        "target": target_name,
        "missing_values_total": int(dataset.isna().sum().sum()),
        "duplicate_rows": int(dataset.duplicated().sum()),
    }


def missing_values_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    summary = (
        dataset.isna()
        .sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    summary["missing_fraction"] = summary["missing_count"] / len(dataset)
    return summary.sort_values(["missing_count", "column"], ascending=[False, True])


def feature_summary(dataset: pd.DataFrame, target_name: str) -> pd.DataFrame:
    features = dataset.drop(columns=[target_name])
    numeric_features = features.select_dtypes(include="number")
    summary = numeric_features.describe().T.reset_index().rename(columns={"index": "feature"})
    return summary


def target_summary(dataset: pd.DataFrame, target_name: str) -> dict[str, float]:
    target = pd.to_numeric(dataset[target_name], errors="raise")
    return {
        "count": float(target.count()),
        "mean": float(target.mean()),
        "std": float(target.std()),
        "min": float(target.min()),
        "q25": float(target.quantile(0.25)),
        "median": float(target.median()),
        "q75": float(target.quantile(0.75)),
        "max": float(target.max()),
        "skew": float(target.skew()),
    }


def feature_target_correlations(dataset: pd.DataFrame, target_name: str) -> pd.DataFrame:
    numeric = dataset.select_dtypes(include="number")
    correlations = (
        numeric.corr(numeric_only=True)[target_name]
        .drop(labels=[target_name])
        .rename("correlation")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    correlations["abs_correlation"] = correlations["correlation"].abs()
    return correlations.sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def validate_dataset(dataset: pd.DataFrame, config: Mapping[str, Any]) -> None:
    target_name = str(config["dataset"]["target"])
    expected_feature_count = int(config["dataset"].get("feature_count", 81))
    overview = dataset_overview(dataset, target_name)

    if overview["n_features"] != expected_feature_count:
        msg = f"Expected {expected_feature_count} features, found {overview['n_features']}."
        raise ValueError(msg)
    if overview["missing_values_total"] != 0:
        msg = (
            "Dataset contains missing values; preprocessing policy must be defined "
            "before modeling."
        )
        raise ValueError(msg)
