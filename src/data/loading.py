from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.utils import ensure_dir, get_path

DATASET_FILENAME = "superconductivity_clean.csv"
FEATURES_FILENAME = "superconductivity_features.csv"
TARGET_FILENAME = "superconductivity_target.csv"
METADATA_FILENAME = "superconductivity_metadata.json"
VARIABLES_FILENAME = "superconductivity_variables.csv"
CLEANING_SUMMARY_FILENAME = "cleaning_summary.json"


def _json_default(value: Any) -> str:
    return str(value)


def fetch_superconductivity_dataset(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    dataset_id = int(config["dataset"]["uci_id"])
    dataset = fetch_ucirepo(id=dataset_id)
    features = dataset.data.features.copy()
    target = dataset.data.targets.copy()
    return features, target, dataset


def build_modeling_table(
    features: pd.DataFrame,
    target: pd.DataFrame,
    target_name: str,
    *,
    drop_duplicate_rows: bool = True,
) -> pd.DataFrame:
    if target_name not in target.columns:
        if target.shape[1] != 1:
            msg = f"Target column '{target_name}' missing and target data has multiple columns."
            raise KeyError(msg)
        target = target.rename(columns={target.columns[0]: target_name})

    table = pd.concat(
        [features.reset_index(drop=True), target[[target_name]].reset_index(drop=True)],
        axis=1,
    )
    if drop_duplicate_rows:
        table = table.drop_duplicates().reset_index(drop=True)
    return table


def cache_dataset(
    config: Mapping[str, Any],
    *,
    force: bool = False,
) -> dict[str, Path]:
    raw_dir = ensure_dir(get_path(config, "raw_data"))
    processed_dir = ensure_dir(get_path(config, "processed_data"))
    target_name = str(config["dataset"]["target"])

    processed_path = processed_dir / DATASET_FILENAME
    features_path = raw_dir / FEATURES_FILENAME
    target_path = raw_dir / TARGET_FILENAME
    metadata_path = raw_dir / METADATA_FILENAME
    variables_path = raw_dir / VARIABLES_FILENAME
    cleaning_summary_path = processed_dir / CLEANING_SUMMARY_FILENAME

    if processed_path.exists() and not force:
        return {
            "features": features_path,
            "target": target_path,
            "metadata": metadata_path,
            "variables": variables_path,
            "cleaning_summary": cleaning_summary_path,
            "processed": processed_path,
        }

    if features_path.exists() and target_path.exists() and not force:
        features = pd.read_csv(features_path)
        target = pd.read_csv(target_path)
        dataset = None
    elif features_path.exists() and target_path.exists() and force:
        features = pd.read_csv(features_path)
        target = pd.read_csv(target_path)
        dataset = None
    else:
        features, target, dataset = fetch_superconductivity_dataset(config)

    raw_modeling_table = build_modeling_table(
        features,
        target,
        target_name,
        drop_duplicate_rows=False,
    )
    modeling_table = raw_modeling_table.drop_duplicates().reset_index(drop=True)
    duplicate_rows_removed = int(len(raw_modeling_table) - len(modeling_table))

    features.to_csv(features_path, index=False)
    target.to_csv(target_path, index=False)
    modeling_table.to_csv(processed_path, index=False)

    cleaning_summary = {
        "raw_rows": int(len(raw_modeling_table)),
        "processed_rows": int(len(modeling_table)),
        "duplicate_rows_removed": duplicate_rows_removed,
        "missing_values_after_cleaning": int(modeling_table.isna().sum().sum()),
    }
    cleaning_summary_path.write_text(json.dumps(cleaning_summary, indent=2), encoding="utf-8")

    if dataset is not None:
        metadata = getattr(dataset, "metadata", {})
        metadata_path.write_text(
            json.dumps(metadata, indent=2, default=_json_default),
            encoding="utf-8",
        )

        variables = getattr(dataset, "variables", None)
        if variables is not None:
            variables.to_csv(variables_path, index=False)

    return {
        "features": features_path,
        "target": target_path,
        "metadata": metadata_path,
        "variables": variables_path,
        "cleaning_summary": cleaning_summary_path,
        "processed": processed_path,
    }


def load_processed_dataset(config: Mapping[str, Any]) -> pd.DataFrame:
    dataset_path = get_path(config, "processed_data") / DATASET_FILENAME
    if not dataset_path.exists():
        msg = f"Processed dataset not found at {dataset_path}. Run scripts/run_eda.py first."
        raise FileNotFoundError(msg)
    return pd.read_csv(dataset_path)
