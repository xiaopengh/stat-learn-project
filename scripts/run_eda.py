from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import (
    cache_dataset,
    dataset_overview,
    feature_family_correlation_summary,
    feature_summary,
    feature_target_correlations,
    load_processed_dataset,
    missing_values_summary,
    target_by_number_of_elements,
    target_summary,
    validate_dataset,
)
from src.utils import ensure_dir, get_logger, get_path, load_config
from src.visualization import (
    plot_feature_family_correlations,
    plot_feature_target_correlations,
    plot_log_target_distribution,
    plot_selected_feature_correlation_matrix,
    plot_target_by_number_of_elements,
    plot_target_distribution,
)

LOGGER = get_logger("run_eda")
LATEX_COLUMN_LABELS = {
    "number_of_elements": "Elements",
    "n_materials": "N",
    "mean_temp": "Mean K",
    "median_temp": "Median K",
    "q25": "Q1 K",
    "q75": "Q3 K",
    "min_temp": "Min K",
    "max_temp": "Max K",
    "high_temp_share": "Share >= 77 K",
    "property_family": "Family",
    "n_features": "Features",
    "mean_abs_correlation": "Mean |corr|",
    "max_abs_correlation": "Max |corr|",
    "strongest_feature": "Strongest feature",
    "strongest_correlation": "Corr.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA for the superconductivity dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def write_latex_table(table, output_path: Path) -> None:
    latex_table = table.copy()
    for column in latex_table.columns:
        dtype_name = str(latex_table[column].dtype)
        if not (dtype_name in {"object", "str"} or dtype_name.startswith("string")):
            continue
        latex_table[column] = latex_table[column].astype(str).str.replace("_", r"\_", regex=False)
    latex_table = latex_table.rename(
        columns={
            column: LATEX_COLUMN_LABELS.get(column, str(column).replace("_", r"\_"))
            for column in latex_table.columns
        }
    )
    output_path.write_text(
        latex_table.to_latex(index=False, float_format=lambda value: f"{value:.3f}"),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    target_name = str(config["dataset"]["target"])

    LOGGER.info("Caching/loading dataset")
    cache_paths = cache_dataset(config, force=args.force_download)
    LOGGER.info("Processed dataset path: %s", cache_paths["processed"])

    dataset = load_processed_dataset(config)
    validate_dataset(dataset, config)

    tables_dir = ensure_dir(get_path(config, "tables"))
    figures_dir = ensure_dir(get_path(config, "figures"))

    overview = dataset_overview(dataset, target_name)
    target_stats = target_summary(dataset, target_name)
    cleaning_summary = {}
    if cache_paths["cleaning_summary"].exists():
        cleaning_summary = json.loads(cache_paths["cleaning_summary"].read_text(encoding="utf-8"))

    (tables_dir / "dataset_overview.json").write_text(
        json.dumps(overview, indent=2),
        encoding="utf-8",
    )
    (tables_dir / "preprocessing_summary.json").write_text(
        json.dumps(cleaning_summary, indent=2),
        encoding="utf-8",
    )
    (tables_dir / "target_summary.json").write_text(
        json.dumps(target_stats, indent=2),
        encoding="utf-8",
    )
    missing_values_summary(dataset).to_csv(tables_dir / "missing_values_summary.csv", index=False)
    feature_summary(dataset, target_name).to_csv(tables_dir / "feature_summary.csv", index=False)
    feature_target_correlations(dataset, target_name).to_csv(
        tables_dir / "feature_target_correlations.csv",
        index=False,
    )
    elements_summary = target_by_number_of_elements(dataset, target_name)
    elements_summary.to_csv(tables_dir / "target_by_number_of_elements.csv", index=False)
    write_latex_table(
        elements_summary,
        tables_dir / "target_by_number_of_elements.tex",
    )

    family_correlations = feature_family_correlation_summary(dataset, target_name)
    family_correlations.to_csv(
        tables_dir / "feature_family_correlation_summary.csv",
        index=False,
    )
    write_latex_table(
        family_correlations[
            [
                "property_family",
                "n_features",
                "mean_abs_correlation",
                "max_abs_correlation",
                "strongest_feature",
                "strongest_correlation",
            ]
        ],
        tables_dir / "feature_family_correlation_summary.tex",
    )

    plot_target_distribution(dataset, target_name, figures_dir)
    plot_log_target_distribution(dataset, target_name, figures_dir)
    plot_feature_target_correlations(dataset, target_name, figures_dir)
    plot_selected_feature_correlation_matrix(dataset, target_name, figures_dir)
    plot_target_by_number_of_elements(dataset, target_name, figures_dir)
    plot_feature_family_correlations(dataset, target_name, figures_dir)

    LOGGER.info("EDA completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
