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
    feature_summary,
    feature_target_correlations,
    load_processed_dataset,
    missing_values_summary,
    target_summary,
    validate_dataset,
)
from src.utils import ensure_dir, get_logger, get_path, load_config
from src.visualization import (
    plot_feature_target_correlations,
    plot_log_target_distribution,
    plot_selected_feature_correlation_matrix,
    plot_target_distribution,
)

LOGGER = get_logger("run_eda")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA for the superconductivity dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


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

    plot_target_distribution(dataset, target_name, figures_dir)
    plot_log_target_distribution(dataset, target_name, figures_dir)
    plot_feature_target_correlations(dataset, target_name, figures_dir)
    plot_selected_feature_correlation_matrix(dataset, target_name, figures_dir)

    LOGGER.info("EDA completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
