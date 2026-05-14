from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import nbformat as nbf

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ensure_dir, get_logger, get_path, load_config

LOGGER = get_logger("build_final_notebook")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final submission notebook.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    return parser.parse_args()


def markdown_cell(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code_cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def _build_legacy_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

    nb.cells = [
        markdown_cell(
            """
            # Subject 4: Prediction of the Critical Temperature of Superconducting Materials

            **Authors:** Xiaopeng Zhang, Marçal Herraiz Bayó, Shuaibo HUANG,
            Carlos Cosentino, Polina Ptukha, and Lyes Bouchoucha.

            This notebook is the clean code submission for the statistical learning project.
            The objective is to predict the critical temperature `critical_temp` from 81
            physical and chemical covariates in the UCI superconductivity dataset.
            """
        ),
        code_cell(
            """
            import subprocess
            import sys
            from pathlib import Path

            import pandas as pd
            from IPython.display import Image, display

            PROJECT_ROOT = Path.cwd()
            if not (PROJECT_ROOT / "pyproject.toml").exists():
                PROJECT_ROOT = PROJECT_ROOT.parent
            """
        ),
        markdown_cell(
            """
            ## Reproducible Pipeline

            The notebook delegates implementation to reusable Python modules and scripts.
            This avoids duplicating notebook-only logic and keeps preprocessing, model
            selection, and evaluation reproducible.
            """
        ),
        code_cell(
            """
            for command in [
                [sys.executable, "scripts/run_eda.py"],
                [sys.executable, "scripts/train_models.py"],
                [sys.executable, "scripts/evaluate_models.py"],
            ]:
                print("$", " ".join(command))
                subprocess.run(command, check=True, cwd=PROJECT_ROOT)
            """
        ),
        markdown_cell("## Dataset Overview"),
        code_cell(
            """
            preprocessing = pd.read_json(
                PROJECT_ROOT / "reports/tables/preprocessing_summary.json",
                typ="series",
            )
            overview = pd.read_json(
                PROJECT_ROOT / "reports/tables/dataset_overview.json",
                typ="series",
            )
            target_summary = pd.read_json(
                PROJECT_ROOT / "reports/tables/target_summary.json",
                typ="series",
            )
            display(preprocessing.to_frame("value"))
            display(overview.to_frame("value"))
            display(target_summary.to_frame("value"))

            elements = pd.read_csv(PROJECT_ROOT / "reports/tables/target_by_number_of_elements.csv")
            display(elements)
            """
        ),
        markdown_cell(
            """
            The raw table contains exact duplicate rows, which are removed before the
            train-test split. The processed table has no missing values. The target is positive
            and right-skewed, so the modeling pipeline uses a `log1p` target transformation and
            reports final errors after transforming predictions back to Kelvin. The grouped
            summary by `number_of_elements` shows that compositions with more elements are much
            more likely to include high critical-temperature observations.
            """
        ),
        code_cell(
            """
            for figure in [
                "reports/figures/target_distribution.png",
                "reports/figures/log_target_distribution.png",
                "reports/figures/target_by_number_of_elements.png",
            ]:
                display(Image(filename=str(PROJECT_ROOT / figure)))
            """
        ),
        markdown_cell("## Exploratory Feature Analysis"),
        code_cell(
            """
            correlations = pd.read_csv(
                PROJECT_ROOT / "reports/tables/feature_target_correlations.csv"
            )
            family_correlations = pd.read_csv(
                PROJECT_ROOT / "reports/tables/feature_family_correlation_summary.csv"
            )
            display(correlations.head(15))
            display(family_correlations)

            for figure in [
                "reports/figures/feature_target_correlations.png",
                "reports/figures/selected_feature_correlation_matrix.png",
                "reports/figures/feature_family_correlations.png",
            ]:
                display(Image(filename=str(PROJECT_ROOT / figure)))
            """
        ),
        markdown_cell(
            """
            ## Model Comparison

            The final comparison below is computed only on the held-out test set. Hyperparameters
            are selected inside the training data by cross-validation where applicable.
            """
        ),
        code_cell(
            """
            comparison = pd.read_csv(PROJECT_ROOT / "reports/tables/final_model_comparison.csv")
            display(comparison)

            full_comparison = pd.read_csv(PROJECT_ROOT / "reports/tables/model_comparison.csv")
            display(full_comparison)
            """
        ),
        markdown_cell(
            """
            ## Diagnosis of the Recommended Model

            The recommended model is the one with the smallest test RMSE. Residual plots are used
            to check whether the error pattern reveals major systematic failures. We also
            summarize the selected model by observed target range because the high-temperature
            tail is scientifically important and harder to predict.
            """
        ),
        code_cell(
            """
            residual_summary = pd.read_csv(
                PROJECT_ROOT / "reports/tables/final_model_residuals_summary.csv"
            )
            range_errors = pd.read_csv(
                PROJECT_ROOT / "reports/tables/best_model_error_by_target_range.csv"
            )
            display(residual_summary)
            display(range_errors)
            """
        ),
        code_cell(
            """
            for figure in [
                "reports/figures/best_model_predicted_vs_observed.png",
                "reports/figures/best_model_residual_diagnostics.png",
                "reports/figures/best_model_error_by_target_range.png",
            ]:
                display(Image(filename=str(PROJECT_ROOT / figure)))
            """
        ),
        markdown_cell(
            """
            ## Conclusion

            On the cleaned table, k-NN regression gives the lowest held-out RMSE in this run,
            narrowly ahead of `HistGradientBoostingRegressor` and clearly ahead of OLS, Ridge,
            and Lasso. Ridge, Lasso, k-NN, and the general evaluation framework are covered by
            the course material; the gradient boosting model is included as one additional
            nonlinear benchmark and must be explained explicitly in the written report.
            """
        ),
    ]
    return nb


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

    nb.cells = [
        markdown_cell(
            """
            # Subject 4: Prediction of the Critical Temperature of Superconducting Materials

            **Authors:** Xiaopeng Zhang, Marçal Herraiz Bayó, Shuaibo Huang,
            Carlos Cosentino, Polina Ptukha, and Lyes Bouchoucha.

            This notebook is the code submission for the statistical learning project.
            It predicts the critical temperature `critical_temp` from the 81 numerical
            physical and chemical covariates in the UCI superconductivity dataset.
            """
        ),
        markdown_cell(
            """
            ## Notebook Reproducibility

            The notebook is the executable analysis artifact. Running the cells from top
            to bottom loads the project configuration, rebuilds the cleaned dataset and
            exploratory summaries, trains all candidate models, writes the tables and
            figures used in the report, and displays the final diagnostics.

            The reusable implementation lives in `src/`, while this notebook fixes the
            analysis order and records the exact configuration used for the submitted
            results. The Python environment is managed with `uv`; from the project root,
            run `uv sync` once and then open or execute this notebook with the synced
            environment.
            """
        ),
        code_cell(
            """
            from __future__ import annotations

            import json
            import sys
            from pathlib import Path
            from time import perf_counter
            from typing import Any

            import joblib
            import numpy as np
            import pandas as pd
            from IPython.display import Image, display

            PROJECT_ROOT = Path.cwd().resolve()
            while (
                not (PROJECT_ROOT / "pyproject.toml").exists()
                and PROJECT_ROOT != PROJECT_ROOT.parent
            ):
                PROJECT_ROOT = PROJECT_ROOT.parent
            if not (PROJECT_ROOT / "pyproject.toml").exists():
                raise FileNotFoundError("Run this notebook from inside the project repository.")
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
            """
        ),
        code_cell(
            """
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
            from src.evaluation import (
                build_model_comparison,
                identify_best_model,
                write_comparison_tables,
                write_residual_diagnostics,
                write_target_range_diagnostics,
            )
            from src.features import (
                select_numeric_features,
                split_features_target,
                split_train_test,
            )
            from src.models import build_model_registry
            from src.utils import ensure_dir, get_path, load_config, set_random_seed
            from src.visualization import (
                plot_feature_family_correlations,
                plot_feature_target_correlations,
                plot_log_target_distribution,
                plot_prediction_diagnostics,
                plot_selected_feature_correlation_matrix,
                plot_target_by_number_of_elements,
                plot_target_distribution,
                plot_target_range_error_summary,
            )
            """
        ),
        code_cell(
            """
            CONFIG_PATH = PROJECT_ROOT / "configs/default.yaml"
            config = load_config(CONFIG_PATH)

            target_name = str(config["dataset"]["target"])
            random_seed = int(config.get("project", {}).get("random_seed", 42))
            split_config = config.get("split", {})
            cv_config = config.get("cross_validation", {})
            metric_names = [str(name) for name in config.get("evaluation", {}).get("metrics", [])]

            set_random_seed(random_seed)

            tables_dir = ensure_dir(get_path(config, "tables"))
            figures_dir = ensure_dir(get_path(config, "figures"))
            models_dir = ensure_dir(get_path(config, "models"))

            pd.set_option("display.max_columns", 50)
            pd.set_option("display.width", 120)

            print(f"Project root: {PROJECT_ROOT}")
            print(f"Target: {target_name}")
            print(f"Random seed: {random_seed}")
            print(
                "Train/test split: "
                f"test_size={split_config.get('test_size')}, "
                f"random_state={split_config.get('random_state')}"
            )
            print(f"Cross-validation folds: {cv_config.get('folds')}")
            print(f"Metrics: {metric_names}")
            """
        ),
        markdown_cell(
            """
            ## Dataset Audit and Exploratory Analysis

            The raw feature and target tables are combined, exact duplicate rows are
            removed before the train-test split, and the resulting modeling table is
            validated before any model is fit. The exploratory summaries below are
            descriptive only; they are not used to select features before final
            evaluation.
            """
        ),
        code_cell(
            """
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


            def write_latex_table(table: pd.DataFrame, output_path: Path) -> None:
                latex_table = table.copy()
                for column in latex_table.columns:
                    dtype_name = str(latex_table[column].dtype)
                    if dtype_name in {"object", "str"} or dtype_name.startswith("string"):
                        latex_table[column] = (
                            latex_table[column].astype(str).str.replace("_", r"\\_", regex=False)
                        )
                latex_table = latex_table.rename(
                    columns={
                        column: LATEX_COLUMN_LABELS.get(column, str(column).replace("_", r"\\_"))
                        for column in latex_table.columns
                    }
                )
                output_path.write_text(
                    latex_table.to_latex(index=False, float_format=lambda value: f"{value:.3f}"),
                    encoding="utf-8",
                )


            cache_paths = cache_dataset(config, force=False)
            dataset = load_processed_dataset(config)
            validate_dataset(dataset, config)

            overview = dataset_overview(dataset, target_name)
            target_stats = target_summary(dataset, target_name)
            cleaning_summary = {}
            if cache_paths["cleaning_summary"].exists():
                cleaning_summary = json.loads(
                    cache_paths["cleaning_summary"].read_text(encoding="utf-8")
                )

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

            missing_values_summary(dataset).to_csv(
                tables_dir / "missing_values_summary.csv",
                index=False,
            )
            feature_summary(dataset, target_name).to_csv(
                tables_dir / "feature_summary.csv",
                index=False,
            )
            correlations = feature_target_correlations(dataset, target_name)
            correlations.to_csv(tables_dir / "feature_target_correlations.csv", index=False)

            elements_summary = target_by_number_of_elements(dataset, target_name)
            elements_summary.to_csv(tables_dir / "target_by_number_of_elements.csv", index=False)
            write_latex_table(elements_summary, tables_dir / "target_by_number_of_elements.tex")

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

            display(pd.Series(cleaning_summary).to_frame("value"))
            display(pd.Series(overview).to_frame("value"))
            display(pd.Series(target_stats).to_frame("value"))
            display(elements_summary)
            """
        ),
        markdown_cell(
            """
            The cleaned table contains no missing values. The target distribution is
            positive and right-skewed, so model fitting uses a `log1p` transformation
            of `critical_temp`, while all reported metrics are computed after inverse
            transformation back to Kelvin.
            """
        ),
        code_cell(
            """
            for figure in [
                "reports/figures/target_distribution.png",
                "reports/figures/log_target_distribution.png",
                "reports/figures/target_by_number_of_elements.png",
            ]:
                display(Image(filename=str(PROJECT_ROOT / figure)))
            """
        ),
        markdown_cell("## Feature Summaries"),
        code_cell(
            """
            display(correlations.head(15))
            display(family_correlations)

            for figure in [
                "reports/figures/feature_target_correlations.png",
                "reports/figures/selected_feature_correlation_matrix.png",
                "reports/figures/feature_family_correlations.png",
            ]:
                display(Image(filename=str(PROJECT_ROOT / figure)))
            """
        ),
        markdown_cell(
            """
            ## Model Fitting

            The learning problem is supervised regression with `critical_temp` as the
            target. The candidate models are a mean baseline, ordinary least squares,
            Ridge regression, Lasso regression, k-nearest-neighbor regression, and
            histogram-based gradient boosting. OLS, Ridge, Lasso, and k-NN are fit
            with feature standardization inside the model pipeline; target
            transformation is handled inside each fitted estimator.
            """
        ),
        code_cell(
            """
            def extract_model_search_metadata(model: Any) -> dict[str, Any]:
                inner = model
                if hasattr(inner, "regressor_"):
                    inner = inner.regressor_
                elif hasattr(inner, "regressor"):
                    inner = inner.regressor

                if hasattr(inner, "named_steps") and "regressor" in inner.named_steps:
                    inner = inner.named_steps["regressor"]

                metadata: dict[str, Any] = {}
                if hasattr(inner, "best_params_") and inner.best_params_ is not None:
                    metadata["best_params"] = dict(inner.best_params_)
                if hasattr(inner, "best_score_") and inner.best_score_ is not None:
                    metadata["best_score"] = float(inner.best_score_)
                if hasattr(inner, "alpha_") and inner.alpha_ is not None:
                    metadata["selected_alpha"] = float(inner.alpha_)
                if hasattr(inner, "n_neighbors") and inner.n_neighbors is not None:
                    metadata["selected_neighbors"] = int(inner.n_neighbors)
                return metadata


            def make_predictions_frame(
                *,
                model_name: str,
                split_name: str,
                y_true: pd.Series,
                y_pred: np.ndarray,
            ) -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        "row_id": y_true.index,
                        "model": model_name,
                        "split": split_name,
                        "y_true": y_true.to_numpy(),
                        "y_pred": y_pred,
                        "residual": y_true.to_numpy() - y_pred,
                    }
                )


            X, y = split_features_target(dataset, target_name)
            X_numeric = select_numeric_features(X)
            split = split_train_test(
                X_numeric,
                y,
                test_size=float(split_config.get("test_size", 0.2)),
                random_state=int(split_config.get("random_state", random_seed)),
            )

            print(f"Rows: {len(dataset)}")
            print(f"Numeric features: {X_numeric.shape[1]}")
            print(f"Training rows: {split.X_train.shape[0]}")
            print(f"Test rows: {split.X_test.shape[0]}")

            model_registry = build_model_registry(config, apply_log_target=True)
            trained_models: dict[str, Any] = {}
            predictions_frames: list[pd.DataFrame] = []
            comparison_rows: list[dict[str, Any]] = []

            for model_name, estimator in model_registry.items():
                start = perf_counter()
                estimator.fit(split.X_train, split.y_train)
                fit_seconds = perf_counter() - start

                train_pred = estimator.predict(split.X_train)
                test_pred = estimator.predict(split.X_test)

                train_frame = make_predictions_frame(
                    model_name=model_name,
                    split_name="train",
                    y_true=split.y_train,
                    y_pred=train_pred,
                )
                test_frame = make_predictions_frame(
                    model_name=model_name,
                    split_name="test",
                    y_true=split.y_test,
                    y_pred=test_pred,
                )
                train_metrics = {
                    key: value
                    for key, value in build_model_comparison(train_frame, metrics=metric_names)
                    .iloc[0]
                    .items()
                    if key != "model"
                }
                test_metrics = {
                    key: value
                    for key, value in build_model_comparison(test_frame, metrics=metric_names)
                    .iloc[0]
                    .items()
                    if key != "model"
                }

                predictions_frames.extend([train_frame, test_frame])

                metadata = extract_model_search_metadata(estimator)
                comparison_row: dict[str, Any] = {
                    "model": model_name,
                    "fit_time_seconds": fit_seconds,
                }
                comparison_row.update(
                    {f"train_{key}": value for key, value in train_metrics.items()}
                )
                comparison_row.update(
                    {f"test_{key}": value for key, value in test_metrics.items()}
                )
                comparison_row["model_metadata"] = json.dumps(metadata, sort_keys=True)
                comparison_rows.append(comparison_row)

                trained_models[model_name] = estimator
                joblib.dump(estimator, models_dir / f"{model_name}.joblib")

            model_bundle = {
                "target": target_name,
                "feature_columns": X_numeric.columns.tolist(),
                "models": trained_models,
            }
            joblib.dump(model_bundle, models_dir / "model_bundle.joblib")

            predictions = pd.concat(predictions_frames, ignore_index=True)
            predictions.to_csv(tables_dir / "model_predictions.csv", index=False)

            full_comparison = pd.DataFrame(comparison_rows).sort_values(
                by="test_rmse",
                ascending=True,
                kind="stable",
            )
            full_comparison.to_csv(tables_dir / "model_comparison.csv", index=False)
            display(full_comparison)
            """
        ),
        markdown_cell(
            """
            ## Model Comparison and Diagnostics

            Hyperparameters for Ridge, Lasso, and k-NN are selected using only the
            training data. The final comparison below uses the held-out test set and
            reports MSE, RMSE, MAE, and \(R^2\) on the original Kelvin scale.
            """
        ),
        code_cell(
            """
            comparison = build_model_comparison(
                predictions,
                metrics=metric_names,
                split="test",
            )
            write_comparison_tables(
                comparison,
                output_dir=tables_dir,
                stem="final_model_comparison",
            )

            residual_paths = write_residual_diagnostics(
                predictions,
                output_dir=tables_dir,
                stem="final_model_residuals",
                split="test",
            )
            best_model = identify_best_model(comparison, metric="rmse")
            target_range_paths = write_target_range_diagnostics(
                predictions,
                output_dir=tables_dir,
                model_name=str(best_model["model"]),
                split="test",
            )
            residual_summary = pd.read_csv(residual_paths["summary_csv"])
            target_range_summary = pd.read_csv(target_range_paths["summary_csv"])

            plot_prediction_diagnostics(
                predictions,
                figures_dir,
                model_name=str(best_model["model"]),
            )
            plot_target_range_error_summary(
                target_range_summary,
                figures_dir,
                model_name=str(best_model["model"]),
            )

            display(comparison)
            print(f"Recommended model by test RMSE: {best_model['model']}")
            display(residual_summary)
            display(target_range_summary)
            """
        ),
        code_cell(
            """
            for figure in [
                "reports/figures/best_model_predicted_vs_observed.png",
                "reports/figures/best_model_residual_diagnostics.png",
                "reports/figures/best_model_error_by_target_range.png",
            ]:
                display(Image(filename=str(PROJECT_ROOT / figure)))
            """
        ),
        markdown_cell(
            """
            ## Conclusion

            On the cleaned table, k-NN regression gives the lowest held-out RMSE in
            this run, narrowly ahead of histogram-based gradient boosting and clearly
            ahead of OLS, Ridge, and Lasso. The selected k-NN model has \(k=3\). Its
            strongest remaining weakness is the high-temperature tail, where residual
            diagnostics show larger errors and average underprediction.
            """
        ),
    ]
    return nb


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    output_path = get_path(config, "final_notebook")
    ensure_dir(output_path.parent)

    notebook = build_notebook()
    nbf.write(notebook, output_path)
    LOGGER.info("Wrote final notebook to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
