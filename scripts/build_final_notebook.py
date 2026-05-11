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

            This notebook is the clean code submission for the statistical learning project.
            The objective is to predict the critical temperature `critical_temp` from 81
            physical and chemical covariates in the UCI superconductivity dataset.
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import subprocess
            import sys

            import matplotlib.pyplot as plt
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
            """
        ),
        markdown_cell(
            """
            The raw table contains exact duplicate rows, which are removed before the
            train-test split. The processed table has no missing values. The target is positive
            and right-skewed, so the modeling pipeline uses a `log1p` target transformation and
            reports final errors after transforming predictions back to Kelvin.
            """
        ),
        code_cell(
            """
            for figure in [
                "reports/figures/target_distribution.png",
                "reports/figures/log_target_distribution.png",
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
            display(correlations.head(15))

            for figure in [
                "reports/figures/feature_target_correlations.png",
                "reports/figures/selected_feature_correlation_matrix.png",
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
            """
        ),
        markdown_cell(
            """
            ## Diagnosis of the Recommended Model

            The recommended model is the one with the smallest test RMSE. Residual plots are used
            to check whether the error pattern reveals major systematic failures.
            """
        ),
        code_cell(
            """
            for figure in [
                "reports/figures/best_model_predicted_vs_observed.png",
                "reports/figures/best_model_residual_diagnostics.png",
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
