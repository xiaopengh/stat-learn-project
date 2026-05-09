# Project Execution Guide

## Objective

Build a reproducible statistical learning project for Subject 4: predicting the critical temperature `critical_temp` of superconducting materials from 81 physical and chemical covariates.

The official code deliverable will be a Jupyter notebook, but development must happen first in modular Python scripts and source files. The final notebook should only assemble clean, tested, reproducible analysis.

## Authoritative Inputs

- Project instructions: `docs/instructions_projets.pdf`
- Subject descriptions: `docs/projets.pdf`
- Course methods: `.tex` files in `poly/`
- Preliminary notebook: `notebooks/preliminary_main.ipynb`

## Submission Constraints

- Submit one PDF report and one notebook by email.
- Deadline from the project instructions: Friday, May 15, 2026, before 23:59.
- File naming pattern:
  - `Subject4_[Names of all authors].pdf`
  - `Subject4_[Names of all authors].ipynb`
- The report first page must list all group members and Subject 4.

## Repository Layout

```text
.
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
├── notebooks/
├── poly/
├── reports/
│   └── figures/
├── scripts/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   ├── utils/
│   └── visualization/
├── AGENTS.md
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Execution Plan

1. Exploratory data analysis.
2. Data preprocessing and feature engineering.
3. Modeling.
4. Model evaluation.
5. Model diagnosis and interpretation.
6. LaTeX report writing.
7. Final notebook generation for submission.

## Course Methods

Before labeling a method as "seen in class", verify it in `poly/`.

Most relevant covered methods for this regression project:

- linear regression and ordinary least squares;
- Ridge regression;
- Lasso regression;
- cross-validation;
- k-nearest neighbors regression;
- neural networks and stochastic optimization, if used.

If a method is not supported by the course `.tex` files, explain its functioning in the report.

## Modeling Requirements

- Define the learning problem explicitly as supervised regression.
- Identify `critical_temp` as the target and the 81 physical/chemical variables as covariates.
- Compare at least two different methods or modeling approaches.
- Use clear metrics with formulas in the report, such as MSE, RMSE, MAE, and `R^2`.
- Avoid data leakage between preprocessing, model selection, and final evaluation.
- Keep a held-out test set for final reporting when possible.
- Justify preprocessing, especially target transformations and feature scaling.

## Code Standards

- Use `uv` for environment and package management.
- Run scripts from the project root.
- Use reusable functions instead of notebook-style code.
- Use `pathlib.Path` for paths.
- Avoid hardcoded absolute paths.
- Set and record random seeds.
- Log important steps, generated files, and model metrics.
- Save generated figures in `reports/figures/` with explicit filenames.
- Keep source code modular under `src/`.

## Report Standards

- Write the report in LaTeX at `reports/report.tex`.
- Use the same figure directory as the Python pipeline.
- In `reports/report.tex`, configure:

```latex
\graphicspath{{figures/}}
```

The report should cover the problem, dataset, EDA, preprocessing, methods, metrics, diagnosis, limitations, and final conclusion. Every included figure or table must be commented on.
