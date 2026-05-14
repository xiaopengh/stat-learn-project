# Superconductivity Critical Temperature Prediction

Statistical learning project for Subject 4: predicting the critical temperature `critical_temp` of superconducting materials from 81 physical and chemical covariates.

## Authors

- Xiaopeng Zhang
- Marçal Herraiz Bayó
- Shuaibo HUANG
- Carlos Cosentino
- Polina Ptukha
- Lyes Bouchoucha

## Project Layout

```text
.
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
├── notebooks/
│   └── preliminary_main.ipynb
├── poly/
├── reports/
│   ├── figures/
│   └── report.tex
├── scripts/
├── src/
├── tests/
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

## Environment

This project uses `uv` for Python environment and dependency management.

```bash
uv sync
```

Run commands inside the environment with:

```bash
uv run python --version
```

If the environment cannot write to the default user caches, use the project-local fallback:

```bash
env MPLCONFIGDIR=.matplotlib-cache uv --cache-dir .uv-cache sync
env MPLCONFIGDIR=.matplotlib-cache uv --cache-dir .uv-cache run python --version
```

The primary dependency source is `pyproject.toml`, with exact resolved versions stored in `uv.lock`. `requirements.txt` is kept as a compatibility file for tools that require it.

## Notes

- The original preliminary notebook is preserved at `notebooks/preliminary_main.ipynb`.
- Generated figures should be written to `reports/figures/`.
- The LaTeX report entry point is `reports/report.tex`.
- Course method references are stored as `.tex` files in `poly/`.

## Reproduce the Current Analysis

```bash
env MPLCONFIGDIR=.matplotlib-cache uv --cache-dir .uv-cache run python scripts/run_eda.py
env MPLCONFIGDIR=.matplotlib-cache uv --cache-dir .uv-cache run python scripts/train_models.py
env MPLCONFIGDIR=.matplotlib-cache uv --cache-dir .uv-cache run python scripts/evaluate_models.py
env MPLCONFIGDIR=.matplotlib-cache uv --cache-dir .uv-cache run python scripts/build_final_notebook.py
```
