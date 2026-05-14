# Superconductivity Critical Temperature Prediction

Statistical learning project for Subject 4: predicting the critical temperature `critical_temp` of superconducting materials from 81 physical and chemical covariates.

## Authors

- Xiaopeng Zhang
- Marçal Herraiz Bayó
- Shuaibo HUANG
- Carlos Cosentino
- Polina Ptukha
- Lyes Bouchoucha

## Run

From this folder:

```bash
uv sync
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/final_submission.ipynb
```

Or open `notebooks/final_submission.ipynb` in Jupyter with the `uv` environment.

## Contents

```text
minimum_submission/
├── configs/
│   └── default.yaml
├── data/
│   └── processed/
│       ├── cleaning_summary.json
│       └── superconductivity_clean.csv
├── notebooks/
│   └── final_submission.ipynb
├── reports/
│   └── report.pdf
├── src/
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

The notebook recreates `reports/figures/`, `reports/tables/`, and
`artifacts/models/` when it is executed. They are not included here because they
are generated outputs, while `reports/report.pdf` is the report deliverable.
