# 10pearl: Forecasting Project

## Project Overview

This repository contains code, models, notebooks, and reports for a time-series forecasting project (AQI / air-quality forecasting). The project implements data processing, model training, inference, SHAP analyses, and a small dashboard for visualizing forecasts.

Important: The project's thinking, design trade-offs, and decision log are recorded in [process.txt](process.txt).

## Repository structure

- [process.txt](process.txt): design notes, decisions, and project thinking.
- [project.txt](project.txt): short project summary and metadata.
- [requirements.txt](requirements.txt) and [requirements-general.txt](requirements-general.txt): Python dependencies.
- code/: main Python scripts:
  - [code/training.py](code/training.py)
  - [code/inferance.py](code/inferance.py)
  - [code/dashboard.py](code/dashboard.py)
  - [code/features.py](code/features.py)
  - [code/shap_analysis.py](code/shap_analysis.py)
- notebooks_experiments/: exploratory notebooks and model selection experiments.
- models/: serialized model artifacts (for example `aqi_xgb_72h.joblib`).
- predictions/: outputs such as `latest_forecast.csv`.
- reports/shap/: SHAP feature importance CSV exports.

## Quickstart

1. Create and activate a Python virtual environment (Windows example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
pip install -r requirements-general.txt
```

3. Typical workflows:

- Train a model (example):

```powershell
python code/training.py
```

- Run inference / produce forecasts:

```powershell
python code/inferance.py
```

- Generate SHAP analyses:

```powershell
python code/shap_analysis.py
```

- Launch the dashboard:

```powershell
python code/dashboard.py
```

Notes: Each script may accept CLI arguments or configuration values. Check the top of each script (for example [code/training.py](code/training.py)) for available options and parameters.

## Notebooks

Open notebooks in `notebooks_experiments/` to review experiments, model selection notes, and exploratory analysis. These complement the decision log in [process.txt](process.txt).

## Models and Outputs

- Trained model files are stored under `models/` (e.g. `models/aqi_xgb_72h.joblib`).
- Generated forecasts are in `predictions/latest_forecast.csv`.
- SHAP exports are under `reports/shap/`.

## Development notes

- The canonical history of design choices, assumptions, and experiments is in [process.txt](process.txt). Before changing model architecture, hyperparameter search, or evaluation protocols, consult `process.txt` to preserve context and rationale.
- Use `project.txt` for high-level project metadata and scope.
