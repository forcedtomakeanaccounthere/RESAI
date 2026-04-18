# Healthcare Bias Prediction Web App

This app demonstrates prediction differences for healthcare utilization across multiple bias-mitigation and privacy-aware modeling approaches using the same MEPS HC-181 pipeline from your updated notebook.

## What this app does

- Trains multiple notebook-aligned variants, including:
  - Logistic Regression (original)
  - Random Forest (original)
  - Logistic Regression + Reweighing
  - Logistic Regression + Disparate Impact Remover (if available)
  - Logistic Regression + SMOTE (if imbalanced-learn is installed)
  - Federated and DP-Federated logistic variants (lightweight simulation)
  - ROC-inspired post-processing threshold variant
- Lets you enter customer details using the same feature set used in the notebook.
- Returns side-by-side predictions for baseline and preferred fair model, plus full per-model prediction table.
- Shows expanded fairness and performance metrics, including balanced accuracy, disparate impact, average odds difference, statistical parity difference, equal opportunity difference, Theil index, precision/recall/F1, and ROC AUC.
- Shows signed-residual group diagnostics and trade-off highlights (accuracy/fairness/privacy/explainability).

## Project structure

- webapp/model_service.py: Shared training and inference logic
- webapp/flask_api.py: Flask API endpoints for schema, metrics, and prediction
- webapp/streamlit_app.py: Streamlit website UI
- webapp/requirements.txt: Python dependencies

## Data path

The app expects h181.csv and auto-searches in several locations including:

- workspace root
- parent folders near webapp
- path provided by environment variable MEPS_CSV_PATH

## Setup

1. Open a terminal in RESAI/webapp.
2. Create and activate a virtual environment.
3. Install dependencies.

Example commands (PowerShell):

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

## Run Flask API

python flask_api.py

Default API URL: http://127.0.0.1:5000

## Run Streamlit UI

In a second terminal:

streamlit run streamlit_app.py

Default UI URL: http://localhost:8501

## API endpoints

- GET /health
- GET /schema
- GET /metrics
- POST /predict

Example prediction payload:

{
  "features": {
    "AGELAST": 52,
    "SEX": 2,
    "RACE": 0,
    "MARRY15X": 1,
    "EDUCYR": 12,
    "FTSTU53X": 0,
    "ACTDTY53": 0,
    "HONRDC53": 0,
    "RTHLTH53": 3,
    "MNHLTH53": 2,
    "HIBPDX": 1,
    "CHDDX": 0,
    "ANGIDX": 0,
    "MIDX": 0,
    "OHRTDX": 0,
    "STRKDX": 0,
    "EMPHDX": 0,
    "CHBRON53": 0,
    "CHOLDX": 1,
    "CANCERDX": 0,
    "DIABDX": 1,
    "JTPAIN53": 0,
    "ARTHDX": 0,
    "ARTHTYPE": 0,
    "ASTHDX": 0,
    "ADHDADDX": 0,
    "PREGNT53": 0,
    "WLKLIM53": 0,
    "ACTLIM53": 0,
    "SOCLIM53": 0,
    "COGLIM53": 0,
    "DFHEAR42": 0,
    "DFSEE42": 0,
    "DFCOG42": 0,
    "DFWLKC42": 0,
    "DFDRSB42": 0,
    "DFERND42": 0,
    "INSCOV15": 1,
    "POVCAT15": 3,
    "REGION15": 2
  }
}

## Notes

- First run can take longer because model variants are trained and cached in webapp/models.
- Retraining can be triggered by deleting the cached artifact file.
