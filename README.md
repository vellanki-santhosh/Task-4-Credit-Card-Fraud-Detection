# Credit Card Fraud Detection — ML Pipeline

This repository contains a self-contained, educational pipeline that demonstrates how to build, evaluate and compare models for credit-card fraud detection on a highly imbalanced dataset.

## Project Overview
- Script: [fraud_detection_pipeline.py](fraud_detection_pipeline.py)
- Output summary: [model_summary.csv](model_summary.csv)

What it does:
- Generates a synthetic dataset modeled after the Kaggle `creditcard.csv` (features `V1`–`V28`, `Time`, `Amount`, `Class`) with ~1% fraud.
- Runs EDA and saves figures (`01_eda.png`, `02_roc_pr_curves.png`, ...).
- Engineers 6 features (log amount, hour, night indicator, two interactions, high-amount flag).
- Handles class imbalance using `SMOTE`, random over-sampling and under-sampling.
- Trains three models: Logistic Regression, Random Forest, HistGradientBoosting.
- Uses stratified 5-fold CV (SMOTE inside folds) and evaluates with AUC-PR, AUC-ROC, F1, precision, recall.
- Performs threshold tuning (F1-optimal and cost-sensitive) and produces confusion matrices and feature importances.

## Key Files
- fraud_detection_pipeline.py — core script that generates data, trains models, and writes outputs.
- model_summary.csv — CSV report of CV and test metrics (already produced in this workspace).

## Requirements
- Python 3.8+ (the script uses modern sklearn APIs)
- Packages: numpy, pandas, matplotlib, seaborn, scikit-learn, imbalanced-learn

Install quickly with pip:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

## Running
1. From the repository root run:

```bash
python fraud_detection_pipeline.py
```

2. By default the script writes outputs to `/mnt/user-data/outputs/` (variable `OUT` at top of the script). If you run on Windows, update the `OUT` path inside `fraud_detection_pipeline.py` to a writable directory (for example `./outputs/`) before running.

Generated artifacts (examples):
- `01_eda.png` — exploratory plots and class imbalance
- `02_roc_pr_curves.png` — ROC and Precision-Recall curves
- `03_threshold_tuning.png` — threshold selection plots
- `04_confusion_matrices.png` — confusion matrices at selected threshold
- `06_feature_importance.png` — Random Forest importances
- `model_summary.csv` — final numeric summary

## Notes on Results (from model_summary.csv)
The included `model_summary.csv` contains the final metrics:

| Model | Test AUC-PR | Test AUC-ROC | Test F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8004 | 0.9903 | 0.4058 | 0.2642 | 0.8750 |
| Random Forest       | 0.4272 | 0.9730 | 0.4783 | 0.3667 | 0.6875 |
| Hist Grad Boosting  | 0.5616 | 0.9595 | 0.6047 | 0.4815 | 0.8125 |

- The pipeline emphasizes AUC-PR (average precision) for rare-event detection — it is the most relevant metric when positives are rare.
- Logistic Regression achieved the highest AUC-PR on the test set here, but `HistGradientBoosting` shows higher F1 at the default 0.5 threshold — indicating a trade-off between ranking quality (AP) and thresholded F1.

## Recommendations & Next Steps
- If you plan to run locally on Windows, change `OUT` to a local folder (e.g. `OUT = "./outputs/"`) and create that folder.
- Add a small CLI wrapper or `argparse` flags to set `OUT`, random seed, or choose a subset of steps (EDA, train, eval).
- Persist the best model (joblib/pickle) and add a saved inference example.
- Replace synthetic data with real transaction data and add proper data ingestion safeguards.
- Add unit/integration tests and a `requirements.txt` or `pyproject.toml` for reproducibility.

## License
This repository is an educational example; adapt freely for experimentation.

---
Created by analysis of the repository files.
# Task-4-Credit-Card-Fraud-Detection
