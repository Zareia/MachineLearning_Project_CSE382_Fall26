# Milestone 1 submission — Thursday, 5 November 2026

## What the team submits tonight

Submit **one ZIP of the team repository at the submitted version**, containing:

- Source code and modular model/preprocessing files.
- Configuration, requirements.txt and setup/run instructions.
- The Milestone 1 team progress report.
- Result tables, figures and supporting analysis.
- A README containing the GitHub repository URL and exact submitted commit SHA.

Do not include fitted/serialized models, the virtual environment, Git internals
or caches. The code and commands must regenerate the artifacts locally.
The report is a progress report, not the complete Milestone 2 project report.

## Your XGBoost contribution

- [x] Untuned XGBRegressor for ADR and XGBClassifier for IsCanceled.
- [x] Python modules and the two required CLI training commands.
- [x] Teammate notebook preprocessing adapted into training-only pipelines.
- [x] Feature exclusions and prediction-point limitations documented.
- [x] Seed 42, train/validation/reserved-test splits and training-only five-fold CV.
- [x] Regression MAE/RMSE/R2; classification accuracy/precision/recall/F1/ROC-AUC and confusion matrices.
- [x] CSV results, confusion-matrix PNG, local fitted pipelines and dependencies.
- [x] Contribution report: [xgboost_milestone1.md](xgboost_milestone1.md).
- [x] Twelve checks and actual runs; saved XGBoost pipelines reproduce validation metrics.

## What still needs team review before submission

- [ ] Combine your report section with the team's EDA, other models, clustering,
  preliminary leakage audit, overall progress and remaining-work plan.
- [ ] Confirm the team member names, student IDs, GitHub accounts and individual contributions.
- [ ] Explain the current feature-timing and split differences honestly. Do not
  declare a model winner or claim a globally common untouched final test set from
  these separate exploratory runs.
- [ ] Review the newer team scripts `src/regression/10_linear_regression.py` and
  `src/tuning/11_linear_regression_tuning.py` before running them for Milestone 1:
  both currently predict and score their test partitions by default. Disable or
  explicitly defer those test-evaluation blocks until final evaluation. The
  XGBoost CLI does not invoke either script; they were merged without altering
  the teammate's implementation or running their test evaluation.
- [ ] Approve the contribution branch, then authorize push/PR separately.
- [ ] Agree the exact team submission revision and put its actual SHA in the ZIP's README.
- [ ] Inspect the ZIP contents and submit through the course's announced channel.
  The provided specification does not identify a submission portal or an exact clock time.

## What can wait until Milestone 2

Three models per task with baseline/tuned comparisons, systematic tuning,
PCA/LDA comparisons, final supervised test evaluation after model choices are
fixed, and the complete saved-model inference/application workflow.

There is no mandatory Milestone 1 discussion; an optional TA discussion may be
requested. Milestone 2 is due 11 December 2026; final discussions are 18–24 December.

Source: the supplied CSE382 project specification, Milestone 1 and Submission
Format and Live Evaluation sections. Submission itself has not been performed.
