# City Hotel Booking Demand — CSE382 ML Project (Fall 2026)

An end-to-end machine learning system on the **City Hotel (H2)** booking dataset covering:

- **Task A – Regression:** predict `ADR` (Average Daily Rate)
- **Task B – Classification:** predict `IsCanceled`
- **Task C – Clustering:** unsupervised booking/customer segments
- PCA / LDA experiments, hyperparameter tuning, saved inference artifacts, and a CLI/GUI application

## Dataset

`data/raw/H2.csv`: 79,330 bookings × 31 variables, arrivals between 2015-07-01 and 2017-08-31.

- Source: Nuno Antonio, Ana de Almeida, Luis Nunes. "Hotel booking demand datasets." *Data in Brief* 22 (2019) 41–49. https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/
- Mirror used: https://raw.githubusercontent.com/afaq-ahmad/Hotel-booking-demand-Case-Study/master/H2.csv

**Column names:** the code uses the original source labels (e.g. `LeadTime`, `ADR`). If any module renames them, document the mapping here.

## Setup

**New to Git/GitHub? Follow the step-by-step guide in [CONTRIBUTING.md](CONTRIBUTING.md).**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11–3.12 is recommended, since some ML libraries may not ship wheels for the newest Python releases yet.

## Project structure

```
config.yaml          shared settings (seed, paths, split, per-task features)
data/raw/            H2.csv (tracked)
data/processed/      derived data (git-ignored)
src/
  data/              data loading
  preprocessing/     imputation, encoding, scaling pipelines
  features/          task-specific feature selection / engineering
  regression/        ADR models
  classification/    IsCanceled models
  clustering/        clustering algorithms + interpretation
  dimensionality/    PCA / LDA
  tuning/            hyperparameter search
  evaluation/        metrics, comparison tables, plots
  inference/         load saved artifacts, predict new bookings
  utils/             shared helpers
artifacts/           fitted pipelines/models (git-ignored, regenerate locally)
outputs/figures/     EDA + result figures
outputs/tables/      comparison tables
notebooks/           exploration only (not the main solution)
reports/             milestone + final reports
tests/
```

## Run commands

XGBoost baselines (from the repository root, after activating the environment):

```bash
python main.py train --task regression --model xgboost
python main.py train --task classification --model xgboost
python -m pytest -q
```

These commands use the team's data/configuration helpers and the seed/split
settings in `config.yaml`. The existing KNN feature settings are preserved.
XGBoost preprocessing is adapted from the current teammate V2 notebook, with
country grouping, imputation and encoding fitted on training data only.
No hyperparameter search or final test evaluation is performed.

Results are saved under `outputs/xgb_*`; fitted pipelines are generated under
`artifacts/` and are excluded from Git and the submission ZIP. Use `--data`,
`--outputs` and `--artifacts` to override paths.

- [XGBoost contribution report](reports/xgboost_milestone1.md)
- [Generated metrics summary](outputs/xgb_baseline_summary.md)
- [Tonight's submission checklist](reports/submission_checklist.md)

KNN and XGBoost currently have different feature sets, populations and split
policies. Their preliminary scores must not be used as a controlled model ranking.
Source attributes are generally pre-arrival snapshots rather than verified
booking-creation histories; the report documents this timing limitation.

KNN regression baseline (PowerShell, from the repository root):

```powershell
.\.venv\Scripts\python.exe -m src.regression.knn_baseline
```

Reports validation MAE, RMSE and R2 alongside a training-mean reference. The final
test set is reserved. Settings are in `config.yaml`; assumptions, leakage notes,
split policy and output descriptions are in
[KNN baseline notes](reports/knn_baseline_notes.md).
The complete multi-task application remains to be implemented.

Training-only data-quality audit (no retraining or record removal):

```powershell
.\.venv\Scripts\python.exe -m src.data.audit_regression
```

Writes `outputs/tables/regression_training_audit.csv` and
`outputs/tables/regression_training_missing.csv`. The KNN notes document the
decisions and the source's pre-arrival snapshot timing limitation.

Plot saved KNN validation predictions and the training ADR distribution:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.plot_knn
```

Writes `outputs/figures/knn_training_adr.png`,
`outputs/figures/knn_validation_diagnostics.png` and
`outputs/tables/knn_error_summary.json`. Run the baseline first. Plotting does not
retrain the model or evaluate the reserved test set.

Decision Tree classification and Agglomerative clustering baselines:

```bash
python main.py train --task classification --model decision_tree
python main.py cluster --algorithm agglomerative
```

The Decision Tree is an untuned `DecisionTreeClassifier` (scikit-learn defaults,
`random_state` 42) behind the shared `build_preprocessor("classification")`, so it
uses the same inputs, leakage exclusions, stratified 70/15/15 split and
training-only five-fold CV as XGBoost. It runs through the same runner and saves
`outputs/decision_tree_classification_*`. The reserved test rows are never
predicted or scored. Its split file is regenerated locally and git-ignored,
because it is byte-identical to `outputs/xgb_classification_splits.csv`.

Agglomerative clustering uses the team's `load_h2()` eligible rows (duplicates
retained) and clusters a seeded 8,000-booking sample on LeadTime,
StaysInWeekendNights, StaysInWeekNights, Adults and ADR. The default command fits
only the untuned baseline in `config.yaml` (Ward, Euclidean, k=3) and writes
`outputs/tables/agglomerative_*` and `outputs/figures/agglomerative_*`.
`--search` (parameter grid with unsupervised selection) and `--stability`
(ARI across seeded resamples) are optional extensions, not the Milestone 1
baseline. The command uses `--data`, `--outputs` and `--artifacts` like `train`.

Limitations:

- `AgglomerativeClustering` cannot assign unseen bookings. The fitted
  preprocessor and the sample's labels are saved under `artifacts/`, and the
  metadata records `predict_supported: false`.
- ADR is a descriptive clustering input, but it is computed from lodging
  transactions, so it is not a verified booking-time value.
- DBSCAN uses a different feature set and population, so its metrics must not be
  ranked against Agglomerative.

Details and results: [Decision Tree and Agglomerative report](reports/decision_tree_agglomerative_milestone1.md).

## Milestones

| Milestone | Deadline | Scope |
|-----------|----------|-------|
| 1 | Thu 5 Nov 2026 | EDA, preprocessing pipelines, leakage analysis, one untuned model per task, progress report |
| 2 | Fri 11 Dec 2026 | All models + tuning, PCA/LDA, final app, final report |
| Discussion | 18–24 Dec 2026 | Live demo + individual discussion |

## Submission version

Before creating the final team submission ZIP, record its approved source commit:

```bash
git rev-parse HEAD
```

Include that exact SHA and this repository URL in the README inside the submission
ZIP. The contribution branch is still under review; its current HEAD must not be
presented as an already approved team submission. See the submission checklist.
