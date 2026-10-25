# City Hotel Booking Demand — CSE382 ML Project (Fall 2026)

An end-to-end machine learning system on the **City Hotel (H2)** booking dataset covering:

- **Task A – Regression:** predict `ADR` (Average Daily Rate)
- **Task B – Classification:** predict `IsCanceled`
- **Task C – Clustering:** unsupervised booking/customer segments
- PCA / LDA experiments, hyperparameter tuning, saved inference artifacts, and a CLI/GUI application

**Repository:** https://github.com/Zareia/MachineLearning_Project_CSE382_Fall26
**Submitted commit SHA:** _TBD_

## Team

| Name | Student ID |
|------|------------|
|      |            |
|      |            |
|      |            |
|      |            |

## Dataset

`data/raw/H2.csv`: 79,330 bookings × 31 variables, arrivals between 2015-07-01 and 2017-08-31.

- Source: Nuno Antonio, Ana de Almeida, Luis Nunes. "Hotel booking demand datasets." *Data in Brief* 22 (2019) 41–49. https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/
- Mirror used: https://raw.githubusercontent.com/afaq-ahmad/Hotel-booking-demand-Case-Study/master/H2.csv

Raw quirks: categorical strings are space-padded (e.g. `"HB       "`), and `Agent`/`Company` use the literal text `NULL` for missing values.

**Column names:** the code uses the original source labels (e.g. `LeadTime`, `ADR`). If any module renames them, document the mapping here.

## Setup

**New to Git/GitHub? Follow the step-by-step guide in [CONTRIBUTING.md](CONTRIBUTING.md).**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Python 3.11–3.12 is recommended, since some ML libraries may not ship wheels for the newest Python releases yet.

## Project structure

```
config.yaml          shared settings (seed, paths, split, per-task features)
data/raw/            H2.csv (tracked)
data/processed/      derived data (git-ignored)
src/
  data/              loading + schema validation
  preprocessing/     imputation, encoding, scaling pipelines
  features/          task-specific feature selection / engineering
  regression/        ADR models
  classification/    IsCanceled models
  clustering/        clustering algorithms + interpretation
  dimensionality/    PCA / LDA
  tuning/            hyperparameter search
  evaluation/        metrics, comparison tables, plots
  inference/         load saved artifacts, predict new bookings
  utils/             config, seeding
artifacts/           fitted pipelines/models (git-ignored, regenerate locally)
outputs/figures/     EDA + result figures
outputs/tables/      comparison tables
notebooks/           exploration only (not the main solution)
reports/             milestone + final reports
tests/
```

## Run commands

_To be added once the CLI/GUI entry point exists._

## Milestones

| Milestone | Deadline | Scope |
|-----------|----------|-------|
| 1 | Thu 5 Nov 2026 | EDA, preprocessing pipelines, leakage analysis, one untuned model per task, progress report |
| 2 | Fri 11 Dec 2026 | All models + tuning, PCA/LDA, final app, final report |
| Discussion | 18–24 Dec 2026 | Live demo + individual discussion |
