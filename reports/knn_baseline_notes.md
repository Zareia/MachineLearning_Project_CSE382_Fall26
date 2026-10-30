# KNN regression baseline: working notes

## Scope and prediction point

Milestone 1 experiment: predict ADR from booking details assumed available when
the booking is created. This is a provisional experiment, pending the team's
shared feature and split review, not a completed Milestone 1 report.
The original article states that attributes were generally extracted relative to
the day before arrival, using change logs where available. This is not a snapshot
at booking creation. Stay length, guest counts, meal, arrival date and reserved
room may differ from their original values. Accordingly this experiment DOES NOT
establish leakage-free performance at booking creation. Before making a deployment
claim, the team must agree a defensible prediction point (for example a pre-arrival
ADR estimate with feature-by-feature availability checks) or obtain original
booking-time histories. These limitations cannot be fixed by merely renaming the
prediction point. ADR is calculated from lodging transactions, not necessarily the
initial quoted price. See the original article's specifications table, Section 1,
and Table 1: https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/

## Features and leakage

The exact input list and categorical list are in config.yaml. Inputs describe
booking lead time, arrival timing, requested stay, guest counts, meal, booking
channels, and reserved room. Codes are one-hot encoded, not ordered numbers.
ADR is the target only; IsCanceled, ReservationStatus, ReservationStatusDate,
AssignedRoomType, and BookingChanges are excluded because they contain outcomes
or later information. Remaining unused fields are omitted to keep this first
experiment compact; omission does not by itself imply leakage.

All string trimming and NULL conversion are stateless. Median imputation and
standardization for numbers, most-frequent imputation for categories, and one-hot
encoding are fitted inside the pipeline on training data only. Unknown validation
categories are ignored by the encoder. KNN uses k=5, uniform weighting, Euclidean
distance; these settings are fixed, not selected by a search.

## Split and duplicates

Repeated rows are retained because the CSV has no unique booking identifier to
prove they are erroneous records. Identical cleaned predictor rows are assigned
to the same split, even if their ADR or outcomes differ. This prevents identical
inputs from appearing across train, validation, and test; it deliberately tests
generalization to distinct input combinations. Approximately 70/15/15 percent of
predictor groups are split with seed 42, so row proportions can differ.
The split uses no target values. This is a random grouped holdout, not a temporal
forecast evaluation; related customers and near duplicates may still cross splits.
The team should reuse this policy across regression models or agree a replacement
before comparing results. Saved row assignments depend on this exact CSV order
and input list; changing either changes the experiment.

No ADR-based filtering is performed. Zero and unusually high ADR values remain;
their meaning requires EDA. Test rows are reserved but never passed to fit,
predict, or metrics. Validation metrics alone are produced for Milestone 1.

## Run and outputs

From the repository root on Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m src.regression.knn_baseline
```

- outputs/tables/knn_baseline_validation.csv: validation MAE, RMSE, R2 for KNN
  and a reference that always predicts the training mean.
- outputs/tables/knn_baseline_run.json: settings, split sizes, timing, results.
- data/processed/knn_split_assignments.csv: source row positions and split labels.
- data/processed/knn_validation_predictions.csv: validation predictions.
- artifacts/knn_baseline.joblib: fitted preprocessing and KNN together.

Artifacts and processed data are ignored by Git. Regenerate with the command;
do not include fitted models in the submission ZIP. Loading the model does not
require retraining, but inputs must first use select_features with the matching
configuration. Each rerun overwrites this baseline's output files.

## Remaining work

Review the assumptions with the team; complete EDA, unusual-value decisions,
plots, and the broader leakage audit; interpret the actual baseline results;
then perform additional models, tuning and PCA comparisons in Milestone 2.

## Training-only unusual-record audit (30 October 2026)

Run `.\.venv\Scripts\python.exe -m src.data.audit_regression` from the project
root. Counts refer ONLY to the existing 56,007 training rows. The audit does not
fit a model, alter splits, inspect validation/test targets or change raw data.

| Finding | Training count | Decision and evidence |
| --- | ---: | --- |
| Missing Children | 4 | Keep existing training-median imputation (median 0); do not claim these are known child-free bookings. |
| Zero total guests | 117 | Retain and flag. No identifier or operational documentation establishes whether these are errors or special records. Use min_count=3 when summing guests so missing counts are not mistaken for zero. |
| Zero total nights | 240 | Retain and flag; all have ADR 0. ADR's published definition divides lodging transactions by staying nights, so a zero-night rate is not interpretable under that formula without an operational convention. No claim that these are ordinary stays. |
| Zero ADR | 880 | Retain. 360 have the dataset's Complementary market-segment label; this is consistent with some free bookings but does not explain every zero or prove its cause. Do not blanket-delete zero targets. |
| ADR above 500 | 1 | Value 5,400; next-highest training ADR is 375.50. Retain and flag. Table 7 of the original article reports the same H2 maximum, ruling out a new discrepancy in this copy but NOT proving validity. Do not invent a corrected rate. |
| Babies above 3 | 1 | A booking has 9 babies and 1 adult. Retain and flag; suspicious is not verified invalid. |

The cutoffs 500 and 3 are exploratory review flags only, not learned thresholds,
deletion rules, or domain-validity boundaries. Counts overlap and must not be
summed as distinct bookings. The generated CSV contains counts and ADR summaries;
training-only examples are saved under data/processed/ (Git-ignored).

Booking outcome fields were consulted descriptively during manual training-data
inspection only. They do not enter predictors, splitting, corrections or removal
rules. No canceled bookings are removed simply because they were canceled.

Decision: preserve the original all-booking baseline and its existing metrics.
There is insufficient evidence to correct or delete these values. The retained
extreme ADR may affect neighbor averages and the training-mean reference. This
limitation should accompany reported results. A later, explicitly scoped study
of positive-night/positive-guest stays would answer a narrower question and must
report exclusions and coverage under a frozen policy, not be substituted merely
because it produces better validation scores. A training-only sensitivity study
can be planned before any subsequent model selection.

Also note the article's definition of Agent/Company NULL as not applicable, rather
than unknown: preserve that meaning if the team later includes those columns.
Neither is an input to this baseline.

## Baseline figures and interpretation

Run `.\.venv\Scripts\python.exe -m src.evaluation.plot_knn` after running the
baseline. It reads the saved validation predictions and training split; no
refitting or test evaluation occurs. The ADR distribution shows the full training
range (logarithmic count axis) alongside a detail view through the training 99th
percentile, ADR 229. The 553 higher-rate training rows are omitted from the detail
view only, not from fitting or the full-range plot.

The validation diagnostic figure includes actual versus predicted ADR, signed
errors versus predicted ADR, the full error histogram and comparison with the
training-mean reference. Error is defined as predicted minus actual: a positive
error is an overestimate. In the existing baseline run, MAE is 15.27, RMSE is
24.44 and R2 is 0.610. Median absolute error is 9.65; the empirical 95th percentile
of absolute error is 49.76. The maximum absolute validation error is 391.00.
Average signed error is -0.54, which can hide offsetting large errors and must
not be confused with MAE. These are descriptive validation results, not a future
performance guarantee or a prediction interval.

The dense region follows the perfect-prediction line, while several zero-rate
bookings are overestimated and some high-rate bookings are underestimated. KNN's
averaging can smooth unusual cases; these plots alone do not establish causes.
No settings or cleaning rules were changed in response to the figures. Both plots
are preliminary report figures, not evidence that all Milestone 1 requirements
are complete.
