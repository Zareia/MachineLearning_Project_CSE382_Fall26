# XGBoost baselines — H2 only

## Method

Intended prediction point: booking creation. Source attributes generally reflect the day before arrival, so these exploratory scores do not establish leakage-free booking-time performance.

Reuse the current V2 notebook: remove Undefined market/channel rows and Adults <= 0; fill Country with Unknown; derive TotalGuests, IsFamily, hasCompany, hasAgent and TotalNights; use top-10 Country grouping and the notebook's feature exclusions. Country grouping is learned from training only. Missing numeric inputs use training medians; categorical inputs use constant Missing then one-hot encoding with unknown-category handling. Raw party/stay/agency columns are replaced by derived features. Trees need no scaling.

Defaults except random_state from config.yaml (42). No search, early stopping or threshold tuning. Configured split: 70/15/15, test reserved first; classification is stratified. Five-fold CV uses only training: shuffled KFold for regression, StratifiedKFold for classification. Report unweighted fold mean and sample standard deviation (ddof=1). Saved pipelines are fitted on training only. Positive class = canceled (1); classification uses the default 0.5 threshold. Test rows are never predicted or scored.

## Leakage and limitations

ADR is calculated from lodging transactions, not established as a booking-time quote; exclude it from classification. DepositType is calculated from payments before arrival/cancellation; exclude it from both tasks. Exclude final status fields and realized post-booking changes. Remaining source attributes still need booking-time availability checks. The inherited row exclusions restrict coverage and do not prove the excluded bookings are errors.

Duplicates are retained to match the notebook. Random row splits/CV can share identical or related bookings and be optimistic. KNN uses a different population, feature set and predictor-group split; these scores cannot rank models or establish a common untouched team test set. Freeze a shared evaluation policy before model comparison and acknowledge previous inspections. No target clipping or performance-driven row deletion is applied.

Source: [António et al., specifications table and Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/).

## Regression

Target: ADR. Split sizes: {'train': 55254, 'validation': 11841, 'test': 11841}.

Raw inputs: LeadTime, ArrivalDateMonth, StaysInWeekendNights, StaysInWeekNights, Adults, Children, Babies, Meal, Country, MarketSegment, DistributionChannel, IsRepeatedGuest, PreviousCancellations, PreviousBookingsNotCanceled, ReservedRoomType, Agent, Company, CustomerType, RequiredCarParkingSpaces, TotalOfSpecialRequests.

Derived features: TotalGuests, IsFamily, hasCompany, hasAgent, TotalNights.

Replaced raw columns: Company, Agent, Children, Babies, StaysInWeekendNights, StaysInWeekNights. Exact estimator inputs are in the companion features CSV.

Dropped predictors:

- ReservationStatus: Final outcome, unavailable when the booking is created.
- ReservationStatusDate: Future date on which the final status was recorded.
- AssignedRoomType: Ultimately assigned room can differ after booking.
- BookingChanges: Counts amendments after booking creation.
- DaysInWaitingList: Realized waiting duration is unknown at booking creation.
- DepositType: Computed from payments before arrival/cancellation, not just an initial deposit policy.
- ArrivalDateYear: Omitted by the notebook's feature selection; not itself a leakage claim.
- ArrivalDateWeekNumber: Omitted by the notebook's feature selection; not itself a leakage claim.
- ArrivalDateDayOfMonth: Omitted by the notebook's feature selection; not itself a leakage claim.
- IsCanceled: Future cancellation outcome, unavailable at booking.
- ADR: target, never a predictor.

| Metric | Validation | Training CV mean +/- std |
|---|---:|---:|
| MAE | 12.3594 | 12.8341 +/- 0.2352 |
| RMSE | 18.8405 | 30.7629 +/- 15.6390 |
| R2 | 0.7659 | 0.4906 +/- 0.3598 |

Cleaning counts: `{"source_rows": 79330, "source_columns": 31, "removed_undefined_MarketSegment": 2, "removed_undefined_DistributionChannel": 2, "removed_no_adult": 390, "retained_rows": 78936, "exact_duplicate_rows_retained": 25885}`.

Versions: `{"numpy": "2.5.4", "pandas": "3.1.1", "scikit-learn": "1.9.1", "matplotlib": "3.11.2", "joblib": "1.6.0", "PyYAML": "6.0.3", "threadpoolctl": "3.7.0", "xgboost": "3.4.1"}`.

## Classification

Target: IsCanceled. Split sizes: {'train': 55254, 'validation': 11841, 'test': 11841}.

Raw inputs: LeadTime, ArrivalDateMonth, StaysInWeekendNights, StaysInWeekNights, Adults, Children, Babies, Meal, Country, MarketSegment, DistributionChannel, IsRepeatedGuest, PreviousCancellations, PreviousBookingsNotCanceled, ReservedRoomType, Agent, Company, CustomerType, RequiredCarParkingSpaces, TotalOfSpecialRequests.

Derived features: TotalGuests, IsFamily, hasCompany, hasAgent, TotalNights.

Replaced raw columns: Company, Agent, Children, Babies, StaysInWeekendNights, StaysInWeekNights. Exact estimator inputs are in the companion features CSV.

Dropped predictors:

- ReservationStatus: Final outcome, unavailable when the booking is created.
- ReservationStatusDate: Future date on which the final status was recorded.
- AssignedRoomType: Ultimately assigned room can differ after booking.
- BookingChanges: Counts amendments after booking creation.
- DaysInWaitingList: Realized waiting duration is unknown at booking creation.
- DepositType: Computed from payments before arrival/cancellation, not just an initial deposit policy.
- ArrivalDateYear: Omitted by the notebook's feature selection; not itself a leakage claim.
- ArrivalDateWeekNumber: Omitted by the notebook's feature selection; not itself a leakage claim.
- ArrivalDateDayOfMonth: Omitted by the notebook's feature selection; not itself a leakage claim.
- ADR: Calculated from lodging transactions/staying nights; not established as an initial quoted price.
- IsCanceled: target, never a predictor.

| Metric | Validation | Training CV mean +/- std |
|---|---:|---:|
| accuracy | 0.8426 | 0.8431 +/- 0.0054 |
| precision | 0.8453 | 0.8410 +/- 0.0067 |
| recall | 0.7630 | 0.7702 +/- 0.0094 |
| F1 | 0.8020 | 0.8040 +/- 0.0071 |
| ROC_AUC | 0.9228 | 0.9256 +/- 0.0034 |
| TN | 6201.0000 | 5759.6000 +/- 30.4352 |
| FP | 691.0000 | 672.6000 +/- 30.5663 |
| FN | 1173.0000 | 1061.2000 +/- 43.7459 |
| TP | 3776.0000 | 3557.4000 +/- 43.2932 |

Cleaning counts: `{"source_rows": 79330, "source_columns": 31, "removed_undefined_MarketSegment": 2, "removed_undefined_DistributionChannel": 2, "removed_no_adult": 390, "retained_rows": 78936, "exact_duplicate_rows_retained": 25885}`.

Versions: `{"numpy": "2.5.4", "pandas": "3.1.1", "scikit-learn": "1.9.1", "matplotlib": "3.11.2", "joblib": "1.6.0", "PyYAML": "6.0.3", "threadpoolctl": "3.7.0", "xgboost": "3.4.1"}`.

## Interpretation (3 lines)

1. Regression validation MAE is 12.36 rate units, RMSE 18.84, and R2 0.766; larger RMSE shows the influence of larger errors.
2. Cancellation validation F1 is 0.802, recall 0.763 and ROC-AUC 0.923 at the default decision threshold.
3. Regression CV RMSE is 30.76 +/- 15.64; unusual target values are retained. These preliminary scores have the timing/duplicate limitations above; final test performance is not known.
