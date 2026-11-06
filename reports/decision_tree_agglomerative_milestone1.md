# Decision Tree Classification and Agglomerative Clustering — Milestone 1

**Student Name:** Seifallah Magdy

**Student ID:** 2301147

**Contribution:** Decision Tree Classification and Agglomerative Clustering.

**GitHub:** seifallahmagdy

**Course:** CSE382 — Introduction to Machine Learning

**Date:** 5 November 2026

**Scope:** an untuned Decision Tree baseline for `IsCanceled` and an untuned Agglomerative clustering baseline,
integrated into the team pipeline. This is a contribution section, not the complete team progress report.

Every number below comes from these two runs on the team's current code and data policy:

```bash
python main.py train --task classification --model decision_tree
python main.py cluster --algorithm agglomerative
```

The outputs were generated with Python 3.14.0, scikit-learn 1.9.1, pandas 3.1.0 and NumPy 2.5.4.
Results from my earlier standalone version are **not** reused (see section 9).

## 1. Objectives

- **Decision Tree (Task B):** predict whether a booking is eventually cancelled (`IsCanceled`; positive class 1 =
  cancelled). It follows the team's intended prediction point, booking creation, and the team's caveat that the source
  attributes are generally pre-arrival snapshots.
- **Agglomerative (Task C):** describe booking segments without any outcome. Clusters come from booking attributes
  only. `IsCanceled` is never an input and never a selection criterion.

## 2. Feature selection

**Decision Tree.** It uses the shared classification policy in `src/preprocessing/booking.py`, so the inputs and
exclusions are exactly the same as XGBoost's:

- **Raw inputs (20):** LeadTime, ArrivalDateMonth, StaysInWeekendNights, StaysInWeekNights, Adults, Children, Babies,
  Meal, Country, MarketSegment, DistributionChannel, IsRepeatedGuest, PreviousCancellations,
  PreviousBookingsNotCanceled, ReservedRoomType, Agent, Company, CustomerType, RequiredCarParkingSpaces,
  TotalOfSpecialRequests.
- **Excluded (team policy):** ReservationStatus, ReservationStatusDate, AssignedRoomType, BookingChanges,
  DaysInWaitingList, DepositType, ADR, ArrivalDateYear, ArrivalDateWeekNumber and ArrivalDateDayOfMonth. The target
  is never an input.
- My earlier standalone configuration also used ADR and DepositType. Both are **not** used here.

**Agglomerative.** Five numeric booking descriptors (`config.yaml → clustering.agglomerative.features`): LeadTime,
StaysInWeekendNights, StaysInWeekNights, Adults and ADR.

- **ADR** is a descriptive clustering input only. It is computed from lodging transactions divided by staying nights,
  so it is not a verified booking-time quote. Clusters that separate on ADR describe realised prices.
- **Rejected inputs:** `check_features` raises an error if an outcome column (IsCanceled, ReservationStatus,
  ReservationStatusDate) or a non-H2 column is configured.
- **Description only:** Children, MarketSegment, CustomerType, DepositType, Meal, IsRepeatedGuest and
  TotalOfSpecialRequests are summarised per cluster for interpretation, never used for fitting.

## 3. Preprocessing

**Shared loading.** Both models load through `load_h2()`. It validates the 79,330 × 31 H2 schema, cleans category
padding and `NULL`, and removes 2 Undefined MarketSegment rows, 2 Undefined DistributionChannel rows and 390 rows
with Adults ≤ 0. That leaves **78,936 eligible bookings**. The 25,885 rows that duplicate another full row are
retained, as in the team policy.

**Decision Tree.** It uses the shared `build_preprocessor("classification")`, so no separate pipeline exists. In order:

- Notebook-derived features: TotalGuests, IsFamily, hasCompany, hasAgent and TotalNights.
- Country grouped into the training top-10 plus Other.
- Missing numeric values filled with the training median.
- Missing categorical values filled with the constant `Missing`, then one-hot encoded; categories never seen in
  training are ignored.

This gives 61 encoded features. Everything is fitted on the training rows only (refitted per CV fold).

**Agglomerative.** Its own numeric pipeline, because distances need scaling:

- Median imputation, then `log1p` (after clipping at 0) to shrink the long right tails of LeadTime, the nights and
  ADR, then `StandardScaler`.
- It is fitted on the clustering sample, which is a simple random sample of 8,000 eligible bookings drawn with seed 42
  (fingerprint `93aedda6576ba581`).
- Hierarchical clustering stores all pairwise distances, so it uses O(n²) memory: about 256 MB for 8,000 rows versus
  about 25 GB for all 78,936.

## 4. Baseline configuration

| Model | Configuration |
|---|---|
| Decision Tree | `DecisionTreeClassifier(random_state=42)` with scikit-learn defaults: gini, unlimited depth, `min_samples_split=2`, `min_samples_leaf=1`, no class weights, decision threshold 0.5. No search or pruning. |
| Agglomerative | `AgglomerativeClustering(n_clusters=3, linkage="ward", metric="euclidean")`, the fixed baseline in `config.yaml`. k=3 was set as a starting point, not chosen from results. |

The parameter search (`--search`) and resampling stability check (`--stability`) are implemented for Milestone 2.
They are optional and are not part of this baseline. Both were run once to confirm they work, and their outputs are
not reported here.

## 5. Evaluation method

**Decision Tree.** It uses the team runner (`src/evaluation/xgb_baseline.train_baseline`, with a new `prefix`
argument) with the same seed and the same stratified 70/15/15 split as XGBoost. Its split file is byte-identical to
`outputs/xgb_classification_splits.csv`.

- **Split:** 55,254 train / 11,841 validation / 11,841 reserved test.
- **Fitting:** the saved pipeline is fitted on train only. Five-fold stratified CV uses training rows only and refits
  the preprocessing in every fold.
- **Reported:** validation metrics plus CV mean ± sample standard deviation.
- **Test split:** never predicted or scored (`test_evaluated: false` in the metadata).

**Agglomerative.** Internal metrics only, with no labels:

- Silhouette (Euclidean; higher is better) and Davies–Bouldin (Euclidean; lower is better).
- Both are reported as undefined, instead of crashing, when a clustering has fewer than 2 or more than n−1 labels.
- Cluster sizes are recorded, and the per-cluster profiles use the original values.

## 6. Results

### Decision Tree (validation, and training 5-fold CV)

| Metric | Validation | CV mean ± std |
|---|---:|---:|
| Accuracy | 0.8246 | 0.8218 ± 0.0031 |
| Precision (cancelled) | 0.7925 | 0.7847 ± 0.0047 |
| Recall (cancelled) | 0.7862 | 0.7905 ± 0.0050 |
| F1 (cancelled) | 0.7893 | 0.7876 ± 0.0036 |
| ROC-AUC | 0.8234 | 0.8213 ± 0.0029 |

- **Validation confusion matrix:** TN 5,873, FP 1,019, FN 1,058, TP 3,891. 41.8% of validation bookings are
  cancelled. Pooled across the CV folds: TN 27,150, FP 5,011, FN 4,837, TP 18,256.
- **Tree size:** the fitted tree has depth 50 and 9,153 leaves. Fitting plus evaluation took 4.5 s.
- **Saved outputs:** `outputs/decision_tree_classification_{metrics,cv_folds,confusion_matrices,features,metadata}`
  and `..._confusion_matrix.png`.

**Reading.** CV and validation agree closely, so the estimate is stable for this split policy. The unpruned tree
grows until almost every leaf is pure, so its probabilities are almost all 0 or 1. That is why its ROC-AUC (0.823)
barely exceeds the average of its true-positive and true-negative rates.

For reference under the identical protocol, the committed untuned XGBoost scores validation F1 0.802 and ROC-AUC
0.923. Neither model is tuned, so this is not a final model ranking.

### Agglomerative (sample of 8,000 bookings)

| Metric | Value |
|---|---|
| Silhouette (Euclidean) | 0.288 |
| Davies–Bouldin (Euclidean) | 1.122 |
| Cluster sizes | 6,481 / 1,428 / 91 (smallest 1.1%) |
| Fit time | 3.7 s |

Figures:
- `outputs/figures/agglomerative_dendrogram.png`: the last 30 merges, with the k=3 cut.
- `agglomerative_pca_2d.png`: PC1 29.8% and PC2 20.7% of the variance; a visualisation only.
- `agglomerative_cluster_sizes.png`.

Tables: `outputs/tables/agglomerative_{metrics,baseline_profiles,results,metadata}`.

A silhouette of 0.29 means the clusters overlap: they are regions of a continuum, not cleanly separated groups. The
dendrogram supports this. The k=3 cut falls between merges at heights 101.2 and 102.6, a narrow band, while k=2 would
sit in a much wider gap (102.6 to 118.0).

## 7. Cluster interpretation (observed profiles only)

| Cluster | Share | What distinguishes it (mean / median of the original values) |
|---|---|---|
| 0 | 81.0% | Parties of 1–4 adults (mean 2.06); longest lead time (median 80 days); ADR median 103.5; most weekend nights (mean 0.84). Top segment Online TA (54%). |
| 1 | 17.9% | Exactly 1 adult in every booking; shorter lead time (median 43.5 days); ADR median 91.7; fewer weekend nights (mean 0.56). The most common segment is still Online TA, but only 29% of the cluster. |
| 2 | 1.1% | Zero-rate stays (ADR median 0); very short lead time (median 3 days); median 1 week night and 0 weekend nights. Top segment Complementary (42%); 38% repeated guests. |

**Post-hoc only (not an input or a selection criterion):** cancellation rates are 43.3%, 38.0% and 8.8% for clusters
0, 1 and 2. These describe the segments; they do not validate the clustering. Adults is a small integer with a single
value for every booking in cluster 1, so party size is one of the strongest drivers of this partition.

## 8. Leakage and other limitations

- **Team feature policy, questions to raise.** The shared policy keeps three inputs that my earlier audit of the full
  H2 file flagged as possibly recorded after booking creation. I did not change the team policy; the team should
  decide on these for Milestone 2.
  - `RequiredCarParkingSpaces`: none of the 1,926 bookings with a value above 0 was cancelled.
  - `Country`: PRT bookings cancel 64.9% of the time vs 26.9% for others. The dataset paper notes that country is
    often confirmed at check-in.
  - `TotalOfSpecialRequests`: the cancellation rate falls from 54.9% to 3.8% as the number of requests rises, and
    requests can be added after booking.
- **Duplicates.** 25,885 duplicate rows are retained, and **40.4% of validation bookings have an input vector
  identical to one in the training split.** A fully grown tree can memorise these, so these validation and CV scores
  are probably optimistic. In my earlier standalone experiment, under a different feature set, collapsing duplicates
  lowered a default tree's CV F1 from 0.75 to 0.50. That figure only shows the size of the risk; it is not a
  measurement for this configuration.
- **Agglomerative.** It runs on a 10% sample of the eligible rows, and the duplicates are in the sample population.
  ADR is a realised value (section 2). The baseline k=3 is unvalidated by design.
- **DBSCAN comparison.** DBSCAN uses six different features (including Children and Babies, without ADR), all
  eligible rows and noise exclusion, so its scores **must not be ranked** against Agglomerative.
- **Reproducibility.** On this Windows checkout (`core.autocrlf=true`), H2.csv has CRLF line endings, so the
  `data_sha256` in the Decision Tree metadata (`6796f5…`) differs from the repository content hash (`510709…`) that the
  XGBoost metadata records. The data rows are identical.

## 9. Inference on unseen bookings

- **Decision Tree:** the fitted pipeline is saved locally as
  `artifacts/decision_tree_classification_pipeline.joblib`, which is git-ignored. It can predict new booking records
  that have the 20 raw input columns. A `predict` command and the application are Milestone 2 work.
- **Agglomerative:** **no prediction for unseen bookings.** `AgglomerativeClustering` has no `predict`; its result is
  a merge history of the sampled bookings. The saved outputs are the fitted preprocessor and the sample's labels
  (`artifacts/agglomerative_*`), and the metadata says `predict_supported: false`. Assigning new bookings would need a
  different, separately justified method.

## 10. Relation to my earlier standalone work

My earlier version was developed before the team pipeline existed. It is **not** the source of any result above.

| | Earlier standalone version | This integration |
|---|---|---|
| Population | Exact duplicates removed (53,428 rows) | Team `load_h2()` eligibility, duplicates kept (78,936) |
| Decision Tree inputs | Own audit, including ADR and DepositType | Team classification policy; ADR and DepositType excluded |
| Split and tuning | Own split, randomized search, test set scored | Team split and runner, untuned, test never scored |
| Agglomerative | Same 5 features and preprocessing, search-selected configuration reported | Same 5 features; untuned baseline is the default; search and stability optional |

## 11. My contribution and files changed

| File | Change |
|---|---|
| `src/classification/decision_tree.py` | New: untuned Decision Tree pipeline on the shared preprocessor; trains through the shared runner and records the tree's depth and leaves. |
| `src/evaluation/xgb_baseline.py` | Small change: `prefix` argument (default `"xgb"`) for output names. XGBoost's version stamp and summary are written only for XGBoost. Re-running both XGBoost commands reproduced the committed metrics, CV folds, features, splits and confusion matrices byte for byte. |
| `src/clustering/agglomerative.py` | New: Agglomerative baseline, optional search and stability, profiles, figures, metadata. |
| `src/evaluation/clustering.py` | New: silhouette / Davies–Bouldin that return None when undefined. |
| `main.py` | Added `--model decision_tree` and `cluster --algorithm agglomerative`; the XGBoost routes are unchanged. |
| `config.yaml` | Added `classification.decision_tree` and `clustering.agglomerative`; all existing settings unchanged. |
| `tests/test_decision_tree.py`, `tests/test_agglomerative.py` | 17 new tests, including CLI routing for the existing XGBoost commands. |
| `README.md`, `.gitignore`, this report | Commands and limitations; git-ignore the duplicate split file. |
| `outputs/decision_tree_classification_*`, `outputs/{tables,figures}/agglomerative_*` | Generated by the two commands above. |

**Verification.** The full suite passes: 29 tests (the team's 12 plus these 17). The existing KNN, DBSCAN and both
XGBoost commands were re-run. Their committed result files were reproduced, apart from run-time and environment
stamps, and those regenerated files were restored, not committed.
