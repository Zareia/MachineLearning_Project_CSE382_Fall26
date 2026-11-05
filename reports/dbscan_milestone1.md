# DBSCAN — Milestone 1
Contributor: Mohamed Tamer

## Objective and Features
Discover booking segments using LeadTime, StaysInWeekendNights,
StaysInWeekNights, Adults, Children, and Babies.

These features describe booking lead time, stay length, and party
composition. They are treated as booking-time attributes for this
preliminary analysis. This assumption must be reviewed because the
source records may reflect later booking changes.

## Data Preparation
- Source: H2.csv, 79,330 rows.
- Normalize surrounding whitespace and textual missing-value markers.
- Reproduce the teammate notebook's baseline filtering:
  exclude Undefined MarketSegment, Undefined DistributionChannel,
  and Adults <= 0.
- Retain 78,936 rows; exclude 394 rows.
- These inherited filtering rules require review: Adults = 0 alone
  does not prove a record is invalid, and undefined channel fields
  are not among the selected clustering features.
- Selected features contain no missing, non-finite, or negative values.
- Apply StandardScaler.
- Retain duplicate bookings and unusual feature values.
- Compress identical feature vectors with occurrence counts as
  DBSCAN sample weights, then restore labels to all retained rows.
  This preserves multiplicity; ambiguous border assignments may
  depend on processing order.

## Preliminary Leakage Audit
- IsCanceled, ReservationStatus, and ReservationStatusDate are excluded.
- No outcome labels are used for feature selection or clustering.
- Scaling is fitted on the full retained dataset for this descriptive,
  transductive clustering experiment.
- PCA is used only for visualization.
- No supervised test-set performance is used in this experiment.
- Coordinate the dataset, features, and evaluation protocol with
  the other clustering experiments before final comparisons.

## Baseline Results
- eps: 0.5; min_samples: 5; Euclidean distance.
- Clusters: 186.
- Noise: 700 bookings (0.89%).
- Silhouette: 0.1880 on 5,000 randomly sampled non-noise bookings;
  seed 42; 132 clusters represented.
- Davies–Bouldin: 1.1341 on all non-noise bookings.
- PCA 2D explained variance: approximately 40.01%.

The largest clusters mainly separate exact stay lengths and party
composition. Noise bookings have longer average stays and higher
average child/baby counts. Many small clusters and limited sampled
separation motivate further investigation.

## Reproduction
Run from the repository root on Windows:

    .venv\Scripts\python.exe -m src.clustering.dbscan_baseline

Outputs:
- outputs/tables/dbscan_baseline_script_metrics.csv
- outputs/tables/dbscan_baseline_script_profiles.csv
- outputs/figures/dbscan_baseline_script_pca.png

## Remaining Work
Review cleaning rules and feature assumptions, integrate shared
configuration settings, perform systematic parameter tuning, and
connect the clustering analysis to the final application.