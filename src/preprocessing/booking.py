"""Adapt notebooks/Logistic regression_V2.ipynb, including cells 50-55.

Fixed eligibility rules are applied before splitting. Learned Country grouping,
imputation and encoding belong inside each fitted training pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data.load import SOURCE_COLUMNS, load_raw, validate_schema

TARGETS = {"regression": "ADR", "classification": "IsCanceled"}
CATEGORICAL = ("ArrivalDateMonth Meal Country MarketSegment DistributionChannel "
               "ReservedRoomType Agent Company CustomerType").split()
ENGINEERED = ["TotalGuests", "IsFamily", "hasCompany", "hasAgent", "TotalNights"]
REPLACED_RAW = ("Company Agent Children Babies StaysInWeekendNights StaysInWeekNights").split()

# Source timing: António et al., specifications table and Table 1.
# https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/
COMMON_DROPS = {
    "ReservationStatus": "Final outcome, unavailable when the booking is created.",
    "ReservationStatusDate": "Future date on which the final status was recorded.",
    "AssignedRoomType": "Ultimately assigned room can differ after booking.",
    "BookingChanges": "Counts amendments after booking creation.",
    "DaysInWaitingList": "Realized waiting duration is unknown at booking creation.",
    "DepositType": "Computed from payments before arrival/cancellation, not just an initial deposit policy.",
    "ArrivalDateYear": "Omitted by the notebook's feature selection; not itself a leakage claim.",
    "ArrivalDateWeekNumber": "Omitted by the notebook's feature selection; not itself a leakage claim.",
    "ArrivalDateDayOfMonth": "Omitted by the notebook's feature selection; not itself a leakage claim.",
}
DROP_REASONS = {
    "regression": {**COMMON_DROPS, "IsCanceled": "Future cancellation outcome, unavailable at booking."},
    "classification": {**COMMON_DROPS, "ADR": "Calculated from lodging transactions/staying nights; not established as an initial quoted price."},
}


def task_features(task):
    """Raw inputs needed either directly or to reproduce notebook features."""
    return [c for c in SOURCE_COLUMNS if c not in DROP_REASONS[task] and c != TARGETS[task]]


def load_h2(path):
    """Only fixed, row-wise notebook eligibility/formatting; no fitted statistics."""
    df = load_raw(path)
    validate_schema(df)
    audit = {"source_rows": len(df), "source_columns": len(df.columns)}
    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].map(clean_category)
    for column in ("MarketSegment", "DistributionChannel"):
        remove = df[column].eq("Undefined").fillna(False)
        audit[f"removed_undefined_{column}"] = int(remove.sum())
        df = df.loc[~remove].copy()  # Notebook cells 8, 13.
    audit["removed_no_adult"] = int((~df["Adults"].gt(0)).sum())
    df = df.loc[df["Adults"].gt(0)].copy()  # Notebook cell 31.
    audit["retained_rows"] = len(df)
    audit["exact_duplicate_rows_retained"] = int(df.duplicated().sum())
    return df, audit


def clean_category(value):
    if pd.isna(value) or str(value).strip() in ("", "NULL"):
        return np.nan
    return str(value).strip()


class BookingFeatures(TransformerMixin, BaseEstimator):
    """Notebook feature engineering and TRAIN-fitted top-10 Country grouping."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        countries = X["Country"].map(clean_category).fillna("Unknown")
        self.top_countries_ = countries.value_counts().nlargest(10).index.tolist()
        return self

    def transform(self, X):
        result = X.copy()
        for column in CATEGORICAL:
            result[column] = result[column].map(clean_category)
        # Cells 23, 33, 35, 50-53; the current notebook already fixes the older typo.
        result["Country"] = result["Country"].fillna("Unknown")
        result["Country"] = result["Country"].where(result["Country"].isin(self.top_countries_), "Other")
        result["TotalGuests"] = result["Adults"] + result["Children"] + result["Babies"]
        result["IsFamily"] = (result["Adults"].gt(0) & (result["Children"] + result["Babies"]).gt(0)).astype(int)
        # NULL agency/company means not applicable in the source, represented by 0.
        result["hasCompany"] = result["Company"].notna().astype(int)
        result["hasAgent"] = result["Agent"].notna().astype(int)
        result["TotalNights"] = result["StaysInWeekendNights"] + result["StaysInWeekNights"]
        return result.loc[:, list(self.feature_names_in_) + ENGINEERED]

    def get_feature_names_out(self, input_features=None):
        return np.append(self.feature_names_in_, ENGINEERED)


def build_preprocessor(task):
    raw = task_features(task)
    numeric = [c for c in raw if c not in CATEGORICAL and c not in REPLACED_RAW] + ENGINEERED
    categorical = [c for c in CATEGORICAL if c in raw and c not in REPLACED_RAW]
    columns = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
        ]), categorical),
    ], remainder="drop")
    # Trees need no scaling. Dense matrices preserve genuine numeric zeros for XGB.
    return Pipeline([("features", BookingFeatures()), ("columns", columns)])
