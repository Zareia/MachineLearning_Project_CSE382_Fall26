"""Load and validate the original City Hotel (H2) dataset without filtering."""

from pathlib import Path

import pandas as pd

from src.utils.config import PROJECT_ROOT, load_config

SOURCE_COLUMNS = (
    "IsCanceled LeadTime ArrivalDateYear ArrivalDateMonth ArrivalDateWeekNumber "
    "ArrivalDateDayOfMonth StaysInWeekendNights StaysInWeekNights Adults Children "
    "Babies Meal Country MarketSegment DistributionChannel IsRepeatedGuest "
    "PreviousCancellations PreviousBookingsNotCanceled ReservedRoomType "
    "AssignedRoomType BookingChanges DepositType Agent Company DaysInWaitingList "
    "CustomerType ADR RequiredCarParkingSpaces TotalOfSpecialRequests "
    "ReservationStatus ReservationStatusDate"
).split()


def load_raw(path=None):
    path = Path(path) if path is not None else PROJECT_ROOT / load_config()["paths"]["raw_data"]
    return pd.read_csv(path, na_values=["NULL"], skipinitialspace=True,
                       dtype={"Agent": str, "Company": str})


def validate_schema(df, expected_rows=79330):
    missing = set(SOURCE_COLUMNS) - set(df.columns)
    unexpected = set(df.columns) - set(SOURCE_COLUMNS)
    if missing or unexpected or len(df.columns) != 31:
        raise ValueError(f"Expected original H2 schema; missing={sorted(missing)}, "
                         f"unexpected={sorted(unexpected)}")
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows:,} H2 rows, received {len(df):,}.")
    if df["IsCanceled"].isna().any() or not df["IsCanceled"].isin([0, 1]).all():
        raise ValueError("IsCanceled must be binary with no missing labels.")
