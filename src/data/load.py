from pathlib import Path

import pandas as pd

from src.utils.config import PROJECT_ROOT

# Exact source labels from the dataset paper (spec section 4).
SOURCE_COLUMNS = [
    "IsCanceled", "LeadTime", "ArrivalDateYear", "ArrivalDateMonth",
    "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth", "StaysInWeekendNights",
    "StaysInWeekNights", "Adults", "Children", "Babies", "Meal", "Country",
    "MarketSegment", "DistributionChannel", "IsRepeatedGuest",
    "PreviousCancellations", "PreviousBookingsNotCanceled", "ReservedRoomType",
    "AssignedRoomType", "BookingChanges", "DepositType", "Agent", "Company",
    "DaysInWaitingList", "CustomerType", "ADR", "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests", "ReservationStatus", "ReservationStatusDate",
]
EXPECTED_ROWS = 79_330


def load_raw(path: str | Path = PROJECT_ROOT / "data/raw/H2.csv") -> pd.DataFrame:
    """Load H2.csv as-is. Note: string fields are space-padded and Agent/Company use the text 'NULL'."""
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> None:
    if list(df.columns) != SOURCE_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, got {len(df)}")
