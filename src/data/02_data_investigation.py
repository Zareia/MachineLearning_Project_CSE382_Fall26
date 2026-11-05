
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Load the dataset
data = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "H2.csv")

# PART 1: Investigate duplicate rows

print("\n========== DUPLICATE ROWS ==========")

duplicates = data[data.duplicated(keep=False)]

print("Number of rows involved in duplicates:", len(duplicates))

print("\nExamples of duplicate rows:")
print(duplicates.head(10))


# PART 2: Investigate bookings with zero ADR

print("\n========== ZERO ADR BOOKINGS ==========")

zero_adr = data[data["ADR"] == 0]

print("Number of zero ADR bookings:", len(zero_adr))

print("\nExamples:")
print(zero_adr[["ADR", "Adults", "Children", "Babies",
                "StaysInWeekendNights", "StaysInWeekNights",
                "CustomerType"]].head(10))


# PART 3: Investigate unusually high ADR

print("\n========== HIGHEST ADR BOOKINGS ==========")

highest_adr = data.sort_values(by="ADR", ascending=False)

print(highest_adr[["ADR", "Adults", "Children", "Babies",
                   "ReservedRoomType", "CustomerType"]].head(10))


# PART 4: Check bookings with no guests

print("\n========== BOOKINGS WITH NO GUESTS ==========")

no_guests = data[
    (data["Adults"] == 0) &
    (data["Children"] == 0) &
    (data["Babies"] == 0)
]

print("Bookings with no guests:", len(no_guests))


# PART 5: Check bookings with zero nights

print("\n========== BOOKINGS WITH ZERO NIGHTS ==========")

zero_nights = data[
    (data["StaysInWeekendNights"] == 0) &
    (data["StaysInWeekNights"] == 0)
]

print("Bookings with zero nights:", len(zero_nights))


# PART 6: Check numerical column ranges

print("\n========== NUMERICAL COLUMN RANGES ==========")

print(data.describe().T[["min", "max", "mean"]])
