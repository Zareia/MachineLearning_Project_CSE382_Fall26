
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Load dataset
data = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "H2.csv")


# PART 1: Investigate the highest ADR booking

print("\n========== ADR = 5400 ==========")

print(data[data["ADR"] == 5400].T)


# PART 2: Investigate other high ADR bookings

print("\n========== ADR ABOVE 300 ==========")

high_adr = data[data["ADR"] > 300]

print("Number of bookings above 300:", len(high_adr))

print(high_adr[["ADR", "Adults", "Children", "Babies",
                "StaysInWeekendNights", "StaysInWeekNights"]])


# PART 3: Investigate bookings with no guests

print("\n========== NO GUEST BOOKINGS ==========")

no_guests = data[
    (data["Adults"] == 0) &
    (data["Children"] == 0) &
    (data["Babies"] == 0)
]

print(no_guests.head(20).to_string())


# PART 4: Calculate dataset size after removing exact duplicates

print("\n========== DUPLICATE CHECK ==========")

clean_duplicates = data.drop_duplicates()

print("Original rows:", len(data))
print("Rows after removing duplicates:", len(clean_duplicates))
print("Removed rows:", len(data) - len(clean_duplicates))


# PART 5: Check ADR percentiles

print("\n========== ADR PERCENTILES ==========")

print(data["ADR"].quantile([0.50, 0.75, 0.90, 0.95, 0.99, 0.999]))
