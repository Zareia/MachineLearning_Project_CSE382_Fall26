
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# STEP 1: Load cleaned dataset
data = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "H2_cleaned.csv")

# STEP 2: Define input features
features = [
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "ReservedRoomType",
    "DepositType",
    "Agent",
    "Company",
    "CustomerType"
]

target = "ADR"


# STEP 3: Define target
target = "ADR"


# STEP 4: Create X and y
X = data[features]
y = data[target]


# STEP 5: Display results
print("FEATURE SELECTION COMPLETED!")

print("\nNumber of input features:", X.shape[1])
print("Number of target variables:", 1)

print("\nInput features:")
print(X.columns.tolist())

print("\nTarget variable:")
print(y.name)

print("\nInput dataset dimensions:", X.shape)
print("Target dataset dimensions:", y.shape)
