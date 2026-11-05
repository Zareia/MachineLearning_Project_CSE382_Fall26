
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# STEP 1: Load the cleaned dataset
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
    "BookingChanges",
    "DepositType",
    "Agent",
    "Company",
]


# STEP 3: Separate X and y
X = data[features]
y = data["ADR"]


# STEP 4: Split into training data and temporary data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)


# STEP 5: Split temporary data into validation and testing
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42
)


# STEP 6: Display dataset sizes
print("DATA SPLITTING COMPLETED!")

print("\nTraining data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Testing data:", X_test.shape)

print("\nTotal bookings:")
print(len(X_train) + len(X_val) + len(X_test))
