
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# STEP 1: Load cleaned dataset
data = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "H2_cleaned.csv")

# STEP 2: Investigate parking spaces
print("REQUIRED CAR PARKING SPACES:")
print(data["RequiredCarParkingSpaces"].value_counts().sort_index())


# STEP 3: Investigate special requests
print("\nTOTAL OF SPECIAL REQUESTS:")
print(data["TotalOfSpecialRequests"].value_counts().sort_index())


# STEP 4: Investigate Agent
print("\nAGENT:")
print("Unique values:", data["Agent"].nunique())
print(data["Agent"].value_counts().head(15))
print("NULL text values:", (data["Agent"].str.strip() == "NULL").sum())


# STEP 5: Investigate Company
print("\nCOMPANY:")
print("Unique values:", data["Company"].nunique())
print(data["Company"].value_counts().head(15))
print("NULL text values:", (data["Company"].str.strip() == "NULL").sum())


# STEP 6: Investigate waiting list
print("\nDAYS IN WAITING LIST:")
print(data["DaysInWaitingList"].describe())


# STEP 7: Investigate booking changes
print("\nBOOKING CHANGES:")
print(data["BookingChanges"].value_counts().sort_index())


print("\nFEATURE INVESTIGATION COMPLETED!")

