import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# STEP 1: Load the original dataset
data = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "H2.csv")
print("Original number of rows:", len(data))


# STEP 2: Remove exact duplicate rows
data = data.drop_duplicates()

print("After removing duplicates:", len(data))


# STEP 3: Remove bookings with ADR equal to zero
data = data[data["ADR"] > 0]

print("After removing zero ADR:", len(data))


# STEP 4: Remove the suspicious ADR value of 5400
data = data[data["ADR"] != 5400]

print("After removing ADR 5400:", len(data))


# STEP 5 Remove rows with missing Children values
data = data.dropna(subset=["Children"])

print("After removing missing Children:", len(data))


# STEP 6 Remove bookings with no recorded guests
data = data[
    (data["Adults"] + data["Children"] + data["Babies"]) > 0
]

print("After removing bookings with no guests:", len(data))



# STEP 7: Replace missing Country values with Unknown
data["Country"] = data["Country"].fillna("Unknown")


# STEP 8: Display the final dataset information
print("\nFINAL DATASET INFORMATION")

print("Number of rows:", data.shape[0])
print("Number of columns:", data.shape[1])

print("\nMissing values:")
print(data.isnull().sum())

print("\nADR statistics:")
print(data["ADR"].describe())


# STEP 9: Save the cleaned dataset
data.to_csv(
    PROJECT_ROOT / "data" / "processed" / "H2_cleaned.csv",
    index=False
)