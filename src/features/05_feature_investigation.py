
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# STEP 1: Load the cleaned dataset
data = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "H2_cleaned.csv")

# STEP 2: Display all column names
print("DATASET COLUMNS:")
print(data.columns.tolist())


# STEP 3: Display data types
print("\nDATA TYPES:")
print(data.dtypes)


# STEP 4: Count unique values in each column
print("\nNUMBER OF UNIQUE VALUES:")
print(data.nunique())


# STEP 5: Display example values from each column
print("\nEXAMPLE VALUES:")

for column in data.columns:
    print("\n", column)
    print(data[column].unique()[:10])


# STEP 6: Display dataset dimensions
print("\nDATASET DIMENSIONS:")
print(data.shape)
