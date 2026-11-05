import pandas as pd
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# STEP 1: Load the dataset
data = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "H2.csv")

# STEP 2: Display the first 5 rows
print("\nFIRST 5 ROWS:")
print(data.head())

# STEP 3: Display the dataset size
print("\nDATASET SIZE:")
print("Number of rows:", data.shape[0])
print("Number of columns:", data.shape[1])

# STEP 4: Display column names and data types
print("\nCOLUMN INFORMATION:")
data.info()

# STEP 5: Check missing values
print("\nMISSING VALUES:")
print(data.isnull().sum())

# STEP 6: Check duplicate rows
print("\nDUPLICATE ROWS:")
print(data.duplicated().sum())

# STEP 7: Display statistical information
print("\nSTATISTICAL SUMMARY:")
print(data.describe())

# STEP 8: Check ADR values
print("\nADR INFORMATION:")
print(data["ADR"].describe())

print("\nNEGATIVE ADR VALUES:")
print((data["ADR"] < 0).sum())

print("\nZERO ADR VALUES:")
print((data["ADR"] == 0).sum())
