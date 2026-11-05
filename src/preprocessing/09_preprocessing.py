
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# STEP 1: Load the cleaned dataset
data = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "H2_cleaned.csv")# STEP 2: Define numerical features
numerical_features = [
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled"
]

# STEP 3: Define categorical features
categorical_features = [
    "ArrivalDateMonth",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "ReservedRoomType",
    "DepositType",
    "Agent",
    "Company",
    "CustomerType"
]

# STEP 4: Clean categorical values
for column in categorical_features:
    data[column] = data[column].astype(str).str.strip()
    data[column] = data[column].replace("NULL", "Unknown")

# STEP 5: Separate input features and target
features = numerical_features + categorical_features

X = data[features]
y = data["ADR"]

# STEP 6: Split the dataset
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42
)

# STEP 7: Create the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# STEP 8: Fit preprocessing using training data only
X_train_processed = preprocessor.fit_transform(X_train)
feature_names = preprocessor.get_feature_names_out()

print("\nALL PROCESSED FEATURE NAMES:")
for i, name in enumerate(feature_names, start=1):
    print(f"{i}. {name}")

# Save all feature names to a CSV file
pd.Series(feature_names, name="Feature Name").to_csv(
    PROJECT_ROOT / "outputs" / "tables" / "processed_feature_names.csv",
    index=False
)

# STEP 9: Transform validation and testing data
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# STEP 10: Display results
print("PREPROCESSING COMPLETED!")

print("\nOriginal training dimensions:", X_train.shape)
print("Processed training dimensions:", X_train_processed.shape)

print("\nOriginal validation dimensions:", X_val.shape)
print("Processed validation dimensions:", X_val_processed.shape)

print("\nOriginal testing dimensions:", X_test.shape)
print("Processed testing dimensions:", X_test_processed.shape)

print("\nNumber of numerical features:", len(numerical_features))
print("Number of categorical features:", len(categorical_features))

print("\nFirst 10 processed feature names:")
print(preprocessor.get_feature_names_out()[:10])
