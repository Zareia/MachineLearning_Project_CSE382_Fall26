
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# STEP 1: Load dataset
data = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "H2_cleaned.csv")

# STEP 2: Define numerical features
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
    "PreviousBookingsNotCanceled",
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


# STEP 4: Clean categorical features
for column in categorical_features:
    data[column] = data[column].astype(str).str.strip()
    data[column] = data[column].replace("NULL", "Unknown")


# STEP 5: Separate features and target
features = numerical_features + categorical_features

X = data[features]
y = data["ADR"]


# STEP 6: Split dataset
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


# STEP 7: Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# STEP 8: Create Ridge Regression pipeline
ridge_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regression", Ridge())
])


# STEP 9: Define hyperparameters to test
parameters = {
    "regression__alpha": [
        0.1,
        0.3,
        0.5,
        0.7,
        1,
        1.5,
        2,
        3,
        5,
        7,
        10
    ]
}

# STEP 10: Perform 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=ridge_pipeline,
    param_grid=parameters,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)


# STEP 11: Train and tune using training data only
print("STARTING RIDGE REGRESSION TUNING...")

grid_search.fit(X_train, y_train)


# STEP 12: Display best parameters
print("\nBEST HYPERPARAMETERS:")
print(grid_search.best_params_)

print("\nBEST CROSS-VALIDATION MAE:")
print(round(-grid_search.best_score_, 3))


# STEP 13: Evaluate best model on validation data
best_model = grid_search.best_estimator_

val_predictions = best_model.predict(X_val)
# FINAL TEST EVALUATION
test_predictions = best_model.predict(X_test)

print("\nFINAL TEST PERFORMANCE:")
print("MAE:", round(mean_absolute_error(y_test, test_predictions), 3))
print("MSE:", round(mean_squared_error(y_test, test_predictions), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, test_predictions)), 3))
print("R2 Score:", round(r2_score(y_test, test_predictions), 3))

mae = mean_absolute_error(y_val, val_predictions)
mse = mean_squared_error(y_val, val_predictions)
rmse = mse ** 0.5
r2 = r2_score(y_val, val_predictions)

print("\nRIDGE REGRESSION VALIDATION PERFORMANCE:")
print("MAE:", round(mae, 3))
print("MSE:", round(mse, 3))
print("RMSE:", round(rmse, 3))
print("R2 Score:", round(r2, 3))
