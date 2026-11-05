import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# STEP 1: Load the cleaned dataset
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


# STEP 4: Clean categorical values
for column in categorical_features:
    data[column] = data[column].astype(str).str.strip()
    data[column] = data[column].replace("NULL", "Unknown")


# STEP 5: Separate input features and target
features = numerical_features + categorical_features

X = data[features]
y = data["ADR"]


# STEP 6: Split the data
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


# STEP 7: Create the preprocessing system
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# STEP 8: Create the Linear Regression model
model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regression", LinearRegression())
])


# STEP 9: Train the model
model.fit(X_train, y_train)

print("LINEAR REGRESSION MODEL TRAINED SUCCESSFULLY!")


# STEP 10: Make predictions
train_predictions = model.predict(X_train)
val_predictions = model.predict(X_val)
# FINAL TEST EVALUATION
test_predictions = model.predict(X_test)

print("\nFINAL TEST PERFORMANCE:")
print("MAE:", round(mean_absolute_error(y_test, test_predictions), 3))
print("MSE:", round(mean_squared_error(y_test, test_predictions), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, test_predictions)), 3))
print("R2 Score:", round(r2_score(y_test, test_predictions), 3))


# STEP 11: Calculate evaluation metrics
def evaluate_model(actual, predicted):

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = mse ** 0.5
    r2 = r2_score(actual, predicted)

    print("MAE:", round(mae, 3))
    print("MSE:", round(mse, 3))
    print("RMSE:", round(rmse, 3))
    print("R2 Score:", round(r2, 3))


# STEP 12: Evaluate training performance
print("\nTRAINING PERFORMANCE:")
evaluate_model(y_train, train_predictions)


# STEP 13: Evaluate validation performance
print("\nVALIDATION PERFORMANCE:")
evaluate_model(y_val, val_predictions)


# STEP 14: Display example predictions
print("\nEXAMPLE PREDICTIONS:")

for i in range(10):
    print(
        "Actual ADR:", round(y_val.iloc[i], 2),
        "| Predicted ADR:", round(val_predictions[i], 2)
    )

# STEP 15: Visualize actual vs predicted ADR

plt.figure(figsize=(8, 6))

plt.scatter(y_val, val_predictions, alpha=0.4)

# Draw the ideal prediction line
plt.plot(
    [y_val.min(), y_val.max()],
    [y_val.min(), y_val.max()],
    linestyle="--"
)

plt.xlabel("Actual ADR")
plt.ylabel("Predicted ADR")
plt.title("Linear Regression: Actual vs Predicted ADR")

plt.grid(True)
plt.show()