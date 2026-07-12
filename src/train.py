# src/train.py

from preprocessing import prepare_data

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import pickle
import numpy as np


# Load data
X, y = prepare_data("../data/student-mat.csv")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
rf = RandomForestRegressor(
    random_state=42
)

# Train model
rf.fit(X_train, y_train)

# Predictions
predictions = rf.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
print("R2 Score:", r2_score(y_test, predictions))

# Save model
with open("../models/student_model.pkl", "wb") as file:
    pickle.dump(rf, file)

print("Model saved successfully.")