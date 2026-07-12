import os
import pickle
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================
# Load Dataset
# ============================================

DATA_PATH = "../data/student-mat.csv"

df = pd.read_csv(DATA_PATH, sep=";")

print("Dataset Loaded Successfully")
print(df.head())

# ============================================
# Selected Features
# ============================================

selected_features = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "internet",
    "higher",
    "famrel",
    "absences"
]

X = df[selected_features]

y = df["G3"]

# ============================================
# Find categorical and numerical columns
# ============================================

categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()

numeric_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

print("\nCategorical Columns:")
print(categorical_columns)

print("\nNumerical Columns:")
print(numeric_columns)

# ============================================
# Preprocessing
# ============================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numeric_columns
        )
    ]
)

# ============================================
# Model
# ============================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# ============================================
# Pipeline
# ============================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# ============================================
# Train Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Model...")

pipeline.fit(X_train, y_train)

print("Training Complete!")

# ============================================
# Prediction
# ============================================

predictions = pipeline.predict(X_test)

# ============================================
# Evaluation
# ============================================

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)

print("\n===============================")
print("MODEL PERFORMANCE")
print("===============================")

print(f"MAE       : {mae:.4f}")
print(f"MSE       : {mse:.4f}")
print(f"RMSE      : {rmse:.4f}")
print(f"R2 Score  : {r2:.4f}")

# ============================================
# Save Pipeline
# ============================================

os.makedirs("../models", exist_ok=True)

MODEL_PATH = "../models/student_pipeline.pkl"

with open(MODEL_PATH, "wb") as file:
    pickle.dump(pipeline, file)

print("\nPipeline Saved Successfully!")
print("Saved To:", MODEL_PATH)