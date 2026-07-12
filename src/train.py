# src/train.py

import pickle

from preprocessing import (
    load_dataset,
    split_features_target
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import pandas as pd
import numpy as np

# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv("../data/student-mat.csv", sep=";")

print(df.columns.tolist())
print(df.head())

X, y = split_features_target(df)

# ----------------------------
# Find categorical columns
# ----------------------------

categorical_columns = X.select_dtypes(
    include=["object"]
).columns

numeric_columns = X.select_dtypes(
    exclude=["object"]
).columns

# ----------------------------
# Preprocessing
# ----------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "num",
            "passthrough",
            numeric_columns
        )
    ]
)

# ----------------------------
# Pipeline
# ----------------------------

pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        RandomForestRegressor(
            random_state=42,
            n_estimators=200
        )
    )
])

# ----------------------------
# Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# Train
# ----------------------------

pipeline.fit(
    X_train,
    y_train
)

# ----------------------------
# Predict
# ----------------------------

predictions = pipeline.predict(
    X_test
)

# ----------------------------
# Evaluation
# ----------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

print()

print("=" * 40)

print("MODEL PERFORMANCE")

print("=" * 40)

print(f"MAE  : {mae:.2f}")

print(f"RMSE : {rmse:.2f}")

print(f"R2   : {r2:.4f}")

print("=" * 40)

# ----------------------------
# Save Pipeline
# ----------------------------

with open(
    "../models/student_pipeline.pkl",
    "wb"
) as file:

    pickle.dump(
        pipeline,
        file
    )

print()

print("Pipeline saved successfully!")