# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(path):
    """
    Load dataset
    """
    df = pd.read_csv(path)
    return df


def encode_data(df):
    """
    Encode categorical columns
    """
    df = df.copy()

    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df


def prepare_data(path):
    """
    Load and encode data
    """
    df = load_data(path)
    df = encode_data(df)

    X = df.drop("G3", axis=1)
    y = df["G3"]

    return X, y