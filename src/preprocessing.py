# src/preprocessing.py

import pandas as pd


def load_dataset(path):
    """
    Load the dataset.
    """
    return pd.read_csv(path)


def split_features_target(df):
    """
    Split dataframe into X and y.
    """
    X = df.drop("G3", axis=1)
    y = df["G3"]

    return X, y