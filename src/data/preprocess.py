# src/data/preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset(path="Data/EEW_features_2024-10-21.csv", target_col="pga_raw"):
    df = pd.read_csv(path)
    # This assumes `pga_raw` is a column; if not, change to your label column.
    # Fillna strategy (simple)
    df = df.dropna(subset=[target_col])  # require a target
    df = df.fillna(df.median(numeric_only=True))
    return df

def prepare_X_y(df, target_col="pga_raw", log_transform=True, features=None):
    if features is None:
        # numeric features only except target
        features = [c for c in df.select_dtypes(include=["number"]).columns if c != target_col]
    X = df[features].values
    y = df[target_col].values
    if log_transform:
        y = np.log1p(y)
    return X, y, features

def train_val_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
