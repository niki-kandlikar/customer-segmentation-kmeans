"""
preprocessing.py
-----------------
Data cleaning, feature selection, and feature scaling for the
customer segmentation pipeline.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

def load_data(path="data/customer_data.csv"):
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and obviously invalid rows."""
    df = df.drop_duplicates(subset="CustomerID")
    df = df.dropna(subset=FEATURES)
    for col in FEATURES:
        df = df[df[col] >= 0]
    return df.reset_index(drop=True)

def scale_features(df: pd.DataFrame, features=FEATURES):
    """Standardize features to zero mean / unit variance for K-Means."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled, columns=[f"{c}_scaled" for c in features])
    return scaled_df, scaler

if __name__ == "__main__":
    raw = load_data()
    clean = clean_data(raw)
    scaled_df, scaler = scale_features(clean)
    print(f"Raw rows: {len(raw)} -> Clean rows: {len(clean)}")
    print(scaled_df.describe())