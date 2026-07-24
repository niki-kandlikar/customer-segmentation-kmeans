"""
clustering.py
--------------
Model selection (elbow method + silhouette score) and K-Means fitting.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def evaluate_k_range(X, k_range=range(2, 11), random_state=42):
    """Return inertia (elbow) and silhouette score for each k."""
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    return pd.DataFrame({
        "k": list(k_range),
        "inertia": inertias,
        "silhouette_score": silhouettes,
    })


def best_k_by_silhouette(eval_df: pd.DataFrame) -> int:
    return int(eval_df.loc[eval_df["silhouette_score"].idxmax(), "k"])


def fit_kmeans(X, k, random_state=42):
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    return km, labels
