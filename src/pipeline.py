"""
pipeline.py
-----------
End-to-end run: load -> clean -> scale -> select k -> fit -> evaluate ->
visualize -> translate clusters into business insights.

Run with:  python src/pipeline.py
Outputs land in results/
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from preprocessing import load_data, clean_data, scale_features, FEATURES
from clustering import evaluate_k_range, best_k_by_silhouette, fit_kmeans

sns.set_style("whitegrid")
RESULTS_DIR = "results"


def plot_elbow_and_silhouette(eval_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(eval_df["k"], eval_df["inertia"], marker="o", color="#2E86AB")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia (WCSS)")

    axes[1].plot(eval_df["k"], eval_df["silhouette_score"], marker="o", color="#A23B72")
    axes[1].set_title("Silhouette Score by k")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/elbow_silhouette.png", dpi=150)
    plt.close()


def plot_clusters(df, best_k):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    sns.scatterplot(
        data=df, x="Annual Income (k$)", y="Spending Score (1-100)",
        hue="Cluster", palette="viridis", s=60, ax=axes[0]
    )
    axes[0].set_title("Clusters: Income vs Spending Score")
    axes[0].set_xlabel("Annual Income (k$)")
    axes[0].set_ylabel("Spending Score")

    sns.scatterplot(
        data=df, x="Age", y="Spending Score (1-100)",
        hue="Cluster", palette="viridis", s=60, ax=axes[1]
    )
    axes[1].set_title("Clusters: Age vs Spending Score")
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Spending Score")

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/cluster_scatter_k{best_k}.png", dpi=150)
    plt.close()


def summarize_clusters(df, features):
    summary = df.groupby("Cluster")[features].mean().round(1)
    summary["count"] = df.groupby("Cluster").size()
    return summary


def label_segments(summary: pd.DataFrame) -> dict:
    """Simple rule-based business labeling from cluster centroids."""
    labels = {}
    income_med = summary["Annual Income (k$)"].median()
    spend_med = summary["Spending Score (1-100)"].median()
    age_med = summary["Age"].median()

    for cluster_id, row in summary.iterrows():
        income_tag = "High Income" if row["Annual Income (k$)"] >= income_med else "Low/Mid Income"
        spend_tag = "High Spender" if row["Spending Score (1-100)"] >= spend_med else "Low Spender"
        age_tag = "Older" if row["Age"] >= age_med else "Younger"
