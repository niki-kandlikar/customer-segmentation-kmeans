# Customer Segmentation with K-Means

Unsupervised clustering project on retail customer data (age, income, spending score, purchase frequency), built to practice the full pipeline: cleaning, scaling, choosing k, and turning clusters into something a non-technical stakeholder could actually use.

## What it does

Given raw customer records, the pipeline:

1. Cleans and validates the data
2. Scales features (income is on a totally different scale than the others, so K-Means needs help)
3. Sweeps k = 2–10 and picks the best k using elbow + silhouette score
4. Fits the final model and plots the clusters
5. Summarizes each cluster's centroid in plain terms

## Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Structure
