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

```
customer-segmentation-kmeans/
├── data/customer_data.csv     # dataset
├── src/
│   ├── generate_data.py       # synthetic data generator (see note below)
│   ├── preprocessing.py       # cleaning + scaling
│   ├── clustering.py          # elbow/silhouette + model fit
│   └── pipeline.py            # runs the full thing end to end
├── notebooks/analysis.ipynb   # exploratory version
├── results/                   # plots + cluster_summary.csv
└── requirements.txt
```

## Running it

```
git clone https://github.com/niki-kandlikar/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans
pip install -r requirements.txt

python src/generate_data.py
cd src
python pipeline.py
```

## Notes on the data

Right now `generate_data.py` builds a synthetic dataset, which makes the clusters look cleaner than they'd be on messy real-world data. Swapping in a real dataset (UCI's Mall Customers or Online Retail sets) is next on my list — the preprocessing interface should handle either without changes.

## Choosing k

Instead of picking a cluster count by eye, the pipeline checks both inertia (elbow method) and silhouette score across k = 2–10, and takes the k with the best silhouette score.

![Elbow and silhouette plots](results/elbow_silhouette.png)

## Results

On the generated data, k = 5 gets a silhouette score of 0.54:

| Cluster | Age | Income | Spending | Read on it |
|---|---|---|---|---|
| 0 | ~30 | High | Low | Cautious, underspending relative to income |
| 1 | ~24 | Low/Mid | High | Young, impulsive |
| 2 | ~42 | High | High | Best customers |
| 3 | ~57 | High | Low | Conservative savers |
| 4 | ~38 | Low/Mid | High | Budget-conscious regulars |

![Cluster scatter](results/cluster_scatter_k5.png)

Full centroids in `results/cluster_summary.csv`.

## What I'd add next

- Real dataset instead of the synthetic one
- PCA so this scales past 2-3 features
- A quick comparison against DBSCAN, since K-Means assumes spherical clusters and I haven't checked whether that assumption actually holds here
