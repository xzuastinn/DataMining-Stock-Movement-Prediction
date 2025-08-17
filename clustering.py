"""
clustering.py
=============

Loads PCA‑transformed features, performs K‑Means and Agglomerative clustering,
reports silhouette & Davies‑Bouldin scores, and saves the data with a
`cluster_label` column to `data/cluster.parquet`.

Call ``from clustering import run`` inside pipeline.ipynb.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
IN_PATH = DATA_DIR / "pca_features.parquet"
OUT_PATH = DATA_DIR / "cluster.parquet"
PLOT_DIR = SCRIPT_DIR / "figures"
PLOT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------------
def choose_k_elbow(X, k_max: int = 15):
    """Return k at the elbow using inertia curve."""
    inertias = []
    ks = range(2, k_max + 1)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        km.fit(X)
        inertias.append(km.inertia_)

    # crude elbow: point with biggest drop in inertia
    drops = np.diff(inertias)
    k_star = ks[np.argmin(drops) + 1]  # +1 because diff shortens array

    # plot for reference
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=list(ks), y=inertias, marker="o")
    plt.axvline(k_star, color="red", linestyle="--", label=f"chosen k = {k_star}")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for K‑Means")
    plt.legend()
    elbow_path = PLOT_DIR / "kmeans_elbow.png"
    plt.tight_layout()
    plt.savefig(elbow_path, dpi=120)
    plt.close()

    return k_star, elbow_path


def run():
    """Execute clustering pipeline and return metric dict + plot paths."""
    if not IN_PATH.exists():
        raise FileNotFoundError(f"PCA file not found at {IN_PATH}")

    df = pd.read_parquet(IN_PATH)
    X = df.drop(columns=["label"]).values  # keep any label col for later

    if len(X) > 25_000:
        X_elbow = X[np.random.choice(len(X), 25_000, replace=False)]
    else:
        X_elbow = X


    # -----------------------------------------------------------------
    # 1. K‑Means with elbow choice
    # -----------------------------------------------------------------
    k_chosen, elbow_plot = choose_k_elbow(X_elbow, k_max=6)  # k_max trimmed
    kmeans     = KMeans(n_clusters=k_chosen, random_state=RANDOM_STATE, n_init="auto")
    km_labels  = kmeans.fit_predict(X)

    AGG_SAMPLES = 0        # 0 → completely skip Agglomerative
    if AGG_SAMPLES >= 2:   # run only if you actually want it
        if len(X) > AGG_SAMPLES:
            X_agg = X[np.random.choice(len(X), AGG_SAMPLES, replace=False)]
        else:
            X_agg = X
        agg = AgglomerativeClustering(n_clusters=k_chosen, linkage="ward")
        _   = agg.fit_predict(X_agg)       # we don’t use these labels
    #agg = AgglomerativeClustering(n_clusters=k_chosen, linkage="ward")
    #agg_labels = agg.fit_predict(X_agg)

    # -----------------------------------------------------------------
    # 3. Quality metrics on K‑Means clusters (primary algorithm)
    # -----------------------------------------------------------------
    sil = silhouette_score(X, km_labels)
    dbi = davies_bouldin_score(X, km_labels)

    # -----------------------------------------------------------------
    # 4. Scatter plot of first two PCs coloured by K‑Means labels
    # -----------------------------------------------------------------
    scatter_path = PLOT_DIR / "pca_clusters.png"
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        x=df.iloc[:, 0],  # PC1
        y=df.iloc[:, 1],  # PC2
        hue=km_labels,
        palette="viridis",
        s=10,
        linewidth=0,
    )
    plt.title("PC1 vs PC2 – K‑Means Clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=120)
    plt.close()

    # -----------------------------------------------------------------
    # 5. Save enriched data
    # -----------------------------------------------------------------
    df["cluster_label"] = km_labels
    df.to_parquet(OUT_PATH, index=False)

    metrics = {
        "k_chosen": int(k_chosen),
        "silhouette": float(sil),
        "davies_bouldin": float(dbi),
    }
    plots = {
        "elbow_plot": str(elbow_plot),
        "scatter_plot": str(scatter_path),
    }

    print("=== Clustering Summary ===")
    for k, v in metrics.items():
        print(f"{k:>14}: {v}")

    return metrics, plots


# ---------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run()
