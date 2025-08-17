
> **Goal** – Explore whether engineered market features can  
> 1. uncover latent structure among S&P 500 tickers (PCA → K‑Means)  
> 2. predict next‑day price direction (logistic baseline & MLP)

---

## 1 Project Structure
final_project/
├── feature_engineering.py # pulls prices → engineers features → CSV
├── pca_baseline.py # PCA, elbow, logistic baseline (CLI script)
├── mlp_classifier.py # MLP model + plots
├── pipeline.ipynb # master notebook that orchestrates everything
├── config.py # shared paths / constants
├── requirements.txt # pinned deps
├── data/
│ └── not included -- click run all in pipeline.ipynb
└── figures also not included in repo

## 2 Quick Start
1. Install Requirements -- pip install -r requirements.txt
2. Generate feature matrix (≈ 2 min) -- $ python Feature\ Engineering.py
3. Run pipeline.ipynb 

## 3 Data Pipeline

1. Feature Engineering (Feature Engineering.py)
- Pulls 5 years of daily bars for current S&P 500 constituents.
- Engineers 11 features per (Date, Ticker) row:
    - Returns: 1 d, 5 d, 20 d, 60 d, 120 d
    - Volatility: 10 d σ, 20 d σ, 60 d σ
    - MACD + signal line
    - RSI‑14

2. Dimensionality Reduction & Clustering (pca_baseline.py)
- Standardises features, retains PCs explaining ≥ 95 % variance.
- Elbow method picks k = 3 clusters.

3. Classification Benchmarks
- Logistic regression baseline on PC space (macro‑avg AUC ≈ 0.75).
- MLP classifier (2‑layer, early‑stopping) with ROC & learning‑curve plots.


## 4 Reproducing Figures

All plots are written to figures/ and auto‑displayed in the notebook:
- pca_variance.png – explained‑variance curve.
- pca_clusters.png – clusters in PC₁ × PC₂ space.
- kmeans_elbow.png – SSE vs. k.
- mlp_training_history.png – loss & validation score.
- Confusion matrices + ROC curves for both models.

