# pip install pyarrow fastparquet

from config import SCRIPT_DIR, OUTPUT_DIR
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


'''
Expected column format of csv file is:
Date, Ticker, Adj Close, Ret_1d, Ret_2d, Ret_3d, Ret_4d, Ret_5d, Ret_20d, Ret_60d, Ret_120d, Vol_10d, Vol_20d, Vol_60d, MACD, MACD_Signal, RSI14

Date is there just to organize the df in chronological order.
Target is taken from Ret_1d, which will become 1 for positive returns and 0 for negative returns.
'''

def main():
    drop_cols = ['Adj Close', 'label', 'Ticker', 'Date', 'Ret_1d']

    # Load data and finalize the preparations for PCA
    path = os.path.join(SCRIPT_DIR, './data/sp500_features.csv')
    df = pd.read_csv(path)
    df = df.dropna()
    df = df.sort_values('Date')
    original_len = len(df)

    # Create the targets based off the 1 day return
    # Multi-class labels
    df['label'] = 1  # default = small/no move
    df.loc[df['Ret_1d'] > 0.015, 'label'] = 2  # big positive move
    df.loc[df['Ret_1d'] < -0.015, 'label'] = 0  # big negative move

    # Separate the feature set and the target, prepare for PCA analysis
    X = df.drop(columns=drop_cols)
    y = df['label'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Don't know the correct number of components, so instead capture 95% of the variance.
    pca = PCA(n_components=0.95)  # keep 95% of variance
    X_pca = pca.fit_transform(X_scaled)

    # Save PCA cumulative variance plot
    plt.figure(figsize=(6, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), drawstyle='steps-post', label='Cumulative Explained Variance')
    plt.axhline(0.95, color='red', linestyle='--', label='95% Threshold')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.legend()
    os.makedirs(os.path.join(SCRIPT_DIR, 'figures'), exist_ok=True)
    pca_plot_path = os.path.join(SCRIPT_DIR, 'figures/pca_variance.png')
    plt.savefig(pca_plot_path)
    plt.close()

    # Save PCA-transformed features
    df_pca = pd.DataFrame(X_pca)
    df_pca['label'] = y

    out_path = os.path.join(OUTPUT_DIR, "pca_features.parquet")
    df_pca.to_parquet(out_path, index=False)

    # Split the data up, but maintain chronological order
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, shuffle=False  # Time-series style split
    )

    # Establish the logistic baseline
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['down', 'neutral', 'up'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Logistic Regression Confusion Matrix")
    cm_path = os.path.join(SCRIPT_DIR, 'figures/confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    # Print the results
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1': f1_score(y_test, y_pred, average='macro')
    }

    print("====== Logistic Regression Baseline Metrics: ======")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")

    print(classification_report(y_test, y_pred, target_names=['down', 'neutral', 'up']))

    # ROC Curve (One-vs-Rest)
    y_score = clf.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(7, 5))
    colors = ['darkorange', 'green', 'blue']
    labels = ['Down', 'Neutral', 'Up']

    for i, color in enumerate(colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'Class {labels[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('One-vs-Rest ROC Curve (Logistic Regression)')
    plt.legend(loc="lower right")

    roc_plot_path = os.path.join(SCRIPT_DIR, 'figures/logreg_roc.png')
    plt.savefig(roc_plot_path)
    plt.close() 

    plots = {
        "pca_variance": str(pca_plot_path),
        "confusion_matrix": str(cm_path),
        "logreg_roc": str(roc_plot_path)
    }

    return plots

# Allow running as script or import
if __name__ == "__main__":
    main()
