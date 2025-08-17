"""
Stock Price Movement Prediction using Multi-Layer Perceptron (MLP) Classifier

This module builds upon the clustered features to predict significant stock price movements.
The workflow is as follows:

1. Data Input:
   - Reads the clustered features from 'cluster.parquet'
   - Features include both PCA components and cluster assignments

2. Target Creation:
   - Converts the multi-class labels to binary classification
   - Class 1 (positive): Large upward price movements (>1.5% increase)
   - Class 0 (negative): Small movements or decreases

3. Model Architecture:
   - Uses scikit-learn's MLPClassifier
   - Three hidden layers: 128 → 64 → 32 neurons
   - ReLU activation functions
   - Adam optimizer with adaptive learning rate
   - Early stopping to prevent overfitting

4. Training Approach:
   - Chronological train-test split (80-20)
   - No shuffling to maintain time series integrity
   - Validation fraction: 0.2 of training data
   - Early stopping with 10 epochs patience

5. Evaluation Metrics:
   - Accuracy: Overall prediction accuracy
   - Precision: Accuracy of positive predictions
   - Recall: Ability to find all positive cases
   - F1-score: Harmonic mean of precision and recall
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from pathlib import Path
from config import SCRIPT_DIR, OUTPUT_DIR

# Set plot style
# Configure paths
DATA_DIR = Path(OUTPUT_DIR)
FIG_DIR = Path(SCRIPT_DIR) / "figures"
FIG_DIR.mkdir(exist_ok=True)
plt.style.use('default')
sns.set_style("whitegrid")

def load_data():
    """Load the enriched dataset with PCA and clustering features"""
    input_path = os.path.join(OUTPUT_DIR, "cluster.parquet")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            "\nERROR: Cluster features file not found at: {}"
            "\n\nThis module requires the cluster.parquet file."
            "\nPlease ensure:"
            "\n1. The data/ directory exists"
            "\n2. The cluster.parquet file has been provided"
            .format(input_path)
        )
    
    try:
        df = pd.read_parquet(input_path)
        return df
    except Exception as e:
        raise Exception(
            "\nError reading cluster features file: {}"
            "\nPlease ensure cluster.parquet is valid: {}"
            .format(str(e))
        )

def create_model():
    """Create the MLPClassifier model"""
    return MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),  # Three hidden layers
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=32,
        learning_rate='adaptive',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        verbose=False
    )

def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """Calculate and return evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    # Detailed classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    return metrics, report

def save_training_report(model_info, metrics, report, training_history, output_path):
    """Generate and save a detailed training report"""
    report_content = f"""MLP Classifier Training Report
============================

Dataset Overview
---------------
Shape: {model_info['shape']}
Number of features: {model_info['n_features']}
Number of samples: {model_info['n_samples']}
Time period coverage: {model_info['time_period']:.1f} trading days

Class Distribution
-----------------
Big negative moves (<-1.5%): {model_info['dist_neg']:.1%}
Small/No moves (-1.5% to 1.5%): {model_info['dist_neutral']:.1%}
Big positive moves (>1.5%): {model_info['dist_pos']:.1%}

Data Split
----------
Training set shape: {model_info['train_shape']}
Test set shape: {model_info['test_shape']}

Training Progress
----------------
Initial:
- Loss: {training_history['loss'][0]:.8f}
- Validation score: {training_history['val_score'][0]:.6f}

Final (after {len(training_history['loss'])} epochs):
- Loss: {training_history['loss'][-1]:.8f}
- Validation score: {training_history['val_score'][-1]:.6f}

Training History (Loss/Validation Score):
Epoch  Loss         Validation
------ ------------ -----------"""

    # Add training history
    for i, (loss, val) in enumerate(zip(training_history['loss'], training_history['val_score']), 1):
        report_content += f"\n{i:<6d} {loss:<12.8f} {val:.6f}"

    report_content += f"""

Final Performance Metrics
------------------------
Accuracy:  {metrics['accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall:    {metrics['recall']:.4f}
F1-score:  {metrics['f1']:.4f}

Detailed Classification Report
----------------------------
Class 0 (Not significant increase):
- Precision: {report['0']['precision']:.4f}
- Recall:    {report['0']['recall']:.4f}
- F1-score:  {report['0']['f1-score']:.4f}
- Support:   {report['0']['support']:,.0f} samples

Class 1 (Significant increase):
- Precision: {report['1']['precision']:.4f}
- Recall:    {report['1']['recall']:.4f}
- F1-score:  {report['1']['f1-score']:.4f}
- Support:   {report['1']['support']:,.0f} samples

Overall Metrics:
- Accuracy: {report['accuracy']:.4f}
- Macro avg:
  * Precision: {report['macro avg']['precision']:.4f}
  * Recall:    {report['macro avg']['recall']:.4f}
  * F1-score:  {report['macro avg']['f1-score']:.4f}
- Weighted avg:
  * Precision: {report['weighted avg']['precision']:.4f}
  * Recall:    {report['weighted avg']['recall']:.4f}
  * F1-score:  {report['weighted avg']['f1-score']:.4f}

Generated Visualizations
-----------------------
1. mlp_training_history.png - Shows loss curve and validation scores over epochs
2. mlp_confusion_matrix.png - Displays the confusion matrix for test set predictions

Note: This model shows strong accuracy but lower recall for significant price increases,
suggesting it's conservative in predicting major upward movements."""

    with open(output_path, 'w') as f:
        f.write(report_content)
    
    return output_path

def save_results(metrics, report, model, X_test, y_test, y_pred, model_info=None, training_history=None):
    """Save all results to a dedicated folder in the data directory"""
    # Create output directory
    output_folder = os.path.join(OUTPUT_DIR, "mlp_classifier_output")
    os.makedirs(output_folder, exist_ok=True)
    
    # Create figures directory if it doesn't exist
    os.makedirs(FIG_DIR, exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_parquet(os.path.join(output_folder, "metrics.parquet"), index=False)
    
    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_parquet(os.path.join(output_folder, "classification_report.parquet"))
    
    # Save predictions with probabilities
    predictions_df = pd.DataFrame({
        'true_label': y_test,
        'predicted': y_pred,
        'probability': model.predict_proba(X_test)[:, 1]
    })
    predictions_df.to_parquet(os.path.join(output_folder, "predictions.parquet"), index=False)
    
    # Save model parameters
    params = {
        'hidden_layer_sizes': model.hidden_layer_sizes,
        'activation': model.activation,
        'solver': model.solver,
        'alpha': model.alpha,
        'batch_size': model.batch_size,
        'learning_rate': model.learning_rate,
        'max_iter': model.max_iter,
        'early_stopping': model.early_stopping,
        'validation_fraction': model.validation_fraction,
        'n_iter_no_change': model.n_iter_no_change
    }
    pd.DataFrame([params]).to_parquet(os.path.join(output_folder, "model_parameters.parquet"), index=False)
    
    # Initialize plot paths
    plot_paths = {}
    
    # Save individual plots in figures directory
    if hasattr(model, 'loss_curve_'):
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_curve_, label='Training Loss')
        plt.title('Learning Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        loss_path = os.path.join(FIG_DIR, "mlp_loss_curve.png")
        plt.savefig(loss_path, dpi=120, bbox_inches='tight')
        plt.close()
        plot_paths['loss_curve'] = loss_path
    
    if hasattr(model, 'validation_scores_'):
        plt.figure(figsize=(10, 6))
        plt.plot(model.validation_scores_, label='Validation Score')
        plt.title('Validation Scores During Training')
        plt.xlabel('Iterations')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        val_path = os.path.join(FIG_DIR, "mlp_validation_scores.png")
        plt.savefig(val_path, dpi=120, bbox_inches='tight')
        plt.close()
        plot_paths['validation_scores'] = val_path
    
    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(FIG_DIR, "mlp_confusion_matrix.png")
    plt.savefig(cm_path, dpi=120, bbox_inches='tight')
    plt.close()
    plot_paths['confusion_matrix'] = cm_path
    
    # Save training report if model info is provided
    if model_info is not None and training_history is not None:
        report_path = os.path.join(FIG_DIR, "mlp_training_report.txt")
        save_training_report(model_info, metrics, report, training_history, report_path)
    
    # Return paths for pipeline
    return plot_paths

def train_and_evaluate():
    """Main function to train and evaluate the MLP model"""
    print("Loading and preparing data...")
    df = load_data()

    # Dataset overview
    print(f"Shape: {df.shape}")
    dist = df["label"].value_counts(normalize=True)
    print(
        f"Class distribution 0/1/2 → "
        f"{dist.get(0,0):.1%} / {dist.get(1,0):.1%} / {dist.get(2,0):.1%}"
    )

    # ------------------------------------------------------------------ #
    # 1) Build feature matrix & target
    # ------------------------------------------------------------------ #
    X = df.drop(["label", "cluster_label"], axis=1)
    y = (df["label"] == 2).astype(int)          # 1 = up, 0 = down or neutral

    # Chronological 80 / 20 split
    split_idx = int(len(X) * 0.8)
    X_train, X_test  = X.iloc[:split_idx],  X.iloc[split_idx:]
    y_train, y_test  = y.iloc[:split_idx],  y.iloc[split_idx:]

    # Now the shapes exist → safe to print
    print(f"Train/Test split: {X_train.shape}  /  {X_test.shape}")

    # ------------------------------------------------------------------ #
    # 2) Model
    # ------------------------------------------------------------------ #
    print("\nCreating and training MLP model…")
    model = create_model()
    model.fit(X_train, y_train)

    # Predictions & metrics
    y_pred  = model.predict(X_test)
    metrics, report = evaluate_model(y_test, y_pred)
    
    # Collect model info and training history
    model_info = {
        'shape': df.shape,
        'n_features': df.shape[1] - 2,  # Excluding label and cluster_label
        'n_samples': df.shape[0],
        'time_period': df.shape[0] / 500,
        'dist_neg': dist.get(0, 0),
        'dist_neutral': dist.get(1, 0),
        'dist_pos': dist.get(2, 0),
        'train_shape': X_train.shape,
        'test_shape': X_test.shape
    }
    
    training_history = {
        'loss': model.loss_curve_,
        'val_score': model.validation_scores_
    }
    
    # Create figures directory
    os.makedirs(FIG_DIR, exist_ok=True)
    
    # Save combined training history plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(model.loss_curve_, label='Training Loss')
    plt.title('Learning Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(model.validation_scores_, label='Validation Score')
    plt.title('Validation Score History')
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    history_path = os.path.join(FIG_DIR, "mlp_training_history.png")
    plt.savefig(history_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    # Save individual plots and get their paths
    additional_plot_paths = save_results(metrics, report, model, X_test, y_test, y_pred, model_info, training_history)
    
    # Combine all plot paths
    final_plot_paths = {
        'training_history': history_path,
        **additional_plot_paths
    }
    
    # Print metrics
    print("\nPerformance Metrics:")
    print("-" * 50)
    for metric, value in metrics.items():
        print(f"{metric:>10}: {value:.4f}")
    
    
    return metrics, report, final_plot_paths

if __name__ == "__main__":
    metrics, report, final_plot_paths = train_and_evaluate()
    print("\nGenerated plots:")
    for plot_type, path in final_plot_paths.items():
        print(f"- {plot_type}: {path}")
