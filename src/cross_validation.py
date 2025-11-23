# src/cross_validation.py
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


def perform_cross_validation(
    model: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Perform k-fold stratified cross-validation and return aggregated metrics.
    
    Args:
        model: Scikit-learn pipeline or model
        X: Feature matrix
        y: Target array
        n_splits: Number of folds for cross-validation
        random_state: Random state for reproducibility
        threshold: Probability threshold for binary classification
        
    Returns:
        Dictionary with mean and std metrics across folds
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Store metrics for each fold
    fold_metrics = {
        "auc": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "accuracy": [],
    }
    
    print(f"\n{'='*60}")
    print(f"Performing {n_splits}-Fold Stratified Cross-Validation")
    print(f"{'='*60}\n")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train model on fold
        model.fit(X_train_fold, y_train_fold)
        
        # Predict on validation fold
        y_proba_fold = model.predict_proba(X_val_fold)[:, 1]
        y_pred_fold = (y_proba_fold >= threshold).astype(int)
        
        # Calculate metrics
        auc = roc_auc_score(y_val_fold, y_proba_fold)
        f1 = f1_score(y_val_fold, y_pred_fold)
        precision = precision_score(y_val_fold, y_pred_fold)
        recall = recall_score(y_val_fold, y_pred_fold)
        accuracy = accuracy_score(y_val_fold, y_pred_fold)
        
        # Store metrics
        fold_metrics["auc"].append(auc)
        fold_metrics["f1"].append(f1)
        fold_metrics["precision"].append(precision)
        fold_metrics["recall"].append(recall)
        fold_metrics["accuracy"].append(accuracy)
        
        print(f"Fold {fold_idx}/{n_splits}:")
        print(f"  AUC: {auc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Accuracy: {accuracy:.4f}")
    
    # Calculate mean and std across folds
    cv_results = {}
    for metric_name, values in fold_metrics.items():
        cv_results[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": [float(v) for v in values],
        }
    
    print(f"\n{'='*60}")
    print("Cross-Validation Results (Mean ± Std)")
    print(f"{'='*60}")
    for metric_name, stats in cv_results.items():
        print(f"{metric_name.upper():12s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"{'='*60}\n")
    
    return cv_results


def get_cv_summary(cv_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Extract mean metrics from CV results for logging.
    
    Args:
        cv_results: Results from perform_cross_validation
        
    Returns:
        Dictionary with mean metrics only
    """
    return {f"cv_{metric}_mean": stats["mean"] for metric, stats in cv_results.items()}

