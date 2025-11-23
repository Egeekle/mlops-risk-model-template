# src/hyperparameter_optimization.py
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import optuna
from optuna import Trial
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .model_utils import build_model_pipeline
from .config import RANDOM_STATE, CV_N_SPLITS, OPTUNA_METRIC


def objective(
    trial: Trial,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    metric: str = "auc",
) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        X: Feature matrix
        y: Target array
        n_splits: Number of folds for cross-validation
        metric: Metric to optimize ("auc" or "f1")
        
    Returns:
        Mean metric value across CV folds
    """
    # Suggest hyperparameters
    C = trial.suggest_float("C", 1e-4, 100.0, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    solver = trial.suggest_categorical(
        "solver", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]
    )
    max_iter = trial.suggest_int("max_iter", 500, 2000, step=100)
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    
    # Handle solver compatibility with penalty
    if penalty == "l1":
        if solver not in ["liblinear", "saga"]:
            solver = "liblinear"
    elif penalty == "elasticnet":
        if solver != "saga":
            solver = "saga"
    elif penalty == "l2":
        if solver not in ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]:
            solver = "lbfgs"
    
    # Suggest l1_ratio for elasticnet
    l1_ratio = None
    if penalty == "elasticnet":
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    
    # Build model with suggested hyperparameters
    model_params = {
        "random_state": RANDOM_STATE,
        "C": C,
        "penalty": penalty,
        "solver": solver,
        "max_iter": max_iter,
        "class_weight": class_weight,
    }
    
    if l1_ratio is not None:
        model_params["l1_ratio"] = l1_ratio
    
    model = build_model_pipeline(**model_params)
    
    # Perform cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train model on fold
        model.fit(X_train_fold, y_train_fold)
        
        # Predict on validation fold
        y_proba_fold = model.predict_proba(X_val_fold)[:, 1]
        
        # Calculate metric
        if metric == "auc":
            score = roc_auc_score(y_val_fold, y_proba_fold)
        elif metric == "f1":
            y_pred_fold = (y_proba_fold >= 0.5).astype(int)
            score = f1_score(y_val_fold, y_pred_fold)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)
    
    # Return mean score across folds
    return float(np.mean(scores))


def optimize_hyperparameters(
    X: pd.DataFrame,
    y: np.ndarray,
    n_trials: int = 50,
    n_splits: int = 5,
    metric: str = "auc",
    timeout: Optional[int] = None,
    study_name: str = "credit_risk_optimization",
    direction: str = "maximize",
    show_progress_bar: bool = True,
) -> Dict[str, Any]:
    """
    Optimize hyperparameters using Optuna.
    
    Args:
        X: Feature matrix
        y: Target array
        n_trials: Number of optimization trials
        n_splits: Number of folds for cross-validation
        metric: Metric to optimize ("auc" or "f1")
        timeout: Timeout in seconds (None = no timeout)
        study_name: Name of the Optuna study
        direction: Optimization direction ("maximize" or "minimize")
        show_progress_bar: Whether to show progress bar
        
    Returns:
        Dictionary with best parameters and best value
    """
    print(f"\n{'='*60}")
    print(f"Optuna Hyperparameter Optimization")
    print(f"{'='*60}")
    print(f"Metric: {metric.upper()}")
    print(f"Trials: {n_trials}")
    print(f"CV Folds: {n_splits}")
    print(f"Direction: {direction}")
    print(f"{'='*60}\n")
    
    # Create study
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, X, y, n_splits=n_splits, metric=metric),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress_bar,
    )
    
    print(f"\n{'='*60}")
    print("Optimization Complete")
    print(f"{'='*60}")
    print(f"Best {metric.upper()}: {study.best_value:.4f}")
    print(f"\nBest Parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    print(f"{'='*60}\n")
    
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "study": study,
    }

