# src/train.py
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

import mlflow
import mlflow.sklearn

from .config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    TARGET_COLUMN,
    RANDOM_STATE,
    LATEST_MODEL_PATH,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    CV_N_SPLITS,
    CV_THRESHOLD,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT,
    OPTUNA_STUDY_NAME,
    OPTUNA_DIRECTION,
    OPTUNA_METRIC,
    METRICS_DIR,
    BASELINE_METRICS_PATH,
)
from .model_utils import split_features_target, build_model_pipeline
from .cross_validation import perform_cross_validation, get_cv_summary
from .hyperparameter_optimization import optimize_hyperparameters


def main() -> None:
    train_path = PROCESSED_DATA_DIR / "train.csv"
    valid_path = PROCESSED_DATA_DIR / "valid.csv"

    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(
            "No se encontraron train.csv / valid.csv. "
            "Ejecuta primero: dvc repro (o python -m src.data_prep)."
        )

    # 1) Cargar datos
    train_df = pd.read_csv(train_path, parse_dates=["application_date"])
    valid_df = pd.read_csv(valid_path, parse_dates=["application_date"])

    X_train, y_train = split_features_target(train_df, TARGET_COLUMN)
    X_valid, y_valid = split_features_target(valid_df, TARGET_COLUMN)

    # 2) Optimize hyperparameters using Optuna
    optimization_results = optimize_hyperparameters(
        X=X_train,
        y=y_train,
        n_trials=OPTUNA_N_TRIALS,
        n_splits=CV_N_SPLITS,
        metric=OPTUNA_METRIC,
        timeout=OPTUNA_TIMEOUT,
        study_name=OPTUNA_STUDY_NAME,
        direction=OPTUNA_DIRECTION,
        show_progress_bar=True,
    )
    
    best_params = optimization_results["best_params"]
    best_value = optimization_results["best_value"]

    # 3) Build model pipeline with optimized hyperparameters
    model_params = {
        "random_state": RANDOM_STATE,
        "C": best_params["C"],
        "penalty": best_params["penalty"],
        "solver": best_params["solver"],
        "max_iter": best_params["max_iter"],
        "class_weight": best_params["class_weight"],
    }
    
    # Add l1_ratio if present (for elasticnet penalty)
    if "l1_ratio" in best_params:
        model_params["l1_ratio"] = best_params["l1_ratio"]
    
    model = build_model_pipeline(**model_params)

    # 4) Configurar MLflow (tracking local)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="logreg_optimized"):

        # 4.1) Log de parámetros
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "random_state": RANDOM_STATE,
                "train_rows": len(train_df),
                "valid_rows": len(valid_df),
                "cv_n_splits": CV_N_SPLITS,
                "cv_threshold": CV_THRESHOLD,
                "optuna_n_trials": OPTUNA_N_TRIALS,
                "optuna_metric": OPTUNA_METRIC,
                **{f"best_{k}": v for k, v in best_params.items()},
            }
        )
        
        # Log best optimization value
        mlflow.log_metric(f"optuna_best_{OPTUNA_METRIC}", best_value)

        # 5) Perform Cross-Validation on training set with optimized model
        cv_model_params = {
            "random_state": RANDOM_STATE,
            "C": best_params["C"],
            "penalty": best_params["penalty"],
            "solver": best_params["solver"],
            "max_iter": best_params["max_iter"],
            "class_weight": best_params["class_weight"],
        }
        
        if "l1_ratio" in best_params:
            cv_model_params["l1_ratio"] = best_params["l1_ratio"]
        
        cv_results = perform_cross_validation(
            model=build_model_pipeline(**cv_model_params),
            X=X_train,
            y=y_train,
            n_splits=CV_N_SPLITS,
            random_state=RANDOM_STATE,
            threshold=CV_THRESHOLD,
        )
        
        # 5.1) Log CV metrics to MLflow
        cv_metrics = get_cv_summary(cv_results)
        mlflow.log_metrics(cv_metrics)
        
        # Also log std metrics
        cv_std_metrics = {f"cv_{metric}_std": stats["std"] for metric, stats in cv_results.items()}
        mlflow.log_metrics(cv_std_metrics)

        # 6) Train final model on all training data with optimized hyperparameters
        print("\nTraining final model on full training set with optimized hyperparameters...")
        model.fit(X_train, y_train)

        # 7) Evaluate on validation set
        y_proba = model.predict_proba(X_valid)[:, 1]
        y_pred = (y_proba >= CV_THRESHOLD).astype(int)

        auc = roc_auc_score(y_valid, y_proba)
        f1 = f1_score(y_valid, y_pred)

        print(f"\nValidation Set Results:")
        print(f"AUC valid: {auc:.4f}")
        print(f"F1  valid: {f1:.4f}")

        # 7.1) Log validation metrics
        mlflow.log_metrics(
            {
                "auc_valid": auc,
                "f1_valid": f1,
            }
        )

        # 8) Log del modelo en MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None,  # puedes usar un Model Registry si quieres
        )

        # 9) Guardar el modelo "oficial" para la API
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, LATEST_MODEL_PATH)
        print(f"Modelo guardado en: {LATEST_MODEL_PATH}")
        
        # 10) Guardar métricas baseline para detección de drift
        import json
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        baseline_metrics = {
            "auc_valid": float(auc),
            "f1_valid": float(f1),
        }
        with open(BASELINE_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(baseline_metrics, f, indent=2)
        print(f"Métricas baseline guardadas en: {BASELINE_METRICS_PATH}")


if __name__ == "__main__":
    main()
