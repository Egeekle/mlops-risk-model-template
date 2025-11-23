# src/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "credit_risk.csv"

MODELS_DIR = BASE_DIR / "models"
LATEST_MODEL_PATH = MODELS_DIR / "model-latest.pkl"

RANDOM_STATE = 42
TARGET_COLUMN = "default_90d"

# Cross-Validation Configuration
CV_N_SPLITS = 5  # Number of folds for k-fold cross-validation
CV_THRESHOLD = 0.5  # Probability threshold for binary classification

# Optuna Hyperparameter Optimization Configuration
OPTUNA_N_TRIALS = 50  # Number of optimization trials
OPTUNA_TIMEOUT = None  # Timeout in seconds (None = no timeout)
OPTUNA_STUDY_NAME = "credit_risk_optimization"  # Study name for Optuna
OPTUNA_DIRECTION = "maximize"  # Direction: "maximize" for AUC, "minimize" for loss
OPTUNA_METRIC = "auc"  # Metric to optimize: "auc", "f1", etc.

# --- Configuración MLflow ---
# Usamos un store local dentro del repo.
MLFLOW_DIR = BASE_DIR / "mlruns"
#MLFLOW_TRACKING_URI = f"file://{MLFLOW_DIR}"
MLFLOW_TRACKING_URI = MLFLOW_DIR.as_uri()
MLFLOW_EXPERIMENT_NAME = "credit_risk_baseline"
METRICS_DIR = BASE_DIR / "metrics"
BASELINE_METRICS_PATH = METRICS_DIR / "baseline_metrics.json"
PRODUCTION_DATA_DIR = DATA_DIR / "production"
PRODUCTION_DATA_PATH = PRODUCTION_DATA_DIR / "production_batch.csv"