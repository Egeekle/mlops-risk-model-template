# src/evaluate.py
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

from .config import PROCESSED_DATA_DIR, LATEST_MODEL_PATH, TARGET_COLUMN
from .model_utils import split_features_target


def main() -> None:
    valid_path = PROCESSED_DATA_DIR / "valid.csv"

    if not valid_path.exists():
        raise FileNotFoundError(
            f"No se encontró {valid_path}. Ejecuta primero el preprocesamiento (data_prep)."
        )

    if not LATEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {LATEST_MODEL_PATH}. Ejecuta primero src.train."
        )

    print(f"Cargando validación desde: {valid_path}")
    valid_df = pd.read_csv(valid_path, parse_dates=["application_date"])

    X_valid, y_valid = split_features_target(valid_df, TARGET_COLUMN)

    print(f"Cargando modelo desde: {LATEST_MODEL_PATH}")
    model = joblib.load(LATEST_MODEL_PATH)

    y_proba = model.predict_proba(X_valid)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_valid, y_proba)
    f1 = f1_score(y_valid, y_pred)

    print("===== MÉTRICAS DE EVALUACIÓN =====")
    print(f"AUC valid: {auc:.4f}")
    print(f"F1  valid: {f1:.4f}")


if __name__ == "__main__":
    main()
