# api/model_loader.py
from functools import lru_cache
import joblib

from .config import MODEL_PATH


class ModelNotFound(Exception):
    pass


@lru_cache(maxsize=1)
def get_model():
    """
    Carga el modelo entrenado desde disco y lo cachea.
    """
    if not MODEL_PATH.exists():
        raise ModelNotFound(f"No se encontró el modelo en {MODEL_PATH}. Ejecuta src/train.py primero.")
    model = joblib.load(MODEL_PATH)
    return model
