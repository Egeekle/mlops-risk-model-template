# api/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "model-latest.pkl"

API_TITLE = "Credit Risk Scoring API"
API_VERSION = "0.1.0"
