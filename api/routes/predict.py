# api/routes/predict.py
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from ..model_loader import get_model, ModelNotFound
from ..schemas import CreditApplication, PredictionResult

router = APIRouter()


@router.post("/", response_model=PredictionResult)
def predict(application: CreditApplication):
    try:
        model = get_model()
    except ModelNotFound as e:
        raise HTTPException(status_code=500, detail=str(e))

    data = pd.DataFrame([application.dict()])

    proba = model.predict_proba(data)[:, 1]
    default_prob = float(proba[0])
    default_class = int(default_prob >= 0.5)

    return PredictionResult(
        default_probability=default_prob,
        default_class=default_class,
    )
