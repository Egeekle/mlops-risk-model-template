# api/routes/__init__.py
from fastapi import APIRouter

from .health import router as health_router
from .predict import router as predict_router

api_router = APIRouter()
api_router.include_router(health_router, prefix="", tags=["health"])
api_router.include_router(predict_router, prefix="/predict", tags=["prediction"])
