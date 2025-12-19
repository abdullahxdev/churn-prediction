"""FastAPI backend module."""

from .main import app
from .schemas import CustomerData, PredictionResponse, BatchPredictionRequest

__all__ = ["app", "CustomerData", "PredictionResponse", "BatchPredictionRequest"]
