"""
FastAPI Main Application
========================

REST API for churn prediction service.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from sqlalchemy.orm import Session

from config import get_config, MODELS_DIR, ROOT_DIR
from .schemas import (
    CustomerData,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    HealthResponse,
    PredictionHistoryItem,
)
from .database import get_db, db_manager, PredictionRecord

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="ML-powered customer churn prediction service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded model and preprocessor
model = None
preprocessor = None
explainer = None
model_name = "xgboost"
feature_names = []


def load_model_and_preprocessor():
    """Load model and preprocessor on startup."""
    global model, preprocessor, model_name, feature_names

    # Try to load the model
    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        # Try alternative paths
        for name in ["xgboost", "random_forest", "lightgbm", "ensemble_voting"]:
            alt_path = MODELS_DIR / f"{name}.joblib"
            if alt_path.exists():
                model_path = alt_path
                model_name = name
                break

    if model_path.exists():
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.warning("No trained model found. API will return errors for predictions.")

    # Try to load preprocessor
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    if preprocessor_path.exists():
        preprocessor = joblib.load(preprocessor_path)
        logger.info(f"Loaded preprocessor from {preprocessor_path}")

    # Load feature names
    feature_names_path = MODELS_DIR / "feature_names.joblib"
    if feature_names_path.exists():
        feature_names = joblib.load(feature_names_path)


@app.on_event("startup")
async def startup_event():
    """Execute on application startup."""
    load_model_and_preprocessor()
    logger.info("Churn Prediction API started")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        preprocessor_loaded=preprocessor is not None,
        database_connected=True,
        timestamp=datetime.now()
    )


def get_risk_level(probability: float) -> str:
    """Convert probability to risk level."""
    if probability >= 0.7:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    else:
        return "Low"


def prepare_features(customer_data: CustomerData) -> pd.DataFrame:
    """Prepare features from customer data."""
    data_dict = customer_data.model_dump()

    # Remove CustomerID if present
    data_dict.pop("CustomerID", None)

    # Create DataFrame
    df = pd.DataFrame([data_dict])

    return df


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_single(
    customer: CustomerData,
    db: Session = Depends(get_db),
    include_factors: bool = Query(False, description="Include top risk factors")
):
    """
    Make a single churn prediction.

    Args:
        customer: Customer data
        db: Database session
        include_factors: Whether to include risk factors

    Returns:
        Prediction response
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )

    try:
        # Prepare features
        df = prepare_features(customer)

        # Transform if preprocessor available
        if preprocessor is not None:
            X = preprocessor.transform(df)
        else:
            X = df.values

        # Make prediction
        prediction = int(model.predict(X)[0])

        # Get probability
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(X)[0][1])
        else:
            probability = float(prediction)

        # Calculate confidence
        confidence = max(probability, 1 - probability)

        # Get risk level
        risk_level = get_risk_level(probability)

        # Get top risk factors if requested
        top_risk_factors = None
        if include_factors and explainer is not None:
            # Implementation would go here
            pass

        # Save to database
        db_manager.save_prediction(
            db=db,
            customer_id=customer.CustomerID,
            prediction=prediction,
            probability=probability,
            risk_level=risk_level,
            model_used=model_name,
            confidence=confidence,
            input_features=customer.model_dump(),
            top_risk_factors=top_risk_factors
        )

        return PredictionResponse(
            customer_id=customer.CustomerID,
            churn_prediction=prediction,
            churn_probability=probability,
            risk_level=risk_level,
            confidence=confidence,
            top_risk_factors=top_risk_factors,
            timestamp=datetime.now(),
            model_used=model_name
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(
    request: BatchPredictionRequest,
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """
    Make batch predictions for multiple customers.

    Args:
        request: Batch prediction request
        db: Database session
        background_tasks: Background task handler

    Returns:
        Batch prediction response
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )

    start_time = time.time()

    try:
        predictions = []

        for customer in request.customers:
            # Prepare features
            df = prepare_features(customer)

            # Transform
            if preprocessor is not None:
                X = preprocessor.transform(df)
            else:
                X = df.values

            # Predict
            pred = int(model.predict(X)[0])
            prob = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else float(pred)
            confidence = max(prob, 1 - prob)
            risk_level = get_risk_level(prob)

            # Save to database (in background for large batches)
            db_manager.save_prediction(
                db=db,
                customer_id=customer.CustomerID,
                prediction=pred,
                probability=prob,
                risk_level=risk_level,
                model_used=model_name,
                confidence=confidence,
                input_features=customer.model_dump()
            )

            predictions.append(PredictionResponse(
                customer_id=customer.CustomerID,
                churn_prediction=pred,
                churn_probability=prob,
                risk_level=risk_level,
                confidence=confidence,
                timestamp=datetime.now(),
                model_used=model_name
            ))

        processing_time = (time.time() - start_time) * 1000

        # Calculate summary statistics
        churn_count = sum(1 for p in predictions if p.churn_prediction == 1)
        high_risk = sum(1 for p in predictions if p.risk_level == "High")
        medium_risk = sum(1 for p in predictions if p.risk_level == "Medium")
        low_risk = sum(1 for p in predictions if p.risk_level == "Low")
        avg_prob = sum(p.churn_probability for p in predictions) / len(predictions)

        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(predictions),
            churn_count=churn_count,
            non_churn_count=len(predictions) - churn_count,
            average_churn_probability=avg_prob,
            high_risk_count=high_risk,
            medium_risk_count=medium_risk,
            low_risk_count=low_risk,
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/history", tags=["History"])
async def get_prediction_history(
    db: Session = Depends(get_db),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    customer_id: Optional[str] = Query(None)
):
    """
    Get prediction history.

    Args:
        db: Database session
        limit: Maximum records
        offset: Records to skip
        customer_id: Filter by customer

    Returns:
        List of prediction records
    """
    records = db_manager.get_predictions(db, limit, offset, customer_id)
    return [r.to_dict() for r in records]


@app.get("/predictions/statistics", tags=["History"])
async def get_prediction_statistics(db: Session = Depends(get_db)):
    """
    Get aggregate prediction statistics.

    Args:
        db: Database session

    Returns:
        Statistics dictionary
    """
    return db_manager.get_prediction_statistics(db)


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded"
        )

    return ModelInfo(
        model_name=model_name,
        model_type=type(model).__name__,
        version="1.0.0",
        training_date=None,
        metrics={},
        feature_count=len(feature_names) if feature_names else 0,
        is_active=True
    )


@app.post("/model/reload", tags=["Model"])
async def reload_model():
    """Reload the model from disk."""
    load_model_and_preprocessor()

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Failed to load model"
        )

    return {"message": "Model reloaded successfully", "model_name": model_name}


@app.post("/feedback/{prediction_id}", tags=["Feedback"])
async def add_feedback(
    prediction_id: int,
    actual_outcome: Optional[int] = Query(None, ge=0, le=1),
    notes: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Add feedback for a prediction.

    Args:
        prediction_id: ID of prediction
        actual_outcome: Actual churn outcome
        notes: Feedback notes
        db: Database session

    Returns:
        Success message
    """
    # Verify prediction exists
    prediction = db_manager.get_prediction_by_id(db, prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    db_manager.add_feedback(db, prediction_id, actual_outcome, notes)
    return {"message": "Feedback recorded successfully"}


# Run with: uvicorn src.api.main:app --reload
if __name__ == "__main__":
    import uvicorn

    config = get_config()
    api_config = config.get("api", {})

    uvicorn.run(
        "src.api.main:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        reload=api_config.get("reload", True)
    )
