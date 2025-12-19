"""
Database Module
===============

SQLAlchemy database models and operations for storing predictions.
"""

from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    DateTime,
    JSON,
    Boolean,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from loguru import logger

from config import get_config, ROOT_DIR


# Get configuration
config = get_config()
db_config = config.get("database", {})

# Database URL
DATABASE_URL = db_config.get("url", f"sqlite:///{ROOT_DIR}/data/churn_predictions.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=db_config.get("echo", False),
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


class PredictionRecord(Base):
    """Database model for storing predictions."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, index=True, nullable=True)
    prediction = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    model_used = Column(String, nullable=False)
    input_features = Column(JSON, nullable=True)
    top_risk_factors = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> Dict:
        """Convert record to dictionary."""
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "prediction": self.prediction,
            "probability": self.probability,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "model_used": self.model_used,
            "input_features": self.input_features,
            "top_risk_factors": self.top_risk_factors,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ModelMetrics(Base):
    """Database model for storing model metrics."""

    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False, index=True)
    model_version = Column(String, nullable=True)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    training_samples = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> Dict:
        """Convert record to dictionary."""
        return {
            "id": self.id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "training_samples": self.training_samples,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class CustomerFeedback(Base):
    """Database model for storing prediction feedback."""

    __tablename__ = "customer_feedback"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, nullable=False)
    actual_outcome = Column(Integer, nullable=True)
    feedback_notes = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DatabaseManager:
    """Manager class for database operations."""

    def __init__(self):
        """Initialize DatabaseManager."""
        create_tables()

    def save_prediction(
        self,
        db: Session,
        customer_id: Optional[str],
        prediction: int,
        probability: float,
        risk_level: str,
        model_used: str,
        confidence: Optional[float] = None,
        input_features: Optional[Dict] = None,
        top_risk_factors: Optional[List] = None
    ) -> PredictionRecord:
        """
        Save a prediction to the database.

        Args:
            db: Database session
            customer_id: Customer identifier
            prediction: Predicted class
            probability: Prediction probability
            risk_level: Risk level category
            model_used: Name of model used
            confidence: Prediction confidence
            input_features: Input features dict
            top_risk_factors: Top risk factors

        Returns:
            Created PredictionRecord
        """
        record = PredictionRecord(
            customer_id=customer_id,
            prediction=prediction,
            probability=probability,
            risk_level=risk_level,
            confidence=confidence,
            model_used=model_used,
            input_features=input_features,
            top_risk_factors=top_risk_factors
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    def get_predictions(
        self,
        db: Session,
        limit: int = 100,
        offset: int = 0,
        customer_id: Optional[str] = None
    ) -> List[PredictionRecord]:
        """
        Get prediction records.

        Args:
            db: Database session
            limit: Maximum number of records
            offset: Number of records to skip
            customer_id: Filter by customer ID

        Returns:
            List of PredictionRecord
        """
        query = db.query(PredictionRecord)

        if customer_id:
            query = query.filter(PredictionRecord.customer_id == customer_id)

        return query.order_by(PredictionRecord.created_at.desc()).offset(offset).limit(limit).all()

    def get_prediction_by_id(
        self,
        db: Session,
        prediction_id: int
    ) -> Optional[PredictionRecord]:
        """Get a single prediction by ID."""
        return db.query(PredictionRecord).filter(PredictionRecord.id == prediction_id).first()

    def get_prediction_statistics(self, db: Session) -> Dict:
        """
        Get aggregate statistics of predictions.

        Args:
            db: Database session

        Returns:
            Dictionary with statistics
        """
        total = db.query(PredictionRecord).count()
        churn_count = db.query(PredictionRecord).filter(PredictionRecord.prediction == 1).count()
        non_churn_count = total - churn_count

        high_risk = db.query(PredictionRecord).filter(PredictionRecord.risk_level == "High").count()
        medium_risk = db.query(PredictionRecord).filter(PredictionRecord.risk_level == "Medium").count()
        low_risk = db.query(PredictionRecord).filter(PredictionRecord.risk_level == "Low").count()

        # Average probability
        from sqlalchemy import func
        avg_prob = db.query(func.avg(PredictionRecord.probability)).scalar() or 0

        return {
            "total_predictions": total,
            "churn_predictions": churn_count,
            "non_churn_predictions": non_churn_count,
            "churn_rate": churn_count / total if total > 0 else 0,
            "high_risk_count": high_risk,
            "medium_risk_count": medium_risk,
            "low_risk_count": low_risk,
            "average_probability": avg_prob,
        }

    def save_model_metrics(
        self,
        db: Session,
        model_name: str,
        metrics: Dict[str, float],
        model_version: Optional[str] = None,
        training_samples: Optional[int] = None,
        is_active: bool = False
    ) -> ModelMetrics:
        """
        Save model metrics to database.

        Args:
            db: Database session
            model_name: Name of the model
            metrics: Dictionary of metrics
            model_version: Version string
            training_samples: Number of training samples
            is_active: Whether this is the active model

        Returns:
            Created ModelMetrics record
        """
        record = ModelMetrics(
            model_name=model_name,
            model_version=model_version,
            accuracy=metrics.get("accuracy"),
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            f1_score=metrics.get("f1"),
            roc_auc=metrics.get("roc_auc"),
            training_samples=training_samples,
            is_active=is_active
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    def get_active_model(self, db: Session) -> Optional[ModelMetrics]:
        """Get the currently active model."""
        return db.query(ModelMetrics).filter(ModelMetrics.is_active == True).first()

    def set_active_model(self, db: Session, model_name: str) -> bool:
        """
        Set a model as the active model.

        Args:
            db: Database session
            model_name: Name of model to activate

        Returns:
            True if successful
        """
        # Deactivate all models
        db.query(ModelMetrics).update({ModelMetrics.is_active: False})

        # Activate specified model
        model = db.query(ModelMetrics).filter(ModelMetrics.model_name == model_name).first()
        if model:
            model.is_active = True
            db.commit()
            return True
        return False

    def add_feedback(
        self,
        db: Session,
        prediction_id: int,
        actual_outcome: Optional[int] = None,
        feedback_notes: Optional[str] = None
    ) -> CustomerFeedback:
        """
        Add feedback for a prediction.

        Args:
            db: Database session
            prediction_id: ID of prediction
            actual_outcome: Actual churn outcome
            feedback_notes: Additional notes

        Returns:
            Created CustomerFeedback record
        """
        feedback = CustomerFeedback(
            prediction_id=prediction_id,
            actual_outcome=actual_outcome,
            feedback_notes=feedback_notes
        )
        db.add(feedback)
        db.commit()
        db.refresh(feedback)
        return feedback


# Create global database manager instance
db_manager = DatabaseManager()
