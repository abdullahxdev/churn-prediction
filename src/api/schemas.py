"""
API Schemas (Pydantic Models)
=============================

Data validation models for API requests and responses.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class CustomerData(BaseModel):
    """Schema for customer data input."""

    # Required fields
    Tenure: int = Field(..., ge=0, le=100, description="Months since customer joined")
    WarehouseToHome: float = Field(..., ge=0, description="Distance from warehouse to home")
    HourSpendOnApp: float = Field(..., ge=0, description="Hours spent on mobile app")
    NumberOfDeviceRegistered: int = Field(..., ge=1, le=10, description="Number of devices registered")
    NumberOfAddress: int = Field(..., ge=1, description="Number of addresses registered")
    OrderAmountHikeFromlastYear: float = Field(..., description="Percentage increase in order amount")
    CouponUsed: int = Field(..., ge=0, description="Number of coupons used")
    OrderCount: int = Field(..., ge=0, description="Total number of orders")
    DaySinceLastOrder: int = Field(..., ge=0, description="Days since last order")
    CashbackAmount: float = Field(..., ge=0, description="Total cashback amount received")

    # Categorical fields
    PreferredLoginDevice: str = Field(..., description="Preferred login device")
    CityTier: int = Field(..., ge=1, le=3, description="City tier (1, 2, or 3)")
    PreferredPaymentMode: str = Field(..., description="Preferred payment mode")
    Gender: str = Field(..., description="Customer gender")
    PreferedOrderCat: str = Field(..., description="Preferred order category")
    SatisfactionScore: int = Field(..., ge=1, le=5, description="Satisfaction score (1-5)")
    MaritalStatus: str = Field(..., description="Marital status")
    Complain: int = Field(..., ge=0, le=1, description="Has complained (0 or 1)")

    # Optional customer ID
    CustomerID: Optional[str] = Field(None, description="Customer identifier")

    @field_validator("PreferredLoginDevice")
    @classmethod
    def validate_login_device(cls, v):
        allowed = ["Mobile Phone", "Phone", "Computer"]
        if v not in allowed:
            raise ValueError(f"PreferredLoginDevice must be one of {allowed}")
        return v

    @field_validator("PreferredPaymentMode")
    @classmethod
    def validate_payment_mode(cls, v):
        allowed = ["Debit Card", "Credit Card", "E wallet", "COD", "UPI", "CC"]
        if v not in allowed:
            raise ValueError(f"PreferredPaymentMode must be one of {allowed}")
        return v

    @field_validator("Gender")
    @classmethod
    def validate_gender(cls, v):
        allowed = ["Male", "Female"]
        if v not in allowed:
            raise ValueError(f"Gender must be one of {allowed}")
        return v

    @field_validator("PreferedOrderCat")
    @classmethod
    def validate_order_cat(cls, v):
        allowed = ["Laptop & Accessory", "Mobile", "Mobile Phone", "Fashion", "Grocery", "Others"]
        if v not in allowed:
            raise ValueError(f"PreferedOrderCat must be one of {allowed}")
        return v

    @field_validator("MaritalStatus")
    @classmethod
    def validate_marital_status(cls, v):
        allowed = ["Single", "Married", "Divorced"]
        if v not in allowed:
            raise ValueError(f"MaritalStatus must be one of {allowed}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "Tenure": 12,
                "WarehouseToHome": 15.5,
                "HourSpendOnApp": 3.0,
                "NumberOfDeviceRegistered": 4,
                "NumberOfAddress": 2,
                "OrderAmountHikeFromlastYear": 15.0,
                "CouponUsed": 2,
                "OrderCount": 5,
                "DaySinceLastOrder": 7,
                "CashbackAmount": 150.0,
                "PreferredLoginDevice": "Mobile Phone",
                "CityTier": 1,
                "PreferredPaymentMode": "Debit Card",
                "Gender": "Male",
                "PreferedOrderCat": "Laptop & Accessory",
                "SatisfactionScore": 3,
                "MaritalStatus": "Single",
                "Complain": 0,
                "CustomerID": "CUST_001"
            }
        }


class PredictionResponse(BaseModel):
    """Schema for single prediction response."""

    customer_id: Optional[str] = Field(None, description="Customer identifier")
    churn_prediction: int = Field(..., description="Predicted churn (0 or 1)")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    risk_level: str = Field(..., description="Risk level category")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    top_risk_factors: Optional[List[Dict[str, Union[str, float]]]] = Field(
        None, description="Top factors contributing to prediction"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    model_used: str = Field(..., description="Model used for prediction")

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "churn_prediction": 1,
                "churn_probability": 0.75,
                "risk_level": "High",
                "confidence": 0.85,
                "top_risk_factors": [
                    {"feature": "DaySinceLastOrder", "impact": "Increases Churn Risk", "shap_value": 0.23},
                    {"feature": "SatisfactionScore", "impact": "Increases Churn Risk", "shap_value": 0.18}
                ],
                "timestamp": "2024-12-19T10:30:00",
                "model_used": "xgboost"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request."""

    customers: List[CustomerData] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""

    predictions: List[PredictionResponse]
    total_customers: int
    churn_count: int
    non_churn_count: int
    average_churn_probability: float
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    processing_time_ms: float


class ModelInfo(BaseModel):
    """Schema for model information."""

    model_name: str
    model_type: str
    version: str
    training_date: Optional[str]
    metrics: Dict[str, float]
    feature_count: int
    is_active: bool


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    database_connected: bool
    timestamp: datetime


class PredictionHistoryItem(BaseModel):
    """Schema for prediction history item."""

    id: int
    customer_id: Optional[str]
    prediction: int
    probability: float
    risk_level: str
    model_used: str
    timestamp: datetime
    input_features: Optional[Dict]
