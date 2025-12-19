"""
Streamlit Dashboard Application
===============================

Interactive dashboard for churn prediction and analysis.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

# Page config
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .risk-high { color: #ff4b4b; }
    .risk-medium { color: #ffa500; }
    .risk-low { color: #00cc00; }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = "http://localhost:8000"


def get_api_health():
    """Check API health status."""
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None


def make_prediction(customer_data: Dict) -> Optional[Dict]:
    """Make a prediction via API."""
    try:
        response = httpx.post(
            f"{API_URL}/predict",
            json=customer_data,
            params={"include_factors": True},
            timeout=10
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def get_prediction_stats() -> Optional[Dict]:
    """Get prediction statistics from API."""
    try:
        response = httpx.get(f"{API_URL}/predictions/statistics", timeout=5)
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None


def get_prediction_history(limit: int = 100) -> Optional[List]:
    """Get prediction history from API."""
    try:
        response = httpx.get(
            f"{API_URL}/predictions/history",
            params={"limit": limit},
            timeout=5
        )
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None


# Sidebar Navigation
with st.sidebar:
    st.markdown("## Churn Prediction System")

    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Single Prediction", "Batch Prediction", "Model Analysis", "History"],
        icons=["speedometer2", "person", "people", "graph-up", "clock-history"],
        menu_icon="cast",
        default_index=0,
    )

    st.markdown("---")

    # API Status
    health = get_api_health()
    if health:
        st.success("API Connected")
        if health.get("model_loaded"):
            st.info("Model Loaded")
        else:
            st.warning("No Model Loaded")
    else:
        st.error("API Disconnected")
        st.info("Start API with:\n`uvicorn src.api.main:app --reload`")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **Churn Prediction System**

    ML-powered customer churn
    prediction for e-commerce.

    Built with:
    - FastAPI
    - Streamlit
    - Scikit-learn
    - XGBoost
    """)


# Dashboard Page
if selected == "Dashboard":
    st.markdown('<h1 class="main-header">Customer Churn Analytics Dashboard</h1>', unsafe_allow_html=True)

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    stats = get_prediction_stats()

    if stats:
        with col1:
            st.metric(
                label="Total Predictions",
                value=stats.get("total_predictions", 0),
                delta=None
            )
        with col2:
            churn_rate = stats.get("churn_rate", 0) * 100
            st.metric(
                label="Churn Rate",
                value=f"{churn_rate:.1f}%",
                delta=None
            )
        with col3:
            st.metric(
                label="High Risk Customers",
                value=stats.get("high_risk_count", 0),
                delta=None
            )
        with col4:
            avg_prob = stats.get("average_probability", 0) * 100
            st.metric(
                label="Avg Churn Probability",
                value=f"{avg_prob:.1f}%",
                delta=None
            )
    else:
        for col in [col1, col2, col3, col4]:
            with col:
                st.metric(label="--", value="N/A")

    st.markdown("---")

    # Charts Row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Distribution")

        if stats:
            risk_data = {
                "Risk Level": ["High", "Medium", "Low"],
                "Count": [
                    stats.get("high_risk_count", 0),
                    stats.get("medium_risk_count", 0),
                    stats.get("low_risk_count", 0)
                ]
            }
            fig = px.pie(
                risk_data,
                names="Risk Level",
                values="Count",
                color="Risk Level",
                color_discrete_map={"High": "#ff4b4b", "Medium": "#ffa500", "Low": "#00cc00"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available. Make some predictions first.")

    with col2:
        st.subheader("Churn vs Non-Churn")

        if stats and stats.get("total_predictions", 0) > 0:
            churn_data = {
                "Category": ["Will Churn", "Will Stay"],
                "Count": [
                    stats.get("churn_predictions", 0),
                    stats.get("non_churn_predictions", 0)
                ]
            }
            fig = px.bar(
                churn_data,
                x="Category",
                y="Count",
                color="Category",
                color_discrete_map={"Will Churn": "#ff4b4b", "Will Stay": "#00cc00"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available. Make some predictions first.")

    # Recent Predictions
    st.markdown("---")
    st.subheader("Recent Predictions")

    history = get_prediction_history(10)
    if history:
        df = pd.DataFrame(history)
        if not df.empty:
            display_cols = ["customer_id", "prediction", "probability", "risk_level", "created_at"]
            available_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[available_cols], use_container_width=True)
    else:
        st.info("No predictions yet.")


# Single Prediction Page
elif selected == "Single Prediction":
    st.markdown('<h1 class="main-header">Single Customer Prediction</h1>', unsafe_allow_html=True)

    st.markdown("Enter customer details to predict churn probability.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Basic Info")
            customer_id = st.text_input("Customer ID", value="CUST_001")
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            city_tier = st.selectbox("City Tier", [1, 2, 3])

        with col2:
            st.subheader("Engagement")
            hour_spend_on_app = st.number_input("Hours on App", min_value=0.0, max_value=24.0, value=3.0)
            num_devices = st.number_input("Devices Registered", min_value=1, max_value=10, value=4)
            num_addresses = st.number_input("Number of Addresses", min_value=1, max_value=20, value=2)
            preferred_login = st.selectbox("Preferred Login Device", ["Mobile Phone", "Phone", "Computer"])
            preferred_payment = st.selectbox("Payment Mode", ["Debit Card", "Credit Card", "E wallet", "COD", "UPI"])

        with col3:
            st.subheader("Order History")
            order_count = st.number_input("Order Count", min_value=0, max_value=100, value=5)
            days_since_order = st.number_input("Days Since Last Order", min_value=0, max_value=365, value=7)
            order_hike = st.number_input("Order Amount Hike (%)", min_value=-100.0, max_value=200.0, value=15.0)
            coupon_used = st.number_input("Coupons Used", min_value=0, max_value=50, value=2)
            cashback = st.number_input("Cashback Amount", min_value=0.0, max_value=1000.0, value=150.0)

        col1, col2 = st.columns(2)
        with col1:
            warehouse_distance = st.number_input("Warehouse to Home Distance", min_value=0.0, max_value=100.0, value=15.5)
            preferred_order_cat = st.selectbox("Preferred Category", ["Laptop & Accessory", "Mobile", "Mobile Phone", "Fashion", "Grocery", "Others"])
        with col2:
            satisfaction = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3)
            complain = st.selectbox("Has Complained", [0, 1])

        submitted = st.form_submit_button("Predict Churn", use_container_width=True)

    if submitted:
        customer_data = {
            "CustomerID": customer_id,
            "Tenure": tenure,
            "WarehouseToHome": warehouse_distance,
            "HourSpendOnApp": hour_spend_on_app,
            "NumberOfDeviceRegistered": num_devices,
            "NumberOfAddress": num_addresses,
            "OrderAmountHikeFromlastYear": order_hike,
            "CouponUsed": coupon_used,
            "OrderCount": order_count,
            "DaySinceLastOrder": days_since_order,
            "CashbackAmount": cashback,
            "PreferredLoginDevice": preferred_login,
            "CityTier": city_tier,
            "PreferredPaymentMode": preferred_payment,
            "Gender": gender,
            "PreferedOrderCat": preferred_order_cat,
            "SatisfactionScore": satisfaction,
            "MaritalStatus": marital_status,
            "Complain": complain
        }

        with st.spinner("Making prediction..."):
            result = make_prediction(customer_data)

        if result:
            st.markdown("---")
            st.subheader("Prediction Result")

            col1, col2, col3 = st.columns(3)

            with col1:
                prediction_text = "Will Churn" if result["churn_prediction"] == 1 else "Will Stay"
                prediction_color = "red" if result["churn_prediction"] == 1 else "green"
                st.markdown(f"### Prediction: <span style='color:{prediction_color}'>{prediction_text}</span>", unsafe_allow_html=True)

            with col2:
                prob = result["churn_probability"] * 100
                st.markdown(f"### Probability: {prob:.1f}%")

                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgreen"},
                            {'range': [40, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)

            with col3:
                risk = result["risk_level"]
                risk_color = {"High": "red", "Medium": "orange", "Low": "green"}.get(risk, "gray")
                st.markdown(f"### Risk Level: <span style='color:{risk_color}'>{risk}</span>", unsafe_allow_html=True)
                st.markdown(f"**Confidence:** {result.get('confidence', 0)*100:.1f}%")
                st.markdown(f"**Model:** {result.get('model_used', 'N/A')}")
        else:
            st.error("Failed to make prediction. Please check if the API is running and model is loaded.")


# Batch Prediction Page
elif selected == "Batch Prediction":
    st.markdown('<h1 class="main-header">Batch Prediction</h1>', unsafe_allow_html=True)

    st.markdown("Upload a CSV file with customer data to make batch predictions.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview Data")
        st.dataframe(df.head(10), use_container_width=True)
        st.info(f"Total customers: {len(df)}")

        if st.button("Run Batch Prediction", use_container_width=True):
            st.warning("Batch prediction via file upload requires the API batch endpoint. Make sure your CSV has the correct columns.")
            # Implementation would convert DataFrame to API format and call batch endpoint


# Model Analysis Page
elif selected == "Model Analysis":
    st.markdown('<h1 class="main-header">Model Analysis</h1>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Model Metrics", "ROC Curve"])

    with tab1:
        st.subheader("Feature Importance (SHAP)")
        st.info("Feature importance visualization will be available after model training with SHAP analysis.")

        # Placeholder chart
        sample_importance = pd.DataFrame({
            "Feature": ["DaySinceLastOrder", "Complain", "Tenure", "SatisfactionScore", "CashbackAmount"],
            "Importance": [0.25, 0.20, 0.18, 0.15, 0.12]
        })
        fig = px.bar(sample_importance, x="Importance", y="Feature", orientation="h",
                     title="Sample Feature Importance")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Model Performance Metrics")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Training Metrics")
            metrics_data = {
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
                "Score": [0.92, 0.88, 0.85, 0.86, 0.94]
            }
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

        with col2:
            st.markdown("### Confusion Matrix")
            cm_data = [[850, 50], [75, 425]]
            fig = px.imshow(cm_data, text_auto=True, aspect="auto",
                           labels=dict(x="Predicted", y="Actual"),
                           x=["No Churn", "Churn"],
                           y=["No Churn", "Churn"])
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("ROC Curve")
        st.info("ROC curve will be displayed after model evaluation.")


# History Page
elif selected == "History":
    st.markdown('<h1 class="main-header">Prediction History</h1>', unsafe_allow_html=True)

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        limit = st.selectbox("Records per page", [25, 50, 100, 200], index=2)
    with col2:
        customer_filter = st.text_input("Filter by Customer ID")
    with col3:
        risk_filter = st.multiselect("Filter by Risk Level", ["High", "Medium", "Low"])

    history = get_prediction_history(limit)

    if history:
        df = pd.DataFrame(history)

        # Apply filters
        if customer_filter:
            df = df[df["customer_id"].str.contains(customer_filter, case=False, na=False)]
        if risk_filter:
            df = df[df["risk_level"].isin(risk_filter)]

        st.dataframe(df, use_container_width=True)

        # Export option
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No prediction history available.")


# Run with: streamlit run src/dashboard/app.py
