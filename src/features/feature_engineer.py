"""
Feature Engineering Module
==========================

Advanced feature engineering for churn prediction.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
    RFE,
)
from sklearn.ensemble import RandomForestClassifier

from config import get_config


class FeatureEngineer:
    """Create and select features for churn prediction."""

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize FeatureEngineer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config()
        self.created_features = []
        self.selected_features = []
        self.feature_importance = {}

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        logger.info("Starting feature engineering...")

        # Create different feature types
        df = self.create_tenure_features(df)
        df = self.create_engagement_features(df)
        df = self.create_value_features(df)
        df = self.create_satisfaction_features(df)
        df = self.create_interaction_features(df)
        df = self.create_ratio_features(df)

        logger.info(f"Created {len(self.created_features)} new features")
        return df

    def create_tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tenure-based features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with tenure features
        """
        if "Tenure" in df.columns:
            # Tenure categories
            df["TenureCategory"] = pd.cut(
                df["Tenure"],
                bins=[0, 6, 12, 24, float("inf")],
                labels=["New", "Developing", "Established", "Loyal"]
            ).astype(str)
            self.created_features.append("TenureCategory")

            # Is new customer
            df["IsNewCustomer"] = (df["Tenure"] <= 6).astype(int)
            self.created_features.append("IsNewCustomer")

            # Tenure squared (polynomial feature)
            df["TenureSquared"] = df["Tenure"] ** 2
            self.created_features.append("TenureSquared")

            logger.debug("Created tenure features")

        return df

    def create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engagement-based features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engagement features
        """
        # App usage score
        if "HourSpendOnApp" in df.columns:
            df["HighAppUsage"] = (df["HourSpendOnApp"] >= 3).astype(int)
            self.created_features.append("HighAppUsage")

            # Log transform for skewed data
            df["LogHourSpendOnApp"] = np.log1p(df["HourSpendOnApp"])
            self.created_features.append("LogHourSpendOnApp")

        # Device engagement
        if "NumberOfDeviceRegistered" in df.columns:
            df["MultiDevice"] = (df["NumberOfDeviceRegistered"] > 3).astype(int)
            self.created_features.append("MultiDevice")

        # Order frequency
        if "OrderCount" in df.columns:
            df["FrequentBuyer"] = (df["OrderCount"] >= 3).astype(int)
            self.created_features.append("FrequentBuyer")

            df["LogOrderCount"] = np.log1p(df["OrderCount"])
            self.created_features.append("LogOrderCount")

        # Days since last order
        if "DaySinceLastOrder" in df.columns:
            df["RecentlyActive"] = (df["DaySinceLastOrder"] <= 7).astype(int)
            self.created_features.append("RecentlyActive")

            df["Inactive30Days"] = (df["DaySinceLastOrder"] > 30).astype(int)
            self.created_features.append("Inactive30Days")

        logger.debug("Created engagement features")
        return df

    def create_value_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create value-based features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with value features
        """
        # Cashback features
        if "CashbackAmount" in df.columns:
            df["HighCashback"] = (df["CashbackAmount"] >= df["CashbackAmount"].median()).astype(int)
            self.created_features.append("HighCashback")

            df["LogCashback"] = np.log1p(df["CashbackAmount"])
            self.created_features.append("LogCashback")

        # Order value growth
        if "OrderAmountHikeFromlastYear" in df.columns:
            df["PositiveGrowth"] = (df["OrderAmountHikeFromlastYear"] > 15).astype(int)
            self.created_features.append("PositiveGrowth")

            df["NegativeGrowth"] = (df["OrderAmountHikeFromlastYear"] < 10).astype(int)
            self.created_features.append("NegativeGrowth")

        # Coupon usage
        if "CouponUsed" in df.columns:
            df["CouponUser"] = (df["CouponUsed"] > 0).astype(int)
            self.created_features.append("CouponUser")

            df["HeavyCouponUser"] = (df["CouponUsed"] >= 2).astype(int)
            self.created_features.append("HeavyCouponUser")

        logger.debug("Created value features")
        return df

    def create_satisfaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create satisfaction-based features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with satisfaction features
        """
        # Satisfaction categories
        if "SatisfactionScore" in df.columns:
            df["Satisfied"] = (df["SatisfactionScore"] >= 4).astype(int)
            self.created_features.append("Satisfied")

            df["Dissatisfied"] = (df["SatisfactionScore"] <= 2).astype(int)
            self.created_features.append("Dissatisfied")

        # Complaint features
        if "Complain" in df.columns:
            df["HasComplained"] = df["Complain"].astype(int)
            self.created_features.append("HasComplained")

        # Combined satisfaction risk
        if "SatisfactionScore" in df.columns and "Complain" in df.columns:
            df["HighRiskCustomer"] = (
                (df["SatisfactionScore"] <= 2) & (df["Complain"] == 1)
            ).astype(int)
            self.created_features.append("HighRiskCustomer")

        logger.debug("Created satisfaction features")
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between existing features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with interaction features
        """
        # Tenure * Satisfaction interaction
        if "Tenure" in df.columns and "SatisfactionScore" in df.columns:
            df["TenureSatisfaction"] = df["Tenure"] * df["SatisfactionScore"]
            self.created_features.append("TenureSatisfaction")

        # App usage * Order count interaction
        if "HourSpendOnApp" in df.columns and "OrderCount" in df.columns:
            df["EngagementScore"] = df["HourSpendOnApp"] * df["OrderCount"]
            self.created_features.append("EngagementScore")

        # Value * Engagement interaction
        if "CashbackAmount" in df.columns and "OrderCount" in df.columns:
            df["ValueEngagement"] = df["CashbackAmount"] * df["OrderCount"]
            self.created_features.append("ValueEngagement")

        # Distance * Order interaction
        if "WarehouseToHome" in df.columns and "OrderCount" in df.columns:
            df["DistanceOrderRatio"] = df["WarehouseToHome"] / (df["OrderCount"] + 1)
            self.created_features.append("DistanceOrderRatio")

        logger.debug("Created interaction features")
        return df

    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ratio-based features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with ratio features
        """
        # Cashback per order
        if "CashbackAmount" in df.columns and "OrderCount" in df.columns:
            df["CashbackPerOrder"] = df["CashbackAmount"] / (df["OrderCount"] + 1)
            self.created_features.append("CashbackPerOrder")

        # Coupons per order
        if "CouponUsed" in df.columns and "OrderCount" in df.columns:
            df["CouponPerOrder"] = df["CouponUsed"] / (df["OrderCount"] + 1)
            self.created_features.append("CouponPerOrder")

        # App time per order
        if "HourSpendOnApp" in df.columns and "OrderCount" in df.columns:
            df["AppTimePerOrder"] = df["HourSpendOnApp"] / (df["OrderCount"] + 1)
            self.created_features.append("AppTimePerOrder")

        # Orders per tenure month
        if "OrderCount" in df.columns and "Tenure" in df.columns:
            df["OrdersPerMonth"] = df["OrderCount"] / (df["Tenure"] + 1)
            self.created_features.append("OrdersPerMonth")

        logger.debug("Created ratio features")
        return df

    def select_features_univariate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        k: int = 20,
        method: str = "f_classif"
    ) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
        """
        Select features using univariate statistical tests.

        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            k: Number of features to select
            method: Selection method ('f_classif', 'chi2', 'mutual_info')

        Returns:
            Tuple of (selected features, feature names, scores)
        """
        if method == "f_classif":
            selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
        elif method == "chi2":
            # chi2 requires non-negative features
            X_non_neg = X - X.min(axis=0)
            selector = SelectKBest(chi2, k=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X_non_neg, y)
            scores = dict(zip(feature_names, selector.scores_))
            selected_mask = selector.get_support()
            selected_names = [f for f, m in zip(feature_names, selected_mask) if m]
            return X_selected, selected_names, scores
        elif method == "mutual_info":
            selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
        else:
            raise ValueError(f"Unknown method: {method}")

        X_selected = selector.fit_transform(X, y)
        scores = dict(zip(feature_names, selector.scores_))
        selected_mask = selector.get_support()
        selected_names = [f for f, m in zip(feature_names, selected_mask) if m]

        logger.info(f"Selected {len(selected_names)} features using {method}")
        return X_selected, selected_names, scores

    def select_features_rfe(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_features: int = 15,
        step: int = 1
    ) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
        """
        Select features using Recursive Feature Elimination.

        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            n_features: Number of features to select
            step: Number of features to remove at each iteration

        Returns:
            Tuple of (selected features, feature names, rankings)
        """
        estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        selector = RFE(
            estimator=estimator,
            n_features_to_select=min(n_features, X.shape[1]),
            step=step
        )

        X_selected = selector.fit_transform(X, y)
        rankings = dict(zip(feature_names, selector.ranking_))
        selected_mask = selector.get_support()
        selected_names = [f for f, m in zip(feature_names, selected_mask) if m]

        logger.info(f"Selected {len(selected_names)} features using RFE")
        return X_selected, selected_names, rankings

    def get_feature_importance_rf(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Get feature importance using Random Forest.

        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names

        Returns:
            Dictionary of feature importance scores
        """
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)

        importance = dict(zip(feature_names, rf.feature_importances_))
        self.feature_importance = importance

        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        return importance

    def get_created_features(self) -> List[str]:
        """Get list of created feature names."""
        return self.created_features

    def get_feature_engineering_summary(self) -> Dict:
        """
        Get summary of feature engineering.

        Returns:
            Dictionary with summary
        """
        return {
            "created_features": self.created_features,
            "num_created": len(self.created_features),
            "selected_features": self.selected_features,
            "feature_importance": self.feature_importance
        }
