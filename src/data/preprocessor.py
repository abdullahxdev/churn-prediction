"""
Data Preprocessor Module
========================

Handles data cleaning, transformation, and preparation for modeling.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from config import get_config


class DataPreprocessor:
    """Preprocess data for churn prediction models."""

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize DataPreprocessor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config()
        self.feature_config = self.config.get("features", {})
        self.numerical_features = self.feature_config.get("numerical", [])
        self.categorical_features = self.feature_config.get("categorical", [])
        self.drop_columns = self.feature_config.get("drop_columns", [])

        self.preprocessor = None
        self.label_encoders = {}
        self.feature_names = []

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        logger.info("Starting data cleaning...")

        # Drop specified columns
        cols_to_drop = [col for col in self.drop_columns if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"Dropped columns: {cols_to_drop}")

        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            logger.info(f"Removed {dropped_rows} duplicate rows")

        # Handle missing values info
        missing_info = df.isnull().sum()
        if missing_info.any():
            logger.info(f"Missing values found:\n{missing_info[missing_info > 0]}")

        logger.info(f"Data cleaned: {len(df)} rows, {len(df.columns)} columns")
        return df

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        numerical_strategy: str = "median",
        categorical_strategy: str = "most_frequent"
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: DataFrame with missing values
            numerical_strategy: Strategy for numerical columns ('mean', 'median', 'most_frequent')
            categorical_strategy: Strategy for categorical columns ('most_frequent', 'constant')

        Returns:
            DataFrame with imputed values
        """
        df = df.copy()

        # Identify columns with missing values
        numerical_cols = [col for col in self.numerical_features if col in df.columns]
        categorical_cols = [col for col in self.categorical_features if col in df.columns]

        # Impute numerical columns
        for col in numerical_cols:
            if df[col].isnull().any():
                if numerical_strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif numerical_strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif numerical_strategy == "most_frequent":
                    df[col] = df[col].fillna(df[col].mode()[0])
                logger.debug(f"Imputed {col} with {numerical_strategy}")

        # Impute categorical columns
        for col in categorical_cols:
            if df[col].isnull().any():
                if categorical_strategy == "most_frequent":
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif categorical_strategy == "constant":
                    df[col] = df[col].fillna("Unknown")
                logger.debug(f"Imputed {col} with {categorical_strategy}")

        return df

    def handle_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 1.5,
        action: str = "clip"
    ) -> pd.DataFrame:
        """
        Handle outliers in numerical features.

        Args:
            df: Input DataFrame
            method: Detection method ('iqr', 'zscore')
            threshold: Threshold for detection
            action: Action to take ('clip', 'remove', 'none')

        Returns:
            DataFrame with handled outliers
        """
        df = df.copy()
        numerical_cols = [col for col in self.numerical_features if col in df.columns]

        for col in numerical_cols:
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                if action == "clip":
                    df[col] = df[col].clip(lower_bound, upper_bound)
                elif action == "remove":
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

            elif method == "zscore":
                from scipy import stats
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                if action == "clip":
                    mean, std = df[col].mean(), df[col].std()
                    df[col] = df[col].clip(
                        mean - threshold * std,
                        mean + threshold * std
                    )
                elif action == "remove":
                    mask = z_scores < threshold
                    df = df[mask]

        return df

    def create_preprocessing_pipeline(
        self,
        numerical_scaler: str = "standard",
        categorical_encoder: str = "onehot"
    ) -> ColumnTransformer:
        """
        Create sklearn preprocessing pipeline.

        Args:
            numerical_scaler: Scaler for numerical features ('standard', 'minmax', 'none')
            categorical_encoder: Encoder for categorical features ('onehot', 'ordinal', 'label')

        Returns:
            ColumnTransformer pipeline
        """
        # Define numerical pipeline
        numerical_steps = [("imputer", SimpleImputer(strategy="median"))]

        if numerical_scaler == "standard":
            numerical_steps.append(("scaler", StandardScaler()))
        elif numerical_scaler == "minmax":
            numerical_steps.append(("scaler", MinMaxScaler()))

        numerical_pipeline = Pipeline(numerical_steps)

        # Define categorical pipeline
        categorical_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent"))
        ]

        if categorical_encoder == "onehot":
            categorical_steps.append(
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            )
        elif categorical_encoder == "ordinal":
            categorical_steps.append(
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            )

        categorical_pipeline = Pipeline(categorical_steps)

        # Combine pipelines
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("numerical", numerical_pipeline, self.numerical_features),
                ("categorical", categorical_pipeline, self.categorical_features)
            ],
            remainder="drop"
        )

        return self.preprocessor

    def fit_transform(
        self,
        df: pd.DataFrame,
        numerical_scaler: str = "standard",
        categorical_encoder: str = "onehot"
    ) -> np.ndarray:
        """
        Fit preprocessor and transform data.

        Args:
            df: Input DataFrame
            numerical_scaler: Scaler type
            categorical_encoder: Encoder type

        Returns:
            Transformed numpy array
        """
        # Filter to existing columns
        available_numerical = [col for col in self.numerical_features if col in df.columns]
        available_categorical = [col for col in self.categorical_features if col in df.columns]

        self.numerical_features = available_numerical
        self.categorical_features = available_categorical

        logger.info(f"Numerical features: {self.numerical_features}")
        logger.info(f"Categorical features: {self.categorical_features}")

        # Create and fit pipeline
        self.create_preprocessing_pipeline(numerical_scaler, categorical_encoder)
        transformed = self.preprocessor.fit_transform(df)

        # Store feature names
        self._extract_feature_names()

        logger.info(f"Transformed shape: {transformed.shape}")
        return transformed

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor.

        Args:
            df: Input DataFrame

        Returns:
            Transformed numpy array
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")

        return self.preprocessor.transform(df)

    def _extract_feature_names(self):
        """Extract feature names after transformation."""
        self.feature_names = []

        # Get numerical feature names
        self.feature_names.extend(self.numerical_features)

        # Get categorical feature names (one-hot encoded)
        if hasattr(self.preprocessor, "named_transformers_"):
            cat_transformer = self.preprocessor.named_transformers_.get("categorical")
            if cat_transformer is not None:
                encoder = cat_transformer.named_steps.get("encoder")
                if encoder is not None and hasattr(encoder, "get_feature_names_out"):
                    cat_features = encoder.get_feature_names_out(self.categorical_features)
                    self.feature_names.extend(cat_features)

    def get_feature_names(self) -> List[str]:
        """Get names of all features after transformation."""
        return self.feature_names

    def inverse_transform_target(
        self,
        y: np.ndarray,
        target_name: str = "Churn"
    ) -> pd.Series:
        """
        Inverse transform target variable.

        Args:
            y: Encoded target array
            target_name: Name of target column

        Returns:
            Original target values
        """
        if target_name in self.label_encoders:
            return pd.Series(
                self.label_encoders[target_name].inverse_transform(y),
                name=target_name
            )
        return pd.Series(y, name=target_name)

    def get_preprocessing_summary(self) -> Dict:
        """
        Get summary of preprocessing steps applied.

        Returns:
            Dictionary with preprocessing summary
        """
        return {
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "dropped_columns": self.drop_columns,
            "total_features": len(self.feature_names),
            "feature_names": self.feature_names
        }
