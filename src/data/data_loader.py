"""
Data Loader Module
==================

Handles loading data from various sources and basic validation.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
from loguru import logger

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, get_config


class DataLoader:
    """Load and manage datasets for churn prediction."""

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize DataLoader.

        Args:
            config: Configuration dictionary. If None, loads from config.yaml
        """
        self.config = config or get_config()
        self.raw_data_path = RAW_DATA_DIR
        self.processed_data_path = PROCESSED_DATA_DIR

    def load_raw_data(
        self,
        filename: str = "ecommerce_customer_churn.csv",
        **kwargs
    ) -> pd.DataFrame:
        """
        Load raw data from CSV file.

        Args:
            filename: Name of the data file
            **kwargs: Additional arguments to pass to pd.read_csv

        Returns:
            DataFrame containing raw data
        """
        file_path = self.raw_data_path / filename

        # Try different file extensions
        if not file_path.exists():
            for ext in [".csv", ".xlsx", ".xls", ".parquet"]:
                alt_path = self.raw_data_path / f"{filename.split('.')[0]}{ext}"
                if alt_path.exists():
                    file_path = alt_path
                    break

        if not file_path.exists():
            logger.error(f"Data file not found: {file_path}")
            raise FileNotFoundError(f"Data file not found: {file_path}")

        logger.info(f"Loading data from {file_path}")

        # Load based on file extension
        ext = file_path.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(file_path, **kwargs)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, **kwargs)
        elif ext == ".parquet":
            df = pd.read_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df

    def load_processed_data(
        self,
        filename: str = "processed_data.parquet"
    ) -> pd.DataFrame:
        """
        Load processed data.

        Args:
            filename: Name of the processed data file

        Returns:
            DataFrame containing processed data
        """
        file_path = self.processed_data_path / filename

        if not file_path.exists():
            logger.error(f"Processed data not found: {file_path}")
            raise FileNotFoundError(f"Processed data not found: {file_path}")

        logger.info(f"Loading processed data from {file_path}")

        ext = file_path.suffix.lower()
        if ext == ".parquet":
            df = pd.read_parquet(file_path)
        elif ext == ".csv":
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        return df

    def save_processed_data(
        self,
        df: pd.DataFrame,
        filename: str = "processed_data.parquet"
    ) -> Path:
        """
        Save processed data.

        Args:
            df: DataFrame to save
            filename: Output filename

        Returns:
            Path to saved file
        """
        file_path = self.processed_data_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving processed data to {file_path}")

        ext = file_path.suffix.lower()
        if ext == ".parquet":
            df.to_parquet(file_path, index=False)
        elif ext == ".csv":
            df.to_csv(file_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        return file_path

    def get_train_test_split(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        test_size: Optional[float] = None,
        val_size: Optional[float] = None,
        random_state: Optional[int] = None,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: Input DataFrame
            target_col: Name of target column
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed
            stratify: Whether to stratify split

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        from sklearn.model_selection import train_test_split

        # Get parameters from config if not provided
        data_config = self.config.get("data", {})
        target_col = target_col or data_config.get("target_column", "Churn")
        test_size = test_size or data_config.get("test_size", 0.2)
        val_size = val_size or data_config.get("val_size", 0.1)
        random_state = random_state or data_config.get("random_state", 42)

        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Stratify if requested
        strat = y if stratify else None

        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=strat
        )

        # Second split: train and val
        val_adjusted = val_size / (1 - test_size)
        strat_temp = y_temp if stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_adjusted,
            random_state=random_state,
            stratify=strat_temp
        )

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Validate data quality.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "duplicates": df.duplicated().sum(),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }

        # Check for target column
        target_col = self.config.get("data", {}).get("target_column", "Churn")
        if target_col in df.columns:
            validation_results["target_distribution"] = df[target_col].value_counts().to_dict()
            validation_results["target_balance"] = df[target_col].value_counts(normalize=True).to_dict()

        return validation_results
