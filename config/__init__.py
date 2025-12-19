"""Configuration module for Churn Prediction System."""

import os
from pathlib import Path

import yaml

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.absolute()

# Load configuration
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def load_config() -> dict:
    """Load configuration from YAML file."""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_config() -> dict:
    """Get configuration dictionary."""
    return load_config()


# Export commonly used paths
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models" / "saved"
MLFLOW_DIR = ROOT_DIR / "models" / "mlflow"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, MLFLOW_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
