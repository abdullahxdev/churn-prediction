"""
Utility Helper Functions
========================

Common utility functions used across the project.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from loguru import logger

from config import ROOT_DIR


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "7 days"
):
    """
    Setup logging configuration.

    Args:
        level: Logging level
        log_file: Optional log file path
        rotation: Log rotation setting
        retention: Log retention setting
    """
    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # Add file handler if specified
    if log_file:
        log_path = ROOT_DIR / "logs" / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_path,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=rotation,
            retention=retention,
            compression="zip"
        )

    logger.info(f"Logging configured at {level} level")


def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get current timestamp string.

    Args:
        format_str: Datetime format string

    Returns:
        Formatted timestamp
    """
    return datetime.now().strftime(format_str)


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> Dict[str, str]:
    """
    Format metric values for display.

    Args:
        metrics: Dictionary of metric values
        precision: Decimal precision

    Returns:
        Dictionary with formatted values
    """
    return {k: f"{v:.{precision}f}" for k, v in metrics.items()}


def create_directories(paths: List[Union[str, Path]]):
    """
    Create directories if they don't exist.

    Args:
        paths: List of directory paths
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division handling zero denominator.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if denominator is zero

    Returns:
        Division result or default
    """
    return numerator / denominator if denominator != 0 else default


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.

    Args:
        old_value: Original value
        new_value: New value

    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0.0 if new_value == 0 else float('inf')
    return ((new_value - old_value) / old_value) * 100


def get_business_metrics(
    y_true: List[int],
    y_pred: List[int],
    avg_customer_value: float = 500,
    retention_cost: float = 50,
    acquisition_cost: float = 200
) -> Dict[str, float]:
    """
    Calculate business-relevant metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        avg_customer_value: Average customer lifetime value
        retention_cost: Cost of retention campaign per customer
        acquisition_cost: Cost to acquire new customer

    Returns:
        Dictionary of business metrics
    """
    import numpy as np

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # True positives: predicted churn, actually churned
    tp = np.sum((y_pred == 1) & (y_true == 1))
    # False positives: predicted churn, didn't churn
    fp = np.sum((y_pred == 1) & (y_true == 0))
    # False negatives: predicted no churn, actually churned
    fn = np.sum((y_pred == 0) & (y_true == 1))
    # True negatives: predicted no churn, didn't churn
    tn = np.sum((y_pred == 0) & (y_true == 0))

    # Calculate costs and savings
    # Retention cost for predicted churners
    total_retention_cost = (tp + fp) * retention_cost

    # Savings from prevented churn (assuming 50% success rate)
    prevented_churn_savings = tp * avg_customer_value * 0.5

    # Cost of missed churners (need to acquire replacement)
    missed_churn_cost = fn * acquisition_cost

    # Unnecessary retention cost
    unnecessary_retention_cost = fp * retention_cost

    # Net savings
    net_savings = prevented_churn_savings - total_retention_cost - missed_churn_cost

    # ROI
    roi = safe_divide(net_savings, total_retention_cost) * 100

    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "total_retention_cost": total_retention_cost,
        "prevented_churn_savings": prevented_churn_savings,
        "missed_churn_cost": missed_churn_cost,
        "net_savings": net_savings,
        "roi_percentage": roi,
    }
