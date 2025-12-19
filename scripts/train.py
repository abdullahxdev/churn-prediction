"""
Training Script
===============

Command-line script to train churn prediction models.

Usage:
    python scripts/train.py --model xgboost --tune
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import joblib
import numpy as np
from loguru import logger

from config import get_config, MODELS_DIR, RAW_DATA_DIR
from src.data import DataLoader, DataPreprocessor
from src.features import FeatureEngineer
from src.models import ModelTrainer, ModelEvaluator, ModelExplainer
from src.utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train churn prediction models")

    parser.add_argument(
        "--data",
        type=str,
        default="ecommerce_customer_churn.csv",
        help="Name of data file in data/raw/"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"],
        help="Model to train"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform hyperparameter tuning"
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=50,
        help="Number of tuning trials"
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Create ensemble model"
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Generate SHAP explanations"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    setup_logging(level=args.log_level, log_file="training.log")
    logger.info("Starting model training...")

    # Load config
    config = get_config()
    logger.info(f"Loaded configuration")

    # Load data
    loader = DataLoader(config)
    try:
        df = loader.load_raw_data(args.data)
    except FileNotFoundError:
        logger.error(f"Data file not found: {args.data}")
        logger.info("Creating sample data for demonstration...")
        # Create sample data
        np.random.seed(42)
        n_samples = 5000
        import pandas as pd
        df = pd.DataFrame({
            'CustomerID': range(1, n_samples + 1),
            'Churn': np.random.choice([0, 1], n_samples, p=[0.83, 0.17]),
            'Tenure': np.random.randint(0, 61, n_samples),
            'PreferredLoginDevice': np.random.choice(['Mobile Phone', 'Computer', 'Phone'], n_samples),
            'CityTier': np.random.choice([1, 2, 3], n_samples),
            'WarehouseToHome': np.random.uniform(5, 35, n_samples),
            'PreferredPaymentMode': np.random.choice(['Debit Card', 'Credit Card', 'E wallet', 'COD', 'UPI'], n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'HourSpendOnApp': np.random.uniform(0, 5, n_samples),
            'NumberOfDeviceRegistered': np.random.randint(1, 7, n_samples),
            'PreferedOrderCat': np.random.choice(['Laptop & Accessory', 'Mobile', 'Fashion', 'Grocery', 'Others'], n_samples),
            'SatisfactionScore': np.random.randint(1, 6, n_samples),
            'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
            'NumberOfAddress': np.random.randint(1, 11, n_samples),
            'Complain': np.random.choice([0, 1], n_samples, p=[0.72, 0.28]),
            'OrderAmountHikeFromlastYear': np.random.uniform(11, 26, n_samples),
            'CouponUsed': np.random.randint(0, 16, n_samples),
            'OrderCount': np.random.randint(1, 16, n_samples),
            'DaySinceLastOrder': np.random.randint(0, 46, n_samples),
            'CashbackAmount': np.random.uniform(0, 325, n_samples)
        })

    # Feature engineering
    fe = FeatureEngineer(config)
    df = fe.create_all_features(df)
    logger.info(f"Created {len(fe.get_created_features())} new features")

    # Preprocessing
    preprocessor = DataPreprocessor(config)
    df = preprocessor.clean_data(df)
    df = preprocessor.handle_missing_values(df)

    # Split data
    target_col = config["data"]["target_column"]
    X_train, X_val, X_test, y_train, y_val, y_test = loader.get_train_test_split(df, target_col=target_col)

    # Transform features
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names()

    # Save preprocessor
    joblib.dump(preprocessor.preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.joblib")
    logger.info("Saved preprocessor and feature names")

    # Train models
    trainer = ModelTrainer(config)

    if args.model == "all":
        models = trainer.train_all_models(X_train_t, y_train, X_val=X_val_t, y_val=y_val)
    else:
        trainer.train_model(X_train_t, y_train, args.model, X_val=X_val_t, y_val=y_val)

    # Hyperparameter tuning
    if args.tune:
        model_to_tune = args.model if args.model != "all" else "xgboost"
        logger.info(f"Tuning {model_to_tune}...")
        trainer.hyperparameter_tuning(X_train_t, y_train, model_to_tune, n_trials=args.tune_trials)

    # Ensemble
    if args.ensemble:
        trainer.create_ensemble(X_train_t, y_train, method="voting")

    # Evaluate
    evaluator = ModelEvaluator(config)
    comparison = evaluator.evaluate_all_models(trainer.get_all_trained_models(), X_test_t, y_test)
    logger.info(f"\nModel Comparison:\n{comparison}")

    # Get best model
    best_name, best_model, best_score = trainer.get_best_model(X_val_t, y_val, metric="f1")
    logger.info(f"Best model: {best_name} (F1={best_score:.4f})")

    # Plot results
    evaluator.plot_confusion_matrix(best_model, X_test_t, y_test, best_name)
    evaluator.plot_roc_curves(trainer.get_all_trained_models(), X_test_t, y_test)
    evaluator.plot_model_comparison(comparison)

    # SHAP analysis
    if args.shap:
        logger.info("Generating SHAP explanations...")
        explainer = ModelExplainer(config)
        explainer.setup_shap_explainer(best_model, X_train_t[:100], model_type="tree")
        explainer.calculate_shap_values(X_test_t[:500], feature_names=feature_names)
        explainer.plot_shap_summary(X_test_t[:500])
        explainer.plot_shap_bar(X_test_t[:500])

    # Save best model
    trainer.save_model(best_model, "best_model")
    for name, model in trainer.get_all_trained_models().items():
        trainer.save_model(model, name)

    # Save metadata
    metadata = {
        "best_model": best_name,
        "metrics": comparison.loc[best_name].to_dict(),
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "training_samples": len(X_train),
    }
    joblib.dump(metadata, MODELS_DIR / "model_metadata.joblib")

    logger.info("Training complete!")
    logger.info(f"Best model saved to: {MODELS_DIR / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
