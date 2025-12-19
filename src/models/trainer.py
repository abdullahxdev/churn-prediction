"""
Model Trainer Module
====================

Handles model training with MLflow experiment tracking.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
from optuna.integration import OptunaSearchCV

from config import get_config, MODELS_DIR, MLFLOW_DIR


class ModelTrainer:
    """Train and manage machine learning models with MLflow tracking."""

    MODELS = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier,
        "lightgbm": LGBMClassifier,
        "catboost": CatBoostClassifier,
        "gradient_boosting": GradientBoostingClassifier,
    }

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize ModelTrainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config()
        self.models_config = self.config.get("models", {})
        self.mlflow_config = self.config.get("mlflow", {})
        self.tuning_config = self.config.get("tuning", {})

        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0

        # Setup MLflow
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        tracking_uri = self.mlflow_config.get("tracking_uri", "mlflow_runs")
        mlflow_path = MLFLOW_DIR / tracking_uri

        mlflow.set_tracking_uri(f"file://{mlflow_path}")
        experiment_name = self.mlflow_config.get("experiment_name", "churn_prediction")

        # Create experiment if it doesn't exist
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        logger.info(f"MLflow tracking URI: {mlflow_path}")
        logger.info(f"MLflow experiment: {experiment_name}")

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str,
        params: Optional[dict] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        log_to_mlflow: bool = True
    ) -> Any:
        """
        Train a single model.

        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Name of model to train
            params: Model parameters (overrides config)
            X_val: Validation features
            y_val: Validation labels
            log_to_mlflow: Whether to log to MLflow

        Returns:
            Trained model
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODELS.keys())}")

        # Get model class and parameters
        model_class = self.MODELS[model_name]
        model_config = self.models_config.get(model_name, {})

        if params is None:
            params = model_config.get("params", {})

        logger.info(f"Training {model_name}...")

        # Create model instance
        model = model_class(**params)

        # Train with MLflow tracking
        if log_to_mlflow:
            with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_params(params)
                mlflow.set_tag("model_type", model_name)

                # Train model
                model.fit(X_train, y_train)

                # Calculate metrics
                train_score = model.score(X_train, y_train)
                mlflow.log_metric("train_accuracy", train_score)

                if X_val is not None and y_val is not None:
                    val_score = model.score(X_val, y_val)
                    mlflow.log_metric("val_accuracy", val_score)

                # Cross-validation score
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=5, scoring="f1"
                )
                mlflow.log_metric("cv_f1_mean", cv_scores.mean())
                mlflow.log_metric("cv_f1_std", cv_scores.std())

                # Log model
                mlflow.sklearn.log_model(model, model_name)

                logger.info(f"{model_name} - Train Acc: {train_score:.4f}, CV F1: {cv_scores.mean():.4f}")
        else:
            model.fit(X_train, y_train)

        self.trained_models[model_name] = model
        return model

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train all enabled models.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Dictionary of trained models
        """
        logger.info("Training all models...")

        for model_name, model_config in self.models_config.items():
            if model_name in self.MODELS and model_config.get("enabled", True):
                try:
                    self.train_model(
                        X_train, y_train,
                        model_name,
                        X_val=X_val,
                        y_val=y_val
                    )
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")

        return self.trained_models

    def hyperparameter_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str,
        n_trials: Optional[int] = None,
        cv: Optional[int] = None
    ) -> Tuple[Any, Dict]:
        """
        Perform hyperparameter tuning using Optuna.

        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Name of model to tune
            n_trials: Number of trials
            cv: Number of CV folds

        Returns:
            Tuple of (best model, best params)
        """
        n_trials = n_trials or self.tuning_config.get("n_trials", 50)
        cv = cv or self.tuning_config.get("cv_folds", 5)
        scoring = self.tuning_config.get("scoring", "f1")

        logger.info(f"Hyperparameter tuning for {model_name} ({n_trials} trials)...")

        def objective(trial):
            if model_name == "random_forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "class_weight": "balanced",
                    "random_state": 42,
                    "n_jobs": -1,
                }
                model = RandomForestClassifier(**params)

            elif model_name == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "random_state": 42,
                    "n_jobs": -1,
                }
                model = XGBClassifier(**params)

            elif model_name == "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                    "class_weight": "balanced",
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbose": -1,
                }
                model = LGBMClassifier(**params)

            elif model_name == "catboost":
                params = {
                    "iterations": trial.suggest_int("iterations", 50, 300),
                    "depth": trial.suggest_int("depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                    "auto_class_weights": "Balanced",
                    "random_state": 42,
                    "verbose": 0,
                }
                model = CatBoostClassifier(**params)

            elif model_name == "logistic_regression":
                params = {
                    "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
                    "max_iter": 1000,
                    "class_weight": "balanced",
                    "random_state": 42,
                }
                model = LogisticRegression(**params)

            else:
                raise ValueError(f"Tuning not implemented for {model_name}")

            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv, scoring=scoring, n_jobs=-1
            )
            return cv_scores.mean()

        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Train best model
        best_params = study.best_params
        logger.info(f"Best params: {best_params}")
        logger.info(f"Best CV score: {study.best_value:.4f}")

        # Log to MLflow
        with mlflow.start_run(run_name=f"{model_name}_tuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_params(best_params)
            mlflow.log_metric("best_cv_score", study.best_value)
            mlflow.set_tag("tuning", "optuna")
            mlflow.set_tag("model_type", model_name)

            # Train final model
            best_model = self.train_model(
                X_train, y_train,
                model_name,
                params=best_params,
                log_to_mlflow=False
            )
            mlflow.sklearn.log_model(best_model, f"{model_name}_tuned")

        return best_model, best_params

    def create_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        models: Optional[List[str]] = None,
        method: str = "voting"
    ) -> Any:
        """
        Create an ensemble model.

        Args:
            X_train: Training features
            y_train: Training labels
            models: List of model names to include
            method: Ensemble method ('voting', 'stacking')

        Returns:
            Ensemble model
        """
        if models is None:
            models = ["random_forest", "xgboost", "lightgbm"]

        logger.info(f"Creating {method} ensemble with {models}...")

        estimators = []
        for name in models:
            if name not in self.trained_models:
                model_config = self.models_config.get(name, {})
                params = model_config.get("params", {})
                model = self.MODELS[name](**params)
            else:
                model = self.trained_models[name]
            estimators.append((name, model))

        if method == "voting":
            ensemble = VotingClassifier(
                estimators=estimators,
                voting="soft",
                n_jobs=-1
            )
        elif method == "stacking":
            ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=1000),
                cv=5,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

        # Train ensemble
        with mlflow.start_run(run_name=f"ensemble_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            ensemble.fit(X_train, y_train)

            cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring="f1")
            mlflow.log_metric("cv_f1_mean", cv_scores.mean())
            mlflow.log_metric("cv_f1_std", cv_scores.std())
            mlflow.set_tag("ensemble_type", method)
            mlflow.set_tag("base_models", str(models))

            mlflow.sklearn.log_model(ensemble, f"ensemble_{method}")

        self.trained_models[f"ensemble_{method}"] = ensemble
        logger.info(f"Ensemble CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return ensemble

    def save_model(
        self,
        model: Any,
        model_name: str,
        filepath: Optional[Path] = None
    ) -> Path:
        """
        Save a trained model to disk.

        Args:
            model: Model to save
            model_name: Name for the model file
            filepath: Optional custom filepath

        Returns:
            Path to saved model
        """
        if filepath is None:
            filepath = MODELS_DIR / f"{model_name}.joblib"

        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")

        return filepath

    def load_model(self, model_name: str, filepath: Optional[Path] = None) -> Any:
        """
        Load a model from disk.

        Args:
            model_name: Name of the model
            filepath: Optional custom filepath

        Returns:
            Loaded model
        """
        if filepath is None:
            filepath = MODELS_DIR / f"{model_name}.joblib"

        if not filepath.exists():
            raise FileNotFoundError(f"Model not found: {filepath}")

        model = joblib.load(filepath)
        self.trained_models[model_name] = model
        logger.info(f"Model loaded from {filepath}")

        return model

    def get_best_model(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = "f1"
    ) -> Tuple[str, Any, float]:
        """
        Get the best performing model.

        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: Evaluation metric

        Returns:
            Tuple of (model name, model, score)
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        )

        metrics = {
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
            "roc_auc": roc_auc_score,
        }

        if metric not in metrics:
            raise ValueError(f"Unknown metric: {metric}")

        metric_func = metrics[metric]
        best_score = 0
        best_name = None
        best_model = None

        for name, model in self.trained_models.items():
            try:
                if metric == "roc_auc":
                    y_prob = model.predict_proba(X_val)[:, 1]
                    score = metric_func(y_val, y_prob)
                else:
                    y_pred = model.predict(X_val)
                    score = metric_func(y_val, y_pred)

                if score > best_score:
                    best_score = score
                    best_name = name
                    best_model = model

            except Exception as e:
                logger.warning(f"Error evaluating {name}: {e}")

        self.best_model = best_model
        self.best_model_name = best_name
        self.best_score = best_score

        logger.info(f"Best model: {best_name} with {metric}={best_score:.4f}")
        return best_name, best_model, best_score

    def get_all_trained_models(self) -> Dict[str, Any]:
        """Get all trained models."""
        return self.trained_models
