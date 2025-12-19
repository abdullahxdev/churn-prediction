"""
Model Evaluator Module
======================

Comprehensive model evaluation and comparison.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    log_loss,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

from config import get_config, FIGURES_DIR


class ModelEvaluator:
    """Evaluate and compare machine learning models."""

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize ModelEvaluator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config()
        self.eval_config = self.config.get("evaluation", {})
        self.threshold = self.eval_config.get("threshold", 0.5)
        self.evaluation_results = {}

    def evaluate_model(
        self,
        model: Any,
        X: np.ndarray,
        y_true: np.ndarray,
        model_name: str = "model",
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single model comprehensively.

        Args:
            model: Trained model
            X: Features
            y_true: True labels
            model_name: Name of model
            threshold: Classification threshold

        Returns:
            Dictionary of metrics
        """
        threshold = threshold or self.threshold

        # Get predictions
        y_pred = model.predict(X)

        # Get probability predictions if available
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
            y_pred_threshold = (y_prob >= threshold).astype(int)
        else:
            y_prob = None
            y_pred_threshold = y_pred

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred_threshold),
            "precision": precision_score(y_true, y_pred_threshold, zero_division=0),
            "recall": recall_score(y_true, y_pred_threshold, zero_division=0),
            "f1": f1_score(y_true, y_pred_threshold, zero_division=0),
        }

        if y_prob is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            metrics["avg_precision"] = average_precision_score(y_true, y_prob)
            metrics["log_loss"] = log_loss(y_true, y_prob)
            metrics["brier_score"] = brier_score_loss(y_true, y_prob)

        # Store results
        self.evaluation_results[model_name] = {
            "metrics": metrics,
            "y_pred": y_pred_threshold,
            "y_prob": y_prob,
            "y_true": y_true,
        }

        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, ROC-AUC: {metrics.get('roc_auc', 'N/A')}")

        return metrics

    def evaluate_all_models(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y_true: np.ndarray
    ) -> pd.DataFrame:
        """
        Evaluate multiple models and create comparison.

        Args:
            models: Dictionary of trained models
            X: Features
            y_true: True labels

        Returns:
            DataFrame with model comparison
        """
        results = []

        for name, model in models.items():
            metrics = self.evaluate_model(model, X, y_true, name)
            metrics["model"] = name
            results.append(metrics)

        df = pd.DataFrame(results)
        df = df.set_index("model")

        # Sort by F1 score
        df = df.sort_values("f1", ascending=False)

        return df

    def get_classification_report(
        self,
        model: Any,
        X: np.ndarray,
        y_true: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> str:
        """
        Get detailed classification report.

        Args:
            model: Trained model
            X: Features
            y_true: True labels
            target_names: Names for classes

        Returns:
            Classification report string
        """
        y_pred = model.predict(X)
        target_names = target_names or ["No Churn", "Churn"]

        report = classification_report(y_true, y_pred, target_names=target_names)
        return report

    def get_confusion_matrix(
        self,
        model: Any,
        X: np.ndarray,
        y_true: np.ndarray,
        normalize: Optional[str] = None
    ) -> np.ndarray:
        """
        Get confusion matrix.

        Args:
            model: Trained model
            X: Features
            y_true: True labels
            normalize: Normalization mode ('true', 'pred', 'all', None)

        Returns:
            Confusion matrix
        """
        y_pred = model.predict(X)
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        return cm

    def find_optimal_threshold(
        self,
        model: Any,
        X: np.ndarray,
        y_true: np.ndarray,
        metric: str = "f1"
    ) -> Tuple[float, float]:
        """
        Find optimal classification threshold.

        Args:
            model: Trained model
            X: Features
            y_true: True labels
            metric: Metric to optimize

        Returns:
            Tuple of (optimal threshold, best score)
        """
        if not hasattr(model, "predict_proba"):
            logger.warning("Model doesn't support probability predictions")
            return 0.5, 0.0

        y_prob = model.predict_proba(X)[:, 1]
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_score = 0

        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)

            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "precision":
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = thresh

        logger.info(f"Optimal threshold: {best_threshold:.2f} with {metric}={best_score:.4f}")
        return best_threshold, best_score

    def plot_confusion_matrix(
        self,
        model: Any,
        X: np.ndarray,
        y_true: np.ndarray,
        model_name: str = "Model",
        save: bool = True,
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Plot confusion matrix heatmap.

        Args:
            model: Trained model
            X: Features
            y_true: True labels
            model_name: Name for title
            save: Whether to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        cm = self.get_confusion_matrix(model, X, y_true)
        cm_norm = self.get_confusion_matrix(model, X, y_true, normalize="true")

        fig, axes = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]))

        # Raw counts
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            ax=axes[0]
        )
        axes[0].set_title(f"{model_name} - Confusion Matrix (Counts)")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")

        # Normalized
        sns.heatmap(
            cm_norm, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            ax=axes[1]
        )
        axes[1].set_title(f"{model_name} - Confusion Matrix (Normalized)")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")

        plt.tight_layout()

        if save:
            filepath = FIGURES_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved confusion matrix plot to {filepath}")

        return fig

    def plot_roc_curves(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y_true: np.ndarray,
        save: bool = True,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot ROC curves for multiple models.

        Args:
            models: Dictionary of trained models
            X: Features
            y_true: True labels
            save: Whether to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)[:, 1]
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc = roc_auc_score(y_true, y_prob)
                ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.500)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves Comparison")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = FIGURES_DIR / "roc_curves_comparison.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved ROC curves plot to {filepath}")

        return fig

    def plot_precision_recall_curves(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y_true: np.ndarray,
        save: bool = True,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot Precision-Recall curves for multiple models.

        Args:
            models: Dictionary of trained models
            X: Features
            y_true: True labels
            save: Whether to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)[:, 1]
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                ap = average_precision_score(y_true, y_prob)
                ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")

        # Baseline (proportion of positive class)
        baseline = y_true.mean()
        ax.axhline(y=baseline, color="k", linestyle="--", label=f"Baseline (AP={baseline:.3f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves Comparison")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = FIGURES_DIR / "pr_curves_comparison.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved PR curves plot to {filepath}")

        return fig

    def plot_calibration_curves(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10,
        save: bool = True,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot calibration curves for multiple models.

        Args:
            models: Dictionary of trained models
            X: Features
            y_true: True labels
            n_bins: Number of bins
            save: Whether to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)[:, 1]
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true, y_prob, n_bins=n_bins
                )
                brier = brier_score_loss(y_true, y_prob)
                ax.plot(
                    mean_predicted_value, fraction_of_positives,
                    marker="o", label=f"{name} (Brier={brier:.3f})"
                )

        ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curves Comparison")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = FIGURES_DIR / "calibration_curves.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved calibration curves plot to {filepath}")

        return fig

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        save: bool = True,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot model comparison bar chart.

        Args:
            comparison_df: DataFrame with model metrics
            save: Whether to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Select metrics to plot
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        available_metrics = [m for m in metrics if m in comparison_df.columns]

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(comparison_df))
        width = 0.15
        multiplier = 0

        for metric in available_metrics:
            offset = width * multiplier
            rects = ax.bar(x + offset, comparison_df[metric], width, label=metric.upper())
            multiplier += 1

        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks(x + width * (len(available_metrics) - 1) / 2)
        ax.set_xticklabels(comparison_df.index, rotation=45, ha="right")
        ax.legend(loc="upper right")
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save:
            filepath = FIGURES_DIR / "model_comparison.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved model comparison plot to {filepath}")

        return fig

    def get_evaluation_summary(self) -> Dict:
        """
        Get summary of all evaluations.

        Returns:
            Dictionary with evaluation summary
        """
        summary = {}
        for model_name, results in self.evaluation_results.items():
            summary[model_name] = results["metrics"]
        return summary
