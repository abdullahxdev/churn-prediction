"""
Model Explainability Module
===========================

SHAP and LIME-based model interpretability.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from loguru import logger

from config import get_config, FIGURES_DIR


class ModelExplainer:
    """Explain model predictions using SHAP and LIME."""

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize ModelExplainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config()
        self.shap_explainer = None
        self.lime_explainer = None
        self.shap_values = None
        self.feature_names = []

    def setup_shap_explainer(
        self,
        model: Any,
        X_background: np.ndarray,
        model_type: str = "tree"
    ) -> shap.Explainer:
        """
        Setup SHAP explainer for the model.

        Args:
            model: Trained model
            X_background: Background data for SHAP
            model_type: Type of model ('tree', 'linear', 'kernel')

        Returns:
            SHAP Explainer object
        """
        logger.info(f"Setting up SHAP explainer ({model_type})...")

        if model_type == "tree":
            self.shap_explainer = shap.TreeExplainer(model)
        elif model_type == "linear":
            self.shap_explainer = shap.LinearExplainer(model, X_background)
        elif model_type == "kernel":
            # Subsample for kernel explainer (slow)
            if len(X_background) > 100:
                indices = np.random.choice(len(X_background), 100, replace=False)
                X_background = X_background[indices]
            self.shap_explainer = shap.KernelExplainer(model.predict_proba, X_background)
        else:
            # Auto-detect
            try:
                self.shap_explainer = shap.TreeExplainer(model)
            except Exception:
                self.shap_explainer = shap.Explainer(model, X_background)

        return self.shap_explainer

    def setup_lime_explainer(
        self,
        X_train: np.ndarray,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None
    ) -> LimeTabularExplainer:
        """
        Setup LIME explainer.

        Args:
            X_train: Training data
            feature_names: List of feature names
            class_names: Names for classes
            categorical_features: Indices of categorical features

        Returns:
            LIME TabularExplainer
        """
        logger.info("Setting up LIME explainer...")

        self.feature_names = feature_names
        class_names = class_names or ["No Churn", "Churn"]

        self.lime_explainer = LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=class_names,
            categorical_features=categorical_features,
            mode="classification"
        )

        return self.lime_explainer

    def calculate_shap_values(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Calculate SHAP values for given data.

        Args:
            X: Data to explain
            feature_names: List of feature names

        Returns:
            SHAP values array
        """
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not setup. Call setup_shap_explainer first.")

        logger.info(f"Calculating SHAP values for {len(X)} samples...")

        self.shap_values = self.shap_explainer.shap_values(X)

        if feature_names:
            self.feature_names = feature_names

        # Handle multi-class output
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Class 1 (Churn)

        return self.shap_values

    def get_feature_importance_shap(
        self,
        X: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Get feature importance based on SHAP values.

        Args:
            X: Data to calculate SHAP values (if not already calculated)

        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            if X is None:
                raise ValueError("SHAP values not calculated. Provide X or call calculate_shap_values first.")
            self.calculate_shap_values(X)

        # Calculate mean absolute SHAP value for each feature
        importance = np.abs(self.shap_values).mean(axis=0)

        df = pd.DataFrame({
            "feature": self.feature_names[:len(importance)] if self.feature_names else [f"f{i}" for i in range(len(importance))],
            "importance": importance
        })
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)

        return df

    def plot_shap_summary(
        self,
        X: np.ndarray,
        max_display: int = 20,
        save: bool = True
    ) -> plt.Figure:
        """
        Plot SHAP summary plot.

        Args:
            X: Data used for SHAP values
            max_display: Maximum features to display
            save: Whether to save figure

        Returns:
            Matplotlib figure
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)

        fig = plt.figure(figsize=(12, 8))

        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )

        plt.title("SHAP Feature Importance Summary")
        plt.tight_layout()

        if save:
            filepath = FIGURES_DIR / "shap_summary.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved SHAP summary plot to {filepath}")

        return fig

    def plot_shap_bar(
        self,
        X: np.ndarray,
        max_display: int = 20,
        save: bool = True
    ) -> plt.Figure:
        """
        Plot SHAP bar plot (feature importance).

        Args:
            X: Data used for SHAP values
            max_display: Maximum features to display
            save: Whether to save figure

        Returns:
            Matplotlib figure
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)

        fig = plt.figure(figsize=(10, 8))

        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False
        )

        plt.title("SHAP Feature Importance (Bar)")
        plt.tight_layout()

        if save:
            filepath = FIGURES_DIR / "shap_bar.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved SHAP bar plot to {filepath}")

        return fig

    def explain_single_prediction_shap(
        self,
        model: Any,
        X_single: np.ndarray,
        save: bool = True,
        plot_type: str = "force"
    ) -> plt.Figure:
        """
        Explain a single prediction using SHAP.

        Args:
            model: Trained model
            X_single: Single sample to explain
            save: Whether to save figure
            plot_type: Type of plot ('force', 'waterfall')

        Returns:
            Matplotlib figure or SHAP plot
        """
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not setup.")

        shap_values_single = self.shap_explainer.shap_values(X_single)

        if isinstance(shap_values_single, list):
            shap_values_single = shap_values_single[1]

        if plot_type == "force":
            # Force plot
            shap.initjs()
            expected_value = self.shap_explainer.expected_value
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value[1]

            fig = shap.force_plot(
                expected_value,
                shap_values_single[0],
                X_single[0],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )

        elif plot_type == "waterfall":
            fig = plt.figure(figsize=(12, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values_single[0],
                    base_values=self.shap_explainer.expected_value[1] if isinstance(self.shap_explainer.expected_value, np.ndarray) else self.shap_explainer.expected_value,
                    data=X_single[0],
                    feature_names=self.feature_names
                ),
                show=False
            )

        if save:
            filepath = FIGURES_DIR / f"shap_{plot_type}_single.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved SHAP {plot_type} plot to {filepath}")

        return fig

    def explain_single_prediction_lime(
        self,
        model: Any,
        X_single: np.ndarray,
        num_features: int = 10,
        save: bool = True
    ) -> Tuple[Any, plt.Figure]:
        """
        Explain a single prediction using LIME.

        Args:
            model: Trained model
            X_single: Single sample to explain
            num_features: Number of features to show
            save: Whether to save figure

        Returns:
            Tuple of (LIME explanation, matplotlib figure)
        """
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not setup.")

        explanation = self.lime_explainer.explain_instance(
            X_single.flatten(),
            model.predict_proba,
            num_features=num_features
        )

        fig = explanation.as_pyplot_figure()
        plt.title("LIME Explanation")
        plt.tight_layout()

        if save:
            filepath = FIGURES_DIR / "lime_explanation.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved LIME explanation to {filepath}")

        return explanation, fig

    def plot_shap_dependence(
        self,
        X: np.ndarray,
        feature: Union[str, int],
        interaction_feature: Optional[Union[str, int]] = "auto",
        save: bool = True
    ) -> plt.Figure:
        """
        Plot SHAP dependence plot for a feature.

        Args:
            X: Data used for SHAP values
            feature: Feature to plot
            interaction_feature: Feature for interaction coloring
            save: Whether to save figure

        Returns:
            Matplotlib figure
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)

        fig = plt.figure(figsize=(10, 6))

        shap.dependence_plot(
            feature,
            self.shap_values,
            X,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )

        plt.title(f"SHAP Dependence Plot - {feature}")
        plt.tight_layout()

        if save:
            feature_name = feature if isinstance(feature, str) else f"feature_{feature}"
            filepath = FIGURES_DIR / f"shap_dependence_{feature_name}.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved SHAP dependence plot to {filepath}")

        return fig

    def get_top_risk_factors(
        self,
        X_single: np.ndarray,
        top_n: int = 5
    ) -> pd.DataFrame:
        """
        Get top risk factors for a single prediction.

        Args:
            X_single: Single sample
            top_n: Number of top factors to return

        Returns:
            DataFrame with top risk factors
        """
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not setup.")

        shap_values_single = self.shap_explainer.shap_values(X_single)

        if isinstance(shap_values_single, list):
            shap_values_single = shap_values_single[1]

        # Create DataFrame of SHAP values
        df = pd.DataFrame({
            "feature": self.feature_names[:len(shap_values_single[0])] if self.feature_names else [f"f{i}" for i in range(len(shap_values_single[0]))],
            "shap_value": shap_values_single[0],
            "feature_value": X_single.flatten()[:len(shap_values_single[0])]
        })

        # Sort by absolute SHAP value
        df["abs_shap"] = np.abs(df["shap_value"])
        df = df.sort_values("abs_shap", ascending=False).head(top_n)
        df = df.drop("abs_shap", axis=1)

        # Add direction
        df["impact"] = df["shap_value"].apply(lambda x: "Increases Churn Risk" if x > 0 else "Decreases Churn Risk")

        return df

    def get_explainability_summary(self) -> Dict:
        """
        Get summary of explainability analysis.

        Returns:
            Dictionary with summary
        """
        summary = {
            "shap_explainer_type": type(self.shap_explainer).__name__ if self.shap_explainer else None,
            "lime_explainer_setup": self.lime_explainer is not None,
            "shap_values_calculated": self.shap_values is not None,
            "num_features": len(self.feature_names) if self.feature_names else 0,
        }

        if self.shap_values is not None:
            summary["shap_values_shape"] = self.shap_values.shape

        return summary
