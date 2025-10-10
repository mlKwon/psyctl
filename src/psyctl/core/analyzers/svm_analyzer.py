"""SVM-based layer analysis."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import cross_val_score  # type: ignore[import-not-found]
from sklearn.svm import LinearSVC  # type: ignore[import-not-found]

from psyctl.core.analyzers.base import BaseLayerAnalyzer
from psyctl.core.logger import get_logger


class SVMAnalyzer(BaseLayerAnalyzer):
    """
    SVM-based separation analysis with cross-validation.

    Uses Linear SVM to measure how well positive and neutral activations
    can be separated. Includes cross-validation to detect overfitting and
    support vector analysis to assess decision boundary complexity.

    Enhancements over basic SVM:
    - Cross-validation accuracy to detect overfitting
    - Support vector count to assess boundary complexity
    - Robust error handling with fallback strategies
    """

    def __init__(self):
        """Initialize SVM analyzer."""
        self.logger = get_logger("svm_analyzer")

    def analyze(
        self,
        positive_activations: list[torch.Tensor],
        neutral_activations: list[torch.Tensor],
    ) -> dict[str, float]:
        """
        Analyze separation using SVM with cross-validation.

        Args:
            positive_activations: List of activation tensors
            neutral_activations: List of activation tensors

        Returns:
            Dictionary with metrics:
            - score: Primary ranking metric (CV accuracy if available, else train accuracy)
            - accuracy: SVM training accuracy
            - cv_accuracy: Cross-validated accuracy (5-fold)
            - margin: Average decision margin
            - support_vector_ratio: Ratio of support vectors to total samples
            - converged: Whether SVM converged
        """
        self.logger.debug(
            f"Analyzing {len(positive_activations)} positive and "
            f"{len(neutral_activations)} neutral activations"
        )

        # Convert to numpy arrays
        x_pos = self._stack_tensors(positive_activations)
        x_neu = self._stack_tensors(neutral_activations)

        # Combine data
        x = np.vstack([x_pos, x_neu])
        y = np.hstack([np.zeros(len(x_pos)), np.ones(len(x_neu))])

        n_samples = len(y)
        n_features = x.shape[1]

        self.logger.debug(f"Dataset: {n_samples} samples x {n_features} features")

        # Check for sufficient samples
        if n_samples < 4:
            self.logger.warning(
                f"Only {n_samples} samples available. SVM analysis may be unreliable."
            )

        metrics = {}
        converged = True

        # Train SVM
        try:
            svm = LinearSVC(C=1.0, max_iter=10000, dual="auto", random_state=42)  # type: ignore[call-arg]
            svm.fit(x, y)

            # 1. Training accuracy
            accuracy = svm.score(x, y)
            metrics["accuracy"] = float(accuracy)
            self.logger.debug(f"SVM training accuracy: {accuracy:.4f}")

            # 2. Decision margin (average distance from decision boundary)
            try:
                decision_values = np.abs(svm.decision_function(x))
                margin = decision_values.mean()
                metrics["margin"] = float(margin)
                self.logger.debug(f"Average margin: {margin:.4f}")
            except Exception as e:
                self.logger.warning(f"Margin calculation failed: {e}")
                metrics["margin"] = 0.0

            # 3. Support vector ratio (complexity indicator)
            try:
                # For LinearSVC, we approximate by counting samples near decision boundary
                # A sample is considered a support vector if within margin threshold
                decision_values = np.abs(svm.decision_function(x))
                # Support vectors are typically within 1.0 of decision boundary
                margin_threshold = 1.0
                n_support_vectors = np.sum(decision_values < margin_threshold)
                support_vector_ratio = n_support_vectors / n_samples
                metrics["support_vector_ratio"] = float(support_vector_ratio)
                self.logger.debug(
                    f"Support vectors: {n_support_vectors}/{n_samples} "
                    f"(ratio: {support_vector_ratio:.4f})"
                )
            except Exception as e:
                self.logger.warning(f"Support vector calculation failed: {e}")
                metrics["support_vector_ratio"] = 0.0

        except Exception as e:
            self.logger.error(f"SVM training failed: {e}")
            metrics["accuracy"] = 0.0
            metrics["margin"] = 0.0
            metrics["support_vector_ratio"] = 0.0
            converged = False

        # 4. Cross-validation accuracy (most reliable metric)
        try:
            if n_samples >= 10:  # Need at least 10 samples for 5-fold CV
                svm_cv = LinearSVC(C=1.0, max_iter=10000, dual="auto", random_state=42)  # type: ignore[call-arg]
                n_folds = min(5, n_samples // 2)
                cv_scores = cross_val_score(svm_cv, x, y, cv=n_folds)
                cv_accuracy = cv_scores.mean()
                cv_std = cv_scores.std()
                metrics["cv_accuracy"] = float(cv_accuracy)
                self.logger.debug(
                    f"CV accuracy: {cv_accuracy:.4f} (±{cv_std:.4f}, {n_folds}-fold)"
                )

                # Use CV accuracy as primary score (more reliable)
                metrics["score"] = float(cv_accuracy)
            else:
                # Not enough samples for CV, fall back to train accuracy
                self.logger.warning(
                    f"Only {n_samples} samples, skipping cross-validation. "
                    "Using training accuracy as score."
                )
                metrics["cv_accuracy"] = metrics.get("accuracy", 0.0)
                metrics["score"] = metrics.get("accuracy", 0.0)
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
            metrics["cv_accuracy"] = 0.0
            metrics["score"] = metrics.get("accuracy", 0.0)
            converged = False

        metrics["converged"] = converged

        self.logger.info(
            f"SVM analysis: score={metrics['score']:.4f}, "
            f"accuracy={metrics.get('accuracy', 0.0):.4f}, "
            f"cv={metrics.get('cv_accuracy', 0.0):.4f}, "
            f"margin={metrics.get('margin', 0.0):.4f}, "
            f"sv_ratio={metrics.get('support_vector_ratio', 0.0):.4f}"
        )

        return metrics
