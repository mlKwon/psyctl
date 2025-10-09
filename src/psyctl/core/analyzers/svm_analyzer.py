"""SVM-based layer analysis."""

from typing import Dict, List

import numpy as np
import torch
from sklearn.svm import LinearSVC

from psyctl.core.analyzers.base import BaseLayerAnalyzer
from psyctl.core.logger import get_logger


class SVMAnalyzer(BaseLayerAnalyzer):
    """
    SVM-based separation analysis.

    Uses Linear SVM training accuracy as a measure of how well
    positive and neutral activations can be separated.
    """

    def __init__(self):
        """Initialize SVM analyzer."""
        self.logger = get_logger("svm_analyzer")

    def analyze(
        self,
        positive_activations: List[torch.Tensor],
        neutral_activations: List[torch.Tensor],
    ) -> Dict[str, float]:
        """
        Analyze separation using SVM training accuracy.

        Args:
            positive_activations: List of activation tensors
            neutral_activations: List of activation tensors

        Returns:
            Dictionary with metrics:
            - score: Primary ranking metric (accuracy)
            - accuracy: SVM training accuracy
            - margin: Average decision margin
            - converged: Whether SVM converged
        """
        self.logger.debug(
            f"Analyzing {len(positive_activations)} positive and "
            f"{len(neutral_activations)} neutral activations"
        )

        # Convert to numpy arrays
        X_pos = self._stack_tensors(positive_activations)
        X_neu = self._stack_tensors(neutral_activations)

        # Combine data
        X = np.vstack([X_pos, X_neu])
        y = np.hstack([np.zeros(len(X_pos)), np.ones(len(X_neu))])

        # Train SVM
        try:
            svm = LinearSVC(C=1.0, max_iter=10000, dual="auto", random_state=42)
            svm.fit(X, y)

            # Calculate accuracy
            accuracy = svm.score(X, y)

            # Calculate margin (average decision function value)
            decision_values = np.abs(svm.decision_function(X))
            margin = decision_values.mean()

            self.logger.debug(f"SVM accuracy: {accuracy:.4f}, margin: {margin:.4f}")

            return {
                "score": float(accuracy),
                "accuracy": float(accuracy),
                "margin": float(margin),
                "converged": True,
            }

        except Exception as e:
            self.logger.error(f"SVM training failed: {e}")
            return {
                "score": 0.0,
                "accuracy": 0.0,
                "margin": 0.0,
                "converged": False,
            }

    def _stack_tensors(self, tensor_list: List[torch.Tensor]) -> np.ndarray:
        """
        Stack list of tensors into numpy array.

        Args:
            tensor_list: List of torch tensors

        Returns:
            Numpy array of shape [N, D]
        """
        stacked = []
        for t in tensor_list:
            t = torch.as_tensor(t)
            if t.ndim == 1:
                stacked.append(t.unsqueeze(0))
            elif t.ndim == 2 and t.shape[0] == 1:
                stacked.append(t)
            else:
                raise ValueError(
                    f"Expected tensor of shape [D] or [1, D], got {t.shape}"
                )

        return torch.vstack(stacked).float().cpu().numpy()
