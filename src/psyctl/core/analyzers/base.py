"""Base class for layer analysis methods."""

from abc import ABC, abstractmethod
from typing import Dict, List

import torch


class BaseLayerAnalyzer(ABC):
    """
    Base class for layer analysis methods.

    All analysis methods should inherit from this class and implement
    the analyze() method.
    """

    @abstractmethod
    def analyze(
        self,
        positive_activations: List[torch.Tensor],
        neutral_activations: List[torch.Tensor],
    ) -> Dict[str, float]:
        """
        Analyze separation quality between positive and neutral activations.

        Args:
            positive_activations: List of activation tensors (shape: [hidden_dim])
            neutral_activations: List of activation tensors (shape: [hidden_dim])

        Returns:
            Dictionary with analysis metrics, must include "score" key
            Example: {"score": 0.95, "accuracy": 0.95, "margin": 1.23}

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement analyze() method")
