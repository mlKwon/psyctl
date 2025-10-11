"""Base class for layer analysis methods."""

from abc import ABC, abstractmethod

import numpy as np
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
        positive_activations: list[torch.Tensor],
        neutral_activations: list[torch.Tensor],
    ) -> dict[str, float]:
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

    def _stack_tensors(self, tensor_list: list[torch.Tensor]) -> np.ndarray:
        """
        Stack list of tensors into numpy array.

        This is a utility method for converting lists of activation tensors
        into a single numpy array suitable for sklearn classifiers.

        Args:
            tensor_list: List of torch tensors with shapes [D] or [1, D]

        Returns:
            Numpy array of shape [N, D] where N = len(tensor_list)

        Raises:
            ValueError: If tensors have unexpected shapes

        Example:
            >>> tensors = [torch.randn(256), torch.randn(256), torch.randn(256)]
            >>> stacked = self._stack_tensors(tensors)
            >>> stacked.shape
            (3, 256)
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
