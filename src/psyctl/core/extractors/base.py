"""Base class for steering vector extraction methods."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer


class BaseVectorExtractor(ABC):
    """
    Base class for steering vector extraction methods.

    All extraction methods should inherit from this class and implement
    the extract() method.
    """

    @abstractmethod
    def extract(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        layers: list[str],
        dataset_path: Path,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Extract steering vectors from specified layers.

        Args:
            model: Loaded language model
            tokenizer: Model tokenizer
            layers: List of layer paths to extract from
            dataset_path: Path to dataset
            **kwargs: Method-specific parameters

        Returns:
            Dictionary mapping layer names to steering vectors
            Example: {
                "model.layers[13].mlp.down_proj": tensor(...),
                "model.layers[14].mlp.down_proj": tensor(...)
            }

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement extract() method")
