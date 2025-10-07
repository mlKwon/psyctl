"""Mean Contrastive Activation Vector extractor."""

from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer

from psyctl.config import INFERENCE_BATCH_SIZE
from psyctl.core.caa_dataset_loader import CAADatasetLoader
from psyctl.core.extractors.base import BaseVectorExtractor
from psyctl.core.hook_manager import ActivationHookManager
from psyctl.core.layer_accessor import LayerAccessor
from psyctl.core.logger import get_logger


class MeanContrastiveActivationVectorExtractor(BaseVectorExtractor):
    """
    Extract steering vectors using mean difference of contrastive activations.

    This extractor implements the CAA (Contrastive Activation Addition) method
    by computing the difference between mean activations from positive and
    neutral personality prompts.

    Algorithm:
    1. Collect activations from positive prompts → compute mean
    2. Collect activations from neutral prompts → compute mean
    3. Steering vector = mean(positive) - mean(neutral)

    Attributes:
        hook_manager: Manager for forward hooks
        dataset_loader: Loader for CAA dataset
        layer_accessor: Accessor for dynamic layer retrieval
        logger: Logger instance
    """

    def __init__(self):
        """Initialize MeanContrastiveActivationVectorExtractor."""
        self.hook_manager = ActivationHookManager()
        self.dataset_loader = CAADatasetLoader()
        self.layer_accessor = LayerAccessor()
        self.logger = get_logger("mcav_extractor")

    def extract(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        layers: List[str],
        dataset_path: Path,
        batch_size: int = None,
        normalize: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract steering vectors from multiple layers simultaneously.

        Args:
            model: Loaded language model
            tokenizer: Model tokenizer
            layers: List of layer paths (e.g., ["model.layers[13].mlp.down_proj"])
            dataset_path: Path to CAA dataset
            batch_size: Batch size for inference (default: from config)
            normalize: Whether to normalize vectors to unit length
            **kwargs: Additional parameters (unused)

        Returns:
            Dictionary mapping layer names to steering vectors
            {
                "model.layers[13].mlp.down_proj": tensor(...),
                "model.layers[14].mlp.down_proj": tensor(...)
            }

        Example:
            >>> extractor = MeanContrastiveActivationVectorExtractor()
            >>> vectors = extractor.extract(
            ...     model=model,
            ...     tokenizer=tokenizer,
            ...     layers=["model.layers[13].mlp.down_proj", "model.layers[14].mlp.down_proj"],
            ...     dataset_path=Path("./dataset/caa"),
            ...     batch_size=16
            ... )
        """
        if batch_size is None:
            batch_size = INFERENCE_BATCH_SIZE

        self.logger.info(f"Extracting steering vectors from {len(layers)} layers")
        self.logger.info(f"Dataset: {dataset_path}")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Normalize: {normalize}")

        # 1. Validate layers
        self.logger.info("Validating layer paths...")
        if not self.layer_accessor.validate_layers(model, layers):
            raise ValueError("Some layer paths are invalid")

        # 2. Load dataset
        self.logger.info("Loading dataset...")
        dataset = self.dataset_loader.load(dataset_path)
        positive_prompts, neutral_prompts = self.dataset_loader.create_prompts(
            dataset, tokenizer
        )

        self.logger.info(
            f"Loaded {len(positive_prompts)} positive and {len(neutral_prompts)} neutral prompts"
        )

        # 3. Get layer modules
        layer_modules = {}
        for layer_str in layers:
            layer_module = self.layer_accessor.get_layer(model, layer_str)
            layer_modules[layer_str] = layer_module

        # 4. Collect positive activations
        self.logger.info("Collecting positive activations...")
        self._collect_activations(
            model, tokenizer, layer_modules, positive_prompts, batch_size, "positive"
        )
        positive_means = self.hook_manager.get_mean_activations()

        # 5. Collect neutral activations
        self.logger.info("Collecting neutral activations...")
        self.hook_manager.reset()
        self._collect_activations(
            model, tokenizer, layer_modules, neutral_prompts, batch_size, "neutral"
        )
        neutral_means = self.hook_manager.get_mean_activations()

        # 6. Compute steering vectors
        self.logger.info("Computing steering vectors...")
        steering_vectors = {}

        for layer_name in layers:
            positive_key = f"{layer_name}_positive"
            neutral_key = f"{layer_name}_neutral"

            if positive_key not in positive_means or neutral_key not in neutral_means:
                self.logger.warning(
                    f"Missing activations for layer '{layer_name}', skipping"
                )
                continue

            steering_vec = positive_means[positive_key] - neutral_means[neutral_key]

            if normalize:
                norm = steering_vec.norm()
                if norm > 1e-8:
                    steering_vec = steering_vec / norm
                    self.logger.debug(f"Normalized vector for '{layer_name}'")
                else:
                    self.logger.warning(
                        f"Vector for '{layer_name}' has near-zero norm, skipping normalization"
                    )

            steering_vectors[layer_name] = steering_vec

            self.logger.info(
                f"Extracted steering vector for '{layer_name}': "
                f"shape={steering_vec.shape}, norm={steering_vec.norm():.4f}"
            )

        self.logger.info(
            f"Successfully extracted {len(steering_vectors)} steering vectors"
        )
        return steering_vectors

    def _collect_activations(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        layer_modules: Dict[str, nn.Module],
        prompts: List[str],
        batch_size: int,
        suffix: str,
    ) -> None:
        """
        Collect activations from multiple layers in one forward pass.

        Args:
            model: Language model
            tokenizer: Tokenizer
            layer_modules: Dictionary of layer name → module
            prompts: List of prompts to process
            batch_size: Batch size for inference
            suffix: Suffix for layer names (e.g., "positive" or "neutral")
        """
        # Register hooks with suffix
        suffixed_layers = {
            f"{name}_{suffix}": module for name, module in layer_modules.items()
        }
        self.hook_manager.register_hooks(suffixed_layers)

        try:
            num_batches = (len(prompts) + batch_size - 1) // batch_size

            with torch.inference_mode():
                for batch_prompts in tqdm(
                    self.dataset_loader.get_batch_iterator(prompts, batch_size),
                    desc=f"Collecting {suffix} activations",
                    total=num_batches,
                ):
                    # Tokenize batch
                    inputs = tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        add_special_tokens=False,
                    )

                    # Move to model device
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    # Forward pass (hooks will collect activations)
                    _ = model(**inputs)

        finally:
            self.hook_manager.remove_all_hooks()

        # Log statistics
        stats = self.hook_manager.get_activation_stats()
        for layer_name, layer_stats in stats.items():
            self.logger.debug(
                f"Collected {layer_stats['count']} activations from '{layer_name}'"
            )
