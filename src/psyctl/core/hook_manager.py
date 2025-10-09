"""Activation hook manager for collecting model activations."""

from typing import Callable, Dict, Optional

import torch
from torch import nn

from psyctl.core.logger import get_logger


class ActivationHookManager:
    """
    Manage forward hooks for multi-layer activation collection.

    This class handles registration of forward hooks on model layers to collect
    activations during inference. It supports incremental mean calculation for
    memory-efficient processing of large datasets.

    Correctly handles both left-padded and right-padded batches by using
    attention masks to identify the last real (non-padding) token in each sequence.

    Attributes:
        hooks: Dictionary mapping layer names to RemovableHandle objects
        activations: Dictionary storing activation statistics per layer
        logger: Logger instance for debugging
    """

    def __init__(self):
        """Initialize ActivationHookManager with empty state."""
        self.hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self.activations: Dict[str, Dict] = {}
        self.logger = get_logger("hook_manager")
        self._attention_mask: Optional[torch.Tensor] = None

    def set_attention_mask(self, attention_mask: torch.Tensor) -> None:
        """
        Set attention mask for the current batch.

        The attention mask is used to find the last real (non-padding) token
        in each sequence. This ensures correct activation extraction regardless
        of padding direction (left or right).

        Args:
            attention_mask: [batch_size, seq_len] tensor where 1=real, 0=padding

        Example:
            >>> # Before model forward pass
            >>> inputs = tokenizer(prompts, padding=True, return_tensors="pt")
            >>> hook_manager.set_attention_mask(inputs['attention_mask'])
            >>> _ = model(**inputs)  # Hooks will use the mask
        """
        self._attention_mask = attention_mask
        self.logger.debug(f"Set attention mask with shape {attention_mask.shape}")

    def _get_last_real_token_position(
        self, attention_mask: torch.Tensor, sample_idx: int
    ) -> int:
        """
        Find the ABSOLUTE position of the last real (non-padding) token.

        Args:
            attention_mask: [seq_len] tensor for one sample (1=real, 0=padding)
            sample_idx: Index of sample in batch (for logging)

        Returns:
            Absolute 0-indexed position in the sequence

        Raises:
            ValueError: If sample is entirely padding

        Example:
            >>> # Right padding: [Token1, Token2, Token3, PAD, PAD]
            >>> mask = torch.tensor([1, 1, 1, 0, 0])
            >>> pos = manager._get_last_real_token_position(mask, 0)
            >>> assert pos == 2  # Absolute position 2

            >>> # Left padding: [PAD, PAD, Token1, Token2, Token3]
            >>> mask = torch.tensor([0, 0, 1, 1, 1])
            >>> pos = manager._get_last_real_token_position(mask, 0)
            >>> assert pos == 4  # Absolute position 4
        """
        # Find all positions where mask is 1
        real_positions = torch.where(attention_mask == 1)[0]

        if len(real_positions) == 0:
            raise ValueError(f"Sample {sample_idx} is entirely padding")

        # Get the last real position (highest index)
        last_position = real_positions[-1].item()
        return last_position

    def collect_activation(self, layer_name: str) -> Callable:
        """
        Create hook callback for collecting activations with incremental mean.

        This method returns a hook function that collects the last REAL token's
        activation from each sample in a batch, using attention mask to handle
        padding correctly.

        Args:
            layer_name: Name identifier for this layer

        Returns:
            Hook function that can be registered with module.register_forward_hook()

        Note:
            Uses attention mask if available to find the last real token.
            Falls back to position -1 if no attention mask is set.
        """

        def hook(module: nn.Module, input: tuple, output: torch.Tensor):
            """
            Forward hook callback to collect activations.

            Args:
                module: The module this hook is attached to
                input: Input tensors to the module
                output: Output tensor from the module (expected shape: [B, T, H])
            """
            # Handle tuple output (some layers return (hidden_states, optional_outputs))
            if isinstance(output, tuple):
                output = output[0]

            # Detach and move to CPU for memory efficiency
            activations = output.detach().cpu()  # Shape: [B, T, H]

            # Initialize storage for this layer if needed
            if layer_name not in self.activations:
                self.activations[layer_name] = {'sum': None, 'count': 0}

            # Process each sample in batch
            batch_size = activations.shape[0]
            for i in range(batch_size):
                # Find last real token position
                if self._attention_mask is not None:
                    # Use attention mask to find last real token
                    mask = self._attention_mask[i].cpu()
                    last_pos = self._get_last_real_token_position(mask, i)
                else:
                    # Fallback: assume no padding (use position -1)
                    last_pos = -1
                    if i == 0:  # Warn once per batch
                        self.logger.warning(
                            f"No attention mask set for layer '{layer_name}'. "
                            f"Using position -1 (may be incorrect for right-padded models)"
                        )

                # Extract activation at last real token
                vec = activations[i, last_pos, :]

                # Update incremental sum
                if self.activations[layer_name]['sum'] is None:
                    self.activations[layer_name]['sum'] = vec.clone()
                else:
                    self.activations[layer_name]['sum'] += vec

                self.activations[layer_name]['count'] += 1

        return hook

    def register_hooks(self, layers: Dict[str, nn.Module]) -> None:
        """
        Register forward hooks on multiple layers.

        Args:
            layers: Dictionary mapping layer names to PyTorch modules

        Example:
            >>> hook_manager = ActivationHookManager()
            >>> layers = {
            ...     "layer_13_positive": model.model.layers[13].mlp.down_proj,
            ...     "layer_14_positive": model.model.layers[14].mlp.down_proj,
            ... }
            >>> hook_manager.register_hooks(layers)
        """
        self.logger.info(f"Registering hooks on {len(layers)} layers")

        for layer_name, layer_module in layers.items():
            hook_callback = self.collect_activation(layer_name)
            handle = layer_module.register_forward_hook(hook_callback)
            self.hooks[layer_name] = handle

            self.logger.debug(
                f"Registered hook on '{layer_name}' ({type(layer_module).__name__})"
            )

        self.logger.info(f"Successfully registered {len(self.hooks)} hooks")

    def remove_all_hooks(self) -> None:
        """
        Remove all registered forward hooks.

        This should be called after activation collection is complete to avoid
        memory leaks and unwanted side effects.
        """
        if not self.hooks:
            self.logger.debug("No hooks to remove")
            return

        self.logger.info(f"Removing {len(self.hooks)} hooks")

        for layer_name, handle in self.hooks.items():
            handle.remove()
            self.logger.debug(f"Removed hook from '{layer_name}'")

        self.hooks.clear()
        self.logger.info("All hooks removed")

    def get_mean_activations(self) -> Dict[str, torch.Tensor]:
        """
        Calculate and return mean activations for all layers.

        Returns:
            Dictionary mapping layer names to mean activation tensors

        Raises:
            ValueError: If no activations have been collected

        Example:
            >>> hook_manager = ActivationHookManager()
            >>> # ... register hooks and run inference ...
            >>> mean_acts = hook_manager.get_mean_activations()
            >>> print(mean_acts['layer_13_positive'].shape)  # [hidden_dim]
        """
        if not self.activations:
            raise ValueError("No activations collected. Run inference first.")

        mean_activations = {}

        for layer_name, stats in self.activations.items():
            if stats['sum'] is None or stats['count'] == 0:
                self.logger.warning(
                    f"Layer '{layer_name}' has no activations collected"
                )
                continue

            mean_act = stats['sum'] / stats['count']
            mean_activations[layer_name] = mean_act

            self.logger.debug(
                f"Calculated mean for '{layer_name}': "
                f"shape={mean_act.shape}, count={stats['count']}"
            )

        self.logger.info(
            f"Calculated mean activations for {len(mean_activations)} layers"
        )
        return mean_activations

    def reset(self) -> None:
        """
        Reset activation storage and clear attention mask.

        Useful when collecting activations for different prompt types
        (e.g., positive vs neutral) using the same hooks.
        """
        self.logger.info("Resetting activation storage")
        self.activations.clear()
        self._attention_mask = None

    def get_activation_stats(self) -> Dict[str, Dict]:
        """
        Get statistics about collected activations.

        Returns:
            Dictionary with stats for each layer including count and shape

        Example:
            >>> stats = hook_manager.get_activation_stats()
            >>> print(stats['layer_13_positive'])
            {'count': 1000, 'shape': [2560], 'mean_norm': 12.34}
        """
        stats = {}

        for layer_name, data in self.activations.items():
            layer_stats = {
                'count': data['count'],
                'shape': list(data['sum'].shape) if data['sum'] is not None else None,
            }

            if data['sum'] is not None and data['count'] > 0:
                mean = data['sum'] / data['count']
                layer_stats['mean_norm'] = mean.norm().item()

            stats[layer_name] = layer_stats

        return stats

    def __del__(self):
        """Cleanup: ensure all hooks are removed when object is destroyed."""
        if hasattr(self, 'hooks') and self.hooks:
            self.remove_all_hooks()
