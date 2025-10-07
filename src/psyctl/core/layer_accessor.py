"""Layer accessor for dynamic model layer access."""

import re
from typing import List

from torch import nn

from psyctl.core.logger import get_logger


class LayerAccessor:
    """
    Parse layer string and access model layers dynamically.

    Supports dot notation with bracket indexing for accessing nested model layers.
    Example: "model.layers[13].mlp.down_proj" → actual PyTorch module

    Attributes:
        logger: Logger instance for debugging
    """

    def __init__(self):
        """Initialize LayerAccessor with logger."""
        self.logger = get_logger("layer_accessor")

    def parse_layer_path(self, layer_str: str) -> List[str]:
        """
        Parse layer path string into components.

        Args:
            layer_str: Layer path string (e.g., "model.layers[13].mlp.down_proj")

        Returns:
            List of path components (e.g., ["model", "layers", "13", "mlp", "down_proj"])

        Example:
            >>> accessor = LayerAccessor()
            >>> accessor.parse_layer_path("model.layers[13].mlp.down_proj")
            ['model', 'layers', '13', 'mlp', 'down_proj']
        """
        # Replace brackets with dots: model.layers[13] → model.layers.13
        normalized = re.sub(r'\[(\d+)\]', r'.\1', layer_str)

        # Split by dots and filter empty strings
        components = [c for c in normalized.split('.') if c]

        self.logger.debug(f"Parsed layer path '{layer_str}' → {components}")
        return components

    def get_layer(self, model: nn.Module, layer_str: str) -> nn.Module:
        """
        Get layer module from model using layer path string.

        Args:
            model: PyTorch model
            layer_str: Layer path string (e.g., "model.layers[13].mlp.down_proj")

        Returns:
            PyTorch module at the specified path

        Raises:
            AttributeError: If layer path is invalid or layer doesn't exist
            IndexError: If array index is out of bounds

        Example:
            >>> accessor = LayerAccessor()
            >>> layer = accessor.get_layer(model, "model.layers[13].mlp.down_proj")
        """
        components = self.parse_layer_path(layer_str)

        try:
            current = model
            path_so_far = []

            for component in components:
                path_so_far.append(component)

                # Check if component is a digit (array index)
                if component.isdigit():
                    index = int(component)
                    if not isinstance(current, (nn.ModuleList, list, tuple)):
                        raise AttributeError(
                            f"Cannot index into {type(current).__name__} "
                            f"at path '{'.'.join(path_so_far[:-1])}'"
                        )
                    if index >= len(current):
                        raise IndexError(
                            f"Index {index} out of range for module list "
                            f"of length {len(current)} at path '{'.'.join(path_so_far[:-1])}'"
                        )
                    current = current[index]
                else:
                    # Regular attribute access
                    if not hasattr(current, component):
                        raise AttributeError(
                            f"Module has no attribute '{component}' "
                            f"at path '{'.'.join(path_so_far[:-1])}'"
                        )
                    current = getattr(current, component)

            self.logger.debug(
                f"Successfully accessed layer '{layer_str}' → {type(current).__name__}"
            )
            return current

        except (AttributeError, IndexError) as e:
            self.logger.error(f"Failed to access layer '{layer_str}': {e}")
            raise

    def validate_layers(self, model: nn.Module, layer_strs: List[str]) -> bool:
        """
        Validate that all layer paths exist in the model.

        Args:
            model: PyTorch model
            layer_strs: List of layer path strings

        Returns:
            True if all layers are valid, False otherwise

        Example:
            >>> accessor = LayerAccessor()
            >>> layers = ["model.layers[13].mlp.down_proj", "model.layers[14].mlp.down_proj"]
            >>> is_valid = accessor.validate_layers(model, layers)
        """
        self.logger.info(f"Validating {len(layer_strs)} layer paths...")

        all_valid = True
        for layer_str in layer_strs:
            try:
                self.get_layer(model, layer_str)
                self.logger.debug(f"Valid layer: {layer_str}")
            except (AttributeError, IndexError) as e:
                self.logger.error(f"Invalid layer '{layer_str}': {e}")
                all_valid = False

        if all_valid:
            self.logger.info("All layer paths are valid")
        else:
            self.logger.error("Some layer paths are invalid")

        return all_valid

    def get_layer_info(self, model: nn.Module, layer_str: str) -> dict:
        """
        Get information about a layer.

        Args:
            model: PyTorch model
            layer_str: Layer path string

        Returns:
            Dictionary with layer information (type, parameters, etc.)

        Example:
            >>> accessor = LayerAccessor()
            >>> info = accessor.get_layer_info(model, "model.layers[13].mlp.down_proj")
            >>> print(info['type'], info['num_parameters'])
        """
        layer = self.get_layer(model, layer_str)

        info = {
            'path': layer_str,
            'type': type(layer).__name__,
            'num_parameters': sum(p.numel() for p in layer.parameters()),
            'trainable_parameters': sum(
                p.numel() for p in layer.parameters() if p.requires_grad
            ),
        }

        # Add shape info for Linear layers
        if isinstance(layer, nn.Linear):
            info['in_features'] = layer.in_features
            info['out_features'] = layer.out_features
            info['bias'] = layer.bias is not None

        self.logger.debug(f"Layer info for '{layer_str}': {info}")
        return info
