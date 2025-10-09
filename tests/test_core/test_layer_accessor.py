"""Tests for layer accessor with wildcard expansion."""

import pytest
import torch
from torch import nn

from psyctl.core.layer_accessor import LayerAccessor


class SimpleModel(nn.Module):
    """Simple model for testing layer access."""

    def __init__(self, num_layers=5):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            nn.Module() for _ in range(num_layers)
        ])
        for i, layer in enumerate(self.model.layers):
            layer.mlp = nn.Module()
            layer.mlp.down_proj = nn.Linear(10, 10)
            layer.mlp.up_proj = nn.Linear(10, 10)
            layer.attention = nn.Linear(10, 10)


class TestLayerAccessor:
    """Test LayerAccessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.accessor = LayerAccessor()
        self.model = SimpleModel(num_layers=5)

    def test_basic_layer_access(self):
        """Test basic layer access without wildcards."""
        layer = self.accessor.get_layer(self.model, "model.layers[0].mlp.down_proj")
        assert isinstance(layer, nn.Linear)

    def test_wildcard_all_layers(self):
        """Test wildcard expansion for all layers."""
        patterns = ["model.layers[*].mlp.down_proj"]
        expanded = self.accessor.expand_layer_patterns(self.model, patterns)

        assert len(expanded) == 5
        assert "model.layers[0].mlp.down_proj" in expanded
        assert "model.layers[4].mlp.down_proj" in expanded

    def test_slice_notation_range(self):
        """Test slice notation with start:end."""
        patterns = ["model.layers[1:3].mlp.down_proj"]
        expanded = self.accessor.expand_layer_patterns(self.model, patterns)

        assert len(expanded) == 2
        assert "model.layers[1].mlp.down_proj" in expanded
        assert "model.layers[2].mlp.down_proj" in expanded

    def test_slice_notation_from_start(self):
        """Test slice notation with start: (to end)."""
        patterns = ["model.layers[3:].mlp.down_proj"]
        expanded = self.accessor.expand_layer_patterns(self.model, patterns)

        assert len(expanded) == 2  # layers 3, 4
        assert "model.layers[3].mlp.down_proj" in expanded
        assert "model.layers[4].mlp.down_proj" in expanded

    def test_slice_notation_to_end(self):
        """Test slice notation with :end (from 0)."""
        patterns = ["model.layers[:2].mlp.down_proj"]
        expanded = self.accessor.expand_layer_patterns(self.model, patterns)

        assert len(expanded) == 2  # layers 0, 1
        assert "model.layers[0].mlp.down_proj" in expanded
        assert "model.layers[1].mlp.down_proj" in expanded

    def test_slice_notation_with_step(self):
        """Test slice notation with start:end:step."""
        patterns = ["model.layers[0:5:2].mlp.down_proj"]
        expanded = self.accessor.expand_layer_patterns(self.model, patterns)

        assert len(expanded) == 3  # layers 0, 2, 4
        assert "model.layers[0].mlp.down_proj" in expanded
        assert "model.layers[2].mlp.down_proj" in expanded
        assert "model.layers[4].mlp.down_proj" in expanded

    def test_no_wildcard(self):
        """Test that patterns without wildcards are passed through."""
        patterns = ["model.layers[0].mlp.down_proj"]
        expanded = self.accessor.expand_layer_patterns(self.model, patterns)

        assert len(expanded) == 1
        assert expanded[0] == "model.layers[0].mlp.down_proj"

    def test_multiple_patterns(self):
        """Test multiple patterns in one call."""
        patterns = [
            "model.layers[0].mlp.down_proj",
            "model.layers[1:3].mlp.up_proj"
        ]
        expanded = self.accessor.expand_layer_patterns(self.model, patterns)

        assert len(expanded) == 3  # 1 + 2
        assert "model.layers[0].mlp.down_proj" in expanded
        assert "model.layers[1].mlp.up_proj" in expanded
        assert "model.layers[2].mlp.up_proj" in expanded

    def test_wildcard_at_different_levels(self):
        """Test wildcard expansion for different module levels."""
        patterns = ["model.layers[*].mlp.up_proj"]
        expanded = self.accessor.expand_layer_patterns(self.model, patterns)

        assert len(expanded) == 5

    def test_has_wildcard(self):
        """Test wildcard detection."""
        assert self.accessor._has_wildcard("model.layers[*].mlp")
        assert self.accessor._has_wildcard("model.layers[1:3].mlp")
        assert self.accessor._has_wildcard("model.layers[:5].mlp")
        assert not self.accessor._has_wildcard("model.layers[3].mlp")

    def test_parse_slice(self):
        """Test slice parsing."""
        # Simple range
        result = self.accessor._parse_slice("1:3", 10)
        assert result == (1, 3, 1)

        # From start
        result = self.accessor._parse_slice("5:", 10)
        assert result == (5, 10, 1)

        # To end
        result = self.accessor._parse_slice(":5", 10)
        assert result == (0, 5, 1)

        # With step
        result = self.accessor._parse_slice("0:10:2", 10)
        assert result == (0, 10, 2)

    def test_format_layer_path(self):
        """Test formatting layer path from components."""
        components = ['model', 'layers', '13', 'mlp', 'down_proj']
        formatted = self.accessor.format_layer_path(components)
        assert formatted == 'model.layers[13].mlp.down_proj'
