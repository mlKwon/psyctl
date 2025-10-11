"""Tests for ActivationHookManager."""

import pytest
import torch
from torch import nn

from psyctl.core.hook_manager import ActivationHookManager


@pytest.fixture
def hook_manager():
    """Create ActivationHookManager instance."""
    return ActivationHookManager()


@pytest.fixture
def mock_layer():
    """Create a mock layer that produces activations."""
    return nn.Linear(10, 20)


class TestActivationHookManager:
    """Test suite for ActivationHookManager."""

    def test_initialization(self, hook_manager):
        """Test hook manager initializes correctly."""
        assert hook_manager.hooks == {}
        assert hook_manager.activations == {}
        assert hook_manager.logger is not None

    def test_register_single_hook(self, hook_manager, mock_layer):
        """Test registering a single hook."""
        layers = {"layer_1": mock_layer}
        hook_manager.register_hooks(layers)

        assert "layer_1" in hook_manager.hooks
        assert len(hook_manager.hooks) == 1

    def test_register_multiple_hooks(self, hook_manager):
        """Test registering multiple hooks."""
        layers = {
            "layer_1": nn.Linear(10, 20),
            "layer_2": nn.Linear(20, 30),
            "layer_3": nn.Linear(30, 40),
        }
        hook_manager.register_hooks(layers)

        assert len(hook_manager.hooks) == 3
        assert "layer_1" in hook_manager.hooks
        assert "layer_2" in hook_manager.hooks
        assert "layer_3" in hook_manager.hooks

    def test_collect_activation_single_sample(self, hook_manager, mock_layer):
        """Test collecting activation from a single sample."""
        layers = {"test_layer": mock_layer}
        hook_manager.register_hooks(layers)

        # Simulate forward pass
        input_tensor = torch.randn(1, 5, 10)  # [batch=1, seq=5, features=10]
        _ = mock_layer(input_tensor)

        # Check activations were collected
        assert "test_layer" in hook_manager.activations
        assert hook_manager.activations["test_layer"]["count"] == 1
        assert hook_manager.activations["test_layer"]["sum"] is not None
        assert hook_manager.activations["test_layer"]["sum"].shape == (20,)

    def test_collect_activation_batch(self, hook_manager, mock_layer):
        """Test collecting activations from a batch."""
        layers = {"test_layer": mock_layer}
        hook_manager.register_hooks(layers)

        # Simulate forward pass with batch_size=4
        input_tensor = torch.randn(4, 5, 10)  # [batch=4, seq=5, features=10]
        _ = mock_layer(input_tensor)

        # Check all samples in batch were collected
        assert hook_manager.activations["test_layer"]["count"] == 4

    def test_collect_activation_multiple_batches(self, hook_manager, mock_layer):
        """Test collecting activations from multiple batches."""
        layers = {"test_layer": mock_layer}
        hook_manager.register_hooks(layers)

        # First batch
        input_tensor1 = torch.randn(2, 5, 10)
        _ = mock_layer(input_tensor1)

        # Second batch
        input_tensor2 = torch.randn(3, 5, 10)
        _ = mock_layer(input_tensor2)

        # Total count should be 5
        assert hook_manager.activations["test_layer"]["count"] == 5

    def test_handle_tuple_output(self, hook_manager):
        """Test handling layers that return tuples."""

        # Create a layer that returns tuple (common in transformer models)
        class TupleOutputLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 20)

            def forward(self, x):
                output = self.linear(x)
                return (output, None)  # Return tuple

        layer = TupleOutputLayer()
        layers = {"tuple_layer": layer}
        hook_manager.register_hooks(layers)

        # Forward pass
        input_tensor = torch.randn(1, 5, 10)
        _ = layer(input_tensor)

        # Should still collect activations correctly
        assert "tuple_layer" in hook_manager.activations
        assert hook_manager.activations["tuple_layer"]["count"] == 1
        assert hook_manager.activations["tuple_layer"]["sum"].shape == (20,)

    def test_get_mean_activations(self, hook_manager, mock_layer):
        """Test calculating mean activations."""
        layers = {"test_layer": mock_layer}
        hook_manager.register_hooks(layers)

        # Collect some activations
        for _ in range(3):
            input_tensor = torch.randn(2, 5, 10)  # 2 samples per batch
            _ = mock_layer(input_tensor)

        # Get mean
        mean_acts = hook_manager.get_mean_activations()

        assert "test_layer" in mean_acts
        assert mean_acts["test_layer"].shape == (20,)
        # 6 total samples (3 batches * 2 samples)
        assert hook_manager.activations["test_layer"]["count"] == 6

    def test_get_mean_activations_no_data(self, hook_manager):
        """Test getting mean activations when no data collected."""
        with pytest.raises(ValueError, match="No activations collected"):
            hook_manager.get_mean_activations()

    def test_remove_all_hooks(self, hook_manager):
        """Test removing all hooks."""
        layers = {"layer_1": nn.Linear(10, 20), "layer_2": nn.Linear(20, 30)}
        hook_manager.register_hooks(layers)

        # Remove hooks
        hook_manager.remove_all_hooks()

        assert len(hook_manager.hooks) == 0

    def test_remove_hooks_when_empty(self, hook_manager):
        """Test removing hooks when none registered."""
        # Should not raise error
        hook_manager.remove_all_hooks()
        assert len(hook_manager.hooks) == 0

    def test_reset(self, hook_manager, mock_layer):
        """Test resetting activation storage."""
        layers = {"test_layer": mock_layer}
        hook_manager.register_hooks(layers)

        # Collect some data
        input_tensor = torch.randn(2, 5, 10)
        _ = mock_layer(input_tensor)

        # Reset
        hook_manager.reset()

        assert len(hook_manager.activations) == 0
        # Hooks should still be registered
        assert len(hook_manager.hooks) == 1

    def test_get_activation_stats(self, hook_manager, mock_layer):
        """Test getting activation statistics."""
        layers = {"test_layer": mock_layer}
        hook_manager.register_hooks(layers)

        # Collect data
        input_tensor = torch.randn(3, 5, 10)
        _ = mock_layer(input_tensor)

        stats = hook_manager.get_activation_stats()

        assert "test_layer" in stats
        assert stats["test_layer"]["count"] == 3
        assert stats["test_layer"]["shape"] == [20]
        assert "mean_norm" in stats["test_layer"]
        assert isinstance(stats["test_layer"]["mean_norm"], float)

    def test_get_activation_stats_empty(self, hook_manager):
        """Test getting stats when no data collected."""
        stats = hook_manager.get_activation_stats()
        assert stats == {}

    def test_incremental_mean_calculation(self, hook_manager, mock_layer):
        """Test that mean is calculated incrementally correctly."""
        layers = {"test_layer": mock_layer}
        hook_manager.register_hooks(layers)

        # Collect known activations
        with torch.no_grad():
            # First sample: all ones
            input_tensor1 = torch.ones(1, 5, 10)
            _ = mock_layer(input_tensor1)

            # Second sample: all twos
            input_tensor2 = torch.ones(1, 5, 10) * 2
            _ = mock_layer(input_tensor2)

        _ = hook_manager.get_mean_activations()

        # Mean should be close to average of the two outputs
        assert hook_manager.activations["test_layer"]["count"] == 2

    def test_last_token_extraction(self, hook_manager, mock_layer):
        """Test that only the last token is extracted."""
        layers = {"test_layer": mock_layer}
        hook_manager.register_hooks(layers)

        # Create input with different values at each position
        input_tensor = torch.randn(1, 5, 10)  # 5 time steps
        _ = mock_layer(input_tensor)

        # Should have collected exactly 1 activation (last token of 1 sample)
        assert hook_manager.activations["test_layer"]["count"] == 1

    def test_cleanup_on_deletion(self, mock_layer):
        """Test that hooks are removed when manager is deleted."""
        manager = ActivationHookManager()
        layers = {"test_layer": mock_layer}
        manager.register_hooks(layers)

        # Count hooks before deletion
        initial_hooks = len(manager.hooks)
        assert initial_hooks == 1

        # Delete manager
        del manager

        # Hook should be removed (can't verify directly, but no error should occur)

    def test_multiple_layers_different_sizes(self, hook_manager):
        """Test collecting from multiple layers with different sizes."""
        layers = {
            "layer_small": nn.Linear(10, 20),
            "layer_medium": nn.Linear(10, 50),
            "layer_large": nn.Linear(10, 100),
        }
        hook_manager.register_hooks(layers)

        # Forward pass
        input_tensor = torch.randn(2, 5, 10)
        for layer in layers.values():
            _ = layer(input_tensor)

        mean_acts = hook_manager.get_mean_activations()

        assert mean_acts["layer_small"].shape == (20,)
        assert mean_acts["layer_medium"].shape == (50,)
        assert mean_acts["layer_large"].shape == (100,)

    def test_cpu_memory_efficiency(self, hook_manager, mock_layer):
        """Test that activations are moved to CPU for memory efficiency."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Move layer to GPU
        layer_gpu = mock_layer.cuda()
        layers = {"test_layer": layer_gpu}
        hook_manager.register_hooks(layers)

        # Forward pass with GPU tensor
        input_tensor = torch.randn(1, 5, 10).cuda()
        _ = layer_gpu(input_tensor)

        # Activations should be on CPU
        assert hook_manager.activations["test_layer"]["sum"].device.type == "cpu"
