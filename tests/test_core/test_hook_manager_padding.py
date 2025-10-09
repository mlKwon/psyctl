"""Tests for ActivationHookManager padding handling."""
import pytest
import torch

from psyctl.core.hook_manager import ActivationHookManager


class TestPaddingExtraction:
    """Test that activation extraction handles padding correctly."""

    def test_left_padding_extraction(self):
        """Test that left-padded batches extract correct tokens.

        Returns ABSOLUTE position in sequence, not relative position.
        """
        hook_manager = ActivationHookManager()

        # Simulate left-padded batch
        # [PAD, PAD, Token1, Token2]  -> last real token at absolute position 3
        # [PAD, Token3, Token4, Token5] -> last real token at absolute position 3
        attention_mask = torch.tensor([
            [0, 0, 1, 1],  # Last real token at index 3
            [0, 1, 1, 1],  # Last real token at index 3
        ])

        hook_manager.set_attention_mask(attention_mask)

        # Sample 0: last real token at absolute position 3
        pos0 = hook_manager._get_last_real_token_position(attention_mask[0], 0)
        assert pos0 == 3, f"Expected position 3, got {pos0}"

        # Sample 1: last real token at absolute position 3
        pos1 = hook_manager._get_last_real_token_position(attention_mask[1], 1)
        assert pos1 == 3, f"Expected position 3, got {pos1}"

    def test_right_padding_extraction(self):
        """Test that right-padded batches extract correct tokens."""
        hook_manager = ActivationHookManager()

        # Simulate right-padded batch
        # [Token1, Token2, PAD, PAD]
        # [Token3, Token4, Token5, PAD]
        attention_mask = torch.tensor([
            [1, 1, 0, 0],  # 2 real tokens at positions 0, 1
            [1, 1, 1, 0],  # 3 real tokens at positions 0, 1, 2
        ])

        hook_manager.set_attention_mask(attention_mask)

        # Sample 0: last real token at position 1
        pos0 = hook_manager._get_last_real_token_position(attention_mask[0], 0)
        assert pos0 == 1, f"Expected position 1, got {pos0}"

        # Sample 1: last real token at position 2
        pos1 = hook_manager._get_last_real_token_position(attention_mask[1], 1)
        assert pos1 == 2, f"Expected position 2, got {pos1}"

    def test_no_padding(self):
        """Test batches without padding."""
        hook_manager = ActivationHookManager()

        # No padding - all tokens are real
        attention_mask = torch.tensor([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ])

        hook_manager.set_attention_mask(attention_mask)

        # Both should point to last position (index 3)
        pos0 = hook_manager._get_last_real_token_position(attention_mask[0], 0)
        assert pos0 == 3, f"Expected position 3, got {pos0}"

        pos1 = hook_manager._get_last_real_token_position(attention_mask[1], 1)
        assert pos1 == 3, f"Expected position 3, got {pos1}"

    def test_single_token(self):
        """Test samples with only one real token."""
        hook_manager = ActivationHookManager()

        # Single token samples
        attention_mask = torch.tensor([
            [1, 0, 0, 0],  # 1 real token at absolute position 0
            [0, 0, 0, 1],  # 1 real token at absolute position 3
        ])

        hook_manager.set_attention_mask(attention_mask)

        # Sample 0: only token at absolute position 0
        pos0 = hook_manager._get_last_real_token_position(attention_mask[0], 0)
        assert pos0 == 0, f"Expected position 0, got {pos0}"

        # Sample 1: only token at absolute position 3
        pos1 = hook_manager._get_last_real_token_position(attention_mask[1], 1)
        assert pos1 == 3, f"Expected position 3, got {pos1}"

    def test_all_padding_raises_error(self):
        """Test that all-padding sample raises error."""
        hook_manager = ActivationHookManager()

        # All padding (invalid)
        attention_mask = torch.tensor([0, 0, 0, 0])

        with pytest.raises(ValueError, match="entirely padding"):
            hook_manager._get_last_real_token_position(attention_mask, 0)

    def test_hook_extraction_with_mask(self):
        """Test full hook extraction with attention mask."""
        hook_manager = ActivationHookManager()

        # Create mock activations (batch_size=2, seq_len=4, hidden_dim=3)
        activations = torch.randn(2, 4, 3)

        # Right-padded attention mask
        attention_mask = torch.tensor([
            [1, 1, 0, 0],  # Last real token at position 1
            [1, 1, 1, 0],  # Last real token at position 2
        ])

        hook_manager.set_attention_mask(attention_mask)

        # Create and call hook
        hook_fn = hook_manager.collect_activation("test_layer")
        hook_fn(None, None, activations)

        # Check that correct activations were collected
        stats = hook_manager.get_activation_stats()
        assert "test_layer" in stats
        assert stats["test_layer"]["count"] == 2

        # Get mean and verify shape
        mean_acts = hook_manager.get_mean_activations()
        assert "test_layer" in mean_acts
        assert mean_acts["test_layer"].shape == (3,)  # Hidden dim

    def test_hook_extraction_without_mask_warns(self):
        """Test that hook warns when no attention mask is set."""
        hook_manager = ActivationHookManager()

        # Create mock activations
        activations = torch.randn(2, 4, 3)

        # Create hook WITHOUT setting attention mask
        hook_fn = hook_manager.collect_activation("test_layer")

        # Should not crash, but will use position -1
        hook_fn(None, None, activations)

        # Should still collect activations
        stats = hook_manager.get_activation_stats()
        assert "test_layer" in stats
        assert stats["test_layer"]["count"] == 2

    def test_reset_clears_mask(self):
        """Test that reset() clears attention mask."""
        hook_manager = ActivationHookManager()

        # Set mask
        attention_mask = torch.tensor([[1, 1, 0, 0]])
        hook_manager.set_attention_mask(attention_mask)
        assert hook_manager._attention_mask is not None

        # Reset
        hook_manager.reset()
        assert hook_manager._attention_mask is None
        assert len(hook_manager.activations) == 0


class TestMixedLengthBatches:
    """Test extraction with realistic mixed-length batches."""

    def test_realistic_caa_batch(self):
        """Test with realistic CAA dataset batch lengths."""
        hook_manager = ActivationHookManager()

        # Simulate CAA batch with various lengths (after tokenization)
        # Lengths: [216, 166, 188, 159, 257, 209, 184, 190]
        # Padded to max length: 257
        batch_size = 8
        max_len = 257
        real_lengths = [216, 166, 188, 159, 257, 209, 184, 190]

        # Create attention mask
        attention_mask = torch.zeros(batch_size, max_len)
        for i, length in enumerate(real_lengths):
            attention_mask[i, :length] = 1

        hook_manager.set_attention_mask(attention_mask)

        # Verify last positions
        expected_positions = [length - 1 for length in real_lengths]
        for i, expected in enumerate(expected_positions):
            pos = hook_manager._get_last_real_token_position(attention_mask[i], i)
            assert pos == expected, f"Sample {i}: expected {expected}, got {pos}"

    def test_extreme_length_variance(self):
        """Test with extreme length variance (short to very long)."""
        hook_manager = ActivationHookManager()

        # Very short to very long
        batch_size = 4
        max_len = 500
        real_lengths = [5, 50, 250, 500]

        # Create attention mask
        attention_mask = torch.zeros(batch_size, max_len)
        for i, length in enumerate(real_lengths):
            attention_mask[i, :length] = 1

        hook_manager.set_attention_mask(attention_mask)

        # All should find correct last positions
        for i, length in enumerate(real_lengths):
            pos = hook_manager._get_last_real_token_position(attention_mask[i], i)
            assert pos == length - 1
