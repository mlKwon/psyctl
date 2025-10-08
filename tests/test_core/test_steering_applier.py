"""Tests for SteeringApplier."""

import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

from psyctl.core.steering_applier import SteeringApplier


@pytest.fixture
def mock_model():
    """Create a mock PyTorch model."""
    model = MagicMock()
    model.device = torch.device("cpu")
    model.config._name_or_path = "test-model"
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    return tokenizer


@pytest.fixture
def mock_layer():
    """Create a mock layer module."""
    layer = MagicMock()
    layer.register_forward_hook = MagicMock(return_value=MagicMock())
    return layer


@pytest.fixture
def steering_applier():
    """Create SteeringApplier instance."""
    return SteeringApplier()


class TestGetSteeringAppliedModel:
    """Test suite for get_steering_applied_model method."""

    def test_validation_both_model_and_name(self, steering_applier):
        """Test that providing both model and model_name raises error."""
        with pytest.raises(ValueError, match="Cannot provide both"):
            steering_applier.get_steering_applied_model(
                steering_vector_path=Path("./test.safetensors"),
                model_name="test-model",
                model=MagicMock(),
                tokenizer=MagicMock(),
            )

    def test_validation_neither_model_nor_name(self, steering_applier):
        """Test that providing neither model nor model_name raises error."""
        with pytest.raises(ValueError, match="Must provide either"):
            steering_applier.get_steering_applied_model(
                steering_vector_path=Path("./test.safetensors"),
            )

    def test_validation_model_without_tokenizer(self, steering_applier, mock_model):
        """Test that providing model without tokenizer raises error."""
        with pytest.raises(ValueError, match="Must provide 'tokenizer'"):
            steering_applier.get_steering_applied_model(
                steering_vector_path=Path("./test.safetensors"),
                model=mock_model,
            )

    def test_validation_file_not_exists(self, steering_applier):
        """Test that non-existent steering vector file raises error."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            steering_applier.get_steering_applied_model(
                steering_vector_path=Path("./nonexistent.safetensors"),
                model_name="test-model",
            )

    @patch("psyctl.core.steering_applier.VectorStore")
    @patch("psyctl.core.steering_applier.LLMLoader")
    @patch("psyctl.core.steering_applier.LayerAccessor")
    def test_model_has_remove_steering_method(
        self,
        mock_accessor_class,
        mock_loader_class,
        mock_store_class,
        steering_applier,
        mock_model,
        mock_tokenizer,
        mock_layer,
        tmp_path,
    ):
        """Test that returned model has remove_steering method."""
        # Setup mocks
        vector_file = tmp_path / "test.safetensors"
        vector_file.touch()

        mock_store = steering_applier.vector_store
        mock_store.load_multi_layer.return_value = (
            {"layer_1": torch.randn(128)},
            {"model_name": "test"},
        )

        mock_accessor = steering_applier.layer_accessor
        mock_accessor.get_layer.return_value = mock_layer

        # Execute
        model, tokenizer = steering_applier.get_steering_applied_model(
            steering_vector_path=vector_file,
            model=mock_model,
            tokenizer=mock_tokenizer,
            strength=2.0,
        )

        # Verify
        assert hasattr(model, "remove_steering")
        assert callable(model.remove_steering)
        assert hasattr(model, "_steering_handles")

    @patch("psyctl.core.steering_applier.VectorStore")
    @patch("psyctl.core.steering_applier.LLMLoader")
    @patch("psyctl.core.steering_applier.LayerAccessor")
    def test_remove_steering_clears_hooks(
        self,
        mock_accessor_class,
        mock_loader_class,
        mock_store_class,
        steering_applier,
        mock_model,
        mock_tokenizer,
        mock_layer,
        tmp_path,
    ):
        """Test that remove_steering properly removes all hooks."""
        # Setup mocks
        vector_file = tmp_path / "test.safetensors"
        vector_file.touch()

        mock_store = steering_applier.vector_store
        mock_store.load_multi_layer.return_value = (
            {"layer_1": torch.randn(128), "layer_2": torch.randn(128)},
            {"model_name": "test"},
        )

        mock_accessor = steering_applier.layer_accessor
        mock_accessor.get_layer.return_value = mock_layer

        mock_handle_1 = MagicMock()
        mock_handle_2 = MagicMock()
        mock_layer.register_forward_hook.side_effect = [mock_handle_1, mock_handle_2]

        # Execute
        model, tokenizer = steering_applier.get_steering_applied_model(
            steering_vector_path=vector_file,
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        # Call remove_steering
        model.remove_steering()

        # Verify hooks were removed
        mock_handle_1.remove.assert_called_once()
        mock_handle_2.remove.assert_called_once()
        assert not hasattr(model, "_steering_handles")

    @patch("psyctl.core.steering_applier.VectorStore")
    @patch("psyctl.core.steering_applier.LLMLoader")
    @patch("psyctl.core.steering_applier.LayerAccessor")
    def test_multiple_layers_registered(
        self,
        mock_accessor_class,
        mock_loader_class,
        mock_store_class,
        steering_applier,
        mock_model,
        mock_tokenizer,
        mock_layer,
        tmp_path,
    ):
        """Test that hooks are registered for all layers."""
        # Setup mocks
        vector_file = tmp_path / "test.safetensors"
        vector_file.touch()

        mock_store = steering_applier.vector_store
        vectors = {
            "layer_1": torch.randn(128),
            "layer_2": torch.randn(128),
            "layer_3": torch.randn(128),
        }
        mock_store.load_multi_layer.return_value = (vectors, {"model_name": "test"})

        mock_accessor = steering_applier.layer_accessor
        mock_accessor.get_layer.return_value = mock_layer

        # Execute
        model, tokenizer = steering_applier.get_steering_applied_model(
            steering_vector_path=vector_file,
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        # Verify
        assert len(model._steering_handles) == 3
        assert mock_layer.register_forward_hook.call_count == 3


class TestApplySteeringVerbose:
    """Test suite for verbose parameter in apply_steering."""

    @patch("psyctl.core.steering_applier.VectorStore")
    @patch("psyctl.core.steering_applier.LLMLoader")
    @patch("psyctl.core.steering_applier.LayerAccessor")
    def test_verbose_logs_full_prompt(
        self,
        mock_accessor_class,
        mock_loader_class,
        mock_store_class,
        steering_applier,
        mock_model,
        mock_tokenizer,
        mock_layer,
        tmp_path,
    ):
        """Test that verbose=True logs full prompt."""
        # Setup mocks
        vector_file = tmp_path / "test.safetensors"
        vector_file.touch()

        mock_loader = steering_applier.llm_loader
        mock_loader.load_model.return_value = (mock_model, mock_tokenizer)

        mock_store = steering_applier.vector_store
        mock_store.load_multi_layer.return_value = (
            {"layer_1": torch.randn(128)},
            {"model_name": "test"},
        )

        mock_accessor = steering_applier.layer_accessor
        mock_accessor.get_layer.return_value = mock_layer

        # Mock tokenizer behavior
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
        }
        mock_tokenizer.decode.return_value = "Test output"

        # Mock model.generate
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        # Patch logger to capture log calls
        with patch.object(steering_applier.logger, "info") as mock_log_info:
            # Execute
            steering_applier.apply_steering(
                steering_vector_path=vector_file,
                model_name="test-model",
                input_text="Test input",
                verbose=True,
            )

            # Verify that full prompt was logged
            log_calls = [str(call) for call in mock_log_info.call_args_list]
            assert any("Full prompt after chat template" in str(call) for call in log_calls)

    @patch("psyctl.core.steering_applier.VectorStore")
    @patch("psyctl.core.steering_applier.LLMLoader")
    @patch("psyctl.core.steering_applier.LayerAccessor")
    def test_verbose_false_uses_debug(
        self,
        mock_accessor_class,
        mock_loader_class,
        mock_store_class,
        steering_applier,
        mock_model,
        mock_tokenizer,
        mock_layer,
        tmp_path,
    ):
        """Test that verbose=False uses debug logging."""
        # Setup mocks
        vector_file = tmp_path / "test.safetensors"
        vector_file.touch()

        mock_loader = steering_applier.llm_loader
        mock_loader.load_model.return_value = (mock_model, mock_tokenizer)

        mock_store = steering_applier.vector_store
        mock_store.load_multi_layer.return_value = (
            {"layer_1": torch.randn(128)},
            {"model_name": "test"},
        )

        mock_accessor = steering_applier.layer_accessor
        mock_accessor.get_layer.return_value = mock_layer

        # Mock tokenizer behavior
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
        }
        mock_tokenizer.decode.return_value = "Test output"

        # Mock model.generate
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        # Patch logger to capture log calls
        with patch.object(steering_applier.logger, "debug") as mock_log_debug:
            with patch.object(steering_applier.logger, "info") as mock_log_info:
                # Execute
                steering_applier.apply_steering(
                    steering_vector_path=vector_file,
                    model_name="test-model",
                    input_text="Test input",
                    verbose=False,
                )

                # Verify that debug was used for prompt
                log_calls = [str(call) for call in mock_log_debug.call_args_list]
                assert any("Prepared prompt" in str(call) for call in log_calls)

                # Verify that "Full prompt" was NOT logged to info
                info_calls = [str(call) for call in mock_log_info.call_args_list]
                assert not any(
                    "Full prompt after chat template" in str(call) for call in info_calls
                )


class TestPerLayerStrength:
    """Test suite for per-layer strength functionality."""

    @patch("psyctl.core.steering_applier.VectorStore")
    @patch("psyctl.core.steering_applier.LLMLoader")
    @patch("psyctl.core.steering_applier.LayerAccessor")
    def test_strength_as_float(
        self,
        mock_accessor_class,
        mock_loader_class,
        mock_store_class,
        steering_applier,
        mock_model,
        mock_tokenizer,
        mock_layer,
        tmp_path,
    ):
        """Test that float strength applies to all layers."""
        # Setup mocks
        vector_file = tmp_path / "test.safetensors"
        vector_file.touch()

        mock_loader = steering_applier.llm_loader
        mock_loader.load_model.return_value = (mock_model, mock_tokenizer)

        mock_store = steering_applier.vector_store
        mock_store.load_multi_layer.return_value = (
            {
                "layer_1": torch.randn(128),
                "layer_2": torch.randn(128),
            },
            {"model_name": "test"},
        )

        mock_accessor = steering_applier.layer_accessor
        mock_accessor.get_layer.return_value = mock_layer

        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "Test output"
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        # Patch _make_steering_hook to track calls
        with patch.object(
            steering_applier, "_make_steering_hook", wraps=steering_applier._make_steering_hook
        ) as mock_hook:
            # Execute with float strength
            steering_applier.apply_steering(
                steering_vector_path=vector_file,
                model_name="test-model",
                input_text="Test input",
                strength=2.5,
            )

            # Verify all hooks received the same strength
            assert mock_hook.call_count == 2
            for call in mock_hook.call_args_list:
                assert call[0][2] == 2.5  # strength is 3rd positional arg

    @patch("psyctl.core.steering_applier.VectorStore")
    @patch("psyctl.core.steering_applier.LLMLoader")
    @patch("psyctl.core.steering_applier.LayerAccessor")
    def test_strength_as_dict(
        self,
        mock_accessor_class,
        mock_loader_class,
        mock_store_class,
        steering_applier,
        mock_model,
        mock_tokenizer,
        mock_layer,
        tmp_path,
    ):
        """Test that dict strength applies per-layer values."""
        # Setup mocks
        vector_file = tmp_path / "test.safetensors"
        vector_file.touch()

        mock_loader = steering_applier.llm_loader
        mock_loader.load_model.return_value = (mock_model, mock_tokenizer)

        mock_store = steering_applier.vector_store
        mock_store.load_multi_layer.return_value = (
            {
                "layer_1": torch.randn(128),
                "layer_2": torch.randn(128),
                "layer_3": torch.randn(128),
            },
            {"model_name": "test"},
        )

        mock_accessor = steering_applier.layer_accessor
        mock_accessor.get_layer.return_value = mock_layer

        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "Test output"
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        # Patch _make_steering_hook to track calls
        with patch.object(
            steering_applier, "_make_steering_hook", wraps=steering_applier._make_steering_hook
        ) as mock_hook:
            # Execute with dict strength
            strength_dict = {
                "layer_1": 1.0,
                "layer_2": 2.5,
                "layer_3": 3.0,
            }
            steering_applier.apply_steering(
                steering_vector_path=vector_file,
                model_name="test-model",
                input_text="Test input",
                strength=strength_dict,
            )

            # Verify hooks received correct per-layer strengths
            assert mock_hook.call_count == 3
            strengths_used = [call[0][2] for call in mock_hook.call_args_list]
            assert 1.0 in strengths_used
            assert 2.5 in strengths_used
            assert 3.0 in strengths_used

    @patch("psyctl.core.steering_applier.VectorStore")
    @patch("psyctl.core.steering_applier.LLMLoader")
    @patch("psyctl.core.steering_applier.LayerAccessor")
    def test_strength_dict_with_missing_layers(
        self,
        mock_accessor_class,
        mock_loader_class,
        mock_store_class,
        steering_applier,
        mock_model,
        mock_tokenizer,
        mock_layer,
        tmp_path,
    ):
        """Test that missing layers in dict use default strength."""
        # Setup mocks
        vector_file = tmp_path / "test.safetensors"
        vector_file.touch()

        mock_loader = steering_applier.llm_loader
        mock_loader.load_model.return_value = (mock_model, mock_tokenizer)

        mock_store = steering_applier.vector_store
        mock_store.load_multi_layer.return_value = (
            {
                "layer_1": torch.randn(128),
                "layer_2": torch.randn(128),
                "layer_3": torch.randn(128),
            },
            {"model_name": "test"},
        )

        mock_accessor = steering_applier.layer_accessor
        mock_accessor.get_layer.return_value = mock_layer

        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "Test output"
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        # Patch _make_steering_hook to track calls
        with patch.object(
            steering_applier, "_make_steering_hook", wraps=steering_applier._make_steering_hook
        ) as mock_hook:
            # Execute with partial dict strength
            strength_dict = {
                "layer_1": 2.0,
                # layer_2 and layer_3 missing - should use default 1.0
            }
            steering_applier.apply_steering(
                steering_vector_path=vector_file,
                model_name="test-model",
                input_text="Test input",
                strength=strength_dict,
            )

            # Verify hooks received correct strengths
            assert mock_hook.call_count == 3
            strengths_used = [call[0][2] for call in mock_hook.call_args_list]
            assert strengths_used.count(1.0) == 2  # layer_2 and layer_3
            assert strengths_used.count(2.0) == 1  # layer_1

    def test_resolve_layer_strength_with_float(self, steering_applier):
        """Test _resolve_layer_strength with float input."""
        result = steering_applier._resolve_layer_strength(2.5, "any_layer")
        assert result == 2.5

    def test_resolve_layer_strength_with_dict_present(self, steering_applier):
        """Test _resolve_layer_strength with dict when layer is present."""
        strength_dict = {"layer_1": 3.0, "layer_2": 1.5}
        result = steering_applier._resolve_layer_strength(strength_dict, "layer_1")
        assert result == 3.0

    def test_resolve_layer_strength_with_dict_missing(self, steering_applier):
        """Test _resolve_layer_strength with dict when layer is missing."""
        strength_dict = {"layer_1": 3.0}
        result = steering_applier._resolve_layer_strength(strength_dict, "layer_2")
        assert result == 1.0  # default

    def test_resolve_layer_strength_with_dict_custom_default(self, steering_applier):
        """Test _resolve_layer_strength with custom default."""
        strength_dict = {"layer_1": 3.0}
        result = steering_applier._resolve_layer_strength(
            strength_dict, "layer_2", default=5.0
        )
        assert result == 5.0
