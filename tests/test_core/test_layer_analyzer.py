"""Tests for LayerAnalyzer."""

import pytest
import torch
from torch import nn
from datasets import Dataset

from psyctl.core.layer_analyzer import LayerAnalyzer


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, num_layers=3):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            nn.Module() for _ in range(num_layers)
        ])
        for i, layer in enumerate(self.model.layers):
            layer.mlp = nn.Linear(10, 10)

        self.config = type('obj', (object,), {'_name_or_path': 'test-model'})()
        self._device = torch.device('cpu')

    @property
    def device(self):
        """Return the device of the model."""
        return self._device

    def forward(self, input_ids, attention_mask=None):
        """Dummy forward pass."""
        batch_size = input_ids.shape[0]
        return type('obj', (object,), {
            'logits': torch.randn(batch_size, input_ids.shape[1], 100)
        })()


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.pad_token = "[PAD]"
        self.pad_token_id = 0
        self.padding_side = "left"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        """Mock chat template application."""
        return messages[0]["content"]

    def __call__(self, prompts, return_tensors="pt", padding=True, truncation=True):
        """Mock tokenization."""
        if isinstance(prompts, str):
            prompts = [prompts]

        max_len = 20
        batch_size = len(prompts)

        input_ids = torch.randint(1, 100, (batch_size, max_len))
        attention_mask = torch.ones(batch_size, max_len)

        class MockBatch:
            def __init__(self):
                self.input_ids = input_ids
                self.attention_mask = attention_mask
                self._data = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }

            def to(self, device):
                return self

            def __getitem__(self, key):
                return self._data[key]

            def keys(self):
                return self._data.keys()

            def values(self):
                return self._data.values()

            def items(self):
                return self._data.items()

        return MockBatch()


class TestLayerAnalyzer:
    """Test LayerAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = LayerAnalyzer()
        self.model = SimpleModel(num_layers=3)
        self.tokenizer = MockTokenizer()

    def test_analyze_with_dict_dataset(self):
        """Test that analyze_layers handles dict-based datasets correctly."""
        # Create a dataset in dict format (like Hugging Face datasets)
        dataset_dict = {
            "situation": ["Test situation 1", "Test situation 2"],
            "positive": ["Positive response 1", "Positive response 2"],
            "neutral": ["Neutral response 1", "Neutral response 2"]
        }
        dataset = Dataset.from_dict(dataset_dict)

        # This should not raise AttributeError
        results = self.analyzer.analyze_layers(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=["model.layers[0].mlp"],
            dataset=dataset,
            batch_size=2,
            method="svm",
            top_k=1
        )

        assert results is not None
        assert "rankings" in results
        assert len(results["rankings"]) == 1

    def test_analyze_with_list_dataset(self):
        """Test that analyze_layers handles list-based datasets correctly."""
        # Create a dataset in list format
        dataset = [
            {
                "situation": "Test situation 1",
                "positive": "Positive response 1",
                "neutral": "Neutral response 1"
            },
            {
                "situation": "Test situation 2",
                "positive": "Positive response 2",
                "neutral": "Neutral response 2"
            }
        ]

        results = self.analyzer.analyze_layers(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=["model.layers[0].mlp"],
            dataset=dataset,
            batch_size=2,
            method="svm",
            top_k=1
        )

        assert results is not None
        assert "rankings" in results

    def test_analyze_with_question_field(self):
        """Test that analyze_layers handles datasets with 'question' field."""
        dataset = [
            {
                "question": "Test question 1",
                "positive": "Positive response 1",
                "neutral": "Neutral response 1"
            },
            {
                "question": "Test question 2",
                "positive": "Positive response 2",
                "neutral": "Neutral response 2"
            }
        ]

        results = self.analyzer.analyze_layers(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=["model.layers[0].mlp"],
            dataset=dataset,
            batch_size=2,
            method="svm",
            top_k=1
        )

        assert results is not None
        assert "rankings" in results

    def test_batch_processing_boundary(self):
        """Test batch processing at dataset boundaries."""
        # Create dataset that doesn't divide evenly by batch size
        dataset = [
            {
                "situation": f"Test situation {i}",
                "positive": f"Positive response {i}",
                "neutral": f"Neutral response {i}"
            }
            for i in range(5)
        ]

        results = self.analyzer.analyze_layers(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=["model.layers[0].mlp"],
            dataset=dataset,
            batch_size=2,  # 5 items / 2 = 2 full batches + 1 partial
            method="svm",
            top_k=1
        )

        assert results is not None
        assert "rankings" in results

    def test_wildcard_expansion(self):
        """Test that wildcard expansion works correctly."""
        dataset = [
            {
                "situation": "Test",
                "positive": "Positive",
                "neutral": "Neutral"
            }
        ]

        results = self.analyzer.analyze_layers(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=["model.layers[*].mlp"],
            dataset=dataset,
            batch_size=1,
            method="svm",
            top_k=3
        )

        assert results is not None
        assert results["total_layers"] == 3
        assert len(results["rankings"]) == 3

    def test_parameter_validation(self):
        """Test parameter validation."""
        dataset = [{"situation": "Test", "positive": "Pos", "neutral": "Neu"}]

        # Should raise error when providing both model and model_name
        with pytest.raises(ValueError, match="Cannot provide both"):
            self.analyzer.analyze_layers(
                model=self.model,
                model_name="test-model",
                tokenizer=self.tokenizer,
                layers=["model.layers[0].mlp"],
                dataset=dataset
            )

        # Should raise error when providing neither model nor model_name
        with pytest.raises(ValueError, match="Must provide either"):
            self.analyzer.analyze_layers(
                layers=["model.layers[0].mlp"],
                dataset=dataset
            )

        # Should raise error when providing both dataset and dataset_path
        with pytest.raises(ValueError, match="Cannot provide both"):
            self.analyzer.analyze_layers(
                model=self.model,
                tokenizer=self.tokenizer,
                layers=["model.layers[0].mlp"],
                dataset=dataset,
                dataset_path="./test"
            )

    def test_top_k_results(self):
        """Test that top_k correctly limits results."""
        dataset = [
            {
                "situation": "Test",
                "positive": "Positive",
                "neutral": "Neutral"
            }
        ]

        results = self.analyzer.analyze_layers(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=["model.layers[*].mlp"],
            dataset=dataset,
            batch_size=1,
            method="svm",
            top_k=2
        )

        assert len(results["top_k_layers"]) == 2
        assert len(results["rankings"]) == 3  # All layers still ranked

    def test_empty_situation_field(self):
        """Test handling of items without situation or question field."""
        dataset = [
            {
                "positive": "Positive response",
                "neutral": "Neutral response"
            }
        ]

        # Should use empty string for situation
        results = self.analyzer.analyze_layers(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=["model.layers[0].mlp"],
            dataset=dataset,
            batch_size=1,
            method="svm",
            top_k=1
        )

        assert results is not None
