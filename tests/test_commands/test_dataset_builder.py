"""Integration tests for DatasetBuilder functionality using real models and data."""

import json
import os
import pytest
from pathlib import Path

from psyctl.core.dataset_builder import DatasetBuilder
from psyctl.core.logger import get_logger, setup_logging

# Setup logging to enable custom logger with success method
setup_logging()
logger = get_logger("test_dataset_builder")


@pytest.fixture(scope="session")
def model_and_tokenizer():
    """Load a real model and tokenizer for integration testing."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Use a smaller model for testing
        model_name = "google/gemma-3-270m-it"  # Good quality
        
        logger.info(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.success(f"Successfully loaded model: {model_name}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        pytest.skip(f"Failed to load model: {e}")


@pytest.fixture
def dataset_builder_instance():
    """Create a DatasetBuilder instance."""
    builder = DatasetBuilder()
    return builder


def test_dataset_builder_initialization():
    """Test DatasetBuilder class initialization."""
    logger.info("Testing DatasetBuilder class initialization")
    
    builder = DatasetBuilder()
    
    assert builder.llm_loader is not None
    assert builder.p2 is None
    assert builder.logger is not None
    assert builder.dataset is None
    assert builder.model is None
    assert builder.tokenizer is None
    
    logger.success("DatasetBuilder initialization test passed")


def test_load_dataset_real(dataset_builder_instance):
    """Test _load_dataset method with real dataset."""
    logger.info("Testing _load_dataset method with real dataset")
    
    # Test the actual _load_dataset method to see HF_TOKEN logging
    try:
        dataset_builder_instance._load_dataset()
        logger.success("Successfully loaded dataset using _load_dataset method")
        
        # Check dataset structure
        assert dataset_builder_instance.dataset is not None
        assert len(dataset_builder_instance.dataset) > 0
        
        sample = dataset_builder_instance.dataset[0]
        assert "speakers" in sample
        assert "dialogue" in sample
        assert "narrative" in sample
        
        logger.success(f"_load_dataset test passed - loaded {len(dataset_builder_instance.dataset)} samples")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        pytest.skip(f"Failed to load dataset: {e}")


def test_get_answer_real(dataset_builder_instance, model_and_tokenizer):
    """Test _get_answer method with real model."""
    logger.info("Testing _get_answer method with real model")
    
    model, tokenizer = model_and_tokenizer
    dataset_builder_instance.model = model
    dataset_builder_instance.tokenizer = tokenizer
    
    # Test with a simple scenario
    result = dataset_builder_instance._get_answer(
        user_name="Alice",
        char_name="Bob",
        p2="Bob is an extroverted person who loves socializing.",
        situation="Alice meets Bob at a party.\nAlice: Hello, how are you?\n",
        verbose=False
    )
    
    assert isinstance(result, str)
    assert len(result) > 0
    
    logger.success("_get_answer test passed")


def test_gen_caa_data(dataset_builder_instance):
    """Test _gen_caa_data method."""
    logger.info("Testing _gen_caa_data method")
    
    char_name = "Alice"
    situation = "Alice meets Bob at a coffee shop.\nBob: Hello, how are you?\n"
    answer_pos = "I'm excited to meet you!"
    answer_neg = "I'm feeling a bit shy today."
    
    result = dataset_builder_instance._gen_caa_data(char_name, situation, answer_pos, answer_neg)
    
    # Check that result is a single string template
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Check template content
    assert "[Situation]" in result
    assert "Alice meets Bob at a coffee shop" in result
    assert "Bob: Hello, how are you?" in result
    assert "[Question]" in result
    assert "You are Alice. What would your response be in this situation?" in result
    assert "1. I'm excited to meet you!" in result
    assert "2. I'm feeling a bit shy today." in result
    assert "[Answer]" in result
    
    logger.success("_gen_caa_data test passed")


def test_save_sample_to_jsonl(dataset_builder_instance, tmp_path):
    """Test _save_sample_to_jsonl method."""
    logger.info("Testing _save_sample_to_jsonl method")
    
    sample = {"question": "test question", "positive": "(1", "neutral": "(2"}
    output_file = tmp_path / "test_output.jsonl"
    
    dataset_builder_instance._save_sample_to_jsonl(sample, output_file)
    
    # Check that file was created
    assert output_file.exists()
    
    # Check file content
    with open(output_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    # Should contain the sample as JSON
    assert '"question"' in content
    assert '"test question"' in content
    assert '"(1"' in content
    assert '"(2"' in content
    
    logger.success("_save_sample_to_jsonl test passed")


def test_build_caa_dataset_real_integration(dataset_builder_instance, model_and_tokenizer, tmp_path):
    """Test complete build_caa_dataset integration with real data."""
    logger.info("Testing complete build_caa_dataset integration with real data")
    
    model, tokenizer = model_and_tokenizer
    
    # Mock the LLMLoader to return our test model
    from unittest.mock import patch
    with patch.object(dataset_builder_instance.llm_loader, 'load_model') as mock_load:
        mock_load.return_value = (model, tokenizer)
        
        # Test the complete build process with 10 samples
        output_dir = tmp_path / "test_dataset_real"
        num_samples = dataset_builder_instance.build_caa_dataset(
            model="test-model",
            personality="Extroversion",
            output_dir=output_dir,
            limit_samples=10
        )
        
        # Check results
        assert num_samples == 10
        assert dataset_builder_instance.model == model
        assert dataset_builder_instance.tokenizer == tokenizer
        assert dataset_builder_instance.personality == "Extroversion"
        
        # Check that output file was created
        output_files = list(output_dir.glob("caa_dataset_*.jsonl"))
        assert len(output_files) == 1
        
        # Check file content
        with open(output_files[0], "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Should have 10 lines (one for each sample)
        assert len(lines) == 10
        
        # Check that each line contains valid JSON        
        for line in lines:
            sample = json.loads(line.strip())
            assert isinstance(sample, dict)
            assert "question" in sample
            assert "positive" in sample
            assert "neutral" in sample
    
    logger.success("build_caa_dataset real integration test passed")


def test_build_caa_dataset_with_small_limit(dataset_builder_instance, model_and_tokenizer, tmp_path):
    """Test build_caa_dataset with small sample limit."""
    logger.info("Testing build_caa_dataset with small sample limit")
    
    model, tokenizer = model_and_tokenizer
    
    # Mock the LLMLoader to return our test model
    from unittest.mock import patch
    with patch.object(dataset_builder_instance.llm_loader, 'load_model') as mock_load:
        mock_load.return_value = (model, tokenizer)
        
        # Test with limit of 3 samples
        output_dir = tmp_path / "test_dataset_small"
        num_samples = dataset_builder_instance.build_caa_dataset(
            model="test-model",
            personality="Extroversion",
            output_dir=output_dir,
            limit_samples=3
        )
        
        # Should generate exactly 3 samples
        assert num_samples == 3
        
        # Check that output file was created
        output_files = list(output_dir.glob("caa_dataset_*.jsonl"))
        assert len(output_files) == 1
        
        # Check file content
        with open(output_files[0], "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Should have 3 lines
        assert len(lines) == 3
    
    logger.success("build_caa_dataset with small limit test passed")


def test_build_caa_dataset_error_handling(dataset_builder_instance):
    """Test build_caa_dataset error handling."""
    logger.info("Testing build_caa_dataset error handling")
    
    # Test with invalid model
    with pytest.raises(Exception):
        dataset_builder_instance.build_caa_dataset(
            model="invalid-model",
            personality="Extroversion",
            output_dir=Path("./test_output"),
            limit_samples=1
        )
    
    logger.success("build_caa_dataset error handling test passed")


if __name__ == "__main__":
    # Instructions for running integration tests
    print("To run integration tests with real models:")
    print("1. Get your Hugging Face token from https://huggingface.co/settings/tokens")
    print("2. Set the environment variable: export HF_TOKEN=your_token_here")
    print("3. Run: pytest tests/test_commands/test_dataset_builder.py -v")
    print("\nOr create a .env file with: HF_TOKEN=your_token_here")
    print("\nExample .env file:")
    print("HF_TOKEN=your_huggingface_token_here")
    print("PSYCTL_LOG_LEVEL=INFO")
