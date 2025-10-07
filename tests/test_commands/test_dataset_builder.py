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
    assert builder.jinja_env is not None
    assert builder.caa_question_template_path is None
    assert builder.roleplay_prompt_template_path is None

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
        output_file = dataset_builder_instance.build_caa_dataset(
            model="test-model",
            personality="Extroversion",
            output_dir=output_dir,
            limit_samples=10
        )

        # Check results
        assert output_file is not None
        assert output_file.exists()
        assert output_file.suffix == ".jsonl"
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
        output_file = dataset_builder_instance.build_caa_dataset(
            model="test-model",
            personality="Extroversion",
            output_dir=output_dir,
            limit_samples=3
        )

        # Should generate output file
        assert output_file is not None
        assert output_file.exists()
        assert output_file.suffix == ".jsonl"
        
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


def test_template_loading(dataset_builder_instance):
    """Test template loading functionality."""
    logger.info("Testing template loading functionality")

    # Test loading default templates
    caa_template = dataset_builder_instance._load_template('caa_question.j2')
    assert caa_template is not None

    roleplay_template = dataset_builder_instance._load_template('roleplay_prompt.j2')
    assert roleplay_template is not None

    logger.success("Template loading test passed")


def test_custom_template_override(dataset_builder_instance, tmp_path):
    """Test custom template override functionality."""
    logger.info("Testing custom template override functionality")

    # Create a custom template
    custom_template_path = tmp_path / "custom_caa.j2"
    custom_template_content = """[Custom Situation]
{{ situation }}
[Custom Question]
{{ char_name }}, what would you say?
1. {{ answer_1 }}
2. {{ answer_2 }}
[Custom Answer]
"""
    with open(custom_template_path, 'w', encoding='utf-8') as f:
        f.write(custom_template_content)

    # Create builder with custom template
    builder = DatasetBuilder(caa_question_template=str(custom_template_path))

    # Test rendering with custom template
    result = builder._gen_caa_data(
        char_name="Alice",
        situation="Test situation",
        answer_1="Answer 1",
        answer_2="Answer 2"
    )

    assert "[Custom Situation]" in result
    assert "[Custom Question]" in result
    assert "[Custom Answer]" in result
    assert "Alice, what would you say?" in result

    logger.success("Custom template override test passed")


def test_template_rendering_with_variables(dataset_builder_instance):
    """Test template rendering with all variables."""
    logger.info("Testing template rendering with variables")

    # Test CAA question template
    result = dataset_builder_instance._gen_caa_data(
        char_name="Bob",
        situation="Bob meets Alice.\nAlice: Hi there!",
        answer_1="Hello, nice to meet you!",
        answer_2="Oh, hi."
    )

    assert "Bob" in result
    assert "Bob meets Alice" in result
    assert "Hello, nice to meet you!" in result
    assert "Oh, hi." in result

    logger.success("Template rendering with variables test passed")


def test_get_default_templates(dataset_builder_instance):
    """Test getting default template content."""
    logger.info("Testing getting default template content")

    # Get CAA question template
    caa_template = dataset_builder_instance.get_caa_question_template()
    assert isinstance(caa_template, str)
    assert len(caa_template) > 0
    assert "[Situation]" in caa_template
    assert "[Question]" in caa_template
    assert "[Answer]" in caa_template

    # Get roleplay prompt template
    roleplay_template = dataset_builder_instance.get_roleplay_prompt_template()
    assert isinstance(roleplay_template, str)
    assert len(roleplay_template) > 0
    assert "# Overview" in roleplay_template
    assert "# Situation" in roleplay_template

    logger.success("Getting default templates test passed")


def test_set_and_get_custom_template_from_string(dataset_builder_instance):
    """Test setting and getting custom template from string."""
    logger.info("Testing setting and getting custom template from string")

    # Custom CAA template
    custom_caa = """[Custom Situation]
{{ situation }}
[Custom Question]
{{ char_name }}, what is your answer?
1. {{ answer_1 }}
2. {{ answer_2 }}
[Custom Answer]
"""

    # Set custom template
    dataset_builder_instance.set_caa_question_template(custom_caa)

    # Get it back
    retrieved = dataset_builder_instance.get_caa_question_template()
    assert retrieved == custom_caa
    assert "[Custom Situation]" in retrieved
    assert "[Custom Question]" in retrieved

    # Test that it actually works in _gen_caa_data
    result = dataset_builder_instance._gen_caa_data(
        char_name="Alice",
        situation="Test situation",
        answer_1="Answer 1",
        answer_2="Answer 2"
    )
    assert "[Custom Situation]" in result
    assert "[Custom Question]" in result
    assert "Alice, what is your answer?" in result

    logger.success("Setting and getting custom template from string test passed")


def test_set_and_get_roleplay_template_from_string(dataset_builder_instance, model_and_tokenizer):
    """Test setting and getting custom roleplay template from string."""
    logger.info("Testing setting and getting custom roleplay template from string")

    model, tokenizer = model_and_tokenizer
    dataset_builder_instance.model = model
    dataset_builder_instance.tokenizer = tokenizer

    # Custom roleplay template
    custom_roleplay = """# Custom Overview
You are {{ char_name }}.
User is {{ user_name }}.

# Personality
{{ p2 }}

# Context
{{ situation }}
"""

    # Set custom template
    dataset_builder_instance.set_roleplay_prompt_template(custom_roleplay)

    # Get it back
    retrieved = dataset_builder_instance.get_roleplay_prompt_template()
    assert retrieved == custom_roleplay
    assert "# Custom Overview" in retrieved
    assert "# Context" in retrieved

    # Test that it actually works in _get_answer
    result = dataset_builder_instance._get_answer(
        user_name="Alice",
        char_name="Bob",
        p2="Bob is friendly.",
        situation="Alice: Hello!",
        verbose=False
    )
    assert isinstance(result, str)
    assert len(result) > 0

    logger.success("Setting and getting custom roleplay template from string test passed")


def test_template_priority(dataset_builder_instance, tmp_path):
    """Test template loading priority: in-memory > file > default."""
    logger.info("Testing template loading priority")

    # 1. Default template
    default_template = dataset_builder_instance.get_caa_question_template()
    assert "[Situation]" in default_template

    # 2. File-based template
    file_template_path = tmp_path / "file_template.j2"
    file_template_content = "[File Situation]\n{{ situation }}"
    with open(file_template_path, 'w', encoding='utf-8') as f:
        f.write(file_template_content)

    builder_with_file = DatasetBuilder(caa_question_template=str(file_template_path))
    file_loaded = builder_with_file.get_caa_question_template()
    assert "[File Situation]" in file_loaded

    # 3. In-memory template (should override file)
    memory_template_content = "[Memory Situation]\n{{ situation }}"
    builder_with_file.set_caa_question_template(memory_template_content)
    memory_loaded = builder_with_file.get_caa_question_template()
    assert "[Memory Situation]" in memory_loaded
    assert "[File Situation]" not in memory_loaded

    logger.success("Template loading priority test passed")


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
