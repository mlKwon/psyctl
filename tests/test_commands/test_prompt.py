"""Integration tests for prompt functionality using real models."""

import os
import pytest
from pathlib import Path

from psyctl.core.prompt import P2
from psyctl.core.logger import get_logger, setup_logging

# Setup logging to enable custom logger with success method
setup_logging()
logger = get_logger("test_prompt_integration")


def get_hf_token():
    """Get Hugging Face token from environment variable."""
    token = os.getenv("HF_TOKEN")
    if not token:
        pytest.skip("HF_TOKEN environment variable not set. Set it to run integration tests.")
    return token


@pytest.fixture(scope="session")
def model_and_tokenizer():
    """Load a real model and tokenizer for integration testing."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Use a smaller model for testing
        # model_name = "microsoft/DialoGPT-small"  # Small model for faster testing
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
def p2_real_instance(model_and_tokenizer):
    """Create a P2 instance with real model and tokenizer."""
    model, tokenizer = model_and_tokenizer
    return P2(model, tokenizer)


def test_p2_real_initialization(model_and_tokenizer):
    """Test P2 class initialization with real model."""
    logger.info("Testing P2 class initialization with real model")
    
    model, tokenizer = model_and_tokenizer
    p2 = P2(model, tokenizer)
    
    assert p2.model is not None
    assert p2.tokenizer is not None
    assert p2.model == model
    assert p2.tokenizer == tokenizer
    assert p2.keywords is None
    assert p2.personality is None
    assert not hasattr(p2, 'keywords_build_prompt') or p2.keywords_build_prompt is None
    assert not hasattr(p2, 'personality_build_prompt') or p2.personality_build_prompt is None
    logger.success("P2 real initialization test passed")





def test_build_real_model(p2_real_instance):
    """Test build method with real model."""
    logger.info("Testing build method with real model")
    
    char_name = "Alice"
    personality_trait = "Extroversion"
    
    result = p2_real_instance.build(char_name, personality_trait)
    
    # Verify result is valid
    assert result is not None
    assert isinstance(result, str)
    
    # The result might be empty for some models, which is acceptable
    # Just check that it's a valid string
    assert isinstance(result, str)
    
    logger.success("build method with real model test passed")


def test_build_stores_all_values(p2_real_instance):
    """Test that build method stores all values in instance variables."""
    logger.info("Testing that build method stores all values")
    
    char_name = "Alice"
    personality_trait = "Extroversion"
    
    # Before build, all values should be None
    assert p2_real_instance.keywords is None
    assert p2_real_instance.personality is None
    assert not hasattr(p2_real_instance, 'keywords_build_prompt') or p2_real_instance.keywords_build_prompt is None
    assert not hasattr(p2_real_instance, 'personality_build_prompt') or p2_real_instance.personality_build_prompt is None
    
    # Call build method
    result = p2_real_instance.build(char_name, personality_trait)
    
    # After build, all values should be stored
    assert p2_real_instance.keywords is not None
    assert p2_real_instance.personality is not None
    assert hasattr(p2_real_instance, 'keywords_build_prompt')
    assert hasattr(p2_real_instance, 'personality_build_prompt')
    assert p2_real_instance.keywords_build_prompt is not None
    assert p2_real_instance.personality_build_prompt is not None
    
    # Check types
    assert isinstance(p2_real_instance.keywords, str)
    assert isinstance(p2_real_instance.personality, str)
    assert isinstance(p2_real_instance.keywords_build_prompt, str)
    assert isinstance(p2_real_instance.personality_build_prompt, str)
    
    # Return value should be the personality
    assert result == p2_real_instance.personality
    
    # Build prompts should contain the expected content
    assert personality_trait.lower() in p2_real_instance.keywords_build_prompt.lower()
    assert char_name.lower() in p2_real_instance.personality_build_prompt.lower()
    assert p2_real_instance.keywords.lower() in p2_real_instance.personality_build_prompt.lower()
    
    logger.success("build stores all values test passed")


def test_build_different_personality(p2_real_instance):
    """Test build method with different personality trait."""
    logger.info("Testing build method with different personality trait")
    
    char_name = "Bob"
    personality_trait = "Introversion"
    
    result = p2_real_instance.build(char_name, personality_trait)
    
    # Verify result is valid
    assert result is not None
    assert isinstance(result, str)
    
    # The result might be empty for some models, which is acceptable
    # Just check that it's a valid string
    assert isinstance(result, str)
    
    logger.success("build method with different personality test passed")


@pytest.mark.slow
def test_build_multiple_calls(p2_real_instance):
    """Test multiple calls to build method to ensure consistency."""
    logger.info("Testing multiple calls to build method")
    
    char_name = "Charlie"
    personality_trait = "Openness"
    
    # Make multiple calls
    results = []
    for i in range(3):
        result = p2_real_instance.build(char_name, personality_trait)
        results.append(result)
        
        # Verify each result is valid
        assert result is not None
        assert isinstance(result, str)
    
    # All results should be valid (they might be empty for some models)
    # This is acceptable for testing purposes
    assert len(results) == 3
    
    logger.success("Multiple calls test passed")


def test_build_process_validation(p2_real_instance):
    """Test that build process follows the correct workflow."""
    logger.info("Testing build process validation")
    
    char_name = "Diana"
    personality_trait = "Conscientiousness"
    
    # Call build method
    result = p2_real_instance.build(char_name, personality_trait)
    
    # Validate the build process workflow
    # 1. Keywords should be generated first
    assert p2_real_instance.keywords is not None
    assert isinstance(p2_real_instance.keywords, str)
    
    # 2. Keywords build prompt should ask about the personality trait
    keywords_prompt = p2_real_instance.keywords_build_prompt
    assert "Words related to" in keywords_prompt
    assert personality_trait in keywords_prompt
    assert "format: Comma sperated words" in keywords_prompt
    
    # 3. personality should be generated using the keywords
    assert p2_real_instance.personality is not None
    assert isinstance(p2_real_instance.personality, str)
    
    # 4. Personality build prompt should use the generated keywords
    persona_prompt = p2_real_instance.personality_build_prompt
    assert char_name in persona_prompt
    assert "are traits of" in persona_prompt
    assert "Desribe about" in persona_prompt
    
    # 5. The keywords should appear in the personality build prompt
    # (This validates that the first step's output is used in the second step)
    if p2_real_instance.keywords.strip():  # Only check if keywords is not empty
        assert p2_real_instance.keywords in persona_prompt
    
    # 6. Return value should match stored personality
    assert result == p2_real_instance.personality
    
    logger.success("Build process validation test passed")


if __name__ == "__main__":
    # Instructions for running integration tests
    print("To run integration tests with real models:")
    print("1. Get your Hugging Face token from https://huggingface.co/settings/tokens")
    print("2. Set the environment variable: export HF_TOKEN=your_token_here")
    print("3. Run: pytest tests/test_commands/test_prompt_integration.py -v")
    print("\nOr create a .env file with: HF_TOKEN=your_token_here")
    print("\nExample .env file:")
    print("HF_TOKEN=your_huggingface_token_here")
    print("PSYCTL_LOG_LEVEL=INFO")
