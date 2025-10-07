"""
OpenRouter-compatible P2 (Personality Prompt) Generator

This module provides an OpenRouter API-based implementation of the P2 class
for generating personality-specific character descriptions without requiring
local model loading.
"""

from typing import Tuple
from psyctl.models.openrouter_client import OpenRouterClient
from psyctl.core.logger import get_logger


class P2OpenRouter:
    """
    P2 implementation using OpenRouter API instead of local models.

    This class mirrors the functionality of the original P2 class but uses
    OpenRouter API for generation, enabling personality prompt creation
    without local GPU resources.

    Attributes:
        client (OpenRouterClient): OpenRouter API client
        model (str): Model identifier on OpenRouter
        keywords (str): Generated personality keywords
        personality (str): Generated personality description
        char_name (str): Character name
    """

    def __init__(self, client: OpenRouterClient, model: str):
        """
        Initialize P2 with OpenRouter client.

        Args:
            client (OpenRouterClient): Initialized OpenRouter client
            model (str): Model identifier (e.g., "qwen/qwen3-next-80b-a3b-instruct")
        """
        self.client = client
        self.model = model
        self.logger = get_logger("p2_openrouter")

        self.keywords = None
        self.personality = None
        self.keywords_build_prompt = None
        self.personality_build_prompt = None
        self.char_name = None

    def build(self, char_name: str, personality_trait: str) -> str:
        """
        Build personality description for a character.

        Args:
            char_name (str): Name of the character
            personality_trait (str): Target personality trait

        Returns:
            str: Generated personality description
        """
        self.logger.info(f"Building P2 for {char_name} with trait: {personality_trait}")

        # Step 1: Generate keywords related to personality trait
        keywords_prompt = f"Words related to {personality_trait}? (format: Comma separated words)"
        _, keywords = self._get_result(keywords_prompt)

        # Step 2: Generate personality description using keywords
        personality_prompt = f"{keywords} are traits of {char_name}.\n\nDescribe about {char_name}"
        prefill = f"Here's a description of {char_name}, built from the traits suggested by the list:"
        _, personality = self._get_result(personality_prompt, prefill=prefill)

        self.char_name = char_name
        self.keywords = keywords
        self.personality = personality
        self.keywords_build_prompt = keywords_prompt
        self.personality_build_prompt = personality_prompt

        self.logger.debug(f"Generated keywords: {keywords}")
        self.logger.debug(f"Generated personality: {personality[:100]}...")

        return self.personality

    def _get_result(self, prompt: str, prefill: str = None) -> Tuple[str, str]:
        """
        Get result from OpenRouter API.

        Args:
            prompt (str): User prompt
            prefill (str, optional): Prefill text for assistant response

        Returns:
            Tuple[str, str]: (full_prompt, generated_text)
        """
        # Construct full prompt with prefill if provided
        full_prompt = prompt
        if prefill:
            full_prompt = f"{prompt}\n\n{prefill}"

        try:
            gen_id, output_text = self.client.generate(
                prompt=full_prompt,
                model=self.model,
                max_tokens=100,
                temperature=0.7,
            )

            # If prefill was used, the output already contains it
            # We want to return just the continuation
            if prefill:
                # The API should continue from the prefill
                result = output_text
            else:
                result = output_text

            return full_prompt, result.strip()

        except Exception as e:
            self.logger.error(f"Failed to get result from OpenRouter: {e}")
            raise


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv(override=True)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY environment variable not set")
        exit(1)

    client = OpenRouterClient(api_key=api_key)
    p2 = P2OpenRouter(client=client, model="qwen/qwen3-next-80b-a3b-instruct")

    # Test personality generation
    personality = p2.build("Alice", "Extroversion")
    print(f"Character: {p2.char_name}")
    print(f"Keywords: {p2.keywords}")
    print(f"Personality: {p2.personality}")
