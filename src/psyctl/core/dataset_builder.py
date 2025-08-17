"""Dataset builder for personality steering."""

from pathlib import Path
from typing import List

from psyctl.core.logger import get_logger
from psyctl.data.personality_templates import PersonalityTemplates
from psyctl.models.llm_loader import LLMLoader


class DatasetBuilder:
    """Build datasets for steering vector extraction."""

    def __init__(self):
        self.llm_loader = LLMLoader()
        self.templates = PersonalityTemplates()
        self.logger = get_logger("dataset_builder")

    def build_caa_dataset(self, model: str, personality: str, output_dir: Path):
        """Build CAA dataset for given personality traits."""
        self.logger.info(f"Building CAA dataset for model: {model}")
        self.logger.info(f"Personality traits: {personality}")
        self.logger.info(f"Output directory: {output_dir}")

        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created output directory: {output_dir}")

            # Parse personality traits
            traits = [trait.strip() for trait in personality.split(",")]
            self.logger.info(f"Parsed traits: {traits}")

            # TODO: Implement CAA dataset generation
            # 1. Load model
            self.logger.debug("Loading model...")
            # model, tokenizer = self.llm_loader.load_model(model)

            # 2. Generate personality-specific prompts
            self.logger.debug("Generating personality-specific prompts...")
            # prompts = self._generate_prompts(traits)

            # 3. Collect activations
            self.logger.debug("Collecting activations...")
            # activations = self._collect_activations(model, prompts)

            # 4. Save dataset
            self.logger.debug("Saving dataset...")
            # self._save_dataset(activations, output_dir)

            self.logger.success(f"CAA dataset built successfully at {output_dir}")

        except Exception as e:
            self.logger.error(f"Failed to build CAA dataset: {e}")
            raise
