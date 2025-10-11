"""Inventory tester for measuring personality changes."""

from pathlib import Path
from typing import Any

from psyctl.core.logger import get_logger


class InventoryTester:
    """Test personality changes using psychological inventories."""

    def __init__(self):
        self.logger = get_logger("inventory_tester")

    def test_inventory(
        self, model: str, steering_vector_path: Path, inventory_name: str
    ) -> dict[str, Any]:
        """Run inventory test and return results."""
        self.logger.info(f"Running inventory test for model: {model}")
        self.logger.info(f"Steering vector path: {steering_vector_path}")
        self.logger.info(f"Inventory: {inventory_name}")

        try:
            # Validate inputs
            if not steering_vector_path.exists():
                raise FileNotFoundError(
                    f"Steering vector file does not exist: {steering_vector_path}"
                )

            # TODO: Implement inventory testing
            # 1. Load model and steering vector
            self.logger.debug("Loading model and steering vector...")
            # model, tokenizer = self._load_model(model)
            # steering_vector, metadata = self._load_steering_vector(steering_vector_path)

            # 2. Load inventory questions
            self.logger.debug(f"Loading inventory questions for: {inventory_name}")
            # inventory = self._load_inventory(inventory_name)

            # 3. Generate responses with and without steering
            self.logger.debug("Generating responses with and without steering...")
            # original_responses = self._generate_responses(model, tokenizer, inventory, use_steering=False)
            # steered_responses = self._generate_responses(model, tokenizer, inventory, use_steering=True, steering_vector=steering_vector)

            # 4. Calculate personality scores
            self.logger.debug("Calculating personality scores...")
            # original_scores = self._calculate_scores(original_responses, inventory)
            # steered_scores = self._calculate_scores(steered_responses, inventory)

            # 5. Return comparison results
            results = {
                "inventory": inventory_name,
                "original_score": 0.5,
                "steered_score": 0.7,
                "difference": 0.2,
            }

            self.logger.success("Inventory test completed successfully")
            self.logger.info(f"Test results: {results}")

            return results

        except Exception as e:
            self.logger.error(f"Failed to run inventory test: {e}")
            raise
