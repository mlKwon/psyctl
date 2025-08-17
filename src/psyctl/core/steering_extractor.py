"""Steering vector extractor using CAA method."""

from pathlib import Path

import torch
from safetensors.torch import save_file

from psyctl.core.logger import get_logger


class SteeringExtractor:
    """Extract steering vectors using various methods."""

    def __init__(self):
        self.logger = get_logger("steering_extractor")

    def extract_caa(
        self, model: str, layer: str, dataset_path: Path, output_path: Path
    ):
        """Extract steering vector using CAA method."""
        self.logger.info(f"Extracting CAA steering vector for model: {model}")
        self.logger.info(f"Target layer: {layer}")
        self.logger.info(f"Dataset path: {dataset_path}")
        self.logger.info(f"Output path: {output_path}")

        try:
            # Validate inputs
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created output directory: {output_path.parent}")

            # TODO: Implement CAA steering vector extraction
            # 1. Load model and dataset
            self.logger.debug("Loading model and dataset...")
            # model, tokenizer = self._load_model(model)
            # dataset = self._load_dataset(dataset_path)

            # 2. Extract activations from specified layer
            self.logger.debug(f"Extracting activations from layer: {layer}")
            # activations = self._extract_activations(model, dataset, layer)

            # 3. Apply CAA algorithm
            self.logger.debug("Applying CAA algorithm...")
            # steering_vector = self._apply_caa_algorithm(activations)

            # 4. Save steering vector with metadata
            self.logger.debug("Saving steering vector...")
            # self._save_steering_vector(steering_vector, model, layer, output_path)

            self.logger.success(
                f"Steering vector extracted successfully to {output_path}"
            )

        except Exception as e:
            self.logger.error(f"Failed to extract steering vector: {e}")
            raise
