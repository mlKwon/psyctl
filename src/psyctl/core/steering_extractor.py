"""Steering vector extractor using various methods."""

from pathlib import Path
from typing import Dict, List, Optional

import torch

from psyctl.core.extractors import (
    BiPOVectorExtractor,
    MeanContrastiveActivationVectorExtractor,
)
from psyctl.core.logger import get_logger
from psyctl.models.llm_loader import LLMLoader
from psyctl.models.vector_store import VectorStore


class SteeringExtractor:
    """Extract steering vectors using various methods."""

    EXTRACTORS = {
        "mean_contrastive": MeanContrastiveActivationVectorExtractor,
        "bipo": BiPOVectorExtractor,
    }

    def __init__(self):
        self.logger = get_logger("steering_extractor")
        self.llm_loader = LLMLoader()
        self.vector_store = VectorStore()

    def extract_caa(
        self,
        model_name: str,
        layers: List[str],
        dataset_path: Path,
        output_path: Path,
        batch_size: Optional[int] = None,
        normalize: bool = False,
        method: str = "mean_contrastive",
        **method_params,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract steering vectors using CAA method.

        Args:
            model_name: Hugging Face model identifier
            layers: List of layer paths to extract from
            dataset_path: Path to CAA dataset
            output_path: Output file path for safetensors
            batch_size: Batch size for inference (optional)
            normalize: Whether to normalize vectors to unit length
            method: Extraction method name (default: "mean_contrastive")

        Returns:
            Dictionary mapping layer names to steering vectors

        Example:
            >>> extractor = SteeringExtractor()
            >>> vectors = extractor.extract_caa(
            ...     model_name="meta-llama/Llama-3.2-3B-Instruct",
            ...     layers=["model.layers[13].mlp.down_proj"],
            ...     dataset_path=Path("./dataset/caa"),
            ...     output_path=Path("./out.safetensors")
            ... )
        """
        self.logger.info(f"Extracting CAA steering vectors for model: {model_name}")
        self.logger.info(f"Target layers: {layers}")
        self.logger.info(f"Dataset path: {dataset_path}")
        self.logger.info(f"Output path: {output_path}")
        self.logger.info(f"Method: {method}")

        try:
            # Validate inputs
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created output directory: {output_path.parent}")

            # 1. Load model
            self.logger.info("Loading model...")
            model, tokenizer = self.llm_loader.load_model(model_name)

            # 2. Get extractor
            extractor_class = self.EXTRACTORS.get(method)
            if extractor_class is None:
                raise ValueError(
                    f"Unknown extraction method: {method}. "
                    f"Available methods: {list(self.EXTRACTORS.keys())}"
                )

            extractor = extractor_class()

            # 3. Extract steering vectors
            self.logger.info(f"Extracting steering vectors using {method}...")
            steering_vectors = extractor.extract(
                model=model,
                tokenizer=tokenizer,
                layers=layers,
                dataset_path=dataset_path,
                batch_size=batch_size,
                normalize=normalize,
                **method_params,
            )

            # 4. Prepare metadata
            metadata = {
                "model": model_name,
                "method": method,
                "layers": layers,
                "dataset_path": str(dataset_path),
                "num_layers": len(layers),
                "normalized": normalize,
            }

            # Add dataset info if available
            try:
                from psyctl.core.caa_dataset_loader import CAADatasetLoader

                loader = CAADatasetLoader()
                dataset_info = loader.get_dataset_info(dataset_path)
                metadata["dataset_samples"] = dataset_info["num_samples"]
            except Exception as e:
                self.logger.debug(f"Could not get dataset info: {e}")

            # 5. Save steering vectors
            self.logger.info("Saving steering vectors...")
            self.vector_store.save_multi_layer(
                vectors=steering_vectors, output_path=output_path, metadata=metadata
            )

            self.logger.info(
                f"Extracted and saved {len(steering_vectors)} steering vectors to {output_path}"
            )

            return steering_vectors

        except Exception as e:
            self.logger.error(f"Failed to extract steering vector: {e}")
            raise
