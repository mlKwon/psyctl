"""Vector storage and loading utilities."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from safetensors.torch import load_file, save_file

from psyctl.core.logger import get_logger


class VectorStore:
    """Store and load steering vectors with metadata."""
    
    def __init__(self):
        self.logger = get_logger("vector_store")
    
    def save_steering_vector(
        self,
        vector: torch.Tensor,
        metadata: Dict[str, Any],
        filepath: Path
    ) -> None:
        """Save steering vector with metadata."""
        self.logger.info(f"Saving steering vector to: {filepath}")
        self.logger.debug(f"Vector shape: {vector.shape}")
        self.logger.debug(f"Metadata: {metadata}")
        
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {filepath.parent}")
            
            # Save vector and metadata
            save_file(
                {"steering_vector": vector},
                filepath,
                metadata=metadata
            )
            
            self.logger.success(f"Steering vector saved successfully to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save steering vector: {e}")
            raise
    
    def load_steering_vector(self, filepath: Path) -> tuple[torch.Tensor, Dict[str, Any]]:
        """Load steering vector and metadata."""
        self.logger.info(f"Loading steering vector from: {filepath}")
        
        try:
            if not filepath.exists():
                raise FileNotFoundError(f"Steering vector file does not exist: {filepath}")
            
            # Load vector and metadata
            tensors = load_file(filepath)
            metadata = tensors.get("metadata", {})
            vector = tensors["steering_vector"]
            
            self.logger.debug(f"Loaded vector shape: {vector.shape}")
            self.logger.debug(f"Loaded metadata: {metadata}")
            self.logger.success(f"Steering vector loaded successfully from {filepath}")
            
            return vector, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load steering vector: {e}")
            raise
