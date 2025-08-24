"""Settings management for psyctl."""

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Model settings
    default_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    default_device: str = "auto"  # auto, cpu, cuda

    # Hugging Face settings
    hf_token: Optional[str] = None

    # Dataset settings
    default_dataset_size: int = 1000
    default_batch_size: int = 8

    # Steering settings
    default_layer: str = "model.layers[13].mlp.down_proj"
    steering_strength: float = 1.0

    # Output settings
    output_dir: Path = Path("./output")
    dataset_dir: Path = Path("./dataset")
    steering_vector_dir: Path = Path("./steering_vector")
    results_dir: Path = Path("./results")

    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    model_config = {
        "env_file": ".env",
        "env_prefix": "PSYCTL_",
        "extra": "ignore",  # Ignore extra fields from environment variables
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Create directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.output_dir,
            self.dataset_dir,
            self.steering_vector_dir,
            self.results_dir,
        ]

        for directory in directories:
            try:
                directory.mkdir(exist_ok=True)
            except Exception as e:
                print(f"Failed to create directory {directory}: {e}")
                raise
