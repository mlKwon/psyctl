"""PSYCTL - LLM Personality Steering Tool."""

__version__ = "0.1.0"
__author__ = "ModuLabs Persona Lab"
__email__ = "rick@caveduck.io"

# Core classes
# Configuration
from . import config

# Commands (for CLI usage)
from .commands import benchmark, dataset, extract, steering
from .core.dataset_builder import DatasetBuilder

# Logger
from .core.logger import get_logger
from .core.prompt import P2

# Models
from .models.llm_loader import LLMLoader
from .models.vector_store import VectorStore

__all__ = [
    "P2",
    # Core classes
    "DatasetBuilder",
    # Models
    "LLMLoader",
    "VectorStore",
    "__author__",
    "__email__",
    # Metadata
    "__version__",
    "benchmark",
    # Configuration
    "config",
    # Commands
    "dataset",
    "extract",
    # Utilities
    "get_logger",
    "steering",
]
