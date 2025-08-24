"""PSYCTL - LLM Personality Steering Tool."""

__version__ = "0.1.0"
__author__ = "ModuLabs Persona Lab"
__email__ = "rick@caveduck.io"

# Core classes
from .core.dataset_builder import DatasetBuilder
from .core.prompt import P2

# Models
from .models.llm_loader import LLMLoader
from .models.vector_store import VectorStore

# Configuration
from .config.settings import Settings

# Commands (for CLI usage)
from .commands import dataset, extract, steering, benchmark

# Logger
from .core.logger import get_logger

__all__ = [
    # Core classes
    "DatasetBuilder",
    "P2",
    
    # Models
    "LLMLoader", 
    "VectorStore",
    
    # Configuration
    "Settings",
    
    # Commands
    "dataset",
    "extract", 
    "steering",
    "benchmark",
    
    # Utilities
    "get_logger",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]
