"""Simple configuration management."""
import os
from pathlib import Path
from typing import Optional

def get_env(key: str, default=None, cast_type: type = str):
    """Get environment variable with type casting."""
    value = os.getenv(key, default)
    if value is None or value == default:
        return default
    if cast_type == bool:
        return value.lower() in ('true', '1', 'yes', 'on')
    elif cast_type == int:
        return int(value)
    elif cast_type == float:
        return float(value)
    elif cast_type == Path:
        return Path(value)
    return value

# Model settings
DEFAULT_MODEL = get_env("PSYCTL_DEFAULT_MODEL", "gemma-3-270m-it")
DEFAULT_DEVICE = get_env("PSYCTL_DEFAULT_DEVICE", "auto")

# Hugging Face settings
HF_TOKEN = get_env("PSYCTL_HF_TOKEN", get_env("HF_TOKEN"))

# Dataset settings
DEFAULT_DATASET_SIZE = get_env("PSYCTL_DEFAULT_DATASET_SIZE", 1000, int)
DEFAULT_BATCH_SIZE = get_env("PSYCTL_DEFAULT_BATCH_SIZE", 8, int)

# Steering settings
DEFAULT_LAYER = get_env("PSYCTL_DEFAULT_LAYER", "model.layers[13].mlp.down_proj")
STEERING_STRENGTH = get_env("PSYCTL_STEERING_STRENGTH", 1.0, float)

# Directory settings
OUTPUT_DIR = get_env("PSYCTL_OUTPUT_DIR", Path("./output"), Path)
DATASET_DIR = get_env("PSYCTL_DATASET_DIR", Path("./dataset"), Path)
STEERING_VECTOR_DIR = get_env("PSYCTL_STEERING_VECTOR_DIR", Path("./steering_vector"), Path)
RESULTS_DIR = get_env("PSYCTL_RESULTS_DIR", Path("./results"), Path)
CACHE_DIR = get_env("PSYCTL_CACHE_DIR", Path("./temp"), Path)

# Logging settings
LOG_LEVEL = get_env("PSYCTL_LOG_LEVEL", "INFO")
LOG_FILE = get_env("PSYCTL_LOG_FILE", None, Path)

def create_directories():
    """Create necessary directories."""
    directories = [OUTPUT_DIR, DATASET_DIR, STEERING_VECTOR_DIR, RESULTS_DIR, CACHE_DIR]

    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Failed to create directory {directory}: {e}")
            raise