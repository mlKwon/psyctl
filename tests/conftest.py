"""Pytest configuration for psyctl tests."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_dataset_path(tmp_path):
    """Create a sample dataset path for testing."""
    return tmp_path / "sample_dataset"


@pytest.fixture
def sample_steering_vector_path(tmp_path):
    """Create a sample steering vector path for testing."""
    return tmp_path / "sample_vector.safetensors"
