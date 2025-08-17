"""Tests for dataset commands."""

import pytest
from click.testing import CliRunner

from psyctl.commands.dataset import build_caa
from psyctl.core.logger import get_logger

logger = get_logger("test_dataset")


def test_build_caa_command():
    """Test dataset build command."""
    logger.info("Testing dataset build command")
    runner = CliRunner()
    result = runner.invoke(
        build_caa,
        [
            "--model",
            "test-model",
            "--personality",
            "Extroversion",
            "--output",
            "./test_output",
        ],
    )
    # Command should run without error (implementation pending)
    assert result.exit_code == 0
    logger.success("Dataset build command test passed")
