"""Tests for CLI functionality."""

from click.testing import CliRunner

from psyctl.cli import main
from psyctl.core.logger import get_logger, setup_logging

# Setup logging to enable custom logger with success method
setup_logging()
logger = get_logger("test_cli")


def test_cli_version():
    """Test CLI version command."""
    logger.info("Testing CLI version command")
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "psyctl" in result.output
    logger.success("CLI version test passed")


def test_cli_help():
    """Test CLI help command."""
    logger.info("Testing CLI help command")
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "PSYCTL" in result.output
    logger.success("CLI help test passed")
