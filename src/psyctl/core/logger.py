"""Logging configuration for psyctl."""

import sys
from pathlib import Path

from loguru import logger

from psyctl.config.settings import Settings


def setup_logging(settings: Settings = None):
    """Setup logging configuration."""
    if settings is None:
        settings = Settings()

    # Remove default handler
    logger.remove()

    # Console handler with colors
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )

    # File handler if log_file is specified
    if settings.log_file:
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            settings.log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=settings.log_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )

    # Add structured logging for development
    if settings.log_level == "DEBUG":
        logger.add(
            settings.output_dir / "debug.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="5 MB",
            retention="3 days",
        )


def get_logger(name: str = None):
    """Get logger instance."""
    if name:
        return logger.bind(name=name)
    return logger
